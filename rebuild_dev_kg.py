"""
Build the knowledge graph from the narratives with schema-constrained LLM extraction.

Usage:
    python rebuild_dev_kg.py --csv development_cohort.csv --old-graph perignnosis_graph.pt --out perignnosis_graph_rebuilt.pt
    python rebuild_dev_kg.py ... --cache rebuild_extraction.jsonl
"""

import argparse, json, os, re, sys, time, collections, hashlib
import numpy as np
import torch
from torch_geometric.data import HeteroData

EMB = 768
PDIN, COMP = 13, 1

NODE_TYPES = ["Condition", "Symptom", "Procedure", "Treatment", "Medication",
              "Device", "BodyPart", "Person", "Place", "Time", "Event",
              "Emotion", "Measurement", "Test"]

REL_TYPES = ["EXPERIENCED", "CAUSED", "TREATED_WITH", "PERFORMED_BY",
             "HAS_SYMPTOM", "DIAGNOSED_WITH", "LOCATED_AT", "TEMPORAL",
             "PART_OF", "ASSOCIATED_WITH"]

WOMAN_REL = {
    "Condition": "EXPERIENCED", "Symptom": "EXPERIENCED",
    "Event": "EXPERIENCED", "Emotion": "EXPERIENCED",
    "Procedure": "UNDERWENT", "Test": "UNDERWENT",
    "Treatment": "RECEIVED", "Medication": "RECEIVED", "Device": "RECEIVED",
    "Person": "MENTIONS", "Place": "MENTIONS", "Time": "MENTIONS",
    "BodyPart": "MENTIONS", "Measurement": "MENTIONS",
}

SYSTEM_PROMPT = """You extract a clinical knowledge graph from a woman's account of giving birth.

Use ONLY these entity types:
Condition, Symptom, Procedure, Treatment, Medication, Device, BodyPart, Person, Place, Time, Event, Emotion, Measurement, Test

Use ONLY these relation types:
EXPERIENCED, CAUSED, TREATED_WITH, PERFORMED_BY, HAS_SYMPTOM, DIAGNOSED_WITH, LOCATED_AT, TEMPORAL, PART_OF, ASSOCIATED_WITH

Rules:
- Entity names must be short, lower-case, generic clinical terms as they appear in the text
  (e.g. "emergency cesarean", "epidural", "fear of dying", "nicu", "hemorrhage").
- Do not invent entities that are not in the text. Do not include the narrator herself.
- Do not use the narrator's or the baby's proper name; use "baby", "partner", "nurse",
  "obstetrician", "midwife" and similar role words instead.
- Return JSON only, in this exact form:
  {"nodes": [{"id": "...", "type": "..."}],
   "edges": [{"source": "...", "target": "...", "type": "..."}]}
"""

SYNONYMS = {
    "c section": "cesarean", "c-section": "cesarean", "csection": "cesarean",
    "caesarean": "cesarean", "cesarean section": "cesarean",
    "emergency c section": "emergency cesarean",
    "emergency caesarean": "emergency cesarean",
    "emergency cesarean section": "emergency cesarean",
    "vaginal delivery": "vaginal birth", "natural birth": "vaginal birth",
    "epidural anesthesia": "epidural", "epidural anaesthesia": "epidural",
    "nicu": "neonatal intensive care unit",
    "postpartum hemorrhage": "hemorrhage", "haemorrhage": "hemorrhage",
    "post partum": "postpartum", "ob": "obstetrician", "obgyn": "obstetrician",
    "doctor": "physician", "dr": "physician",
    "labor": "labour", "contraction": "contractions",
    "baby boy": "baby", "baby girl": "baby", "newborn": "baby", "infant": "baby",
    "husband": "partner", "wife": "partner", "spouse": "partner",
    "anxious": "anxiety", "scared": "fear", "afraid": "fear", "terrified": "fear",
    "panicked": "panic", "helpless": "helplessness", "powerless": "helplessness",
}
_STOP = {"the", "a", "an", "my", "her", "his", "their", "this", "that", "some"}


def canon_surface(s):
    s = str(s).lower().strip()
    s = re.sub(r"[^a-z0-9 \-]", " ", s)
    s = s.replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    toks = [t for t in s.split() if t not in _STOP]
    s = " ".join(toks)
    s = SYNONYMS.get(s, s)
    if len(s) > 4 and s.endswith("s") and not s.endswith("ss"):
        s2 = s[:-1]
        s = SYNONYMS.get(s2, s2)
    return s


TYPE_FALLBACK = {
    "disease": "Condition", "diagnosis": "Condition", "illness": "Condition",
    "complication": "Condition", "injury": "Condition",
    "sign": "Symptom", "pain": "Symptom",
    "surgery": "Procedure", "operation": "Procedure", "intervention": "Procedure",
    "therapy": "Treatment", "care": "Treatment", "service": "Treatment",
    "drug": "Medication", "medicine": "Medication", "anesthesia": "Medication",
    "substance": "Medication",
    "equipment": "Device", "instrument": "Device", "tool": "Device",
    "anatomy": "BodyPart", "organ": "BodyPart",
    "role": "Person", "occupation": "Person", "people": "Person",
    "family": "Person", "staff": "Person", "baby": "Person",
    "location": "Place", "facility": "Place", "hospital": "Place",
    "organization": "Place", "room": "Place",
    "date": "Time", "duration": "Time", "period": "Time",
    "experience": "Event", "activity": "Event", "action": "Event",
    "feeling": "Emotion", "mood": "Emotion", "emotionalstate": "Emotion",
    "quantity": "Measurement", "number": "Measurement", "weight": "Measurement",
    "examination": "Test", "screening": "Test", "scan": "Test",
}


def canon_type(t):
    t = re.sub(r"[^a-z]", "", str(t).lower())
    if not t:
        return None
    for nt in NODE_TYPES:
        if t == nt.lower():
            return nt
    if t in TYPE_FALLBACK:
        return TYPE_FALLBACK[t]
    for nt in NODE_TYPES:
        if nt.lower() in t:
            return nt
    for k, v in TYPE_FALLBACK.items():
        if k in t:
            return v
    return None


def canon_rel(r):
    r = re.sub(r"[^a-z]", "", str(r).lower())
    for rt in REL_TYPES:
        if r == rt.replace("_", "").lower():
            return rt
    return None


def parse_json_block(txt):
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        return None
    for cand in (m.group(0), m.group(0).replace("'", '"')):
        try:
            return json.loads(cand)
        except Exception:
            continue
    return None


def extract_once(llm, narrative):
    from langchain_core.messages import SystemMessage, HumanMessage
    out = llm.invoke([SystemMessage(content=SYSTEM_PROMPT),
                      HumanMessage(content=narrative.strip()[:6000])])
    obj = parse_json_block(getattr(out, "content", str(out)))
    if not obj:
        return set(), set()
    nodes, edges = set(), set()
    typed = {}
    for n in obj.get("nodes", []) or []:
        if not isinstance(n, dict):
            continue
        nid, nt = canon_surface(n.get("id", "")), canon_type(n.get("type", ""))
        if nid and nt and 1 < len(nid) <= 60:
            typed[nid] = nt
            nodes.add((nid, nt))
    for e in obj.get("edges", []) or []:
        if not isinstance(e, dict):
            continue
        s, d = canon_surface(e.get("source", "")), canon_surface(e.get("target", ""))
        r = canon_rel(e.get("type", ""))
        if s in typed and d in typed and r and s != d:
            edges.add((s, r, d))
    return nodes, edges


def extract_narrative(llm, narrative, passes, min_votes):
    nvote, evote = collections.Counter(), collections.Counter()
    for _ in range(passes):
        n, e = extract_once(llm, narrative)
        nvote.update(n)
        evote.update(e)
    keep_n = {k for k, v in nvote.items() if v >= min_votes}
    names = {nid for nid, _ in keep_n}
    keep_e = {k for k, v in evote.items() if v >= min_votes and k[0] in names and k[2] in names}
    return sorted(keep_n), sorted(keep_e)


def run_extraction(df, args):
    done = {}
    if os.path.exists(args.cache):
        with open(args.cache) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done[str(r["record_id"])] = r
                except Exception:
                    pass
        print(f"resuming: {len(done)} narratives already extracted", flush=True)

    todo = [i for i, r in df.iterrows() if str(r[args.id_col]) not in done]
    if todo:
        from langchain_ollama import ChatOllama
        llm = ChatOllama(model=args.model, temperature=args.temperature,
                         num_predict=1200, format="json")
        t0 = time.time()
        fails = empties = 0
        with open(args.cache, "a") as f:
            for k, i in enumerate(todo):
                row = df.loc[i]
                rid = str(row[args.id_col])
                try:
                    nodes, edges = extract_narrative(llm, str(row[args.text_col]),
                                                     args.passes, args.min_votes)
                except ImportError:
                    raise
                except Exception as ex:
                    print(f"  !! {rid}: {type(ex).__name__}: {ex}", flush=True)
                    nodes, edges = [], []
                    fails += 1
                empties += not nodes
                if k + 1 >= args.health_check and empties > args.max_empty_frac * (k + 1):
                    sys.exit(f"\nABORTING: {empties} of the first {k+1} narratives "
                             f"produced no entities ({fails} raised errors). Something is "
                             f"wrong with the model or the prompt, not with the data. "
                             f"Fix it, delete {args.cache}, and rerun.")
                rec = {"record_id": rid, "nodes": nodes, "edges": edges}
                f.write(json.dumps(rec) + "\n")
                f.flush()
                done[rid] = rec
                if (k + 1) % 10 == 0 or k == len(todo) - 1:
                    el = time.time() - t0
                    print(f"  extracted {k+1}/{len(todo)}  ({el:.0f}s, "
                          f"{el/(k+1):.1f}s each, eta {(len(todo)-k-1)*el/(k+1)/60:.0f} min, "
                          f"empty {empties}, errors {fails})", flush=True)
    n_empty = sum(1 for r in done.values() if not r["nodes"])
    if n_empty:
        print(f"note: {n_empty}/{len(done)} narratives yielded no entities", flush=True)
    return done


def load_embedder(args):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(args.embed_model, trust_remote_code=True,
                            device="cuda" if torch.cuda.is_available() else "cpu")
    m.max_seq_length = args.max_seq_length
    return m


def embed(model, texts, bs=16):
    v = model.encode(list(texts), batch_size=bs, convert_to_numpy=True,
                     normalize_embeddings=True, show_progress_bar=False)
    return torch.tensor(np.asarray(v), dtype=torch.float)


def audit_alignment(df, args, best, keep, Xo, yo, S=None):
    import pandas as pd
    n_graph = Xo.size(0)
    k = keep.numpy()
    matched = set(best[keep].tolist())
    extra = sorted(set(range(n_graph)) - matched)
    ids = df[args.id_col].astype(str).values
    print(f"\n--- CSV vs published graph ---")
    print(f"graph rows matched by no CSV row: {len(extra)} {extra[:10]}")
    if extra:
        print(f"  labels of those rows: {[int(yo[i]) for i in extra]}")

    if S is not None:
        for gi in extra[:10]:
            c = S[:, gi]
            i = int(c.argmax())
            print(f"  graph row {gi}: closest CSV row {i} (id {ids[i]}) "
                  f"cosine {float(c[i]):.4f}")
        for i in np.where(~k)[0]:
            print(f"  DROPPED CSV row {i} (id {ids[i]}): best graph row "
                  f"{int(best[i])}, cosine {float(S[i].max()):.4f}"
                  + (f", {args.pcl_col}="
                     f"{pd.to_numeric(df[args.pcl_col], errors='coerce').iloc[i]}"
                     if args.pcl_col in df.columns else ""))
        dup = {g: c for g, c in collections.Counter(best[keep].tolist()).items() if c > 1}
        if dup:
            print(f"  graph rows matched by MORE THAN ONE kept CSV row: {dup}")

    if args.pcl_col in df.columns:
        csv_y = (pd.to_numeric(df[args.pcl_col], errors="coerce").values
                 >= args.cutoff).astype(int)
        gy = yo[best].numpy()
        print(f"positives in the WHOLE CSV ({len(df)} rows): {int(csv_y.sum())}"
              f"  <- compare with the manuscript")
        agree = int((csv_y[k] == gy[k]).sum())
        print(f"label agreement, CSV({args.pcl_col}>={args.cutoff}) vs graph y: "
              f"{agree}/{int(k.sum())}")
        if agree < int(k.sum()):
            mism = np.where((csv_y != gy) & k)[0]
            print(f"  ** MISMATCHED CSV rows: {mism.tolist()[:20]}")
        print(f"positives: CSV {int(csv_y[k].sum())}  matched graph rows "
              f"{int(gy[k].sum())}  whole graph {int(yo.sum())}/{n_graph}")

    pdi = sorted([c for c in df.columns if re.fullmatch(r"pdi_?q?\d+", c, re.I)],
                 key=lambda c: int(re.findall(r"\d+", c)[-1]))
    if len(pdi) == PDIN:
        csv_p = df[pdi].apply(pd.to_numeric, errors="coerce").values
        gp = Xo[best][:, EMB:EMB + PDIN].numpy()
        miss = np.isnan(csv_p) & k[:, None]
        if miss.any():
            vals = gp[miss]
            print(f"PDI cells missing in the CSV: {int(miss.sum())}; the graph holds "
                  f"{np.unique(np.round(vals, 3))[:6]} there (mean {vals.mean():.2f}) "
                  f"- that is what the original pipeline filled them with")
        ok = (~np.isnan(csv_p)) & k[:, None]
        if ok.any():
            d = np.abs(csv_p[ok] - gp[ok])
            print(f"PDI agreement on cells present in both: max |diff| {d.max():.3f}, "
                  f"{int((d > 1e-6).sum())} of {int(ok.sum())} cells differ")
    print("--- end cross-check ---\n", flush=True)


def align_to_old_graph(df, args, embedder):
    old = torch.load(args.old_graph, map_location="cpu", weights_only=False)
    Xo = old["Woman"].x
    yo = old["Woman"].y.view(-1)
    print(f"old graph: {Xo.shape[0]} women, {Xo.shape[1]} features, "
          f"{int(yo.sum())} positive", flush=True)
    Vn = embed(embedder, df[args.text_col].astype(str).tolist(), bs=args.embed_batch)
    A = torch.nn.functional.normalize(Vn, dim=1)
    B = torch.nn.functional.normalize(Xo[:, :EMB], dim=1)
    S = A @ B.t()
    best = S.argmax(1)
    sim = S.max(1).values
    uniq = len(set(best.tolist()))
    print(f"alignment: median cosine {sim.median():.4f}  min {sim.min():.4f}  "
          f"unique matches {uniq}/{len(best)}", flush=True)
    if sim.median() < 0.95 or uniq < len(best) * 0.95:
        print("!! alignment is poor - the CSV narratives do not look like the ones "
              "used to build the published graph. Check --text-col and the file.",
              flush=True)
        if not args.force_align:
            sys.exit(1)
    keep = sim >= args.align_min
    if (~keep).any():
        print(f"   dropping {int((~keep).sum())} rows below --align-min", flush=True)
    audit_alignment(df, args, best, keep, Xo, yo, S)
    return best, keep, Xo, yo


def build_features_from_csv(df, args, embedder):
    pdi = sorted([c for c in df.columns if re.fullmatch(r"pdi_?q?\d+", c, re.I)],
                 key=lambda c: int(re.findall(r"\d+", c)[-1]))
    assert len(pdi) == PDIN, f"expected 13 PDI columns, found {pdi}"
    import pandas as pd
    V = embed(embedder, df[args.text_col].astype(str).tolist(), bs=args.embed_batch)
    Pdf = df[pdi].apply(pd.to_numeric, errors="coerce")
    Cdf = df[[args.comp_col]].apply(pd.to_numeric, errors="coerce")
    n_miss = int(Pdf.isna().sum().sum() + Cdf.isna().sum().sum())
    if n_miss:
        where = {k: int(v) for k, v in Pdf.isna().sum().items() if v}
        print(f"{n_miss} missing PDI/complication cells -> filling with '{args.impute}' "
              f"({where}, {args.comp_col}={int(Cdf.isna().sum().iloc[0])})", flush=True)
    if args.impute == "zero":
        Pdf, Cdf = Pdf.fillna(0), Cdf.fillna(0)
    else:
        Pdf, Cdf = Pdf.fillna(Pdf.median()), Cdf.fillna(Cdf.median())
    P = torch.tensor(Pdf.values, dtype=torch.float)
    C = torch.tensor(Cdf.values, dtype=torch.float)
    y = torch.tensor((df[args.pcl_col].astype(float).values >= args.cutoff).astype(int),
                     dtype=torch.long)
    return torch.cat([V, P, C], 1), y


def build_graph(df, recs, Xw, yw, args, embedder):
    ids = [str(r[args.id_col]) for _, r in df.iterrows()]
    per_woman = [recs.get(i, {"nodes": [], "edges": []}) for i in ids]

    df_count = collections.Counter()
    for rec in per_woman:
        df_count.update({nid for nid, _ in rec["nodes"]})
    max_df = args.max_df_frac * len(per_woman)
    kept = {k for k, c in df_count.items() if c >= args.min_df and c <= max_df}
    print(f"entities: {len(df_count)} unique, {len(kept)} kept "
          f"(min_df={args.min_df}, max_df={max_df:.0f} = "
          f"{args.max_df_frac:.0%} of narratives)", flush=True)
    if args.max_df_frac < 1.0:
        hubs = sorted(((c, k) for k, c in df_count.items() if c > max_df), reverse=True)
        print(f"  dropped {len(hubs)} hub entities: "
              + ", ".join(f"{k} ({c})" for c, k in hubs[:15]), flush=True)

    tvote = collections.defaultdict(collections.Counter)
    for rec in per_woman:
        for nid, nt in rec["nodes"]:
            if nid in kept:
                tvote[nid][nt] += 1
    ent_type = {k: v.most_common(1)[0][0] for k, v in tvote.items()}

    by_fam = collections.defaultdict(list)
    idx = {}
    for nid in sorted(kept):
        fam = ent_type[nid]
        idx[nid] = (fam, len(by_fam[fam]))
        by_fam[fam].append(nid)

    data = HeteroData()
    data["Woman"].x = Xw
    data["Woman"].y = yw
    for fam, names in by_fam.items():
        data[fam].x = embed(embedder, names, bs=args.embed_batch)
        data[fam].names = names

    buckets = collections.defaultdict(set)
    for wi, rec in enumerate(per_woman):
        for nid, _ in rec["nodes"]:
            if nid not in idx:
                continue
            fam, ei = idx[nid]
            buckets[("Woman", WOMAN_REL[fam], fam)].add((wi, ei))
        for s, r, d in rec["edges"]:
            if s not in idx or d not in idx:
                continue
            sf, si = idx[s]
            dfm, di = idx[d]
            buckets[(sf, r, dfm)].add((si, di))

    for k, pairs in buckets.items():
        data[k].edge_index = torch.tensor(sorted(pairs), dtype=torch.long).t().contiguous()
    return data


def describe(data, tag):
    n = sum(data[t].num_nodes for t in data.node_types)
    e = sum(data[t].edge_index.size(1) for t in data.edge_types)
    print(f"\n=== rebuilt graph {tag} ===")
    print(f"nodes={n}  node_types={len(data.node_types)}  "
          f"edges={e}  edge_types={len(data.edge_types)}")
    nW = data["Woman"].num_nodes
    deg = torch.zeros(nW)
    for (s, r, d) in data.edge_types:
        ei = data[s, r, d].edge_index
        if s == "Woman":
            deg += torch.bincount(ei[0], minlength=nW).float()
        if d == "Woman":
            deg += torch.bincount(ei[1], minlength=nW).float()
    print(f"edges per woman: mean {deg.mean():.1f}  median {deg.median():.0f}  "
          f"min {deg.min():.0f}  max {deg.max():.0f}  isolated: {int((deg==0).sum())}")
    sizes = sorted(((t, data[t].num_nodes) for t in data.node_types), key=lambda x: -x[1])
    print("node types:", ", ".join(f"{t}={c}" for t, c in sizes))
    rel = collections.Counter()
    for (s, r, d) in data.edge_types:
        rel[r] += data[s, r, d].edge_index.size(1)
    print("relations:", ", ".join(f"{r}={c}" for r, c in rel.most_common()))
    pairs = collections.defaultdict(set)
    for (s, r, d) in data.edge_types:
        if s == "Woman" and d != "Woman":
            ei = data[s, r, d].edge_index
            for w, e in zip(ei[0].tolist(), ei[1].tolist()):
                pairs[d].add((w, e))
    per_entity = []
    for d, ps in pairs.items():
        cnt = collections.Counter(e for _, e in ps)
        per_entity.extend(cnt.get(i, 0) for i in range(data[d].num_nodes))
    if per_entity:
        arr = np.asarray(per_entity, dtype=float)
        print(f"women per entity: mean {arr.mean():.2f}  max {arr.max():.0f}  "
              f"entities linked to >=2 women: {int((arr >= 2).sum())}/{len(arr)}")


def main(a):
    import pandas as pd
    df = pd.read_csv(a.csv)
    if a.text_col not in df.columns:
        sys.exit(f"--text-col '{a.text_col}' not in {list(df.columns)[:20]}")
    df = df[df[a.text_col].notna()].reset_index(drop=True)
    print(f"{len(df)} narratives in {a.csv}", flush=True)

    embedder = load_embedder(a)
    if a.old_graph:
        best, keep, Xo, yo = align_to_old_graph(df, a, embedder)
        df = df[keep.numpy()].reset_index(drop=True)
        rows = best[keep]
        Xw, yw = Xo[rows].clone(), yo[rows].clone()
    else:
        Xw, yw = build_features_from_csv(df, a, embedder)
    print(f"woman block: {tuple(Xw.shape)}  positives {int(yw.sum())}", flush=True)
    if a.align_only:
        print("\n--align-only: stopping before extraction.")
        return

    recs = run_extraction(df, a)
    data = build_graph(df, recs, Xw, yw, a, embedder)
    describe(data, f"(passes={a.passes}, min_votes={a.min_votes}, min_df={a.min_df})")
    torch.save(data, a.out)
    print(f"\nsaved {a.out}")
    print("next:")
    print(f"  python fair_benchmark.py --graph {a.out} --seeds 1 2 3 4 5 --out results_fair_rebuilt/")
    print(f"  python compact_gnn.py   --graph {a.out} --seeds 1 2 3 4 5 --out results_compact_rebuilt/")
    print(f"  python compact_gnn.py   --graph {a.out} --seeds 1 2 3 4 5 --no-pdi --out results_compact_rebuilt_nopdi/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="development cohort CSV")
    p.add_argument("--out", default="perignnosis_graph_rebuilt.pt")
    p.add_argument("--cache", default="rebuild_extraction.jsonl")
    p.add_argument("--old-graph", default="perignnosis_graph.pt",
                   help="reuse the published Woman features so only structure changes; "
                        "pass an empty string to build features from the CSV instead, "
                        "which is what the corrected 301-woman cohort needs")
    p.add_argument("--impute", choices=["median", "zero"], default="median",
                   help="how to fill missing PDI/complication cells when features are "
                        "built from the CSV ('zero' reproduces the original pipeline)")
    p.add_argument("--id-col", default="record_id")
    p.add_argument("--text-col", default="cb_delivery_narrative")
    p.add_argument("--pcl-col", default="spcl5_total")
    p.add_argument("--comp-col", default="complication")
    p.add_argument("--cutoff", type=int, default=32)
    p.add_argument("--model", default="llama3.1:8b")
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--passes", type=int, default=3)
    p.add_argument("--min-votes", type=int, default=2)
    p.add_argument("--min-df", type=int, default=2)
    p.add_argument("--max-df-frac", type=float, default=1.0,
                   help="drop entities appearing in more than this fraction of "
                        "narratives (hub removal; 1.0 keeps everything). Set this "
                        "BEFORE running, not after seeing the AUC.")
    p.add_argument("--embed-model", default="jinaai/jina-embeddings-v2-base-en")
    p.add_argument("--max-seq-length", type=int, default=8192)
    p.add_argument("--embed-batch", type=int, default=8)
    p.add_argument("--health-check", type=int, default=15,
                   help="abort if too many of the first N narratives come back empty")
    p.add_argument("--max-empty-frac", type=float, default=0.5)
    p.add_argument("--align-min", type=float, default=0.90)
    p.add_argument("--force-align", action="store_true")
    p.add_argument("--align-only", action="store_true",
                   help="run the CSV-vs-graph cross-check and stop (a few minutes)")
    main(p.parse_args())
