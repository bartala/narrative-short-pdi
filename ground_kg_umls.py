"""
Ground knowledge-graph entities in UMLS concepts.

Usage:
    python ground_kg_umls.py --graph perignnosis_graph_rebuilt.pt --out perignnosis_graph_grounded.pt
    python ground_kg_umls.py --graph ... --out ... --no-linker
    python ground_kg_umls.py --graph ... --dump-links links.csv --diagnose-only
"""

import argparse, collections, json, os, re, sys
import numpy as np
import torch
from torch_geometric.data import HeteroData

EMB = 768

CATEGORIES = [
    ("obstetric_emergency",
     r"\bemergency (cesarean|c section|caesarean)\b|\bcrash (section|cesarean)\b"
     r"|\b(uterine )?rupture\b|\bplacental abruption\b|\bcord prolapse\b"
     r"|\bshoulder dystocia\b|\bfetal distress\b", 1),
    ("hemorrhage",
     r"\b(postpartum )?h(a)?emorrhage\b|\bblood loss\b|\btransfusion\b", 1),
    ("neonatal_compromise",
     r"\bneonatal intensive care\b|\bnicu\b|\bbaby (was )?taken\b|\bseparat"
     r"|\bresuscitat|\bstillbirth\b|\bneonatal death\b|\bbaby died\b", 1),
    ("hypertensive_event",
     r"\bpre ?eclampsia\b|\beclampsia\b|\bhellp\b|\bseizure\b", 1),
    ("instrumental_delivery",
     r"\bforceps\b|\bvacuum\b|\bventouse\b|\binstrumental delivery\b", 1),
    ("maternal_critical_care",
     r"\bgeneral an(a)?esthesia\b|\bintubat|\bintensive care unit\b|\bicu\b"
     r"|\bhysterectomy\b|\breoperation\b", 1),
    ("perineal_trauma",
     r"\b(third|fourth) degree tear\b|\bsevere tear\b|\bepisiotomy\b", 1),
    ("life_threat",
     r"\bfear of dying\b|\bthought (i|we) (would|might) die\b|\bnear death\b"
     r"|\bthought (i|we) (was|were) dying\b", 0),
    ("dissociation",
     r"\bdissociat|\bunreal\b|\bout of (my )?body\b|\bdetach|\bnumb\b", 0),
    ("loss_of_control",
     r"\bhelpless|\bpowerless|\bout of control\b|\blost control\b", 0),
    ("acute_fear",
     r"\bterror\b|\bpanic\b|\bhorror\b|\bterrified\b", 0),
    ("care_dismissal",
     r"\bnot listened to\b|\bignored\b|\bdismissed\b|\bnot believed\b", 0),
    ("severe_pain",
     r"\bsevere pain\b|\bunbearable pain\b|\bexcruciating\b", 0),
    ("abandonment",
     r"\babandoned\b|\bno one (came|helped)\b|\bleft alone\b", 0),
    ("shame_violation",
     r"\bshame\b|\bhumiliat|\bviolat|\bassault", 0),
    ("self_blame",
     r"\bguilt\b|\bfail(ed|ure)\b|\bblame", 0),
]
CAT_RE = [(n, re.compile(p, re.I), ob) for n, p, ob in CATEGORIES]

SEM_GROUPS = ["DISO", "PROC", "CHEM", "ANAT", "DEVI", "LIVB",
              "PHYS", "PHEN", "ACTI", "ORGG", "OTHER"]
TUI_GROUP = {}
for g, tuis in {
    "DISO": "T020 T190 T049 T019 T047 T050 T033 T037 T048 T191 T046 T184",
    "PROC": "T060 T065 T058 T059 T063 T062 T061",
    "CHEM": ("T116 T195 T123 T122 T103 T120 T104 T200 T196 T126 T131 T125 "
             "T129 T130 T197 T114 T109 T121 T192 T127"),
    "ANAT": "T017 T029 T023 T030 T031 T022 T025 T026 T018 T021 T024",
    "DEVI": "T203 T074 T075",
    "LIVB": ("T001 T002 T004 T005 T007 T008 T010 T011 T012 T013 T014 T015 T016 "
             "T096 T097 T098 T099 T100 T101 T032"),
    "PHYS": "T039 T040 T041 T042 T043 T044 T045 T201 T034",
    "PHEN": "T038 T069 T068 T070 T067",
    "ACTI": "T052 T053 T056 T051 T064 T055 T066 T057 T054",
    "ORGG": "T092 T093 T094 T095 T083",
}.items():
    for t in tuis.split():
        TUI_GROUP[t] = g


CAT_OFF = 2
SEM_OFF = CAT_OFF + len(CATEGORIES)
EXTRA = SEM_OFF + len(SEM_GROUPS)
ATTR_NAMES = (["risk_factor", "obstetric"] + [f"cat_{n}" for n, _, _ in CATEGORIES]
              + [f"sem_{g}" for g in SEM_GROUPS])


def attribute_block(surface, canonical, tuis):
    v = np.zeros(EXTRA, dtype="float32")
    text = f"{surface} {canonical or ''}"
    for i, (name, rx, ob) in enumerate(CAT_RE):
        if rx.search(text):
            v[CAT_OFF + i] = 1.0
            v[0] = 1.0
            v[1] = max(v[1], float(ob))
    groups = {TUI_GROUP.get(t, "OTHER") for t in (tuis or [])} or {"OTHER"}
    for g in groups:
        v[SEM_OFF + SEM_GROUPS.index(g)] = 1.0
    return v


def load_linker(args):
    if args.no_linker:
        return None, None
    try:
        import spacy, scispacy
        from scispacy.linking import EntityLinker
    except Exception as ex:
        sys.exit(f"scispaCy is not installed ({type(ex).__name__}: {ex}).\n"
                 f"  pip install scispacy\n"
                 f"  pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/"
                 f"releases/v0.5.4/en_core_sci_md-0.5.4.tar.gz\n"
                 f"Or rerun with --no-linker for lexicon-only grounding.")
    import spacy
    nlp = spacy.load(args.spacy_model)
    nlp.add_pipe("scispacy_linker", config={
        "resolve_abbreviations": True, "linker_name": "umls",
        "max_entities_per_mention": 1, "threshold": args.link_threshold})
    linker = nlp.get_pipe("scispacy_linker")
    return nlp, linker


def link_entities(names, nlp, linker, args):
    out = {}
    if nlp is None:
        return {n: {} for n in names}
    for i, name in enumerate(names):
        doc = nlp(name)
        best = None
        for ent in doc.ents:
            for cui, score in ent._.kb_ents:
                if best is None or score > best[1]:
                    best = (cui, score)
        if best and best[1] >= args.link_threshold:
            cui, score = best
            e = linker.kb.cui_to_entity[cui]
            out[name] = {"cui": cui, "canonical": e.canonical_name,
                         "tuis": list(e.types), "score": float(score)}
        else:
            out[name] = {}
        if (i + 1) % 200 == 0:
            print(f"  linked {i+1}/{len(names)}", flush=True)
    return out


def collect_names(data):
    names = {}
    for nt in data.node_types:
        if nt == "Woman":
            continue
        n = getattr(data[nt], "names", None)
        if n is None:
            sys.exit(f"node type '{nt}' has no surface forms stored. Grounding needs "
                     f"a graph from rebuild_dev_kg.py, which saves data[type].names.")
        names[nt] = list(n)
    return names


def ground(data, links, args):
    names = collect_names(data)

    ident, mapping = {}, {}
    for nt, ns in names.items():
        for i, nm in enumerate(ns):
            L = links.get(nm, {})
            cui = L.get("cui")
            key = ("cui", cui) if (cui and not args.no_merge) else ("str", nt, nm)
            if key not in ident:
                ident[key] = {"type": nt, "surface": nm, "link": L,
                              "rows": [], "idx": None}
            ident[key]["rows"].append((nt, i))
            mapping[(nt, i)] = key

    by_type = collections.defaultdict(list)
    for key, rec in ident.items():
        rec["idx"] = len(by_type[rec["type"]])
        by_type[rec["type"]].append(rec)

    new = HeteroData()
    new["Woman"].x = data["Woman"].x
    new["Woman"].y = data["Woman"].y

    for nt, recs in by_type.items():
        base = []
        for r in recs:
            src_t, src_i = r["rows"][0]
            base.append(data[src_t].x[src_i][:EMB])
        E = torch.stack(base)
        A = torch.tensor(np.stack([
            attribute_block(r["surface"], r["link"].get("canonical"),
                            r["link"].get("tuis")) for r in recs]))
        new[nt].x = torch.cat([E, A], 1)
        new[nt].names = [r["surface"] for r in recs]
        new[nt].cuis = [r["link"].get("cui") or "" for r in recs]

    buckets = collections.defaultdict(set)
    for (s, r, d) in data.edge_types:
        ei = data[s, r, d].edge_index
        for a, b in zip(ei[0].tolist(), ei[1].tolist()):
            if s == "Woman":
                sk, si = "Woman", a
            else:
                key = mapping.get((s, a))
                if key is None:
                    continue
                sk, si = ident[key]["type"], ident[key]["idx"]
            if d == "Woman":
                dk, di = "Woman", b
            else:
                key = mapping.get((d, b))
                if key is None:
                    continue
                dk, di = ident[key]["type"], ident[key]["idx"]
            buckets[(sk, r, dk)].add((si, di))

    if not args.no_hubs:
        hub_rows, hub_idx = [], {}
        for g in SEM_GROUPS:
            hub_idx[g] = len(hub_rows)
            v = np.zeros(EMB + EXTRA, dtype="float32")
            v[EMB + SEM_OFF + SEM_GROUPS.index(g)] = 1.0
            hub_rows.append(v)
        new["SemanticGroup"].x = torch.tensor(np.stack(hub_rows))
        new["SemanticGroup"].names = list(SEM_GROUPS)
        for nt, recs in by_type.items():
            for r in recs:
                tuis = r["link"].get("tuis") or []
                gs = {TUI_GROUP.get(t, "OTHER") for t in tuis} or {"OTHER"}
                for g in gs:
                    buckets[(nt, "IS_A", "SemanticGroup")].add((r["idx"], hub_idx[g]))

    for k, pairs in buckets.items():
        new[k].edge_index = torch.tensor(sorted(pairs), dtype=torch.long).t().contiguous()
    return new, ident


def sharing_stats(data):
    counts = []
    for (s, r, d) in data.edge_types:
        if s == "Woman" and d != "Woman":
            counts.append(torch.bincount(data[s, r, d].edge_index[1],
                                         minlength=data[d].num_nodes).float())
    if not counts:
        return 0.0, 0, 0
    c = torch.cat(counts)
    return float(c.mean()), int((c >= 2).sum()), len(c)


def describe(data, tag):
    n = sum(data[t].num_nodes for t in data.node_types)
    e = sum(data[t].edge_index.size(1) for t in data.edge_types)
    mean_share, shared, total = sharing_stats(data)
    print(f"\n=== {tag} ===")
    print(f"nodes={n}  node_types={len(data.node_types)}  edges={e}  "
          f"edge_types={len(data.edge_types)}")
    print(f"women per entity: mean {mean_share:.2f}  "
          f"entities reached by >=2 women: {shared}/{total}")


def main(a):
    data = torch.load(a.graph, map_location="cpu", weights_only=False)
    describe(data, "before grounding")

    names = collect_names(data)
    flat = sorted({n for ns in names.values() for n in ns})
    print(f"\n{len(flat)} unique entity surface forms", flush=True)

    nlp, linker = load_linker(a)
    links = link_entities(flat, nlp, linker, a)
    n_linked = sum(1 for v in links.values() if v.get("cui"))
    print(f"linked to UMLS: {n_linked}/{len(flat)} ({n_linked/max(len(flat),1):.0%})")
    n_risk = sum(1 for n in flat
                 if attribute_block(n, links[n].get("canonical"), None)[0] > 0)
    print(f"matched a curated risk factor: {n_risk}/{len(flat)}")

    if a.dump_links:
        import csv
        with open(a.dump_links, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["surface", "cui", "canonical", "tuis", "score",
                        "risk_factor", "obstetric", "categories"])
            for n in flat:
                L = links[n]
                v = attribute_block(n, L.get("canonical"), L.get("tuis"))
                cats = [nm for i, (nm, _, _) in enumerate(CATEGORIES)
                        if v[CAT_OFF + i] > 0]
                w.writerow([n, L.get("cui", ""), L.get("canonical", ""),
                            "|".join(L.get("tuis", [])), round(L.get("score", 0), 3),
                            int(v[0]), int(v[1]), "|".join(cats)])
        print(f"wrote {a.dump_links} - read it before training on this graph")
    if a.diagnose_only:
        return

    grounded, ident = ground(data, links, a)
    describe(grounded, "after grounding")
    merged = len(names and [x for ns in names.values() for x in ns]) - \
        sum(grounded[t].num_nodes for t in grounded.node_types
            if t not in ("Woman", "SemanticGroup"))
    print(f"entities merged by shared concept: {merged}")
    torch.save(grounded, a.out)
    print(f"\nsaved {a.out}")
    print("next:")
    print(f"  python fair_benchmark.py --graph {a.out} --seeds 1 2 3 4 5 --out results_grounded/")
    print(f"  python compact_gnn.py   --graph {a.out} --seeds 1 2 3 4 5 --out results_compact_grounded/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--graph", default="perignnosis_graph_rebuilt.pt")
    p.add_argument("--out", default="perignnosis_graph_grounded.pt")
    p.add_argument("--spacy-model", default="en_core_sci_md")
    p.add_argument("--link-threshold", type=float, default=0.80)
    p.add_argument("--no-linker", action="store_true",
                   help="skip UMLS, use the curated lexicon only")
    p.add_argument("--no-merge", action="store_true",
                   help="attach attributes but do not merge entities by concept")
    p.add_argument("--no-hubs", action="store_true")
    p.add_argument("--dump-links", default="umls_links.csv")
    p.add_argument("--diagnose-only", action="store_true")
    main(p.parse_args())
