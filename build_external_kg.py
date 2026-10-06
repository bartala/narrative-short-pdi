"""
Build a knowledge graph for a second cohort, aligned to the development graph schema.

Usage:
    python build_external_kg.py --csv external_cohort.csv --dev-graph perignnosis_graph.pt --out external_graph.pt
"""

import argparse, json, os, re, time
import numpy as np, pandas as pd, torch
from torch_geometric.data import HeteroData
from tqdm import tqdm

EMB, PDIN, COMP = 768, 13, 1
MIN_WORDS = 30


def norm_type(s):
    return re.sub(r"[^0-9A-Za-z_]", "_", str(s).strip()).strip("_") or "Entity"


def main(a):
    dev = torch.load(a.dev_graph, map_location="cpu", weights_only=False)
    dev_nodes = set(dev.node_types)
    dev_rels = {r for (_, r, _) in dev.edge_types}
    print(f"development graph: {len(dev_nodes)} node types, {len(dev_rels)} relation types", flush=True)

    df = pd.read_csv(a.csv)
    pdi_cols = [c for c in df.columns if re.fullmatch(r"pdi_q?\d+", c)]
    pdi_cols = sorted(pdi_cols, key=lambda c: int(re.findall(r"\d+", c)[-1]))
    assert len(pdi_cols) == PDIN, f"expected 13 PDI columns, found {pdi_cols}"
    df["word_count"] = df[a.text_col].fillna("").astype(str).str.split().str.len()
    df = df[(df.word_count >= MIN_WORDS) & df[pdi_cols].notna().all(axis=1)
            & df[a.label_col].notna()].reset_index(drop=True)
    df["record_id"] = df["record_id"].astype(str)
    print(f"external cohort: {len(df)} women, positives={int(df[a.label_col].sum())}", flush=True)

    from langchain_core.documents import Document
    from langchain_experimental.graph_transformers import LLMGraphTransformer
    from langchain_ollama import ChatOllama

    llm = ChatOllama(model=a.llm, temperature=0.0, num_ctx=8192)
    transformer = LLMGraphTransformer(llm=llm, allowed_nodes=[], strict=False)

    cache_path = a.out + ".extraction.jsonl"
    done = {}
    if os.path.exists(cache_path):
        for line in open(cache_path):
            rec = json.loads(line)
            done[rec["record_id"]] = rec
        print(f"resuming: {len(done)} narratives already extracted", flush=True)

    with open(cache_path, "a") as fh:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="LLM extraction"):
            rid = row["record_id"]
            if rid in done:
                continue
            doc = Document(page_content=str(row[a.text_col]), metadata={"id": rid, "record_id": rid})
            try:
                gdocs = transformer.convert_to_graph_documents([doc])
                nodes = [{"id": str(n.id), "type": norm_type(n.type)} for g in gdocs for n in g.nodes]
                rels = [{"src": str(r.source.id), "src_type": norm_type(r.source.type),
                         "dst": str(r.target.id), "dst_type": norm_type(r.target.type),
                         "rel": norm_type(r.type).upper()} for g in gdocs for r in g.relationships]
            except Exception as e:
                nodes, rels = [], []
                print(f"  extraction failed for {rid}: {e}", flush=True)
            rec = {"record_id": rid, "nodes": nodes, "rels": rels}
            done[rid] = rec
            fh.write(json.dumps(rec) + "\n"); fh.flush()

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(a.embed_model, trust_remote_code=True)
    try:
        model.max_seq_length = a.max_seq
    except Exception:
        pass
    print("embedding narratives...", flush=True)
    narr = model.encode(df[a.text_col].astype(str).tolist(), normalize_embeddings=True,
                        batch_size=a.batch, show_progress_bar=True)

    entity_texts, entity_key = [], {}
    for rid, rec in done.items():
        for n in rec["nodes"]:
            t = norm_type(n["type"]); t = t if t in dev_nodes else "__Entity__"
            key = (t, n["id"])
            if key not in entity_key:
                entity_key[key] = len(entity_texts); entity_texts.append(n["id"])
    print(f"embedding {len(entity_texts)} entity nodes...", flush=True)
    ent_emb = (model.encode(entity_texts, normalize_embeddings=True, batch_size=a.batch,
                            show_progress_bar=True) if entity_texts else np.zeros((0, EMB)))

    data = HeteroData()
    comp = (df[a.comp_col].fillna(0).astype(float).values if a.comp_col in df.columns
            else np.zeros(len(df)))
    if a.comp_col not in df.columns:
        print("note: obstetric complication column absent - filled with 0", flush=True)
    woman_x = np.hstack([narr, df[pdi_cols].values.astype(float), comp.reshape(-1, 1)])
    data["Woman"].x = torch.tensor(woman_x, dtype=torch.float)
    data["Woman"].y = torch.tensor(df[a.label_col].values.astype(int), dtype=torch.long)
    data["Woman"].record_id = df["record_id"].tolist()
    woman_idx = {r: i for i, r in enumerate(df["record_id"])}

    by_type = {}
    for (t, eid), pos in entity_key.items():
        by_type.setdefault(t, []).append((eid, pos))
    local_idx = {}
    for t, items in by_type.items():
        data[t].x = torch.tensor(np.stack([ent_emb[p] for _, p in items]), dtype=torch.float)
        for i, (eid, _) in enumerate(items):
            local_idx[(t, eid)] = i

    def resolve(eid, etype):
        t = norm_type(etype); t = t if t in dev_nodes else "__Entity__"
        return (t, local_idx.get((t, eid)))

    edges = {}
    for rid, rec in done.items():
        if rid not in woman_idx:
            continue
        w = woman_idx[rid]
        for n in rec["nodes"]:
            t, i = resolve(n["id"], n["type"])
            if i is None:
                continue
            edges.setdefault(("Woman", "MENTIONS", t), [[], []])
            edges[("Woman", "MENTIONS", t)][0].append(w)
            edges[("Woman", "MENTIONS", t)][1].append(i)
        for r in rec["rels"]:
            st, si = resolve(r["src"], r["src_type"])
            dt, di = resolve(r["dst"], r["dst_type"])
            if si is None or di is None:
                continue
            rel = r["rel"] if r["rel"] in dev_rels else "MENTIONS"
            edges.setdefault((st, rel, dt), [[], []])
            edges[(st, rel, dt)][0].append(si)
            edges[(st, rel, dt)][1].append(di)
    for k, (s, d) in edges.items():
        data[k].edge_index = torch.tensor([s, d], dtype=torch.long)

    n_edges = sum(data[e].edge_index.size(1) for e in data.edge_types)
    print(f"external graph: {sum(data[n].num_nodes for n in data.node_types)} nodes, "
          f"{len(data.node_types)} node types, {len(data.edge_types)} edge types, {n_edges} edges", flush=True)
    torch.save(data, a.out)
    df.to_csv(a.out.replace(".pt", "_metadata.csv"), index=False)
    print("saved", a.out, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--dev-graph", default="perignnosis_graph.pt")
    ap.add_argument("--out", default="external_graph.pt")
    ap.add_argument("--text-col", default="cb_delivery_narrative")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--comp-col", default="obstetric_complication")
    ap.add_argument("--llm", default="llama3.1:8b")
    ap.add_argument("--embed-model", default="jinaai/jina-embeddings-v2-base-en")
    ap.add_argument("--max-seq", type=int, default=8192)
    ap.add_argument("--batch", type=int, default=16)
    main(ap.parse_args())
