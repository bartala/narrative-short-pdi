"""
Re-embed knowledge-graph entity nodes with SapBERT.

Usage:
    python sapbert_entities.py --graph perignnosis_graph_pooled.pt --out perignnosis_graph_pooled_sapbert.pt
    python pca_gnn.py --graph perignnosis_graph_pooled_sapbert.pt --out results_pca_gnn_sapbert/
"""

import argparse, os
import numpy as np, torch
import torch.nn.functional as F

EMB = 768
DEFAULT_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"


def encode(names, tok, model, device, bs=128, max_len=25, pooling="cls"):
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(names), bs):
            b = tok(list(names[i:i + bs]), padding=True, truncation=True,
                    max_length=max_len, return_tensors="pt").to(device)
            h = model(**b).last_hidden_state
            if pooling == "cls":
                v = h[:, 0]
            else:
                m = b["attention_mask"].unsqueeze(-1).float()
                v = (h * m).sum(1) / m.sum(1)
            out.append(F.normalize(v, dim=1).cpu())
    return torch.cat(out, 0)


def neighbours(E, names, probe, k=5):
    E = F.normalize(E, dim=1)
    i = names.index(probe)
    s = E @ E[i]
    s[i] = -2
    top = torch.topk(s, min(k, len(names) - 1)).indices.tolist()
    return ", ".join(names[j] for j in top)


def main(a):
    if a.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoTokenizer, AutoModel

    g = torch.load(a.graph, map_location="cpu", weights_only=False)
    ent_types = [nt for nt in g.node_types if nt != "Woman"]
    missing = [nt for nt in ent_types if getattr(g[nt], "names", None) is None]
    if missing:
        raise SystemExit(f"no entity names stored for {missing}. This script needs a "
                         f"graph built by rebuild_dev_kg.py (it stores data[type].names).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=a.offline)
    model = AutoModel.from_pretrained(a.model, local_files_only=a.offline).to(device)
    hid = model.config.hidden_size
    if hid != EMB:
        raise SystemExit(f"{a.model} gives {hid}-dim vectors; the graph expects {EMB}")

    all_names, old_all, new_all = [], [], []
    for nt in ent_types:
        names = [str(n) for n in g[nt].names]
        x = g[nt].x
        if len(names) != x.size(0):
            raise SystemExit(f"{nt}: {len(names)} names but {x.size(0)} nodes")
        new = encode(names, tok, model, device, a.batch, a.max_len, a.pooling)
        old_all.append(x[:, :EMB].clone()); new_all.append(new); all_names += names
        x = x.clone()
        x[:, :EMB] = new
        g[nt].x = x
        print(f"  {nt:<22} {len(names):>4} entities re-embedded", flush=True)

    old = torch.cat(old_all); new = torch.cat(new_all)
    n = len(all_names)
    iu = torch.triu_indices(n, n, 1)
    for tag, E in [("old", old), ("new", new)]:
        S = F.normalize(E, dim=1) @ F.normalize(E, dim=1).T
        print(f"mean pairwise cosine ({tag}): {S[iu[0], iu[1]].mean():.3f}  "
              f"(lower = entities better spread out)")

    probes = [p for p in a.probe if p in all_names] or all_names[:3]
    print("\nnearest neighbours, old -> new (sanity check):")
    for p in probes:
        print(f"  {p}\n    old: {neighbours(old, all_names, p)}"
              f"\n    new: {neighbours(new, all_names, p)}")

    g.entity_encoder = a.model
    torch.save(g, a.out)
    print(f"\n{n} entities across {len(ent_types)} types; Woman nodes unchanged.")
    print(f"saved {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="perignnosis_graph_pooled.pt")
    ap.add_argument("--out", default="perignnosis_graph_pooled_sapbert.pt")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--pooling", choices=["cls", "mean"], default="cls",
                    help="SapBERT uses cls; use mean for e.g. BioLORD")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--max-len", type=int, default=25)
    ap.add_argument("--offline", action="store_true",
                    help="use only locally cached weights; no network at all")
    ap.add_argument("--probe", nargs="*",
                    default=["epidural", "emergency cesarean", "fear", "pain",
                             "postpartum hemorrhage"])
    main(ap.parse_args())
