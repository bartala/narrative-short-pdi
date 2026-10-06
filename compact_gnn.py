"""
Compact heterogeneous graph model on the knowledge graph.

Usage:
    python compact_gnn.py --graph perignnosis_graph_norm.pt --seeds 1 2 3 4 5 --out results_compact/
    python compact_gnn.py --graph perignnosis_graph_norm.pt --no-pdi --out results_compact_nopdi/
"""

import argparse, itertools, os, time
import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GCNConv
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from fair_benchmark import EMB, PDIN, COMP, TOTAL, PDI6, C_GRID, sanitize, set_seed

GRID = {"hidden": [32, 64], "dropout": [0.0, 0.3], "lr": [1e-3, 5e-4]}
TUNE_EPOCHS, FINAL_EPOCHS, WD = 30, 60, 1e-2


def perm_test(y, p1, p2, n=10000, seed=0):
    rng = np.random.RandomState(seed)
    obs = roc_auc_score(y, p1) - roc_auc_score(y, p2)
    hits = 0
    for _ in range(n):
        s = rng.rand(len(y)) < 0.5
        a = np.where(s, p2, p1)
        b = np.where(s, p1, p2)
        if abs(roc_auc_score(y, a) - roc_auc_score(y, b)) >= abs(obs) - 1e-12:
            hits += 1
    return obs, (hits + 1) / (n + 1)


def to_homogeneous(data, woman):
    node_types = list(data.node_types)
    width = max([TOTAL] + [data[nt].x.size(1) for nt in node_types])
    offset, xs, tids = {}, [], []
    n = 0
    for i, nt in enumerate(node_types):
        x = data[nt].x
        if x.size(1) < width:
            x = torch.cat([x, torch.zeros(x.size(0), width - x.size(1))], 1)
        offset[nt] = n
        n += x.size(0)
        xs.append(x)
        tids.append(torch.full((x.size(0),), i, dtype=torch.long))
    X = torch.cat(xs, 0)
    T = torch.cat(tids, 0)
    rel_names = sorted({r for (_, r, _) in data.edge_types})
    rel_id = {r: i for i, r in enumerate(rel_names)}
    src, dst, rel = [], [], []
    for (s, r, d) in data.edge_types:
        ei = data[s, r, d].edge_index
        if ei is None or ei.numel() == 0:
            continue
        src.append(ei[0] + offset[s]); dst.append(ei[1] + offset[d])
        rel.append(torch.full((ei.size(1),), rel_id[r], dtype=torch.long))
    edge_index = torch.stack([torch.cat(src), torch.cat(dst)])
    edge_type = torch.cat(rel)
    woman_idx = torch.arange(data[woman].num_nodes) + offset[woman]
    return (X, T, edge_index, edge_type, woman_idx,
            len(node_types), len(rel_names), width)


class CompactGNN(nn.Module):

    def __init__(self, n_types, n_rels, hidden=64, dropout=0.0, use_clinical=True,
                 in_dim=TOTAL, conv="gatv2"):
        super().__init__()
        self.use_clinical = use_clinical
        self.dropout = dropout
        self.conv_kind = conv
        self.enc = nn.Linear(in_dim, hidden)
        self.type_emb = nn.Embedding(n_types, hidden)
        self.norm = nn.LayerNorm(hidden)
        if conv == "gcn":
            self.rel_emb = None
            self.conv1 = GCNConv(hidden, hidden, add_self_loops=True)
            self.conv2 = GCNConv(hidden, hidden, add_self_loops=True)
        else:
            self.rel_emb = nn.Embedding(n_rels, hidden)
            self.conv1 = GATv2Conv(hidden, hidden, heads=2, concat=False,
                                   edge_dim=hidden, add_self_loops=True)
            self.conv2 = GATv2Conv(hidden, hidden, heads=1, concat=False,
                                   edge_dim=hidden, add_self_loops=True)
        head_in = hidden + (PDIN + COMP if use_clinical else 0)
        self.head = nn.Linear(head_in, 2)

    def forward(self, X, T, edge_index, edge_type, woman_idx):
        h = self.norm(self.enc(X) + self.type_emb(T))
        if self.conv_kind == "gcn":
            h1 = F.dropout(F.relu(self.conv1(h, edge_index)), self.dropout, self.training)
            h2 = F.dropout(F.relu(self.conv2(h1, edge_index)), self.dropout, self.training)
        else:
            e = self.rel_emb(edge_type)
            h1 = F.dropout(F.relu(self.conv1(h, edge_index, edge_attr=e)),
                           self.dropout, self.training)
            h2 = F.dropout(F.relu(self.conv2(h1, edge_index, edge_attr=e)),
                           self.dropout, self.training)
        z = h2[woman_idx] + h[woman_idx]
        if self.use_clinical:
            z = torch.cat([z, X[woman_idx][:, EMB:EMB + PDIN + COMP]], -1)
        return self.head(z)


def mask_test_edges(edge_index, edge_type, woman_idx, test_local):
    blocked = torch.zeros(int(edge_index.max()) + 1, dtype=torch.bool)
    blocked[woman_idx[torch.as_tensor(test_local, dtype=torch.long)]] = True
    keep = ~blocked[edge_index[0]]
    return edge_index[:, keep], edge_type[keep]


def run_fold(pack, tr, te, y_t, device, seed, use_clinical, conv="gatv2"):
    X, T, ei, et, widx, n_types, n_rels, in_dim = pack
    ei, et = mask_test_edges(ei, et, widx, te)
    X, T, ei, et, widx = X.to(device), T.to(device), ei.to(device), et.to(device), widx.to(device)
    sub_tr, sub_val = train_test_split(tr, test_size=0.2, stratify=y_t[tr].cpu(), random_state=seed)
    best, best_cfg = -1, None
    keys, vals = zip(*GRID.items())
    for combo in itertools.product(*vals):
        cfg = dict(zip(keys, combo))
        m = CompactGNN(n_types, n_rels, cfg["hidden"], cfg["dropout"],
                       use_clinical, in_dim, conv).to(device)
        opt = torch.optim.AdamW(m.parameters(), lr=cfg["lr"], weight_decay=WD)
        for _ in range(TUNE_EPOCHS):
            m.train(); opt.zero_grad()
            F.cross_entropy(m(X, T, ei, et, widx)[sub_tr], y_t[sub_tr]).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            p = F.softmax(m(X, T, ei, et, widx)[sub_val], 1)[:, 1].cpu().numpy()
        a = roc_auc_score(y_t[sub_val].cpu().numpy(), p)
        if a > best:
            best, best_cfg = a, cfg
        del m, opt
    m = CompactGNN(n_types, n_rels, best_cfg["hidden"], best_cfg["dropout"],
                   use_clinical, in_dim, conv).to(device)
    n_par = sum(p.numel() for p in m.parameters())
    opt = torch.optim.AdamW(m.parameters(), lr=best_cfg["lr"], weight_decay=WD)
    for _ in range(FINAL_EPOCHS):
        m.train(); opt.zero_grad()
        F.cross_entropy(m(X, T, ei, et, widx)[tr], y_t[tr]).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        p = F.softmax(m(X, T, ei, et, widx)[te], 1)[:, 1].cpu().numpy()
    return p, n_par


def main(a):
    os.makedirs(a.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = torch.load(a.graph, map_location="cpu", weights_only=False)
    data, woman = sanitize(raw, legacy=a.legacy_sanitize)
    y = data[woman].y.view(-1).numpy().astype(int)
    Xw = data[woman].x.numpy()
    Xw_full = data[woman].x.numpy().copy()

    keep_pdi = list(range(PDIN))
    if a.pdi_subset:
        keep_pdi = sorted({int(t) for t in a.pdi_subset.replace(",", " ").split()})
        assert all(0 <= i < PDIN for i in keep_pdi), f"PDI indices out of range: {keep_pdi}"
        drop = [EMB + i for i in range(PDIN) if i not in keep_pdi]
        data[woman].x = data[woman].x.clone()
        data[woman].x[:, drop] = 0.0
        Xw = data[woman].x.numpy()
        print(f"PDI subset: items {[i + 1 for i in keep_pdi]} kept, "
              f"{len(drop)} zeroed", flush=True)
    if a.no_pdi:
        data[woman].x = data[woman].x.clone()
        data[woman].x[:, EMB:] = 0.0
        Xw = data[woman].x.numpy()
    pack = to_homogeneous(data, woman)
    print(f"homogeneous graph: {pack[0].size(0)} nodes, {pack[2].size(1)} edges, "
          f"{pack[5]} node types, {pack[6]} relation types, width {pack[7]}, "
          f"device={device}", flush=True)

    rows = []
    for seed in a.seeds:
        t0 = time.time(); set_seed(seed)
        y_t = torch.as_tensor(y, dtype=torch.long, device=device)
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        tail = [EMB + PDIN] if a.lr_include_comp else []
        feat_cols = list(range(EMB)) + tail if a.no_pdi else \
            list(range(EMB)) + [EMB + i for i in keep_pdi] + tail
        tracks = {"PeriGNNosis-Compact": np.zeros(len(y)),
                  "LR (same features)": np.zeros(len(y))}
        if not a.no_pdi:
            tracks["LR (PDI items only)"] = np.zeros(len(y))
            tracks["LR (PDI-13)"] = np.zeros(len(y))
        if a.lr_include_comp:
            tracks["LR (COMP slot only)"] = np.zeros(len(y))
        n_par = 0
        for fold, (tr, te) in enumerate(skf.split(Xw, y)):
            set_seed(seed + fold)
            tracks["PeriGNNosis-Compact"][te], n_par = run_fold(
                pack, tr, te, y_t, device, seed, use_clinical=not a.no_pdi, conv=a.conv)

            def lr_oof(cols, src=None):
                src = Xw if src is None else src
                gs = GridSearchCV(
                    Pipeline([("sc", StandardScaler()),
                              ("clf", LogisticRegression(class_weight="balanced",
                                                         solver="liblinear",
                                                         max_iter=2000))]),
                    C_GRID, scoring="roc_auc",
                    cv=StratifiedKFold(3, shuffle=True, random_state=seed), n_jobs=-1
                ).fit(src[tr][:, cols], y[tr])
                return gs.best_estimator_.predict_proba(src[te][:, cols])[:, 1]

            tracks["LR (same features)"][te] = lr_oof(feat_cols)
            if not a.no_pdi:
                tracks["LR (PDI items only)"][te] = lr_oof(
                    [EMB + i for i in keep_pdi] + tail)
                tracks["LR (PDI-13)"][te] = lr_oof(
                    list(range(EMB, EMB + PDIN)) + tail, src=Xw_full)
            if a.lr_include_comp:
                tracks["LR (COMP slot only)"][te] = lr_oof([EMB + PDIN],
                                                           src=Xw_full)
            print(f"  seed {seed} fold {fold+1}/5 ({time.time()-t0:.0f}s)", flush=True)

        for name, p in tracks.items():
            rows.append({"seed": seed, "model": name, "auc": roc_auc_score(y, p)})
        pd.DataFrame(tracks | {"y_true": y}).to_csv(
            os.path.join(a.out, f"oof_seed{seed}.csv"), index=False)
        print("  seed %d: " % seed + " | ".join(
            f"{n} {roc_auc_score(y, p):.4f}" for n, p in tracks.items())
            + f" | params {n_par:,}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "compact_auc.csv"), index=False)
    print("\n=== pooled OOF AUC ===")
    print(df.groupby("model")["auc"].agg(["mean", "std"]).round(4).to_string())
    print("\n=== PeriGNNosis-Compact vs each baseline "
          "(paired permutation on OOF probabilities, 10,000 draws) ===")
    for name in [m for m in df.model.unique() if m != "PeriGNNosis-Compact"]:
        ds, ps = [], []
        for seed in a.seeds:
            o = pd.read_csv(os.path.join(a.out, f"oof_seed{seed}.csv"))
            d, pv = perm_test(o["y_true"].values, o["PeriGNNosis-Compact"].values,
                              o[name].values, n=10000, seed=seed)
            ds.append(d); ps.append(pv)
        print(f"  vs {name}: Δ {np.mean(ds):+.4f}  perm p {min(ps):.4f}-{max(ps):.4f}"
              f"  wins {sum(d > 0 for d in ds)}/{len(ds)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="perignnosis_graph_norm.pt")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--out", default="results_compact")
    ap.add_argument("--conv", choices=["gatv2", "gcn"], default="gatv2",
                    help="gatv2 = attention with relation embeddings (default); "
                         "gcn = no attention, no relation types, mean aggregation")
    ap.add_argument("--no-pdi", action="store_true", help="no-questionnaire setting")
    ap.add_argument("--lr-include-comp", action="store_true",
                    help="give the LR baselines the COMP slot as well (the "
                         "complication flag, or the cohort indicator in a pooled "
                         "run). Required for the pooled analysis to be fair.")
    ap.add_argument("--pdi-subset", default="",
                    help="zero-based PDI item indices to keep, e.g. '0,1,2,4,11,12'. "
                         "Derive the set with pdi_subset_analysis.py first - do not "
                         "reuse the set from the original submission, which was "
                         "selected on the flawed cohort.")
    ap.add_argument("--legacy-sanitize", action="store_true")
    main(ap.parse_args())
