"""
Benchmark of the graph neural network, feed-forward network, LR and SVM with nested cross-validation.

Usage:
    python fair_benchmark.py --graph perignnosis_graph.pt --seeds 1 2 3 4 5 --out results/
    python fair_benchmark.py --graph ... --legacy-sanitize
"""

import argparse, json, os, random, time
import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, HeteroConv, LayerNorm
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

EMB, PDIN, COMP = 768, 13, 1
TOTAL = EMB + PDIN + COMP
PDI6 = [0, 1, 2, 4, 11, 12]
NN_GRID = {"hidden_channels": [32, 64], "dropout": [0.0, 0.3], "lr": [1e-3, 5e-4]}
TUNE_EPOCHS, FINAL_EPOCHS, WD = 15, 30, 1e-2
C_GRID = {"clf__C": [0.01, 0.1, 1, 10, 100]}


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def sanitize(raw, legacy=False):
    raw = raw.cpu()
    node_map, used = {}, set()
    for nt in raw.node_types:
        if legacy:
            new = nt.replace(" ", "_").replace("-", "_") if nt != "__Entity__" else "Entity"
        else:
            new = "EntityAll" if nt == "__Entity__" else nt.replace(" ", "_").replace("-", "_")
            while new in used:
                new += "_x"
        used.add(new); node_map[nt] = new
    tmp = HeteroData()
    for nt, new in node_map.items():
        tmp[new].x = raw[nt].x
        tmp[new].num_nodes = int(raw[nt].x.size(0))
        if "y" in raw[nt]:
            tmp[new].y = raw[nt].y
    for (s, r, d) in raw.edge_types:
        tmp[node_map[s], r.replace(" ", "_").replace("-", "_"), node_map[d]].edge_index = raw[s, r, d].edge_index
    data = T.ToUndirected()(tmp)
    woman = node_map["Woman"]
    for nt in data.node_types:
        x = data[nt].x
        target = TOTAL if nt == woman else max(EMB, x.size(1))
        if x.size(1) < target:
            data[nt].x = torch.cat([x, torch.zeros(x.size(0), target - x.size(1))], 1)
        elif x.size(1) > target:
            data[nt].x = x[:, :target]
    for (s, r, d) in data.edge_types:
        ei = data[s, r, d].edge_index
        if ei is not None and ei.numel():
            m = (ei[0] < data[s].num_nodes) & (ei[1] < data[d].num_nodes)
            data[s, r, d].edge_index = ei[:, m]
    return data, woman


def block_edges(data, blocked_idx, woman):
    data = data.clone()
    blocked = torch.zeros(data[woman].num_nodes, dtype=torch.bool)
    blocked[torch.as_tensor(blocked_idx, dtype=torch.long)] = True
    for (s, r, d) in data.edge_types:
        if s == woman:
            ei = data[s, r, d].edge_index
            data[s, r, d].edge_index = ei[:, ~blocked[ei[0]]]
    return data


class HeteroGAT(nn.Module):
    def __init__(self, metadata, woman, hidden_channels=64, dropout=0.0, heads=4,
                 in_dims=None):
        super().__init__()
        self.woman, self.dropout = woman, dropout
        nts, ets = metadata
        in_dims = in_dims or {}
        self.encoder = nn.ModuleDict({
            nt: nn.Linear(in_dims.get(nt, TOTAL if nt == woman else EMB), hidden_channels)
            for nt in nts})
        self.norms = nn.ModuleDict({nt: LayerNorm(hidden_channels) for nt in nts})
        self.conv1 = HeteroConv({et: GATConv(hidden_channels, hidden_channels, heads=heads,
                                             concat=False, add_self_loops=False) for et in ets}, aggr="sum")
        self.conv2 = HeteroConv({et: GATConv(hidden_channels, hidden_channels, heads=1,
                                             concat=False, add_self_loops=False) for et in ets}, aggr="sum")
        self.res_lin = nn.Linear(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels + PDIN + COMP, 2)

    def forward(self, x_dict, edge_index_dict):
        clinical = x_dict[self.woman][:, EMB:EMB + PDIN + COMP]
        h = {nt: self.norms[nt](self.encoder[nt](x)) for nt, x in x_dict.items()}
        h0 = h[self.woman]
        x1 = self.conv1(h, edge_index_dict)
        x1 = {k: F.dropout(F.relu(x1.get(k, h[k])), p=self.dropout, training=self.training) for k in h}
        x2 = self.conv2(x1, edge_index_dict)
        w = F.dropout(F.relu(x2.get(self.woman, x1[self.woman])), p=self.dropout, training=self.training)
        return self.lin(torch.cat([w + self.res_lin(h0), clinical], -1))


class MLP(nn.Module):
    def __init__(self, in_dim, woman, hidden_channels=64, dropout=0.0):
        super().__init__()
        self.woman, self.in_dim = woman, in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_channels), nn.LayerNorm(hidden_channels), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_channels, 2))

    def forward(self, x_dict, edge_index_dict=None):
        return self.net(x_dict[self.woman][:, :self.in_dim] if self.in_dim <= EMB
                        else x_dict[self.woman])


def tune_train_nn(make_model, g, tr, te, y_t, device, woman, seed):
    import itertools
    sub_tr, sub_val = train_test_split(tr, test_size=0.2, stratify=y_t[tr].cpu(), random_state=seed)
    best, best_cfg = -1, None
    keys, vals = zip(*NN_GRID.items())
    for combo in itertools.product(*vals):
        cfg = dict(zip(keys, combo))
        m = make_model({k: v for k, v in cfg.items() if k != "lr"}).to(device)
        opt = torch.optim.AdamW(m.parameters(), lr=cfg["lr"], weight_decay=WD)
        for _ in range(TUNE_EPOCHS):
            m.train(); opt.zero_grad()
            F.cross_entropy(m(g.x_dict, g.edge_index_dict)[sub_tr], y_t[sub_tr]).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            p = F.softmax(m(g.x_dict, g.edge_index_dict)[sub_val], 1)[:, 1].cpu().numpy()
        a = roc_auc_score(y_t[sub_val].cpu().numpy(), p) if len(set(y_t[sub_val].tolist())) > 1 else 0.5
        if a > best:
            best, best_cfg = a, cfg
        del m, opt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    m = make_model({k: v for k, v in best_cfg.items() if k != "lr"}).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=best_cfg["lr"], weight_decay=WD)
    for _ in range(FINAL_EPOCHS):
        m.train(); opt.zero_grad()
        F.cross_entropy(m(g.x_dict, g.edge_index_dict)[tr], y_t[tr]).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        return F.softmax(m(g.x_dict, g.edge_index_dict)[te], 1)[:, 1].cpu().numpy()


def linear_oof(kind, X, y, tr, te, seed):
    clf = (LogisticRegression(class_weight="balanced", solver="liblinear", max_iter=2000, random_state=seed)
           if kind == "lr" else SVC(probability=True, class_weight="balanced", random_state=seed))
    gs = GridSearchCV(Pipeline([("sc", StandardScaler()), ("clf", clf)]), C_GRID,
                      scoring="roc_auc", cv=StratifiedKFold(3, shuffle=True, random_state=seed),
                      n_jobs=-1).fit(X[tr], y[tr])
    return gs.best_estimator_.predict_proba(X[te])[:, 1]


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = torch.load(args.graph, map_location="cpu", weights_only=False)
    data, woman = sanitize(raw, legacy=args.legacy_sanitize)
    y = data[woman].y.view(-1).numpy().astype(int)
    X = data[woman].x.numpy()
    n_edges = sum(data[e].edge_index.size(1) for e in data.edge_types)
    print(f"women={len(y)} positives={y.sum()} node_types={len(data.node_types)} "
          f"edge_types={len(data.edge_types)} edges={n_edges} device={device}", flush=True)

    feature_sets = {
        "LR (fusion)":     ("lr", X),
        "SVM (fusion)":    ("svm", X),
        "LR (PDI-13)":     ("lr", X[:, EMB:EMB + PDIN]),
        "LR (PDI-6)":      ("lr", X[:, [EMB + i for i in PDI6]]),
        "LR (narrative)":  ("lr", X[:, :EMB]),
    }
    rows = []
    os.makedirs(args.out, exist_ok=True)
    for seed in args.seeds:
        t0 = time.time()
        set_seed(seed)
        y_t = torch.as_tensor(y, dtype=torch.long, device=device)
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        oof = {k: np.zeros(len(y)) for k in
               ["PeriGNNosis", "FFNN (fusion)", "FFNN (PDI-13)", "FFNN (narrative)"] + list(feature_sets)}
        for fold, (tr, te) in enumerate(skf.split(X, y)):
            set_seed(seed + fold)
            safe = block_edges(data, te, woman).to(device)
            oof["PeriGNNosis"][te] = tune_train_nn(
                lambda kw: HeteroGAT(safe.metadata(), woman,
                                     in_dims={nt: safe[nt].x.size(1) for nt in safe.node_types}, **kw), safe, tr, te, y_t, device, woman, seed)
            oof["FFNN (fusion)"][te] = tune_train_nn(
                lambda kw: MLP(TOTAL, woman, **kw), safe, tr, te, y_t, device, woman, seed)
            g_pdi = safe.clone()
            g_pdi[woman].x = safe[woman].x[:, EMB:EMB + PDIN]
            oof["FFNN (PDI-13)"][te] = tune_train_nn(
                lambda kw: MLP(PDIN, woman, **kw), g_pdi, tr, te, y_t, device, woman, seed)
            g_txt = safe.clone()
            g_txt[woman].x = safe[woman].x[:, :EMB]
            oof["FFNN (narrative)"][te] = tune_train_nn(
                lambda kw: MLP(EMB, woman, **kw), g_txt, tr, te, y_t, device, woman, seed)
            for name, (kind, Xf) in feature_sets.items():
                oof[name][te] = linear_oof(kind, Xf, y, tr, te, seed)
            del safe, g_pdi, g_txt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"  seed {seed} fold {fold+1}/5 done ({time.time()-t0:.0f}s)", flush=True)
        for name, p in oof.items():
            rows.append({"seed": seed, "model": name, "auc": roc_auc_score(y, p)})
        pd.DataFrame({k: v for k, v in oof.items()} | {"y_true": y}).to_csv(
            os.path.join(args.out, f"oof_seed{seed}.csv"), index=False)
        print(pd.DataFrame(rows).query("seed == @seed").to_string(index=False), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "fair_benchmark_auc.csv"), index=False)
    summary = (df.groupby("model")["auc"].agg(["mean", "std", "min", "max"])
                 .sort_values("mean", ascending=False).round(4))
    print("\n=== pooled OOF AUC across seeds ===")
    print(summary.to_string())
    summary.to_csv(os.path.join(args.out, "fair_benchmark_summary.csv"))

    from scipy import stats
    base = df[df.model == "PeriGNNosis"].set_index("seed")["auc"]
    print("\n=== PeriGNNosis minus baseline (paired across seeds) ===")
    comp = []
    for m in df.model.unique():
        if m == "PeriGNNosis":
            continue
        other = df[df.model == m].set_index("seed")["auc"]
        d = (base - other).dropna()
        t = stats.ttest_rel(base[d.index], other[d.index])
        ci = stats.t.interval(0.95, len(d) - 1, d.mean(), stats.sem(d)) if len(d) > 1 else (np.nan, np.nan)
        comp.append({"model": m, "delta": round(d.mean(), 4),
                     "ci_low": round(ci[0], 4), "ci_high": round(ci[1], 4),
                     "p": round(float(t.pvalue), 4), "wins": int((d > 0).sum()), "n_seeds": len(d)})
    comp = pd.DataFrame(comp).sort_values("delta")
    print(comp.to_string(index=False))
    comp.to_csv(os.path.join(args.out, "fair_benchmark_paired.csv"), index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="perignnosis_graph.pt")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--out", default="results_fair_benchmark")
    ap.add_argument("--legacy-sanitize", action="store_true",
                    help="reproduce the original __Entity__ -> Entity collision")
    run(ap.parse_args())
