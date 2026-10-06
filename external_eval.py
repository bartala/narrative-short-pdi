"""
Evaluate the graph model and baselines on a second cohort.

Usage:
    python external_eval.py --dev-graph perignnosis_graph.pt --ext-graph external_graph.pt --seeds 1 2 3 4 5 --out results_external/
"""

import argparse, os, random
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

from fair_benchmark import (EMB, PDIN, COMP, TOTAL, PDI6, NN_GRID, TUNE_EPOCHS,
                            FINAL_EPOCHS, WD, C_GRID, HeteroGAT, MLP, sanitize, set_seed)


def align_to(dev_data, ext_raw, woman):
    from torch_geometric.data import HeteroData
    import torch_geometric.transforms as T
    tmp = HeteroData()
    keep_nodes = set(dev_data.node_types)
    for nt in ext_raw.node_types:
        new = "EntityAll" if nt == "__Entity__" else nt.replace(" ", "_").replace("-", "_")
        if new not in keep_nodes:
            new = "EntityAll"
        x = ext_raw[nt].x
        if new in tmp.node_types and "x" in tmp[new]:
            tmp[new].x = torch.cat([tmp[new].x, x], 0)
        else:
            tmp[new].x = x
        if "y" in ext_raw[nt]:
            tmp[new].y = ext_raw[nt].y
    for (s, r, d) in ext_raw.edge_types:
        ns = "EntityAll" if s == "__Entity__" else s.replace(" ", "_").replace("-", "_")
        nd = "EntityAll" if d == "__Entity__" else d.replace(" ", "_").replace("-", "_")
        ns = ns if ns in keep_nodes else "EntityAll"
        nd = nd if nd in keep_nodes else "EntityAll"
        key = (ns, r, nd)
        if key not in dev_data.edge_types:
            key = (ns, "MENTIONS", nd)
            if key not in dev_data.edge_types:
                continue
        ei = ext_raw[s, r, d].edge_index
        if key in tmp.edge_types and "edge_index" in tmp[key]:
            tmp[key].edge_index = torch.cat([tmp[key].edge_index, ei], 1)
        else:
            tmp[key].edge_index = ei
    for nt in dev_data.node_types:
        if nt not in tmp.node_types or "x" not in tmp[nt]:
            tmp[nt].x = torch.zeros(1, TOTAL if nt == woman else EMB)
    data = T.ToUndirected()(tmp)
    for (s, r, d) in data.edge_types:
        ei = data[s, r, d].edge_index
        if ei is not None and ei.numel():
            m = (ei[0] < data[s].num_nodes) & (ei[1] < data[d].num_nodes)
            data[s, r, d].edge_index = ei[:, m]
    return data


def train_full_nn(make_model, g_dev, y_dev_t, device, seed, woman):
    import itertools
    idx = np.arange(len(y_dev_t))
    tr, val = train_test_split(idx, test_size=0.2, stratify=y_dev_t.cpu().numpy(), random_state=seed)
    best, best_cfg = -1, None
    keys, vals = zip(*NN_GRID.items())
    for combo in itertools.product(*vals):
        cfg = dict(zip(keys, combo))
        m = make_model({k: v for k, v in cfg.items() if k != "lr"}).to(device)
        opt = torch.optim.AdamW(m.parameters(), lr=cfg["lr"], weight_decay=WD)
        for _ in range(TUNE_EPOCHS):
            m.train(); opt.zero_grad()
            F.cross_entropy(m(g_dev.x_dict, g_dev.edge_index_dict)[tr], y_dev_t[tr]).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            p = F.softmax(m(g_dev.x_dict, g_dev.edge_index_dict)[val], 1)[:, 1].cpu().numpy()
        a = roc_auc_score(y_dev_t[val].cpu().numpy(), p)
        if a > best:
            best, best_cfg = a, cfg
        del m, opt
    m = make_model({k: v for k, v in best_cfg.items() if k != "lr"}).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=best_cfg["lr"], weight_decay=WD)
    for _ in range(FINAL_EPOCHS):
        m.train(); opt.zero_grad()
        F.cross_entropy(m(g_dev.x_dict, g_dev.edge_index_dict), y_dev_t).backward(); opt.step()
    m.eval()
    return m, best_cfg


def boot_ci(y, p, q=None, n=2000, seed=42):
    rng = np.random.RandomState(seed); out, diff = [], []
    while len(out) < n:
        i = rng.choice(len(y), len(y), replace=True)
        if y[i].min() == y[i].max():
            continue
        out.append(roc_auc_score(y[i], p[i]))
        if q is not None:
            diff.append(roc_auc_score(y[i], p[i]) - roc_auc_score(y[i], q[i]))
    lo, hi = np.percentile(out, [2.5, 97.5])
    if q is None:
        return lo, hi, None, None
    dl, dh = np.percentile(diff, [2.5, 97.5])
    return lo, hi, dl, dh


def main(a):
    os.makedirs(a.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev_raw = torch.load(a.dev_graph, map_location="cpu", weights_only=False)
    dev, woman = sanitize(dev_raw, legacy=a.legacy_sanitize)
    y_dev = dev[woman].y.view(-1).numpy().astype(int)
    X_dev = dev[woman].x.numpy()

    ext_raw = torch.load(a.ext_graph, map_location="cpu", weights_only=False)
    ext = align_to(dev, ext_raw, woman)
    y_ext = ext[woman].y.view(-1).numpy().astype(int)
    X_ext = ext[woman].x.numpy()
    print(f"development: {len(y_dev)} women ({y_dev.sum()} positive) | "
          f"external: {len(y_ext)} women ({y_ext.sum()} positive, {y_ext.mean():.1%})", flush=True)

    preds, rows = {}, []
    for name, cols in {"LR (fusion)": list(range(TOTAL)),
                       "LR (PDI-13)": list(range(EMB, EMB + PDIN)),
                       "LR (PDI-6)": [EMB + i for i in PDI6],
                       "LR (narrative)": list(range(EMB))}.items():
        gs = GridSearchCV(Pipeline([("sc", StandardScaler()),
                                    ("clf", LogisticRegression(class_weight="balanced",
                                                               solver="liblinear", max_iter=2000))]),
                          C_GRID, scoring="roc_auc",
                          cv=StratifiedKFold(5, shuffle=True, random_state=42), n_jobs=-1
                          ).fit(X_dev[:, cols], y_dev)
        preds[name] = gs.best_estimator_.predict_proba(X_ext[:, cols])[:, 1]
        rows.append({"model": name, "seed": "-", "auc": roc_auc_score(y_ext, preds[name])})
        print(f"{name:<16} external AUC {rows[-1]['auc']:.3f}", flush=True)

    y_dev_t = torch.as_tensor(y_dev, dtype=torch.long, device=device)
    g_dev = dev.to(device)
    p_gnn, p_ffnn = [], []
    for seed in a.seeds:
        set_seed(seed)
        m, cfg = train_full_nn(lambda kw: HeteroGAT(g_dev.metadata(), woman,
                                  in_dims={nt: g_dev[nt].x.size(1) for nt in g_dev.node_types}, **kw),
                               g_dev, y_dev_t, device, seed, woman)
        with torch.no_grad():
            g_ext = ext.to(device)
            p = F.softmax(m(g_ext.x_dict, g_ext.edge_index_dict), 1)[:, 1].cpu().numpy()[:len(y_ext)]
        p_gnn.append(p)
        rows.append({"model": "PeriGNNosis", "seed": seed, "auc": roc_auc_score(y_ext, p)})
        print(f"PeriGNNosis seed {seed}: external AUC {rows[-1]['auc']:.3f}  (cfg {cfg})", flush=True)
        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        set_seed(seed)
        m2, _ = train_full_nn(lambda kw: MLP(TOTAL, woman, **kw), g_dev, y_dev_t, device, seed, woman)
        with torch.no_grad():
            p2 = F.softmax(m2(ext.to(device).x_dict, None), 1)[:, 1].cpu().numpy()[:len(y_ext)]
        p_ffnn.append(p2)
        rows.append({"model": "FFNN (fusion)", "seed": seed, "auc": roc_auc_score(y_ext, p2)})
        del m2
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    preds["PeriGNNosis"] = np.mean(p_gnn, 0)
    preds["FFNN (fusion)"] = np.mean(p_ffnn, 0)

    print("\n=== EXTERNAL AUC (seed-averaged predictions for neural models) ===")
    summary = []
    for name, p in preds.items():
        lo, hi, dl, dh = boot_ci(y_ext, p, preds["PeriGNNosis"] if name != "PeriGNNosis" else None)
        auc = roc_auc_score(y_ext, p)
        delta = auc - roc_auc_score(y_ext, preds["PeriGNNosis"]) if name != "PeriGNNosis" else 0.0
        summary.append({"model": name, "auc": round(auc, 3), "ci_low": round(lo, 3), "ci_high": round(hi, 3),
                        "minus_PeriGNNosis": round(delta, 3),
                        "delta_ci_low": None if dl is None else round(-dh, 3),
                        "delta_ci_high": None if dh is None else round(-dl, 3)})
    summary = pd.DataFrame(summary).sort_values("auc", ascending=False)
    print(summary.to_string(index=False))
    summary.to_csv(os.path.join(a.out, "external_summary.csv"), index=False)

    p = preds["PeriGNNosis"]
    print("\n=== PeriGNNosis operating points on the external cohort ===")
    for thr in (0.225, 0.5):
        yh = (p >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_ext, yh, labels=[0, 1]).ravel()
        print(f"thr={thr}: F1={f1_score(y_ext, yh):.3f} sens={tp/(tp+fn):.3f} "
              f"spec={tn/(tn+fp):.3f} PPV={tp/max(tp+fp,1):.3f}")

    out = pd.DataFrame({"y_true": y_ext})
    for k, v in preds.items():
        out[k.replace(" ", "_")] = v
    out.to_csv(os.path.join(a.out, "external_predictions.csv"), index=False)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "external_per_seed.csv"), index=False)
    print("\nsaved to", a.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-graph", default="perignnosis_graph.pt")
    ap.add_argument("--ext-graph", default="external_graph.pt")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--out", default="results_external")
    ap.add_argument("--legacy-sanitize", action="store_true")
    main(ap.parse_args())
