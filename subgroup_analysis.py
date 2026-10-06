"""
Performance of the graph model within subgroups.

Usage:
    python subgroup_analysis.py --graph perignnosis_graph_norm.pt --oof-dir results_fair/ --seeds 1 2 3 4 5 --out results_subgroup/
"""

import argparse, glob, os
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from fair_benchmark import (EMB, PDIN, COMP, TOTAL, C_GRID, HeteroGAT, MLP,
                            sanitize, set_seed, block_edges, tune_train_nn)


def part_a(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = torch.load(args.graph, map_location="cpu", weights_only=False)
    data, woman = sanitize(raw, legacy=args.legacy_sanitize)
    y = data[woman].y.view(-1).numpy().astype(int)
    X = data[woman].x.numpy()
    print(f"\n=== A. no-questionnaire setting (clinical block removed) ===", flush=True)
    rows = []
    for seed in args.seeds:
        set_seed(seed)
        y_t = torch.as_tensor(y, dtype=torch.long, device=device)
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        oof = {k: np.zeros(len(y)) for k in
               ["PeriGNNosis (no PDI)", "FFNN (narrative)", "LR (narrative)"]}
        for fold, (tr, te) in enumerate(skf.split(X, y)):
            set_seed(seed + fold)
            safe = block_edges(data, te, woman)
            safe[woman].x = safe[woman].x.clone()
            safe[woman].x[:, EMB:] = 0.0
            safe = safe.to(device)
            oof["PeriGNNosis (no PDI)"][te] = tune_train_nn(
                lambda kw: HeteroGAT(safe.metadata(), woman,
                                     in_dims={nt: safe[nt].x.size(1) for nt in safe.node_types}, **kw), safe, tr, te, y_t, device, woman, seed)
            g_txt = safe.clone()
            g_txt[woman].x = safe[woman].x[:, :EMB]
            oof["FFNN (narrative)"][te] = tune_train_nn(
                lambda kw: MLP(EMB, woman, **kw), g_txt, tr, te, y_t, device, woman, seed)
            gs = GridSearchCV(Pipeline([("sc", StandardScaler()),
                                        ("clf", LogisticRegression(class_weight="balanced",
                                                                   solver="liblinear", max_iter=2000))]),
                              C_GRID, scoring="roc_auc",
                              cv=StratifiedKFold(3, shuffle=True, random_state=seed), n_jobs=-1
                              ).fit(X[tr][:, :EMB], y[tr])
            oof["LR (narrative)"][te] = gs.best_estimator_.predict_proba(X[te][:, :EMB])[:, 1]
            del safe, g_txt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"  seed {seed} fold {fold+1}/5", flush=True)
        for k, p in oof.items():
            rows.append({"seed": seed, "model": k, "auc": roc_auc_score(y, p)})
        pd.DataFrame(oof | {"y_true": y}).to_csv(
            os.path.join(args.out, f"noPDI_oof_seed{seed}.csv"), index=False)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "noPDI_auc.csv"), index=False)
    summ = df.groupby("model")["auc"].agg(["mean", "std"]).sort_values("mean", ascending=False).round(4)
    print(summ.to_string())
    if "PeriGNNosis (no PDI)" in summ.index and len(args.seeds) > 1:
        from scipy import stats
        g = df[df.model == "PeriGNNosis (no PDI)"].set_index("seed")["auc"]
        for m in ["FFNN (narrative)", "LR (narrative)"]:
            o = df[df.model == m].set_index("seed")["auc"]
            d = (g - o).dropna()
            t = stats.ttest_rel(g[d.index], o[d.index])
            print(f"  PeriGNNosis(no PDI) - {m}: {d.mean():+.4f}  p={t.pvalue:.4f}  "
                  f"wins {int((d>0).sum())}/{len(d)}")
    return summ


def part_b(args):
    files = sorted(glob.glob(os.path.join(args.oof_dir, "oof_seed*.csv")))
    if not files:
        print(f"\n(no OOF files in {args.oof_dir}; skipping part B)")
        return
    print(f"\n=== B. does the graph help where the questionnaire is ambiguous? ===", flush=True)
    rows = []
    for f in files:
        d = pd.read_csv(f)
        y = d["y_true"].values
        pdi = d["LR (PDI-13)"].values
        gnn = d["PeriGNNosis"].values
        lo, hi = np.percentile(pdi, [33, 67])
        bands = {"low risk (PDI)": pdi < lo, "ambiguous (PDI)": (pdi >= lo) & (pdi <= hi),
                 "high risk (PDI)": pdi > hi}
        for name, m in bands.items():
            if y[m].min() == y[m].max():
                continue
            rows.append({"file": os.path.basename(f), "band": name, "n": int(m.sum()),
                         "positives": int(y[m].sum()),
                         "auc_PDI": roc_auc_score(y[m], pdi[m]),
                         "auc_PeriGNNosis": roc_auc_score(y[m], gnn[m])})
    b = pd.DataFrame(rows)
    if b.empty:
        print("  no usable bands")
        return
    summ = (b.groupby("band")[["n", "positives", "auc_PDI", "auc_PeriGNNosis"]]
              .mean().round(3))
    summ["delta"] = (summ["auc_PeriGNNosis"] - summ["auc_PDI"]).round(3)
    print(summ.to_string())
    b.to_csv(os.path.join(args.out, "subgroup_bands.csv"), index=False)
    return summ


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="perignnosis_graph_norm.pt")
    ap.add_argument("--oof-dir", default="results_fair/")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--out", default="results_subgroup")
    ap.add_argument("--legacy-sanitize", action="store_true")
    ap.add_argument("--skip-a", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if not a.skip_a:
        part_a(a)
    part_b(a)
    print("\nsaved to", a.out)
