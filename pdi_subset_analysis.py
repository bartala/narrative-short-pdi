"""
Nested selection of PDI items by univariate AUC.

Usage:
    python pdi_subset_analysis.py --graph perignnosis_graph_rebuilt.pt --seeds 1 2 3 4 5 --out results_pdi_subset/
"""

import argparse, collections, os
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from fair_benchmark import EMB, PDIN, C_GRID, sanitize, set_seed

ORIGINAL_SIX = [0, 1, 2, 4, 11, 12]


def perm_test(y, p1, p2, n=2000, seed=0):
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


def lr():
    return Pipeline([("sc", StandardScaler()),
                     ("clf", LogisticRegression(class_weight="balanced",
                                                solver="liblinear", max_iter=2000))])


def fit_lr(X, y, cols, seed):
    gs = GridSearchCV(lr(), C_GRID, scoring="roc_auc",
                      cv=StratifiedKFold(3, shuffle=True, random_state=seed),
                      n_jobs=-1).fit(X[:, cols], y)
    return gs.best_estimator_


def rank_items(P, y):
    scores = []
    for j in range(P.shape[1]):
        v = P[:, j]
        if len(np.unique(v)) < 2:
            scores.append(0.5)
            continue
        a = roc_auc_score(y, v)
        scores.append(max(a, 1.0 - a))
    return list(np.argsort(scores)[::-1]), scores


def choose_k(P, y, seed, kmax):
    inner = StratifiedKFold(3, shuffle=True, random_state=seed)
    per_k = np.zeros(kmax + 1)
    for itr, ival in inner.split(P, y):
        order, _ = rank_items(P[itr], y[itr])
        for k in range(1, kmax + 1):
            cols = order[:k]
            m = fit_lr(P[itr], y[itr], cols, seed)
            per_k[k] += roc_auc_score(y[ival], m.predict_proba(P[ival][:, cols])[:, 1])
    per_k /= inner.get_n_splits()
    best = max(range(1, kmax + 1), key=lambda k: (per_k[k], -k))
    return best, per_k


def main(a):
    os.makedirs(a.out, exist_ok=True)
    import torch
    raw = torch.load(a.graph, map_location="cpu", weights_only=False)
    data, woman = sanitize(raw)
    y = data[woman].y.view(-1).numpy().astype(int)
    X = data[woman].x.numpy()
    P = X[:, EMB:EMB + PDIN]
    print(f"{len(y)} women, {int(y.sum())} positive, {PDIN} PDI items", flush=True)

    rows, picks, kcurves, curve_rows = [], [], [], []
    for seed in a.seeds:
        set_seed(seed)
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        oof = {k: np.zeros(len(y)) for k in
               ["LR (nested-selected PDI)", "LR (PDI-13)", "LR (original six)"]}
        oof_k = {k: np.zeros(len(y)) for k in range(1, PDIN + 1)}
        for fold, (tr, te) in enumerate(skf.split(P, y)):
            set_seed(seed + fold)
            k_star, curve = choose_k(P[tr], y[tr], seed, PDIN)
            kcurves.append({"seed": seed, "fold": fold,
                            **{f"k{k}": curve[k] for k in range(1, PDIN + 1)}})
            order, _ = rank_items(P[tr], y[tr])
            sel = sorted(order[:k_star])
            picks.append({"seed": seed, "fold": fold, "k": k_star,
                          "items": "|".join(str(i + 1) for i in sel)})

            m = fit_lr(P[tr], y[tr], sel, seed)
            oof["LR (nested-selected PDI)"][te] = m.predict_proba(P[te][:, sel])[:, 1]

            m = fit_lr(P[tr], y[tr], list(range(PDIN)), seed)
            oof["LR (PDI-13)"][te] = m.predict_proba(P[te])[:, 1]

            m = fit_lr(P[tr], y[tr], ORIGINAL_SIX, seed)
            oof["LR (original six)"][te] = m.predict_proba(P[te][:, ORIGINAL_SIX])[:, 1]

            for k in range(1, PDIN + 1):
                cols = sorted(order[:k])
                m = fit_lr(P[tr], y[tr], cols, seed)
                oof_k[k][te] = m.predict_proba(P[te][:, cols])[:, 1]

            print(f"  seed {seed} fold {fold+1}/5: k*={k_star} items "
                  f"{[int(i)+1 for i in sel]}", flush=True)

        for name, p in oof.items():
            rows.append({"seed": seed, "model": name, "auc": roc_auc_score(y, p)})
        for k, p in oof_k.items():
            d, pv = perm_test(y, p, oof_k[PDIN], n=a.perms, seed=seed)
            curve_rows.append({"seed": seed, "k": k, "auc": roc_auc_score(y, p),
                               "delta_vs_13": d, "perm_p": pv})
        pd.DataFrame(oof | {"y_true": y}).to_csv(
            os.path.join(a.out, f"pdi_oof_seed{seed}.csv"), index=False)
        pd.DataFrame(oof_k | {"y_true": y}).to_csv(
            os.path.join(a.out, f"pdi_oof_by_k_seed{seed}.csv"), index=False)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "pdi_subset_auc.csv"), index=False)
    pd.DataFrame(picks).to_csv(os.path.join(a.out, "pdi_selections.csv"), index=False)
    pd.DataFrame(kcurves).to_csv(os.path.join(a.out, "pdi_k_curves.csv"), index=False)

    print("\n=== pooled OOF AUC ===")
    print(df.groupby("model")["auc"].agg(["mean", "std"]).round(4).to_string())

    n_folds = len(picks)
    freq = collections.Counter()
    for r in picks:
        for it in r["items"].split("|"):
            freq[int(it)] += 1
    print(f"\n=== item selection frequency over {n_folds} outer folds ===")
    for it in range(1, PDIN + 1):
        c = freq.get(it, 0)
        flag = "  <- robust (>=80%)" if c >= 0.8 * n_folds else ""
        print(f"  Q{it:<2} {c}/{n_folds}  ({c/n_folds:.0%}){flag}")
    robust = sorted(it for it in freq if freq[it] >= 0.8 * n_folds)
    print(f"\nrobust set: {robust}  (size {len(robust)})")
    print(f"manuscript's original six: {[i+1 for i in ORIGINAL_SIX]}")
    print(f"overlap: {sorted(set(robust) & {i+1 for i in ORIGINAL_SIX})}")
    ks = collections.Counter(r["k"] for r in picks)
    print(f"chosen k across folds: {dict(sorted(ks.items()))}")

    cv = pd.DataFrame(curve_rows)
    cv.to_csv(os.path.join(a.out, "pdi_auc_by_k.csv"), index=False)
    g = cv.groupby("k")
    print(f"\n=== out-of-fold AUC by number of items "
          f"(items ranked inside each fold; paired permutation test vs all 13) ===")
    smallest = None
    for k in range(1, PDIN + 1):
        r = g.get_group(k)
        auc, d = r.auc.mean(), r.delta_vs_13.mean()
        pmin, pmax = r.perm_p.min(), r.perm_p.max()
        ns = pmin > 0.05
        if k < PDIN and ns and smallest is None:
            smallest = k
        print(f"  k={k:<3} AUC {auc:.4f}  Δ {d:+.4f}  "
              f"perm p {pmin:.3f}-{pmax:.3f}"
              + ("   not significantly worse in any seed" if ns else ""))
    if smallest:
        print(f"\nsmallest k not significantly worse than 13 items in any seed: "
              f"{smallest}")
    else:
        print("\nno reduced set was non-significant across all seeds")

    print("\n=== vs all 13 items, paired permutation test per seed ===")
    for name in ["LR (nested-selected PDI)", "LR (original six)"]:
        ds, ps = [], []
        for seed in a.seeds:
            o = pd.read_csv(os.path.join(a.out, f"pdi_oof_seed{seed}.csv"))
            d, pv = perm_test(o["y_true"].values, o[name].values,
                              o["LR (PDI-13)"].values, n=10000, seed=seed)
            ds.append(d); ps.append(pv)
        print(f"  {name}: Δ {np.mean(ds):+.4f}  "
              f"perm p {min(ps):.4f}-{max(ps):.4f}")

    print(f"\nsaved to {a.out}")
    print("Use the robust set above as --pdi-subset for compact_gnn.py "
          "(zero-based indices, i.e. Q1 -> 0).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="perignnosis_graph_rebuilt.pt")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--out", default="results_pdi_subset")
    ap.add_argument("--perms", type=int, default=2000,
                    help="permutations for the per-k curve (headline comparisons "
                         "always use 10,000)")
    main(ap.parse_args())
