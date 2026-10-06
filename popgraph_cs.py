"""
Population graph with Correct & Smooth on top of logistic regression; gradient boosting baseline.

Usage:
    python popgraph_cs.py --seeds 1 2 3 4 5 --out results_popgraph_cs/
"""

import argparse, itertools, os, time
import numpy as np, pandas as pd, torch
from scipy.special import logit, expit
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import clone

from fair_benchmark import EMB, PDIN, sanitize, set_seed
from pca_narrative_diagnostic import C_VALUES, K_GRID, EMB_COLS, PDI_COLS, perm_test
from graph_signal_diagnostic import pipe

K_NB = [5, 10, 25, 50]
S_GRID = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
SPACES = ["pdi", "narrative", "both"]
M_LR, M_CS, M_LP, M_GBM = ("LR (narrative + PDI-13)", "PeriGNNosis-CS",
                           "LR + label propagation", "GBM (narrative + PDI-13)")
MODELS = [M_LR, M_CS, M_LP, M_GBM]


def lr_tuned(X, y, tr, seed):
    gs = GridSearchCV(pipe([("pca", PCA(10, random_state=0), EMB_COLS),
                            ("pdi", "passthrough", PDI_COLS)]),
                      {"clf__C": C_VALUES, "ct__pca__n_components": K_GRID},
                      scoring="roc_auc", n_jobs=-1,
                      cv=StratifiedKFold(3, shuffle=True, random_state=seed))
    return gs.fit(X[tr], y[tr]).best_estimator_


def space(X, tr, kind, n_comp=20):
    parts = []
    if kind in ("narrative", "both"):
        Z = PCA(n_comp, random_state=0).fit(X[tr][:, EMB_COLS]).transform(X[:, EMB_COLS])
        parts.append(Z)
    if kind in ("pdi", "both"):
        parts.append(X[:, PDI_COLS])
    Z = np.concatenate(parts, 1)
    mu, sd = Z[tr].mean(0), Z[tr].std(0).clip(1e-9)
    Z = (Z - mu) / sd
    if kind == "both":
        nz = Z.shape[1] - PDIN
        Z[:, :nz] /= np.sqrt(nz); Z[:, nz:] /= np.sqrt(PDIN)
    return Z / np.linalg.norm(Z, axis=1, keepdims=True).clip(1e-9)


def neighbours(Z, src, dst, kmax):
    S = Z[dst] @ Z[src].T
    S[np.asarray(dst)[:, None] == np.asarray(src)[None, :]] = -np.inf
    return np.asarray(src)[np.argsort(-S, 1)[:, :kmax]]


def main(a):
    os.makedirs(a.out, exist_ok=True)
    d, w = sanitize(torch.load(a.graph, map_location="cpu", weights_only=False))
    X = d[w].x.numpy()[:, :EMB + PDIN].astype(np.float64)
    y = d[w].y.view(-1).numpy().astype(int)
    coh = pd.read_csv(a.cohort_csv).cohort.values.astype(int)
    strata = y * 2 + coh
    kmax = max(K_NB)
    print(f"{len(y)} women, {y.sum()} positive; spaces {SPACES}, k {K_NB}, s {S_GRID}",
          flush=True)

    gbm = Pipeline([("ct", ColumnTransformer([("pca", PCA(20, random_state=0), EMB_COLS),
                                              ("pdi", "passthrough", PDI_COLS)])),
                    ("clf", HistGradientBoostingClassifier(random_state=0,
                                                           class_weight="balanced"))])
    g_gbm = {"clf__learning_rate": [0.03, 0.1], "clf__max_depth": [2, 3],
             "clf__max_iter": [100, 300], "clf__l2_regularization": [0.0, 1.0]}

    rows, picks = [], []
    for seed in a.seeds:
        set_seed(seed)
        t0 = time.time()
        oof = {m: np.zeros(len(y)) for m in MODELS}
        for fold, (tr, te) in enumerate(
                StratifiedKFold(5, shuffle=True, random_state=seed).split(X, strata)):
            best = lr_tuned(X, y, tr, seed)
            p_te = best.predict_proba(X[te])[:, 1]
            p_tr = np.zeros(len(y))
            for a_, b_ in StratifiedKFold(5, shuffle=True, random_state=seed).split(
                    tr, strata[tr]):
                m = clone(best).fit(X[tr[a_]], y[tr[a_]])
                p_tr[tr[b_]] = m.predict_proba(X[tr[b_]])[:, 1]
            p_all = p_tr.copy(); p_all[te] = p_te
            eps = 1e-6
            L = logit(np.clip(p_all, eps, 1 - eps))
            resid = y - p_all

            best_cs, best_lp = (-1, None), (-1, None)
            NB = {}
            for sp in SPACES:
                Z = space(X, tr, sp)
                NB[sp] = (neighbours(Z, tr, tr, kmax), neighbours(Z, tr, te, kmax))
                for k in K_NB:
                    nb_tr = NB[sp][0][:, :k]
                    corr_r = resid[nb_tr].mean(1)
                    corr_y = y[nb_tr].mean(1) - y[tr].mean()
                    for s in S_GRID:
                        auc_r = roc_auc_score(y[tr], L[tr] + s * corr_r)
                        auc_y = roc_auc_score(y[tr], L[tr] + s * corr_y)
                        if auc_r > best_cs[0]: best_cs = (auc_r, (sp, k, s))
                        if auc_y > best_lp[0]: best_lp = (auc_y, (sp, k, s))
            sp, k, s = best_cs[1]
            oof[M_CS][te] = expit(L[te] + s * resid[NB[sp][1][:, :k]].mean(1))
            sp2, k2, s2 = best_lp[1]
            oof[M_LP][te] = expit(L[te] + s2 * (y[NB[sp2][1][:, :k2]].mean(1) - y[tr].mean()))
            oof[M_LR][te] = p_te
            gg = GridSearchCV(gbm, g_gbm, scoring="roc_auc", n_jobs=-1,
                              cv=StratifiedKFold(3, shuffle=True, random_state=seed))
            oof[M_GBM][te] = gg.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
            picks.append({"seed": seed, "fold": fold, "cs_space": sp, "cs_k": k, "cs_s": s,
                          "lp_space": sp2, "lp_k": k2, "lp_s": s2})
        for m in MODELS:
            rows.append({"seed": seed, "model": m, "auc": roc_auc_score(y, oof[m])})
        pd.DataFrame(oof | {"y_true": y}).to_csv(
            os.path.join(a.out, f"oof_seed{seed}.csv"), index=False)
        print(f"  seed {seed} ({time.time()-t0:.0f}s): " + " | ".join(
            f"{m} {roc_auc_score(y, oof[m]):.4f}" for m in MODELS), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "auc.csv"), index=False)
    pk = pd.DataFrame(picks); pk.to_csv(os.path.join(a.out, "choices.csv"), index=False)
    print("\n=== pooled OOF AUC ===")
    print(df.groupby("model").auc.agg(["mean", "std"]).round(4)
            .sort_values("mean", ascending=False).to_string())
    print("\nCorrect & Smooth choices: space " + str(dict(pk.cs_space.value_counts())) +
          ", k " + str(dict(pk.cs_k.value_counts())) + ", s " + str(dict(pk.cs_s.value_counts())))
    print(f"\n=== paired permutation vs {M_LR} ({a.perms} draws) ===")
    for m in [M_CS, M_LP, M_GBM]:
        ds, ps = [], []
        for seed in a.seeds:
            o = pd.read_csv(os.path.join(a.out, f"oof_seed{seed}.csv"))
            dd, pv = perm_test(o.y_true.values, o[m].values, o[M_LR].values, n=a.perms, seed=seed)
            ds.append(dd); ps.append(pv)
        print(f"  {m:<26} Δ {np.mean(ds):+.4f}  perm p {min(ps):.3f}-{max(ps):.3f}"
              f"  wins {sum(x > 0 for x in ds)}/{len(ds)}")
    print(f"\nsaved to {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="perignnosis_graph_pooled.pt")
    ap.add_argument("--cohort-csv", default="pooled_cohort.csv")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--perms", type=int, default=2000)
    ap.add_argument("--out", default="results_popgraph_cs")
    main(ap.parse_args())
