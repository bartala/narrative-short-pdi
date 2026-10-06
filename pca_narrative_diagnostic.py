"""
Logistic regression on the PCA-compressed narrative with and without PDI items.

Usage:
    python pca_narrative_diagnostic.py --graph perignnosis_graph_pooled.pt --seeds 1 2 3 4 5 --out results_pca_diagnostic/
"""

import argparse, os, time
import numpy as np, pandas as pd, torch
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fair_benchmark import EMB, PDIN, sanitize, set_seed

C_VALUES = [0.01, 0.1, 1, 10, 100]
K_GRID = [5, 10, 20, 50, 100]
EMB_COLS = list(range(EMB))
PDI_COLS = list(range(EMB, EMB + PDIN))


def lr():
    return LogisticRegression(class_weight="balanced", solver="liblinear",
                              max_iter=2000)


def pipe_pca(with_pdi, k=None):
    parts = [("pca", PCA(n_components=k or 10, random_state=0), EMB_COLS)]
    if with_pdi:
        parts.append(("pdi", "passthrough", PDI_COLS))
    return Pipeline([("ct", ColumnTransformer(parts)),
                     ("sc", StandardScaler()), ("clf", lr())])


def pipe_cols(cols):
    return Pipeline([("sel", ColumnTransformer([("keep", "passthrough", cols)])),
                     ("sc", StandardScaler()), ("clf", lr())])


def fit_predict(pipe, grid, X, y, tr, te, seed):
    gs = GridSearchCV(pipe, grid, scoring="roc_auc",
                      cv=StratifiedKFold(3, shuffle=True, random_state=seed),
                      n_jobs=-1).fit(X[tr], y[tr])
    return gs.best_estimator_.predict_proba(X[te])[:, 1], gs.best_params_


def perm_test(y, p1, p2, n=10000, seed=0):
    rng = np.random.RandomState(seed)
    obs = roc_auc_score(y, p1) - roc_auc_score(y, p2)
    hits = 0
    for _ in range(n):
        s = rng.rand(len(y)) < 0.5
        a, b = np.where(s, p2, p1), np.where(s, p1, p2)
        if abs(roc_auc_score(y, a) - roc_auc_score(y, b)) >= abs(obs) - 1e-12:
            hits += 1
    return obs, (hits + 1) / (n + 1)


def main(a):
    os.makedirs(a.out, exist_ok=True)
    raw = torch.load(a.graph, map_location="cpu", weights_only=False)
    data, woman = sanitize(raw)
    X = data[woman].x.numpy()[:, :EMB + PDIN].astype(np.float64)
    y = data[woman].y.view(-1).numpy().astype(int)
    print(f"{len(y)} women, {int(y.sum())} positive; narrative dim {EMB}, "
          f"PDI items {PDIN}", flush=True)

    M_PDI = "LR (PDI-13)"
    M_PCA_PDI = "LR (narrative PCA + PDI)"
    M_FULL = "LR (narrative 768 + PDI)"
    M_PCA = "LR (narrative PCA only)"
    grid_c = {"clf__C": C_VALUES}
    grid_pca = {"clf__C": C_VALUES, "ct__pca__n_components": K_GRID}

    rows, chosen, curve = [], [], []
    for seed in a.seeds:
        set_seed(seed)
        t0 = time.time()
        oof = {m: np.zeros(len(y)) for m in [M_PDI, M_PCA_PDI, M_FULL, M_PCA]}
        oof_k = {k: np.zeros(len(y)) for k in K_GRID}
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(skf.split(X, y)):
            oof[M_PDI][te], _ = fit_predict(pipe_cols(PDI_COLS), grid_c, X, y, tr, te, seed)
            oof[M_PCA_PDI][te], bp = fit_predict(pipe_pca(True), grid_pca, X, y,
                                                 tr, te, seed)
            chosen.append({"seed": seed, "fold": fold,
                           "k": bp["ct__pca__n_components"], "C": bp["clf__C"]})
            oof[M_FULL][te], _ = fit_predict(pipe_cols(EMB_COLS + PDI_COLS), grid_c,
                                             X, y, tr, te, seed)
            oof[M_PCA][te], _ = fit_predict(pipe_pca(False), grid_pca, X, y,
                                            tr, te, seed)
            for k in K_GRID:
                oof_k[k][te], _ = fit_predict(pipe_pca(True, k), grid_c, X, y,
                                              tr, te, seed)
        for m, p in oof.items():
            rows.append({"seed": seed, "model": m, "auc": roc_auc_score(y, p)})
        for k, p in oof_k.items():
            curve.append({"seed": seed, "k": k, "auc": roc_auc_score(y, p)})
        pd.DataFrame(oof | {"y_true": y}).to_csv(
            os.path.join(a.out, f"oof_seed{seed}.csv"), index=False)
        print(f"  seed {seed} ({time.time()-t0:.0f}s): " + " | ".join(
            f"{m} {roc_auc_score(y, p):.4f}" for m, p in oof.items()), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "pca_auc.csv"), index=False)
    pd.DataFrame(chosen).to_csv(os.path.join(a.out, "pca_chosen.csv"), index=False)
    cv = pd.DataFrame(curve)
    cv.to_csv(os.path.join(a.out, "pca_curve.csv"), index=False)

    print("\n=== pooled OOF AUC ===")
    print(df.groupby("model")["auc"].agg(["mean", "std"]).round(4)
            .sort_values("mean", ascending=False).to_string())
    ch = pd.DataFrame(chosen)
    print(f"\ncomponents chosen by the inner CV: "
          f"{dict(ch.k.value_counts().sort_index())}")
    print("\n=== secondary: fixed number of components, LR (narrative PCA_k + PDI) ===")
    ref = df[df.model == M_PDI].auc.mean()
    for k, g in cv.groupby("k"):
        print(f"  k={k:<4} AUC {g.auc.mean():.4f}   vs PDI-13 {g.auc.mean() - ref:+.4f}")

    print("\n=== PRIMARY: paired permutation vs LR (PDI-13), per seed ===")
    for m in [M_PCA_PDI, M_FULL]:
        ds, ps = [], []
        for seed in a.seeds:
            o = pd.read_csv(os.path.join(a.out, f"oof_seed{seed}.csv"))
            d, pv = perm_test(o["y_true"].values, o[m].values, o[M_PDI].values,
                              n=10000, seed=seed)
            ds.append(d); ps.append(pv)
        print(f"  {m} - {M_PDI}: Δ {np.mean(ds):+.4f}  perm p {min(ps):.4f}-{max(ps):.4f}"
              f"  wins {sum(d > 0 for d in ds)}/{len(ds)}")

    print("\nHow to read this:")
    print("  If 'narrative PCA + PDI' clearly beats PDI-13, the narrative carries")
    print("  information beyond the questionnaire and a PCA-compressed PeriGNNosis")
    print("  is worth building. If it does not, no model built on these narrative")
    print("  embeddings - graph or otherwise - can be expected to beat the PDI.")
    print(f"\nsaved to {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="perignnosis_graph_pooled.pt")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--out", default="results_pca_diagnostic")
    main(ap.parse_args())
