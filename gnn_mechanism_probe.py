"""
Probes for neighbourhood information that message passing could use.

Usage:
    python gnn_mechanism_probe.py
"""

import argparse, os
import numpy as np, pandas as pd, torch
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from fair_benchmark import EMB, PDIN, sanitize, set_seed
from pca_narrative_diagnostic import fit_predict, perm_test, C_VALUES, EMB_COLS, PDI_COLS
from graph_signal_diagnostic import incidence, pipe


def knn_mean(Z, r, k, metric="cos"):
    if metric == "cos":
        Zn = Z / np.linalg.norm(Z, axis=1, keepdims=True).clip(1e-9)
        S = Zn @ Zn.T
    else:
        inter = Z @ Z.T
        n = Z.sum(1)
        S = inter / np.maximum(n[:, None] + n[None, :] - inter, 1e-9)
    np.fill_diagonal(S, -np.inf)
    nb = np.argsort(-S, 1)[:, :k]
    return r[nb].mean(1)


def main(a):
    d, w = sanitize(torch.load(a.graph, map_location="cpu", weights_only=False))
    X = d[w].x.numpy()[:, :EMB + PDIN].astype(np.float64)
    y = d[w].y.view(-1).numpy().astype(int)
    coh = pd.read_csv(a.cohort_csv).cohort.values.astype(int)
    A, _ = incidence(d, w)
    s = np.load(a.sentences)
    n = len(y)
    mx = np.full((n, EMB), -np.inf); mean = np.zeros((n, EMB))
    np.maximum.at(mx, s["woman"], s["emb"])
    np.add.at(mean, s["woman"], s["emb"])
    mean /= np.bincount(s["woman"], minlength=n)[:, None]

    print("=== P1 residual homophily (Mark 1 OOF, LR narrative + PDI-13) ===")
    oof = pd.read_csv(a.mark1_oof)
    p = oof["LR (narrative + PDI-13)"].values
    r = y - p
    spaces = {"narrative (document)": X[:, :EMB], "narrative (mean of sentences)": mean,
              "PDI profile": X[:, EMB:], "narrative + PDI (standardised)":
              np.c_[X[:, :EMB] / X[:, :EMB].std(), X[:, EMB:] / X[:, EMB:].std() * 0.2]}
    for name, Z in spaces.items():
        for k in [5, 10, 25]:
            rho, pv = stats.spearmanr(r, knn_mean(Z, r, k))
            print(f"  {name:<32} k={k:<3} residual correlation {rho:+.3f} (p={pv:.3f})")
    has = A.sum(1) > 0
    for k in [5, 10, 25]:
        rho, pv = stats.spearmanr(r[has], knn_mean(A, r, k, "jac")[has])
        print(f"  {'shared entities (Jaccard)':<32} k={k:<3} residual correlation "
              f"{rho:+.3f} (p={pv:.3f})")
    rho, _ = stats.spearmanr(y, knn_mean(X[:, :EMB], y.astype(float), 10))
    print(f"  (for scale: raw label homophily, narrative k=10: {rho:+.3f})")

    print("\n=== P2 does sentence-level content add to the document embedding? ===")
    Xs = np.c_[X, mx, mean]
    MX = list(range(EMB + PDIN, 2 * EMB + PDIN))
    MN = list(range(2 * EMB + PDIN, 3 * EMB + PDIN))
    doc = ("pca", PCA(10, random_state=0), EMB_COLS)
    pdi = ("pdi", "passthrough", PDI_COLS)
    base_g = {"clf__C": C_VALUES, "ct__pca__n_components": [10, 20]}
    specs = {
        "document + PDI": ([doc, pdi], base_g),
        "document + PDI + max-pooled sentences": (
            [doc, pdi, ("s", PCA(10, random_state=0), MX)],
            base_g | {"ct__s__n_components": [5, 10, 20]}),
        "document + PDI + mean-pooled sentences": (
            [doc, pdi, ("s", PCA(10, random_state=0), MN)],
            base_g | {"ct__s__n_components": [5, 10, 20]}),
        "max-pooled sentences + PDI (no document)": (
            [("s", PCA(10, random_state=0), MX), pdi],
            {"clf__C": C_VALUES, "ct__s__n_components": [10, 20]}),
    }
    strata = y * 2 + coh
    res = {m: [] for m in specs}
    oofs = []
    for seed in a.seeds:
        set_seed(seed)
        o = {m: np.zeros(n) for m in specs}
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xs, strata):
            for m, (parts, g) in specs.items():
                o[m][te], _ = fit_predict(pipe(parts), g, Xs, y, tr, te, seed)
        for m in specs:
            res[m].append(roc_auc_score(y, o[m]))
        oofs.append(o)
        print(f"  seed {seed}: " + " | ".join(f"{m} {res[m][-1]:.4f}" for m in specs),
              flush=True)
    base = "document + PDI"
    for m in specs:
        line = f"  {m:<44} AUC {np.mean(res[m]):.4f}"
        if m != base:
            ds, ps = zip(*[perm_test(y, o[m], o[base], n=2000, seed=s)
                           for s, o in zip(a.seeds, oofs)])
            line += (f"   Δ {np.mean(ds):+.4f}  perm p {min(ps):.3f}-{max(ps):.3f}"
                     f"  wins {sum(x > 0 for x in ds)}/{len(ds)}")
        print(line)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="perignnosis_graph_pooled.pt")
    ap.add_argument("--cohort-csv", default="pooled_cohort.csv")
    ap.add_argument("--sentences", default="narrative_sentences.npz")
    ap.add_argument("--mark1-oof", default="MARK1/results/results_narrative_short_pdi/oof_mean.csv")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    main(ap.parse_args())
