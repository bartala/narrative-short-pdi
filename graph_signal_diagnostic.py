"""
Does the knowledge graph add information beyond the compressed narrative?

Usage:
    python graph_signal_diagnostic.py --graph perignnosis_graph_pooled.pt --seeds 1 2 3 4 5 --out results_graph_signal/
    python graph_signal_diagnostic.py --graph perignnosis_graph_pooled_grounded.pt --seeds 1 2 3 4 5 --out results_graph_signal_grounded/
"""

import argparse, os, time
import numpy as np, pandas as pd, torch
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fair_benchmark import EMB, PDIN, sanitize, set_seed
from pca_narrative_diagnostic import (lr, fit_predict, perm_test, C_VALUES, K_GRID,
                                      EMB_COLS, PDI_COLS)

MIN_DF = 5
SVD_K = [5, 10, 20]
M_PDI = "LR (PDI-13)"
M_BASE = "LR (narrative PCA + PDI)"
M_ENT = "+ entities"
M_SVD = "+ entity SVD"
M_NBR = "+ neighbour label rate"
M_ATTR = "+ UMLS attributes"
M_ENTONLY = "LR (entities only)"
MODELS = [M_PDI, M_BASE, M_ENT, M_SVD, M_NBR, M_ATTR, M_ENTONLY]


def incidence(data, woman):
    nw = data[woman].num_nodes
    blocks, attrs = [], []
    width = max(data[nt].x.size(1) for nt in data.node_types if nt != woman)
    for nt in data.node_types:
        if nt == woman:
            continue
        M = np.zeros((nw, data[nt].num_nodes), dtype=np.float64)
        for (s, _, t) in data.edge_types:
            if s == woman and t == nt:
                ei = data[s, _, t].edge_index.numpy()
                M[ei[0], ei[1]] = 1.0
        blocks.append(M)
        x = data[nt].x[:, EMB:].numpy().astype(np.float64)
        attrs.append(np.pad(x, ((0, 0), (0, width - EMB - x.shape[1]))))
    return np.concatenate(blocks, 1), np.concatenate(attrs, 0)


def nbr_rate(A, y, src, dst, prior):
    idf = np.log(len(src) / (1.0 + A[src].sum(0)))
    idf = np.clip(idf, 0, None)
    S = (A[dst] * idf) @ A[src].T
    den = S.sum(1)
    rate = np.where(den > 0, (S @ y[src]) / np.maximum(den, 1e-12), prior)
    return rate, (den > 0).astype(float)


def nbr_features(A, y, tr, seed):
    prior = y[tr].mean()
    f = np.zeros((len(y), 2))
    inner = StratifiedKFold(5, shuffle=True, random_state=seed)
    for a, b in inner.split(tr, y[tr]):
        r, has = nbr_rate(A, y, tr[a], tr[b], prior)
        f[tr[b]] = np.c_[r, has]
    rest = np.setdiff1d(np.arange(len(y)), tr)
    r, has = nbr_rate(A, y, tr, rest, prior)
    f[rest] = np.c_[r, has]
    return f


def pipe(parts):
    return Pipeline([("ct", ColumnTransformer(parts)),
                     ("sc", StandardScaler()), ("clf", lr())])


def main(a):
    os.makedirs(a.out, exist_ok=True)
    raw = torch.load(a.graph, map_location="cpu", weights_only=False)
    data, woman = sanitize(raw)
    Xw = data[woman].x.numpy()[:, :EMB + PDIN].astype(np.float64)
    y = data[woman].y.view(-1).numpy().astype(int)
    A, attr = incidence(data, woman)
    attr = attr[:, (attr != 0).any(0)]
    W_attr = np.log1p(A @ attr)
    models = MODELS if W_attr.shape[1] else [m for m in MODELS if m != M_ATTR]
    dfreq = A.sum(0)
    print(f"{len(y)} women, {int(y.sum())} positive; {A.shape[1]} entities, "
          f"{A.sum(1).mean():.1f} per woman, {(A.sum(1) == 0).sum()} women with none; "
          f"{(dfreq >= MIN_DF).sum()} entities linked to >= {MIN_DF} women; "
          f"{W_attr.shape[1]} external attributes", flush=True)

    pca = ("pca", PCA(10, random_state=0), EMB_COLS)
    pdi = ("pdi", "passthrough", PDI_COLS)
    g_c = {"clf__C": C_VALUES}
    g_pca = {"clf__C": C_VALUES, "ct__pca__n_components": K_GRID}

    rows = []
    for seed in a.seeds:
        set_seed(seed)
        t0 = time.time()
        oof = {m: np.zeros(len(y)) for m in models}
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(skf.split(Xw, y)):
            keep = np.where(A[tr].sum(0) >= MIN_DF)[0]
            E = A[:, keep]
            N = nbr_features(A, y, tr, seed)
            X = np.c_[Xw, E, N, W_attr]
            ent = list(range(EMB + PDIN, EMB + PDIN + len(keep)))
            nbr = [EMB + PDIN + len(keep), EMB + PDIN + len(keep) + 1]
            att = list(range(nbr[-1] + 1, X.shape[1]))

            oof[M_PDI][te], _ = fit_predict(pipe([pdi]), g_c, X, y, tr, te, seed)
            oof[M_BASE][te], _ = fit_predict(pipe([pca, pdi]), g_pca, X, y, tr, te, seed)
            oof[M_ENT][te], _ = fit_predict(
                pipe([pca, pdi, ("ent", "passthrough", ent)]), g_pca, X, y, tr, te, seed)
            oof[M_SVD][te], _ = fit_predict(
                pipe([pca, pdi, ("svd", TruncatedSVD(10, random_state=0), ent)]),
                g_pca | {"ct__svd__n_components": SVD_K}, X, y, tr, te, seed)
            oof[M_NBR][te], _ = fit_predict(
                pipe([pca, pdi, ("nbr", "passthrough", nbr)]), g_pca, X, y, tr, te, seed)
            if M_ATTR in models:
                oof[M_ATTR][te], _ = fit_predict(
                    pipe([pca, pdi, ("att", "passthrough", att)]), g_pca, X, y, tr, te,
                    seed)
            oof[M_ENTONLY][te], _ = fit_predict(
                pipe([("ent", "passthrough", ent)]), g_c, X, y, tr, te, seed)
        for m in models:
            rows.append({"seed": seed, "model": m, "auc": roc_auc_score(y, oof[m])})
        pd.DataFrame(oof | {"y_true": y}).to_csv(
            os.path.join(a.out, f"oof_seed{seed}.csv"), index=False)
        print(f"  seed {seed} ({time.time()-t0:.0f}s): " + " | ".join(
            f"{m} {roc_auc_score(y, oof[m]):.4f}" for m in models), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "graph_signal_auc.csv"), index=False)
    print("\n=== pooled OOF AUC ===")
    print(df.groupby("model")["auc"].agg(["mean", "std"]).round(4)
            .sort_values("mean", ascending=False).to_string())

    print(f"\n=== paired permutation vs {M_BASE} (10,000 draws) ===")
    for m in [m for m in [M_ENT, M_SVD, M_NBR, M_ATTR] if m in models]:
        ds, ps = [], []
        for seed in a.seeds:
            o = pd.read_csv(os.path.join(a.out, f"oof_seed{seed}.csv"))
            d, pv = perm_test(o["y_true"].values, o[m].values, o[M_BASE].values,
                              n=10000, seed=seed)
            ds.append(d); ps.append(pv)
        print(f"  {m:<24} Δ {np.mean(ds):+.4f}  perm p {min(ps):.4f}-{max(ps):.4f}"
              f"  wins {sum(d > 0 for d in ds)}/{len(ds)}")
    print("\nIf no '+' model beats the base, the graph carries nothing beyond the")
    print("compressed narrative, and a GNN on it cannot be expected to either.")
    print(f"\nsaved to {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="perignnosis_graph_pooled.pt")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--out", default="results_graph_signal")
    main(ap.parse_args())
