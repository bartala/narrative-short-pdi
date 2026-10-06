"""
Number of PDI items needed with and without the narrative (top-k items by univariate AUC).

Usage:
    python pdi_replacement.py --graph perignnosis_graph_pooled.pt --seeds 1 2 3 4 5 --out results_pdi_replacement/
"""

import argparse, os, time
import numpy as np, pandas as pd, torch
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from fair_benchmark import EMB, PDIN, sanitize, set_seed
from pca_narrative_diagnostic import (fit_predict, perm_test, C_VALUES, K_GRID,
                                      EMB_COLS, PDI_COLS)
from graph_signal_diagnostic import incidence, nbr_features, pipe

K_ITEMS = [0, 1, 2, 3, 4, 6, 8, 10, 13]
M_PDI, M_NAR, M_GR = "PDI top-k", "narrative + top-k", "graph + narrative + top-k"
MODELS = [M_PDI, M_NAR, M_GR]


def rank_items(X, y, tr):
    auc = [roc_auc_score(y[tr], X[tr, c]) for c in PDI_COLS]
    return [PDI_COLS[i] for i in np.argsort([-max(a, 1 - a) for a in auc])]


def main(a):
    os.makedirs(a.out, exist_ok=True)
    raw = torch.load(a.graph, map_location="cpu", weights_only=False)
    data, woman = sanitize(raw)
    Xw = data[woman].x.numpy()[:, :EMB + PDIN].astype(np.float64)
    y = data[woman].y.view(-1).numpy().astype(int)
    A, _ = incidence(data, woman)
    print(f"{len(y)} women, {int(y.sum())} positive; k items {K_ITEMS}", flush=True)

    g_c = {"clf__C": C_VALUES}
    g_pca = {"clf__C": C_VALUES, "ct__pca__n_components": K_GRID}
    pca = ("pca", PCA(10, random_state=0), EMB_COLS)

    rows, ranks = [], []
    for seed in a.seeds:
        set_seed(seed)
        t0 = time.time()
        oof = {(m, k): np.zeros(len(y)) for m in MODELS for k in K_ITEMS}
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(skf.split(Xw, y)):
            order = rank_items(Xw, y, tr)
            ranks.append({"seed": seed, "fold": fold,
                          "order": " ".join(f"Q{c - EMB + 1}" for c in order)})
            X = np.c_[Xw, nbr_features(A, y, tr, seed)]
            nbr = [X.shape[1] - 2, X.shape[1] - 1]
            for k in K_ITEMS:
                pdi = [("pdi", "passthrough", order[:k])] if k else []
                if k:
                    oof[M_PDI, k][te], _ = fit_predict(pipe(pdi), g_c, X, y, tr, te, seed)
                else:
                    oof[M_PDI, k][te] = y[tr].mean()
                oof[M_NAR, k][te], _ = fit_predict(pipe([pca] + pdi), g_pca,
                                                   X, y, tr, te, seed)
                oof[M_GR, k][te], _ = fit_predict(
                    pipe([pca] + pdi + [("nbr", "passthrough", nbr)]), g_pca,
                    X, y, tr, te, seed)
        for (m, k), p in oof.items():
            rows.append({"seed": seed, "model": m, "k": k, "auc": roc_auc_score(y, p)})
        pd.DataFrame({f"{m}|{k}": p for (m, k), p in oof.items()} | {"y_true": y}
                     ).to_csv(os.path.join(a.out, f"oof_seed{seed}.csv"), index=False)
        print(f"  seed {seed} ({time.time()-t0:.0f}s): " + " | ".join(
            f"k={k} {roc_auc_score(y, oof[M_PDI, k]):.3f}/"
            f"{roc_auc_score(y, oof[M_NAR, k]):.3f}/{roc_auc_score(y, oof[M_GR, k]):.3f}"
            for k in K_ITEMS), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "replacement_auc.csv"), index=False)
    pd.DataFrame(ranks).to_csv(os.path.join(a.out, "item_ranks.csv"), index=False)
    oofs = {s: pd.read_csv(os.path.join(a.out, f"oof_seed{s}.csv")) for s in a.seeds}

    def test(m1, k1, m2, k2):
        ds, ps = [], []
        for s, o in oofs.items():
            d, p = perm_test(o["y_true"].values, o[f"{m1}|{k1}"].values,
                             o[f"{m2}|{k2}"].values, n=a.perms, seed=s)
            ds.append(d); ps.append(p)
        return np.mean(ds), min(ps), max(ps), sum(d > 0 for d in ds)

    print(f"\n=== OOF AUC by number of PDI items (mean over seeds; "
          f"perm tests {a.perms} draws) ===")
    print(f"  {'k':>3}  {'PDI':>6}  {'+narr':>6}  {'+graph':>6}   "
          f"narrative gain (p range, wins)      graph gain (p range, wins)")
    mean = df.groupby(["model", "k"]).auc.mean()
    for k in K_ITEMS:
        dn = test(M_NAR, k, M_PDI, k)
        dg = test(M_GR, k, M_NAR, k)
        print(f"  {k:>3}  {mean[M_PDI, k]:.4f}  {mean[M_NAR, k]:.4f}  {mean[M_GR, k]:.4f}"
              f"   {dn[0]:+.4f} (p {dn[1]:.3f}-{dn[2]:.3f}, {dn[3]}/5)"
              f"   {dg[0]:+.4f} (p {dg[1]:.3f}-{dg[2]:.3f}, {dg[3]}/5)", flush=True)

    print(f"\n=== replacement: vs LR on all 13 items ({mean[M_PDI, 13]:.4f}) ===")
    for m in MODELS:
        ok = None
        for k in K_ITEMS:
            d, pmin, _, _ = test(m, k, M_PDI, 13)
            if d >= 0 or pmin >= 0.05:
                ok = k
                break
        print(f"  {m:<28} smallest k not significantly worse in any seed: {ok}")
    print("\nThe gap between the 'PDI top-k' and 'narrative + top-k' answers is the")
    print("number of questionnaire items the narrative can stand in for.")
    print(f"\nsaved to {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="perignnosis_graph_pooled.pt")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--perms", type=int, default=2000)
    ap.add_argument("--out", default="results_pdi_replacement")
    main(ap.parse_args())
