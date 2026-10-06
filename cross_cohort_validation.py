"""
Cross-cohort validation: train on one cohort, test on the other.

Usage:
    python cross_cohort_validation.py --out results_cross_cohort/
"""

import argparse, os
import numpy as np, pandas as pd, torch
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from fair_benchmark import EMB, PDIN, sanitize
from pca_narrative_diagnostic import C_VALUES, K_GRID, EMB_COLS, PDI_COLS
from graph_signal_diagnostic import pipe
from narrative_short_pdi import (SIX_COLS, MARGIN, M_REF, M_PRI, M_SIX, M_NAR,
                                 boot_diff, boot_auc, delong)


def main(a):
    os.makedirs(a.out, exist_ok=True)
    d, w = sanitize(torch.load(a.graph, map_location="cpu", weights_only=False))
    X = d[w].x.numpy()[:, :EMB + PDIN].astype(np.float64)
    y = d[w].y.view(-1).numpy().astype(int)
    meta = pd.read_csv(a.cohort_csv)
    if not np.allclose(meta[[f"pdi_q{i}" for i in range(1, 14)]].values, X[:, EMB:]):
        raise SystemExit("cohort csv is not row-aligned with the graph")
    coh = meta["cohort"].values.astype(int)
    pca = ("pca", PCA(10, random_state=0), EMB_COLS)
    g_c = {"clf__C": C_VALUES}
    g_p = {"clf__C": C_VALUES, "ct__pca__n_components": K_GRID}
    specs = {M_REF: ([("p", "passthrough", PDI_COLS)], g_c),
             M_PRI: ([pca, ("p", "passthrough", SIX_COLS)], g_p),
             M_SIX: ([("p", "passthrough", SIX_COLS)], g_c),
             M_NAR: ([pca], g_p)}
    rows = []
    for train_c in sorted(set(coh)):
        tr, te = np.where(coh == train_c)[0], np.where(coh != train_c)[0]
        print(f"\n=== train cohort {train_c} (n={len(tr)}, {y[tr].sum()} pos) -> "
              f"test cohort {1 - train_c} (n={len(te)}, {y[te].sum()} pos) ===")
        P = {}
        for m, (parts, grid) in specs.items():
            gs = GridSearchCV(pipe(parts), grid, scoring="roc_auc", n_jobs=-1,
                              cv=StratifiedKFold(5, shuffle=True, random_state=0))
            P[m] = gs.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
            v, lo, hi = boot_auc(y[te], P[m], y[te], a.boot)
            print(f"  {m:<24} AUC {v:.4f} [{lo:.4f}, {hi:.4f}]")
        for tag, m1, m2, mg in [("narrative + six vs PDI-13", M_PRI, M_REF, MARGIN),
                                ("narrative + six vs six", M_PRI, M_SIX, None)]:
            dd, lo, hi = boot_diff(y[te], P[m1], P[m2], y[te], a.boot)
            _, dlo, dhi, dp = delong(y[te], P[m1], P[m2])
            ni = "" if mg is None else ("  -> NON-INFERIOR" if lo > -mg else "  -> not shown")
            print(f"  {tag:<28} Δ {dd:+.4f}  bootstrap [{lo:+.4f}, {hi:+.4f}]  "
                  f"DeLong p={dp:.3f}{ni}")
            rows.append({"train": train_c, "test": 1 - train_c, "comparison": tag,
                         "delta": dd, "lo": lo, "hi": hi, "delong_p": dp})
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "cross_cohort.csv"), index=False)
    print(f"\nsaved to {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="perignnosis_graph_pooled.pt")
    ap.add_argument("--cohort-csv", default="pooled_cohort.csv")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default="results_cross_cohort")
    main(ap.parse_args())
