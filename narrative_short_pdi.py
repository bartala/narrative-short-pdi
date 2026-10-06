"""
Main analysis: narrative + six PDI items vs all 13 PDI items (non-inferiority).

Usage:
    python narrative_short_pdi.py --graph perignnosis_graph_pooled.pt --cohort-csv pooled_cohort.csv --out results_narrative_short_pdi/
"""

import argparse, os, time
import numpy as np, pandas as pd, torch
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from fair_benchmark import EMB, PDIN, sanitize, set_seed
from pca_narrative_diagnostic import fit_predict, C_VALUES, K_GRID, EMB_COLS, PDI_COLS
from graph_signal_diagnostic import pipe

SIX = [1, 2, 3, 5, 12, 13]
MARGIN = 0.02
SIX_COLS = [EMB + q - 1 for q in SIX]
COH, WC = EMB + PDIN, EMB + PDIN + 1

M_REF = "LR (PDI-13)"
M_PRI = "LR (narrative + six)"
M_SIX = "LR (six)"
M_N13 = "LR (narrative + PDI-13)"
M_NAR = "LR (narrative only)"
M_LEN = "LR (word count + six)"
M_REFC = "LR (PDI-13 + cohort)"
M_PRIC = "LR (narrative + six + cohort)"
M_SIXC = "LR (six + cohort)"


def specs():
    pca = ("pca", PCA(10, random_state=0), EMB_COLS)
    g_c = {"clf__C": C_VALUES}
    g_p = {"clf__C": C_VALUES, "ct__pca__n_components": K_GRID}
    cols = lambda name, c: (name, "passthrough", c)
    return {
        M_REF: ([cols("pdi", PDI_COLS)], g_c),
        M_PRI: ([pca, cols("pdi", SIX_COLS)], g_p),
        M_SIX: ([cols("pdi", SIX_COLS)], g_c),
        M_N13: ([pca, cols("pdi", PDI_COLS)], g_p),
        M_NAR: ([pca], g_p),
        M_LEN: ([cols("pdi", SIX_COLS + [WC])], g_c),
        M_REFC: ([cols("pdi", PDI_COLS + [COH])], g_c),
        M_PRIC: ([pca, cols("pdi", SIX_COLS + [COH])], g_p),
        M_SIXC: ([cols("pdi", SIX_COLS + [COH])], g_c),
    }


def delong(y, p1, p2):
    pos, neg = p1[y == 1], p1[y == 0]
    def v(p):
        a, b = p[y == 1][:, None], p[y == 0][None, :]
        psi = (a > b) + 0.5 * (a == b)
        return psi.mean(1), psi.mean(0), psi.mean()
    v10a, v01a, A1 = v(p1)
    v10b, v01b, A2 = v(p2)
    s10 = np.cov(np.vstack([v10a, v10b]))
    s01 = np.cov(np.vstack([v01a, v01b]))
    var = ((s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / len(pos)
           + (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / len(neg))
    d, se = A1 - A2, np.sqrt(var)
    return d, d - 1.96 * se, d + 1.96 * se, 2 * stats.norm.sf(abs(d) / se)


def boot_diff(y, p1, p2, strata, n, seed=0):
    rng = np.random.RandomState(seed)
    groups = [np.where(strata == s)[0] for s in np.unique(strata)]
    ds = np.empty(n)
    for i in range(n):
        idx = np.concatenate([rng.choice(g, len(g)) for g in groups])
        ds[i] = roc_auc_score(y[idx], p1[idx]) - roc_auc_score(y[idx], p2[idx])
    return roc_auc_score(y, p1) - roc_auc_score(y, p2), *np.percentile(ds, [2.5, 97.5])


def boot_auc(y, p, strata, n, seed=0):
    rng = np.random.RandomState(seed)
    groups = [np.where(strata == s)[0] for s in np.unique(strata)]
    a = [roc_auc_score(y[i], p[i]) for i in
         (np.concatenate([rng.choice(g, len(g)) for g in groups]) for _ in range(n))]
    return roc_auc_score(y, p), *np.percentile(a, [2.5, 97.5])


def main(a):
    os.makedirs(a.out, exist_ok=True)
    raw = torch.load(a.graph, map_location="cpu", weights_only=False)
    data, woman = sanitize(raw)
    Xw = data[woman].x.numpy()[:, :EMB + PDIN].astype(np.float64)
    y = data[woman].y.view(-1).numpy().astype(int)
    meta = pd.read_csv(a.cohort_csv)
    pdi_csv = meta[[f"pdi_q{i}" for i in range(1, 14)]].values
    if len(meta) != len(y) or not np.allclose(pdi_csv, Xw[:, EMB:EMB + PDIN]):
        raise SystemExit("cohort csv is not row-aligned with the graph")
    cohort = meta["cohort"].values.astype(int)
    wc = np.log(meta["cb_delivery_narrative"].fillna("").str.split().str.len()
                .clip(lower=1).values)
    X = np.c_[Xw, cohort, wc]
    strata = y * 2 + cohort
    names = {c: f"cohort {c}" for c in np.unique(cohort)}
    print(f"{len(y)} women, {y.sum()} positive; " + "; ".join(
        f"{names[c]}: n={int((cohort == c).sum())}, {y[cohort == c].mean():.1%} positive"
        for c in names) + f"; {len(a.seeds)} repeats of 5-fold CV", flush=True)

    sp = specs()
    rows, oofs = [], []
    for seed in a.seeds:
        path = os.path.join(a.out, f"oof_seed{seed}.csv")
        if os.path.exists(path):
            oofs.append(pd.read_csv(path)); print(f"  seed {seed}: cached"); continue
        set_seed(seed)
        t0 = time.time()
        oof = {m: np.zeros(len(y)) for m in sp}
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        for tr, te in skf.split(X, strata):
            for m, (parts, grid) in sp.items():
                oof[m][te], _ = fit_predict(pipe(parts), grid, X, y, tr, te, seed)
        o = pd.DataFrame(oof | {"y_true": y})
        o.to_csv(path, index=False)
        oofs.append(o)
        print(f"  seed {seed} ({time.time()-t0:.0f}s): " + " | ".join(
            f"{m} {roc_auc_score(y, oof[m]):.4f}" for m in sp), flush=True)
    for s, o in zip(a.seeds, oofs):
        for m in sp:
            rows.append({"seed": s, "model": m, "auc": roc_auc_score(y, o[m])})
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(os.path.join(a.out, "auc_per_seed.csv"), index=False)
    P = {m: np.mean([o[m].values for o in oofs], 0) for m in sp}
    pd.DataFrame(P | {"y_true": y, "cohort": cohort}).to_csv(
        os.path.join(a.out, "oof_mean.csv"), index=False)

    out = []
    def report(tag, m1, m2, idx=slice(None), margin=None):
        yy, s = y[idx], strata[idx]
        d, lo, hi = boot_diff(yy, P[m1][idx], P[m2][idx], s, a.boot)
        _, dlo, dhi, dp = delong(yy, P[m1][idx], P[m2][idx])
        line = (f"  {tag:<34} Δ {d:+.4f}  bootstrap 95% CI [{lo:+.4f}, {hi:+.4f}]  "
                f"DeLong [{dlo:+.4f}, {dhi:+.4f}] p={dp:.3f}")
        if margin is not None:
            line += "  -> " + ("NON-INFERIOR" if lo > -margin else "not shown")
        print(line, flush=True)
        out.append({"comparison": tag, "model": m1, "reference": m2, "delta": d,
                    "boot_lo": lo, "boot_hi": hi, "delong_lo": dlo, "delong_hi": dhi,
                    "delong_p": dp})

    print(f"\n=== AUC (mean OOF prediction over {len(a.seeds)} repeats), "
          f"bootstrap 95% CI, {a.boot} draws ===")
    aucs = []
    for m in sp:
        v, lo, hi = boot_auc(y, P[m], strata, a.boot)
        sd = per_seed[per_seed.model == m].auc.std()
        aucs.append({"model": m, "auc": v, "lo": lo, "hi": hi, "sd_over_seeds": sd})
        print(f"  {m:<32} {v:.4f} [{lo:.4f}, {hi:.4f}]  (sd over seeds {sd:.4f})")
    pd.DataFrame(aucs).to_csv(os.path.join(a.out, "auc_ci.csv"), index=False)

    print(f"\n=== PRIMARY: non-inferiority, margin {MARGIN} ===")
    report("narrative + six vs PDI-13", M_PRI, M_REF, margin=MARGIN)
    d, lo, hi = out[-1]["delta"], out[-1]["boot_lo"], out[-1]["boot_hi"]
    print("  margin sensitivity: " + ", ".join(
        f"{mg}: {'non-inferior' if lo > -mg else 'not shown'}" for mg in [0.01, 0.02, 0.03]))

    print("\n=== S1 narrative added value / S3 length / S4 context ===")
    report("S1 narrative + six vs six", M_PRI, M_SIX)
    report("S3 narrative + six vs wordcount + six", M_PRI, M_LEN)
    report("S3 wordcount + six vs six", M_LEN, M_SIX)
    report("S4 six vs PDI-13", M_SIX, M_REF)
    report("S4 narrative + PDI-13 vs PDI-13", M_N13, M_REF)

    print("\n=== S2a within cohort (same OOF predictions) ===")
    for c in names:
        idx = cohort == c
        print(f" {names[c]} (n={idx.sum()}, {y[idx].sum()} positive): " + ", ".join(
            f"{m} {roc_auc_score(y[idx], P[m][idx]):.3f}"
            for m in [M_REF, M_PRI, M_SIX, M_NAR]))
        report(f"{names[c]}: narrative + six vs PDI-13", M_PRI, M_REF, idx, MARGIN)
        report(f"{names[c]}: narrative + six vs six", M_PRI, M_SIX, idx)
    print("\n=== S2b cohort-adjusted models ===")
    report("narrative+six+cohort vs PDI-13+cohort", M_PRIC, M_REFC, margin=MARGIN)
    report("narrative+six+cohort vs six+cohort", M_PRIC, M_SIXC)
    pc = cross_val_predict(pipe([("pca", PCA(20, random_state=0), EMB_COLS)]), X, cohort,
                           cv=StratifiedKFold(5, shuffle=True, random_state=0),
                           method="predict_proba")[:, 1]
    print(f"  cohort alone predicts the outcome with AUC {roc_auc_score(y, cohort == 0):.3f};"
          f" the narrative (PCA-20, 5-fold CV) predicts the cohort with AUC "
          f"{roc_auc_score(cohort, pc):.3f}")

    pd.DataFrame(out).to_csv(os.path.join(a.out, "comparisons.csv"), index=False)
    print(f"\nsaved to {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="perignnosis_graph_pooled.pt")
    ap.add_argument("--cohort-csv", default="pooled_cohort.csv")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 11)))
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default="results_narrative_short_pdi")
    main(ap.parse_args())
