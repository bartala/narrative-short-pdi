"""
Sensitivity analysis: recalibration fitted inside the model's own outer folds; net benefit and sensitivity at 80% specificity.

Usage:
    python nested_recalibration.py
"""

import glob, re
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from clinical_utility import (cv_recalibrate, net_benefit, sens_at_spec, boot_ci,
                              logit, DCA_GRID, PRIMARY_RANGE)

OOF_DIR = "results_narrative_short_pdi"
COHORT_CSV = "pooled_cohort.csv"
M6N, M13, M6 = "LR (narrative + six)", "LR (PDI-13)", "LR (six)"
PAIRS = [(M6N, M13), (M6N, M6), (M13, M6)]
BOOT = 1000


def nested_recalibrate(y, p, strata, seed, k=5):
    out = np.zeros_like(p, dtype=float)
    lp = logit(p).reshape(-1, 1)
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(lp, strata):
        m = LogisticRegression(C=1e6, max_iter=1000).fit(lp[tr], y[tr])
        out[te] = m.predict_proba(lp[te])[:, 1]
    return out


def main():
    files = sorted(glob.glob(f"{OOF_DIR}/oof_seed*.csv"),
                   key=lambda f: int(re.findall(r"seed(\d+)", f)[0]))
    seeds = [int(re.findall(r"seed(\d+)", f)[0]) for f in files]
    raw = [pd.read_csv(f) for f in files]
    y = raw[0]["y_true"].values.astype(int)
    cohort = pd.read_csv(COHORT_CSV).cohort.values.astype(int)
    assert len(cohort) == len(y)
    strata = y * 2 + cohort
    models = [M6N, M13, M6]
    print(f"{len(y)} women, {y.sum()} positive, {len(files)} repetitions")

    versions = {
        "original (separate 5-fold split)":
            [{m: cv_recalibrate(y, d[m].values, s) for m in models}
             for d, s in zip(raw, seeds)],
        "nested (model's outer folds)":
            [{m: nested_recalibrate(y, d[m].values, strata, s) for m in models}
             for d, s in zip(raw, seeds)],
    }
    lo_t, hi_t = PRIMARY_RANGE
    mask = (DCA_GRID >= lo_t - 1e-9) & (DCA_GRID <= hi_t + 1e-9)

    def mean_nb(yy, pp):
        return float(net_benefit(yy, pp, DCA_GRID[mask]).mean())

    for name, recal in versions.items():
        print(f"\n=== {name} ===")
        for m1, m2 in PAIRS:
            d = np.mean([mean_nb(y, r[m1]) - mean_nb(y, r[m2]) for r in recal])
            lo, hi, _ = boot_ci(y, recal, mean_nb, m1, m2, BOOT)
            print(f"  NB diff 0.10-0.30  {m1} - {m2}: {d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
        for m in models:
            raw_s = np.mean([sens_at_spec(y, d[m].values, 0.80) for d in raw])
            rec_s = np.mean([sens_at_spec(y, r[m], 0.80) for r in recal])
            print(f"  sens@spec0.80  {m}: raw scores {raw_s:.3f} | recalibrated {rec_s:.3f}")


if __name__ == "__main__":
    main()
