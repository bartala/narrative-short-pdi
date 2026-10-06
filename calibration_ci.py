"""
Calibration intercept and slope of the raw out-of-fold probabilities, with bootstrap 95% CIs.

Usage:
    python calibration_ci.py
"""

import glob, re, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from clinical_utility import calib_stats

warnings.filterwarnings("ignore")
OOF_DIR = "results_narrative_short_pdi"
MODELS = ["LR (narrative + six)", "LR (PDI-13)", "LR (six)"]
DRAWS = 1000


def main():
    files = sorted(glob.glob(f"{OOF_DIR}/oof_seed*.csv"),
                   key=lambda f: int(re.findall(r"seed(\d+)", f)[0]))
    raw = [pd.read_csv(f) for f in files]
    y = raw[0]["y_true"].values.astype(int)
    P = {m: np.stack([d[m].values for d in raw]) for m in MODELS}

    def stat(idx):
        return {m: np.mean([calib_stats(y[idx], P[m][s, idx])[1:]
                            for s in range(len(raw))], axis=0) for m in MODELS}

    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    rng = np.random.RandomState(0)
    draws = [np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
             for _ in range(DRAWS)]
    boot = Parallel(n_jobs=-1)(delayed(stat)(i) for i in draws)
    point = stat(np.arange(len(y)))

    print(f"{len(y)} women, {y.sum()} positive, {len(raw)} repetitions, "
          f"{DRAWS} stratified paired draws")
    for m in MODELS:
        b = np.array([d[m] for d in boot])
        lo, hi = np.percentile(b, [2.5, 97.5], axis=0)
        print(f"{m:<24} intercept {point[m][0]:+.3f} [{lo[0]:+.3f}, {hi[0]:+.3f}]   "
              f"slope {point[m][1]:.3f} [{lo[1]:.3f}, {hi[1]:.3f}]")


if __name__ == "__main__":
    main()
