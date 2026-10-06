"""
AUC comparison within each cohort.

Usage:
    python per_cohort_check.py --oof-dir results_pca_diagnostic/ --model "LR (narrative PCA + PDI)" --reference "LR (PDI-13)" --cohort-csv pooled_cohort.csv
"""

import argparse, glob, os, re
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score


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
    files = sorted(glob.glob(os.path.join(a.oof_dir, "oof_seed*.csv")),
                   key=lambda f: int(re.findall(r"seed(\d+)", f)[0]))
    if not files:
        raise SystemExit(f"no oof_seed*.csv in {a.oof_dir}")
    cohort = pd.read_csv(a.cohort_csv)["cohort"].values.astype(int)
    first = pd.read_csv(files[0])
    if len(first) != len(cohort):
        raise SystemExit(f"row mismatch: predictions {len(first)}, cohort file "
                         f"{len(cohort)} - the files are not aligned")
    for m in (a.model, a.reference):
        if m not in first.columns:
            raise SystemExit(f"'{m}' not in {list(first.columns)}")

    names = {0: "CBEx", 1: "COVID"}
    groups = [("pooled", np.ones(len(cohort), bool))] + \
             [(names.get(c, f"cohort {c}"), cohort == c) for c in sorted(set(cohort))]
    print(f"{a.model}  vs  {a.reference}\n")
    rows = []
    for gname, gm in groups:
        ds, ps, am, ar = [], [], [], []
        for f in files:
            d = pd.read_csv(f)
            s = int(re.findall(r"seed(\d+)", f)[0])
            y = d["y_true"].values[gm]
            p1, p2 = d[a.model].values[gm], d[a.reference].values[gm]
            am.append(roc_auc_score(y, p1)); ar.append(roc_auc_score(y, p2))
            dd, pv = perm_test(y, p1, p2, n=a.perms, seed=s)
            ds.append(dd); ps.append(pv)
        y0 = pd.read_csv(files[0])["y_true"].values[gm]
        print(f"  {gname:7s} n={gm.sum():<5} positives={int(y0.sum()):<4} "
              f"{a.model} {np.mean(am):.4f} | {a.reference} {np.mean(ar):.4f} | "
              f"Δ {np.mean(ds):+.4f}  perm p {min(ps):.4f}-{max(ps):.4f}  "
              f"wins {sum(x > 0 for x in ds)}/{len(ds)}")
        rows.append({"group": gname, "n": int(gm.sum()), "positives": int(y0.sum()),
                     "auc_model": np.mean(am), "auc_reference": np.mean(ar),
                     "delta": np.mean(ds), "perm_p_min": min(ps),
                     "perm_p_max": max(ps), "wins": sum(x > 0 for x in ds)})
    out = os.path.join(a.oof_dir, "per_cohort_check.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nsaved {out}")
    print("\nHow to read this: if the gain is positive within BOTH cohorts, it is not")
    print("an artefact of mixing two studies. If it appears only in the pooled row,")
    print("the model was mostly detecting which study a woman came from.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--cohort-csv", default="pooled_cohort.csv")
    ap.add_argument("--perms", type=int, default=10000)
    main(ap.parse_args())
