"""
Calibration, decision curves on recalibrated probabilities, and operating points.

Usage:
    python clinical_utility.py --oof-dir results_selection_symmetric_pooled/ --model "PeriGNNosis (own selection)" --references "LR (own selection)" "LR (PDI-13)" --band-model "LR (PDI-13)" --cohort-csv pooled_cohort.csv --out results_clinical_utility/
"""

import argparse, glob, os, re
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold

EPS = 1e-6
DCA_GRID = np.round(np.arange(0.02, 0.605, 0.01), 2)
DCA_REPORT = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
PRIMARY_RANGE = (0.10, 0.30)


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def calib_stats(y, p):
    lp = logit(p).reshape(-1, 1)
    slope_fit = LogisticRegression(C=1e6, max_iter=1000).fit(lp, y)
    slope = float(slope_fit.coef_[0, 0])
    off = lp.ravel()
    a = 0.0
    for _ in range(50):
        mu = 1 / (1 + np.exp(-(off + a)))
        g = np.sum(y - mu)
        h = np.sum(mu * (1 - mu)) + 1e-9
        a += g / h
    return brier_score_loss(y, np.clip(p, 0, 1)), float(a), slope


def cv_recalibrate(y, p, seed, k=5):
    out = np.zeros_like(p, dtype=float)
    lp = logit(p).reshape(-1, 1)
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(lp, y):
        m = LogisticRegression(C=1e6, max_iter=1000).fit(lp[tr], y[tr])
        out[te] = m.predict_proba(lp[te])[:, 1]
    return out


def net_benefit(y, p, pts):
    n = len(y)
    res = np.empty(len(pts))
    for i, pt in enumerate(pts):
        pos = p >= pt
        tp = np.sum(pos & (y == 1))
        fp = np.sum(pos & (y == 0))
        res[i] = tp / n - fp / n * pt / (1 - pt)
    return res


def treat_all(y, pts):
    prev = y.mean()
    return prev - (1 - prev) * pts / (1 - pts)


def sens_at_spec(y, s, spec):
    thr = np.quantile(s[y == 0], spec)
    return float(np.mean(s[y == 1] > thr))


def spec_at_sens(y, s, sens):
    thr = np.quantile(s[y == 1], 1 - sens)
    return float(np.mean(s[y == 0] < thr))


def f1_nested(y, p, seed, k=5):
    tp = fp = fn = 0
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(
            y.reshape(-1, 1), y):
        cand = np.unique(np.quantile(p[tr], np.linspace(0.01, 0.99, 99)))
        best_t, best_f = cand[0], -1.0
        for t in cand:
            pr = p[tr] >= t
            a_tp = np.sum(pr & (y[tr] == 1))
            a_fp = np.sum(pr & (y[tr] == 0))
            a_fn = np.sum(~pr & (y[tr] == 1))
            f = 2 * a_tp / max(2 * a_tp + a_fp + a_fn, 1)
            if f > best_f:
                best_f, best_t = f, t
        pr = p[te] >= best_t
        tp += np.sum(pr & (y[te] == 1))
        fp += np.sum(pr & (y[te] == 0))
        fn += np.sum(~pr & (y[te] == 1))
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    sens = tp / max(tp + fn, 1)
    ppv = tp / max(tp + fp, 1)
    return f1, sens, ppv


def perm_auc(y, p1, p2, n=2000, seed=0):
    rng = np.random.RandomState(seed)
    obs = roc_auc_score(y, p1) - roc_auc_score(y, p2)
    hits = 0
    for _ in range(n):
        sw = rng.rand(len(y)) < 0.5
        a, b = np.where(sw, p2, p1), np.where(sw, p1, p2)
        if abs(roc_auc_score(y, a) - roc_auc_score(y, b)) >= abs(obs) - 1e-12:
            hits += 1
    return obs, (hits + 1) / (n + 1)


def strat_boot_idx(y, rng):
    i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]
    return np.concatenate([rng.choice(i1, len(i1)), rng.choice(i0, len(i0))])


def boot_ci(y, preds_by_seed, stat, model, ref, B, seed=0):
    rng = np.random.RandomState(seed)
    diffs = np.empty(B)
    for b in range(B):
        idx = strat_boot_idx(y, rng)
        vals = [stat(y[idx], d[model][idx]) - stat(y[idx], d[ref][idx])
                for d in preds_by_seed]
        diffs[b] = np.mean(vals)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return lo, hi, float(np.mean(diffs > 0))


def main(a):
    os.makedirs(a.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.oof_dir, "oof_seed*.csv")),
                   key=lambda f: int(re.findall(r"seed(\d+)", f)[0]))
    if not files:
        raise SystemExit(f"no oof_seed*.csv in {a.oof_dir}")
    raw = [pd.read_csv(f) for f in files]
    y = raw[0]["y_true"].values.astype(int)
    for d in raw[1:]:
        assert (d["y_true"].values == y).all(), "seed files are not row-aligned"
    models = [a.model] + a.references
    for m in models + [a.band_model]:
        assert m in raw[0].columns, f"column '{m}' not in {list(raw[0].columns)}"
    seeds = [int(re.findall(r"seed(\d+)", f)[0]) for f in files]
    print(f"{len(y)} women, {int(y.sum())} positive ({y.mean():.1%}); "
          f"{len(files)} seeds; model = {a.model}", flush=True)

    cohort = None
    if a.cohort_csv:
        c = pd.read_csv(a.cohort_csv)
        if len(c) != len(y):
            print(f"!! cohort file has {len(c)} rows, predictions have {len(y)} - "
                  f"skipping the per-cohort split")
        else:
            cohort = c["cohort"].values.astype(int)

    print("\n=== A. calibration of raw OOF probabilities (mean over seeds) ===")
    rows = []
    for m in models:
        st = np.array([calib_stats(y, d[m].values) for d in raw])
        rows.append({"model": m, "brier": st[:, 0].mean(),
                     "calib_intercept": st[:, 1].mean(), "calib_slope": st[:, 2].mean()})
    calib = pd.DataFrame(rows)
    print(calib.round(4).to_string(index=False))
    print("  (ideal: intercept 0, slope 1. LR baselines are trained with balanced "
          "class weights, so their raw probabilities are inflated by design.)")
    calib.to_csv(os.path.join(a.out, "A_calibration_raw.csv"), index=False)

    recal = [{m: cv_recalibrate(y, d[m].values, s) for m in models}
             for d, s in zip(raw, seeds)]
    nb = {m: np.mean([net_benefit(y, r[m], DCA_GRID) for r in recal], axis=0)
          for m in models}
    nb_all = treat_all(y, DCA_GRID)
    dca = pd.DataFrame({"threshold": DCA_GRID, "treat_all": nb_all,
                        "treat_none": 0.0} | {m: nb[m] for m in models})
    dca.to_csv(os.path.join(a.out, "B_decision_curve.csv"), index=False)

    print("\n=== B. net benefit, cross-validated recalibrated probabilities "
          "(mean over seeds) ===")
    rep = dca[dca.threshold.isin(DCA_REPORT)].set_index("threshold")
    print(rep.round(4).to_string())

    lo_t, hi_t = PRIMARY_RANGE
    mask = (DCA_GRID >= lo_t - 1e-9) & (DCA_GRID <= hi_t + 1e-9)

    def mean_nb(yy, pp):
        return float(net_benefit(yy, pp, DCA_GRID[mask]).mean())

    print(f"\n  PRIMARY: mean net-benefit difference over thresholds "
          f"{lo_t:.2f}-{hi_t:.2f}")
    prim_rows = []
    for ref in a.references:
        d_pt = float((nb[a.model][mask] - nb[ref][mask]).mean())
        lo, hi, pgt = boot_ci(y, recal, mean_nb, a.model, ref, a.boot)
        better = DCA_GRID[(nb[a.model] - nb[ref]) > 0]
        rng_txt = (f"{better.min():.2f}-{better.max():.2f}" if len(better)
                   else "none")
        print(f"  {a.model} - {ref}: {d_pt:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
              f"P(boot>0)={pgt:.2f}   thresholds where model NB > ref: {rng_txt}")
        prim_rows.append({"endpoint": "mean NB diff 0.10-0.30", "reference": ref,
                          "diff": d_pt, "ci_low": lo, "ci_high": hi,
                          "p_boot_gt0": pgt})

    print("\n=== C. operating points from raw scores (mean over seeds) ===")
    op_defs = [("sens@spec0.80", lambda yy, s: sens_at_spec(yy, s, 0.80), True),
               ("sens@spec0.90", lambda yy, s: sens_at_spec(yy, s, 0.90), False),
               ("spec@sens0.80", lambda yy, s: spec_at_sens(yy, s, 0.80), False),
               ("spec@sens0.90", lambda yy, s: spec_at_sens(yy, s, 0.90), False)]
    raw_dicts = [{m: d[m].values for m in models} for d in raw]
    op_rows = []
    for name, fn, primary in op_defs:
        vals = {m: np.mean([fn(y, d[m]) for d in raw_dicts]) for m in models}
        line = f"  {name}{' (PRIMARY)' if primary else ''}: " + " | ".join(
            f"{m} {v:.3f}" for m, v in vals.items())
        print(line)
        for ref in a.references:
            lo, hi, pgt = boot_ci(y, raw_dicts, fn, a.model, ref, a.boot)
            diff = vals[a.model] - vals[ref]
            print(f"      {a.model} - {ref}: {diff:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]"
                  f"  P(boot>0)={pgt:.2f}")
            op_rows.append({"endpoint": name, "reference": ref, "diff": diff,
                            "ci_low": lo, "ci_high": hi, "p_boot_gt0": pgt,
                            "primary": primary})
            if primary:
                prim_rows.append({"endpoint": name, "reference": ref, "diff": diff,
                                  "ci_low": lo, "ci_high": hi, "p_boot_gt0": pgt})
    pd.DataFrame(op_rows).to_csv(os.path.join(a.out, "C_operating_points.csv"),
                                 index=False)

    print(f"\n=== D. AUC within terciles of '{a.band_model}' predicted risk ===")
    band_rows = []
    groups = [("all", np.ones(len(y), bool))]
    if cohort is not None:
        groups += [(f"cohort {c}", cohort == c) for c in sorted(np.unique(cohort))]
    for gname, gmask in groups:
        for d, s in zip(raw, seeds):
            ref_risk = d[a.band_model].values
            lo_q, hi_q = np.quantile(ref_risk[gmask], [1 / 3, 2 / 3])
            bands = {"low": ref_risk < lo_q,
                     "middle (ambiguous)": (ref_risk >= lo_q) & (ref_risk <= hi_q),
                     "high": ref_risk > hi_q}
            for bname, bm in bands.items():
                sel = bm & gmask
                yy = y[sel]
                if len(np.unique(yy)) < 2:
                    continue
                row = {"group": gname, "band": bname, "seed": s, "n": int(sel.sum()),
                       "positives": int(yy.sum())}
                for m in models:
                    row[f"auc {m}"] = roc_auc_score(yy, d[m].values[sel])
                for ref in a.references:
                    dd, pv = perm_auc(yy, d[a.model].values[sel], d[ref].values[sel],
                                      n=a.perms, seed=s)
                    row[f"diff vs {ref}"] = dd
                    row[f"perm p vs {ref}"] = pv
                band_rows.append(row)
    bdf = pd.DataFrame(band_rows)
    bdf.to_csv(os.path.join(a.out, "D_band_auc.csv"), index=False)
    for (gname, bname), g in bdf.groupby(["group", "band"], sort=False):
        head = (f"  [{gname}] {bname:20s} n={int(g.n.mean())} "
                f"pos={int(g.positives.mean())}  ")
        head += " | ".join(f"{m} {g[f'auc {m}'].mean():.3f}" for m in models)
        print(head)
        for ref in a.references:
            dd = g[f"diff vs {ref}"]
            pv = g[f"perm p vs {ref}"]
            tag = "   (PRIMARY)" if (gname == "all" and bname.startswith("middle")) else ""
            print(f"      vs {ref}: Δ {dd.mean():+.4f}  perm p {pv.min():.3f}-{pv.max():.3f}"
                  f"  wins {int((dd > 0).sum())}/{len(dd)}{tag}")
            if gname == "all" and bname.startswith("middle"):
                prim_rows.append({"endpoint": "AUC diff, middle tercile",
                                  "reference": ref, "diff": float(dd.mean()),
                                  "perm_p_min": float(pv.min()),
                                  "perm_p_max": float(pv.max()),
                                  "wins": f"{int((dd > 0).sum())}/{len(dd)}"})

    prim = pd.DataFrame(prim_rows)
    prim.to_csv(os.path.join(a.out, "PRIMARY_endpoints.csv"), index=False)
    print("\n=== PRIMARY endpoints (pre-specified) ===")
    print(prim.round(4).to_string(index=False))

    try:
        plot_dca(dca, models, a.out)
    except Exception as ex:
        print(f"(figure skipped: {type(ex).__name__}: {ex})")
    print(f"\nsaved to {a.out}")


def plot_dca(dca, models, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
    SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
    DASH = ["-", (0, (6, 2)), (0, (2, 2))]
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=200, facecolor=SURF)
    ax.set_facecolor(SURF)
    x = dca.threshold.values
    ax.plot(x, dca.treat_all, color=MUTED, lw=1.5, ls=(0, (1, 2)), label="Treat all")
    ax.plot(x, dca.treat_none, color=MUTED, lw=1.5, ls="-", label="Treat none")
    ymax = max(float(dca[m].max()) for m in models) + 0.02
    ymin = -0.02
    for i, m in enumerate(models[:3]):
        ax.plot(x, dca[m], color=SERIES[i], lw=2, ls=DASH[i], label=m)
    ends = sorted(((float(dca[m].values[-1]), m) for m in models[:3]))
    gap = 0.045 * (ymax - ymin)
    placed = []
    for yv, m in ends:
        yl = max(yv, placed[-1] + gap) if placed else yv
        placed.append(yl)
        ax.annotate(m, (x[-1], yv), xytext=(x[-1] + 0.008, yl),
                    textcoords="data", va="center", fontsize=7, color=INK2,
                    annotation_clip=False)
    ax.axvspan(*PRIMARY_RANGE, color=GRID, alpha=0.5, lw=0)
    ax.text(np.mean(PRIMARY_RANGE), ymax - 0.005, "pre-specified range",
            ha="center", va="top", fontsize=7, color=INK2)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Threshold probability", color=INK2, fontsize=9)
    ax.set_ylabel("Net benefit", color=INK2, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(axis="y", color=GRID, lw=0.6)
    leg = ax.legend(frameon=False, fontsize=7, loc="upper right")
    for t in leg.get_texts():
        t.set_color(INK)
    ax.set_title("Decision curves (cross-validated recalibration)",
                 color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "B_decision_curve.png"), facecolor=SURF)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof-dir", required=True)
    ap.add_argument("--model", default="PeriGNNosis (own selection)")
    ap.add_argument("--references", nargs="+",
                    default=["LR (own selection)", "LR (PDI-13)"])
    ap.add_argument("--band-model", default="LR (PDI-13)")
    ap.add_argument("--cohort-csv", default="")
    ap.add_argument("--out", default="results_clinical_utility")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--perms", type=int, default=2000)
    main(ap.parse_args())
