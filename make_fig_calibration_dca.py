"""
Figure: calibration before and after recalibration, and decision curves.

Usage:
    python make_fig_calibration_dca.py
"""

import glob, os, re, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from clinical_utility import cv_recalibrate, net_benefit, treat_all

OOF_DIR = "results_narrative_short_pdi"
OUT = "results_revision_extras"
GRID = np.round(np.arange(0.02, 0.4001, 0.01), 2)
SHADE = (0.10, 0.30)
MODELS = [
    ("LR (PDI-13)", "All 13 PDI items", "#2a78d6", "-", "o"),
    ("LR (narrative + six)", "Narrative + six PDI items", "#eb6834", "-", "s"),
    ("LR (six)", "Six PDI items", "#1baf7a", "--", "^"),
]
INK, GRIDC, BAND = "#52514e", "#d9d8d2", "#f3f2ee"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "text.color": "black", "axes.labelcolor": "black",
    "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": INK,
})


def deciles(y, p, g=10):
    order = np.argsort(p, kind="mergesort")
    return [(p[i].mean(), y[i].mean(), len(i)) for i in np.array_split(order, g)]


def style(ax):
    ax.grid(True, color=GRIDC, linewidth=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.6)


def main():
    files = sorted(glob.glob(os.path.join(OOF_DIR, "oof_seed*.csv")),
                   key=lambda f: int(re.findall(r"seed(\d+)", f)[0]))
    seeds = [int(re.findall(r"seed(\d+)", f)[0]) for f in files]
    raw = [pd.read_csv(f) for f in files]
    need = [m[0] for m in MODELS] + ["y_true"]
    missing = [c for c in need if c not in raw[0].columns]
    if missing:
        print("missing columns:", missing)
        print("available:", list(raw[0].columns))
        sys.exit(1)
    y = raw[0]["y_true"].values.astype(int)
    for d in raw[1:]:
        assert (d["y_true"].values == y).all(), "seed files are not row-aligned"
    print(f"read {len(y)} women, {int(y.sum())} positive, {len(files)} repetitions")

    rows, curves = [], {}
    for col, label, *_ in MODELS:
        p_raw = np.mean([d[col].values for d in raw], axis=0)
        rec = [cv_recalibrate(y, d[col].values, s) for d, s in zip(raw, seeds)]
        p_rec = np.mean(rec, axis=0)
        for panel, p in (("A_raw", p_raw), ("B_recalibrated", p_rec)):
            for g, (mp, obs, n) in enumerate(deciles(y, p), 1):
                rows.append({"panel": panel, "model": label, "group": g,
                             "mean_predicted": mp, "observed_rate": obs, "n": n})
        curves[label] = np.mean([net_benefit(y, r, GRID) for r in rec], axis=0)
    curves["Refer all"] = treat_all(y, GRID)
    curves["Refer none"] = np.zeros_like(GRID)
    for label, v in curves.items():
        rows += [{"panel": "C_net_benefit", "model": label, "threshold": t,
                  "net_benefit": nb} for t, nb in zip(GRID, v)]
    data = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.9))
    b_max = data.loc[data.panel == "B_recalibrated",
                     ["mean_predicted", "observed_rate"]].values.max()
    b_lim = np.ceil(b_max * 10 + 0.5) / 10
    for ax, panel, lim, title in ((axes[0], "A_raw", 1.0, "A  Before recalibration"),
                                  (axes[1], "B_recalibrated", b_lim,
                                   "B  After recalibration")):
        ax.plot([0, lim], [0, lim], ":", color="gray", linewidth=0.9)
        for col, label, c, ls, mk in MODELS:
            s = data[(data.panel == panel) & (data.model == label)]
            ax.plot(s.mean_predicted, s.observed_rate, ls=ls, color=c, marker=mk,
                    markersize=3.6, markerfacecolor="white", markeredgewidth=0.9,
                    linewidth=1.1)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal")
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed rate of probable CB-PTSD")
        ax.set_title(title, loc="left", fontweight="bold")
        style(ax)

    ax = axes[2]
    ax.axvspan(*SHADE, color=BAND, zorder=0, linewidth=0)
    ax.plot(GRID, curves["Refer all"], color=INK, linewidth=0.8)
    ax.plot(GRID, curves["Refer none"], color=INK, linewidth=0.8)
    for col, label, c, ls, mk in MODELS:
        ax.plot(GRID, curves[label], ls=ls, color=c, marker=mk, markevery=4,
                markersize=3.2, markerfacecolor="white", markeredgewidth=0.8,
                linewidth=1.1)
    ylo, yhi = -0.02, 0.125
    ax.set_xlim(GRID[0], GRID[-1]); ax.set_ylim(ylo, yhi)
    ax.text(0.124, 0.02, "refer all", color=INK, fontsize=7, ha="left", va="center")
    ax.text(GRID[-1], 0.002, "refer none", color=INK, fontsize=7,
            ha="right", va="bottom")
    ax.set_box_aspect(1)
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title("C  Decision curves", loc="left", fontweight="bold")
    style(ax)

    handles = [Line2D([], [], ls=ls, color=c, marker=mk, markersize=4,
                      markerfacecolor="white", markeredgewidth=0.9, linewidth=1.1,
                      label=label) for _, label, c, ls, mk in MODELS]
    handles.append(Line2D([], [], ls=":", color="gray", linewidth=0.9,
                          label="Perfect calibration"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.0), handlelength=2.6, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=1.6)

    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, "fig_calibration_dca.pdf"))
    fig.savefig(os.path.join(OUT, "fig_calibration_dca.png"), dpi=300)
    data.to_csv(os.path.join(OUT, "fig_calibration_dca_data.csv"), index=False)

    band = (GRID >= SHADE[0] - 1e-9) & (GRID <= SHADE[1] + 1e-9)
    print("mean net benefit, thresholds 0.10-0.30:",
          {m[1]: round(float(curves[m[1]][band].mean()), 4) for m in MODELS})
    print(f"panel B axis limit {b_lim}; saved to {OUT}/")


if __name__ == "__main__":
    main()
