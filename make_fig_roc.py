"""
Figure: ROC curves of narrative + six PDI items vs all 13 PDI items.

Usage:
    python make_fig_roc.py
"""

import glob, os, re
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

OOF_DIR = "results_narrative_short_pdi"
OUT = "results_revision_extras"
MODELS = [
    ("LR (PDI-13)", "All 13 PDI items", "#2a78d6", "-"),
    ("LR (narrative + six)", "Narrative + six PDI items", "#eb6834", "-"),
]
INK, GRIDC = "#52514e", "#d9d8d2"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": INK,
})


def main():
    files = sorted(glob.glob(os.path.join(OOF_DIR, "oof_seed*.csv")),
                   key=lambda f: int(re.findall(r"seed(\d+)", f)[0]))
    raw = [pd.read_csv(f) for f in files]
    y = raw[0]["y_true"].values.astype(int)
    for d in raw[1:]:
        assert (d["y_true"].values == y).all(), "seed files are not row-aligned"

    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.plot([0, 1], [0, 1], ":", color="gray", linewidth=0.9)
    rows = []
    for col, label, c, ls in MODELS:
        p = np.mean([d[col].values for d in raw], axis=0)
        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)
        rows.append(pd.DataFrame({"model": col, "fpr": fpr, "tpr": tpr}))
        ax.plot(fpr, tpr, ls=ls, color=c, linewidth=1.3,
                label=f"{label} (AUC {auc:.3f})")
        print(f"{label}: {len(fpr)} points, AUC {auc:.4f}")

    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xlabel("False positive rate (1 − specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.grid(True, color=GRIDC, linewidth=0.5); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()

    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, "fig_roc.pdf"))
    fig.savefig(os.path.join(OUT, "fig_roc.png"), dpi=300)
    pd.concat(rows, ignore_index=True).to_csv(os.path.join(OUT, "roc_points.csv"),
                                              index=False)
    print(f"{len(files)} repetitions, {len(y)} women, {y.sum()} positive; "
          f"saved to {OUT}/")


if __name__ == "__main__":
    main()
