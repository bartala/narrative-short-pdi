"""
Graphical abstract with the ROC curves.

Usage:
    python make_graphical_abstract_roc.py --oof-dir results_narrative_short_pdi/ --out results_revision_extras/
"""

import argparse, glob, os, re
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Polygon
from sklearn.metrics import roc_curve, roc_auc_score

INK, MUTED = "#14181f", "#4a5361"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1b7f4f"
P_BG, P_BLUE, P_PINK, P_GREEN = "#f1f5fb", "#dbe8f8", "#fbe3e1", "#e4f4e6"
MODELS = [("LR (PDI-13)", "All 13 PDI items", BLUE),
          ("LR (narrative + six)", "Narrative + six PDI items", ORANGE)]
W, H = 130.0, 50.0


def load(oof_dir):
    files = sorted(glob.glob(os.path.join(oof_dir, "oof_seed*.csv")),
                   key=lambda f: int(re.findall(r"seed(\d+)", f)[0]))
    if not files:
        raise SystemExit(f"no oof_seed*.csv in {oof_dir}")
    raw = [pd.read_csv(f) for f in files]
    y = raw[0]["y_true"].values.astype(int)
    for d in raw[1:]:
        assert (d["y_true"].values == y).all(), "seed files are not row-aligned"
    missing = [c for c, *_ in MODELS if c not in raw[0].columns]
    if missing:
        raise SystemExit(f"columns {missing} not found. Available: "
                         f"{[c for c in raw[0].columns if c != 'y_true']}")
    P = {c: np.mean([d[c].values for d in raw], axis=0) for c, *_ in MODELS}
    print(f"{len(y)} women, {int(y.sum())} positive; {len(files)} repetitions")
    return y, P


def synthetic(seed=0):
    r = np.random.default_rng(seed)
    y = np.r_[np.ones(166), np.zeros(1098)].astype(int)
    z = r.normal(0, 1, y.size) + 1.9 * y
    return y, {"LR (PDI-13)": z + r.normal(0, .25, y.size),
               "LR (narrative + six)": z + r.normal(0, .25, y.size) + .05 * y}


def main(a):
    y, P = synthetic() if a.synthetic else load(a.oof_dir)
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    font = next((f for f in ("Liberation Sans", "Arial", "Helvetica") if f in have), "DejaVu Sans")
    plt.rcParams.update({"font.family": font, "pdf.fonttype": 42})

    fig = plt.figure(figsize=(13, 5))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

    def box(x, y0, w, h, fill, r=1.6):
        ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                    fc=fill, ec="none"))

    def txt(x, y0, s, size=15, bold=False, color=INK, ha="center", va="center"):
        ax.text(x, y0, s, ha=ha, va=va, fontsize=size, color=color, linespacing=1.25,
                fontweight="bold" if bold else "normal")

    def arrow(x1, y1, x2, y2, lw=2.2, col=MUTED, ms=20):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=ms,
                                     lw=lw, color=col, shrinkA=0, shrinkB=0))

    def doc_icon(x, y0, w=4.2, h=5.6):
        c = w * 0.3
        ax.add_patch(Polygon([(x, y0), (x + w, y0), (x + w, y0 + h - c), (x + w - c, y0 + h), (x, y0 + h)],
                             closed=True, fc="white", ec=INK, lw=1.8, joinstyle="round"))
        for i in range(3):
            yy = y0 + h * (0.24 + 0.2 * i)
            ax.plot([x + w * .2, x + w * (.8 if i else .6)], [yy, yy], color=INK, lw=1.8,
                    solid_capstyle="round")

    def list_icon(x, y0, w=4.2, h=5.6):
        ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0,rounding_size=0.5",
                                    fc="white", ec=INK, lw=1.8))
        for i in range(3):
            yy = y0 + h * (0.24 + 0.26 * i)
            ax.add_patch(Rectangle((x + w * .16, yy - .45), .9, .9, fc=INK, ec="none"))
            ax.plot([x + w * .5, x + w * .84], [yy, yy], color=INK, lw=1.8, solid_capstyle="round")

    box(1, 1.5, 33, 47, P_BG)
    txt(17.5, 43.8, f"{len(y):,} postpartum women", 17, True)
    txt(17.5, 39.2, f"{int(y.sum())} with probable CB-PTSD", 14.5, color=MUTED)
    box(3, 21.5, 29, 13.5, P_BLUE)
    doc_icon(5.2, 25.4)
    txt(11.6, 30.4, "Birth narrative", 16, True, ha="left")
    txt(11.6, 25.9, "in her own words", 14.5, ha="left")
    txt(17.5, 18.4, "+", 24, True)
    box(3, 3.5, 29, 12, P_PINK)
    list_icon(5.2, 6.7)
    txt(11.6, 11.7, "PDI questionnaire", 16, True, ha="left")
    txt(11.6, 7.2, "13 distress items", 14.5, ha="left")
    arrow(34.6, 25, 37.4, 25, lw=3, ms=24)

    box(38, 1.5, 31, 47, P_BG)
    txt(53.5, 43.8, "Model", 17, True)
    box(40, 28, 13, 11, P_BLUE); txt(46.5, 33.5, "Narrative\nembedding", 14.5, True)
    box(54, 28, 13, 11, P_PINK); txt(60.5, 33.5, "Six of 13\nPDI items", 14.5, True)
    ax.plot([46.5, 46.5, 60.5, 60.5], [28, 24.5, 24.5, 28], color=INK, lw=2)
    arrow(53.5, 24.5, 53.5, 20.6, lw=2, col=INK, ms=18)
    box(41, 11, 25, 9, "#cfdcf0"); txt(53.5, 15.5, "Logistic regression", 16, True)
    txt(53.5, 6, "repeated nested\ncross-validation", 14.5, color=MUTED)
    arrow(69.6, 25, 72.4, 25, lw=3, ms=24)

    box(73, 1.5, 35, 47, "white")
    ax.add_patch(FancyBboxPatch((73, 1.5), 35, 47, boxstyle="round,pad=0,rounding_size=1.6",
                                fc="none", ec="#c9d3e0", lw=1.2))
    txt(90.5, 43.8, "Identifying probable CB-PTSD", 16, True)
    rx = fig.add_axes([(73 + 7.2) / W, (1.5 + 9.3) / H, 25.6 / W, 28.8 / H])
    rx.plot([0, 1], [0, 1], color="#9aa3ad", lw=1.2, ls=(0, (2, 3)))
    handles = []
    for col_name, lab, col in MODELS:
        fpr, tpr, _ = roc_curve(y, P[col_name])
        auc = roc_auc_score(y, P[col_name])
        lab2 = lab.replace("Narrative + six", "Narrative +\nsix").replace("All 13", "All 13")
        print(f"{lab}: AUC {auc:.4f}")
        h, = rx.plot(fpr, tpr, color=col, lw=2.6, label=f"{lab2}\nAUC {auc:.3f}")
        handles.append(h)
    rx.set_xlim(0, 1); rx.set_ylim(0, 1.005); rx.set_aspect("equal", adjustable="box")
    rx.set_xticks([0, .5, 1]); rx.set_yticks([0, .5, 1])
    rx.set_xticklabels(["0", "0.5", "1"]); rx.set_yticklabels(["0", "0.5", "1"])
    rx.tick_params(labelsize=14, colors=INK, length=4, width=1)
    rx.set_xlabel("1 − specificity", fontsize=14.5, color=INK, labelpad=2)
    rx.set_ylabel("Sensitivity", fontsize=14.5, color=INK, labelpad=2)
    for s in ("top", "right"):
        rx.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        rx.spines[s].set_color(INK); rx.spines[s].set_linewidth(1)
    leg = rx.legend(handles=handles[::-1], loc="lower right", fontsize=14, frameon=True,
                    handlelength=1.0, handletextpad=0.5, labelspacing=0.6, borderaxespad=0.1,
                    borderpad=0.25)
    leg.get_frame().set_facecolor("white"); leg.get_frame().set_edgecolor("none")
    leg.get_frame().set_alpha(1)

    box(110, 1.5, 19, 47, P_GREEN)
    txt(119.5, 25, "Narrative +\n6 PDI items\nwas\nnon-inferior\nto all 13\nPDI items", 17, True, GREEN)

    if a.watermark:
        fig.text(.5, .5, a.watermark, ha="center", va="center", fontsize=60, color="red",
                 alpha=.22, rotation=18)
    os.makedirs(a.out, exist_ok=True)
    fig.savefig(os.path.join(a.out, "graphical_abstract.pdf"))
    fig.savefig(os.path.join(a.out, "graphical_abstract.svg"))
    fig.savefig(os.path.join(a.out, "graphical_abstract_600dpi.png"), dpi=600)
    print("wrote graphical_abstract.pdf / .svg / _600dpi.png (7800 x 3000 px)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof-dir", default="results_narrative_short_pdi/")
    ap.add_argument("--out", default="results_revision_extras/")
    ap.add_argument("--synthetic", action="store_true", help="layout test only, no data needed")
    ap.add_argument("--watermark", default="")
    main(ap.parse_args())
