"""
Do sentence-level emotion and sentiment scores add to the narrative embedding?

Usage:
    python emotion_confirmatory.py --mark1 results_narrative_short_pdi/ --out results_emotion_confirmatory/
"""

import argparse, os, time
import numpy as np, pandas as pd, torch
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from fair_benchmark import EMB, PDIN, sanitize, set_seed
from pca_narrative_diagnostic import fit_predict, C_VALUES, K_GRID, EMB_COLS, PDI_COLS
from graph_signal_diagnostic import pipe
from narrative_short_pdi import SIX_COLS, boot_diff, boot_auc, delong

M_SIXE = "LR (narrative + six + emotions)"
M_13E = "LR (narrative + PDI-13 + emotions)"
REF = {M_SIXE: "LR (narrative + six)", M_13E: "LR (narrative + PDI-13)"}


def main(a):
    os.makedirs(a.out, exist_ok=True)
    d, w = sanitize(torch.load(a.graph, map_location="cpu", weights_only=False))
    X0 = d[w].x.numpy()[:, :EMB + PDIN].astype(np.float64)
    y = d[w].y.view(-1).numpy().astype(int)
    meta = pd.read_csv(a.cohort_csv)
    if not np.allclose(meta[[f"pdi_q{i}" for i in range(1, 14)]].values, X0[:, EMB:]):
        raise SystemExit("cohort csv is not row-aligned with the graph")
    E = pd.read_csv(a.emotions)
    if not (E.record_id.values == meta.record_id.values).all():
        raise SystemExit("emotion features are not row-aligned")
    coh = meta.cohort.values.astype(int)
    strata = y * 2 + coh
    sc = [c for c in E.columns if c.startswith("sent_")]
    ec = [c for c in E.columns if c.startswith("emo_")]
    X = np.c_[X0, E[sc].values, E[ec].values]
    S = list(range(EMB + PDIN, EMB + PDIN + len(sc)))
    M = list(range(S[-1] + 1, X.shape[1]))
    pca = ("pca", PCA(10, random_state=0), EMB_COLS)
    emo = [("sen", "passthrough", S), ("emo", PCA(10, random_state=0), M)]
    g = {"clf__C": C_VALUES, "ct__pca__n_components": K_GRID,
         "ct__emo__n_components": [5, 10, 20]}
    specs = {M_SIXE: ([pca, ("pdi", "passthrough", SIX_COLS)] + emo, g),
             M_13E: ([pca, ("pdi", "passthrough", PDI_COLS)] + emo, g)}
    print(f"{len(y)} women, {y.sum()} positive; {len(a.seeds)} repeats; "
          f"{len(sc)} sentiment + {len(ec)} emotion features", flush=True)

    oofs = []
    for seed in a.seeds:
        path = os.path.join(a.out, f"oof_seed{seed}.csv")
        if os.path.exists(path):
            oofs.append(pd.read_csv(path)); continue
        set_seed(seed)
        t0 = time.time()
        o = {m: np.zeros(len(y)) for m in specs}
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, strata):
            for m, (parts, gg) in specs.items():
                o[m][te], _ = fit_predict(pipe(parts), gg, X, y, tr, te, seed)
        oo = pd.DataFrame(o | {"y_true": y}); oo.to_csv(path, index=False); oofs.append(oo)
        print(f"  seed {seed} ({time.time()-t0:.0f}s): " + " | ".join(
            f"{m} {roc_auc_score(y, o[m]):.4f}" for m in specs), flush=True)

    mk1 = [pd.read_csv(os.path.join(a.mark1, f"oof_seed{s}.csv")) for s in a.seeds]
    if not all(np.array_equal(m.y_true.values, y) for m in mk1):
        raise SystemExit("Mark 1 predictions are not row-aligned")
    P = {m: np.mean([o[m].values for o in oofs], 0) for m in specs}
    P |= {r: np.mean([m[r].values for m in mk1], 0) for r in REF.values()}

    print(f"\n=== AUC (mean OOF prediction over {len(a.seeds)} repeats), 95% bootstrap CI ===")
    for m in [M_SIXE, REF[M_SIXE], M_13E, REF[M_13E]]:
        v, lo, hi = boot_auc(y, P[m], strata, a.boot)
        print(f"  {m:<36} {v:.4f} [{lo:.4f}, {hi:.4f}]")

    rows = []
    def report(tag, m1, m2, idx=slice(None), primary=False):
        dd, lo, hi = boot_diff(y[idx], P[m1][idx], P[m2][idx], strata[idx], a.boot)
        _, dlo, dhi, dp = delong(y[idx], P[m1][idx], P[m2][idx])
        verdict = ("  -> SUPERIOR (CI excludes 0)" if lo > 0 else "  -> not shown") if primary else ""
        print(f"  {tag:<40} Δ {dd:+.4f}  bootstrap [{lo:+.4f}, {hi:+.4f}]  "
              f"DeLong [{dlo:+.4f}, {dhi:+.4f}] p={dp:.3f}{verdict}")
        rows.append({"comparison": tag, "delta": dd, "boot_lo": lo, "boot_hi": hi,
                     "delong_p": dp})

    print("\n=== PRIMARY: emotions added to narrative + six ===")
    report("six items: + emotions vs Mark 1", M_SIXE, REF[M_SIXE], primary=True)
    print("\n=== SECONDARY ===")
    report("13 items: + emotions vs Mark 1", M_13E, REF[M_13E])
    for c in sorted(set(coh)):
        idx = coh == c
        report(f"cohort {c}, six items: + emotions", M_SIXE, REF[M_SIXE], idx)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "comparisons.csv"), index=False)
    print(f"\nsaved to {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="perignnosis_graph_pooled.pt")
    ap.add_argument("--cohort-csv", default="pooled_cohort.csv")
    ap.add_argument("--emotions", default="emotion_features.csv")
    ap.add_argument("--mark1", default="results_narrative_short_pdi")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 11)))
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default="results_emotion_confirmatory")
    main(ap.parse_args())
