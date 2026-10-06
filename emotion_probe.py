"""
Exploratory: emotion, sentiment and LIWC features added to logistic regression.

Usage:
    python emotion_probe.py --seeds 1 2 3 4 5
"""

import argparse
import numpy as np, pandas as pd, torch
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from fair_benchmark import EMB, PDIN, sanitize, set_seed
from pca_narrative_diagnostic import fit_predict, perm_test, C_VALUES, K_GRID, EMB_COLS, PDI_COLS
from graph_signal_diagnostic import pipe
from narrative_short_pdi import SIX_COLS

LIWC_AFFECT = ["affect", "posemo", "negemo", "anx", "anger", "sad", "Tone"]


def run(specs, X, y, strata, seeds, tests, title):
    res, oofs = {m: [] for m in specs}, []
    for seed in seeds:
        set_seed(seed)
        o = {m: np.zeros(len(y)) for m in specs}
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, strata):
            for m, (parts, g) in specs.items():
                o[m][te], _ = fit_predict(pipe(parts), g, X, y, tr, te, seed)
        for m in specs:
            res[m].append(roc_auc_score(y, o[m]))
        oofs.append(o)
        print(f"  seed {seed}: " + " | ".join(f"{m} {res[m][-1]:.4f}" for m in specs), flush=True)
    print(f"\n=== {title} ===")
    for m in specs:
        print(f"  {m:<44} AUC {np.mean(res[m]):.4f}")
    for m1, m2 in tests:
        r = [perm_test(y, o[m1], o[m2], n=2000, seed=s) for s, o in zip(seeds, oofs)]
        print(f"  {m1} - {m2}: Δ {np.mean([x[0] for x in r]):+.4f}  perm p "
              f"{min(x[1] for x in r):.3f}-{max(x[1] for x in r):.3f}  "
              f"wins {sum(x[0] > 0 for x in r)}/{len(r)}")


def main(a):
    d, w = sanitize(torch.load(a.graph, map_location="cpu", weights_only=False))
    X0 = d[w].x.numpy()[:, :EMB + PDIN].astype(np.float64)
    y = d[w].y.view(-1).numpy().astype(int)
    meta = pd.read_csv(a.cohort_csv)
    coh = meta.cohort.values.astype(int)
    E = pd.read_csv(a.emotions)
    if not (E.record_id.values == meta.record_id.values).all():
        raise SystemExit("emotion features are not row-aligned with the cohort csv")
    sent_cols = [c for c in E.columns if c.startswith("sent_")]
    emo_cols = [c for c in E.columns if c.startswith("emo_")]
    X = np.c_[X0, E[sent_cols].values, E[emo_cols].values]
    S = list(range(EMB + PDIN, EMB + PDIN + len(sent_cols)))
    M = list(range(S[-1] + 1, X.shape[1]))
    print(f"{len(y)} women, {y.sum()} positive; {len(sent_cols)} sentiment + "
          f"{len(emo_cols)} emotion features", flush=True)

    from scipy.stats import spearmanr
    rho = sorted(((spearmanr(E[c], y)[0], c) for c in sent_cols + emo_cols),
                 key=lambda t: -abs(t[0]))[:10]
    print("strongest single features (Spearman with outcome): " +
          ", ".join(f"{c} {r:+.2f}" for r, c in rho))

    pca = ("pca", PCA(10, random_state=0), EMB_COLS)
    pdi = ("pdi", "passthrough", PDI_COLS)
    six = ("pdi", "passthrough", SIX_COLS)
    sen = ("sen", "passthrough", S)
    emo = ("emo", PCA(10, random_state=0), M)
    g_p = {"clf__C": C_VALUES, "ct__pca__n_components": K_GRID}
    g_e = {"ct__emo__n_components": [5, 10, 20]}
    specs = {
        "base (narrative + PDI-13)": ([pca, pdi], g_p),
        "+ sentiment": ([pca, pdi, sen], g_p),
        "+ GoEmotions": ([pca, pdi, emo], g_p | g_e),
        "+ both": ([pca, pdi, sen, emo], g_p | g_e),
        "six: base (narrative + six)": ([pca, six], g_p),
        "six: + both": ([pca, six, sen, emo], g_p | g_e),
        "PDI-13 + emotions (no narrative embedding)": (
            [pdi, sen, emo], {"clf__C": C_VALUES} | g_e),
        "PDI-13 only": ([pdi], {"clf__C": C_VALUES}),
        "emotions only": ([sen, emo], {"clf__C": C_VALUES} | g_e),
    }
    B = "base (narrative + PDI-13)"
    run(specs, X, y, y * 2 + coh, a.seeds,
        [("+ sentiment", B), ("+ GoEmotions", B), ("+ both", B),
         ("six: + both", "six: base (narrative + six)"),
         ("PDI-13 + emotions (no narrative embedding)", "PDI-13 only"),
         ("PDI-13 + emotions (no narrative embedding)", B)],
        "pooled sample, OOF AUC (mean over seeds)")

    L = pd.read_csv(a.liwc)
    ids = meta.record_id.str.replace("covid_", "", regex=False)
    L["record_id"] = L.record_id.astype(str)
    Lm = pd.DataFrame({"record_id": ids}).merge(L[["record_id"] + LIWC_AFFECT], how="left")
    idx = np.where((coh == 1) & Lm[LIWC_AFFECT].notna().all(1).values)[0]
    print(f"\nLIWC: {len(idx)} COVID women matched (of {(coh == 1).sum()}), "
          f"{y[idx].sum()} positive", flush=True)
    XL = np.c_[X0[idx], Lm.loc[idx, LIWC_AFFECT].values, E[sent_cols + emo_cols].values[idx]]
    LC = list(range(EMB + PDIN, EMB + PDIN + len(LIWC_AFFECT)))
    specs_l = {"base (narrative + PDI-13)": ([pca, pdi], g_p),
               "+ LIWC affect": ([pca, pdi, ("liwc", "passthrough", LC)], g_p)}
    run(specs_l, XL, y[idx], y[idx], a.seeds, [("+ LIWC affect", B)],
        "COVID cohort only, OOF AUC (mean over seeds)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="perignnosis_graph_pooled.pt")
    ap.add_argument("--cohort-csv", default="pooled_cohort.csv")
    ap.add_argument("--emotions", default="emotion_features.csv")
    ap.add_argument("--liwc", default="covid_LIWC_narratives.csv")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    main(ap.parse_args())
