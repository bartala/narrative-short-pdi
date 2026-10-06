"""
Does the woman-word graph add to the narrative embedding?

Usage:
    python word_graph_confirmatory.py --mark1 results_narrative_short_pdi/ --out results_word_graph_confirmatory/
"""

import argparse, os, time
import numpy as np, pandas as pd, torch
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fair_benchmark import EMB, PDIN, sanitize, set_seed
from pca_narrative_diagnostic import lr, C_VALUES
from narrative_short_pdi import SIX, boot_diff, boot_auc, delong

M_13W = "LR (narrative + PDI-13 + word graph)"
M_SIXW = "LR (narrative + six + word graph)"
REF = {M_13W: "LR (narrative + PDI-13)", M_SIXW: "LR (narrative + six)"}
EMB_NAMES = [f"e{i}" for i in range(EMB)]
PDI_NAMES = [f"pdi_q{i}" for i in range(1, PDIN + 1)]


def model(items):
    words = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=5,
                                                sublinear_tf=True, lowercase=True,
                                                token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z']+\b")),
                      ("svd", TruncatedSVD(50, random_state=0))])
    ct = ColumnTransformer([("pca", PCA(10, random_state=0), EMB_NAMES),
                            ("pdi", "passthrough", items),
                            ("words", words, "text")])
    return Pipeline([("ct", ct), ("sc", StandardScaler()), ("clf", lr())])


GRID = {"clf__C": C_VALUES, "ct__pca__n_components": [10, 20, 50],
        "ct__words__svd__n_components": [20, 50, 100]}


def main(a):
    os.makedirs(a.out, exist_ok=True)
    d, w = sanitize(torch.load(a.graph, map_location="cpu", weights_only=False))
    X0 = d[w].x.numpy()[:, :EMB + PDIN].astype(np.float64)
    y = d[w].y.view(-1).numpy().astype(int)
    meta = pd.read_csv(a.cohort_csv)
    if not np.allclose(meta[PDI_NAMES].values, X0[:, EMB:]):
        raise SystemExit("cohort csv is not row-aligned with the graph")
    coh = meta.cohort.values.astype(int)
    strata = y * 2 + coh
    df = pd.DataFrame(X0, columns=EMB_NAMES + PDI_NAMES)
    df["text"] = meta.cb_delivery_narrative.fillna("").values
    specs = {M_13W: model(PDI_NAMES), M_SIXW: model([f"pdi_q{q}" for q in SIX])}
    print(f"{len(y)} women, {y.sum()} positive; {len(a.seeds)} repeats", flush=True)

    oofs = []
    for seed in a.seeds:
        path = os.path.join(a.out, f"oof_seed{seed}.csv")
        if os.path.exists(path):
            oofs.append(pd.read_csv(path)); continue
        set_seed(seed)
        t0 = time.time()
        o = {m: np.zeros(len(y)) for m in specs}
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(df, strata):
            for m, pipe in specs.items():
                gs = GridSearchCV(pipe, GRID, scoring="roc_auc", n_jobs=-1,
                                  cv=StratifiedKFold(3, shuffle=True, random_state=seed))
                o[m][te] = gs.fit(df.iloc[tr], y[tr]).predict_proba(df.iloc[te])[:, 1]
        oo = pd.DataFrame(o | {"y_true": y}); oo.to_csv(path, index=False); oofs.append(oo)
        print(f"  seed {seed} ({time.time()-t0:.0f}s): " + " | ".join(
            f"{m} {roc_auc_score(y, o[m]):.4f}" for m in specs), flush=True)

    mk1 = [pd.read_csv(os.path.join(a.mark1, f"oof_seed{s}.csv")) for s in a.seeds]
    if not all(np.array_equal(m.y_true.values, y) for m in mk1):
        raise SystemExit("Mark 1 predictions are not row-aligned")
    P = {m: np.mean([o[m].values for o in oofs], 0) for m in specs}
    P |= {r: np.mean([m[r].values for m in mk1], 0) for r in REF.values()}

    print(f"\n=== AUC (mean OOF prediction over {len(a.seeds)} repeats), 95% bootstrap CI ===")
    for m in [M_13W, REF[M_13W], M_SIXW, REF[M_SIXW]]:
        v, lo, hi = boot_auc(y, P[m], strata, a.boot)
        print(f"  {m:<38} {v:.4f} [{lo:.4f}, {hi:.4f}]")
    rows = []

    def report(tag, m1, m2, idx=slice(None), primary=False):
        dd, lo, hi = boot_diff(y[idx], P[m1][idx], P[m2][idx], strata[idx], a.boot)
        _, dlo, dhi, dp = delong(y[idx], P[m1][idx], P[m2][idx])
        v = ("  -> SUPERIOR (CI excludes 0)" if lo > 0 else "  -> not shown") if primary else ""
        print(f"  {tag:<38} Δ {dd:+.4f}  bootstrap [{lo:+.4f}, {hi:+.4f}]  "
              f"DeLong [{dlo:+.4f}, {dhi:+.4f}] p={dp:.3f}{v}")
        rows.append({"comparison": tag, "delta": dd, "boot_lo": lo, "boot_hi": hi,
                     "delong_p": dp})

    print("\n=== PRIMARY: woman-word graph added to narrative + PDI-13 ===")
    report("13 items: + word graph vs Mark 1", M_13W, REF[M_13W], primary=True)
    print("\n=== SECONDARY ===")
    report("six items: + word graph vs Mark 1", M_SIXW, REF[M_SIXW])
    for c in sorted(set(coh)):
        report(f"cohort {c}, 13 items: + word graph", M_13W, REF[M_13W], coh == c)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "comparisons.csv"), index=False)
    print(f"\nsaved to {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="perignnosis_graph_pooled.pt")
    ap.add_argument("--cohort-csv", default="pooled_cohort.csv")
    ap.add_argument("--mark1", default="results_narrative_short_pdi")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 11)))
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default="results_word_graph_confirmatory")
    main(ap.parse_args())
