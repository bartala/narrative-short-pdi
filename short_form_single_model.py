"""
One logistic model trained with random item masking for any PDI short form.

Usage:
    python short_form_single_model.py --mark1 results_narrative_short_pdi/ --out results_short_form_single/
"""

import argparse, os, time
import numpy as np, pandas as pd, torch
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split

from fair_benchmark import EMB, PDIN, sanitize, set_seed
from pca_narrative_diagnostic import C_VALUES, K_GRID, EMB_COLS, PDI_COLS, pipe_pca
from narrative_short_pdi import SIX, boot_diff, boot_auc, delong
import item_gnn as ig

MARGIN = 0.01
N_RAND = 20
SIX_IDX = [q - 1 for q in SIX]
M_SINGLE = "Single model (masked training)"
M_IMP = "Single LR, mean-imputed"
DEDICATED = {"13 items": "LR (narrative + PDI-13)", "six items": "LR (narrative + six)",
             "narrative only": "LR (narrative only)"}
MASKS = {"13 items": list(range(PDIN)), "six items": SIX_IDX, "narrative only": []}


def train_single(X, y, tr, strata, seed, device):
    pca = PCA(ig.N_PCA, random_state=0).fit(X[tr][:, EMB_COLS])
    Zn = pca.transform(X[:, EMB_COLS]); Zn = (Zn - Zn[tr].mean(0)) / Zn[tr].std(0)
    It = X[:, PDI_COLS]; It = (It - It[tr].mean(0)) / It[tr].std(0).clip(1e-9)
    nar = torch.as_tensor(Zn, dtype=torch.float, device=device)
    items = torch.as_tensor(It, dtype=torch.float, device=device)
    fit, val = train_test_split(tr, test_size=0.2, stratify=strata[tr], random_state=seed)
    models = [ig.train(nar, items, y, fit, val, False, seed * 100 + e, device)
              for e in range(ig.N_ENS)]

    def predict(idx, mask):
        t = torch.as_tensor(idx, device=device)
        mk = torch.as_tensor(np.broadcast_to(np.asarray(mask, float), (len(idx), PDIN)).copy(),
                             dtype=torch.float, device=device)
        with torch.no_grad():
            return np.mean([m(nar[t], items[t], mk).cpu().numpy() for m in models], 0)
    return predict


def main(a):
    os.makedirs(a.out, exist_ok=True)
    d, w = sanitize(torch.load(a.graph, map_location="cpu", weights_only=False))
    X = d[w].x.numpy()[:, :EMB + PDIN].astype(np.float64)
    y = d[w].y.view(-1).numpy().astype(int)
    meta = pd.read_csv(a.cohort_csv)
    if not np.allclose(meta[[f"pdi_q{i}" for i in range(1, 14)]].values, X[:, EMB:]):
        raise SystemExit("cohort csv is not row-aligned with the graph")
    coh = meta.cohort.values.astype(int)
    strata = y * 2 + coh
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{len(y)} women, {y.sum()} positive; {len(a.seeds)} repeats; ensemble "
          f"{ig.N_ENS}; {N_RAND} random subsets per size; device={device}", flush=True)

    oofs, rnd_rows = [], []
    for seed in a.seeds:
        path = os.path.join(a.out, f"oof_seed{seed}.csv")
        rpath = os.path.join(a.out, f"random_seed{seed}.csv")
        if os.path.exists(path) and os.path.exists(rpath):
            oofs.append(pd.read_csv(path)); rnd_rows.append(pd.read_csv(rpath))
            print(f"  seed {seed}: cached"); continue
        set_seed(seed)
        t0 = time.time()
        rng = np.random.RandomState(1000 + seed)
        o = {f"{m}|{s}": np.zeros(len(y)) for m in [M_SINGLE, M_IMP] for s in MASKS}
        rr = []
        for fold, (tr, te) in enumerate(
                StratifiedKFold(5, shuffle=True, random_state=seed).split(X, strata)):
            single = train_single(X, y, tr, strata, seed * 10 + fold, device)
            gs = GridSearchCV(pipe_pca(True), {"clf__C": C_VALUES,
                                               "ct__pca__n_components": K_GRID},
                              scoring="roc_auc", n_jobs=-1,
                              cv=StratifiedKFold(3, shuffle=True, random_state=seed)).fit(X[tr], y[tr])
            mu = X[tr][:, PDI_COLS].mean(0)

            def imputed(mask):
                Xt = X[te].copy()
                Xt[:, PDI_COLS] = np.where(np.broadcast_to(mask, (len(te), PDIN)) > 0,
                                           Xt[:, PDI_COLS], mu)
                return gs.best_estimator_.predict_proba(Xt)[:, 1]

            for s, items in MASKS.items():
                mask = np.zeros(PDIN); mask[items] = 1
                o[f"{M_SINGLE}|{s}"][te] = single(te, mask)
                o[f"{M_IMP}|{s}"][te] = imputed(mask)
            for kk in range(1, PDIN):
                for _ in range(N_RAND):
                    mask = np.zeros(PDIN); mask[rng.choice(PDIN, kk, replace=False)] = 1
                    rr.append({"seed": seed, "fold": fold, "k": kk,
                               "single": roc_auc_score(y[te], single(te, mask)),
                               "imputed": roc_auc_score(y[te], imputed(mask))})
        oo = pd.DataFrame(o | {"y_true": y}); oo.to_csv(path, index=False); oofs.append(oo)
        rdf = pd.DataFrame(rr); rdf.to_csv(rpath, index=False); rnd_rows.append(rdf)
        print(f"  seed {seed} ({time.time()-t0:.0f}s): " + " | ".join(
            f"{s}: single {roc_auc_score(y, oo[f'{M_SINGLE}|{s}']):.4f} / imputed "
            f"{roc_auc_score(y, oo[f'{M_IMP}|{s}']):.4f}" for s in MASKS), flush=True)

    mk1 = [pd.read_csv(os.path.join(a.mark1, f"oof_seed{s}.csv")) for s in a.seeds]
    if not all(np.array_equal(m.y_true.values, y) for m in mk1):
        raise SystemExit("Mark 1 predictions are not row-aligned")
    P = {c: np.mean([o[c].values for o in oofs], 0) for c in oofs[0].columns if c != "y_true"}
    for c in DEDICATED.values():
        P[c] = np.mean([m[c].values for m in mk1], 0)

    print(f"\n=== AUC (mean OOF prediction over {len(a.seeds)} repeats), 95% bootstrap CI ===")
    for s in MASKS:
        for c in [f"{M_SINGLE}|{s}", DEDICATED[s], f"{M_IMP}|{s}"]:
            v, lo, hi = boot_auc(y, P[c], strata, a.boot)
            print(f"  {s:<15} {c.split('|')[0]:<28} {v:.4f} [{lo:.4f}, {hi:.4f}]")

    out = []
    print(f"\n=== H1-H3: single model vs the dedicated Mark 1 model, margin {MARGIN} ===")
    for h, s in zip(["H1", "H2", "H3"], ["six items", "13 items", "narrative only"]):
        for ref, tag in [(DEDICATED[s], "dedicated"), (f"{M_IMP}|{s}", "mean-imputed")]:
            dd, lo, hi = boot_diff(y, P[f"{M_SINGLE}|{s}"], P[ref], strata, a.boot)
            _, dlo, dhi, dp = delong(y, P[f"{M_SINGLE}|{s}"], P[ref])
            ni = ("  -> NON-INFERIOR" if lo > -MARGIN else "  -> not shown") if tag == "dedicated" else ""
            print(f"  {h if tag == 'dedicated' else '  '} {s:<15} vs {tag:<13} Δ {dd:+.4f}  "
                  f"bootstrap [{lo:+.4f}, {hi:+.4f}]  DeLong p={dp:.3f}{ni}")
            out.append({"hypothesis": h, "items": s, "reference": ref, "delta": dd,
                        "boot_lo": lo, "boot_hi": hi, "delong_p": dp})
    pd.DataFrame(out).to_csv(os.path.join(a.out, "comparisons.csv"), index=False)

    R = pd.concat(rnd_rows)
    R.to_csv(os.path.join(a.out, "random_subsets.csv"), index=False)
    print("\n=== secondary: random item subsets (AUC per fold, mean over seeds x folds x draws) ===")
    print(f"  {'items':>5}  {'single':>7}  {'imputed':>7}  {'Δ':>7}  single better in")
    for kk, g in R.groupby("k"):
        dlt = g.single - g.imputed
        print(f"  {kk:>5}  {g.single.mean():.4f}  {g.imputed.mean():.4f}  {dlt.mean():+.4f}"
              f"  {(dlt > 0).mean():.0%} of {len(g)} fold x subset draws")
    print(f"\nsaved to {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="perignnosis_graph_pooled.pt")
    ap.add_argument("--cohort-csv", default="pooled_cohort.csv")
    ap.add_argument("--mark1", default="results_narrative_short_pdi")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 11)))
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default="results_short_form_single")
    main(ap.parse_args())
