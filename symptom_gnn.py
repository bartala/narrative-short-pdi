"""
Graph model over PCL-5 symptom nodes.

Usage:
    python symptom_gnn.py --seeds 1 2 3 4 5 6 7 8 9 10 --out results_symptom_gnn/
"""

import argparse, os, time
import numpy as np, pandas as pd, torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split, ParameterGrid
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from fair_benchmark import EMB, PDIN, sanitize, set_seed
from pca_narrative_diagnostic import fit_predict, pipe_pca, C_VALUES, K_GRID, EMB_COLS, PDI_COLS
from narrative_short_pdi import boot_diff, boot_auc, delong

N_SYM, CUTOFF = 20, 32
N_PCA, HID, N_ENS, MAX_EP, PATIENCE, LR_RATE, WD, DROPOUT = 20, 32, 5, 600, 50, 3e-3, 1e-2, 0.2
TOP_EDGES = 4
M_LR, M_RIDGE, M_PLS = "LR (narrative + PDI-13)", "Ridge on PCL-5 total", "PLS on 20 items"
M_SN, M_SN0 = "PeriGNNosis-SN", "PeriGNNosis-SN without symptom graph"
MODELS = [M_LR, M_RIDGE, M_PLS, M_SN0, M_SN]


def load_items(meta):
    cb = pd.read_csv(os.path.expanduser("~/cbptsd_data/CBEX.csv"), low_memory=False)
    cv = pd.read_csv("covid_msb.csv", low_memory=False)
    cols = [f"pclchildbirth_q{i}" for i in range(1, N_SYM + 1)]
    parts = []
    for d, pre in [(cb, "cbex_"), (cv, "covid_")]:
        t = d[["record_id"] + cols].copy()
        t["record_id"] = pre + t.record_id.astype(str)
        parts.append(t)
    items = meta[["record_id"]].merge(pd.concat(parts), on="record_id", how="left")
    Y = items[cols].values.astype(float)
    tot = np.nansum(Y, 1)
    if not np.array_equal(tot, meta.spcl5_total.values):
        raise SystemExit("item scores do not reproduce the PCL-5 total")
    return Y


def symptom_graph(Y, tr):
    Z = Y[tr]
    Z = np.where(np.isnan(Z), np.nanmean(Z, 0), Z)
    C = np.corrcoef(Z.T)
    P = np.linalg.inv(C + 0.1 * np.eye(N_SYM))
    pc = -P / np.sqrt(np.outer(np.diag(P), np.diag(P)))
    np.fill_diagonal(pc, 0)
    W = np.abs(pc)
    A = np.zeros_like(W)
    for j in range(N_SYM):
        k = np.argsort(-W[j])[:TOP_EDGES]
        A[j, k] = W[j, k]
    A = np.maximum(A, A.T) + np.eye(N_SYM)
    d = A.sum(1)
    return A / np.sqrt(np.outer(d, d))


class SymptomGNN(nn.Module):
    def __init__(self, d_in, A, graph=True):
        super().__init__()
        self.graph = graph
        self.register_buffer("A", torch.as_tensor(A, dtype=torch.float))
        self.lin = nn.Linear(d_in, N_SYM)
        self.enc = nn.Sequential(nn.Linear(d_in, HID), nn.GELU(), nn.Dropout(DROPOUT),
                                 nn.Linear(HID, HID))
        self.sym = nn.Parameter(torch.randn(N_SYM, HID) * 0.1)
        if graph:
            self.g1 = nn.Linear(HID, HID); self.g2 = nn.Linear(HID, HID)
            self.gamma = nn.Parameter(torch.zeros(1))
        self.cal = nn.Parameter(torch.tensor([0.3, 0.0]))

    def forward(self, x):
        S = self.sym
        if self.graph:
            S = S + self.g2(self.A @ F.gelu(self.g1(self.A @ S)))
        z = self.lin(x) + self.enc(x) @ S.T
        if self.graph:
            z = z + self.gamma * (z @ self.A - z)
        yhat = 4 * torch.sigmoid(z)
        total = yhat.sum(1)
        return yhat, total, F.softplus(self.cal[0]) * (total - CUTOFF) + self.cal[1]


def train_sn(x, Y, y, fit, val, A, graph, seed, device):
    torch.manual_seed(seed)
    m = SymptomGNN(x.size(1), A, graph).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=LR_RATE, weight_decay=WD)
    Yt = torch.as_tensor(np.nan_to_num(Y), dtype=torch.float, device=device)
    M = torch.as_tensor(~np.isnan(Y), dtype=torch.float, device=device)
    yt = torch.as_tensor(y, dtype=torch.float, device=device)
    fi = torch.as_tensor(fit, device=device); vi = torch.as_tensor(val, device=device)
    best, state, wait = -1, None, 0
    for ep in range(MAX_EP):
        m.train(); opt.zero_grad()
        yhat, _, logit = m(x[fi])
        mse = ((yhat - Yt[fi]) ** 2 * M[fi]).sum() / M[fi].sum()
        bce = F.binary_cross_entropy_with_logits(logit, yt[fi])
        (mse + 0.5 * bce).backward()
        opt.step()
        m.eval()
        with torch.no_grad():
            _, tot, _ = m(x[vi])
        auc = roc_auc_score(y[val], tot.cpu().numpy())
        if auc > best + 1e-5:
            best, wait, state = auc, 0, {k: v.clone() for k, v in m.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    m.load_state_dict(state); m.eval()
    return m


def linear_reg(X, target, y, tr, te, seed, kind):
    if kind == "ridge":
        est, grid = Ridge(), {"r__alpha": [0.1, 1, 10, 100, 1000]}
    else:
        est, grid = PLSRegression(scale=False), {"r__n_components": [2, 5, 10]}
    pipe = Pipeline([("ct", ColumnTransformer([("pca", PCA(10, random_state=0), EMB_COLS),
                                               ("pdi", "passthrough", PDI_COLS)])),
                     ("sc", StandardScaler()), ("r", est)])
    grid = grid | {"ct__pca__n_components": K_GRID}
    inner = list(StratifiedKFold(3, shuffle=True, random_state=seed).split(tr, y[tr]))
    score = lambda P: P.sum(1) if P.ndim == 2 else P
    best = (-1, None)
    for prm in ParameterGrid(grid):
        sc = []
        for a, b in inner:
            mm = clone(pipe).set_params(**prm).fit(X[tr[a]], target[tr[a]])
            sc.append(roc_auc_score(y[tr[b]], score(mm.predict(X[tr[b]]))))
        if np.mean(sc) > best[0]:
            best = (np.mean(sc), prm)
    mm = clone(pipe).set_params(**best[1]).fit(X[tr], target[tr])
    return score(mm.predict(X[te]))


def main(a):
    os.makedirs(a.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d, w = sanitize(torch.load(a.graph, map_location="cpu", weights_only=False))
    X = d[w].x.numpy()[:, :EMB + PDIN].astype(np.float64)
    y = d[w].y.view(-1).numpy().astype(int)
    meta = pd.read_csv(a.cohort_csv)
    if not np.allclose(meta[[f"pdi_q{i}" for i in range(1, 14)]].values, X[:, EMB:]):
        raise SystemExit("cohort csv is not row-aligned with the graph")
    Y = load_items(meta)
    Yf = np.where(np.isnan(Y), 0, Y)
    total = meta.spcl5_total.values.astype(float)
    coh = meta.cohort.values.astype(int)
    strata = y * 2 + coh
    print(f"{len(y)} women, {y.sum()} positive; 20 PCL-5 items ({int(np.isnan(Y).sum())} "
          f"missing values); {len(a.seeds)} seeds; device={device}", flush=True)

    oofs = []
    for seed in a.seeds:
        path = os.path.join(a.out, f"oof_seed{seed}.csv")
        if os.path.exists(path):
            oofs.append(pd.read_csv(path)); continue
        set_seed(seed)
        t0 = time.time()
        o = {m: np.zeros(len(y)) for m in MODELS}
        for fold, (tr, te) in enumerate(
                StratifiedKFold(5, shuffle=True, random_state=seed).split(X, strata)):
            o[M_LR][te], _ = fit_predict(pipe_pca(True), {"clf__C": C_VALUES,
                                                          "ct__pca__n_components": K_GRID},
                                         X, y, tr, te, seed)
            o[M_RIDGE][te] = linear_reg(X, total, y, tr, te, seed, "ridge")
            o[M_PLS][te] = linear_reg(X, Yf, y, tr, te, seed, "pls")
            pca = PCA(N_PCA, random_state=0).fit(X[tr][:, EMB_COLS])
            Zn = pca.transform(X[:, EMB_COLS])
            xin = np.c_[Zn, X[:, PDI_COLS]]
            xin = (xin - xin[tr].mean(0)) / xin[tr].std(0).clip(1e-9)
            xt = torch.as_tensor(xin, dtype=torch.float, device=device)
            A = symptom_graph(Y, tr)
            fit, val = train_test_split(tr, test_size=0.2, stratify=strata[tr],
                                        random_state=seed * 10 + fold)
            tt = torch.as_tensor(te, device=device)
            for name, graph in [(M_SN, True), (M_SN0, False)]:
                preds = []
                for e in range(N_ENS):
                    mm = train_sn(xt, Y, y, fit, val, A, graph, seed * 100 + fold * 10 + e, device)
                    with torch.no_grad():
                        preds.append(mm(xt[tt])[1].cpu().numpy())
                o[name][te] = np.mean(preds, 0)
        oo = pd.DataFrame(o | {"y_true": y}); oo.to_csv(path, index=False); oofs.append(oo)
        print(f"  seed {seed} ({time.time()-t0:.0f}s): " + " | ".join(
            f"{m} {roc_auc_score(y, o[m]):.4f}" for m in MODELS), flush=True)

    P = {m: np.mean([o[m].values for o in oofs], 0) for m in MODELS}
    print(f"\n=== AUC (mean OOF prediction over {len(a.seeds)} seeds), 95% bootstrap CI ===")
    for m in MODELS:
        v, lo, hi = boot_auc(y, P[m], strata, a.boot)
        sd = np.std([roc_auc_score(y, o[m]) for o in oofs])
        print(f"  {m:<40} {v:.4f} [{lo:.4f}, {hi:.4f}]  (sd over seeds {sd:.4f})")
    rows = []
    print("\n=== comparisons (paired bootstrap, DeLong) ===")
    for tag, ref in [("PRIMARY vs Mark 1 LR", M_LR), ("vs ridge on total (same labels)", M_RIDGE),
                     ("vs PLS on 20 items (same labels)", M_PLS),
                     ("vs no symptom graph (ablation)", M_SN0)]:
        dd, lo, hi = boot_diff(y, P[M_SN], P[ref], strata, a.boot)
        _, dlo, dhi, dp = delong(y, P[M_SN], P[ref])
        v = "  -> SUPERIOR (CI excludes 0)" if lo > 0 else ("  -> worse (CI below 0)" if hi < 0 else "  -> not shown")
        print(f"  {tag:<36} Δ {dd:+.4f}  bootstrap [{lo:+.4f}, {hi:+.4f}]  "
              f"DeLong p={dp:.3f}{v}")
        rows.append({"comparison": tag, "delta": dd, "lo": lo, "hi": hi, "delong_p": dp})
    for tag, m1, m2 in [("ridge on total vs Mark 1 LR", M_RIDGE, M_LR),
                        ("PLS on items vs Mark 1 LR", M_PLS, M_LR)]:
        dd, lo, hi = boot_diff(y, P[m1], P[m2], strata, a.boot)
        print(f"  {tag:<36} Δ {dd:+.4f}  bootstrap [{lo:+.4f}, {hi:+.4f}]")
        rows.append({"comparison": tag, "delta": dd, "lo": lo, "hi": hi})
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "comparisons.csv"), index=False)
    print(f"\nsaved to {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="perignnosis_graph_pooled.pt")
    ap.add_argument("--cohort-csv", default="pooled_cohort.csv")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 11)))
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default="results_symptom_gnn")
    main(ap.parse_args())
