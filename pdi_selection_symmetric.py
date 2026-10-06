"""
PDI item selection in which each model chooses its own number of items.

Usage:
    python pdi_selection_symmetric.py --graph perignnosis_graph_pooled.pt --seeds 1 2 3 4 5 --out results_selection_symmetric_pooled/
"""

import argparse, collections, json, os, time
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from fair_benchmark import EMB, PDIN, C_GRID, sanitize, set_seed
from compact_gnn import (to_homogeneous, CompactGNN, mask_test_edges, run_fold,
                         perm_test, WD)

INNER_CFG = {"hidden": 64, "dropout": 0.3, "lr": 1e-3}
INNER_EPOCHS = 60

M_GNN_OWN = "PeriGNNosis (own selection)"
M_LR_OWN = "LR (own selection)"
M_LR_GSEL = "LR (PeriGNNosis's selection)"
M_LR_13 = "LR (PDI-13)"
MODELS = [M_GNN_OWN, M_LR_OWN, M_LR_GSEL, M_LR_13]


def lr_fit_predict(P, y, fit_idx, pred_idx, cols, seed):
    pipe = Pipeline([("sc", StandardScaler()),
                     ("clf", LogisticRegression(class_weight="balanced",
                                                solver="liblinear", max_iter=2000))])
    gs = GridSearchCV(pipe, C_GRID, scoring="roc_auc",
                      cv=StratifiedKFold(3, shuffle=True, random_state=seed), n_jobs=-1)
    gs.fit(P[fit_idx][:, cols], y[fit_idx])
    return gs.best_estimator_.predict_proba(P[pred_idx][:, cols])[:, 1]


def rank_items(P, y):
    s = []
    for j in range(P.shape[1]):
        v = P[:, j]
        if len(np.unique(v)) < 2:
            s.append(0.5)
            continue
        a = roc_auc_score(y, v)
        s.append(max(a, 1 - a))
    return [int(i) for i in np.argsort(s)[::-1]]


def restrict(pack, keep):
    X, T, ei, et, widx, nt, nr, w = pack
    drop = [EMB + i for i in range(PDIN) if i not in set(keep)]
    if not drop:
        return pack
    Xk = X.clone()
    Xk[widx.unsqueeze(1), torch.tensor(drop).unsqueeze(0)] = 0.0
    return (Xk, T, ei, et, widx, nt, nr, w)


def gnn_fixed(pack, fit_idx, pred_idx, block_idx, y_t, device, conv):
    X, T, ei, et, widx, nt, nr, w = pack
    ei2, et2 = mask_test_edges(ei, et, widx, block_idx)
    X, T = X.to(device), T.to(device)
    ei2, et2, widx = ei2.to(device), et2.to(device), widx.to(device)
    m = CompactGNN(nt, nr, INNER_CFG["hidden"], INNER_CFG["dropout"], True, w,
                   conv).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=INNER_CFG["lr"], weight_decay=WD)
    fit_t = torch.as_tensor(fit_idx, dtype=torch.long, device=device)
    pred_t = torch.as_tensor(pred_idx, dtype=torch.long, device=device)
    for _ in range(INNER_EPOCHS):
        m.train(); opt.zero_grad()
        F.cross_entropy(m(X, T, ei2, et2, widx)[fit_t], y_t[fit_t]).backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        p = F.softmax(m(X, T, ei2, et2, widx)[pred_t], 1)[:, 1]
    return p.cpu().numpy()


def choose_k(selector, P, y, tr, outer_te, pack, y_t, device, seed, kgrid, conv):
    inner = StratifiedKFold(3, shuffle=True, random_state=seed)
    score = {k: 0.0 for k in kgrid}
    for itr_l, ival_l in inner.split(P[tr], y[tr]):
        itr, ival = tr[itr_l], tr[ival_l]
        order = rank_items(P[itr], y[itr])
        for k in kgrid:
            cols = sorted(order[:k])
            if selector == "lr":
                p = lr_fit_predict(P, y, itr, ival, cols, seed)
            else:
                p = gnn_fixed(restrict(pack, cols), itr, ival,
                              np.concatenate([ival, outer_te]), y_t, device, conv)
            score[k] += roc_auc_score(y[ival], p) / inner.get_n_splits()
    best = max(kgrid, key=lambda k: (score[k], -k))
    return best, score


def main(a):
    os.makedirs(a.out, exist_ok=True)
    cache_dir = os.path.join(a.out, "fold_cache")
    os.makedirs(cache_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw = torch.load(a.graph, map_location="cpu", weights_only=False)
    data, woman = sanitize(raw)
    y = data[woman].y.view(-1).numpy().astype(int)
    P = data[woman].x.numpy()[:, EMB:EMB + PDIN]
    pack = to_homogeneous(data, woman)
    y_t = torch.as_tensor(y, dtype=torch.long, device=device)
    kgrid = sorted({int(k) for k in a.k_grid})
    print(f"{len(y)} women, {int(y.sum())} positive; k grid {kgrid}; "
          f"conv={a.conv}; device={device}", flush=True)

    rows, picks = [], []
    for seed in a.seeds:
        set_seed(seed)
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        oof = {m: np.zeros(len(y)) for m in MODELS}
        t0 = time.time()
        for fold, (tr, te) in enumerate(skf.split(P, y)):
            cpath = os.path.join(cache_dir, f"seed{seed}_fold{fold}.json")
            if os.path.exists(cpath):
                c = json.load(open(cpath))
                for m in MODELS:
                    oof[m][te] = np.asarray(c["pred"][m])
                picks.append(c["pick"])
                print(f"  seed {seed} fold {fold+1}/5: cached "
                      f"(k LR={c['pick']['k_lr']}, k GNN={c['pick']['k_gnn']})",
                      flush=True)
                continue

            set_seed(seed + fold)
            k_lr, s_lr = choose_k("lr", P, y, tr, te, pack, y_t, device, seed,
                                  kgrid, a.conv)
            k_gnn, s_gnn = choose_k("gnn", P, y, tr, te, pack, y_t, device, seed,
                                    kgrid, a.conv)
            order = rank_items(P[tr], y[tr])
            sel_lr, sel_gnn = sorted(order[:k_lr]), sorted(order[:k_gnn])

            pred = {
                M_LR_OWN: lr_fit_predict(P, y, tr, te, sel_lr, seed),
                M_LR_GSEL: lr_fit_predict(P, y, tr, te, sel_gnn, seed),
                M_LR_13: lr_fit_predict(P, y, tr, te, list(range(PDIN)), seed),
            }
            pred[M_GNN_OWN], _ = run_fold(restrict(pack, sel_gnn), tr, te, y_t,
                                          device, seed, True, a.conv)
            for m in MODELS:
                oof[m][te] = pred[m]

            pick = {"seed": seed, "fold": fold, "k_lr": k_lr, "k_gnn": k_gnn,
                    "items_lr": [i + 1 for i in sel_lr],
                    "items_gnn": [i + 1 for i in sel_gnn],
                    "inner_auc_lr": {str(k): v for k, v in s_lr.items()},
                    "inner_auc_gnn": {str(k): v for k, v in s_gnn.items()}}
            picks.append(pick)
            json.dump({"pred": {m: pred[m].tolist() for m in MODELS}, "pick": pick},
                      open(cpath, "w"))
            print(f"  seed {seed} fold {fold+1}/5 ({time.time()-t0:.0f}s): "
                  f"k LR={k_lr} {pick['items_lr']} | k GNN={k_gnn} "
                  f"{pick['items_gnn']}", flush=True)

        for m in MODELS:
            rows.append({"seed": seed, "model": m, "auc": roc_auc_score(y, oof[m])})
        pd.DataFrame(oof | {"y_true": y}).to_csv(
            os.path.join(a.out, f"oof_seed{seed}.csv"), index=False)
        print("  seed %d: " % seed + " | ".join(
            f"{m} {roc_auc_score(y, oof[m]):.4f}" for m in MODELS), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "selection_symmetric_auc.csv"), index=False)
    pd.DataFrame(picks).to_csv(os.path.join(a.out, "selections.csv"), index=False)

    print("\n=== pooled OOF AUC ===")
    print(df.groupby("model")["auc"].agg(["mean", "std"]).round(4)
            .sort_values("mean", ascending=False).to_string())

    print("\n=== chosen k per selector, across outer folds ===")
    for key, name in [("k_lr", "LR"), ("k_gnn", "PeriGNNosis")]:
        print(f"  {name:12s} {dict(sorted(collections.Counter(p[key] for p in picks).items()))}")

    print("\n=== item selection frequency per selector ===")
    n = len(picks)
    for key, name in [("items_lr", "LR"), ("items_gnn", "PeriGNNosis")]:
        f = collections.Counter(i for p in picks for i in p[key])
        robust = [i for i in range(1, PDIN + 1) if f.get(i, 0) >= 0.8 * n]
        print(f"  {name:12s} " + " ".join(f"Q{i}:{f.get(i, 0)}" for i in range(1, PDIN + 1))
              + f"   robust(>=80%): {robust}")

    print("\n=== paired permutation tests on OOF probabilities (10,000 draws) ===")
    for other in [M_LR_OWN, M_LR_GSEL, M_LR_13]:
        ds, ps = [], []
        for seed in a.seeds:
            o = pd.read_csv(os.path.join(a.out, f"oof_seed{seed}.csv"))
            d, pv = perm_test(o["y_true"].values, o[M_GNN_OWN].values,
                              o[other].values, n=10000, seed=seed)
            ds.append(d); ps.append(pv)
        print(f"  {M_GNN_OWN} vs {other}: Δ {np.mean(ds):+.4f}  "
              f"perm p {min(ps):.4f}-{max(ps):.4f}  wins {sum(d > 0 for d in ds)}/{len(ds)}")
    print(f"\nsaved to {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="perignnosis_graph_pooled.pt")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--out", default="results_selection_symmetric")
    ap.add_argument("--k-grid", type=int, nargs="+", default=list(range(1, 14)))
    ap.add_argument("--conv", choices=["gatv2", "gcn"], default="gatv2")
    main(ap.parse_args())
