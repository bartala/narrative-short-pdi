"""
Graph neural network on PCA-compressed narrative and entity embeddings.

Usage:
    python pca_gnn.py --graph perignnosis_graph_pooled.pt --seeds 1 2 3 4 5 --out results_pca_gnn/
"""

import argparse, itertools, json, os, time
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from fair_benchmark import EMB, PDIN, sanitize, set_seed
from compact_gnn import (to_homogeneous, CompactGNN, mask_test_edges, perm_test,
                         GRID, TUNE_EPOCHS, FINAL_EPOCHS, WD)
from pca_narrative_diagnostic import (pipe_pca, pipe_cols, fit_predict, C_VALUES,
                                      K_GRID, PDI_COLS)

K_GNN = [10, 20, 50]
M_GNN = "PeriGNNosis-PCA"
M_LRPCA = "LR (narrative PCA + PDI)"
M_LR13 = "LR (PDI-13)"
MODELS = [M_GNN, M_LRPCA, M_LR13]


def pca_pack(pack, Zw, Ze, ent_rows, k):
    X, T, ei, et, widx, nt, nr, w = pack
    Xk = X.clone()
    Xk[:, :EMB] = 0.0
    Xk[widx, :k] = Zw[:, :k]
    Xk[ent_rows, :k] = Ze[:, :k]
    return (Xk, T, ei, et, widx, nt, nr, w)


def train_eval(pack, fit_idx, eval_idx, block_idx, y_t, device, cfg, epochs, conv):
    X, T, ei, et, widx, nt, nr, w = pack
    ei2, et2 = mask_test_edges(ei, et, widx, block_idx)
    X, T = X.to(device), T.to(device)
    ei2, et2, widx = ei2.to(device), et2.to(device), widx.to(device)
    m = CompactGNN(nt, nr, cfg["hidden"], cfg["dropout"], True, w, conv).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=cfg["lr"], weight_decay=WD)
    fi = torch.as_tensor(fit_idx, dtype=torch.long, device=device)
    ev = torch.as_tensor(eval_idx, dtype=torch.long, device=device)
    for _ in range(epochs):
        m.train(); opt.zero_grad()
        F.cross_entropy(m(X, T, ei2, et2, widx)[fi], y_t[fi]).backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        p = F.softmax(m(X, T, ei2, et2, widx)[ev], 1)[:, 1]
    return p.cpu().numpy()


def main(a):
    os.makedirs(a.out, exist_ok=True)
    cache = os.path.join(a.out, "fold_cache")
    os.makedirs(cache, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw = torch.load(a.graph, map_location="cpu", weights_only=False)
    data, woman = sanitize(raw)
    y = data[woman].y.view(-1).numpy().astype(int)
    Xw = data[woman].x.numpy()[:, :EMB + PDIN].astype(np.float64)
    pack = to_homogeneous(data, woman)
    X, widx = pack[0], pack[4]
    is_w = torch.zeros(X.size(0), dtype=torch.bool)
    is_w[widx] = True
    ent_rows = torch.where(~is_w)[0]
    y_t = torch.as_tensor(y, dtype=torch.long, device=device)

    kmax = max(K_GNN)
    n_ent_comp = min(kmax, len(ent_rows) - 1)
    ent_pca = PCA(n_ent_comp, random_state=0).fit(X[ent_rows, :EMB].numpy())
    Ze = torch.tensor(ent_pca.transform(X[ent_rows, :EMB].numpy()), dtype=torch.float)
    if n_ent_comp < kmax:
        Ze = torch.cat([Ze, torch.zeros(Ze.size(0), kmax - n_ent_comp)], 1)
    print(f"{len(y)} women, {int(y.sum())} positive; {len(ent_rows)} entity nodes; "
          f"k grid {K_GNN}; entity PCA explains "
          f"{ent_pca.explained_variance_ratio_[:20].sum():.1%} at k=20; "
          f"conv={a.conv}; device={device}", flush=True)

    keys, vals = zip(*GRID.items())
    configs = [dict(zip(keys, c)) for c in itertools.product(*vals)]

    rows, picks = [], []
    for seed in a.seeds:
        set_seed(seed)
        t0 = time.time()
        oof = {m: np.zeros(len(y)) for m in MODELS}
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(skf.split(Xw, y)):
            cpath = os.path.join(cache, f"seed{seed}_fold{fold}.json")
            if os.path.exists(cpath):
                c = json.load(open(cpath))
                for m in MODELS:
                    oof[m][te] = np.asarray(c["pred"][m])
                picks.append(c["pick"])
                print(f"  seed {seed} fold {fold+1}/5: cached (k={c['pick']['k']})",
                      flush=True)
                continue
            set_seed(seed + fold)

            wp = PCA(kmax, random_state=0).fit(Xw[tr, :EMB])
            Zw = torch.tensor(wp.transform(Xw[:, :EMB]), dtype=torch.float)
            packs = {k: pca_pack(pack, Zw, Ze, ent_rows, k) for k in K_GNN}

            sub_tr, sub_val = train_test_split(tr, test_size=0.2, stratify=y[tr],
                                               random_state=seed)
            block = np.concatenate([sub_val, te])
            best = (-1.0, None, None)
            for k in K_GNN:
                for cfg in configs:
                    p = train_eval(packs[k], sub_tr, sub_val, block, y_t, device,
                                   cfg, TUNE_EPOCHS, a.conv)
                    auc = roc_auc_score(y[sub_val], p)
                    if auc > best[0]:
                        best = (auc, k, cfg)
            _, k_best, cfg_best = best
            pred = {M_GNN: train_eval(packs[k_best], tr, te, te, y_t, device,
                                      cfg_best, FINAL_EPOCHS, a.conv)}
            pred[M_LRPCA], _ = fit_predict(
                pipe_pca(True), {"clf__C": C_VALUES, "ct__pca__n_components": K_GRID},
                Xw, y, tr, te, seed)
            pred[M_LR13], _ = fit_predict(pipe_cols(PDI_COLS), {"clf__C": C_VALUES},
                                          Xw, y, tr, te, seed)
            for m in MODELS:
                oof[m][te] = pred[m]
            pick = {"seed": seed, "fold": fold, "k": k_best, **cfg_best,
                    "val_auc": best[0]}
            picks.append(pick)
            json.dump({"pred": {m: pred[m].tolist() for m in MODELS}, "pick": pick},
                      open(cpath, "w"))
            print(f"  seed {seed} fold {fold+1}/5 ({time.time()-t0:.0f}s): k={k_best} "
                  f"{cfg_best}", flush=True)

        for m in MODELS:
            rows.append({"seed": seed, "model": m, "auc": roc_auc_score(y, oof[m])})
        pd.DataFrame(oof | {"y_true": y}).to_csv(
            os.path.join(a.out, f"oof_seed{seed}.csv"), index=False)
        print("  seed %d: " % seed + " | ".join(
            f"{m} {roc_auc_score(y, oof[m]):.4f}" for m in MODELS), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "pca_gnn_auc.csv"), index=False)
    pk = pd.DataFrame(picks)
    pk.to_csv(os.path.join(a.out, "pca_gnn_choices.csv"), index=False)

    print("\n=== pooled OOF AUC ===")
    print(df.groupby("model")["auc"].agg(["mean", "std"]).round(4)
            .sort_values("mean", ascending=False).to_string())
    print(f"\ncomponents chosen for PeriGNNosis-PCA: "
          f"{dict(pk.k.value_counts().sort_index())}")

    print("\n=== paired permutation tests on OOF probabilities (10,000 draws) ===")
    for ref, tag in [(M_LRPCA, "PRIMARY"), (M_LR13, "secondary")]:
        ds, ps = [], []
        for seed in a.seeds:
            o = pd.read_csv(os.path.join(a.out, f"oof_seed{seed}.csv"))
            d, pv = perm_test(o["y_true"].values, o[M_GNN].values, o[ref].values,
                              n=10000, seed=seed)
            ds.append(d); ps.append(pv)
        print(f"  [{tag}] {M_GNN} - {ref}: Δ {np.mean(ds):+.4f}  "
              f"perm p {min(ps):.4f}-{max(ps):.4f}  wins {sum(d > 0 for d in ds)}/{len(ds)}")
    print("\nThe PRIMARY line is the one that shows whether the graph adds anything")
    print("beyond the compressed narrative. The secondary line alone cannot.")
    print(f"\nsaved to {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="perignnosis_graph_pooled.pt")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--out", default="results_pca_gnn")
    ap.add_argument("--conv", choices=["gatv2", "gcn"], default="gatv2")
    main(ap.parse_args())
