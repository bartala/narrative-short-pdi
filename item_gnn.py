"""
One model for any subset of PDI items, with item nodes.

Usage:
    python item_gnn.py --seeds 1 2 3 4 5 --out results_item_gnn/
"""

import argparse, os, time
import numpy as np, pandas as pd, torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from fair_benchmark import EMB, PDIN, sanitize, set_seed
from pca_narrative_diagnostic import fit_predict, perm_test, C_VALUES, K_GRID, EMB_COLS, PDI_COLS
from graph_signal_diagnostic import pipe
from pdi_replacement import rank_items

K_ITEMS = [0, 1, 2, 3, 4, 6, 8, 10, 13]
N_PCA, N_ENS, MAX_EP, PATIENCE, N_RAND = 20, 5, 400, 40, 20
CFG = {"hid": 32, "dropout": 0.2, "wd_graph": 1e-2, "graph_l2": 0.0}
M_OR, M_MI, M_LIN, M_G = ("LR per subset (oracle)", "LR single, mean-imputed",
                          "Linear single, masked", "PeriGNNosis-Items")
SINGLE = [M_MI, M_LIN, M_G]


class ItemGNN(nn.Module):
    def __init__(self, d_nar, n_items, hid, graph=True, dropout=0.2):
        super().__init__()
        self.graph = graph
        self.lin_nar = nn.Linear(d_nar, 1)
        self.lin_item = nn.Parameter(torch.zeros(n_items))
        if graph:
            self.item_emb = nn.Parameter(torch.randn(n_items, hid) * 0.1)
            self.val = nn.Linear(1, hid)
            self.nar = nn.Sequential(nn.Linear(d_nar, hid), nn.GELU(), nn.Dropout(dropout))
            self.mp = nn.ModuleList(nn.MultiheadAttention(hid, 2, dropout=dropout,
                                                          batch_first=True) for _ in range(2))
            self.norm = nn.ModuleList(nn.LayerNorm(hid) for _ in range(2))
            self.readout = nn.MultiheadAttention(hid, 2, dropout=dropout, batch_first=True)
            self.head = nn.Sequential(nn.LayerNorm(hid), nn.Dropout(dropout), nn.Linear(hid, 1))
            nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)

    def forward(self, nar, items, mask):
        out = self.lin_nar(nar).squeeze(1) + (items * mask * self.lin_item).sum(1)
        if not self.graph:
            return out
        h = self.item_emb.unsqueeze(0) + self.val(items.unsqueeze(-1))
        pad = mask == 0
        allpad = pad.all(1)
        pad_safe = pad.clone(); pad_safe[allpad, 0] = False
        for att, norm in zip(self.mp, self.norm):
            m, _ = att(h, h, h, key_padding_mask=pad_safe)
            h = norm(h + m)
        q = self.nar(nar).unsqueeze(1)
        z, _ = self.readout(q, h, h, key_padding_mask=pad_safe)
        z = torch.where(allpad[:, None, None], torch.zeros_like(z), z)
        self.g_out = self.head(q.squeeze(1) + z.squeeze(1)).squeeze(1)
        return out + self.g_out


def random_mask(n, n_items, gen, device):
    k = torch.randint(0, n_items + 1, (n,), generator=gen, device=device)
    r = torch.rand(n, n_items, generator=gen, device=device)
    rank = r.argsort(1).argsort(1)
    return (rank < k[:, None]).float()


def train(nar, items, y, fit, val, graph, seed, device):
    torch.manual_seed(seed)
    gen = torch.Generator(device=device); gen.manual_seed(seed)
    m = ItemGNN(nar.size(1), items.size(1), CFG["hid"], graph, CFG["dropout"]).to(device)
    lin = [p for n_, p in m.named_parameters() if n_.startswith("lin_")]
    gr = [p for n_, p in m.named_parameters() if not n_.startswith("lin_")]
    opt = torch.optim.AdamW([{"params": lin, "weight_decay": 1e-2},
                             {"params": gr, "weight_decay": CFG["wd_graph"]}], lr=3e-3)
    yt = torch.as_tensor(y, dtype=torch.float, device=device)
    pw = torch.tensor((1 - y[fit].mean()) / y[fit].mean(), device=device)
    fi = torch.as_tensor(fit, device=device)
    vi = torch.as_tensor(val, device=device)
    vgen = torch.Generator(device=device); vgen.manual_seed(12345)
    vmasks = [random_mask(len(val), items.size(1), vgen, device) for _ in range(4)]
    best, state, wait = -1, None, 0
    for ep in range(MAX_EP):
        m.train(); opt.zero_grad()
        mk = random_mask(len(fit), items.size(1), gen, device)
        loss = F.binary_cross_entropy_with_logits(m(nar[fi], items[fi], mk), yt[fi],
                                                  pos_weight=pw)
        if graph and CFG["graph_l2"] > 0:
            loss = loss + CFG["graph_l2"] * m.g_out.pow(2).mean()
        loss.backward()
        opt.step()
        m.eval()
        with torch.no_grad():
            auc = np.mean([roc_auc_score(y[val], m(nar[vi], items[vi], vm).cpu().numpy())
                           for vm in vmasks])
        if auc > best + 1e-5:
            best, wait, state = auc, 0, {k: v.clone() for k, v in m.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    m.load_state_dict(state); m.eval()
    return m


def main(a):
    for kv in a.cfg:
        k, v = kv.split("="); CFG[k] = type(CFG[k])(float(v))
    os.makedirs(a.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d, w = sanitize(torch.load(a.graph, map_location="cpu", weights_only=False))
    X = d[w].x.numpy()[:, :EMB + PDIN].astype(np.float64)
    y = d[w].y.view(-1).numpy().astype(int)
    n = len(y)
    coh = pd.read_csv(a.cohort_csv).cohort.values.astype(int)
    strata = y * 2 + coh
    print(f"{n} women, {y.sum()} positive; k {K_ITEMS}; {N_RAND} random subsets per size; "
          f"device={device}; {CFG}", flush=True)
    g_p = {"clf__C": C_VALUES, "ct__pca__n_components": K_GRID}
    pca_step = ("pca", PCA(10, random_state=0), EMB_COLS)

    rows = []
    for seed in a.seeds:
        set_seed(seed)
        t0 = time.time()
        topk = {(m, k): np.zeros(n) for m in [M_OR] + SINGLE for k in K_ITEMS}
        rnd = {(m, k): [] for m in SINGLE for k in K_ITEMS}
        rng = np.random.RandomState(seed)
        for fold, (tr, te) in enumerate(
                StratifiedKFold(5, shuffle=True, random_state=seed).split(X, strata)):
            order = [c - EMB for c in rank_items(X, y, tr)]
            for k in K_ITEMS:
                cols = [EMB + i for i in order[:k]]
                parts = [pca_step] + ([("pdi", "passthrough", cols)] if k else [])
                topk[M_OR, k][te], _ = fit_predict(pipe(parts), g_p, X, y, tr, te, seed)
            lr_all, _ = None, None
            from sklearn.model_selection import GridSearchCV
            gs = GridSearchCV(pipe([pca_step, ("pdi", "passthrough", PDI_COLS)]), g_p,
                              scoring="roc_auc", n_jobs=-1,
                              cv=StratifiedKFold(3, shuffle=True, random_state=seed)).fit(X[tr], y[tr])
            mu = X[tr][:, PDI_COLS].mean(0)

            def lr_masked(mask):
                Xt = X[te].copy()
                Xt[:, PDI_COLS] = np.where(mask > 0, Xt[:, PDI_COLS], mu)
                return gs.best_estimator_.predict_proba(Xt)[:, 1]

            pca = PCA(N_PCA, random_state=0).fit(X[tr][:, EMB_COLS])
            Zn = pca.transform(X[:, EMB_COLS]); Zn = (Zn - Zn[tr].mean(0)) / Zn[tr].std(0)
            It = X[:, PDI_COLS]; It = (It - It[tr].mean(0)) / It[tr].std(0).clip(1e-9)
            nar = torch.as_tensor(Zn, dtype=torch.float, device=device)
            items = torch.as_tensor(It, dtype=torch.float, device=device)
            fit, val = train_test_split(tr, test_size=0.2, stratify=strata[tr],
                                        random_state=seed + fold)
            models = {M_LIN: [train(nar, items, y, fit, val, False, seed * 100 + fold * 10 + e, device)
                              for e in range(N_ENS)],
                      M_G: [train(nar, items, y, fit, val, True, seed * 100 + fold * 10 + e, device)
                            for e in range(N_ENS)]}
            tt = torch.as_tensor(te, device=device)

            def torch_pred(name, mask):
                mk = torch.as_tensor(mask, dtype=torch.float, device=device)
                with torch.no_grad():
                    return np.mean([mm(nar[tt], items[tt], mk).cpu().numpy()
                                    for mm in models[name]], 0)

            for k in K_ITEMS:
                mask = np.zeros((len(te), PDIN)); mask[:, order[:k]] = 1
                topk[M_MI, k][te] = lr_masked(mask)
                for mname in [M_LIN, M_G]:
                    topk[mname, k][te] = torch_pred(mname, mask)
                for _ in range(N_RAND if 0 < k < PDIN else 1):
                    mask = np.zeros((len(te), PDIN))
                    mask[:, rng.choice(PDIN, k, replace=False)] = 1
                    rnd[M_MI, k].append(roc_auc_score(y[te], lr_masked(mask)))
                    for mname in [M_LIN, M_G]:
                        rnd[mname, k].append(roc_auc_score(y[te], torch_pred(mname, mask)))
        for (m, k), p in topk.items():
            rows.append({"seed": seed, "model": m, "k": k, "subset": "top-k",
                         "auc": roc_auc_score(y, p)})
        for (m, k), v in rnd.items():
            rows.append({"seed": seed, "model": m, "k": k, "subset": "random",
                         "auc": np.mean(v)})
        pd.DataFrame({f"{m}|{k}": p for (m, k), p in topk.items()} | {"y_true": y}).to_csv(
            os.path.join(a.out, f"oof_seed{seed}.csv"), index=False)
        print(f"  seed {seed} ({time.time()-t0:.0f}s): top-k AUC at k=0/6/13  " + " | ".join(
            f"{m} {roc_auc_score(y, topk[m, 0]):.3f}/{roc_auc_score(y, topk[m, 6]):.3f}/"
            f"{roc_auc_score(y, topk[m, 13]):.3f}" for m in [M_OR] + SINGLE), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "auc.csv"), index=False)
    oofs = {s: pd.read_csv(os.path.join(a.out, f"oof_seed{s}.csv")) for s in a.seeds}
    for sub in ["top-k", "random"]:
        t = df[df.subset == sub].groupby(["k", "model"]).auc.mean().unstack()
        cols = [c for c in [M_OR, M_MI, M_LIN, M_G] if c in t.columns]
        print(f"\n=== AUC by number of items, {sub} subsets (mean over seeds"
              f"{'; random = mean over folds x draws' if sub == 'random' else ''}) ===")
        print(t[cols].round(4).to_string())
    print(f"\n=== top-k: PeriGNNosis-Items vs each model, paired permutation "
          f"({a.perms} draws) ===")
    for k in K_ITEMS:
        parts = []
        for ref in [M_OR, M_MI, M_LIN]:
            r = [perm_test(o.y_true.values, o[f"{M_G}|{k}"].values, o[f"{ref}|{k}"].values,
                           n=a.perms, seed=s) for s, o in oofs.items()]
            parts.append(f"vs {ref.split(',')[0].split(' (')[0]}: {np.mean([x[0] for x in r]):+.4f}"
                         f" (p {min(x[1] for x in r):.2f}-{max(x[1] for x in r):.2f}, "
                         f"{sum(x[0] > 0 for x in r)}/{len(r)})")
        print(f"  k={k:<3} " + " | ".join(parts))
    print(f"\nsaved to {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="perignnosis_graph_pooled.pt")
    ap.add_argument("--cohort-csv", default="pooled_cohort.csv")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--perms", type=int, default=1000)
    ap.add_argument("--out", default="results_item_gnn")
    ap.add_argument("--cfg", nargs="*", default=[], help="e.g. hid=16 graph_l2=1")
    main(ap.parse_args())
