"""
Graph over the sentences of each narrative, combined with logistic regression.

Usage:
    python sentence_gnn.py --seeds 1 2 3 4 5 --out results_sentence_gnn/
"""

import argparse, os, time
import numpy as np, pandas as pd, torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import logit
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import softmax as seg_softmax, scatter

from fair_benchmark import EMB, PDIN, sanitize, set_seed
from pca_narrative_diagnostic import perm_test
from popgraph_cs import lr_tuned

N_ENS, MAX_EP, PATIENCE = 5, 300, 30
CFG = {"n_comp": 32, "hid": 32, "wd": 1e-2, "corr_l2": 0.0, "dropout": 0.3}
M_LR, M_S, M_S0 = "LR (narrative + PDI-13)", "PeriGNNosis-S", "PeriGNNosis-S (no edges)"
MODELS = [M_LR, M_S, M_S0]


class SentenceGNN(nn.Module):
    def __init__(self, d_in, hid, edges=True, dropout=0.3):
        super().__init__()
        self.edges = edges
        self.inp = nn.Sequential(nn.Linear(d_in, hid), nn.GELU(), nn.Dropout(dropout))
        if edges:
            self.c1 = GATv2Conv(hid, hid, heads=2, concat=False, add_self_loops=True)
            self.c2 = GATv2Conv(hid, hid, heads=2, concat=False, add_self_loops=True)
        else:
            self.c1 = nn.Linear(hid, hid); self.c2 = nn.Linear(hid, hid)
        self.drop = nn.Dropout(dropout)
        self.att_v = nn.Linear(hid, hid); self.att_u = nn.Linear(hid, hid)
        self.att_w = nn.Linear(hid, 1)
        self.head = nn.Linear(hid, 1)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, x, ei, batch, n_graphs):
        h = self.inp(x)
        if self.edges:
            h = h + self.drop(F.gelu(self.c1(h, ei)))
            h = h + self.drop(F.gelu(self.c2(h, ei)))
        else:
            h = h + self.drop(F.gelu(self.c1(h)))
            h = h + self.drop(F.gelu(self.c2(h)))
        a = self.att_w(torch.tanh(self.att_v(h)) * torch.sigmoid(self.att_u(h))).squeeze(1)
        a = seg_softmax(a, batch, num_nodes=n_graphs)
        z = scatter(a.unsqueeze(1) * h, batch, dim=0, dim_size=n_graphs, reduce="sum")
        return self.head(z).squeeze(1), a


def build_graph(s, n_women):
    w, pos = s["woman"], s["pos"]
    order = np.lexsort((pos, w))
    w_o = w[order]
    same = w_o[1:] == w_o[:-1]
    src, dst = order[:-1][same], order[1:][same]
    ei = np.r_[np.c_[src, dst], np.c_[dst, src]].T
    n_s = np.bincount(w, minlength=n_women)
    frac = pos / np.maximum(n_s[w] - 1, 1)
    return torch.as_tensor(ei, dtype=torch.long), frac


def train_one(Xs, ei, batch, n, offset, y, fit, val, edges, seed, device):
    torch.manual_seed(seed)
    m = SentenceGNN(Xs.size(1), CFG["hid"], edges, CFG["dropout"]).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=CFG["wd"])
    yt = torch.as_tensor(y, dtype=torch.float, device=device)
    fi = torch.as_tensor(fit, device=device)
    best, best_state, wait = -1, None, 0
    for ep in range(MAX_EP):
        m.train(); opt.zero_grad()
        corr, _ = m(Xs, ei, batch, n)
        loss = F.binary_cross_entropy_with_logits((offset + corr)[fi], yt[fi])
        (loss + CFG["corr_l2"] * corr[fi].pow(2).mean()).backward()
        opt.step()
        m.eval()
        with torch.no_grad():
            corr, _ = m(Xs, ei, batch, n)
        auc = roc_auc_score(y[val], (offset + corr)[val].cpu().numpy())
        if auc > best + 1e-5:
            best, wait = auc, 0
            best_state = {k: v.clone() for k, v in m.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    m.load_state_dict(best_state); m.eval()
    with torch.no_grad():
        corr, att = m(Xs, ei, batch, n)
    return corr.cpu().numpy(), att.cpu().numpy()


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
    s = np.load(a.sentences)
    ei, frac = build_graph(s, n)
    batch = torch.as_tensor(s["woman"], dtype=torch.long, device=device)
    ei = ei.to(device)
    print(f"{n} women, {y.sum()} positive; {len(s['woman'])} sentence nodes, "
          f"{ei.size(1)} story-order edges; device={device}; {CFG}", flush=True)

    rows, att_rows = [], []
    for seed in a.seeds:
        set_seed(seed)
        t0 = time.time()
        oof = {m: np.zeros(n) for m in MODELS}
        for fold, (tr, te) in enumerate(
                StratifiedKFold(5, shuffle=True, random_state=seed).split(X, strata)):
            best = lr_tuned(X, y, tr, seed)
            p = np.zeros(n)
            p[te] = best.predict_proba(X[te])[:, 1]
            for a_, b_ in StratifiedKFold(5, shuffle=True, random_state=seed).split(
                    tr, strata[tr]):
                p[tr[b_]] = clone(best).fit(X[tr[a_]], y[tr[a_]]).predict_proba(
                    X[tr[b_]])[:, 1]
            off = logit(np.clip(p, 1e-6, 1 - 1e-6))
            oof[M_LR][te] = off[te]
            offset = torch.as_tensor(off, dtype=torch.float, device=device)

            in_tr = np.isin(s["woman"], tr)
            pca = PCA(CFG["n_comp"], random_state=0).fit(s["emb"][in_tr])
            Z = pca.transform(s["emb"])
            Z = (Z - Z[in_tr].mean(0)) / Z[in_tr].std(0)
            Xs = torch.as_tensor(np.c_[Z, frac], dtype=torch.float, device=device)

            fit, val = train_test_split(tr, test_size=0.2, stratify=strata[tr],
                                        random_state=seed + fold)
            for m_name, edges in [(M_S, True), (M_S0, False)]:
                corrs = []
                for e in range(N_ENS):
                    c, att = train_one(Xs, ei, batch, n, offset, y, fit, val, edges,
                                       seed * 100 + fold * 10 + e, device)
                    corrs.append(c)
                    if m_name == M_S and e == 0:
                        sel = np.isin(s["woman"], te)
                        att_rows.append(pd.DataFrame({"seed": seed, "woman": s["woman"][sel],
                                                      "pos": s["pos"][sel], "att": att[sel]}))
                oof[m_name][te] = off[te] + np.mean(corrs, 0)[te]
        for m in MODELS:
            rows.append({"seed": seed, "model": m, "auc": roc_auc_score(y, oof[m])})
        pd.DataFrame(oof | {"y_true": y}).to_csv(
            os.path.join(a.out, f"oof_seed{seed}.csv"), index=False)
        print(f"  seed {seed} ({time.time()-t0:.0f}s): " + " | ".join(
            f"{m} {roc_auc_score(y, oof[m]):.4f}" for m in MODELS), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "auc.csv"), index=False)
    pd.concat(att_rows).to_csv(os.path.join(a.out, "attention.csv"), index=False)
    print("\n=== pooled OOF AUC ===")
    print(df.groupby("model").auc.agg(["mean", "std"]).round(4)
            .sort_values("mean", ascending=False).to_string())
    print(f"\n=== paired permutation tests ({a.perms} draws) ===")
    for m1, m2 in [(M_S, M_LR), (M_S0, M_LR), (M_S, M_S0)]:
        ds, ps = [], []
        for seed in a.seeds:
            o = pd.read_csv(os.path.join(a.out, f"oof_seed{seed}.csv"))
            dd, pv = perm_test(o.y_true.values, o[m1].values, o[m2].values, n=a.perms, seed=seed)
            ds.append(dd); ps.append(pv)
        print(f"  {m1} - {m2}: Δ {np.mean(ds):+.4f}  perm p {min(ps):.3f}-{max(ps):.3f}"
              f"  wins {sum(x > 0 for x in ds)}/{len(ds)}")
    print(f"\nsaved to {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="perignnosis_graph_pooled.pt")
    ap.add_argument("--cohort-csv", default="pooled_cohort.csv")
    ap.add_argument("--sentences", default="narrative_sentences.npz")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--perms", type=int, default=2000)
    ap.add_argument("--out", default="results_sentence_gnn")
    ap.add_argument("--cfg", nargs="*", default=[], help="overrides, e.g. hid=16 corr_l2=1")
    main(ap.parse_args())
