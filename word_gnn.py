"""
Graph neural network on the woman-word graph.

Usage:
    python word_gnn.py --seeds 1 2 3 4 5 --out results_word_gnn/
"""

import argparse, os, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np, pandas as pd, torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import logit
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.preprocessing import normalize

from fair_benchmark import EMB, PDIN, sanitize, set_seed
from pca_narrative_diagnostic import perm_test, C_VALUES, K_GRID, EMB_COLS, lr
from graph_signal_diagnostic import pipe
from narrative_short_pdi import SIX, SIX_COLS
from word_graph_probe import vectorizer
import word_graph_confirmatory as wgc
from word_embed_probe import word_vectors

HID, DROP, EDGE_DROP, WD, LRATE, N_ENS, MAX_EP, PATIENCE = 32, 0.3, 0.2, 1e-2, 3e-3, 5, 300, 30
CORR_L2 = 0.0
M_LR, M_LSA = "LR (narrative + six)", "LR (narrative + six + LSA)"
M_W, M_W0 = "PeriGNNosis-W", "PeriGNNosis-W without graph"
MODELS = [M_LR, M_LSA, M_W, M_W0]


def sparse(M, device):
    M = M.tocoo()
    return torch.sparse_coo_tensor(np.vstack([M.row, M.col]), M.data.astype(np.float32),
                                   M.shape, device=device).coalesce()


def drop_edges(S, p, training):
    if not training or p == 0:
        return S
    keep = torch.rand(S._nnz(), device=S.device) > p
    idx, val = S.indices()[:, keep], S.values()[keep]
    rs = torch.zeros(S.shape[0], device=S.device).index_add_(0, idx[0], val)
    return torch.sparse_coo_tensor(idx, val / rs[idx[0]].clamp(min=1e-9), S.shape).coalesce()


class WordGNN(nn.Module):
    def __init__(self, d_w, d_v, graph=True):
        super().__init__()
        self.graph = graph
        self.enc_w = nn.Sequential(nn.Linear(d_w, HID), nn.GELU(), nn.Dropout(DROP))
        if graph:
            self.enc_v = nn.Sequential(nn.Linear(d_v, HID), nn.GELU(), nn.Dropout(DROP))
            self.v_self, self.v_nb = nn.Linear(HID, HID), nn.Linear(HID, HID)
            self.w_self, self.w_nb = nn.Linear(HID, HID), nn.Linear(HID, HID)
        else:
            self.w_self = nn.Linear(HID, HID)
        self.drop = nn.Dropout(DROP)
        self.head = nn.Linear(HID, 1)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, xw, xv, Wn, Wc):
        hw = self.enc_w(xw)
        if self.graph:
            Wn_, Wc_ = drop_edges(Wn, EDGE_DROP, self.training), drop_edges(Wc, EDGE_DROP, self.training)
            hv = self.enc_v(xv)
            hv = F.gelu(self.v_self(hv) + self.v_nb(torch.sparse.mm(Wc_, hw)))
            hw = F.gelu(self.w_self(hw) + self.w_nb(torch.sparse.mm(Wn_, self.drop(hv))))
        else:
            hw = F.gelu(self.w_self(hw))
        return self.head(self.drop(hw)).squeeze(1)


def train(xw, xv, Wn, Wc, off, y, fit, val, graph, seed, device):
    torch.manual_seed(seed)
    m = WordGNN(xw.size(1), xv.size(1), graph).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=LRATE, weight_decay=WD)
    yt = torch.as_tensor(y, dtype=torch.float, device=device)
    fi = torch.as_tensor(fit, device=device)
    best, state, wait = -1, None, 0
    for ep in range(MAX_EP):
        m.train(); opt.zero_grad()
        c = m(xw, xv, Wn, Wc)
        (F.binary_cross_entropy_with_logits((off + c)[fi], yt[fi])
         + CORR_L2 * c[fi].pow(2).mean()).backward()
        opt.step()
        m.eval()
        with torch.no_grad():
            s = (off + m(xw, xv, Wn, Wc))[val].cpu().numpy()
        auc = roc_auc_score(y[val], s)
        if auc > best + 1e-5:
            best, wait, state = auc, 0, {k: v.clone() for k, v in m.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    m.load_state_dict(state); m.eval()
    with torch.no_grad():
        return m(xw, xv, Wn, Wc).cpu().numpy()


def main(a):
    global HID, WD, CORR_L2
    if a.shrink:
        HID, WD, CORR_L2 = 16, 0.1, 1.0
    from sentence_transformers import SentenceTransformer
    os.makedirs(a.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = SentenceTransformer("jinaai/jina-embeddings-v2-base-en", trust_remote_code=True,
                              device=str(device))
    d, w = sanitize(torch.load(a.graph, map_location="cpu", weights_only=False))
    X = d[w].x.numpy()[:, :EMB + PDIN].astype(np.float64)
    y = d[w].y.view(-1).numpy().astype(int)
    meta = pd.read_csv(a.cohort_csv)
    coh = meta.cohort.values.astype(int)
    texts = meta.cb_delivery_narrative.fillna("").values
    strata = y * 2 + coh
    n = len(y)
    df = pd.DataFrame(X, columns=wgc.EMB_NAMES + wgc.PDI_NAMES); df["text"] = texts
    print(f"{n} women, {y.sum()} positive; six items {SIX}; device={device}; "
          f"hidden {HID}, weight decay {WD}, correction penalty {CORR_L2}", flush=True)

    res, oofs = {m: [] for m in MODELS}, []
    for seed in a.seeds:
        path = os.path.join(a.out, f"oof_seed{seed}.csv")
        if os.path.exists(path):
            o = pd.read_csv(path); oofs.append({m: o[m].values for m in MODELS})
            for m in MODELS: res[m].append(roc_auc_score(y, o[m]))
            continue
        set_seed(seed)
        t0 = time.time()
        o = {m: np.zeros(n) for m in MODELS}
        for fold, (tr, te) in enumerate(
                StratifiedKFold(5, shuffle=True, random_state=seed).split(X, strata)):
            gs = GridSearchCV(pipe([("pca", PCA(10, random_state=0), EMB_COLS),
                                    ("pdi", "passthrough", SIX_COLS)]),
                              {"clf__C": C_VALUES, "ct__pca__n_components": K_GRID},
                              scoring="roc_auc", n_jobs=-1,
                              cv=StratifiedKFold(3, shuffle=True, random_state=seed)).fit(X[tr], y[tr])
            p = np.zeros(n)
            p[te] = gs.best_estimator_.predict_proba(X[te])[:, 1]
            for a_, b_ in StratifiedKFold(5, shuffle=True, random_state=seed).split(tr, strata[tr]):
                p[tr[b_]] = clone(gs.best_estimator_).fit(X[tr[a_]], y[tr[a_]]).predict_proba(X[tr[b_]])[:, 1]
            off_np = logit(np.clip(p, 1e-6, 1 - 1e-6))
            o[M_LR][te] = off_np[te]
            g2 = GridSearchCV(wgc.model([f"pdi_q{q}" for q in SIX]), wgc.GRID, scoring="roc_auc",
                              n_jobs=-1, cv=StratifiedKFold(3, shuffle=True, random_state=seed))
            o[M_LSA][te] = g2.fit(df.iloc[tr], y[tr]).predict_proba(df.iloc[te])[:, 1]
            vec = vectorizer().fit(texts[tr])
            T = vec.transform(texts).tocsr()
            Wn = normalize(T, norm="l1", axis=1)
            Ttr = T.multiply(np.isin(np.arange(n), tr)[:, None]).tocsr()
            Wc = normalize(Ttr.T.tocsr(), norm="l1", axis=1)
            E = word_vectors(list(vec.get_feature_names_out()), enc)
            xv_np = PCA(32, random_state=0).fit_transform(E)
            xv_np = (xv_np - xv_np.mean(0)) / xv_np.std(0)
            pz = PCA(20, random_state=0).fit(X[tr][:, EMB_COLS])
            xw_np = np.c_[pz.transform(X[:, EMB_COLS]), X[:, SIX_COLS]]
            xw_np = (xw_np - xw_np[tr].mean(0)) / xw_np[tr].std(0).clip(1e-9)
            t = lambda A: torch.as_tensor(A, dtype=torch.float, device=device)
            xw, xv, off = t(xw_np), t(xv_np), t(off_np)
            Wn_t, Wc_t = sparse(Wn, device), sparse(Wc, device)
            fit, val = train_test_split(tr, test_size=0.2, stratify=strata[tr],
                                        random_state=seed * 10 + fold)
            for name, graph in [(M_W, True), (M_W0, False)]:
                corr = np.mean([train(xw, xv, Wn_t, Wc_t, off, y, fit, val, graph,
                                      seed * 100 + fold * 10 + e, device)
                                for e in range(N_ENS)], 0)
                o[name][te] = off_np[te] + corr[te]
        pd.DataFrame(o | {"y_true": y}).to_csv(path, index=False)
        oofs.append(o)
        for m in MODELS:
            res[m].append(roc_auc_score(y, o[m]))
        print(f"  seed {seed} ({time.time()-t0:.0f}s): " + " | ".join(
            f"{m} {res[m][-1]:.4f}" for m in MODELS), flush=True)

    print("\n=== OOF AUC (mean over seeds) ===")
    for m in MODELS:
        print(f"  {m:<30} {np.mean(res[m]):.4f}  (sd {np.std(res[m]):.4f})")
    print(f"\n=== paired permutation tests ({a.perms} draws) ===")
    for m1, m2 in [(M_W, M_LR), (M_W, M_LSA), (M_W, M_W0), (M_LSA, M_LR)]:
        r = [perm_test(y, o[m1], o[m2], n=a.perms, seed=s) for s, o in zip(a.seeds, oofs)]
        print(f"  {m1} - {m2}: Δ {np.mean([x[0] for x in r]):+.4f}  perm p "
              f"{min(x[1] for x in r):.3f}-{max(x[1] for x in r):.3f}  "
              f"wins {sum(x[0] > 0 for x in r)}/{len(r)}")
    print(f"\nsaved to {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="perignnosis_graph_pooled.pt")
    ap.add_argument("--cohort-csv", default="pooled_cohort.csv")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--perms", type=int, default=2000)
    ap.add_argument("--out", default="results_word_gnn")
    ap.add_argument("--shrink", action="store_true",
                    help="hidden 16, weight decay 0.1, correction penalty 1")
    main(ap.parse_args())
