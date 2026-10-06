"""
Self-supervised GNN on the woman-word graph.

Usage:
    python ssl_word_gnn.py --seeds 1 2 3 4 5 --out results_ssl_word_gnn/
"""

import argparse, os, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np, pandas as pd, torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize

from fair_benchmark import EMB, PDIN, sanitize, set_seed
from pca_narrative_diagnostic import fit_predict, perm_test, C_VALUES, EMB_COLS
from graph_signal_diagnostic import pipe
from narrative_short_pdi import SIX_COLS
from word_graph_probe import vectorizer
from word_embed_probe import word_vectors
from word_gnn import sparse

HID, EPOCHS, LRATE, WD, NEG, BATCH = 32, 300, 3e-3, 1e-4, 5, 4096
M_LR, M_LSA = "LR (narrative + six)", "LR (narrative + six + LSA)"
M_SSL, M_BOTH = "LR (narrative + six + SSL-GNN)", "LR (narrative + six + LSA + SSL-GNN)"
MODELS = [M_LR, M_LSA, M_SSL, M_BOTH]


class Encoder(nn.Module):
    def __init__(self, d_w, d_v):
        super().__init__()
        self.ew, self.ev = nn.Linear(d_w, HID), nn.Linear(d_v, HID)
        self.v1s, self.v1n = nn.Linear(HID, HID), nn.Linear(HID, HID)
        self.w1s, self.w1n = nn.Linear(HID, HID), nn.Linear(HID, HID)
        self.v2s, self.v2n = nn.Linear(HID, HID), nn.Linear(HID, HID)
        self.w2s, self.w2n = nn.Linear(HID, HID), nn.Linear(HID, HID)

    def forward(self, xw, xv, Wn, Wc):
        hw, hv = F.gelu(self.ew(xw)), F.gelu(self.ev(xv))
        hv1 = F.gelu(self.v1s(hv) + self.v1n(torch.sparse.mm(Wc, hw)))
        hw1 = F.gelu(self.w1s(hw) + self.w1n(torch.sparse.mm(Wn, hv)))
        hv2 = self.v2s(hv1) + self.v2n(torch.sparse.mm(Wc, hw1))
        hw2 = self.w2s(hw1) + self.w2n(torch.sparse.mm(Wn, hv1))
        return hw2, hv2


def ssl_embed(xw, xv, T, tr, seed, device):
    n, V = T.shape
    is_tr = np.isin(np.arange(n), tr)
    Ttr = T.multiply(is_tr[:, None]).tocsr()
    Wn_tr = sparse(normalize(Ttr, norm="l1", axis=1), device)
    Wn_all = sparse(normalize(T, norm="l1", axis=1), device)
    Wc = sparse(normalize(Ttr.T.tocsr(), norm="l1", axis=1), device)
    coo = Ttr.tocoo()
    ew, ev, ewt = coo.row, coo.col, coo.data / coo.data.sum()
    torch.manual_seed(seed); rng = np.random.RandomState(seed)
    m = Encoder(xw.size(1), xv.size(1)).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=LRATE, weight_decay=WD)
    for ep in range(EPOCHS):
        m.train(); opt.zero_grad()
        hw, hv = m(xw, xv, Wn_tr, Wc)
        idx = rng.choice(len(ew), BATCH, p=ewt)
        wi = torch.as_tensor(ew[idx], device=device)
        vi = torch.as_tensor(ev[idx], device=device)
        pos = (hw[wi] * hv[vi]).sum(1)
        vn = torch.as_tensor(rng.randint(0, V, (BATCH, NEG)), device=device)
        neg = (hw[wi].unsqueeze(1) * hv[vn]).sum(2)
        loss = (F.binary_cross_entropy_with_logits(pos, torch.ones_like(pos))
                + F.binary_cross_entropy_with_logits(neg, torch.zeros_like(neg)))
        loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        hw, _ = m(xw, xv, Wn_all, Wc)
    return hw.cpu().numpy(), float(loss)


def main(a):
    from sentence_transformers import SentenceTransformer
    os.makedirs(a.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = SentenceTransformer("jinaai/jina-embeddings-v2-base-en", trust_remote_code=True,
                              device=str(device))
    d, w = sanitize(torch.load(a.graph, map_location="cpu", weights_only=False))
    X0 = d[w].x.numpy()[:, :EMB + PDIN].astype(np.float64)
    y = d[w].y.view(-1).numpy().astype(int)
    meta = pd.read_csv(a.cohort_csv)
    coh = meta.cohort.values.astype(int)
    texts = meta.cb_delivery_narrative.fillna("").values
    strata = y * 2 + coh
    n = len(y)
    print(f"{n} women, {y.sum()} positive; device={device}", flush=True)

    res, oofs = {m: [] for m in MODELS}, []
    for seed in a.seeds:
        set_seed(seed)
        t0 = time.time()
        o = {m: np.zeros(n) for m in MODELS}
        for fold, (tr, te) in enumerate(
                StratifiedKFold(5, shuffle=True, random_state=seed).split(X0, strata)):
            vec = vectorizer().fit(texts[tr])
            T = vec.transform(texts).tocsr()
            E = word_vectors(list(vec.get_feature_names_out()), enc)
            xv = PCA(32, random_state=0).fit_transform(E); xv = (xv - xv.mean(0)) / xv.std(0)
            pz = PCA(20, random_state=0).fit(X0[tr][:, EMB_COLS])
            xw = pz.transform(X0[:, EMB_COLS]); xw = (xw - xw[tr].mean(0)) / xw[tr].std(0)
            t = lambda A: torch.as_tensor(A, dtype=torch.float, device=device)
            H, last = ssl_embed(t(xw), t(xv), T, tr, seed * 10 + fold, device)
            X = np.c_[X0, H, T.toarray()]
            hc = list(range(EMB + PDIN, EMB + PDIN + HID))
            tc = list(range(hc[-1] + 1, X.shape[1]))
            pca = ("pca", PCA(10, random_state=0), EMB_COLS)
            six = ("pdi", "passthrough", SIX_COLS)
            lsa = ("svd", TruncatedSVD(50, random_state=0), tc)
            ssl = ("ssl", "passthrough", hc)
            g = {"clf__C": C_VALUES, "ct__pca__n_components": [10, 20, 50]}
            gs = {"ct__svd__n_components": [20, 50, 100]}
            specs = {M_LR: ([pca, six], g), M_LSA: ([pca, six, lsa], g | gs),
                     M_SSL: ([pca, six, ssl], g), M_BOTH: ([pca, six, lsa, ssl], g | gs)}
            for m, (parts, gg) in specs.items():
                o[m][te], _ = fit_predict(pipe(parts), gg, X, y, tr, te, seed)
        pd.DataFrame(o | {"y_true": y}).to_csv(os.path.join(a.out, f"oof_seed{seed}.csv"), index=False)
        oofs.append(o)
        for m in MODELS:
            res[m].append(roc_auc_score(y, o[m]))
        print(f"  seed {seed} ({time.time()-t0:.0f}s; final SSL loss {last:.3f}): " + " | ".join(
            f"{m} {res[m][-1]:.4f}" for m in MODELS), flush=True)

    print("\n=== OOF AUC (mean over seeds) ===")
    for m in MODELS:
        print(f"  {m:<40} {np.mean(res[m]):.4f}  (sd {np.std(res[m]):.4f})")
    print(f"\n=== paired permutation tests ({a.perms} draws) ===")
    for m1, m2 in [(M_SSL, M_LR), (M_SSL, M_LSA), (M_BOTH, M_LSA), (M_LSA, M_LR)]:
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
    ap.add_argument("--out", default="results_ssl_word_gnn")
    main(ap.parse_args())
