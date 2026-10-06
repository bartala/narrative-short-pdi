"""
Woman-word graph features (spectral embedding, label propagation).

Usage:
    python word_graph_probe.py --seeds 1 2 3 4 5
"""

import argparse, time
import numpy as np, pandas as pd, torch
from scipy import sparse
from joblib import Parallel, delayed
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from fair_benchmark import EMB, PDIN, sanitize, set_seed
from pca_narrative_diagnostic import perm_test, C_VALUES, K_GRID, EMB_COLS, PDI_COLS

SVD_K = [20, 50, 100]
PRIOR_STRENGTH = 10.0
KNN = 25


def vectorizer():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=5, sublinear_tf=True,
                           token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z']+\b", lowercase=True)


def word_messages(T, y, src, dst):
    prior = y[src].mean()
    B = (T[src] > 0).astype(float)
    users = np.asarray(B.sum(0)).ravel()
    pos = np.asarray(B.T @ y[src]).ravel()
    rate = (pos + PRIOR_STRENGTH * prior) / (users + PRIOR_STRENGTH)
    Td = T[dst]
    wsum = np.asarray(Td.sum(1)).ravel()
    mean_rate = np.asarray(Td @ rate).ravel() / np.maximum(wsum, 1e-9)
    Bd = (Td > 0).tocsr()
    max_rate = np.array([rate[Bd.indices[Bd.indptr[i]:Bd.indptr[i + 1]]].max()
                         if Bd.indptr[i + 1] > Bd.indptr[i] else prior
                         for i in range(Bd.shape[0])])
    hot = rate >= np.quantile(rate[users > 0], 0.95)
    share_hot = np.asarray(Bd @ hot.astype(float)).ravel() / np.maximum(
        np.asarray(Bd.sum(1)).ravel(), 1)
    S = (Td @ T[src].T).toarray()
    same = np.asarray(dst)[:, None] == np.asarray(src)[None, :]
    S[same] = -1
    nb = np.argsort(-S, 1)[:, :KNN]
    w = np.take_along_axis(S, nb, 1).clip(min=0)
    nbr = (w * y[src][nb]).sum(1) / np.maximum(w.sum(1), 1e-9)
    return np.c_[mean_rate, max_rate, share_hot, nbr]


def graph_features(T, y, tr, seed):
    F = np.zeros((T.shape[0], 4))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(tr, y[tr]):
        F[tr[b]] = word_messages(T, y, tr[a], tr[b])
    rest = np.setdiff1d(np.arange(T.shape[0]), tr)
    F[rest] = word_messages(T, y, tr, rest)
    return F


def fit_lr(blocks_tr, blocks_te, y_tr, seed, grid):
    inner = list(StratifiedKFold(3, shuffle=True, random_state=seed).split(
        np.zeros(len(y_tr)), y_tr))
    def score(prm):
        sc = []
        for a, b in inner:
            A, Bm = blocks_tr(prm, a, b)
            m = LogisticRegression(C=prm["C"], class_weight="balanced", solver="liblinear",
                                   max_iter=3000).fit(A, y_tr[a])
            sc.append(roc_auc_score(y_tr[b], m.predict_proba(Bm)[:, 1]))
        return np.mean(sc)
    scores = Parallel(n_jobs=-1, backend="threading")(delayed(score)(p) for p in grid)
    best = (max(scores), grid[int(np.argmax(scores))])
    A, Bm = blocks_te(best[1])
    m = LogisticRegression(C=best[1]["C"], class_weight="balanced", solver="liblinear",
                           max_iter=3000).fit(A, y_tr)
    return m.predict_proba(Bm)[:, 1], best[1]


def main(a):
    d, w = sanitize(torch.load(a.graph, map_location="cpu", weights_only=False))
    X = d[w].x.numpy()[:, :EMB + PDIN].astype(np.float64)
    y = d[w].y.view(-1).numpy().astype(int)
    meta = pd.read_csv(a.cohort_csv)
    coh = meta.cohort.values.astype(int)
    texts = meta.cb_delivery_narrative.fillna("").values
    strata = y * 2 + coh
    print(f"{len(y)} women, {y.sum()} positive", flush=True)

    models = ["base", "+ word features (SVD)", "+ word-graph messages", "+ both",
              "words only (TF-IDF LR)", "word-graph messages only"]
    res, oofs, top_words = {m: [] for m in models}, [], []
    for seed in a.seeds:
        set_seed(seed)
        t0 = time.time()
        o = {m: np.zeros(len(y)) for m in models}
        for fold, (tr, te) in enumerate(
                StratifiedKFold(5, shuffle=True, random_state=seed).split(X, strata)):
            vec = vectorizer().fit(texts[tr])
            T = vec.transform(texts).tocsr()
            G = graph_features(T, y, tr, seed)
            if seed == a.seeds[0]:
                B = (T[tr] > 0).astype(float)
                users = np.asarray(B.sum(0)).ravel()
                pos = np.asarray(B.T @ y[tr]).ravel()
                p0 = y[tr].mean()
                rate = (pos + PRIOR_STRENGTH * p0) / (users + PRIOR_STRENGTH)
                voc = np.array(vec.get_feature_names_out())
                ok = users >= 15
                for j in np.argsort(-rate * ok)[:25]:
                    top_words.append({"fold": fold, "word": voc[j], "rate": rate[j],
                                      "users": int(users[j])})
            ytr = y[tr]

            def dense(prm, idx_fit, idx_apply, with_pdi=True, with_nar=True, extra=None):
                parts_f, parts_a = [], []
                if with_nar:
                    p = PCA(prm["k"], random_state=0).fit(X[idx_fit][:, EMB_COLS])
                    parts_f.append(p.transform(X[idx_fit][:, EMB_COLS]))
                    parts_a.append(p.transform(X[idx_apply][:, EMB_COLS]))
                if with_pdi:
                    parts_f.append(X[idx_fit][:, PDI_COLS]); parts_a.append(X[idx_apply][:, PDI_COLS])
                if extra is not None:
                    ef, ea = extra(prm, idx_fit, idx_apply)
                    parts_f.append(ef); parts_a.append(ea)
                F, A_ = np.concatenate(parts_f, 1), np.concatenate(parts_a, 1)
                sc = StandardScaler().fit(F)
                return sc.transform(F), sc.transform(A_)

            def svd_extra(prm, i_f, i_a):
                s = TruncatedSVD(prm["s"], random_state=0).fit(T[i_f])
                return s.transform(T[i_f]), s.transform(T[i_a])

            def g_extra(prm, i_f, i_a):
                return G[i_f], G[i_a]

            def both_extra(prm, i_f, i_a):
                a1, b1 = svd_extra(prm, i_f, i_a)
                return np.c_[a1, G[i_f]], np.c_[b1, G[i_a]]

            gk = [{"C": c, "k": k} for c in C_VALUES for k in K_GRID]
            gks = [{"C": c, "k": k, "s": s} for c in C_VALUES for k in [10, 20, 50]
                   for s in SVD_K]
            specs = {
                "base": (dict(), gk),
                "+ word features (SVD)": (dict(extra=svd_extra), gks),
                "+ word-graph messages": (dict(extra=g_extra), gk),
                "+ both": (dict(extra=both_extra), gks),
                "word-graph messages only": (dict(with_pdi=False, with_nar=False, extra=g_extra),
                                             [{"C": c, "k": 0} for c in C_VALUES]),
            }
            for m, (kw, grid) in specs.items():
                o[m][te], _ = fit_lr(
                    lambda prm, a_, b_: dense(prm, tr[a_], tr[b_], **kw),
                    lambda prm: dense(prm, tr, te, **kw), ytr, seed, grid)
            o["words only (TF-IDF LR)"][te], _ = fit_lr(
                lambda prm, a_, b_: (T[tr[a_]], T[tr[b_]]),
                lambda prm: (T[tr], T[te]), ytr, seed, [{"C": c} for c in C_VALUES])
        for m in models:
            res[m].append(roc_auc_score(y, o[m]))
        oofs.append(o)
        print(f"  seed {seed} ({time.time()-t0:.0f}s): " + " | ".join(
            f"{m} {res[m][-1]:.4f}" for m in models), flush=True)

    tw = pd.DataFrame(top_words).groupby("word").agg(rate=("rate", "mean"),
                                                     users=("users", "mean"),
                                                     folds=("fold", "nunique"))
    print("\nwords with the highest sickness rate among training women (used by >= 15; "
          f"base rate {y.mean():.2f}), listed in >= 3 of 5 folds:")
    print(tw[tw.folds >= 3].sort_values("rate", ascending=False).head(25).round(3).to_string())

    print("\n=== OOF AUC (mean over seeds) ===")
    for m in models:
        line = f"  {m:<28} {np.mean(res[m]):.4f}"
        if m.startswith("+"):
            r = [perm_test(y, o[m], o["base"], n=2000, seed=s) for s, o in zip(a.seeds, oofs)]
            line += (f"   Δ {np.mean([x[0] for x in r]):+.4f}  perm p "
                     f"{min(x[1] for x in r):.3f}-{max(x[1] for x in r):.3f}  "
                     f"wins {sum(x[0] > 0 for x in r)}/{len(r)}")
        print(line)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="perignnosis_graph_pooled.pt")
    ap.add_argument("--cohort-csv", default="pooled_cohort.csv")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    main(ap.parse_args())
