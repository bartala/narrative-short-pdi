"""
Woman-word graph with word-node embeddings.

Usage:
    python word_embed_probe.py --seeds 1 2 3 4 5
"""

import argparse, os, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np, pandas as pd, torch
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize

from fair_benchmark import EMB, PDIN, sanitize, set_seed
from pca_narrative_diagnostic import fit_predict, perm_test, C_VALUES, EMB_COLS, PDI_COLS
from graph_signal_diagnostic import pipe
from word_graph_probe import vectorizer

_cache = {}


def word_vectors(terms, model):
    new = [t for t in terms if t not in _cache]
    if new:
        v = model.encode(new, batch_size=256, convert_to_numpy=True,
                         normalize_embeddings=True, show_progress_bar=False)
        _cache.update(zip(new, v))
    return np.stack([_cache[t] for t in terms])


def main(a):
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = SentenceTransformer("jinaai/jina-embeddings-v2-base-en", trust_remote_code=True,
                              device=device)
    d, w = sanitize(torch.load(a.graph, map_location="cpu", weights_only=False))
    X0 = d[w].x.numpy()[:, :EMB + PDIN].astype(np.float64)
    y = d[w].y.view(-1).numpy().astype(int)
    meta = pd.read_csv(a.cohort_csv)
    coh = meta.cohort.values.astype(int)
    texts = meta.cb_delivery_narrative.fillna("").values
    strata = y * 2 + coh
    n = len(y)
    print(f"{n} women, {y.sum()} positive", flush=True)

    B = list(range(EMB + PDIN))
    pca = ("pca", PCA(10, random_state=0), EMB_COLS)
    pdi = ("pdi", "passthrough", PDI_COLS)
    g = {"clf__C": C_VALUES, "ct__pca__n_components": [10, 20, 50]}
    models = ["base", "+ SVD", "+ 1-hop word vectors", "+ 2-hop propagation",
              "+ SVD + 2-hop"]
    res, oofs = {m: [] for m in models}, []
    for seed in a.seeds:
        set_seed(seed)
        t0 = time.time()
        o = {m: np.zeros(n) for m in models}
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X0, strata):
            vec = vectorizer().fit(texts[tr])
            T = vec.transform(texts).tocsr()
            Wn = normalize(T, norm="l1", axis=1)
            terms = list(vec.get_feature_names_out())
            E = word_vectors(terms, enc)
            H1 = np.asarray(Wn @ E)
            Bin = (T > 0).astype(float)
            Wc = normalize(Bin, norm="l1", axis=0)
            Wnode = np.asarray(Wc.T @ X0[:, EMB_COLS])
            H2 = np.asarray(Wn @ Wnode)
            X = np.c_[X0, H1, H2, T.toarray()]
            c1 = list(range(EMB + PDIN, EMB + PDIN + EMB))
            c2 = list(range(c1[-1] + 1, c1[-1] + 1 + EMB))
            ct = list(range(c2[-1] + 1, X.shape[1]))
            svd = ("svd", TruncatedSVD(50, random_state=0), ct)
            h1 = ("h1", PCA(10, random_state=0), c1)
            h2 = ("h2", PCA(10, random_state=0), c2)
            specs = {
                "base": ([pca, pdi], g),
                "+ SVD": ([pca, pdi, svd], g | {"ct__svd__n_components": [20, 50, 100]}),
                "+ 1-hop word vectors": ([pca, pdi, h1], g | {"ct__h1__n_components": [5, 10, 20]}),
                "+ 2-hop propagation": ([pca, pdi, h2], g | {"ct__h2__n_components": [5, 10, 20]}),
                "+ SVD + 2-hop": ([pca, pdi, svd, h2], g | {"ct__svd__n_components": [20, 50, 100],
                                                           "ct__h2__n_components": [5, 10, 20]}),
            }
            for m, (parts, gg) in specs.items():
                o[m][te], _ = fit_predict(pipe(parts), gg, X, y, tr, te, seed)
        for m in models:
            res[m].append(roc_auc_score(y, o[m]))
        oofs.append(o)
        print(f"  seed {seed} ({time.time()-t0:.0f}s): " + " | ".join(
            f"{m} {res[m][-1]:.4f}" for m in models), flush=True)

    print("\n=== OOF AUC (mean over seeds) ===")
    for m in models:
        line = f"  {m:<24} {np.mean(res[m]):.4f}"
        if m != "base":
            r = [perm_test(y, o[m], o["base"], n=2000, seed=s) for s, o in zip(a.seeds, oofs)]
            line += (f"   Δ vs base {np.mean([x[0] for x in r]):+.4f}  perm p "
                     f"{min(x[1] for x in r):.3f}-{max(x[1] for x in r):.3f}  "
                     f"wins {sum(x[0] > 0 for x in r)}/{len(r)}")
        if m not in ("base", "+ SVD"):
            r = [perm_test(y, o[m], o["+ SVD"], n=2000, seed=s) for s, o in zip(a.seeds, oofs)]
            line += (f" | vs SVD {np.mean([x[0] for x in r]):+.4f} "
                     f"({sum(x[0] > 0 for x in r)}/{len(r)})")
        print(line)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="perignnosis_graph_pooled.pt")
    ap.add_argument("--cohort-csv", default="pooled_cohort.csv")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    main(ap.parse_args())
