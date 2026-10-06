"""
Sensitivity analysis: alternative narrative embedding models in the logistic regression.

Usage:
    python embedding_benchmark.py embed --models all
    python embedding_benchmark.py evaluate --seeds 1 2 3 4 5
"""

import argparse, gc, os, time
import numpy as np, pandas as pd, torch

CANDIDATES = {
    "jina-v2-base (Mark 1)": ("jinaai/jina-embeddings-v2-base-en", 8192, "", True),
    "MiniLM-L6": ("sentence-transformers/all-MiniLM-L6-v2", 256, "", False),
    "mpnet-base": ("sentence-transformers/all-mpnet-base-v2", 384, "", False),
    "bge-base": ("BAAI/bge-base-en-v1.5", 512, "", False),
    "bge-large": ("BAAI/bge-large-en-v1.5", 512, "", False),
    "e5-large-v2": ("intfloat/e5-large-v2", 512, "passage: ", False),
    "gte-large-v1.5": ("Alibaba-NLP/gte-large-en-v1.5", 8192, "", True),
    "Qwen3-Embedding-0.6B": ("Qwen/Qwen3-Embedding-0.6B", 8192, "", False),
    "Qwen3-Embedding-4B": ("Qwen/Qwen3-Embedding-4B", 8192, "", False),
}
CACHE = "narrative_embeddings"


def slug(name):
    return "".join(c if c.isalnum() else "_" for c in name)


def chunks(text, max_tokens):
    words = str(text).split()
    size = max(32, int(max_tokens * 0.7))
    if len(words) <= size:
        return [" ".join(words)]
    step = int(size * 0.75)
    return [" ".join(words[i:i + size]) for i in range(0, len(words) - size // 4, step)]


def embed(a):
    from sentence_transformers import SentenceTransformer
    os.makedirs(CACHE, exist_ok=True)
    texts = pd.read_csv(a.cohort_csv).cb_delivery_narrative.fillna("").values
    names = list(CANDIDATES) if a.models == ["all"] else a.models
    for name in names:
        path = os.path.join(CACHE, slug(name) + ".npy")
        if os.path.exists(path):
            print(f"{name}: cached"); continue
        hf, max_tok, prefix, trust = CANDIDATES[name]
        t0 = time.time()
        if name.startswith("jina"):
            from fair_benchmark import EMB, sanitize
            d, w = sanitize(torch.load(a.graph, map_location="cpu", weights_only=False))
            np.save(path, d[w].x.numpy()[:, :EMB].astype(np.float32))
            print(f"{name}: taken from the graph (Mark 1 embedding)", flush=True)
            continue
        kw = {"model_kwargs": {"torch_dtype": torch.float16}} if "Qwen3" in name else {}
        m = SentenceTransformer(hf, trust_remote_code=trust, device="cuda", **kw)
        m.max_seq_length = min(max_tok, 4096)
        parts, owner = [], []
        for i, t in enumerate(texts):
            cs = chunks(t, max_tok)
            parts += [prefix + c for c in cs]; owner += [i] * len(cs)
        bs = 2 if "4B" in name else (4 if max_tok > 1024 else 32)
        V = m.encode(parts, batch_size=bs, convert_to_numpy=True, normalize_embeddings=True,
                     show_progress_bar=False).astype(np.float32)
        owner = np.array(owner)
        E = np.zeros((len(texts), V.shape[1]), np.float32)
        np.add.at(E, owner, V)
        E /= np.linalg.norm(E, axis=1, keepdims=True).clip(1e-9)
        np.save(path, E)
        print(f"{name}: {E.shape[1]} dims, {len(parts)} chunks for {len(texts)} narratives "
              f"({time.time()-t0:.0f}s)", flush=True)
        del m; gc.collect(); torch.cuda.empty_cache()


def evaluate(a):
    from sklearn.decomposition import PCA
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from fair_benchmark import EMB, PDIN, sanitize, set_seed
    from pca_narrative_diagnostic import fit_predict, perm_test, C_VALUES, K_GRID
    from graph_signal_diagnostic import pipe
    from narrative_short_pdi import SIX

    d, w = sanitize(torch.load(a.graph, map_location="cpu", weights_only=False))
    X0 = d[w].x.numpy()[:, :EMB + PDIN].astype(np.float64)
    y = d[w].y.view(-1).numpy().astype(int)
    coh = pd.read_csv(a.cohort_csv).cohort.values.astype(int)
    strata = y * 2 + coh
    pdi = X0[:, EMB:]
    names = [n for n in CANDIDATES if os.path.exists(os.path.join(CACHE, slug(n) + ".npy"))]
    rows, oof = [], {}
    for name in names:
        E = np.load(os.path.join(CACHE, slug(name) + ".npy")).astype(np.float64)
        X = np.c_[E, pdi]
        dim = E.shape[1]
        ecols = list(range(dim))
        six = [dim + q - 1 for q in SIX]
        all13 = list(range(dim, dim + PDIN))
        pca = ("pca", PCA(10, random_state=0), ecols)
        g = {"clf__C": C_VALUES, "ct__pca__n_components": K_GRID}
        specs = {"six": [pca, ("pdi", "passthrough", six)],
                 "13": [pca, ("pdi", "passthrough", all13)],
                 "narrative only": [pca]}
        t0 = time.time()
        for form, parts in specs.items():
            for seed in a.seeds:
                set_seed(seed)
                p = np.zeros(len(y))
                for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, strata):
                    p[te], _ = fit_predict(pipe(parts), g, X, y, tr, te, seed)
                oof[(name, form, seed)] = p
                rows.append({"model": name, "form": form, "seed": seed,
                             "auc": roc_auc_score(y, p)})
        r = pd.DataFrame(rows)
        r = r[r.model == name].groupby("form").auc.mean()
        print(f"  {name:<24} six {r['six']:.4f} | 13 {r['13']:.4f} | narrative only "
              f"{r['narrative only']:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("embedding_benchmark.csv", index=False)
    ref = "jina-v2-base (Mark 1)"
    print("\n=== mean OOF AUC over seeds; Δ vs jina (Mark 1), paired permutation per seed ===")
    for form in ["six", "13", "narrative only"]:
        print(f"\n  form: {form}")
        for name in names:
            v = df[(df.model == name) & (df.form == form)].auc.mean()
            line = f"    {name:<24} {v:.4f}"
            if name != ref:
                r = [perm_test(y, oof[(name, form, s)], oof[(ref, form, s)], n=1000, seed=s)
                     for s in a.seeds]
                line += (f"   Δ {np.mean([x[0] for x in r]):+.4f}  p {min(x[1] for x in r):.3f}-"
                         f"{max(x[1] for x in r):.3f}  wins {sum(x[0] > 0 for x in r)}/{len(r)}")
            print(line)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["embed", "evaluate"])
    ap.add_argument("--models", nargs="+", default=["all"])
    ap.add_argument("--graph", default="perignnosis_graph_pooled.pt")
    ap.add_argument("--cohort-csv", default="pooled_cohort.csv")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    a = ap.parse_args()
    embed(a) if a.step == "embed" else evaluate(a)
