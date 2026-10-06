"""
Graph centrality features added to logistic regression on narrative + PDI.

Usage:
    python centrality_probe.py --graph perignnosis_graph_pooled.pt --seeds 1 2 3 4 5
"""

import argparse, os, time
import numpy as np, pandas as pd, torch, networkx as nx
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from fair_benchmark import EMB, PDIN, sanitize, set_seed
from pca_narrative_diagnostic import fit_predict, perm_test, C_VALUES, K_GRID, EMB_COLS, PDI_COLS
from graph_signal_diagnostic import pipe


def centralities(data, woman):
    offs, n = {}, 0
    for nt in data.node_types:
        offs[nt] = n; n += data[nt].num_nodes
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for (s, _, t) in data.edge_types:
        ei = data[s, _, t].edge_index.numpy()
        G.add_edges_from(zip(ei[0] + offs[s], ei[1] + offs[t]))
    G.remove_edges_from(nx.selfloop_edges(G))
    t0 = time.time()
    c = {"degree": dict(G.degree()),
         "betweenness": nx.betweenness_centrality(G),
         "closeness": nx.closeness_centrality(G),
         "harmonic": nx.harmonic_centrality(G),
         "pagerank": nx.pagerank(G),
         "eigenvector": nx.eigenvector_centrality_numpy(G),
         "core": nx.core_number(G)}
    print(f"graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges; "
          f"centralities in {time.time()-t0:.0f}s", flush=True)
    nw = data[woman].num_nodes
    W = list(range(offs[woman], offs[woman] + nw))
    feats = {f"woman_{k}": np.array([v[i] for i in W], float) for k, v in c.items()}
    for k in ["betweenness", "pagerank", "degree"]:
        mean, mx = np.zeros(nw), np.zeros(nw)
        for j, i in enumerate(W):
            nb = [u for u in G.neighbors(i) if not offs[woman] <= u < offs[woman] + nw]
            if nb:
                vals = np.array([c[k][u] for u in nb])
                mean[j], mx[j] = vals.mean(), vals.max()
        feats[f"entities_mean_{k}"], feats[f"entities_max_{k}"] = mean, mx
    F = pd.DataFrame(feats)
    return np.log1p(F.clip(lower=0) * np.where(F.max() < 1, 1000, 1)), F


def main(a):
    d, w = sanitize(torch.load(a.graph, map_location="cpu", weights_only=False))
    X0 = d[w].x.numpy()[:, :EMB + PDIN].astype(np.float64)
    y = d[w].y.view(-1).numpy().astype(int)
    meta = pd.read_csv(a.cohort_csv)
    coh = meta.cohort.values.astype(int)
    wc = np.log(meta.cb_delivery_narrative.fillna("").str.split().str.len().clip(lower=1).values)
    C, raw = centralities(d, w)
    raw.assign(y=y, log_words=wc).to_csv(a.out_csv, index=False)
    print("\nSpearman correlation with the outcome and with log word count:")
    from scipy.stats import spearmanr
    for col in C.columns:
        print(f"  {col:<28} outcome {spearmanr(C[col], y)[0]:+.3f}   words {spearmanr(C[col], wc)[0]:+.3f}")

    X = np.c_[X0, C.values, wc]
    cen = list(range(EMB + PDIN, EMB + PDIN + C.shape[1]))
    wcc = [X.shape[1] - 1]
    pca = ("pca", PCA(10, random_state=0), EMB_COLS)
    pdi = ("pdi", "passthrough", PDI_COLS)
    g_p = {"clf__C": C_VALUES, "ct__pca__n_components": K_GRID}
    specs = {"base": ([pca, pdi], g_p),
             "+ centralities": ([pca, pdi, ("c", "passthrough", cen)], g_p),
             "+ word count": ([pca, pdi, ("w", "passthrough", wcc)], g_p),
             "centralities only": ([("c", "passthrough", cen)], {"clf__C": C_VALUES})}
    strata = y * 2 + coh
    res, oofs = {m: [] for m in specs}, []
    for seed in a.seeds:
        set_seed(seed)
        o = {m: np.zeros(len(y)) for m in specs}
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, strata):
            for m, (parts, g) in specs.items():
                o[m][te], _ = fit_predict(pipe(parts), g, X, y, tr, te, seed)
        for m in specs:
            res[m].append(roc_auc_score(y, o[m]))
        oofs.append(o)
        print(f"  seed {seed}: " + " | ".join(f"{m} {res[m][-1]:.4f}" for m in specs), flush=True)
    print("\n=== OOF AUC ===")
    for m in specs:
        line = f"  {m:<20} {np.mean(res[m]):.4f}"
        if m in ("+ centralities", "+ word count"):
            r = [perm_test(y, o[m], o["base"], n=2000, seed=s) for s, o in zip(a.seeds, oofs)]
            line += (f"   Δ {np.mean([x[0] for x in r]):+.4f}  perm p "
                     f"{min(x[1] for x in r):.3f}-{max(x[1] for x in r):.3f}"
                     f"  wins {sum(x[0] > 0 for x in r)}/{len(r)}")
        print(line)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="perignnosis_graph_pooled.pt")
    ap.add_argument("--cohort-csv", default="pooled_cohort.csv")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--out-csv", default="woman_centralities.csv")
    main(ap.parse_args())
