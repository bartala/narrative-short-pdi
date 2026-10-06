"""
Univariate AUC of each PDI item, pooled and per cohort, and top-six frequency across folds.

Usage:
    python item_auc_by_cohort.py
"""

import sys; sys.path.insert(0,'.')
import torch, numpy as np, pandas as pd
from collections import Counter
from sklearn.metrics import roc_auc_score
from fair_benchmark import sanitize, EMB, PDIN
d,w=sanitize(torch.load("perignnosis_graph_pooled.pt",map_location="cpu",weights_only=False))
X=d[w].x.numpy()[:,EMB:EMB+PDIN]; y=d[w].y.view(-1).numpy().astype(int)
coh=pd.read_csv(sys.argv[1]).cohort.values.astype(int)
r=pd.read_csv("results_pdi_replacement/item_ranks.csv")
top6=Counter(q for o in r.order for q in o.split()[:6])
print("top-6 frequency across", len(r), "folds:", dict(sorted(top6.items(), key=lambda t:-t[1])))
rows={}
for name,m in [("pooled",np.ones(len(y),bool)),("cohort0",coh==0),("cohort1",coh==1)]:
    a=[roc_auc_score(y[m],X[m,j]) for j in range(PDIN)]
    rows[f"{name} (n={m.sum()}, pos={y[m].sum()})"]=a
t=pd.DataFrame(rows,index=[f"Q{j+1}" for j in range(PDIN)]).round(3)
for c in t: t[c+" rank"]=t[c].rank(ascending=False).astype(int)
print(t.to_string())
