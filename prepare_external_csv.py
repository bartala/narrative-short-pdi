"""
Prepare the second-cohort CSV for graph construction.

Usage:
    python prepare_external_csv.py --in "Delivery Narratives COVID-19_MSB.csv" --out external_cohort.csv
"""

import argparse, re
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", required=True)
ap.add_argument("--out", default="external_cohort.csv")
ap.add_argument("--text-col", default="cb_delivery_narrative")
ap.add_argument("--pcl-col", default="spcl5_total")
ap.add_argument("--cutoff", type=int, default=32)
ap.add_argument("--min-words", type=int, default=30)
a = ap.parse_args()

df = pd.read_csv(a.inp)
pdi = sorted([c for c in df.columns if re.fullmatch(r"pdi_q?\d+", c)],
             key=lambda c: int(re.findall(r"\d+", c)[-1]))
assert len(pdi) == 13, f"expected 13 PDI columns, found {pdi}"
for c in pdi + [a.pcl_col]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

n0 = len(df)
df["word_count"] = df[a.text_col].fillna("").astype(str).str.split().str.len()
keep = (df[a.text_col].notna() & (df.word_count >= a.min_words)
        & df[pdi].notna().all(axis=1) & df[a.pcl_col].notna())
out = df.loc[keep, ["record_id", a.text_col] + pdi + [a.pcl_col, "word_count"]].copy()
out["label"] = (out[a.pcl_col] >= a.cutoff).astype(int)

print(f"input rows: {n0}")
print(f"excluded  : {n0 - len(out)} (short narrative / incomplete PDI / missing PCL-5)")
print(f"included  : {len(out)}  positives: {out.label.sum()} ({out.label.mean():.1%})")
print(f"median narrative length: {int(out.word_count.median())} words")
out.drop(columns=["word_count"]).to_csv(a.out, index=False)
print("saved", a.out)

exc = df.loc[~keep]
if len(exc):
    print("\nexcluded-vs-included comparison (for the selection-bias paragraph):")
    for col in [a.pcl_col] + pdi[:3]:
        if col in exc.columns:
            print(f"  {col}: excluded mean {exc[col].mean():.2f} (n={exc[col].notna().sum()}) | "
                  f"included mean {out[col].mean():.2f}")
