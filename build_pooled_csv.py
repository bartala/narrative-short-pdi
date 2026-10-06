"""
Merge the two cohorts into one analysis CSV (complete cases, narratives of 30+ words).

Usage:
    python build_pooled_csv.py --cbex CBEX.csv --covid "Delivery Narratives COVID-19_MSB.csv" --out pooled_cohort.csv
"""

import argparse, re
import pandas as pd

PDI = [f"pdi_q{i}" for i in range(1, 14)]


def prep(path, tag, args):
    df = pd.read_csv(path)
    missing = [c for c in [args.id_col, args.text_col, args.pcl_col] + PDI
               if c not in df.columns]
    if missing:
        raise SystemExit(f"{path}: missing columns {missing}")
    for c in PDI + [args.pcl_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    n0 = len(df)
    words = df[args.text_col].fillna("").astype(str).str.split().str.len()
    keep = (df[args.text_col].notna() & (words >= args.min_words)
            & df[PDI].notna().all(axis=1) & df[args.pcl_col].notna())
    out = df.loc[keep, [args.id_col, args.text_col] + PDI + [args.pcl_col]].copy()
    out[args.id_col] = tag + "_" + out[args.id_col].astype(str)
    out["cohort"] = 0 if tag == "cbex" else 1
    out["label"] = (out[args.pcl_col] >= args.cutoff).astype(int)
    print(f"{tag}: {n0} rows -> {len(out)} included "
          f"({n0 - len(out)} excluded: short narrative / incomplete PDI / no PCL-5); "
          f"{int(out.label.sum())} positive ({out.label.mean():.1%}); "
          f"median narrative {int(words[keep].median())} words")
    return out


def main(a):
    cb = prep(a.cbex, "cbex", a)
    cv = prep(a.covid, "covid", a)
    pooled = pd.concat([cb, cv], ignore_index=True)
    assert pooled[a.id_col].is_unique, "record_id collision after prefixing"

    print(f"\npooled: {len(pooled)} women, {int(pooled.label.sum())} positive "
          f"({pooled.label.mean():.1%})")
    print(pooled.groupby("cohort").agg(n=("label", "size"),
                                       positive=("label", "sum"),
                                       rate=("label", "mean")).round(3).to_string())
    print("\nPDI item means by cohort (a large gap here means cohort is doing work "
          "the narrative cannot be credited with):")
    print(pooled.groupby("cohort")[PDI].mean().round(2).to_string())

    pooled["comp_placeholder"] = 0.0
    cols = [a.id_col, a.text_col] + PDI + ["comp_placeholder", "cohort", a.pcl_col]
    pooled[cols].to_csv(a.out, index=False)
    print(f"\nsaved {a.out}  (models see 768 narrative + 13 PDI; cohort is kept "
          f"in the file for post-hoc reporting only, not as an input)")
    print("next:")
    print(f"  python remap_extraction_cache.py   # reuse the 301 CBEx extractions")
    print(f"  python rebuild_dev_kg.py --csv {a.out} --old-graph \"\" "
          f"--comp-col comp_placeholder --cache pooled_extraction.jsonl "
          f"--out perignnosis_graph_pooled.pt")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cbex", required=True)
    p.add_argument("--covid", required=True)
    p.add_argument("--out", default="pooled_cohort.csv")
    p.add_argument("--id-col", default="record_id")
    p.add_argument("--text-col", default="cb_delivery_narrative")
    p.add_argument("--pcl-col", default="spcl5_total")
    p.add_argument("--cutoff", type=int, default=32)
    p.add_argument("--min-words", type=int, default=30)
    main(p.parse_args())
