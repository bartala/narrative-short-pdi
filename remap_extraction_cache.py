"""
Reuse cached extractions under pooled record identifiers.

Usage:
    python remap_extraction_cache.py --in rebuild_extraction.jsonl --out pooled_extraction.jsonl --prefix cbex
"""

import argparse, json, os

p = argparse.ArgumentParser()
p.add_argument("--in", dest="inp", default="rebuild_extraction.jsonl")
p.add_argument("--out", default="pooled_extraction.jsonl")
p.add_argument("--prefix", default="cbex")
a = p.parse_args()

if os.path.exists(a.out):
    raise SystemExit(f"{a.out} already exists - refusing to overwrite. Delete it "
                     f"first if you really want to rebuild the cache.")

n = 0
with open(a.inp) as fi, open(a.out, "w") as fo:
    for line in fi:
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        rid = str(r["record_id"])
        r["record_id"] = rid if rid.startswith(a.prefix + "_") else f"{a.prefix}_{rid}"
        fo.write(json.dumps(r) + "\n")
        n += 1

empties = 0
with open(a.out) as f:
    for line in f:
        if not json.loads(line)["nodes"]:
            empties += 1
print(f"remapped {n} extractions -> {a.out} ({empties} of them yielded no entities)")
print(f"the original {a.inp} is unchanged")
