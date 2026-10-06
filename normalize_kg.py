"""
Diagnose and normalize the knowledge graph.

Usage:
    python normalize_kg.py --graph perignnosis_graph.pt --out perignnosis_graph_norm.pt
    python normalize_kg.py --graph perignnosis_graph.pt --diagnose-only
"""

import argparse, re, collections
import numpy as np, torch
from torch_geometric.data import HeteroData

EMB = 768

TYPE_FAMILIES = {
    "Condition": ["condition", "medicalcondition", "healthcondition", "diagnosis", "disease",
                  "illness", "physicalcondition", "psychologicalcondition", "mentalhealthcondition",
                  "medicalissue", "complication", "concern"],
    "Symptom": ["symptom", "sign", "symptoms"],
    "Procedure": ["procedure", "medicalprocedure", "medicalprocess", "surgery", "operation",
                  "method", "process", "intervention"],
    "Treatment": ["treatment", "medicaltreatment", "therapy", "medicalcare", "care", "service"],
    "Medication": ["medication", "drug", "medicine", "medicalsubstance", "substance", "anesthesia"],
    "Device": ["device", "medicaldevice", "equipment", "medicalequipment", "instrument",
               "medicalinstrument", "medicaltool", "medicalsupply", "object", "tool"],
    "BodyPart": ["bodypart", "anatomy", "anatomicalstructure", "organ", "bodyfluid"],
    "Person": ["person", "role", "occupation", "woman_other", "people", "family", "group",
               "staff", "professional"],
    "Place": ["place", "location", "medicalfacility", "facility", "hospital", "room",
              "organization", "transport", "transportation", "vehicle"],
    "Time": ["time", "date", "duration", "timeperiod", "period", "age"],
    "Event": ["event", "medicalevent", "experience", "activity", "action", "plan", "state"],
    "Emotion": ["emotion", "feeling", "mood", "emotionalstate"],
    "Measurement": ["measurement", "quantity", "weight", "length", "ordinal", "medicalmeasurement",
                    "number", "color", "attribute", "gender", "concept", "field", "resource",
                    "legalconcept", "medicalrecord", "record", "animal", "food"],
    "Test": ["test", "medicaltest", "examination", "screening"],
}
REL_FAMILIES = {
    "MENTIONS": ["mentions", "mentioned", "includes", "included", "has", "have"],
    "EXPERIENCED": ["experienced", "experiencedby", "underwent", "felt", "suffered", "hadcondition",
                    "affected", "affectedby", "involved", "involves", "participatedin"],
    "CAUSED": ["cause", "caused", "causes", "resultedin", "resultsin", "leadsto", "ledto", "dueto",
               "triggered", "because"],
    "TREATED_WITH": ["treatedwith", "received", "administered", "used", "usedfor", "given",
                     "performed", "prescribed", "required", "assisted", "attempted"],
    "LOCATED_AT": ["location", "locatedat", "occurredat", "at", "in", "wasat", "admittedto"],
    "TEMPORAL": ["during", "followedby", "before", "after", "duration", "when", "while", "time"],
    "PART_OF": ["partof", "contains", "consistsof", "component", "hascomponent"],
    "ASSOCIATED_WITH": ["associatedwith", "relatedto", "related", "linkedto", "connectedto",
                        "correlatedwith"],
}
TYPE_LOOKUP = {v: fam for fam, vals in TYPE_FAMILIES.items() for v in vals}
REL_LOOKUP = {v: fam for fam, vals in REL_FAMILIES.items() for v in vals}


def key(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def canon_type(t):
    if t in ("Woman",):
        return "Woman"
    return TYPE_LOOKUP.get(key(t), "Entity")


def canon_rel(r):
    return REL_LOOKUP.get(key(r), "ASSOCIATED_WITH")


def norm_surface(s):
    s = str(s).lower().strip()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"s$", "", s) if len(s) > 4 else s
    return s


def diagnose(raw, tag=""):
    print(f"\n=== KG diagnostics {tag} ===")
    n_nodes = sum(raw[n].num_nodes for n in raw.node_types)
    n_edges = sum(raw[e].edge_index.size(1) for e in raw.edge_types)
    print(f"nodes={n_nodes} node_types={len(raw.node_types)} "
          f"edges={n_edges} edge_types={len(raw.edge_types)}")
    nW = raw["Woman"].num_nodes
    deg = torch.zeros(nW)
    for (s, r, d) in raw.edge_types:
        ei = raw[s, r, d].edge_index
        if s == "Woman":
            deg += torch.bincount(ei[0], minlength=nW).float()
        if d == "Woman":
            deg += torch.bincount(ei[1], minlength=nW).float()
    print(f"edges per woman: mean {deg.mean():.1f}  median {deg.median():.0f}  "
          f"min {deg.min():.0f}  max {deg.max():.0f}  women with none: {int((deg == 0).sum())}")
    sizes = sorted(((nt, raw[nt].num_nodes) for nt in raw.node_types), key=lambda x: -x[1])
    print("largest node types:", ", ".join(f"{n}={c}" for n, c in sizes[:8]))
    rel_counts = collections.Counter()
    for (s, r, d) in raw.edge_types:
        rel_counts[r] += raw[s, r, d].edge_index.size(1)
    print("most frequent relations:", ", ".join(f"{r}={c}" for r, c in rel_counts.most_common(8)))
    singleton_rels = sum(1 for r, c in rel_counts.items() if c <= 2)
    print(f"relation types used <=2 times: {singleton_rels} of {len(rel_counts)}")
    return deg


def build_type_index(raw):
    idx = {}
    for nt in raw.node_types:
        if nt in ("Woman", "__Entity__"):
            continue
        fam = canon_type(nt)
        for v in raw[nt].x:
            idx[hash(np.round(v[:16].numpy(), 5).tobytes())] = fam
    return idx


def normalize(raw, retype=True):
    type_index = build_type_index(raw) if retype else {}
    fams, feats, mapping = {}, {}, {}
    for nt in raw.node_types:
        if nt == "Woman":
            continue
        x = raw[nt].x
        for i in range(x.size(0)):
            v = x[i]
            h = hash(np.round(v[:16].numpy(), 5).tobytes())
            fam = type_index.get(h, canon_type(nt)) if nt == "__Entity__" else canon_type(nt)
            k = (fam, h)
            if k not in fams:
                fams[k] = len(feats.setdefault(fam, []))
                feats[fam].append(v)
            mapping[(nt, i)] = (fam, fams[k])

    data = HeteroData()
    data["Woman"].x = raw["Woman"].x
    data["Woman"].y = raw["Woman"].y
    for fam, vs in feats.items():
        data[fam].x = torch.stack(vs)

    buckets = collections.defaultdict(set)
    for (s, r, d) in raw.edge_types:
        ei = raw[s, r, d].edge_index
        rel = canon_rel(r)
        for a, b in zip(ei[0].tolist(), ei[1].tolist()):
            if s == "Woman":
                sf, si = "Woman", a
            else:
                if (s, a) not in mapping:
                    continue
                sf, si = mapping[(s, a)]
            if d == "Woman":
                df, di = "Woman", b
            else:
                if (d, b) not in mapping:
                    continue
                df, di = mapping[(d, b)]
            buckets[(sf, rel, df)].add((si, di))
    for k, pairs in buckets.items():
        idx = torch.tensor(sorted(pairs), dtype=torch.long).t().contiguous()
        data[k].edge_index = idx
    return data


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="perignnosis_graph.pt")
    ap.add_argument("--out", default="perignnosis_graph_norm.pt")
    ap.add_argument("--diagnose-only", action="store_true")
    ap.add_argument("--no-retype", action="store_true",
                    help="keep __Entity__ as one generic type instead of restoring clinical types")
    a = ap.parse_args()

    raw = torch.load(a.graph, map_location="cpu", weights_only=False)
    diagnose(raw, "(original)")
    if a.diagnose_only:
        raise SystemExit
    norm = normalize(raw, retype=not a.no_retype)
    diagnose(norm, "(normalized)")
    torch.save(norm, a.out)
    print(f"\nsaved {a.out}")
    print("next:  python fair_benchmark.py --graph", a.out, "--seeds 1 2 3 4 5 --out results_fair_norm/")
