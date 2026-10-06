"""
Sentence-level embeddings of each narrative.

Usage:
    python embed_sentences.py --csv pooled_cohort.csv --out narrative_sentences.npz
"""

import argparse, os, re
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np, pandas as pd, torch

SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def sentences(text, min_words=3, max_words=80):
    out, carry = [], ""
    for s in SPLIT.split(str(text)):
        s = (carry + " " + s).strip() if carry else s.strip()
        if not s:
            continue
        if len(s.split()) < min_words:
            carry = s
            continue
        carry = ""
        w = s.split()
        out += [" ".join(w[i:i + max_words]) for i in range(0, len(w), max_words)]
    if carry:
        if out:
            out[-1] += " " + carry
        else:
            out.append(carry)
    return out or [str(text)]


def main(a):
    from sentence_transformers import SentenceTransformer
    df = pd.read_csv(a.csv)
    texts, woman, pos = [], [], []
    for i, t in enumerate(df["cb_delivery_narrative"].fillna("")):
        ss = sentences(t)
        texts += ss; woman += [i] * len(ss); pos += list(range(len(ss)))
    n_s = np.bincount(woman, minlength=len(df))
    print(f"{len(df)} narratives -> {len(texts)} sentences; per narrative median "
          f"{np.median(n_s):.0f}, max {n_s.max()}", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = SentenceTransformer(a.model, trust_remote_code=True, device=device)
    emb = m.encode(texts, batch_size=128, convert_to_numpy=True,
                   normalize_embeddings=True, show_progress_bar=False)
    np.savez_compressed(a.out, emb=emb.astype(np.float32), woman=np.array(woman),
                        pos=np.array(pos),
                        n_words=np.array([len(t.split()) for t in texts]))
    print(f"saved {a.out}: {emb.shape}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="pooled_cohort.csv")
    ap.add_argument("--out", default="narrative_sentences.npz")
    ap.add_argument("--model", default="jinaai/jina-embeddings-v2-base-en")
    main(ap.parse_args())
