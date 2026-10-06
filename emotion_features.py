"""
Sentence-level emotion and sentiment scores for each narrative.

Usage:
    python emotion_features.py --csv pooled_cohort.csv --out emotion_features.csv
"""

import argparse, os
import numpy as np, pandas as pd, torch

from embed_sentences import sentences

for _v in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
    os.environ.pop(_v, None)

GOEMO = "SamLowe/roberta-base-go_emotions"
SENT = "cardiffnlp/twitter-roberta-base-sentiment-latest"


def score(texts, name, device, multilabel, bs=64, offline=False):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(name, local_files_only=offline)
    m = AutoModelForSequenceClassification.from_pretrained(
        name, local_files_only=offline).to(device).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            b = tok(texts[i:i + bs], padding=True, truncation=True, max_length=128,
                    return_tensors="pt").to(device)
            z = m(**b).logits
            out.append((torch.sigmoid(z) if multilabel else torch.softmax(z, 1)).cpu())
    labels = [m.config.id2label[i].lower() for i in range(m.config.num_labels)]
    return torch.cat(out).numpy(), labels


def main(a):
    if a.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
    df = pd.read_csv(a.csv)
    texts, woman = [], []
    for i, t in enumerate(df["cb_delivery_narrative"].fillna("")):
        ss = sentences(t)
        texts += ss; woman += [i] * len(ss)
    woman = np.array(woman)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{len(df)} narratives, {len(texts)} sentences; device={device}", flush=True)
    ge, ge_lab = score(texts, GOEMO, device, True, offline=a.offline)
    se, se_lab = score(texts, SENT, device, False, offline=a.offline)
    print(f"GoEmotions labels: {len(ge_lab)}; sentiment labels: {se_lab}", flush=True)

    n = len(df)
    feats = {}
    for M, labs, tag in [(ge, ge_lab, "emo"), (se, se_lab, "sent")]:
        mean = np.zeros((n, M.shape[1])); np.add.at(mean, woman, M)
        mean /= np.bincount(woman, minlength=n)[:, None]
        mx = np.zeros((n, M.shape[1])); np.maximum.at(mx, woman, M)
        for j, l in enumerate(labs):
            feats[f"{tag}_mean_{l}"] = mean[:, j]
            feats[f"{tag}_max_{l}"] = mx[:, j]
    neg = se_lab.index("negative")
    is_neg = (se.argmax(1) == neg).astype(float)
    feats["sent_frac_negative"] = np.bincount(woman, is_neg, minlength=n) / np.bincount(woman, minlength=n)
    out = pd.DataFrame(feats)
    out.insert(0, "record_id", df["record_id"])
    out.to_csv(a.out, index=False)
    np.savez_compressed(a.sent_out, woman=woman, goemotions=ge.astype(np.float32),
                        sentiment=se.astype(np.float32), goemo_labels=np.array(ge_lab),
                        sent_labels=np.array(se_lab))
    print(f"saved {a.out} ({out.shape[1] - 1} features) and {a.sent_out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="pooled_cohort.csv")
    ap.add_argument("--out", default="emotion_features.csv")
    ap.add_argument("--sent-out", default="emotion_sentences.npz")
    ap.add_argument("--offline", action="store_true")
    main(ap.parse_args())
