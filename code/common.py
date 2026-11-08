import json, os
from pathlib import Path

# All models are cached on Jubail; run fully offline so no step depends on the network or on the (expired)
# Hugging Face token. Must be set before transformers / vLLM are imported (scripts import common first).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["pubhealth", "climatefever", "finfact"]
ATTACK_ORDER = ["CLEAN", "TR_fr", "TR_es", "TR_ar", "TR_hi", "TR_zh", "SUM", "MCQ", "XML", "JSON",
                "TR>MCQ", "MCQ>JSON", "MCQ>XML", "TR>XML", "SUM>JSON", "TR>MCQ>JSON"]


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(p, rows):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def claims(ds, split=None):
    rows = read_jsonl(ROOT / f"data/processed/{ds}/claims.jsonl")
    return [r for r in rows if split is None or r["split"] == split]


def kb(ds):
    return read_jsonl(ROOT / f"data/processed/{ds}/kb.jsonl")


def attacks(ds):
    return read_jsonl(ROOT / f"data/attacks/{ds}.jsonl")


def choose_threshold(y_dev, p_dev):
    """Decision threshold maximising macro-F1 on clean dev; candidates = 0.5 and the 1..99% quantiles of p_dev;
    ties are broken towards 0.5. Prediction rule everywhere: false (1) iff p >= threshold."""
    y_dev, p_dev = np.asarray(y_dev), np.asarray(p_dev, dtype=float)
    grid = np.unique(np.concatenate([[0.5], np.quantile(p_dev, np.linspace(0.01, 0.99, 99))]))
    return float(max(grid, key=lambda t: (metrics(y_dev, (p_dev >= t).astype(int))["macro_f1"], -abs(t - 0.5))))


def yes_prob_from_logprobs(top):
    """top: iterable of (decoded_token, logprob) for the first generated position.
    Returns P(yes) / (P(yes) + P(no)) over the listed tokens (case/quote/space-insensitive), or 0.5 if neither."""
    import math
    y = n = 0.0
    for tok, lp in top:
        s = (tok or "").strip().strip("'\"").lower()
        if s in ("yes", "y", "ye"):
            y += math.exp(lp)
        elif s in ("no", "n"):
            n += math.exp(lp)
    return y / (y + n) if y + n > 0 else 0.5


def check_preds(ds, rows):
    """Integrity check for a prediction file: one row per (id, condition), gold labels, full coverage."""
    gold = {r["id"]: r["label"] for r in claims(ds)}
    keys = [(r["id"], r["attack"]) for r in rows]
    assert len(keys) == len(set(keys)), f"{ds}: duplicate (id, condition) rows"
    bad = [k for k, r in zip(keys, rows) if r["label"] != gold[r["id"]]]
    assert not bad, f"{ds}: {len(bad)} rows with a label different from the benchmark, e.g. {bad[:3]}"
    expect = {(r["id"], r["attack"]) for r in attacks(ds)} | {(c["id"], "DEV_CLEAN") for c in claims(ds, "dev")}
    missing = expect - set(keys)
    assert not missing, f"{ds}: {len(missing)} (id, condition) pairs missing, e.g. {sorted(missing)[:3]}"
    assert all(r["pred"] in (0, 1) for r in rows), f"{ds}: non-binary predictions"


def metrics(y, p):
    """y, p: 1 = false information (positive class), 0 = true."""
    y, p = np.asarray(y), np.asarray(p)
    tp = int(((y == 1) & (p == 1)).sum()); fn = int(((y == 1) & (p == 0)).sum())
    tn = int(((y == 0) & (p == 0)).sum()); fp = int(((y == 0) & (p == 1)).sum())
    f1_pos = 2 * tp / max(1, 2 * tp + fp + fn)
    f1_neg = 2 * tn / max(1, 2 * tn + fn + fp)
    return dict(n=len(y), acc=(tp + tn) / max(1, len(y)), macro_f1=(f1_pos + f1_neg) / 2,
                false_acc=tp / max(1, tp + fn), true_acc=tn / max(1, tn + fp),
                asr=fn / max(1, tp + fn))  # ASR: share of false items that get through as "true"
