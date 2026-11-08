"""Evidence-based veracity classifier of Kotonya & Toni (EMNLP 2020), PUBHEALTH's best published model:
S-BERT ranks evidence sentences by cosine similarity to the claim, the top k = 5 are kept, and SciBERT is fine-tuned
on (claim, selected evidence). Hyper-parameters follow the authors' protocol (appendix): grid search on the dev set over
batch {8,16,32} x lr {1e-5,5e-6,1e-6} x epochs {2,3,4}; they report batch 16, lr 1e-6, 4 epochs for PUBHEALTH. We run
the same grid on each dataset's own dev set (--grid), then train three seeds with the selected setting.
Their code release contains data scripts only, so the model is implemented from the paper.

Adaptation to the train-only-archive protocol (stated in the paper): the original selects sentences from the claim's
OWN fact-check article; test claims' articles are not available here, so sentences are ranked over all evidence
passages of the training archive, and for a training claim its own passages are excluded so that training matches
test conditions. S-BERT model: the original Sentence-BERT (bert-base-nli-mean-tokens, Reimers & Gurevych 2019).

usage: python kotonya_run.py --dataset pubhealth --grid          (once per dataset)
       python kotonya_run.py --dataset pubhealth [--seed 13]   (uses data/kotonya/<dataset>.best.json)
"""
import argparse, json, re
import numpy as np
import torch
from common import ROOT, claims, attacks, kb, read_jsonl, write_jsonl, metrics, choose_threshold, check_preds

SBERT = "sentence-transformers/bert-base-nli-mean-tokens"
SCIBERT = "allenai/scibert_scivocab_uncased"
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def split_sentences(text):
    return [s.strip() for s in _SENT.split(text or "") if len(s.strip().split()) >= 3]


def build_sentence_pool(ds):
    """sentences of the archive's evidence passages, with the claim id they belong to"""
    sents, owner = [], []
    for r in kb(ds):
        if r["type"] != "evidence":
            continue
        for s in split_sentences(r["text"]):
            sents.append(s); owner.append(r["claim_id"])
    return sents, np.array(owner)


def select_evidence(queries, query_owner, sents, owner, model, k=5, bs=512):
    """top-k sentences by cosine similarity; sentences of the query's own claim (training claims) are excluded"""
    S = model.encode(sents, batch_size=256, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=True)
    Qe = model.encode(queries, batch_size=256, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=True)
    owner_idx = {}
    for i, o in enumerate(owner):
        owner_idx.setdefault(o, []).append(i)
    out = []
    for s in range(0, len(queries), bs):
        sim = Qe[s:s + bs] @ S.T
        for j in range(sim.shape[0]):
            own = owner_idx.get(query_owner[s + j])
            if own:
                sim[j, own] = -2.0
        top = torch.topk(sim, k, dim=1).indices.cpu().numpy()
        out += [[int(t) for t in row] for row in top]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--grid", action="store_true", help="run the authors' dev grid search and store the best setting")
    ap.add_argument("--aug", action="store_true", help="Setting B: add the transformed training claims (data/train_aug)")
    a = ap.parse_args()
    assert not (a.grid and a.aug), "the grid search selects hyper-parameters on clean training data only"
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments,
                              DataCollatorWithPadding, set_seed)
    from sentence_transformers import SentenceTransformer
    set_seed(a.seed)
    tr, dv, te = claims(a.dataset, "train"), claims(a.dataset, "dev"), attacks(a.dataset)
    # training items: clean training claims (+ in Setting B the transformed training claims, labelled by their claim)
    tr_items = [(c["claim"], c["id"], c["label"]) for c in tr]
    if a.aug:
        lab = {c["id"]: c["label"] for c in tr}
        tr_items += [(r["anchor"], r["id"], lab[r["id"]]) for r in read_jsonl(ROOT / f"data/train_aug/{a.dataset}.jsonl")]

    sents, owner = build_sentence_pool(a.dataset)
    sb = SentenceTransformer(SBERT, device="cuda")
    texts = [t for t, _, _ in tr_items] + [c["claim"] for c in dv] + [r["query"] for r in te]
    owners = [o for _, o, _ in tr_items] + [None] * (len(dv) + len(te))  # only training claims have archive passages
    sel = select_evidence(texts, owners, sents, owner, sb, k=5)
    for i, (_, o, _) in enumerate(tr_items):  # leakage guard: a training item never sees its own claim's article
        assert not any(owner[j] == o for j in sel[i])
    del sb; torch.cuda.empty_cache()
    evid = [" ".join(sents[j] for j in row) for row in sel]

    tok = AutoTokenizer.from_pretrained(SCIBERT)
    enc = lambda xs, ev: tok(xs, ev, truncation="longest_first", max_length=512)  # attacked inputs can exceed 512 alone

    class DS(torch.utils.data.Dataset):
        def __init__(self, e, y): self.e, self.y = e, y
        def __len__(self): return len(self.y)
        def __getitem__(self, i):
            d = {k: v[i] for k, v in self.e.items()}; d["labels"] = int(self.y[i]); return d

    ntr, ndv = len(tr_items), len(dv)
    dtr = DS(enc(texts[:ntr], evid[:ntr]), [y for _, _, y in tr_items])
    ddv = DS(enc(texts[ntr:ntr + ndv], evid[ntr:ntr + ndv]), [c["label"] for c in dv])
    dte = DS(enc(texts[ntr + ndv:], evid[ntr + ndv:]), [r["label"] for r in te])
    y_dev = [c["label"] for c in dv]

    def fit(bs, lr, ep, seed):
        set_seed(seed)
        model = AutoModelForSequenceClassification.from_pretrained(SCIBERT, num_labels=2)
        odir = f"/scratch/naa9497/misinfo/ckpt/{a.dataset}_kotonya_s{seed}"
        args = TrainingArguments(odir, per_device_train_batch_size=bs, per_device_eval_batch_size=64, learning_rate=lr,
                                 num_train_epochs=ep, save_strategy="no", report_to=[], seed=seed, bf16=True,
                                 logging_steps=100)
        t = Trainer(model=model, args=args, train_dataset=dtr, data_collator=DataCollatorWithPadding(tok))
        t.train()
        return t

    def prob(t, d):
        lo = t.predict(d).predictions
        e = np.exp(lo - lo.max(-1, keepdims=True))
        return (e / e.sum(-1, keepdims=True))[:, 1]

    best_path = ROOT / f"data/kotonya/{a.dataset}.best.json"
    if a.grid:
        # the authors' protocol: grid search on the dev set over batch {8,16,32} x lr {1e-5,5e-6,1e-6} x epochs {2,3,4},
        # selected by dev macro-F1 at the default 0.5 decision rule (seed 13), then 3 seeds with the chosen setting
        res = []
        for bs in (8, 16, 32):
            for lr in (1e-5, 5e-6, 1e-6):
                for ep in (2, 3, 4):
                    t = fit(bs, lr, ep, 13)
                    f = metrics(y_dev, (prob(t, ddv) >= 0.5).astype(int))["macro_f1"]
                    res.append(dict(bs=bs, lr=lr, epochs=ep, dev_macro_f1=f))
                    print("grid", res[-1], flush=True)
                    del t; torch.cuda.empty_cache()
        best = max(res, key=lambda r: (r["dev_macro_f1"], -r["lr"]))
        best_path.parent.mkdir(parents=True, exist_ok=True)
        json.dump(dict(best=best, grid=res), open(best_path, "w"), indent=1)
        print("best", best)
        return
    cfg = json.load(open(best_path))["best"]
    t = fit(cfg["bs"], cfg["lr"], cfg["epochs"], a.seed)
    p_dev, p_te = prob(t, ddv), prob(t, dte)
    thr = choose_threshold([c["label"] for c in dv], p_dev)
    rows = [dict(id=c["id"], attack="DEV_CLEAN", label=c["label"], p_false=float(p), pred=int(p >= thr), pred05=int(p >= 0.5))
            for c, p in zip(dv, p_dev)]
    rows += [dict(id=r["id"], attack=r["attack"], label=r["label"], p_false=float(p), pred=int(p >= thr), pred05=int(p >= 0.5))
             for r, p in zip(te, p_te)]
    check_preds(a.dataset, rows)
    name = ("kotonya-advtrain" if a.aug else "kotonya") + (f"__s{a.seed}" if a.seed != 13 else "")
    write_jsonl(ROOT / f"results/preds/{a.dataset}/{name}.jsonl", rows)
    by = {}
    for r in rows:
        by.setdefault(r["attack"], []).append(r)
    for att, rs in by.items():
        m = metrics([r["label"] for r in rs], [r["pred"] for r in rs])
        print(f"{name:10s} {att:12s} acc={m['acc']:.3f} mF1={m['macro_f1']:.3f} falseAcc={m['false_acc']:.3f} "
              f"trueAcc={m['true_acc']:.3f}", flush=True)


if __name__ == "__main__":
    main()
