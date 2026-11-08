"""Fine-tuned encoder classifiers (claim-only misinformation detectors) trained on CLEAN train claims.

usage: python clf.py --dataset pubhealth --model FacebookAI/xlm-roberta-large --tag xlmr [--seed 13] [--aug]
  --aug   adversarial-training baseline: add the train-split augmentations used for AIR (data/train_aug, seen
          attack families only, different generator than the test attacks) with their claim's label
Also writes predictions on canonicalized inputs (data/canon/<ds>.ours.jsonl) as clf_<tag>+canon, if available.
"""
import argparse, json
from common import ROOT, claims, attacks, write_jsonl, metrics, read_jsonl, choose_threshold, check_preds  # first: sets offline mode
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments,
                          DataCollatorWithPadding, set_seed)



class DS(Dataset):
    def __init__(self, enc, y):
        self.enc, self.y = enc, y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        d = {k: v[i] for k, v in self.enc.items()}
        d["labels"] = int(self.y[i])
        return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--aug", action="store_true")
    a = ap.parse_args()
    set_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model)
    tr, dv = claims(a.dataset, "train"), claims(a.dataset, "dev")
    enc = lambda xs: tok(xs, truncation=True, max_length=512)
    xtr, ytr = [c["claim"] for c in tr], [c["label"] for c in tr]
    if a.aug:
        lab = {c["id"]: c["label"] for c in tr}
        for r in read_jsonl(ROOT / f"data/train_aug/{a.dataset}.jsonl"):
            xtr.append(r["anchor"]); ytr.append(lab[r["id"]])
    dtr = DS(enc(xtr), ytr)
    ddv = DS(enc([c["claim"] for c in dv]), [c["label"] for c in dv])
    # class-weighted loss for imbalanced sets
    w = np.bincount(ytr, minlength=2)
    w = torch.tensor(len(ytr) / (2 * w), dtype=torch.float)

    class WTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            y = inputs.pop("labels")
            out = model(**inputs)
            loss = torch.nn.functional.cross_entropy(out.logits, y, weight=w.to(out.logits.device))
            return (loss, out) if return_outputs else loss

    def cm(ev):
        return {"macro_f1": metrics(ev.label_ids, ev.predictions.argmax(-1))["macro_f1"]}

    model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=2)
    odir = f"/scratch/naa9497/misinfo/ckpt/{a.dataset}_{a.tag}_s{a.seed}"
    args = TrainingArguments(odir, per_device_train_batch_size=16, per_device_eval_batch_size=64,
                             learning_rate=a.lr, num_train_epochs=a.epochs, warmup_ratio=0.1, weight_decay=0.01,
                             eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
                             metric_for_best_model="macro_f1", save_total_limit=1, bf16=True, report_to=[],
                             seed=a.seed, logging_steps=50)
    t = WTrainer(model=model, args=args, train_dataset=dtr, eval_dataset=ddv,
                 data_collator=DataCollatorWithPadding(tok), compute_metrics=cm)
    t.train()

    def prob(texts):
        d = DS(enc(texts), [0] * len(texts))
        lo = t.predict(d).predictions
        e = np.exp(lo - lo.max(-1, keepdims=True))
        return (e / e.sum(-1, keepdims=True))[:, 1]

    y_dev = np.array([c["label"] for c in dv])
    threshold = lambda p_dev: choose_threshold(y_dev, p_dev)

    test = attacks(a.dataset)
    # each variant is calibrated on its own view of the clean dev claims (raw, or canonicalized for +canon)
    variants = {"": ([c["claim"] for c in dv], [r["query"] for r in test])}
    cpath = ROOT / f"data/canon/{a.dataset}.ours.jsonl"
    if cpath.exists():
        cc = {(r["id"], r["attack"]): r["claims"] for r in read_jsonl(cpath)}
        variants["+canon"] = ([" ".join(cc[(c["id"], "DEV_CLEAN")]) or c["claim"] for c in dv],
                              [" ".join(cc[(r["id"], r["attack"])]) or r["query"] for r in test])
    for suffix, (dev_texts, texts) in variants.items():
        p_dev = prob(dev_texts)
        thr = threshold(p_dev)
        p = prob(texts)
        rows = [dict(id=c["id"], attack="DEV_CLEAN", label=c["label"], p_false=float(x), pred=int(x >= thr),
                     pred05=int(x >= 0.5)) for c, x in zip(dv, p_dev)]
        rows += [dict(id=r["id"], attack=r["attack"], label=r["label"], p_false=float(x), pred=int(x >= thr),
                      pred05=int(x >= 0.5)) for r, x in zip(test, p)]
        name = f"clf_{a.tag}{suffix}" + (f"__s{a.seed}" if a.seed != 13 else "")
        check_preds(a.dataset, rows)
        out = ROOT / f"results/preds/{a.dataset}/{name}.jsonl"
        write_jsonl(out, rows)
        json.dump(dict(threshold=float(thr), args=vars(a)), open(str(out).replace(".jsonl", ".meta.json"), "w"))
        by = {}
        for r in rows:
            by.setdefault(r["attack"], []).append(r)
        for att, rs in by.items():
            m = metrics([r["label"] for r in rs], [r["pred"] for r in rs])
            print(f"{name:20s} {att:12s} acc={m['acc']:.3f} mF1={m['macro_f1']:.3f} "
                  f"falseAcc={m['false_acc']:.3f} trueAcc={m['true_acc']:.3f}", flush=True)
    import shutil
    shutil.rmtree(odir, ignore_errors=True)


if __name__ == "__main__":
    main()
