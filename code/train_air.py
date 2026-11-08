"""Attack-invariant retrieval (AIR): contrastive fine-tuning of the BGE-M3 dense encoder so that a transformed
TRAIN claim (anchor) embeds close to its own archive record (positive) and away from other train claims
(in-batch negatives + one mined hard negative). Only train-split data are used. One shared model is trained on
the union of the three training splits for a fixed single epoch (no early stopping; dev data are not used here).

usage: python train_air.py --out /scratch/naa9497/misinfo/models/bgem3-air
"""
import argparse, random
from common import ROOT, claims, read_jsonl  # first: sets offline mode
import numpy as np
import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments, losses
from sentence_transformers.training_args import BatchSamplers


DATASETS = ["pubhealth", "climatefever", "finfact"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--max_steps", type=int, default=-1)
    a = ap.parse_args()
    random.seed(0)
    model = SentenceTransformer("BAAI/bge-m3", device="cuda")
    model.max_seq_length = 1024

    rows = []
    for ds in a.datasets:
        tr = {c["id"]: c for c in claims(ds, "train")}
        ids = list(tr)
        aug = read_jsonl(ROOT / f"data/train_aug/{ds}.jsonl")
        # hard negatives: nearest OTHER train claim under the original encoder
        emb = model.encode([tr[i]["claim"] for i in ids], batch_size=128, normalize_embeddings=True,
                           convert_to_tensor=True, show_progress_bar=False)
        sim = emb @ emb.T
        sim.fill_diagonal_(-1)
        nn = sim.topk(5, dim=1).indices.cpu().numpy()
        pos = {i: k for k, i in enumerate(ids)}
        for r in aug:
            k = pos[r["id"]]
            neg = ids[random.choice(list(nn[k]))]
            rows.append(dict(anchor=r["anchor"], positive=tr[r["id"]]["claim"], negative=tr[neg]["claim"]))
        # clean anchors keep the clean-input behaviour intact
        for i in ids:
            k = pos[i]
            rows.append(dict(anchor=tr[i]["claim"], positive=tr[i]["claim"], negative=tr[ids[nn[k][0]]]["claim"]))
    random.shuffle(rows)
    print("training pairs", len(rows))
    train = Dataset.from_list(rows)
    loss = losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=16)
    args = SentenceTransformerTrainingArguments(
        output_dir=a.out + "_ckpt", num_train_epochs=a.epochs, per_device_train_batch_size=a.bs, learning_rate=a.lr,
        warmup_ratio=0.1, bf16=True, max_steps=a.max_steps, batch_sampler=BatchSamplers.NO_DUPLICATES, save_strategy="no",
        logging_steps=50, report_to=[], seed=0)
    SentenceTransformerTrainer(model=model, args=args, train_dataset=train, loss=loss).train()
    model.save(a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
