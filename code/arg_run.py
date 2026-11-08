"""ARG baseline (Hu et al., AAAI 2024, "Bad Actor, Good Advisor"), run with the OFFICIAL model and trainer
(baselines/arg_official, vendored from github.com/ICTMCG/ARG; only change: a pandas-2 fix in the data loader).

stage "rationales": LLM rationales from the two perspectives used by ARG, textual description (td) and commonsense
  (cs), with the paper's zero-shot CoT perspective-specific template (Table 8):
     Q: Given the following message, predict its veracity. If it is more likely to be a real message, return 1;
        otherwise, return 0. Please refrain from providing ambiguous assessments such as undetermined: <message>
     A: Let's think from the perspective of <textual description | commonsense>.
  The answer is continued by the LLM (Llama-3.1-8B-Instruct instead of GPT-3.5-turbo: open model, stated in the
  paper). td_pred/cs_pred: "return 1"/"1" -> real (0), "return 0"/"0" -> fake (1), otherwise other (2);
  td_acc/cs_acc = prediction equals the gold label (ARG's rationale-usefulness supervision).
  Rationales are produced for train, dev and every test input (clean and attacked), since ARG needs them at test time.
stage "train": official ARG Trainer with the hyper-parameters of the official run_en.sh (bert-base-uncased, lr 5e-5,
  batch 64, max_len 170, 50 epochs with early stopping 5 on validation macro-F1, weights 1.5 / 1.0, weight decay
  5e-5). Train = train split, val = dev split. Then predict every test input. Threshold calibrated on clean dev.

usage: python arg_run.py --dataset pubhealth --stage rationales
       python arg_run.py --dataset pubhealth --stage train --seed 3759
"""
import argparse, json, os, re, sys
from pathlib import Path

import numpy as np
from common import ROOT, claims, attacks, read_jsonl, write_jsonl, metrics, choose_threshold, check_preds

LLAMA = "meta-llama/Meta-Llama-3.1-8B-Instruct"
Q_TMPL = ("Given the following message, predict its veracity. If it is more likely to be a real message, return 1; "
          "otherwise, return 0. Please refrain from providing ambiguous assessments such as undetermined: {m}")
PERSP = {"td": "textual description", "cs": "commonsense"}


def parse_pred(text):
    """ARG judgment from a rationale: 1 -> real (0), 0 -> fake (1), else other (2). Uses the last explicit verdict."""
    t = text.lower()
    m = re.findall(r"return\s*\**\s*([01])\b", t) or re.findall(r"answer[^0-9]{0,40}?\b([01])\b", t) \
        or re.findall(r"(?:^|\s)([01])\s*\.?\s*$", t.strip())
    if not m:
        return 2
    return 0 if m[-1] == "1" else 1


def items(ds, aug=False):
    out = [dict(key=f"{c['id']}|TRAIN", content=c["claim"], label=c["label"], split="train") for c in claims(ds, "train")]
    if aug:  # Setting B (attack-aware): the same transformed training claims used by adversarial training / AIR
        lab = {c["id"]: c["label"] for c in claims(ds, "train")}
        out += [dict(key=f"{r['id']}|AUG|{r['family']}", content=r["anchor"], label=lab[r["id"]], split="train")
                for r in read_jsonl(ROOT / f"data/train_aug/{ds}.jsonl")]
    out += [dict(key=f"{c['id']}|DEV_CLEAN", content=c["claim"], label=c["label"], split="dev") for c in claims(ds, "dev")]
    out += [dict(key=f"{r['id']}|{r['attack']}", content=r["query"], label=r["label"], split="test", id=r["id"],
                 attack=r["attack"]) for r in attacks(ds)]
    return out


def stage_rationales(ds, aug=False):
    from vllm import LLM, SamplingParams
    path = ROOT / f"data/arg/{ds}.rationales.jsonl"
    old = {r["key"]: r for r in read_jsonl(path)} if path.exists() else {}
    all_its = items(ds, aug)
    its = [it for it in all_its if it["key"] not in old]  # only generate what is missing (same prompts, greedy)
    print(ds, "rationales to generate:", len(its), "already available:", len(all_its) - len(its), flush=True)
    if not its:
        return
    llm = LLM(LLAMA, gpu_memory_utilization=0.85, max_model_len=8192, seed=0)
    tok = llm.get_tokenizer()

    def fit(m):  # keep long attacked inputs inside the context window (middle trimmed)
        ids = tok(m, add_special_tokens=False).input_ids
        return m if len(ids) < 7000 else tok.decode(ids[:3400]) + " ... " + tok.decode(ids[-3400:])

    res = {it["key"]: {} for it in its}
    for p, name in PERSP.items():
        msgs = [[{"role": "user", "content": Q_TMPL.format(m=fit(it["content"]))},
                 {"role": "assistant", "content": f"Let's think from the perspective of {name}."}] for it in its]
        outs = llm.chat(msgs, SamplingParams(temperature=0, max_tokens=256), use_tqdm=True,
                        add_generation_prompt=False, continue_final_message=True)
        for it, o in zip(its, outs):
            r = f"Let's think from the perspective of {name}." + o.outputs[0].text
            res[it["key"]][f"{p}_rationale"] = r
            res[it["key"]][f"{p}_pred"] = parse_pred(r)
    merged = dict(old)
    merged.update({k: dict(key=k, **v) for k, v in res.items()})
    write_jsonl(path, list(merged.values()))
    for p in PERSP:
        preds = [res[it["key"]][f"{p}_pred"] for it in its]
        print(ds, p, "pred distribution (0 real / 1 fake / 2 other):", np.bincount(preds, minlength=3).tolist())


def stage_train(ds, seed, aug=False):
    off = Path(__file__).resolve().parent / "baselines" / "arg_official"
    sys.path.insert(0, str(off))
    import random, torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    from models.arg import Trainer                     # official
    from utils.dataloader import get_dataloader        # official
    from utils.utils import get_tensorboard_writer     # official

    R = {r["key"]: r for r in read_jsonl(ROOT / f"data/arg/{ds}.rationales.jsonl")}
    its = items(ds, aug)
    work = Path(f"/scratch/naa9497/misinfo/arg_work/{ds}_s{seed}" + ("_aug" if aug else ""))
    work.mkdir(parents=True, exist_ok=True)

    def rec(it, sid):
        r = R[it["key"]]
        return dict(content=it["content"], label=it["label"], source_id=sid,
                    td_rationale=r["td_rationale"], td_pred=r["td_pred"], td_acc=int(r["td_pred"] == it["label"]),
                    cs_rationale=r["cs_rationale"], cs_pred=r["cs_pred"], cs_acc=int(r["cs_pred"] == it["label"]))

    split = {"train": [], "dev": [], "test": []}
    for sid, it in enumerate(its):
        split[it["split"]].append((sid, it))
    json.dump([rec(it, sid) for sid, it in split["train"]], open(work / "train.json", "w"))
    json.dump([rec(it, sid) for sid, it in split["dev"]], open(work / "val.json", "w"))
    json.dump([rec(it, sid) for sid, it in split["test"] if it["attack"] == "CLEAN"], open(work / "test.json", "w"))
    json.dump([rec(it, sid) for sid, it in split["test"]], open(work / "all_test.json", "w"))

    bert = "google-bert/bert-base-uncased"
    config = {  # official main.py config with the official run_en.sh arguments
        'use_cuda': True, 'seed': seed, 'batchsize': 64, 'max_len': 170, 'early_stop': 5, 'language': 'en',
        'root_path': str(work), 'weight_decay': 5e-5,
        'model': {'mlp': {'dims': [384], 'dropout': 0.2}, 'llm_judgment_predictor_weight': 1.0,
                  'rationale_usefulness_evaluator_weight': 1.5, 'kd_loss_weight': 1},
        'emb_dim': 768, 'co_attention_dim': 300, 'lr': 5e-5, 'epoch': 50, 'model_name': 'ARG',
        'save_log_dir': str(work / 'logs'), 'save_param_dir': str(work / 'param_model'),
        'param_log_dir': str(work / 'logs/param'), 'tensorboard_dir': str(work / 'logs/tensorlog'),
        'bert_path': bert, 'data_type': 'rationale', 'data_name': f'{ds}-arg', 'eval_mode': False,
        'teacher_path': None, 'month': 1}
    os.chdir(work)  # the official trainer writes ./logs relative to the working directory
    trainer = Trainer(config, get_tensorboard_writer(config))
    trainer.train()
    # trainer.model now holds the best-validation parameters (loaded at the end of official train())

    def predict(path):
        loader = get_dataloader(str(path), 170, 64, shuffle=False, bert_path=bert, data_type='rationale', language='en')
        _, label, pred, _, _, _ = trainer.predict(loader)
        return np.array(pred, dtype=float)

    p_dev = predict(work / "val.json")
    p_test = predict(work / "all_test.json")
    y_dev = [it["label"] for _, it in split["dev"]]
    thr = choose_threshold(y_dev, p_dev)
    rows = [dict(id=it["key"].split("|")[0], attack="DEV_CLEAN", label=it["label"], p_false=float(p), pred=int(p >= thr),
                 pred05=int(p >= 0.5)) for (_, it), p in zip(split["dev"], p_dev)]
    rows += [dict(id=it["id"], attack=it["attack"], label=it["label"], p_false=float(p), pred=int(p >= thr),
                  pred05=int(p >= 0.5)) for (_, it), p in zip(split["test"], p_test)]
    check_preds(ds, rows)
    name = ("arg-advtrain" if aug else "arg") + (f"__s{seed}" if seed != 3759 else "")
    write_jsonl(ROOT / f"results/preds/{ds}/{name}.jsonl", rows)
    by = {}
    for r in rows:
        by.setdefault(r["attack"], []).append(r)
    for att, rs in by.items():
        m = metrics([r["label"] for r in rs], [r["pred"] for r in rs])
        print(f"{name:10s} {att:12s} acc={m['acc']:.3f} mF1={m['macro_f1']:.3f} falseAcc={m['false_acc']:.3f} "
              f"trueAcc={m['true_acc']:.3f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--stage", required=True, choices=["rationales", "train"])
    ap.add_argument("--seed", type=int, default=3759)  # official default seed
    ap.add_argument("--aug", action="store_true", help="Setting B: add the transformed training claims (data/train_aug)")
    a = ap.parse_args()
    stage_rationales(a.dataset, a.aug) if a.stage == "rationales" else stage_train(a.dataset, a.seed, a.aug)
