"""Train UNWRAP's verifier (Llama-3.1-8B-Instruct + LoRA) on the TRAINING split of all three datasets.

Each training example is exactly what the verifier sees at test time (same retrieval, reranking and prompt code as
detect.py): the canonical claims of a training claim + the top-5 retrieved archive records with verdicts; the target
is "Yes" (spreads false information) for false claims and "No" for true claims.
Because training claims ARE in the archive while test claims never are, retrieval for a training claim excludes its
own record, so the verifier learns to judge from related verified claims and evidence, never from an exact copy.
Only clean training claims are used (no attacked or transformed text): UNWRAP stays attack-agnostic.
Model selection: epochs in {1, 2} chosen by mean clean-dev macro-F1 over the three datasets (dev claims are never in
the archive). One adapter serves all datasets.

stages (separate processes, so GPU memory is released between them):
  python train_verifier.py --stage canon      canonicalise training claims (vLLM)   -> data/canon/<ds>.ours.train.jsonl
  python train_verifier.py --stage data       retrieval + prompts                   -> data/verifier/{train,dev}.jsonl
  python train_verifier.py --stage train      LoRA fine-tuning + dev selection      -> models/verifier_lora
"""
import argparse, json, random
import numpy as np

from common import ROOT, DATASETS, claims, read_jsonl, write_jsonl, metrics, choose_threshold

LLAMA = "meta-llama/Meta-Llama-3.1-8B-Instruct"
ADAPTER_DIR = "/scratch/naa9497/misinfo/models/verifier_lora"


def train_items(ds, aug):
    """(key, text, claim_id, label): clean training claims, plus in Setting B the transformed training claims"""
    tr = claims(ds, "train")
    out = [(f"{c['id']}|TRAIN", c["claim"], c["id"], c["label"]) for c in tr]
    if aug:
        lab = {c["id"]: c["label"] for c in tr}
        out += [(f"{r['id']}|AUG|{r['family']}", r["anchor"], r["id"], lab[r["id"]])
                for r in read_jsonl(ROOT / f"data/train_aug/{ds}.jsonl")]
    return out


def stage_canon(aug=False):
    from canonicalize import PROMPTS, parse_claims
    from detect import LLM
    llm = LLM(LLAMA, mem=0.85)
    for ds in DATASETS:
        its = train_items(ds, aug)
        outs = llm.generate([PROMPTS["ours"].format(q=t) for _, t, _, _ in its], max_tokens=200)
        write_jsonl(ROOT / f"data/canon/{ds}.ours.train{'aug' if aug else ''}.jsonl",
                    [dict(key=k, id=c, attack="TRAIN", claims=parse_claims(o)) for (k, _, c, _), o in zip(its, outs)])
        print(ds, "canonicalised", len(its), flush=True)


def stage_data(aug=False):
    """Setting A: clean training claims, original BGE-M3 retriever (UNWRAP). Setting B: + transformed training claims,
    AIR retriever (the attack-aware UNWRAP uses AIR at test time, so training examples are built with it too)."""
    from detect import retrieve_candidates, verify_prompt, load_canon
    enc = "air" if aug else "bgem3"
    train, dev = [], []
    for ds in DATASETS:
        its = train_items(ds, aug)
        ctr = {r["key"]: r["claims"] for r in read_jsonl(ROOT / f"data/canon/{ds}.ours.train{'aug' if aug else ''}.jsonl")}
        cdv = load_canon(ds, "ours")
        dv = claims(ds, "dev")
        views = [[t] + ctr[k] for k, t, _, _ in its] + [[c["claim"]] + cdv[(c["id"], "DEV_CLEAN")] for c in dv]
        # archive claim positions, built exactly as retrieval.Index does (contiguous blocks, order of appearance)
        rows = read_jsonl(ROOT / f"data/processed/{ds}/kb.jsonl")
        cid = [r["claim_id"] for r in rows]
        order = [cid[0]] + [cid[i] for i in range(1, len(cid)) if cid[i] != cid[i - 1]]
        claim_pos = {c: i for i, c in enumerate(order)}
        exclude = [claim_pos[c] for _, _, c, _ in its] + [None] * len(dv)
        idx, cand = retrieve_candidates(ds, views, enc, sparse=True, rerank=True, exclude=exclude)
        for i, (_, _, c, _) in enumerate(its):  # leakage guard: own record never retrieved
            assert all(idx.claim_ids[ci] != c for ci, _ in cand[i])
        for i, (k, _, c, y) in enumerate(its):
            train.append(dict(dataset=ds, id=c, key=k, label=y, prompt=verify_prompt(idx, views[i], cand[i])))
        for j, c in enumerate(dv):
            i = len(its) + j
            dev.append(dict(dataset=ds, id=c["id"], label=c["label"], prompt=verify_prompt(idx, views[i], cand[i])))
        print(ds, "train", len(its), "dev", len(dv), flush=True)
        del idx
    suf = "_aug" if aug else ""
    write_jsonl(ROOT / f"data/verifier/train{suf}.jsonl", train)
    write_jsonl(ROOT / f"data/verifier/dev{suf}.jsonl", dev)


def encode_example(tok, prompt, answer=None, max_len=3072):
    """chat-template ids of the user prompt (+ generation header); if answer is given, append it and an end-of-turn
    token and return labels that are -100 everywhere except on the answer and end-of-turn tokens."""
    p = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True)
    if len(p) > max_len:  # keep the beginning (instructions, records) and the end (claims + question)
        p = p[: max_len // 2] + p[-(max_len // 2):]
    if answer is None:
        return p
    a = tok(answer, add_special_tokens=False).input_ids + [tok.convert_tokens_to_ids("<|eot_id|>")]
    return p + a, [-100] * len(p) + a


def adapter_path(aug=False, seed=0):
    """seed 0 keeps the original path (models/verifier_lora[_aug]); other seeds get the suffix _s<seed>"""
    return ADAPTER_DIR + ("_aug" if aug else "") + (f"_s{seed}" if seed else "")


def stage_train(aug=False, epochs_grid=(1, 2), lr=1e-4, seed=0):
    import torch, shutil
    from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
    from peft import LoraConfig, get_peft_model
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    ADAPTER = adapter_path(aug, seed)
    suf = "_aug" if aug else ""
    tok = AutoTokenizer.from_pretrained(LLAMA)
    tr, dv = read_jsonl(ROOT / f"data/verifier/train{suf}.jsonl"), read_jsonl(ROOT / f"data/verifier/dev{suf}.jsonl")
    ex = [encode_example(tok, r["prompt"], "Yes" if r["label"] == 1 else "No") for r in tr]
    yes_id, no_id = tok("Yes", add_special_tokens=False).input_ids[0], tok("No", add_special_tokens=False).input_ids[0]
    model = AutoModelForCausalLM.from_pretrained(LLAMA, torch_dtype=torch.bfloat16).cuda()
    model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                             "gate_proj", "up_proj", "down_proj"]))
    bs, accum = 4, 4
    steps = (len(ex) + bs * accum - 1) // (bs * accum) * max(epochs_grid)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    sch = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)

    def dev_eval():
        model.eval(); p_yes = []
        with torch.no_grad():
            for r in dv:
                ids = torch.tensor([encode_example(tok, r["prompt"])]).cuda()
                lo = model(input_ids=ids).logits[0, -1, [yes_id, no_id]].float()
                p_yes.append(torch.softmax(lo, -1)[0].item())
        model.train()
        res = {}
        for ds in DATASETS:
            ii = [i for i, r in enumerate(dv) if r["dataset"] == ds]
            y, p = [dv[i]["label"] for i in ii], [p_yes[i] for i in ii]
            res[ds] = metrics(y, (np.array(p) >= choose_threshold(y, p)).astype(int))["macro_f1"]
        return res

    results = {}
    model.train()
    for ep in range(1, max(epochs_grid) + 1):
        order = np.random.permutation(len(ex))
        for b in range(0, len(order), bs):
            batch = [ex[i] for i in order[b:b + bs]]
            L = max(len(x) for x, _ in batch)
            ids = torch.tensor([x + [tok.eos_token_id] * (L - len(x)) for x, _ in batch]).cuda()
            lab = torch.tensor([y + [-100] * (L - len(y)) for _, y in batch]).cuda()
            att = torch.tensor([[1] * len(x) + [0] * (L - len(x)) for x, _ in batch]).cuda()
            loss = model(input_ids=ids, attention_mask=att, labels=lab).loss / accum
            loss.backward()
            if (b // bs + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sch.step(); opt.zero_grad()
            if (b // bs) % 200 == 0:
                print(f"epoch {ep} step {b // bs} loss {loss.item() * accum:.4f}", flush=True)
        if ep in epochs_grid:
            r = dev_eval(); results[ep] = r
            print(f"epoch {ep} clean-dev macro-F1 {r} mean {np.mean(list(r.values())):.4f}", flush=True)
            model.save_pretrained(f"{ADAPTER}_e{ep}")
    best = max(results, key=lambda e: (np.mean(list(results[e].values())), -e))
    shutil.rmtree(ADAPTER, ignore_errors=True)
    shutil.copytree(f"{ADAPTER}_e{best}", ADAPTER)
    json.dump(dict(dev=results, best_epoch=best), open(f"{ADAPTER}/selection.json", "w"), indent=1)
    print("selected epoch", best, results[best])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["canon", "data", "train"])
    ap.add_argument("--aug", action="store_true", help="Setting B: clean + transformed training claims, AIR retriever")
    ap.add_argument("--seed", type=int, default=0, help="train stage only: training seed (data order, LoRA init)")
    a = ap.parse_args()
    if a.stage == "train":
        stage_train(aug=a.aug, seed=a.seed)
    else:
        assert a.seed == 0, "--seed applies to the train stage only (canon and data are deterministic)"
        {"canon": stage_canon, "data": stage_data}[a.stage](aug=a.aug)
