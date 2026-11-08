"""Transformations of TRAIN claims for attack-invariant retrieval (AIR) training.

Deliberately different from the test-time attacks (gen_attacks.py):
  * generator: Llama-3.1-8B-Instruct (test attacks: Qwen2.5-32B-Instruct)
  * paraphrased instructions and wrappers
  * SEEN families only (the preprint's): translation to French/Spanish, summarization, MCQ.
    HELD-OUT at training time: XML, JSON, Arabic/Hindi/Chinese, and all chained attacks.

out: data/train_aug/<dataset>.jsonl {id, family, anchor}
"""
import argparse, json, random
from common import ROOT, claims, write_jsonl

OPS = {
    "short": "Write a short paragraph (3-5 sentences) that elaborates on this statement as if it were a news story.",
    "long": "Write a detailed news-style article of about 400 words based on this statement, in a single paragraph.",
    "mcq": "Rewrite this statement as a single multiple-choice question that begins with 'Why', with four options "
           "labelled A-D, one of which restates the statement.",
    "fr": "Give the French version of the following text. Return only the French text.",
    "es": "Give the Spanish version of the following text. Return only the Spanish text.",
}
WRAP = {
    "TR": ["Please render this passage in English:", "Could you convert the following into English?",
           "English translation needed for:"],
    "SUM": ["Give me a brief summary of this article:", "Condense the passage below into a few sentences:",
            "TL;DR of the following:"],
    "MCQ": ["Pick the right option:", "Which answer is correct?", "Solve this quiz question:"],
}
FAMILIES = {"TR_fr": ["short", "fr"], "TR_es": ["short", "es"], "SUM": ["long"], "MCQ": ["mcq"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["pubhealth", "climatefever", "finfact"])
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--max_per_dataset", type=int, default=4000)
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    llm = LLM(a.model, max_model_len=8192, gpu_memory_utilization=0.85, seed=1)
    sp = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=1200, seed=1)
    rng = random.Random(1)
    for ds in a.datasets:
        tr = claims(ds, "train")
        rng.shuffle(tr)
        tr = tr[: a.max_per_dataset]
        memo = {(c["id"], ()): c["claim"] for c in tr}
        for depth in range(2):
            todo = {}
            for c in tr:
                for fam, ops in FAMILIES.items():
                    if depth < len(ops):
                        key = (c["id"], tuple(ops[: depth + 1]))
                        if key not in memo:
                            todo[key] = OPS[ops[depth]] + "\n\n" + memo[(c["id"], tuple(ops[:depth]))]
            keys = list(todo)
            outs = llm.chat([[{"role": "user", "content": todo[k]}] for k in keys], sp, use_tqdm=True)
            for k, o in zip(keys, outs):
                memo[k] = o.outputs[0].text.strip()
        rows = []
        for c in tr:
            for fam, ops in FAMILIES.items():
                w = rng.choice(WRAP[fam.split("_")[0]])
                rows.append(dict(id=c["id"], dataset=ds, family=fam, anchor=f"{w}\n\n{memo[(c['id'], tuple(ops))]}"))
        write_jsonl(ROOT / f"data/train_aug/{ds}.jsonl", rows)
        print(ds, len(rows))


if __name__ == "__main__":
    main()
