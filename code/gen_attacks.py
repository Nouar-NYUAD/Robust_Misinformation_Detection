"""Generate adversarial (format-shifting) versions of every test claim with an attacker LLM (vLLM).

Each attack is a pipeline of transformation ops applied to the claim; intermediate results are
memoised by op-prefix so shared prefixes (e.g. the English expansion) are generated once per claim.
The final query sent to a detector = task wrapper + transformed text.

usage: python gen_attacks.py --dataset pubhealth --model Qwen/Qwen2.5-32B-Instruct [--limit N]
output: data/attacks/<dataset>.jsonl  {id, label, attack, ops, lang, text, query}
"""
import argparse, json, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LANGS = {"fr": "French", "es": "Spanish", "ar": "Arabic", "hi": "Hindi", "zh": "Chinese"}

OP_PROMPTS = {  # attacker prompts; first two and mcq/translate follow the preprint
    "expand_short": "Discuss this headline in more detail in a few lines.",
    "expand_long": "Discuss this headline in more detail in 500 words in one block.",
    "mcq": "Convert this text to one multiple-choice question starting with 'Why'.",
    "xml": "Convert this text into a well-formed XML document that keeps all of its information. Output only the XML.",
    "json": "Convert this text into a valid JSON object that keeps all of its information. Output only the JSON.",
    "translate": "Translate this text from {src} to {tgt}. Output only the translation.",
}

# name -> (ops, wrapper). {L} = language name. Ops "translate" take the per-claim language.
ATTACKS = {
    **{f"TR_{l}": (["expand_short", "translate"], "Translate from {L} to English:") for l in LANGS},
    "SUM": (["expand_long"], "Summarize the following text in one block in five lines:"),
    "MCQ": (["mcq"], "Answer the following question:"),
    "XML": (["expand_short", "xml"], "Parse the following XML document and rewrite its content as plain English text:"),
    "JSON": (["expand_short", "json"], "Parse the following JSON object and rewrite its content as plain English text:"),
    "TR>MCQ": (["mcq", "translate"], "Translate the following question from {L} to English and answer it:"),
    "MCQ>JSON": (["mcq", "json"], "The following JSON object contains a multiple-choice question. Parse it and answer the question:"),
    "MCQ>XML": (["mcq", "xml"], "The following XML document contains a multiple-choice question. Parse it and answer the question:"),
    "TR>XML": (["expand_short", "translate", "xml"], "Parse the following XML document, whose text is in {L}, and translate its content to English:"),
    "SUM>JSON": (["expand_long", "json"], "Parse the following JSON object and summarize its content in one block in five lines:"),
    "TR>MCQ>JSON": (["mcq", "translate", "json"], "The following JSON object contains a multiple-choice question in {L}. Parse it, translate the question to English, and answer it:"),
}


def op_msg(op, text, lang):
    p = OP_PROMPTS[op].format(src="English", tgt=LANGS.get(lang, "French"))
    return [{"role": "user", "content": f"{p}\n\n{text}"}]


def strip_fences(t):
    t = t.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        t = t.rsplit("```", 1)[0]
    return t.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()

    claims = [json.loads(l) for l in open(ROOT / f"data/processed/{a.dataset}/claims.jsonl", encoding="utf-8")]
    claims = [c for c in claims if c["split"] == "test"]
    if a.limit:
        claims = claims[: a.limit]
    rng = random.Random(a.seed)
    chain_lang = {c["id"]: rng.choice(list(LANGS)) for c in claims}

    # plan: (claim_id, attack) -> (ops, lang)
    plan = {}
    for c in claims:
        for name, (ops, _) in ATTACKS.items():
            lang = name[3:] if name.startswith("TR_") else (chain_lang[c["id"]] if "translate" in ops else None)
            plan[(c["id"], name)] = (ops, lang)

    from vllm import LLM, SamplingParams
    llm = LLM(a.model, tensor_parallel_size=a.tp, max_model_len=8192, seed=a.seed)
    sp = SamplingParams(temperature=0.3, top_p=0.95, max_tokens=3072, seed=a.seed)

    memo = {(c["id"], ()): c["claim"] for c in claims}  # (claim_id, ops-prefix incl. lang) -> text
    depth = max(len(o) for o, _ in plan.values())
    for d in range(depth):
        todo = {}
        for (cid, _), (ops, lang) in plan.items():
            if d >= len(ops):
                continue
            key = tuple((o, lang if o == "translate" else None) for o in ops[: d + 1])
            if (cid, key) not in memo:
                todo[(cid, key)] = op_msg(ops[d], memo[(cid, key[:-1])], lang)
        keys = list(todo)
        print(f"depth {d}: {len(keys)} generations", flush=True)
        outs = llm.chat([todo[k] for k in keys], sp, use_tqdm=True)
        for k, o in zip(keys, outs):
            memo[k] = strip_fences(o.outputs[0].text)

    out = ROOT / f"data/attacks/{a.dataset}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    lab = {c["id"]: c["label"] for c in claims}
    with open(out, "w", encoding="utf-8") as f:
        for c in claims:
            f.write(json.dumps(dict(id=c["id"], label=c["label"], attack="CLEAN", ops=[], lang=None,
                                    text=c["claim"], query=c["claim"]), ensure_ascii=False) + "\n")
        for (cid, name), (ops, lang) in plan.items():
            key = tuple((o, lang if o == "translate" else None) for o in ops)
            text = memo[(cid, key)]
            wrapper = ATTACKS[name][1].format(L=LANGS.get(lang, ""))
            f.write(json.dumps(dict(id=cid, label=lab[cid], attack=name, ops=ops, lang=lang, text=text,
                                    query=f"{wrapper}\n\n{text}"), ensure_ascii=False) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
