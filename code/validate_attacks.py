"""Check that each attacked text still conveys the original claim (semantic preservation),
using an independent judge LLM from a third model family (not the attacker, not the detector).

usage: python validate_attacks.py --dataset pubhealth [--judge google/gemma-3-12b-it]
out:   data/attacks/<dataset>.valid.jsonl {id, attack, p_preserved}
"""
import argparse, math
from collections import defaultdict
from common import ROOT, attacks, write_jsonl

PROMPT = ("Original claim: {c}\n\nText (it may be in another language, formatted as XML/JSON, phrased as a question "
          "or multiple-choice question, or be a longer article):\n{t}\n\n"
          "Does the text assert, repeat, or take for granted the original claim (not deny it)? "
          "Answer with 'Yes' or 'No' only.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--judge", default="google/gemma-3-12b-it")
    a = ap.parse_args()
    rows = [r for r in attacks(a.dataset) if r["attack"] != "CLEAN"]
    claim = {r["id"]: r["text"] for r in attacks(a.dataset) if r["attack"] == "CLEAN"}
    from vllm import LLM, SamplingParams
    llm = LLM(a.judge, max_model_len=8192, gpu_memory_utilization=0.85)
    tok = llm.get_tokenizer()

    def fit(t):
        ids = tok(t, add_special_tokens=False).input_ids
        return t if len(ids) < 7000 else tok.decode(ids[:3400]) + " ... " + tok.decode(ids[-3400:])

    msgs = [[{"role": "user", "content": PROMPT.format(c=claim[r["id"]], t=fit(r["text"]))}] for r in rows]
    outs = llm.chat(msgs, SamplingParams(temperature=0, max_tokens=1, logprobs=20), use_tqdm=True)
    res, agg = [], defaultdict(list)
    for r, o in zip(rows, outs):
        y = n = 0.0
        for t in o.outputs[0].logprobs[0].values():
            s = (t.decoded_token or "").strip().lower()
            y += math.exp(t.logprob) if s == "yes" else 0
            n += math.exp(t.logprob) if s == "no" else 0
        p = y / (y + n) if y + n else 0.5
        if not r["text"].strip():  # attacker produced nothing usable: carries no claim -> not preserved
            p = 0.0
        res.append(dict(id=r["id"], attack=r["attack"], p_preserved=p))
        agg[r["attack"]].append(p >= 0.5)
    write_jsonl(ROOT / f"data/attacks/{a.dataset}.valid.jsonl", res)
    for k, v in agg.items():
        print(f"{a.dataset} {k:12s} preserved {sum(v) / len(v):.3f}")


if __name__ == "__main__":
    # Preservation is an analysis step: if it crashes, report loudly (the watcher greps ^FAILED) but exit 0 so the
    # downstream detection jobs (afterok dependencies) still run; it can be re-run on its own.
    import sys, traceback
    try:
        main()
    except Exception:
        traceback.print_exc()
        print(f"FAILED_SOFT: validate_attacks {' '.join(sys.argv[1:])}", flush=True)
