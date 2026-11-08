"""Frontier-model reference: Claude Opus 5.5 via the Message Batches API (50% price).

1. zero-shot detection with the same prompt as the open-model zero-shot baseline (detect.ZS_PROMPT), on the stratified
   subsample of contamination.subsample_ids (100 test claims per dataset) x all 16 conditions (CLEAN + 15 attacks).
   The output is a hard Yes/No (no logprobs), so no threshold is calibrated; refusals / unparsable answers are pred=None
   and are counted and reported.
2. contamination check (contamination.py): guided and general completion prompts for the same claims.

Opus 5.5 always thinks (thinking cannot be disabled); effort "low" keeps it short. Thinking tokens are billed as output.

usage:  python api_zs.py pilot [--k 30]   synchronous sample of the real requests -> measured tokens and cost estimate
        python api_zs.py submit           -> results/api/batch_id.txt, results/api/index.json
        python api_zs.py collect          -> results/preds_api/<ds>/zs_claude.jsonl, results/contamination/claude.completions.jsonl
"""
import argparse, json, random, re, sys
from collections import Counter

from common import ROOT, DATASETS, attacks, write_jsonl
from contamination import subsample_ids, items as ct_items, OUT as CT_OUT

MODEL = "claude-opus-5-5"
PROMPT = "Does this text contain misinformation? Answer with 'Yes' or 'No' only.\n\nText: {q}"  # = detect.ZS_PROMPT
PARAMS = dict(model=MODEL, max_tokens=2048, output_config={"effort": "low"})
PRICE_IN, PRICE_OUT = 4.0, 20.0  # USD per million tokens (standard); batches are billed at 50%
API = ROOT / "results/api"


def zs_rows(ds, n=100):
    keep = subsample_ids(ds, n)
    return [r for r in attacks(ds) if not r["attack"].startswith("ADAPT_") and r["id"] in keep]


def build():
    """(custom_id, prompt, meta) for every request of the batch"""
    out = []
    for ds in DATASETS:
        for k, r in enumerate(zs_rows(ds)):
            out.append((f"zs-{ds}-{k}", PROMPT.format(q=r["query"]),
                        dict(task="zs", dataset=ds, id=r["id"], attack=r["attack"], label=r["label"])))
    for k, it in enumerate(ct_items()):
        out.append((f"ct-{k}", it["prompt"], dict(task="ct", **{x: it[x] for x in ("dataset", "id", "kind", "first", "second", "prompt")})))
    assert len({c for c, _, _ in out}) == len(out) and all(re.fullmatch(r"[A-Za-z0-9_-]{1,64}", c) for c, _, _ in out)
    return out


def text_of(msg):
    """(text, status): refusal and missing text are distinguished"""
    if msg.stop_reason == "refusal":
        return "", "refusal"
    t = next((b.text for b in msg.content if b.type == "text"), "")
    return t, ("ok" if t.strip() else f"empty_{msg.stop_reason}")


def parse_yes_no(text):
    m = re.match(r"\W*(yes|no)\b", (text or "").strip().lower())
    return None if not m else int(m.group(1) == "yes")


def pilot(k):
    import anthropic
    client = anthropic.Anthropic()
    reqs = build()
    rng = random.Random(0)
    zs = [r for r in reqs if r[2]["task"] == "zs"]; ct = [r for r in reqs if r[2]["task"] == "ct"]
    sample = rng.sample(zs, k) + rng.sample(ct, max(2, k // 5))
    tin = tout = 0; st = Counter(); per = Counter()
    for cid, prompt, meta in sample:
        msg = client.messages.create(messages=[{"role": "user", "content": prompt}], **PARAMS)
        t, s = text_of(msg); st[s] += 1
        tin += msg.usage.input_tokens; tout += msg.usage.output_tokens
        per[meta["task"]] += 1
        print(cid, meta.get("attack", meta.get("kind")), s, repr(t[:60]), msg.usage.input_tokens, msg.usage.output_tokens)
    n_zs, n_ct = len(zs), len(ct)
    avg_in, avg_out = tin / len(sample), tout / len(sample)
    est = (n_zs + n_ct) * (avg_in * PRICE_IN + avg_out * PRICE_OUT) / 1e6 * 0.5
    spent = (tin * PRICE_IN + tout * PRICE_OUT) / 1e6
    print(f"status {dict(st)} | mean tokens in {avg_in:.0f} out {avg_out:.0f} | pilot cost ${spent:.2f} | "
          f"full batch: {n_zs} zs + {n_ct} contamination requests, estimated ${est:.2f} at batch price")


def submit():
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    client = anthropic.Anthropic()
    reqs = build()
    API.mkdir(parents=True, exist_ok=True)
    assert not (API / "batch_id.txt").exists(), "a batch was already submitted (results/api/batch_id.txt)"
    json.dump({c: m for c, _, m in reqs}, open(API / "index.json", "w"))
    batch = client.messages.batches.create(requests=[
        Request(custom_id=c, params=MessageCreateParamsNonStreaming(messages=[{"role": "user", "content": p}], **PARAMS))
        for c, p, _ in reqs])
    (API / "batch_id.txt").write_text(batch.id)
    print("submitted", batch.id, len(reqs), "requests")


def collect():
    import anthropic
    client = anthropic.Anthropic()
    bid = (API / "batch_id.txt").read_text().strip()
    batch = client.messages.batches.retrieve(bid)
    if batch.processing_status != "ended":
        print("status", batch.processing_status, batch.request_counts); sys.exit(1)
    index = json.load(open(API / "index.json"))
    zs, ct, st = {ds: [] for ds in DATASETS}, [], Counter()
    tin = tout = 0
    for res in client.messages.batches.results(bid):
        meta = index[res.custom_id]
        text, s = "", res.result.type
        if res.result.type == "succeeded":
            msg = res.result.message
            text, s = text_of(msg)
            tin += msg.usage.input_tokens; tout += msg.usage.output_tokens
        st[(meta["task"], s)] += 1
        if meta["task"] == "zs":
            pred = parse_yes_no(text)
            if s == "ok" and pred is None:
                st[("zs", "unparsed")] += 1
            zs[meta["dataset"]].append(dict(id=meta["id"], attack=meta["attack"], label=meta["label"], pred=pred,
                                            status=s, raw=text[:200], subsample=True))
        else:
            ct.append(dict({k: v for k, v in meta.items() if k != "task"}, completion=text, status=s))
    assert sum(len(v) for v in zs.values()) + len(ct) == len(index), "missing results"
    for ds, rows in zs.items():
        write_jsonl(ROOT / f"results/preds_api/{ds}/zs_claude.jsonl", rows)
    CT_OUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(CT_OUT / "claude.completions.jsonl", ct)
    cost = (tin * PRICE_IN + tout * PRICE_OUT) / 1e6 * 0.5
    json.dump(dict(batch=bid, status={f"{a}:{b}": c for (a, b), c in st.items()}, input_tokens=tin, output_tokens=tout,
                   cost_usd_batch=cost), open(API / "collect_summary.json", "w"), indent=1)
    print({f"{a}:{b}": c for (a, b), c in st.items()}, f"tokens in {tin} out {tout}, cost ${cost:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pilot", "submit", "collect"])
    ap.add_argument("--k", type=int, default=30)
    a = ap.parse_args()
    {"pilot": lambda: pilot(a.k), "submit": submit, "collect": collect}[a.cmd]()
