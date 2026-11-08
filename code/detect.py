"""LLM / RAG misinformation detectors evaluated on clean and attacked test queries.

methods
  zs          zero-shot LLM ("Does this text contain misinformation?")
  rag         standard RAG: dense retrieval with the raw input, top-k records + verdicts, LLM verdict
  minicheck   Bespoke-MiniCheck-7B over top-k retrieved evidence passages
  ours        UNWRAP: canonicalize -> multi-view hybrid retrieval (BGE-M3 dense+sparse, RRF) ->
              cross-encoder rerank -> verdict-aware verification
              --encoder air  -> UNWRAP + attack-invariant retrieval training (train_air.py)
              ablation flags: --no_canon --no_sparse --no_rerank --encoder me5

--canon_input ours|norm  feeds a baseline the cached canonical claims (canonicalize.py) instead of the raw
                         input (tests canonicalization as a plug-in defence for existing detectors)
--canon_kind norm        UNWRAP with generic claim normalization instead of our canonicalization

Every detector is calibrated on CLEAN dev claims only (threshold maximising macro-F1); attacked
data are never used for training or calibration.

usage: python detect.py --dataset pubhealth --method ours [--llm meta-llama/Meta-Llama-3.1-8B-Instruct]
"""
import argparse, json, math, re
import numpy as np

from common import ROOT, claims, attacks, write_jsonl, metrics, choose_threshold, yes_prob_from_logprobs, check_preds

LLAMA = "meta-llama/Meta-Llama-3.1-8B-Instruct"

ZS_PROMPT = "Does this text contain misinformation? Answer with 'Yes' or 'No' only.\n\nText: {q}"

RAG_PROMPT = (
    "You are a fact-checking assistant. Decide whether the user text spreads false information, using the "
    "fact-checked records below (verdicts were assigned by professional fact-checkers).\n\n{records}\n\n"
    "User text:\n{q}\n\n"
    "If the user text makes the same claim as a record verified FALSE, it spreads false information. If it makes "
    "the same claim as a record verified TRUE, it does not. If it matches no record, judge using the evidence.\n"
    "Does the user text spread false information? Answer with 'Yes' or 'No' only.")

VERIFY_PROMPT = (
    "You are a fact-checking assistant. Below are claims extracted from a user's text and fact-checked records "
    "retrieved from an archive (verdicts were assigned by professional fact-checkers).\n\n{records}\n\n"
    "Claims extracted from the user's text:\n{claims}\n\n"
    "If the central claim is the same claim as a record verified FALSE (same event, entity and assertion, even "
    "if worded or framed differently), the text spreads false information. If it is the same claim as a record "
    "verified TRUE, it does not. If it matches no record, judge using the evidence.\n"
    "Does the user's text spread false information? Answer with 'Yes' or 'No' only.")

# Bespoke-MiniCheck-7B, reproduced from the official implementation (github.com/Liyan06/MiniCheck,
# minicheck/utils.py and minicheck/inference.py): system + user message through the model's chat template,
# trust_remote_code (InternLM2), greedy first token, support score = sum of the probabilities of top-5 first tokens
# whose decoded text lower-cases to exactly "yes" (not normalised).
MINICHECK_SYSTEM = ("Determine whether the provided claim is consistent with the corresponding document. Consistency "
                    "in this context implies that all information presented in the claim is substantiated by the "
                    "document. If not, it should be considered inconsistent. Please assess the claim's consistency "
                    "with the document by responding with either \"Yes\" or \"No\".")
MINICHECK_USER = "Document: [DOCUMENT]\nClaim: [CLAIM]"


def trunc(t, n):
    w = t.split()
    return " ".join(w[:n]) + (" ..." if len(w) > n else "")


class LLM:
    def __init__(self, model, mem=0.55, max_len=8192, trust_remote_code=False, lora=None):
        from vllm import LLM as V, SamplingParams
        kw = dict(enable_lora=True, max_lora_rank=64) if lora else {}
        self.llm = V(model, gpu_memory_utilization=mem, max_model_len=max_len, seed=0, enforce_eager=False,
                     trust_remote_code=trust_remote_code, **kw)
        self.lora = None
        if lora:  # trained verifier adapter (train_verifier.py)
            from vllm.lora.request import LoRARequest
            self.lora = LoRARequest("verifier", 1, lora)
        self.tok = self.llm.get_tokenizer()
        self.SP = SamplingParams
        self.max_len = max_len

    def _fit(self, msgs, reserve=300):
        # keep prompt + chat template + generation inside the context window by trimming the user text from
        # the middle (rare: only very long serialised summaries get near the limit)
        budget = self.max_len - reserve
        for m in msgs:
            ids = self.tok(m[-1]["content"], add_special_tokens=False).input_ids
            if len(ids) > budget:
                m[-1]["content"] = (self.tok.decode(ids[: budget // 2]) + " ... " +
                                    self.tok.decode(ids[-(budget // 2):]))
        return msgs

    def yes_prob(self, prompts):
        msgs = self._fit([[{"role": "user", "content": p}] for p in prompts])
        outs = self.llm.chat(msgs, self.SP(temperature=0, max_tokens=1, logprobs=20), use_tqdm=True,
                             lora_request=self.lora)
        return [yes_prob_from_logprobs((t.decoded_token, t.logprob) for t in o.outputs[0].logprobs[0].values())
                for o in outs]

    def minicheck_support(self, docs, claims_):
        """Official Bespoke-MiniCheck scoring (see MINICHECK_* above)."""
        msgs = [[{"role": "system", "content": MINICHECK_SYSTEM},
                 {"role": "user", "content": MINICHECK_USER.replace("[DOCUMENT]", d).replace("[CLAIM]", c)}]
                for d, c in zip(docs, claims_)]
        outs = self.llm.chat(self._fit(msgs), self.SP(temperature=0, max_tokens=1, logprobs=5), use_tqdm=True)
        return [sum(math.exp(t.logprob) for t in o.outputs[0].logprobs[0].values()
                    if (t.decoded_token or "").lower() == "yes") for o in outs]

    def generate(self, prompts, max_tokens=200):
        msgs = self._fit([[{"role": "user", "content": p}] for p in prompts])
        outs = self.llm.chat(msgs, self.SP(temperature=0, max_tokens=max_tokens), use_tqdm=True)
        return [o.outputs[0].text.strip() for o in outs]


def fmt_records(idx, recs):
    lines = []
    for k, (ci, di) in enumerate(recs, 1):
        c = idx.claim_rec[ci]
        ev = idx.rows[di]["text"] if idx.rows[di]["type"] == "evidence" else ""  # di = best evidence passage
        lines.append(f"[{k}] Claim: {c['claim']}\n    Verdict: {c['label_str'].upper()}"
                     + (f"\n    Evidence: {trunc(ev, 80)}" if ev else ""))
    return "\n".join(lines)


def rerank_order(cand, own, scores, top=20):
    """cand[q] = candidate list; own[i] = (q, k) for the i-th scored pair (k = position in cand[q]); scores[i] =
    cross-encoder score. Each candidate gets its max score over its pairs; the first `top` candidates of each
    query are reordered by that score (desc, stable); candidates beyond `top` are dropped."""
    best = {}
    for (q, k), s in zip(own, scores):
        best[(q, k)] = max(best.get((q, k), -np.inf), float(s))
    return [[c for k, c in sorted(enumerate(cs[:top]), key=lambda kc: (-best.get((q, kc[0]), -np.inf), kc[0]))]
            for q, cs in enumerate(cand)]


def retrieve_candidates(ds, views, enc_name, sparse=True, rerank=True, exclude=None):
    """Multi-view retrieval (+ optional cross-encoder reranking) over the dataset's archive.
    views[q] = [raw input, canonical claim 1, ...]; exclude[q] = archive claim index q may not retrieve (verifier
    training data only). Returns (index, candidates) with candidates[q] = [(claim_idx, evidence_row), ...]."""
    import gc, torch
    from retrieval import Index, Encoder, fuse_views
    idx = Index(ds, enc_name)
    enc = Encoder(enc_name)
    flat = [v for vs in views for v in vs]
    owner = np.repeat(np.arange(len(views)), [len(v) for v in views])
    qd, qsp = enc.encode(flat, is_query=True, max_length=2048)
    cand = fuse_views(idx, qd, qsp if sparse else None, owner, len(views), exclude=exclude)  # tested in tests.py
    del enc; gc.collect(); torch.cuda.empty_cache()
    if rerank:
        from FlagEmbedding import FlagReranker
        rr = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True, devices=["cuda"])
        pairs, own = [], []
        for qi, cands in enumerate(cand):
            anchors = views[qi][1:] or views[qi][:1]
            for k, (ci, di) in enumerate(cands[:20]):
                doc = idx.claim_rec[ci]["claim"] + " " + trunc(idx.rows[di]["text"], 120)
                for an in anchors:
                    pairs.append((trunc(an, 200), doc)); own.append((qi, k))
        sc = rr.compute_score(pairs, batch_size=128, max_length=512) if pairs else []
        if not isinstance(sc, list):
            sc = [sc]
        assert len(sc) == len(pairs)
        cand = rerank_order(cand, own, sc, top=20)
        del rr; gc.collect(); torch.cuda.empty_cache()
    return idx, cand


def verify_prompt(idx, views_q, cand_q, k=5):
    claims_txt = "\n".join(f"- {x}" for x in (views_q[1:] or views_q[:1]))
    return VERIFY_PROMPT.format(records=fmt_records(idx, cand_q[:k]), claims=claims_txt)


def load_canon(ds, kind):
    from common import read_jsonl
    return {(r["id"], r["attack"]): r["claims"] for r in read_jsonl(ROOT / f"data/canon/{ds}.{kind}.jsonl")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--method", required=True, choices=["zs", "rag", "minicheck", "ours"])
    ap.add_argument("--llm", default=LLAMA)
    ap.add_argument("--encoder", default=None)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--no_canon", action="store_true")
    ap.add_argument("--canon_kind", default="ours", choices=["ours", "norm"])
    ap.add_argument("--canon_input", default=None, choices=["ours", "norm"])
    ap.add_argument("--no_sparse", action="store_true")
    ap.add_argument("--no_rerank", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--verifier_lora", default=None, help="LoRA adapter of the trained verifier (train_verifier.py)")
    a = ap.parse_args()

    test = attacks(a.dataset)
    if a.limit:
        keep = {r["id"] for r in test[: a.limit]}
        test = [r for r in test if r["id"] in keep]
    dev = [dict(id=c["id"], label=c["label"], attack="DEV_CLEAN", query=c["claim"]) for c in claims(a.dataset, "dev")]
    Q = dev + test
    qs = [r["query"] for r in Q]
    if a.canon_input:  # baseline sees canonical claims instead of the raw (possibly attacked) input
        cc = load_canon(a.dataset, a.canon_input)
        qs = [" ".join(cc[(r["id"], r["attack"])]) or r["query"] for r in Q]
    name = a.method + (f"_{a.tag}" if a.tag else "")
    extra = {}

    if a.method == "zs":
        llm = LLM(a.llm)
        p = llm.yes_prob([ZS_PROMPT.format(q=q) for q in qs])
    elif a.method == "minicheck":
        a.llm = "bespokelabs/Bespoke-MiniCheck-7B"
    if a.method in ("rag", "minicheck", "ours"):
        enc_name = a.encoder or ("bgem3" if a.method == "ours" else "me5")
        views = [[q] for q in qs]
        if a.method == "ours" and not a.no_canon:
            cc = load_canon(a.dataset, a.canon_kind)
            cl = [cc[(r["id"], r["attack"])] for r in Q]
            views = [[q] + c for q, c in zip(qs, cl)]
            extra["canon"] = cl
        idx, cand = retrieve_candidates(a.dataset, views, enc_name, sparse=not a.no_sparse,
                                        rerank=(a.method == "ours" and not a.no_rerank))
        llm = LLM(a.llm, trust_remote_code=(a.method == "minicheck"), lora=a.verifier_lora)

        if a.method == "minicheck":
            docs = ["\n".join(trunc(idx.rows[di]["text"], 150) for _, di in c[:3]) for c in cand]
            p = [1 - s for s in llm.minicheck_support(docs, qs)]  # not supported by the evidence -> false
        else:
            if a.method == "rag":
                prompts = [RAG_PROMPT.format(records=fmt_records(idx, c[: a.k]), q=q) for q, c in zip(qs, cand)]
            else:
                prompts = [verify_prompt(idx, v, c, a.k) for v, c in zip(views, cand)]
            p = llm.yes_prob(prompts)
        extra["top_claims"] = [[idx.claim_ids[x[0]] for x in c[: a.k]] for c in cand]

    # calibrate on clean dev
    assert len(p) == len(Q)
    thr = choose_threshold([r["label"] for r in Q[: len(dev)]], p[: len(dev)])
    rows = []
    for i, r in enumerate(Q):
        o = dict(id=r["id"], attack=r["attack"], label=r["label"], p_false=float(p[i]),
                 pred=int(p[i] >= thr), pred05=int(p[i] >= 0.5))
        for k, v in extra.items():
            o[k] = v[i]
        rows.append(o)
    if not a.limit:
        check_preds(a.dataset, rows)
    out = ROOT / f"results/preds/{a.dataset}/{name}.jsonl"
    write_jsonl(out, rows)
    json.dump(dict(threshold=float(thr), args=vars(a)), open(str(out).replace(".jsonl", ".meta.json"), "w"))
    by = {}
    for r in rows:
        by.setdefault(r["attack"], []).append(r)
    for att, rs in by.items():
        m = metrics([r["label"] for r in rs], [r["pred"] for r in rs])
        h = ""
        print(f"{name:20s} {att:12s} acc={m['acc']:.3f} mF1={m['macro_f1']:.3f} falseAcc={m['false_acc']:.3f} "
              f"trueAcc={m['true_acc']:.3f}{h}", flush=True)


if __name__ == "__main__":
    main()
