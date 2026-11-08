"""ProgramFC baseline (Pan et al., ACL 2023), open-book setting, run with the OFFICIAL code where possible.

Official pieces used unchanged (baselines/programfc_official, vendored from github.com/teacherpeterpan/ProgramFC):
  * FEVEROUS_PROGRAM_FC few-shot prompt (the binary supports/refutes prompt set), claim inserted at [[CLAIM]]
  * Program_Execution.get_command_type / parse_verify_command / parse_question_command /
    derive_final_answer / map_direct_answer_to_label, and the per-program control flow of parse_program
    (including final_answer reset per command and random fallbacks on errors)
  * T5_Question_Answering.answer_verify_question / answer_question_directly prompt strings (captured from the
    official methods), FLAN-T5-XL as the executor, greedy decoding, max_new_tokens 8 / 32
  * evidence = top-5 BM25 hits (k1=0.9, b=0.4) joined by newlines, cut at 3,000 words
Adaptations (stated in the paper): the program generator is Llama-3.1-8B-Instruct (Codex is retired) in completion
mode with the official stop word "# The claim is", temperature 0, one program per claim; the corpus is the
training-split archive's evidence passages (claim records excluded); FLAN-T5 calls are batched (same outputs as
one-by-one greedy decoding up to numerical noise).
Prediction: supports -> true (0), refutes -> false (1). Hard labels: no threshold calibration.

usage: python programfc_run.py --dataset pubhealth
"""
import argparse, json, random, sys, types
from pathlib import Path

from common import ROOT, claims, attacks, kb, write_jsonl, metrics, check_preds

OFF = Path(__file__).resolve().parent / "baselines" / "programfc_official"
sys.path.insert(0, str(OFF))
# the official module imports a Pyserini retriever and an evaluation helper at import time; we provide our own
# retrieval (bm25.py) and evaluation, so register inert stand-ins for those two imports only.
sys.modules["retriever"] = types.SimpleNamespace(PyseriniRetriever=None)
sys.modules["evaluate"] = types.SimpleNamespace(print_evaluation_results=None)
from prompts import FEVEROUS_PROGRAM_FC          # noqa: E402  (official)
from program_execution import Program_Execution  # noqa: E402  (official)
from question_answering import T5_Question_Answering  # noqa: E402  (official)

LLAMA = "meta-llama/Meta-Llama-3.1-8B-Instruct"


class _Capture:
    """Use the official QA methods only to build their exact prompt strings."""
    def __init__(self):
        self.qa = T5_Question_Answering(None, None)
        self.last = None
        self.qa.generate = self._gen

    def _gen(self, input_string, **kw):
        self.last = (input_string, kw.get("max_new_tokens"))
        return [""]

    def verify(self, claim, evidence):
        self.qa.answer_verify_question(claim, evidence, False)
        return self.last

    def question(self, q, evidence):
        self.qa.answer_question_directly(q, evidence, False)
        return self.last


def run_programs(programs, retrieve, t5_generate, seed=0):
    """Lock-step execution of all programs with the official command semantics.
    programs: list of list-of-lines; retrieve(texts)->list of evidence strings; t5_generate(prompts, max_new)->answers.
    Returns list of final answers (True / False / None) exactly as Program_Execution.parse_program would."""
    random.seed(seed)
    PE = Program_Execution.__new__(Program_Execution)  # only its pure parsing helpers are used
    cap = _Capture()
    st = [dict(prog=p, i=0, vm={}, final=None) for p in programs]
    while True:
        pending = []
        for k, s in enumerate(st):
            while s["i"] < len(s["prog"]):
                cmd = s["prog"][s["i"]]
                s["final"] = None                      # official: final_answer = None at every command
                ctype = PE.get_command_type(cmd)
                if ctype in ("VERIFY", "QUESTION"):
                    try:
                        if ctype == "VERIFY":
                            var, text = PE.parse_verify_command(cmd, s["vm"])
                        else:
                            var, text = PE.parse_question_command(cmd, s["vm"])
                        pending.append((k, ctype, var, text))
                    except Exception:
                        s["error"] = True; s["i"] = len(s["prog"])
                    break
                if ctype == "FINAL":
                    try:
                        s["final"] = PE.derive_final_answer(cmd, s["vm"])
                    except Exception:
                        s["final"] = random.sample([True, False], 1)[0]
                s["i"] += 1
        if not pending:
            break
        evid = retrieve([t for _, _, _, t in pending])
        prompts, maxnew = [], []
        for (k, ctype, var, text), ev in zip(pending, evid):
            p, m = cap.verify(text, ev) if ctype == "VERIFY" else cap.question(text, ev)
            prompts.append(p); maxnew.append(m)
        answers = [None] * len(prompts)
        for m in sorted(set(maxnew)):
            idx = [j for j, x in enumerate(maxnew) if x == m]
            for j, a in zip(idx, t5_generate([prompts[j] for j in idx], m)):
                answers[j] = a
        for (k, ctype, var, text), a in zip(pending, answers):
            s = st[k]
            s["vm"][var] = PE.map_direct_answer_to_label(a) if ctype == "VERIFY" else a
            s["i"] += 1
    out = []
    for s in st:
        if s.get("error"):  # official: execution error -> random prediction
            out.append(random.sample([True, False], 1)[0])
        else:
            out.append(s["final"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--t5", default="google/flan-t5-xl")
    a = ap.parse_args()
    test = attacks(a.dataset)
    if a.limit:
        keep = {r["id"] for r in test[: a.limit]}
        test = [r for r in test if r["id"] in keep]
    dev = [dict(id=c["id"], label=c["label"], attack="DEV_CLEAN", query=c["claim"]) for c in claims(a.dataset, "dev")]
    Q = dev + test

    # 1) program generation (official prompt, completion mode, official stop word, temperature 0)
    from vllm import LLM, SamplingParams
    llm = LLM(LLAMA, gpu_memory_utilization=0.6, max_model_len=8192, seed=0)
    prompts = [FEVEROUS_PROGRAM_FC.replace("[[CLAIM]]", r["query"]) for r in Q]
    outs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=1024, stop=["# The claim is"]), use_tqdm=True)
    programs = [[op.strip() for op in o.outputs[0].text.strip().split("\n")] for o in outs]  # official update_results
    del llm
    import gc, torch
    gc.collect(); torch.cuda.empty_cache()

    # 2) retrieval corpus: training-archive evidence passages
    from bm25 import BM25
    docs = [r["text"] for r in kb(a.dataset) if r["type"] == "evidence"]
    bm = BM25(docs, k1=0.9, b=0.4)

    def retrieve(texts):
        res = []
        for hits in bm.topk(texts, 5):
            ev = "\n".join(docs[h].strip() for h in hits)
            w = ev.split()
            res.append(" ".join(w[:3000]) if len(w) > 3000 else ev)  # official max_evidence_length
        return res

    # 3) executor: FLAN-T5-XL, greedy, batched
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    tok = T5Tokenizer.from_pretrained(a.t5)
    t5 = T5ForConditionalGeneration.from_pretrained(a.t5, torch_dtype=torch.bfloat16).cuda().eval()

    def t5_generate(texts, max_new, bs=16):
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        res = [None] * len(texts)
        for s in range(0, len(order), bs):
            ids = order[s:s + bs]
            enc = tok([texts[i] for i in ids], return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                g = t5.generate(**enc, max_length=None, max_new_tokens=max_new)
            for i, t in zip(ids, tok.batch_decode(g, skip_special_tokens=True)):
                res[i] = t.strip()
        return res

    finals = run_programs(programs, retrieve, t5_generate)
    rows = []
    for r, prog, f in zip(Q, programs, finals):
        supports = (f is True)  # official: true_count > false_count with one program -> True only if f is True
        pred = 0 if supports else 1
        rows.append(dict(id=r["id"], attack=r["attack"], label=r["label"], p_false=float(pred), pred=pred,
                         pred05=pred, program=prog, final=f))
    if not a.limit:
        check_preds(a.dataset, rows)
    write_jsonl(ROOT / f"results/preds/{a.dataset}/programfc.jsonl", rows)
    by = {}
    for r in rows:
        by.setdefault(r["attack"], []).append(r)
    for att, rs in by.items():
        m = metrics([r["label"] for r in rs], [r["pred"] for r in rs])
        print(f"programfc {att:12s} acc={m['acc']:.3f} mF1={m['macro_f1']:.3f} falseAcc={m['false_acc']:.3f} "
              f"trueAcc={m['true_acc']:.3f}", flush=True)


if __name__ == "__main__":
    main()
