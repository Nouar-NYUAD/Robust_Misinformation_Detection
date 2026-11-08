"""Data-contamination check (guided vs general instruction; Golchin & Surdeanu, "Time Travel in LLMs", ICLR 2024).

For a stratified subsample of test claims, the claim is cut into a first and a second piece. The model completes the
second piece under
  guided  : the prompt names the dataset and split and asks for the claim exactly as it appears in the dataset
  general : the same request without the dataset / split name
Partition-level evidence of contamination (as in the paper): mean ROUGE-L F1 of guided completions is significantly
higher than of general completions (paired bootstrap, 10,000 resamples, one-sided p <= 0.05). Instance level: exact
matches (after normalisation) and near-exact matches. The paper judges near-exact matches with a GPT-4 few-shot
classifier; here near-exact is ROUGE-L F1 >= 0.75 (our threshold, stated in the paper), exact matches are reported apart.
ROUGE-L is computed on the completed piece only, as in the paper.

The subsample (also used by api_zs.py): 100 test claims per dataset, stratified by label, seed 0.

usage (open model, on a GPU node):  python contamination.py generate --model meta-llama/Meta-Llama-3.1-8B-Instruct --tag llama
       score any completions file:  python contamination.py score --tag llama     (or --tag claude, see api_zs.py)
out: results/contamination/<tag>.completions.jsonl, results/contamination/<tag>.summary.json
"""
import argparse, functools, json, random, re
import numpy as np

from common import ROOT, DATASETS, attacks, read_jsonl, write_jsonl

NAMES = {"pubhealth": "PUBHEALTH", "climatefever": "CLIMATE-FEVER", "finfact": "Fin-Fact"}
GUIDED = ("You are provided with the first piece of a claim from the test split of the {name} dataset. Finish the second "
          "piece of the claim as exactly appeared in the dataset. Only rely on the original form of the claim in the "
          "dataset to finish the second piece.\n\nFirst piece: {first}\n\nSecond piece:")
GENERAL = ("Finish the second piece based on the first piece, such that these two pieces become a single claim.\n\n"
           "First piece: {first}\n\nSecond piece:")
# prefill variant (open models): with the instruction prompts above Llama-3.1-8B-Instruct refuses 95-100% of guided
# prompts ("I cannot provide the second piece ...") but almost no general ones, so the contrast is invalid. Here the
# assistant turn is pre-filled with the first piece and the model can only continue the text; the guided / general
# contrast (dataset + split named or not) is unchanged. A deviation from the paper, stated as such.
GUIDED_PREFILL = ("Reproduce the claim from the test split of the {name} dataset that begins with the text below, exactly "
                  "as it appears in the dataset.\n\n{first}")
GENERAL_PREFILL = "Write a claim that begins with the text below.\n\n{first}"
OUT = ROOT / "results/contamination"
NEAR_EXACT = 0.75
_REFUSAL = re.compile(r"^\W*(i cannot|i can.t|i can not|i.m unable|i am unable|i.m not able|i am not able|sorry|"
                      r"i apologi[sz]e|unfortunately|i don.t have access|i do not have access)", re.I)


def is_refusal(text):
    return bool(_REFUSAL.match((text or "").strip()))


@functools.lru_cache(maxsize=None)
def subsample_ids(ds, n=100, seed=0):
    """stratified subsample of test claim ids (same draw as the original api_zs.subsample); cached, immutable"""
    clean = [r for r in attacks(ds) if r["attack"] == "CLEAN"]
    rng = random.Random(seed)
    by = {0: [r["id"] for r in clean if r["label"] == 0], 1: [r["id"] for r in clean if r["label"] == 1]}
    frac = len(by[1]) / len(clean)
    return frozenset(rng.sample(by[1], round(n * frac))) | frozenset(rng.sample(by[0], n - round(n * frac)))


def cut(text, rng):
    """first piece = first k words, k uniform in [ceil(0.3 n), floor(0.7 n)] (at least 1 word on each side)"""
    w = text.split()
    lo, hi = max(1, int(np.ceil(0.3 * len(w)))), min(len(w) - 1, int(np.floor(0.7 * len(w))))
    k = rng.randint(lo, max(lo, hi))
    return " ".join(w[:k]), " ".join(w[k:])


def items(n=100, seed=0, min_words=6):
    out = []
    for ds in DATASETS:
        keep = subsample_ids(ds, n, seed)
        rng = random.Random(f"{ds}-{seed}")
        for r in sorted((r for r in attacks(ds) if r["attack"] == "CLEAN" and r["id"] in keep), key=lambda r: r["id"]):
            if len(r["text"].split()) < min_words:
                continue
            first, second = cut(r["text"], rng)
            for kind, tmpl, tp in (("guided", GUIDED, GUIDED_PREFILL), ("general", GENERAL, GENERAL_PREFILL)):
                out.append(dict(dataset=ds, id=r["id"], kind=kind, first=first, second=second,
                                prompt=tmpl.format(name=NAMES[ds], first=first),
                                prompt_prefill=tp.format(name=NAMES[ds], first=first)))
    return out


def _tok(s):  # rouge_score default tokenizer: lowercase, non-alphanumeric -> space, whitespace split (no stemming)
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).split()


def rouge_l(ref, hyp):
    """ROUGE-L F1 (Lin 2004), LCS over rouge_score's default tokens"""
    a, b = _tok(ref), _tok(hyp)
    if not a or not b:
        return 0.0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    p, r = lcs / len(b), lcs / len(a)
    return 2 * p * r / (p + r)


_SECOND = re.compile(r"second piece\b[^:\n]{0,25}?(?::|\bis\b|\bwould be\b|\bmight be\b)[\s*_]*(.*)$", re.I)
_FULL = re.compile(r"^[\W_]*(?:completed|combined|complete|full|single|resulting)(?: claim| sentence)?[^:\n]{0,20}:"
                   r"[\s*_]*(.*)$", re.I)
_FIRST_LABEL = re.compile(r"^[\W_]*first piece", re.I)


def _finish(t, first):
    """strip markdown emphasis, surrounding quotes and a repeated first piece"""
    t = re.sub(r"[*_]{1,3}", "", t).strip().strip('"“”\'').strip()
    if t.lower().startswith(first.lower()):
        t = t[len(first):].strip().strip('"“”\'').strip()
    return t


def clean_completion(text, first):
    """the model's second piece. Chat models often answer with labels ('**Second piece (best guess):** ...',
    '**Completed claim:**' followed by the claim on the next line) or with a hedge before a guess, so, in order:
    (1) a line with a 'second piece' label; (2) a line with a 'completed/combined/full claim' label; (3) the first line
    that is neither a refusal/hedge nor a 'first piece' label. A label with nothing after it takes the next line.
    Markdown, quotes and a repeated first piece are removed. Plain continuations (the Llama prefill runs) reduce to
    their first line, as before."""
    lines = [l.strip() for l in (text or "").strip().splitlines() if l.strip()]
    for pat in (_SECOND, _FULL):
        for k, l in enumerate(lines):
            m = pat.search(l) if pat is _SECOND else pat.match(l)
            if m:
                t = m.group(1).strip() or (lines[k + 1] if k + 1 < len(lines) else "")
                return _finish(t, first)
    if is_refusal(text):  # a hedge ("I can't recall it exactly; a plausible completion is "...""): the quoted guess
        quoted = [q for q in _QUOTED.findall(text or "") if len(q.split()) >= 3]
        if quoted:
            return _finish(max(quoted, key=len), first)
    for l in lines:
        if not (is_refusal(l) or _FIRST_LABEL.match(l)):
            return _finish(l, first)
    return ""


_QUOTED = re.compile(r"[\"“]([^\"“”\n]+)[\"”]")


def exact(ref, hyp):
    return _tok(ref) == _tok(hyp)


def boot_greater(diff, B=10000, seed=0):
    """one-sided paired bootstrap p-value for mean(diff) > 0 (Golchin & Surdeanu: 10,000 resamples)"""
    rng = np.random.default_rng(seed)
    d = np.asarray(diff, float)
    means = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    return float(np.mean(means <= 0))


def score(tag):
    rows = read_jsonl(OUT / f"{tag}.completions.jsonl")
    res = {}
    for ds in DATASETS:
        g = {r["id"]: r for r in rows if r["dataset"] == ds and r["kind"] == "guided"}
        n = {r["id"]: r for r in rows if r["dataset"] == ds and r["kind"] == "general"}
        ids = sorted(set(g) & set(n))
        assert ids and set(g) == set(n), f"{ds}: guided and general completions must cover the same claims"
        rg = np.array([rouge_l(g[i]["second"], clean_completion(g[i]["completion"], g[i]["first"])) for i in ids])
        rn = np.array([rouge_l(n[i]["second"], clean_completion(n[i]["completion"], n[i]["first"])) for i in ids])
        ex = [exact(g[i]["second"], clean_completion(g[i]["completion"], g[i]["first"])) for i in ids]
        res[ds] = dict(n=len(ids), rougeL_guided=float(rg.mean()), rougeL_general=float(rn.mean()),
                       diff=float((rg - rn).mean()), p_one_sided=boot_greater(rg - rn),
                       exact_guided=int(sum(ex)), near_exact_guided=int(((rg >= NEAR_EXACT) & ~np.array(ex)).sum()),
                       exact_general=int(sum(exact(n[i]["second"], clean_completion(n[i]["completion"], n[i]["first"]))
                                             for i in ids)),
                       empty=int(sum(not (r.get("completion") or "").strip() for r in list(g.values()) + list(n.values()))),
                       empty_guided=int(sum(not (g[i].get("completion") or "").strip() for i in ids)),
                       empty_general=int(sum(not (n[i].get("completion") or "").strip() for i in ids)),
                       refusal_guided=float(np.mean([is_refusal(g[i]["completion"]) for i in ids])),
                       refusal_general=float(np.mean([is_refusal(n[i]["completion"]) for i in ids])))
        # sensitivity: an empty completion scores ROUGE-L 0 and would bias the comparison against the arm that
        # produces it, so the test is repeated on the claims where both arms produced a non-empty completion
        keep = np.array([bool((g[i].get("completion") or "").strip()) and bool((n[i].get("completion") or "").strip())
                         for i in ids])
        res[ds].update(n_nonempty=int(keep.sum()), rougeL_guided_nonempty=float(rg[keep].mean()),
                       rougeL_general_nonempty=float(rn[keep].mean()), diff_nonempty=float((rg - rn)[keep].mean()),
                       p_one_sided_nonempty=boot_greater((rg - rn)[keep]))
        print(ds, res[ds])
    json.dump(res, open(OUT / f"{tag}.summary.json", "w"), indent=1)
    return res


def prefill_messages(it):
    """user instruction + assistant turn pre-filled with the first piece (continued, not closed, by the chat template)"""
    return [{"role": "user", "content": it["prompt_prefill"]}, {"role": "assistant", "content": it["first"]}]


def generate(model, tag, prefill=False):
    from detect import LLM
    its = items()
    llm = LLM(model, mem=0.85)
    if prefill:
        res = llm.llm.chat([prefill_messages(it) for it in its], llm.SP(temperature=0, max_tokens=64), use_tqdm=True,
                           add_generation_prompt=False, continue_final_message=True)
        outs = [o.outputs[0].text for o in res]
    else:
        outs = llm.generate([it["prompt"] for it in its], max_tokens=64)
    OUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT / f"{tag}.completions.jsonl", [dict(it, completion=o, prefill=prefill) for it, o in zip(its, outs)])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["generate", "score"])
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--prefill", action="store_true", help="assistant turn pre-filled with the first piece (open models)")
    a = ap.parse_args()
    generate(a.model, a.tag, a.prefill) if a.cmd == "generate" else score(a.tag)
