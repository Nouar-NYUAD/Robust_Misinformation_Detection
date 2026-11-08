"""Correctness tests: every piece of pipeline logic that produces a number or a decision is checked against an
independent reference implementation or a hand-computed answer, plus integrity checks on all data files.
CPU only.  usage: python tests.py        (exits non-zero on the first failure)
"""
import itertools, json, random, sys
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

from common import ROOT, DATASETS, ATTACK_ORDER, claims, kb, read_jsonl, metrics, choose_threshold, \
    yes_prob_from_logprobs, check_preds

rng = np.random.default_rng(0)
passed = 0


def ok(cond, msg):
    global passed
    if not cond:
        print("FAIL:", msg); sys.exit(1)
    passed += 1


# ---------------------------------------------------------------- metrics
for _ in range(300):
    n = int(rng.integers(1, 60)); y = rng.integers(0, 2, n); p = rng.integers(0, 2, n)
    m = metrics(y, p)
    ok(abs(m["macro_f1"] - f1_score(y, p, average="macro", labels=[0, 1], zero_division=0)) < 1e-12, "macro-F1 vs sklearn")
    ok(abs(m["acc"] - accuracy_score(y, p)) < 1e-12, "accuracy vs sklearn")
    if (y == 1).any():
        ok(abs(m["false_acc"] - ((p == 1) & (y == 1)).sum() / (y == 1).sum()) < 1e-12, "false-claim recall")
        ok(abs(m["asr"] - (1 - m["false_acc"])) < 1e-12, "ASR = 1 - recall on false claims")
    if (y == 0).any():
        ok(abs(m["true_acc"] - ((p == 0) & (y == 0)).sum() / (y == 0).sum()) < 1e-12, "true-claim recall")
print("metrics ok")

# ---------------------------------------------------------------- threshold selection
for _ in range(200):
    n = int(rng.integers(5, 80)); y = rng.integers(0, 2, n); p = rng.random(n).round(int(rng.integers(1, 4)))
    t = choose_threshold(y, p)
    grid = sorted(set([0.5] + list(np.quantile(p, np.linspace(0.01, 0.99, 99)))))
    best = max(f1_score(y, (p >= g).astype(int), average="macro", labels=[0, 1], zero_division=0) for g in grid)
    ok(abs(f1_score(y, (p >= t).astype(int), average="macro", labels=[0, 1], zero_division=0) - best) < 1e-12,
       "threshold reaches the best dev macro-F1 on its grid")
    ties = [g for g in grid if abs(f1_score(y, (p >= g).astype(int), average="macro", labels=[0, 1], zero_division=0) - best) < 1e-12]
    ok(abs(abs(t - 0.5) - min(abs(g - 0.5) for g in ties)) < 1e-12, "ties broken towards 0.5")
print("threshold ok")

# ---------------------------------------------------------------- yes/no probability
L = np.log
ok(abs(yes_prob_from_logprobs([("Yes", L(.6)), ("No", L(.2)), ("The", L(.2))]) - .75) < 1e-12, "yes/no basic")
ok(abs(yes_prob_from_logprobs([(" yes", L(.3)), ("YES", L(.3)), ("'No", L(.2)), ("n", L(.2))]) - .6) < 1e-12, "case/space/quote variants summed")
ok(yes_prob_from_logprobs([("Maybe", L(.9))]) == 0.5, "neither token -> 0.5")
ok(yes_prob_from_logprobs([("Not", L(.9)), ("Yesterday", L(.1))]) == 0.5, "no partial-word matches")
print("yes-prob ok")

# ---------------------------------------------------------------- claim-level retrieval, best evidence, RRF fusion
from retrieval import Index, fuse_views


class FakeIndex(Index):  # same logic as Index, numpy instead of GPU
    def __init__(self, cid, types, dense):
        self.rows = [dict(claim_id=c, type=t) for c, t in zip(cid, types)]
        self.dense, self.sparse = dense, None
        self.starts = np.array([0] + [i for i in range(1, len(cid)) if cid[i] != cid[i - 1]])
        self.claim_ids = [cid[s] for s in self.starts]

    def doc_scores(self, qd, qs=None, w_sparse=0.3):
        return qd @ self.dense.T


for trial in range(30):
    n_claims = int(rng.integers(3, 25))
    cid, types = [], []
    for c in range(n_claims):
        k = int(rng.integers(0, 5))                      # number of evidence passages (0 allowed)
        cid += [f"c{c}"] * (k + 1); types += ["claim"] + ["evidence"] * k
    D = rng.normal(size=(len(cid), 8)); idx = FakeIndex(cid, types, D)
    nq = int(rng.integers(1, 6)); views_per = rng.integers(1, 4, nq)
    owner = np.repeat(np.arange(nq), views_per); Qv = rng.normal(size=(len(owner), 8))
    S = Qv @ D.T
    # brute force
    claim_rows = {c: [i for i, x in enumerate(cid) if x == c] for c in idx.claim_ids}
    cs_ref = np.array([[S[v, claim_rows[c]].max() for c in idx.claim_ids] for v in range(len(owner))])
    ok(np.allclose(idx.claim_scores(S), cs_ref), "claim score = max over claim record + evidence")
    for v in range(len(owner)):
        for ci, c in enumerate(idx.claim_ids):
            ev = [i for i in claim_rows[c] if types[i] == "evidence"]
            want = max(ev, key=lambda i: S[v, i]) if ev else claim_rows[c][0]
            ok(idx.best_evidence(S[v], ci) == want, "best evidence passage (never the claim record if evidence exists)")
    per_view, keep = int(rng.integers(1, 6)), int(rng.integers(1, 8))
    got = fuse_views(idx, Qv, None, owner, nq, per_view=per_view, keep=keep, bs=int(rng.integers(1, 4)))
    for q in range(nq):
        score, best_v = {}, {}
        for v in np.where(owner == q)[0]:
            order = sorted(range(len(idx.claim_ids)), key=lambda ci: (-cs_ref[v, ci], ci))[:per_view]
            for r, ci in enumerate(order):
                score[ci] = score.get(ci, 0) + 1 / (60 + r)
                if ci not in best_v or cs_ref[v, ci] > cs_ref[best_v[ci], ci]:
                    best_v[ci] = v
        ref = sorted(score, key=lambda ci: (-score[ci], ci))[:keep]
        ok([c for c, _ in got[q]] == ref, "RRF fused ranking vs brute force")
        ok(all(e == idx.best_evidence(S[best_v[c]], c) for c, e in got[q]), "evidence taken from the claim's best view")
print("retrieval/fusion ok")

# exclusion (verifier training data): the excluded claim never appears; the rest equals brute force without it
for trial in range(20):
    n_claims = int(rng.integers(3, 15)); cid, types = [], []
    for c in range(n_claims):
        k = int(rng.integers(0, 4)); cid += [f"c{c}"] * (k + 1); types += ["claim"] + ["evidence"] * k
    D = rng.normal(size=(len(cid), 6)); idx = FakeIndex(cid, types, D)
    nq = int(rng.integers(1, 4)); owner = np.repeat(np.arange(nq), rng.integers(1, 3, nq)); Qv = rng.normal(size=(len(owner), 6))
    excl = [int(rng.integers(0, n_claims)) if rng.random() < 0.7 else None for _ in range(nq)]
    got = fuse_views(idx, Qv, None, owner, nq, per_view=4, keep=6, exclude=excl)
    S = Qv @ D.T; cs = idx.claim_scores(S)
    for q in range(nq):
        if excl[q] is not None:
            ok(all(c != excl[q] for c, _ in got[q]), "excluded claim never retrieved")
        score = {}
        for v in np.where(owner == q)[0]:
            order = [ci for ci in sorted(range(len(idx.claim_ids)), key=lambda ci: (-cs[v, ci], ci)) if ci != excl[q]][:4]
            for r, ci in enumerate(order):
                score[ci] = score.get(ci, 0) + 1 / (60 + r)
        ok([c for c, _ in got[q]] == sorted(score, key=lambda ci: (-score[ci], ci))[:6], "exclusion ranking vs brute force")
print("exclusion ok")

# ---------------------------------------------------------------- reranker reordering
from detect import rerank_order
for _ in range(100):
    nq = int(rng.integers(1, 4))
    cand = [[(int(c), int(c) + 100) for c in rng.permutation(30)[: int(rng.integers(0, 25))]] for _ in range(nq)]
    own, sc = [], []
    for q, cs in enumerate(cand):
        for k in range(min(20, len(cs))):
            for _a in range(int(rng.integers(1, 4))):
                own.append((q, k)); sc.append(float(rng.normal()))
    got = rerank_order(cand, own, sc, top=20)
    for q, cs in enumerate(cand):
        best = {k: max(s for (qq, kk), s in zip(own, sc) if (qq, kk) == (q, k)) for k in range(min(20, len(cs)))}
        ref = [cs[k] for k in sorted(best, key=lambda k: (-best[k], k))]
        ok(got[q] == ref, "rerank order = max score per candidate, top-20 only")
print("rerank ok")

# ---------------------------------------------------------------- canonical-claim parsing
from canonicalize import parse_claims
ok(parse_claims("Here are the claims:\n1. A is B.\n- C did D.\n\"E is F.\"\nG.\nH is I.") == ["A is B.", "C did D.", "E is F."],
   "parse: preface dropped, numbering/bullets/quotes stripped, short lines dropped, max 3")
ok(parse_claims("") == [], "parse: empty")
ok(parse_claims("Claims:\nThe vaccine causes autism.") == ["The vaccine causes autism."], "parse: 'Claims:' header")
print("parse ok")

# ---------------------------------------------------------------- BM25 (Lucene formula) vs brute force
from bm25 import BM25
import math
docs = ["The vaccine causes autism in children", "Vaccines are safe and effective", "Bitcoin price rose sharply",
        "The Federal Reserve raised interest rates", "Children vaccines schedule", "", "autism autism rates rising"]
bm = BM25(docs, k1=0.9, b=0.4)
toks = [bm.analyse(d) for d in docs]
N = len(docs); avgdl = sum(map(len, toks)) / N
for q in ["vaccine autism children", "interest rates rising", "nothing matches here", "autism autism"]:
    ref = []
    for t in toks:
        s = 0.0
        for term in bm.analyse(q):
            df = sum(term in tt for tt in toks); tf = t.count(term)
            if df == 0 or tf == 0:
                continue
            idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
            s += idf * tf * 1.9 / (tf + 0.9 * (1 - 0.4 + 0.4 * len(t) / avgdl))
        ref.append(s)
    ok(np.allclose(bm.scores([q])[0], ref), f"BM25 scores vs brute force: {q}")
    ok(bm.topk([q], 3)[0] == sorted(range(N), key=lambda i: (-ref[i], i))[:3], "BM25 top-k order")
print("bm25 ok")

# ---------------------------------------------------------------- ARG rationale verdict parsing
from arg_run import parse_pred, Q_TMPL
ok(parse_pred("Let's think ... Therefore, this message is more likely to be false. Return 0.") == 1, "ARG: return 0 -> fake")
ok(parse_pred("...so it is likely real. Return 1") == 0, "ARG: return 1 -> real")
ok(parse_pred("the answer (arabic numerals) is 1") == 0, "ARG: answer is 1 -> real")
ok(parse_pred("I cannot determine this.") == 2, "ARG: no verdict -> other")
ok(parse_pred("At first return 1 seemed right, but on reflection return 0.") == 1, "ARG: last verdict wins")
ok(Q_TMPL.startswith("Given the following message, predict its veracity. If it is more likely to be a real message, "
                     "return 1; otherwise, return 0. Please refrain from providing ambiguous assessments such as "
                     "undetermined: "), "ARG: prompt = paper Table 8 wording")
print("arg parsing ok")

# ---------------------------------------------------------------- ProgramFC: batched executor == official parse_program
import programfc_run as pfr
from program_execution import Program_Execution
from question_answering import T5_Question_Answering


def fake_answer(prompt):  # deterministic stand-in for FLAN-T5, depends only on the prompt text
    h = sum(map(ord, prompt)) % 5
    if "True or false?" in prompt:  # verify prompts: answers the official label map knows (no random fallback)
        return ["true", "false", "yes", "No", "it's impossible to say"][h]
    return ["Paris", "Obama", "1999", "the WHO", "Bitcoin"][h]


def fake_retrieve(texts):
    return [f"evidence about {t[:20]}" for t in texts]


class FakeSearcher:
    def retrieve(self, q, k):
        return [{"text": fake_retrieve([q])[0], "doc_id": 0, "score": 1.0}]


def official_final(program):
    PE = Program_Execution.__new__(Program_Execution)
    PE.args = type("A", (), {"setting": "open-book", "num_retrieved": 5, "max_evidence_length": 3000})()
    PE.searcher = FakeSearcher()
    qa = T5_Question_Answering(None, None)
    qa.generate = lambda s, **kw: [fake_answer(s)]
    PE.QA_module = qa
    try:
        return PE.parse_program("x", program, None)[0]
    except Exception:
        return "RANDOM"


cases = [
    ['def program():', 'fact_1 = Verify("A is B.")', 'fact_2 = Verify("C is D.")', 'label = Predict(fact_1 and fact_2)'],
    ['def program():', 'answer_1 = Question("Who founded X?")', 'fact_1 = Verify(f"{answer_1} founded Y.")', 'label = Predict(fact_1)'],
    ['def program():', 'fact_1 = Verify("A.")', 'label = Predict(fact_1)', ''],          # trailing line resets final
    ['def program():', 'fact_1 = Verify("A.")'],                                          # no Predict
    ['def program():', 'label = Predict(fact_9)'],                                        # unknown variable
    ['garbage', 'more garbage'],
]
for _ in range(60):  # random well-formed programs
    n = int(rng.integers(1, 4)); lines = ['def program():']
    for j in range(n):
        lines.append(f'fact_{j+1} = Verify("claim {int(rng.integers(0, 99))} holds.")')
    lines.append("label = Predict(" + " and ".join(f"fact_{j+1}" for j in range(n)) + ")")
    cases.append(lines)
got = pfr.run_programs(cases, fake_retrieve, lambda ps, m: [fake_answer(p) for p in ps])
for c, g in zip(cases, got):
    ref = official_final(c)
    if ref != "RANDOM":
        ok(g == ref, f"ProgramFC batched == official parse_program: {c}")
print("programfc ok")

# ---------------------------------------------------------------- Kotonya: evidence selection excludes the claim's own article
from kotonya_run import select_evidence, split_sentences
import torch


class FakeSB:  # bag-of-letters encoder: deterministic, similarity = shared letters
    def encode(self, texts, **kw):
        v = torch.zeros(len(texts), 26)
        for i, t in enumerate(texts):
            for ch in t.lower():
                if "a" <= ch <= "z":
                    v[i, ord(ch) - 97] += 1
        return torch.nn.functional.normalize(v, dim=1)


sents = ["zzz yyy xxx", "aaa bbb ccc", "aaa bbb ccd", "qqq www eee", "aab bba abc"]
owner = np.array(["c1", "c1", "c2", "c3", "c2"])
sel = select_evidence(["aaa bbb ccc", "aaa bbb ccc"], ["c1", None], sents, owner, FakeSB(), k=2)
ok(all(owner[j] != "c1" for j in sel[0]), "Kotonya: own article excluded for training claims")
ok(1 in sel[1], "Kotonya: test queries may use any archive sentence (none are their own)")
ok(split_sentences("Dr. Smith said so. It is false! Really? Yes it is true.") != [], "sentence splitter runs")
print("kotonya ok")

# ---------------------------------------------------------------- verifier training examples
try:
    from transformers import AutoTokenizer
    from train_verifier import encode_example, LLAMA
    tok = AutoTokenizer.from_pretrained(LLAMA)
    for ans in ("Yes", "No"):
        ids, lab = encode_example(tok, "Does it spread false information?", ans)
        p = encode_example(tok, "Does it spread false information?")
        ok(ids[: len(p)] == p, "training input starts with the exact inference prompt (chat template + header)")
        ok(all(x == -100 for x in lab[: len(p)]), "prompt tokens are masked from the loss")
        ok(tok.decode([x for x in lab if x != -100]) == ans + "<|eot_id|>", "only the answer + end-of-turn are learned")
    ref = tok.apply_chat_template([{"role": "user", "content": "x"}], add_generation_prompt=True, tokenize=True)
    ok(encode_example(tok, "x") == ref, "same chat template as vLLM chat inference")
    print("verifier encoding ok")
except OSError:
    print("verifier encoding: tokenizer not available here, skipped")

# ---------------------------------------------------------------- Setting B (attack-aware) training sets
from train_verifier import train_items
import arg_run
for ds in DATASETS:
    if not (ROOT / f"data/train_aug/{ds}.jsonl").exists():
        continue
    tr = claims(ds, "train"); aug = read_jsonl(ROOT / f"data/train_aug/{ds}.jsonl")
    lab = {c["id"]: c["label"] for c in tr}
    A, B = train_items(ds, False), train_items(ds, True)
    ok(len(A) == len(tr) and len(B) == len(tr) + len(aug), f"{ds}: Setting A = clean train only; B = clean + transformed")
    ok(B[: len(A)] == A, f"{ds}: Setting B contains Setting A unchanged")
    ok(all(y == lab[c] for _, _, c, y in B), f"{ds}: transformed copies carry their claim's label")
    ok(len({k for k, _, _, _ in B}) == len(B), f"{ds}: unique training keys")
    ok(all(c in lab for _, _, c, _ in B), f"{ds}: every training item belongs to a TRAIN claim (no dev/test)")
    ia, ib = arg_run.items(ds, False), arg_run.items(ds, True)
    ok(sum(i["split"] == "train" for i in ib) - sum(i["split"] == "train" for i in ia) == len(aug), f"{ds}: ARG Setting B adds exactly the transformed copies")
    ok([i for i in ia if i["split"] != "train"] == [i for i in ib if i["split"] != "train"], f"{ds}: ARG dev/test identical in A and B")
    ok(all(i["label"] == lab[i["key"].split("|")[0]] for i in ib if i["split"] == "train"), f"{ds}: ARG training labels")
    print(f"setting B training sets ok ({ds})")

# ---------------------------------------------------------------- check_preds catches errors
import common
for ds in DATASETS:
    if not (ROOT / f"data/attacks/{ds}.jsonl").exists():
        continue
    good = [dict(id=r["id"], attack=r["attack"], label=r["label"], pred=0) for r in common.attacks(ds)]
    good += [dict(id=c["id"], attack="DEV_CLEAN", label=c["label"], pred=1) for c in claims(ds, "dev")]
    check_preds(ds, good)
    for bad, why in ((good[:-1], "missing row"), (good + good[:1], "duplicate"),
                     ([dict(good[0], label=1 - good[0]["label"])] + good[1:], "wrong label")):
        try:
            check_preds(ds, bad); ok(False, f"check_preds missed: {why}")
        except AssertionError:
            ok(True, "")
    print(f"check_preds ok ({ds})")

# ---------------------------------------------------------------- data integrity
from gen_attacks import ATTACKS, LANGS
for ds in DATASETS:
    C = claims(ds); K = kb(ds)
    split = {c["id"]: c["split"] for c in C}; lab = {c["id"]: c["label"] for c in C}
    ok(len(split) == len(C), f"{ds}: unique claim ids")
    ok(all(split[r["claim_id"]] == "train" for r in K), f"{ds}: archive holds train claims only")
    ok(all(lab[r["claim_id"]] == r["label"] for r in K), f"{ds}: archive verdicts = benchmark labels")
    ok(all(r["label_str"] == ("false" if r["label"] else "true") for r in K + C), f"{ds}: label 1 <=> 'false'")
    blocks = [k for k, _ in itertools.groupby(r["claim_id"] for r in K)]
    ok(len(blocks) == len(set(blocks)), f"{ds}: archive blocks contiguous")
    ok(all(K[i]["type"] == "claim" for i in [0] + [j for j in range(1, len(K)) if K[j]["claim_id"] != K[j - 1]["claim_id"]]),
       f"{ds}: each block starts with its claim record")
    tr = {c["claim"].lower() for c in C if c["split"] == "train"}
    ok(not any(c["claim"].lower() in tr for c in C if c["split"] != "train"), f"{ds}: no dev/test claim text in train")
    p = ROOT / f"data/attacks/{ds}.jsonl"
    if p.exists():
        A = read_jsonl(p); test = {c["id"] for c in C if c["split"] == "test"}
        conds = {}
        for r in A:
            conds.setdefault(r["id"], []).append(r["attack"])
        import os
        if os.environ.get("SUBSET_OK"):  # smoke-test sandbox: a subset of the test claims, nothing outside it
            ok(set(conds) <= test and len(conds) > 0, f"{ds}: attacks cover a subset of the test claims only")
        else:
            ok(set(conds) == test, f"{ds}: attacks cover exactly the test claims")
        base = set(ATTACK_ORDER)
        ok(all(base <= set(v) and len(v) == len(set(v)) for v in conds.values()), f"{ds}: every test claim has all 16 conditions once")
        ok(all(r["label"] == lab[r["id"]] for r in A), f"{ds}: attack labels = benchmark labels")
        empties = []
        clean = {r["id"]: r["text"] for r in A if r["attack"] == "CLEAN"}
        ok(all(clean[i] == next(c["claim"] for c in C if c["id"] == i) for i in list(clean)[:200]), f"{ds}: CLEAN = original claim")
        for r in A:
            if r["attack"] in ATTACKS:
                ops, wrap = ATTACKS[r["attack"]]
                ok(r["ops"] == ops, f"{ds}: ops recorded")
                if r["attack"].startswith("TR_"):
                    ok(r["lang"] == r["attack"][3:], f"{ds}: TR_x language")
                ok(r["query"] == wrap.format(L=LANGS.get(r["lang"], "")) + "\n\n" + r["text"], f"{ds}: query = wrapper + text")
                if not r["text"].strip():
                    empties.append((r["id"], r["attack"]))
            elif r["attack"].startswith("ADAPT_"):
                kind, b = r["attack"][6:].split(":")
                wrap = ATTACKS[b][1].format(L=LANGS.get(r["lang"], ""))
                ok(r["query"].startswith(wrap + "\n\n") and r["query"][len(wrap) + 2:] == r["text"], f"{ds}: adaptive keeps base wrapper")
        ok(len(empties) <= 0.005 * len(A), f"{ds}: too many empty attack texts ({len(empties)})")
        pv = ROOT / f"data/attacks/{ds}.valid.jsonl"
        if pv.exists():
            V = {(r["id"], r["attack"]): r["p_preserved"] for r in read_jsonl(pv)}
            ok(set(V) == {(r["id"], r["attack"]) for r in A if r["attack"] != "CLEAN"}, f"{ds}: judge covers every attacked text")
            ok(all(V[k] < 0.5 for k in empties), f"{ds}: empty attack texts are excluded as not preserved")
        print(f"attacks ok ({ds}: {len(A)} rows, {len(empties)} empty attack texts -> excluded)")
    pa = ROOT / f"data/train_aug/{ds}.jsonl"
    if pa.exists():
        T = read_jsonl(pa)
        ok(all(split[r["id"]] == "train" for r in T), f"{ds}: AIR/adv-training augmentations from train claims only")
        ok({r["family"] for r in T} <= {"TR_fr", "TR_es", "SUM", "MCQ"}, f"{ds}: only seen attack families in training")
        print(f"train_aug ok ({ds})")
    for kind in ("ours", "norm"):
        pc = ROOT / f"data/canon/{ds}.{kind}.jsonl"
        if pc.exists() and p.exists():
            got = {(r["id"], r["attack"]) for r in read_jsonl(pc)}
            want = {(r["id"], r["attack"]) for r in read_jsonl(p)} | {(c["id"], "DEV_CLEAN") for c in C if c["split"] == "dev"}
            ok(got == want, f"{ds}: canonical cache ({kind}) covers every input exactly")
# ---------------------------------------------------------------- aggregation macro-F1 and paired significance test
import aggregate
from compare import paired_diff
from train_verifier import adapter_path, ADAPTER_DIR
for _ in range(300):
    n = int(rng.integers(1, 60)); y = rng.integers(0, 2, n); p = rng.integers(0, 2, n)
    ok(abs(aggregate.macro_f1(y, p) - f1_score(y, p, average="macro", labels=[0, 1], zero_division=0)) < 1e-12,
       "aggregate.macro_f1 vs sklearn")
ok(adapter_path() == ADAPTER_DIR and adapter_path(True) == ADAPTER_DIR + "_aug" and adapter_path(False, 2) == ADAPTER_DIR + "_s2"
   and adapter_path(True, 1) == ADAPTER_DIR + "_aug_s1", "verifier adapter paths: seed 0 = original path")


def fake_R(ids_by_cond, y_of, preds_by_seed):
    """load_method-style dict: {attack: (ids, y, [pred arrays per seed])}; preds_by_seed[s][(id, attack)]"""
    return {a: (ids, np.array([y_of[i] for i in ids]), [np.array([P[(i, a)] for i in ids]) for P in preds_by_seed])
            for a, ids in ids_by_cond.items()}


def ref_paired(ids_by_cond, y_of, PA, PB, B, seed, subset=None):
    """independent reference: resample claim ids explicitly, rebuild each condition's item list by id lookup"""
    rng_ = np.random.default_rng(seed)
    clean = ids_by_cond["CLEAN"]
    atts = [a for a in ATTACK_ORDER if a != "CLEAN" and a in ids_by_cond and (subset is None or a in subset)]
    f1 = lambda yy, pp: f1_score(yy, pp, average="macro", labels=[0, 1], zero_division=0)
    bc, ba = [], []
    for _ in range(B):
        draw = rng_.integers(0, len(clean), len(clean))
        mult = {}
        for k in draw:
            mult[clean[k]] = mult.get(clean[k], 0) + 1
        v = {}
        for a in ["CLEAN"] + atts:
            items = [i for i in ids_by_cond[a] for _ in range(mult.get(i, 0))]
            sa, sb = PA[rng_.integers(0, len(PA))], PB[rng_.integers(0, len(PB))]
            yy = [y_of[i] for i in items]
            v[a] = f1(yy, [sa[(i, a)] for i in items]) - f1(yy, [sb[(i, a)] for i in items]) if items else np.nan
        bc.append(v["CLEAN"]); ba.append(np.nanmean([v[a] for a in atts]))
    return np.array(bc), np.array(ba)


for trial in range(6):
    n = int(rng.integers(20, 60)); ids = [f"c{k}" for k in range(n)]
    y_of = {i: int(rng.integers(0, 2)) for i in ids}
    conds = {"CLEAN": ids}
    for a in ["TR_fr", "SUM", "MCQ>JSON"]:  # attacked conditions keep a random subset (preservation filter)
        conds[a] = [i for i in ids if rng.random() < 0.8]
    mk = lambda acc: {(i, a): (y_of[i] if rng.random() < acc else 1 - y_of[i]) for a in conds for i in conds[a]}
    PA = [mk(0.85) for _ in range(int(rng.integers(1, 4)))]; PB = [mk(0.6) for _ in range(int(rng.integers(1, 4)))]
    RA, RB = fake_R(conds, y_of, PA), fake_R(conds, y_of, PB)
    r = paired_diff(RA, RB, B=200, seed=trial)
    f1 = lambda yy, pp: f1_score(yy, pp, average="macro", labels=[0, 1], zero_division=0)
    ptm = lambda P, a: np.mean([f1([y_of[i] for i in conds[a]], [S[(i, a)] for i in conds[a]]) for S in P])
    atts = [a for a in ATTACK_ORDER if a in conds and a != "CLEAN"]
    ok(abs(r["clean_diff"] - (ptm(PA, "CLEAN") - ptm(PB, "CLEAN"))) < 1e-12, "paired: clean point difference")
    ok(abs(r["attacked_diff"] - (np.mean([ptm(PA, a) for a in atts]) - np.mean([ptm(PB, a) for a in atts]))) < 1e-12,
       "paired: attacked point difference")
    bc, ba = ref_paired(conds, y_of, PA, PB, 200, trial)
    for k, bs in (("clean", bc), ("attacked", ba)):
        ok(abs(r[f"{k}_lo"] - np.percentile(bs, 2.5)) < 1e-12 and abs(r[f"{k}_hi"] - np.percentile(bs, 97.5)) < 1e-12,
           f"paired: {k} CI vs reference")
        ok(abs(r[f"{k}_p"] - min(1.0, 2 * min(np.mean(bs <= 0), np.mean(bs >= 0)))) < 1e-12, f"paired: {k} p vs reference")
    sub = ["SUM", "MCQ>JSON"]  # attack subset: point difference and bootstrap restricted to these attacks only
    rs = paired_diff(RA, RB, B=200, seed=trial, subset=sub)
    ok(abs(rs["attacked_diff"] - (np.mean([ptm(PA, a) for a in sub]) - np.mean([ptm(PB, a) for a in sub]))) < 1e-12,
       "paired subset: point difference")
    _, ba_s = ref_paired(conds, y_of, PA, PB, 200, trial, subset=sub)
    ok(abs(rs["attacked_lo"] - np.percentile(ba_s, 2.5)) < 1e-12 and abs(rs["attacked_hi"] - np.percentile(ba_s, 97.5)) < 1e-12
       and abs(rs["attacked_p"] - min(1.0, 2 * min(np.mean(ba_s <= 0), np.mean(ba_s >= 0)))) < 1e-12 and rs["n_attacks"] == 2,
       "paired subset: CI and p vs reference")
    same = paired_diff(RA, RA, B=100, seed=trial)
    ok(same["clean_diff"] == 0 and same["attacked_diff"] == 0 and same["clean_p"] == 1 and same["attacked_p"] == 1
       if len(PA) == 1 else same["clean_diff"] == 0, "paired: a method against itself")
try:
    bad = dict(RB); bad["SUM"] = (RB["SUM"][0][1:], RB["SUM"][1][1:], [p[1:] for p in RB["SUM"][2]])
    paired_diff(RA, bad, B=10); ok(False, "paired: mismatched items not detected")
except AssertionError:
    ok(True, "")
import tempfile, pathlib
from compare import method_files
with tempfile.TemporaryDirectory() as td:
    td = pathlib.Path(td)
    for n_ in ["m.jsonl", "m__s1.jsonl", "m__s2.jsonl", "m_x.jsonl", "m_x__s1.jsonl", "m.meta.json"]:
        (td / n_).write_text("")
    ok([f.name for f in method_files(td, "m")] == ["m.jsonl", "m__s1.jsonl", "m__s2.jsonl"], "method_files: all seeds")
    ok([f.name for f in method_files(td, "m@s0")] == ["m.jsonl"], "method_files: first seed only")
    ok([f.name for f in method_files(td, "m_x")] == ["m_x.jsonl", "m_x__s1.jsonl"] and method_files(td, "zz") == []
       and method_files(td, "zz@s0") == [], "method_files: other names")
print("paired significance test ok")

# ---------------------------------------------------------------- seen / unseen attack groups
import seen_unseen as su
from gen_train_aug import FAMILIES
fams_in_train = set()
for ds in DATASETS:
    pa = ROOT / f"data/train_aug/{ds}.jsonl"
    if pa.exists():
        fams_in_train |= {r["family"] for r in read_jsonl(pa)}
ok(set(su.SEEN) == set(FAMILIES) and (not fams_in_train or fams_in_train == set(su.SEEN)),
   "seen group = the families actually present in the attack-aware training data")
allatt = [a for a in ATTACK_ORDER if a != "CLEAN"]
ok(sorted(su.SEEN + su.UNSEEN_SINGLE + su.UNSEEN_CHAIN) == sorted(allatt) and len(allatt) == 15, "groups partition the 15 attacks")
ok(su.UNSEEN_CHAIN == ["TR>MCQ", "MCQ>JSON", "MCQ>XML", "TR>XML", "SUM>JSON", "TR>MCQ>JSON"], "chain group")
ok(not set(su.SEEN) & set(su.UNSEEN_SINGLE + su.UNSEEN_CHAIN), "seen and unseen disjoint")
gm, cl = su.group_means(RA)  # group means on the synthetic method above (attacks present: TR_fr, SUM, MCQ>JSON)
ok(abs(gm["seen"] - np.mean([ptm(PA, "TR_fr"), ptm(PA, "SUM")])) < 1e-12 and abs(gm["unseen_chain"] - ptm(PA, "MCQ>JSON")) < 1e-12
   and abs(cl - ptm(PA, "CLEAN")) < 1e-12, "group means")
print("seen/unseen groups ok")

# ---------------------------------------------------------------- contamination check and Claude requests
import contamination as ct
words = "the a vaccine covid 19 climate warming bank stock 2020 rise fall is not cause".split()
try:
    from rouge_score import rouge_scorer
    sc = rouge_scorer.RougeScorer(["rougeL"])
    for _ in range(500):
        ref = " ".join(rng.choice(words, int(rng.integers(0, 12)))); hyp = " ".join(rng.choice(words, int(rng.integers(0, 12))))
        hyp = hyp + rng.choice(["", ",", ".", "!"]) ; ref = ref.upper() if rng.random() < 0.2 else ref
        ok(abs(ct.rouge_l(ref, hyp) - sc.score(ref, hyp)["rougeL"].fmeasure) < 1e-12, "ROUGE-L F1 vs rouge_score")
    print("ROUGE-L vs rouge_score ok")
except ImportError:
    print("ROUGE-L: rouge_score not installed here, reference comparison skipped")
ok(ct.rouge_l("a b c d", "a b c d") == 1.0 and ct.rouge_l("a b", "c d") == 0.0
   and abs(ct.rouge_l("a b c d", "a c") - 2 * (1.0 * 0.5) / 1.5) < 1e-12, "ROUGE-L hand cases")
ok(ct.clean_completion('Second piece: "cures cancer."\nExplanation...', "garlic") == "cures cancer."
   and ct.clean_completion("garlic cures cancer", "garlic") == "cures cancer"
   and ct.clean_completion("", "x") == "" and ct.clean_completion("**Second piece:** rises", "x") == "rises", "completion cleaning")
# chat-style answers (Claude): labels, label on its own line, full claim, hedge with a quoted guess, pure refusal
ok(ct.clean_completion("**First piece:** Sea ice\n\n**Second piece:** is melting fast.\n\n**Combined claim:** x", "Sea ice")
   == "is melting fast."
   and ct.clean_completion("**Completed claim:**\n\n\"Sea ice is melting fast.\"\n\nNote: guess", "Sea ice") == "is melting fast."
   and ct.clean_completion("I can't recall it exactly. My best guess:\n\n**Second piece (best guess):** \"Own Vanguard\"",
                           "No, Pfizer Does Not") == "Own Vanguard"
   and ct.clean_completion("I can't recall it exactly; a plausible completion is \"Global warming causes more storms.\"",
                           "Global warming") == "causes more storms."
   and ct.clean_completion("I can't reproduce this claim from the dataset.", "x") == ""
   and ct.clean_completion("I believe the second piece is \"to Mexico.\"", "Ford is moving") == "to Mexico.",
   "chat-style completion extraction")
ok(ct.exact("Cures, cancer!", "cures cancer") and not ct.exact("cures cancer", "cures"), "exact match after normalisation")
ok(ct.boot_greater([0.2] * 30) == 0.0 and ct.boot_greater([-0.1] * 30) == 1.0, "bootstrap one-sided p")
for ds in DATASETS:
    if not (ROOT / f"data/attacks/{ds}.jsonl").exists():
        continue
    # the shared subsample = the draw of the original api_zs.subsample (reimplemented here verbatim)
    rows = [r for r in common.attacks(ds) if not r["attack"].startswith("ADAPT_")]
    cl = [r for r in rows if r["attack"] == "CLEAN"]; rr = random.Random(0)
    by = {0: [r["id"] for r in cl if r["label"] == 0], 1: [r["id"] for r in cl if r["label"] == 1]}
    fr = len(by[1]) / len(cl)
    ref_keep = set(rr.sample(by[1], round(100 * fr))) | set(rr.sample(by[0], 100 - round(100 * fr)))
    S = ct.subsample_ids(ds)
    ok(S == ref_keep and len(S) == 100, f"{ds}: shared subsample")
    lab = {r["id"]: r["label"] for r in cl}
    ok(abs(sum(lab[i] for i in S) / 100 - fr) <= 0.01, f"{ds}: subsample stratified by label")
its = ct.items() if all((ROOT / f"data/attacks/{d}.jsonl").exists() for d in DATASETS) else []
if its:
    text_of = {(r["id"]): r["text"] for d in DATASETS for r in common.attacks(d) if r["attack"] == "CLEAN"}
    pairs = {}
    for it in its:
        pairs.setdefault((it["dataset"], it["id"]), []).append(it)
        ok(it["first"] + " " + it["second"] == " ".join(text_of[it["id"]].split()), "pieces rebuild the claim")
        ok(it["first"] and it["second"] and it["id"] in ct.subsample_ids(it["dataset"]), "non-empty pieces of subsample claims")
        ok((ct.NAMES[it["dataset"]] in it["prompt"]) == (it["kind"] == "guided") and it["prompt"].endswith("Second piece:")
           and ("First piece: " + it["first"]) in it["prompt"], "guided names the dataset, general does not")
    ok(all(sorted(x["kind"] for x in v) == ["general", "guided"] and v[0]["first"] == v[1]["first"] for v in pairs.values()),
       "each claim has one guided and one general prompt with the same cut")
    import api_zs, detect
    ok(api_zs.PROMPT == detect.ZS_PROMPT, "Claude zero-shot prompt = open-model zero-shot prompt")
    B_ = api_zs.build()
    zs = [m for _, _, m in B_ if m["task"] == "zs"]
    ok(len(zs) == 3 * 100 * 16 and len(B_) - len(zs) == len(its), "batch size: 3 datasets x 100 claims x 16 conditions + contamination")
    ok(all(sum(1 for m in zs if m["dataset"] == d and m["id"] == i) == 16 for d in DATASETS for i in ct.subsample_ids(d)),
       "every subsample claim has all 16 conditions")
    q = {(r["id"], r["attack"]): r["query"] for d in DATASETS for r in common.attacks(d)}
    ok(all(p == api_zs.PROMPT.format(q=q[(m["id"], m["attack"])]) for _, p, m in B_ if m["task"] == "zs"), "zs prompts = query")
    ok(api_zs.parse_yes_no("Yes.") == 1 and api_zs.parse_yes_no("**No**") == 0 and api_zs.parse_yes_no("Maybe") is None
       and api_zs.parse_yes_no("") is None and api_zs.parse_yes_no("Yesterday") is None, "yes/no parsing")
    print(f"contamination + Claude batch ok ({len(its)} contamination prompts, {len(B_)} requests)")
    for it in its:
        ok((ct.NAMES[it["dataset"]] in it["prompt_prefill"]) == (it["kind"] == "guided")
           and it["prompt_prefill"].endswith("\n\n" + it["first"]), "prefill prompts")
    try:
        from transformers import AutoTokenizer
        tk = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
        for it in its[:50]:
            s = tk.apply_chat_template(ct.prefill_messages(it), tokenize=False, add_generation_prompt=False,
                                       continue_final_message=True)
            tail = s.rsplit("<|eot_id|>", 1)[1]  # everything after the last closed turn = the open assistant turn
            ok(s.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n" + it["first"]) and it["prompt_prefill"] in s
               and tail == "<|start_header_id|>assistant<|end_header_id|>\n\n" + it["first"],
               "prefill: assistant turn left open after the first piece")
        print("prefill chat template ok")
    except OSError:
        print("prefill chat template: tokenizer not available here, skipped")
ok(ct.is_refusal("I cannot provide the second piece") and ct.is_refusal("  I can't help") and ct.is_refusal("Sorry, no")
   and not ct.is_refusal("cures cancer in 2020") and not ct.is_refusal("") and not ct.is_refusal("Icelandic banks"), "refusal detector")

print(f"\nALL {passed} CHECKS PASSED")
