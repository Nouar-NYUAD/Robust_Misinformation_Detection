"""Sample for the human validation of the claim-preservation judge (Gemma-3-12B, validate_attacks.py).

English-language attacks only (the annotators read English): SUM, MCQ, XML, JSON, MCQ>JSON, MCQ>XML, SUM>JSON.
Per dataset x attack cell: PER_CELL items, of which up to NEG_MAX are judge-negative (the judge said the text does
NOT preserve the claim) so that both kinds of judge error can be measured; the rest judge-positive; within each
group, false and true claims alternate. Items are shuffled with a fixed seed; dataset, attack, label and the judge's
decision are kept out of the page (key file only).

out: annotation/items.json (what the page shows: id, claim, text)
     annotation/key.json   (id -> dataset, claim id, attack, label, judge p_preserved, judge decision)
"""
import json, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["pubhealth", "climatefever", "finfact"]
ENGLISH = ["SUM", "MCQ", "XML", "JSON", "MCQ>JSON", "MCQ>XML", "SUM>JSON"]
PER_CELL, NEG_MAX, SEED = 10, 3, 2026


def read(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def pick(pool, n, rng):
    """n items alternating false/true claims as far as the pool allows"""
    f = [r for r in pool if r["label"] == 1]; t = [r for r in pool if r["label"] == 0]
    rng.shuffle(f); rng.shuffle(t)
    out = []
    while len(out) < n and (f or t):
        for src in (f, t):
            if src and len(out) < n:
                out.append(src.pop())
    return out


def main():
    rng = random.Random(SEED)
    items, key = [], {}
    for ds in DATASETS:
        rows = [r for r in read(ROOT / f"data/attacks/{ds}.jsonl") if r["attack"] in ENGLISH]
        claim = {r["id"]: r["text"] for r in read(ROOT / f"data/attacks/{ds}.jsonl") if r["attack"] == "CLEAN"}
        p = {(r["id"], r["attack"]): r["p_preserved"] for r in read(ROOT / f"data/attacks/{ds}.valid.jsonl")}
        for att in ENGLISH:
            cell = [r for r in rows if r["attack"] == att]
            neg = [r for r in cell if p[r["id"], att] < 0.5]
            pos = [r for r in cell if p[r["id"], att] >= 0.5]
            chosen = pick(neg, min(NEG_MAX, len(neg)), rng)
            chosen += pick(pos, PER_CELL - len(chosen), rng)
            assert len(chosen) == PER_CELL, (ds, att, len(neg), len(pos))
            for r in chosen:
                items.append(dict(claim=claim[r["id"]], text=r["text"].strip(),
                                  _key=dict(dataset=ds, claim_id=r["id"], attack=att, label=r["label"],
                                            p_preserved=p[r["id"], att], judge=int(p[r["id"], att] >= 0.5))))
    rng.shuffle(items)
    out_items = []
    for k, it in enumerate(items, 1):
        iid = f"i{k:03d}"
        key[iid] = it.pop("_key")
        out_items.append(dict(id=iid, **it))
    assert len({(v["dataset"], v["claim_id"], v["attack"]) for v in key.values()}) == len(key)  # no duplicates
    d = ROOT / "annotation"; d.mkdir(exist_ok=True)
    json.dump(out_items, open(d / "items.json", "w", encoding="utf-8"), ensure_ascii=False)
    json.dump(key, open(d / "key.json", "w", encoding="utf-8"), indent=1)
    neg = sum(v["judge"] == 0 for v in key.values())
    print(f"{len(out_items)} items, judge-negative {neg}, false claims {sum(v['label'] == 1 for v in key.values())}")
    words = sorted(len(it["text"].split()) for it in out_items)
    print("text words: median", words[len(words) // 2], "max", words[-1])


if __name__ == "__main__":
    main()
