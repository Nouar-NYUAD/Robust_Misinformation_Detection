"""Claude Opus 5.5 zero-shot (api_zs.py) vs every other detector on the SAME stratified subsample of test claims.

Claude was run on 100 test claims per dataset (contamination.subsample_ids) under CLEAN and all 15 attacks. Every
other method is restricted to exactly these claims and scored with the same preservation filter (aggregate.py), so
all numbers in the output are directly comparable. An answer that is neither Yes nor No (pred None: a hedge or a
refusal) counts as "not flagged" (pred 0); the number of such answers is reported.

usage: python subsample_eval.py [--all]
out:   results/tables/claude_subsample[_all].csv   (method, seeds, clean, attacked, drop per dataset)
       results/tables/compare_claude[_all].csv      (paired bootstrap: ours_vtrained and others vs zs_claude)
"""
import csv, json
import numpy as np

from common import ROOT, DATASETS, ATTACK_ORDER
from aggregate import macro_f1, load_valid, load_method
import aggregate
from compare import paired_diff, method_files

CLAUDE = "zs_claude"
METHODS = ["kotonya", "arg", "programfc", "minicheck", "rag", "zs", "zs_qwen", "clf_xlmr", "ours_vtrained",
           "kotonya-advtrain", "arg-advtrain", "clf_xlmr-advtrain", "ours_air_vtrained_aug"]
COMPARE = ["ours_vtrained", "ours_air_vtrained_aug", "clf_xlmr-advtrain", "zs", "rag"]


def claude_file(ds, none_as=0):
    """zs_claude predictions with pred None mapped to none_as, written to a temporary file for load_method"""
    src = ROOT / f"results/preds_api/{ds}/{CLAUDE}.jsonl"
    rows = [json.loads(l) for l in open(src, encoding="utf-8")]
    n_none = sum(r["pred"] is None for r in rows)
    for r in rows:
        r["pred"] = none_as if r["pred"] is None else r["pred"]
    tmp = ROOT / f"results/preds_api/{ds}/{CLAUDE}.resolved.jsonl"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)
    return tmp, n_none, len(rows)


def restrict(R, keep):
    """keep only the claims in `keep` (ids), for every condition"""
    out = {}
    for a, (ids, y, preds) in R.items():
        m = np.array([i in keep for i in ids])
        out[a] = ([i for i in ids if i in keep], y[m], [p[m] for p in preds])
    return out


def scores(R):
    atts = [a for a in ATTACK_ORDER if a != "CLEAN" and a in R and R[a][2]]
    pt = lambda a: float(np.mean([macro_f1(R[a][1], p) for p in R[a][2]]))
    clean, att = pt("CLEAN"), float(np.mean([pt(a) for a in atts]))
    return dict(clean=clean, attacked=att, drop=clean - att, n_attacks=len(atts), seeds=len(R["CLEAN"][2]))


def main():
    rows, comp = [], []
    for ds in DATASETS:
        valid = load_valid(ds)
        f, n_none, n_rows = claude_file(ds)
        RC = load_method([f], valid)
        keep = set(RC["CLEAN"][0])
        assert len(keep) == 100 and all(set(RC[a][0]) <= keep for a in RC), ds
        s = scores(RC)
        assert s["n_attacks"] == 15
        rows.append(dict(dataset=ds, method=CLAUDE, n_claims=len(keep), unparsed=n_none, **s))
        ds_dir = ROOT / "results/preds" / ds
        for m in METHODS:
            Rm = restrict(load_method(method_files(ds_dir, m), valid), keep)
            for a in RC:  # the same items as Claude, condition by condition
                assert Rm[a][0] == RC[a][0] and (Rm[a][1] == RC[a][1]).all(), (ds, m, a)
            rows.append(dict(dataset=ds, method=m, n_claims=len(keep), unparsed=0, **scores(Rm)))
            if m in COMPARE:
                r = paired_diff(Rm, RC)
                comp.append(dict(dataset=ds, a=m, b=CLAUDE, **r))
        print(ds, f"unparsed {n_none}/{n_rows}")
        for r in rows:
            if r["dataset"] == ds:
                print(f"   {r['method']:22s} clean {r['clean']:.3f} attacked {r['attacked']:.3f} drop {r['drop']:+.3f}")
    for name, rr in ((f"claude_subsample{aggregate.SUF}", rows), (f"compare_claude{aggregate.SUF}", comp)):
        with open(ROOT / f"results/tables/{name}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, list(rr[0])); w.writeheader(); w.writerows(rr)
    for r in comp:
        print(f"{r['dataset']:13s} {r['a']:22s} - claude  clean {r['clean_diff']:+.3f} p={r['clean_p']:.3f} | "
              f"attacked {r['attacked_diff']:+.3f} [{r['attacked_lo']:+.3f},{r['attacked_hi']:+.3f}] p={r['attacked_p']:.3f}")


if __name__ == "__main__":
    main()
