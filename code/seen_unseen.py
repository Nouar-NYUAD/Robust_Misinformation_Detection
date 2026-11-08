"""Seen vs unseen attacks. Attack-aware training (Setting B) uses transformed training claims of four families only
(gen_train_aug.FAMILIES: TR_fr, TR_es, SUM, MCQ). Test attacks are grouped by whether such a family was seen:
  seen            TR_fr TR_es SUM MCQ
  unseen_single   TR_ar TR_hi TR_zh XML JSON          (new languages, new wrappers)
  unseen_chain    TR>MCQ MCQ>JSON MCQ>XML TR>XML SUM>JSON TR>MCQ>JSON   (compositions; never seen as such)
For every method: mean macro-F1 per group (same preservation filter, seeds averaged, as aggregate.py), and paired
bootstrap differences of method A against each baseline per group (compare.paired_diff with subset).

usage: python seen_unseen.py ours_vtrained clf_xlmr-advtrain arg-advtrain kotonya-advtrain [--all] [--out=NAME]
out:   results/tables/NAME[_all].csv, NAME_compare[_all].csv  (NAME defaults to seen_unseen)
"""
import csv, sys
import numpy as np

from common import ROOT, ATTACK_ORDER
from aggregate import macro_f1, load_valid, load_method
import aggregate
from compare import paired_diff, method_files, out_name
from gen_train_aug import FAMILIES

SEEN = list(FAMILIES)
UNSEEN_SINGLE = ["TR_ar", "TR_hi", "TR_zh", "XML", "JSON"]
UNSEEN_CHAIN = [a for a in ATTACK_ORDER if ">" in a]
GROUPS = {"seen": SEEN, "unseen_single": UNSEEN_SINGLE, "unseen_chain": UNSEEN_CHAIN,
          "unseen_all": UNSEEN_SINGLE + UNSEEN_CHAIN}
# the groups partition the 15 attacks exactly
assert sorted(SEEN + UNSEEN_SINGLE + UNSEEN_CHAIN) == sorted(a for a in ATTACK_ORDER if a != "CLEAN")


def group_means(R):
    f1 = {a: float(np.mean([macro_f1(R[a][1], p) for p in R[a][2]])) for a in R if R[a][2]}
    return {g: float(np.mean([f1[a] for a in atts if a in f1])) for g, atts in GROUPS.items()}, f1["CLEAN"]


def main():
    names = [a for a in sys.argv[1:] if not a.startswith("--")]
    a_name, others = names[0], names[1:]
    rows, comp = [], []
    for ds_dir in sorted(p for p in (ROOT / "results/preds").glob("*") if p.is_dir()):
        ds = ds_dir.name
        valid = load_valid(ds)
        files = lambda m: method_files(ds_dir, m)
        R = {m: load_method(files(m), valid) for m in names if files(m)}
        if a_name not in R:
            continue
        for m, Rm in R.items():
            g, clean = group_means(Rm)
            rows.append(dict(dataset=ds, method=m, seeds=len(Rm["CLEAN"][2]), clean=clean, **g))
            print(f"{ds:13s} {m:20s} clean {clean:.3f} | " + " ".join(f"{k} {v:.3f}" for k, v in g.items()))
        for b in others:
            if b not in R:
                continue
            for gname, atts in GROUPS.items():
                r = paired_diff(R[a_name], R[b], subset=atts)
                comp.append(dict(dataset=ds, a=a_name, b=b, group=gname, diff=r["attacked_diff"], lo=r["attacked_lo"],
                                 hi=r["attacked_hi"], p=r["attacked_p"], n_attacks=r["n_attacks"]))
                print(f"   {a_name} - {b:20s} {gname:14s} {r['attacked_diff']:+.3f} "
                      f"[{r['attacked_lo']:+.3f},{r['attacked_hi']:+.3f}] p={r['attacked_p']:.3f}")
    base = out_name("seen_unseen")
    for name, rr in ((f"{base}{aggregate.SUF}", rows), (f"{base}_compare{aggregate.SUF}", comp)):
        if rr:
            with open(ROOT / f"results/tables/{name}.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, list(rr[0])); w.writeheader(); w.writerows(rr)


if __name__ == "__main__":
    main()
