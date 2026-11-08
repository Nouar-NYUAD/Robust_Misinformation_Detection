"""Independent re-computation of every macro-F1 in results/tables/summary.csv (and summary_all.csv with --all)
directly from the raw prediction files, WITHOUT using aggregate.py / common.py: plain json + sklearn.

Semantics (as documented in the paper): an attacked text is scored only if the judge found it to preserve the
claim (p_preserved >= 0.5); clean claims are always scored; seeds (name.jsonl, name__s*.jsonl) are averaged;
'attacked' is the mean over the 15 attacks. Prints every mismatch > 1e-9 and exits non-zero if any.
"""
import csv, glob, json, os, sys
from collections import defaultdict
import numpy as np
from sklearn.metrics import f1_score

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ALL = "--all" in sys.argv
ATTACKS = ["TR_fr", "TR_es", "TR_ar", "TR_hi", "TR_zh", "SUM", "MCQ", "XML", "JSON",
           "TR>MCQ", "MCQ>JSON", "MCQ>XML", "TR>XML", "SUM>JSON", "TR>MCQ>JSON"]


def jl(p):
    with open(p, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def main():
    summ = list(csv.DictReader(open(os.path.join(ROOT, "results/tables", "summary_all.csv" if ALL else "summary.csv"))))
    bad, checked = [], 0
    for ds in sorted({r["dataset"] for r in summ}):
        keep = {(r["id"], r["attack"]): r["p_preserved"] >= 0.5 for r in jl(os.path.join(ROOT, f"data/attacks/{ds}.valid.jsonl"))}
        for row in (r for r in summ if r["dataset"] == ds):
            m = row["method"]
            files = sorted(f for f in glob.glob(os.path.join(ROOT, f"results/preds/{ds}/*.jsonl"))
                           if os.path.basename(f)[:-6].split("__s")[0] == m)
            per = defaultdict(list)   # attack -> [f1 per seed]
            for f in files:
                by = defaultdict(lambda: ([], []))
                for r in jl(f):
                    a = r["attack"]
                    if a != "CLEAN" and a not in ATTACKS:
                        continue
                    if a != "CLEAN" and not ALL and not keep.get((r["id"], a), True):
                        continue
                    by[a][0].append(r["label"]); by[a][1].append(r["pred"])
                for a, (y, p) in by.items():
                    per[a].append(f1_score(y, p, labels=[0, 1], average="macro", zero_division=0))
            assert len(files) == int(row["seeds"]), (ds, m, len(files), row["seeds"])
            clean = float(np.mean(per["CLEAN"]))
            att = float(np.mean([np.mean(per[a]) for a in ATTACKS]))
            checks = [("clean_f1", clean), ("attacked_f1", att)] + [(f"f1_{a}", float(np.mean(per[a]))) for a in ATTACKS]
            for k, v in checks:
                checked += 1
                if abs(v - float(row[k])) > 1e-9:
                    bad.append((ds, m, k, v, float(row[k])))
    print(f"checked {checked} numbers ({'all texts' if ALL else 'preservation-filtered'}); mismatches: {len(bad)}")
    for b in bad[:20]:
        print("  MISMATCH", b)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
