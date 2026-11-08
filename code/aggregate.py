"""Aggregate predictions into clean-vs-attacked tables with paired bootstrap 95% CIs.

Attack-preservation filter (default ON): an attacked text is scored only if the independent judge
(validate_attacks.py) found that it still asserts / presupposes the original claim (p_preserved >= 0.5). An
attacker LLM sometimes debunks a false claim while "discussing" it; such a text no longer spreads the claim, so
its gold label would be wrong. CLEAN rows are always kept. `python aggregate.py --all` scores every text instead
(reported in the appendix); outputs then carry the suffix _all.

Methods trained with several seeds (files <name>__s<seed>.jsonl next to <name>.jsonl) are averaged over seeds;
the bootstrap also draws a seed per resample.

out: results/tables/summary[_all].csv   (dataset, method): clean / attacked-mean macro-F1 with CIs, drop, worst, ASR
     results/tables/long[_all].csv      (dataset, method, attack): all metrics
     results/tables/adaptive[_all].csv  (dataset, method, adaptive attack)
     results/tables/preservation.csv    (dataset, attack): share preserved, by label
"""
import csv, sys
from collections import defaultdict
import numpy as np

from common import ROOT, ATTACK_ORDER, read_jsonl, metrics

ALL = "--all" in sys.argv
SUF = "_all" if ALL else ""
B = 1000
base = ROOT / "results/preds"
out = ROOT / "results/tables"; out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)


def macro_f1(y, p):
    tp = np.sum((y == 1) & (p == 1)); fp = np.sum((y == 0) & (p == 1))
    fn = np.sum((y == 1) & (p == 0)); tn = np.sum((y == 0) & (p == 0))
    return (2 * tp / max(1, 2 * tp + fp + fn) + 2 * tn / max(1, 2 * tn + fn + fp)) / 2


def load_valid(ds):
    p = ROOT / f"data/attacks/{ds}.valid.jsonl"
    if not p.exists():
        return None
    return {(r["id"], r["attack"]): r["p_preserved"] >= 0.5 for r in read_jsonl(p)}


def load_method(files, valid):
    """{attack: (ids, y, [preds per seed])}; rows dropped by the preservation filter are excluded."""
    per = defaultdict(lambda: {"y": {}, "p": []})
    for f in files:
        by = defaultdict(dict)
        for r in read_jsonl(f):
            a = r["attack"]
            if a not in ("CLEAN", "DEV_CLEAN") and valid is not None and not ALL and not valid.get((r["id"], a), True):
                continue
            by[a][r["id"]] = r
            per[a]["y"][r["id"]] = r["label"]
        for att, d in by.items():
            per[att]["p"].append(d)
    res = {}
    for att, v in per.items():
        ids = sorted(v["y"])
        res[att] = (ids, np.array([v["y"][i] for i in ids]),
                    [np.array([d[i]["pred"] for i in ids]) for d in v["p"] if all(i in d for i in ids)])
    return res


def main():
    long, summ, adapt, pres = [], [], [], []
    for ds_dir in sorted(p for p in base.glob("*") if p.is_dir()):
        ds = ds_dir.name
        valid = load_valid(ds)
        if valid is not None:
            lab = {(r["id"]): r["label"] for r in read_jsonl(ROOT / f"data/attacks/{ds}.jsonl") if r["attack"] == "CLEAN"}
            agg = defaultdict(list)
            for (i, a), v in valid.items():
                agg[(a, lab[i])].append(v)
            for a in sorted({a for a, _ in agg}):
                f_, t_ = agg.get((a, 1), []), agg.get((a, 0), [])
                pres.append(dict(dataset=ds, attack=a, preserved=np.mean(f_ + t_), preserved_false=np.mean(f_) if f_ else np.nan,
                                 preserved_true=np.mean(t_) if t_ else np.nan, n=len(f_ + t_)))
        groups = defaultdict(list)
        for f in sorted(ds_dir.glob("*.jsonl")):
            groups[f.stem.split("__s")[0]].append(f)
        for method, files in groups.items():
            R = load_method(files, valid)
            if "CLEAN" not in R or not R["CLEAN"][2]:
                continue
            atts = [a for a in ATTACK_ORDER if a in R and a != "CLEAN" and R[a][2]]
            conds = ["CLEAN"] + atts
            pt = {a: float(np.mean([macro_f1(R[a][1], p) for p in R[a][2]])) for a in conds}
            for a in conds:
                ms = [metrics(R[a][1], p) for p in R[a][2]]
                long.append(dict(dataset=ds, method=method, attack=a, seeds=len(ms),
                                 **{k: float(np.mean([m[k] for m in ms])) for k in ms[0]}))
            # paired bootstrap over CLEAN claim ids; each condition keeps the resampled ids it contains
            ids = R["CLEAN"][0]; n = len(ids)
            pos = {i: k for k, i in enumerate(ids)}
            align = {a: np.array([pos[i] for i in R[a][0]]) for a in conds}       # condition row -> clean position
            bs_clean, bs_att, bs_drop = [], [], []
            for _ in range(B):
                counts = np.bincount(rng.integers(0, n, n), minlength=n)        # multiplicity of each clean id
                vals = {}
                for a in conds:
                    w = counts[align[a]]
                    keep = np.repeat(np.arange(len(w)), w)
                    pa = R[a][2][rng.integers(0, len(R[a][2]))]
                    vals[a] = macro_f1(R[a][1][keep], pa[keep]) if len(keep) else np.nan
                bs_clean.append(vals["CLEAN"])
                if atts:
                    m = np.nanmean([vals[a] for a in atts]); bs_att.append(m); bs_drop.append(vals["CLEAN"] - m)
            ci = lambda x: (float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))) if x else (np.nan, np.nan)
            att_mean = float(np.mean([pt[a] for a in atts])) if atts else np.nan
            # ASR relative to clean: false claims detected when clean but missed under attack (first seed)
            cl_ids, y0, p0 = R["CLEAN"][0], R["CLEAN"][1], R["CLEAN"][2][0]
            caught = {i for i, y, p in zip(cl_ids, y0, p0) if y == 1 and p == 1}
            asr = []
            for a in atts:
                m_ = [p == 0 for i, p in zip(R[a][0], R[a][2][0]) if i in caught]
                if m_:
                    asr.append(np.mean(m_))
            summ.append(dict(dataset=ds, method=method, seeds=len(R["CLEAN"][2]),
                             clean_f1=pt["CLEAN"], clean_lo=ci(bs_clean)[0], clean_hi=ci(bs_clean)[1],
                             attacked_f1=att_mean, attacked_lo=ci(bs_att)[0], attacked_hi=ci(bs_att)[1],
                             drop_f1=pt["CLEAN"] - att_mean, drop_lo=ci(bs_drop)[0], drop_hi=ci(bs_drop)[1],
                             worst_f1=min(pt[a] for a in atts) if atts else np.nan,
                             asr_rel=float(np.mean(asr)) if asr else np.nan, **{f"f1_{a}": pt[a] for a in atts}))
            for a in sorted(x for x in R if x.startswith("ADAPT_") and R[x][2]):
                ms = [metrics(R[a][1], p) for p in R[a][2]]
                adapt.append(dict(dataset=ds, method=method, attack=a,
                                  macro_f1=float(np.mean([m["macro_f1"] for m in ms])),
                                  false_acc=float(np.mean([m["false_acc"] for m in ms]))))

    for name, rows in ((f"long{SUF}", long), (f"summary{SUF}", summ), (f"adaptive{SUF}", adapt), ("preservation", pres)):
        if rows:
            keys = list(dict.fromkeys(k for r in rows for k in r))
            with open(out / f"{name}.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, keys); w.writeheader(); w.writerows(rows)
    print("preservation filter:", "OFF (--all)" if ALL else "ON")
    for r in summ:
        print(f"{r['dataset']:13s} {r['method']:24s} clean {r['clean_f1']:.3f} | attacked {r['attacked_f1']:.3f} "
              f"[{r['attacked_lo']:.3f},{r['attacked_hi']:.3f}] drop {r['drop_f1']:+.3f} worst {r['worst_f1']:.3f} "
              f"ASR {r['asr_rel']:.3f}")


if __name__ == "__main__":
    main()
