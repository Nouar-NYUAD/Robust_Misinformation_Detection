"""Paired significance tests between two methods on the same test items (paired bootstrap; Koehn 2004,
Berg-Kirkpatrick et al. 2012, Dror et al. 2018).

Both methods are scored on exactly the same items (same preservation filter as aggregate.py), and each bootstrap
resample draws the same clean claim ids for both methods (an attacked text is kept with the multiplicity of its claim,
as in aggregate.py). Methods with several seeds draw one seed per resample. Reported: the difference A - B in clean
macro-F1 and in attacked macro-F1 (mean over the attacks both methods have), with a 95% CI and a two-sided p-value
p = min(1, 2 * min(P*(d <= 0), P*(d >= 0))) from the bootstrap distribution of the difference.

usage: python compare.py ours_vtrained clf_xlmr-advtrain [more_baselines ...]  [--all] [--out=NAME]
out:   results/tables/NAME[_all].csv  (NAME defaults to compare)
"""
import csv, sys
import numpy as np

from common import ROOT, ATTACK_ORDER
from aggregate import macro_f1, load_valid, load_method
import aggregate


def method_files(ds_dir, spec):
    """prediction files of a method: 'name' = all seeds (name.jsonl + name__s*.jsonl); 'name@s0' = name.jsonl only
    (the first seed, e.g. to compare single-seed ablations with the matching seed of the full model)"""
    if spec.endswith("@s0"):
        f = ds_dir / f"{spec[:-3]}.jsonl"
        return [f] if f.exists() else []
    return sorted(f for f in ds_dir.glob("*.jsonl") if f.stem.split("__s")[0] == spec)


def paired_diff(RA, RB, B=1000, seed=0, subset=None):
    """RA, RB: load_method() outputs for one dataset. Returns dict with point differences, CIs and p-values.
    subset: optional list of attack names; the attacked score is then the mean over these attacks only."""
    rng = np.random.default_rng(seed)
    atts = [a for a in ATTACK_ORDER if a != "CLEAN" and a in RA and a in RB and RA[a][2] and RB[a][2]
            and (subset is None or a in subset)]
    assert atts, "no attack of the subset is available for both methods"
    conds = ["CLEAN"] + atts
    for a in conds:
        assert RA[a][0] == RB[a][0], f"{a}: the two methods are scored on different items"
        assert (RA[a][1] == RB[a][1]).all(), f"{a}: gold labels differ"
    ids = RA["CLEAN"][0]; n = len(ids)
    pos = {i: k for k, i in enumerate(ids)}
    align = {a: np.array([pos[i] for i in RA[a][0]]) for a in conds}
    pt = lambda R, a: float(np.mean([macro_f1(R[a][1], p) for p in R[a][2]]))
    d_clean = pt(RA, "CLEAN") - pt(RB, "CLEAN")
    d_att = float(np.mean([pt(RA, a) for a in atts]) - np.mean([pt(RB, a) for a in atts]))
    bc, ba = [], []
    for _ in range(B):
        counts = np.bincount(rng.integers(0, n, n), minlength=n)
        v = {}
        for a in conds:
            keep = np.repeat(np.arange(len(align[a])), counts[align[a]])
            pa = RA[a][2][rng.integers(0, len(RA[a][2]))]
            pb = RB[a][2][rng.integers(0, len(RB[a][2]))]
            y = RA[a][1][keep]
            v[a] = macro_f1(y, pa[keep]) - macro_f1(y, pb[keep]) if len(keep) else np.nan  # as in aggregate.py
        bc.append(v["CLEAN"]); ba.append(np.nanmean([v[a] for a in atts]))
    out = {}
    for k, d, bs in (("clean", d_clean, np.array(bc)), ("attacked", d_att, np.array(ba))):
        out[f"{k}_diff"] = d
        out[f"{k}_lo"], out[f"{k}_hi"] = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
        out[f"{k}_p"] = float(min(1.0, 2 * min(np.mean(bs <= 0), np.mean(bs >= 0))))
    out["n_attacks"] = len(atts)
    return out


def out_name(default="compare"):
    """--out=NAME on the command line, else the default"""
    names = [a[len("--out="):] for a in sys.argv[1:] if a.startswith("--out=")]
    assert len(names) <= 1 and all(names), "give --out=NAME at most once"
    return names[0] if names else default


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    a_name, others = args[0], args[1:]
    rows = []
    for ds_dir in sorted(p for p in (ROOT / "results/preds").glob("*") if p.is_dir()):
        ds = ds_dir.name
        valid = load_valid(ds)
        files = lambda m: method_files(ds_dir, m)
        if not files(a_name):
            continue
        RA = load_method(files(a_name), valid)
        for b_name in others:
            if not files(b_name):
                continue
            r = paired_diff(RA, load_method(files(b_name), valid))
            rows.append(dict(dataset=ds, a=a_name, b=b_name, **r))
            print(f"{ds:13s} {a_name} - {b_name:22s} clean {r['clean_diff']:+.3f} [{r['clean_lo']:+.3f},{r['clean_hi']:+.3f}] "
                  f"p={r['clean_p']:.3f} | attacked {r['attacked_diff']:+.3f} [{r['attacked_lo']:+.3f},"
                  f"{r['attacked_hi']:+.3f}] p={r['attacked_p']:.3f}")
    if rows:
        with open(ROOT / f"results/tables/{out_name()}{aggregate.SUF}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
