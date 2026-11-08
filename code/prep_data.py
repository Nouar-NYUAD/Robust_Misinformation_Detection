"""Build binary (true=0 / false=1) claim sets and evidence knowledge bases for the three benchmarks.

Outputs, per dataset, in data/processed/<name>/:
  claims.jsonl  {id, claim, label, label_str, split}
  kb.jsonl      {doc_id, claim_id, type, claim, label, label_str, text}
                verified-claim archive built from the TRAIN split only: one "claim" record per fact-checked
                train claim (true AND false) plus its evidence/justification passages, each carrying the
                parent claim's verdict. Dev (calibration) and test claims are never stored in the archive.

Splits
  pubhealth     official train/dev/test (Kotonya & Toni, 2020)
  climatefever  no official split -> stratified 60/10/30, seed 13
  finfact       shared-task test labels are withheld -> stratified 70/10/20, seed 13
"""
import csv, io, json, random, re, sys, zipfile, ast
from pathlib import Path
from collections import Counter, defaultdict

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed"
SEED = 13
csv.field_size_limit(10**9)


def norm(s):
    return re.sub(r"\s+", " ", (s or "")).strip()


def chunk(text, max_words=120, stride=100):
    w = norm(text).split()
    if not w:
        return []
    out = []
    for i in range(0, max(1, len(w) - (max_words - stride)), stride):
        out.append(" ".join(w[i:i + max_words]))
        if i + max_words >= len(w):
            break
    return out


def stratified(ids_labels, fracs, seed=SEED):
    rng = random.Random(seed)
    by = defaultdict(list)
    for i, y in ids_labels:
        by[y].append(i)
    split = {}
    names = ["train", "dev", "test"]
    for y, ids in sorted(by.items()):
        rng.shuffle(ids)
        n = len(ids)
        a, b = round(fracs[0] * n), round((fracs[0] + fracs[1]) * n)
        for k, i in enumerate(ids):
            split[i] = names[0] if k < a else names[1] if k < b else names[2]
    return split


def dedup(rows):
    seen, out = set(), []
    for r in rows:
        k = norm(r["claim"]).lower()
        if k and k not in seen:
            seen.add(k)
            out.append(r)
    return out


def kb_rows(r, passages, prefix=None):
    base = dict(claim_id=r["id"], claim=r["claim"], label=r["label"], label_str=r["label_str"])
    rows = [dict(doc_id=f"{r['id']}_c", type="claim", text=r["claim"], **base)]
    for j, t in enumerate(x for x in passages if x):
        rows.append(dict(doc_id=f"{prefix or r['id']}_{j}", type="evidence", text=t, **base))
    return rows


def write(name, claims, kb):
    train = {r["id"] for r in claims if r["split"] == "train"}
    kb = [r for r in kb if r["claim_id"] in train]
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "claims.jsonl", "w", encoding="utf-8") as f:
        for r in claims:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(d / "kb.jsonl", "w", encoding="utf-8") as f:
        for r in kb:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    c = Counter((r["split"], r["label_str"]) for r in claims)
    k = Counter((r["type"], r["label_str"]) for r in kb)
    print(name, "claims", len(claims), dict(sorted(c.items())), "| kb", len(kb), dict(sorted(k.items())))


def pubhealth():
    z = zipfile.ZipFile(RAW / "pubhealth.zip")
    claims, kb = [], []
    for split in ["train", "dev", "test"]:
        rows = csv.DictReader(io.TextIOWrapper(z.open(f"PUBHEALTH/{split}.tsv"), encoding="utf-8"), delimiter="\t")
        for r in rows:
            lab = (r.get("label") or "").strip()
            if lab not in ("true", "false") or not norm(r.get("claim")):
                continue
            cid = f"ph_{r['claim_id']}"
            claims.append(dict(id=cid, claim=norm(r["claim"]), label=int(lab == "false"), label_str=lab, split=split,
                               _ev=[norm(r.get("explanation"))] + chunk(r.get("main_text"))))
    claims = dedup(claims)
    for r in claims:
        kb += kb_rows(r, r.pop("_ev"))
    write("pubhealth", claims, kb)


def climatefever():
    t = pq.read_table(RAW / "climate_fever_test.parquet").to_pylist()
    lab_map = {0: "true", 1: "false"}  # 0 SUPPORTS, 1 REFUTES; 2 NEI and 3 DISPUTED dropped
    rows = [r for r in t if r["claim_label"] in lab_map]
    claims = dedup([dict(id=f"cf_{r['claim_id']}", claim=norm(r["claim"]), label=int(r["claim_label"] == 1),
                         label_str=lab_map[r["claim_label"]], split=None, _ev=r["evidences"]) for r in rows])
    sp = stratified([(r["id"], r["label"]) for r in claims], (0.6, 0.1))
    kb = []
    for r in claims:
        r["split"] = sp[r["id"]]
        kb += kb_rows(r, [f"{e['article']}: {norm(e['evidence'])}" for e in r.pop("_ev")])
    write("climatefever", claims, kb)


def finfact():
    d = json.load(open(RAW / "finfact.json", encoding="utf-8"))
    rows = []
    for i, r in enumerate(d):
        if r.get("label") not in ("true", "false"):
            continue
        ev = []
        try:
            ev = [norm(x.get("sentence")) for x in ast.literal_eval(r.get("evidence") or "[]")]
        except Exception:
            pass
        rows.append(dict(id=f"ff_{i}", claim=norm(r["claim"]), label=int(r["label"] == "false"), label_str=r["label"],
                         split=None, _ev=chunk(r.get("justification")) + [" ".join(x for x in ev if x)]))
    claims = dedup(rows)
    sp = stratified([(r["id"], r["label"]) for r in claims], (0.7, 0.1))
    kb = []
    for r in claims:
        r["split"] = sp[r["id"]]
        kb += kb_rows(r, r.pop("_ev"))
    write("finfact", claims, kb)


if __name__ == "__main__":
    pubhealth(); climatefever(); finfact()
