"""Find DOIs / stable links for references.bib entries that have neither (no changes are written; prints a proposal).
Crossref title search; a DOI is accepted only if title similarity >= 0.95, the first author's surname matches and
the year is within 1 of ours. arXiv preprints get their arXiv abs URL. Others are listed for a manual link."""
import re, time, urllib.parse
from difflib import SequenceMatcher
from verify_refs import parse, clean, norm, surnames, get, BIB

for kind, key, f in parse(BIB.read_text(encoding="utf-8")):
    if "doi" in f or "url" in f:
        continue
    m = re.search(r"arXiv:(\d{4}\.\d{4,5})", f.get("journal", ""))
    if m:
        print(f"ARXIV  {key}: https://arxiv.org/abs/{m.group(1)}"); continue
    q = urllib.parse.quote(clean(f.get("title", "")))
    try:
        items = get(f"https://api.crossref.org/works?query.bibliographic={q}&rows=8")["message"]["items"]
    except Exception as e:
        print(f"ERROR  {key}: {e}"); continue
    ours = surnames(f.get("author", ""))
    best = None
    for it in items:
        t = re.sub(r"<[^>]+>", "", (it.get("title") or [""])[0])
        sim = SequenceMatcher(None, norm(f.get("title", "")), norm(t)).ratio()
        fa = ((it.get("author") or [{}])[0].get("family") or "").lower()
        y = (it.get("issued", {}).get("date-parts") or [[0]])[0][0] or 0
        ok = sim >= 0.95 and ours and norm(fa) == norm(ours[0]) and abs(int(y) - int(f.get("year", 0))) <= 1
        cand = (ok, sim, it.get("DOI"), t, fa, y, it.get("type"), (it.get("container-title") or [""])[0])
        if best is None or cand[:2] > best[:2]:
            best = cand
    if best and best[0]:
        print(f"DOI    {key}: {best[2]}  [{best[6]}; {best[7]}; {best[5]}]")
    else:
        print(f"MANUAL {key}: best={best[2:] if best else None}")
    time.sleep(1)
