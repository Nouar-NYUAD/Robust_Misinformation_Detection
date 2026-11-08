"""Lighter reference check, in batches: preprints are looked up EXACTLY by arXiv id (arXiv API), other papers by
title (Semantic Scholar, Crossref fallback). Reports title similarity, year, and the full author lists side by side.

usage: python verify_refs2.py <first> <last>   (entry indices, 0-based, inclusive) [--skip key1,key2,...]
"""
import re, sys, time, urllib.parse, urllib.request
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher
from verify_refs import parse, clean, norm, surnames, dblp as s2_search, crossref, BIB

ATOM = "{http://www.w3.org/2005/Atom}"


def arxiv(aid):
    url = f"http://export.arxiv.org/api/query?id_list={aid}"
    req = urllib.request.Request(url, headers={"User-Agent": "reference-check (mailto:naa9497@nyu.edu)"})
    with urllib.request.urlopen(req, timeout=30) as r:
        root = ET.fromstring(r.read())
    e = root.find(f"{ATOM}entry")
    if e is None or e.find(f"{ATOM}title") is None:
        return None
    return dict(title=re.sub(r"\s+", " ", e.find(f"{ATOM}title").text).strip(),
                year=e.find(f"{ATOM}published").text[:4],
                authors=[a.find(f"{ATOM}name").text for a in e.findall(f"{ATOM}author")], venue="arXiv")


def arxiv_title(title):
    """arXiv title search: best of the top 5 hits by title similarity (authors/title check for published papers)"""
    q = urllib.parse.quote(f'ti:"{clean(title)}"')
    url = f"http://export.arxiv.org/api/query?search_query={q}&max_results=5"
    req = urllib.request.Request(url, headers={"User-Agent": "reference-check (mailto:naa9497@nyu.edu)"})
    with urllib.request.urlopen(req, timeout=30) as r:
        root = ET.fromstring(r.read())
    best = None
    for e in root.findall(f"{ATOM}entry"):
        t = re.sub(r"\s+", " ", e.find(f"{ATOM}title").text).strip()
        sim = SequenceMatcher(None, norm(title), norm(t)).ratio()
        if best is None or sim > best[0]:
            best = (sim, dict(title=t, year=e.find(f"{ATOM}published").text[:4], venue="arXiv",
                              authors=[a.find(f"{ATOM}name").text for a in e.findall(f"{ATOM}author")]))
    return best


def main():
    a, b = int(sys.argv[1]), int(sys.argv[2])
    skip = set(sys.argv[sys.argv.index("--skip") + 1].split(",")) if "--skip" in sys.argv else set()
    entries = parse(BIB.read_text(encoding="utf-8"))
    for i, (kind, key, f) in enumerate(entries):
        if i < a or i > b or key in skip:
            continue
        title = f.get("title", "")
        if kind == "misc":
            print(f"[{i}] MANUAL {key}: {clean(title)} ({f.get('year', '')})", flush=True)
            continue
        m = re.search(r"arXiv:(\d{4}\.\d{4,5})", f.get("journal", "") + " " + f.get("eprint", ""))
        try:
            if m:
                r, src = arxiv(m.group(1)), "arxiv"
                time.sleep(3)
                sim = SequenceMatcher(None, norm(title), norm(r["title"])).ratio() if r else 0
            else:
                best, _ = s2_search(title); src = "s2"
                if not best or best[0] < 0.9:
                    b3 = arxiv_title(title); time.sleep(3)
                    if b3 and (not best or b3[0] > best[0]):
                        best, src = b3, "arxiv-title"
                if not best or best[0] < 0.9:
                    b2, _ = crossref(title)
                    if b2 and (not best or b2[0] > best[0]):
                        best, src = b2, "crossref"
                sim, r = best if best else (0, None)
        except Exception as e:
            print(f"[{i}] ERROR {key}: {e}", flush=True); continue
        if not r:
            print(f"[{i}] !! {key}: NOT FOUND", flush=True); continue
        ours = surnames(f.get("author", ""))
        theirs = [x.split()[-1].lower() for x in r["authors"]]
        flags = []
        if sim < 0.9: flags.append(f"title sim {sim:.2f}")
        if src not in ("arxiv", "arxiv-title") and str(f.get("year", "")) != str(r["year"]):
            flags.append(f"year {f.get('year')} vs {r['year']}")
        if src in ("arxiv", "arxiv-title") and int(f.get("year", 0) or 0) < int(r["year"]):
            flags.append(f"year {f.get('year')} < arXiv {r['year']}")  # a venue year >= the arXiv year is expected
        if ours and theirs and norm(ours[0]) != norm(theirs[0]): flags.append(f"first author {ours[0]} vs {theirs[0]}")
        if "others" not in f.get("author", "") and theirs and len(ours) != len(theirs):
            flags.append(f"#authors {len(ours)} vs {len(theirs)}")
        print(f"[{i}] {'!!' if flags else 'ok'} {key} [{src}] {'; '.join(flags)}\n      ours : {clean(title)} | "
              f"{f.get('year')} | {', '.join(ours)}\n      found: {r['title']} | {r['year']} | "
              f"{', '.join(x.split()[-1].lower() for x in r['authors'])}", flush=True)


if __name__ == "__main__":
    main()
