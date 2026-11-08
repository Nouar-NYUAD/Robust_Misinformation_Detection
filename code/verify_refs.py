"""Check every entry of paper/references.bib against DBLP (fallback: Crossref): the title must exist, and the year,
the first author's surname and the number of authors are compared. Entries without a scholarly record (reports,
web pages, standards) are listed for manual checking. Prints a line per entry; mismatches are marked '!!'."""
import json, re, sys, time, urllib.parse, urllib.request
from difflib import SequenceMatcher
from pathlib import Path

BIB = Path(__file__).resolve().parents[1] / "paper/references.bib"


def parse(text):
    out = []
    for m in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}", text, flags=re.S):
        kind, key, body = m.group(1).lower(), m.group(2).strip(), m.group(3)
        f = {k.lower(): v for k, v in re.findall(r"(\w+)\s*=\s*\{((?:[^{}]|\{[^{}]*\})*)\}", body)}
        out.append((kind, key, f))
    return out


def clean(s):
    s = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", s or "")
    s = re.sub(r"\{\\[`'^\"~=.uvHtcdb]\s*\{?(\w)\}?\}", r"\1", s)
    return re.sub(r"[{}\\]", "", s).strip()


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", clean(s).lower())


def surnames(authors):
    out = []
    for a in re.split(r"\s+and\s+", clean(authors)):
        a = a.strip()
        if not a or a == "others":
            continue
        out.append((a.split(",")[0] if "," in a else a.split()[-1]).strip().lower())
    return out


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "reference-check (mailto:naa9497@nyu.edu)"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def dblp(title):
    """Semantic Scholar title search (DBLP rate-limits scripted queries); retried on HTTP 429."""
    q = urllib.parse.quote(clean(title))
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={q}&limit=5&fields=title,year,venue,authors,externalIds"
    for attempt in range(5):
        try:
            data = get(url).get("data", [])
            break
        except Exception as e:
            if "429" in str(e):
                time.sleep(5 * (attempt + 1)); continue
            return None, f"s2 error {e}"
    else:
        return None, "s2 rate-limited"
    best = None
    for i in data:
        sim = SequenceMatcher(None, norm(title), norm(i.get("title", ""))).ratio()
        if best is None or sim > best[0]:
            best = (sim, dict(title=i.get("title", ""), year=str(i.get("year", "")), venue=i.get("venue", ""),
                              authors=[a["name"] for a in i.get("authors", [])]))
    return best, None


def crossref(title):
    q = urllib.parse.quote(clean(title))
    try:
        items = get(f"https://api.crossref.org/works?query.title={q}&rows=5")["message"]["items"]
    except Exception as e:
        return None, f"crossref error {e}"
    best = None
    for it in items:
        t = (it.get("title") or [""])[0]
        sim = SequenceMatcher(None, norm(title), norm(t)).ratio()
        if best is None or sim > best[0]:
            y = (it.get("issued", {}).get("date-parts") or [[""]])[0][0]
            best = (sim, dict(title=t, year=str(y), venue=(it.get("container-title") or [""])[0],
                              authors=[f"{a.get('given', '')} {a.get('family', '')}".strip() for a in it.get("author", [])]))
    return best, None


def main():
    entries = parse(BIB.read_text(encoding="utf-8"))
    print(f"{len(entries)} entries\n")
    for kind, key, f in entries:
        title = f.get("title", "")
        if kind == "misc" or "url" in f and kind == "misc":
            print(f"MANUAL  {key}: {clean(title)} ({f.get('year', '')})")
            continue
        best, err = dblp(title)
        src = "semanticscholar"
        if not best or best[0] < 0.9:
            b2, err2 = crossref(title)
            if b2 and (not best or b2[0] > best[0]):
                best, src = b2, "crossref"
        time.sleep(3.0)
        if not best:
            print(f"!! {key}: NOT FOUND ({err})")
            continue
        sim, r = best
        ours_s = surnames(f.get("author", ""))
        their_s = [a.split()[-1].lower() for a in r["authors"]] if r["authors"] else []
        flags = []
        if sim < 0.9: flags.append(f"title sim {sim:.2f}")
        if str(f.get("year", "")) != str(r["year"]): flags.append(f"year {f.get('year')} vs {r['year']}")
        if ours_s and their_s and ours_s[0] != their_s[0]: flags.append(f"first author {ours_s[0]} vs {their_s[0]}")
        if their_s and "others" not in f.get("author", "") and len(ours_s) != len(their_s):
            flags.append(f"#authors {len(ours_s)} vs {len(their_s)}")
        mark = "!!" if flags else "ok"
        print(f"{mark} {key} [{src}] {'; '.join(flags)}\n     ours : {clean(title)} | {f.get('year')} | "
              f"{clean(f.get('booktitle') or f.get('journal') or '')}\n     found: {r['title']} | {r['year']} | {r['venue']}"
              + (f"\n     authors found: {', '.join(r['authors'])}" if flags else ""))


if __name__ == "__main__":
    main()
