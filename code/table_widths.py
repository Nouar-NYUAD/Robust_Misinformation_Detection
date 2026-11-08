"""Estimate the natural width of every generated table (paper/tables/*.tex), using Times font metrics, and compare
it with the ACL text width (455pt) or column width (219pt). No LaTeX needed. A table whose estimate exceeds the
available width is shrunk by adjustbox, i.e. printed in a smaller font; the goal is a scale of 1.0 everywhere.

Approximation: each cell's text is measured after removing LaTeX markup, in Times at the table's font size
(\\small = 9pt in an 11pt document, \\footnotesize = 8pt, \\scriptsize = 7pt), and every column gets two times
\\tabcolsep. Multicolumn header cells are ignored (they rarely set the width). Expect about +-5% accuracy.
"""
import re, sys
from pathlib import Path
from matplotlib.textpath import TextToPath
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parents[1]
TEXT, COL = 455.24, 219.08
SIZE = {"\\small": 9.0, "\\footnotesize": 8.0, "\\scriptsize": 7.0, "\\normalsize": 10.95}
FP = FontProperties(family="serif", fname=None)
FP.set_name("Times New Roman")
T2P = TextToPath()


def width(text, size):
    if not text.strip():
        return 0.0
    w, _, _ = T2P.get_text_width_height_descent(text, FP, ismath=False)
    return w * size / FP.get_size_in_points()


def clean(cell):
    c = re.sub(r"\$\^\{[^}]*\}\$", "*", cell)                   # superscript markers
    c = c.replace("\\method{}", "UNWRAP").replace("\\air{}", "AIR")
    c = re.sub(r"\\citep\{[^}]*\}", "(Author et al., 2024)", c)
    c = re.sub(r"\\(emph|textbf|textit)\{([^}]*)\}", r"\2", c)
    c = c.replace("$\\to$", ">").replace("$\\rightarrow$", "->").replace("\\%", "%").replace("\\quad", "    ")
    c = re.sub(r"\$([^$]*)\$", r"\1", c)
    c = re.sub(r"\\[a-zA-Z]+", "", c).replace("{", "").replace("}", "").replace("~", " ").replace("\\", "")
    return c.strip()


def table_width(tex):
    size = next((v for k, v in SIZE.items() if k in tex.split("\\begin{tabular}")[0]), 10.95)
    m = re.search(r"\\setlength\{\\tabcolsep\}\{([\d.]+)pt\}", tex)
    sep = float(m.group(1)) if m else 6.0
    body = tex.split("\\begin{tabular}")[1].split("\\end{tabular}")[0]
    rows = [r for r in body.split("\\\\") if "&" in r and "multicolumn" not in r]
    ncol = max(r.count("&") + 1 for r in rows)
    colw = [0.0] * ncol
    for r in rows:
        r = re.sub(r"\\(toprule|midrule|bottomrule|cmidrule\(lr\)\{[^}]*\})", "", r)
        for j, c in enumerate(r.split("&")):
            colw[j] = max(colw[j], width(clean(c), size))
    return sum(colw) + 2 * sep * ncol, size


if __name__ == "__main__":
    bad = 0
    for p in sorted((ROOT / "paper/tables").glob("*.tex")):
        tex = p.read_text(encoding="utf-8")
        avail = TEXT if "\\begin{table*}" in tex else COL
        w, size = table_width(tex)
        scale = min(1.0, avail / w)
        flag = "OK" if scale >= 0.97 else "SHRUNK"
        bad += flag != "OK"
        print(f"{p.name:30s} {size:4.1f}pt  natural {w:6.1f}pt / {avail:5.1f}pt  scale {scale:.2f}  {flag}")
    sys.exit(1 if bad else 0)
