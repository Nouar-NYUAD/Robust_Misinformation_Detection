"""Appendix material generated from the code and data, so that it always matches what was run.

paper/tables/example.tex  one PUBHEALTH test claim under all fifteen attacks (two tables: single, chained)
paper/tables/prompts.tex  every prompt used, read from the source files with `ast` (no module is imported, so no
                          model is loaded); placeholders such as {q} are shown as <q>
"""
import ast, json, re
from pathlib import Path
from common import display

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
OUT = ROOT / "paper/tables"
EXAMPLE_ID, EXAMPLE_DS = "ph_9008", "pubhealth"
MAX_WORDS = 55
NON_LATIN = {"ar": "Arabic", "hi": "Hindi", "zh": "Chinese"}


def consts(path):
    """module-level constants (str, dict, tuple of str) of a source file, evaluated literally"""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass  # not a literal (e.g. a dict built with ** or a call): skipped
    return out


def esc(t):
    """escape LaTeX specials; keep line breaks as \\newline"""
    rep = {"\\": "\\textbackslash{}", "&": "\\&", "%": "\\%", "$": "\\$", "#": "\\#", "_": "\\_", "{": "\\{",
           "}": "\\}", "~": "\\textasciitilde{}", "^": "\\textasciicircum{}", "<": "\\textless{}",
           ">": "\\textgreater{}"}
    t = "".join(rep.get(ch, ch) for ch in t)
    t = t.replace("\u201c", "``").replace("\u201d", "''").replace("\u2019", "'").replace("\u2018", "`")
    t = t.replace("\u2014", "---").replace("\u2013", "--")
    bad = sorted({ch for ch in t if ord(ch) > 0xFF})
    assert not bad, f"characters pdfLaTeX cannot typeset: {bad}"
    return re.sub(r"\n\s*\n+", lambda m: " \\newline ", t).replace("\n", " ")  # function: no escape processing


def placeholders(t):
    return re.sub(r"\{(\w*)\}", lambda m: f"<{m.group(1) or 'text'}>", t)


def shorten(t):
    w = t.split()
    return " ".join(w[:MAX_WORDS]) + (" [...]" if len(w) > MAX_WORDS else "")


def example_tables():
    rows = {r["attack"]: r for r in map(json.loads, open(ROOT / f"data/attacks/{EXAMPLE_DS}.jsonl", encoding="utf-8"))
            if r["id"] == EXAMPLE_ID and not r["attack"].startswith("ADAPT_")}
    valid = {r["attack"]: r["p_preserved"] >= 0.5 for r in map(json.loads, open(
        ROOT / f"data/attacks/{EXAMPLE_DS}.valid.jsonl", encoding="utf-8")) if r["id"] == EXAMPLE_ID}
    assert len(rows) == 16 and all(valid[a] for a in rows if a != "CLEAN"), "example must preserve the claim everywhere"
    label = "false" if rows["CLEAN"]["label"] == 1 else "true"

    def cell(a):
        """English task wrapper; any text in another language is described, not shown, the same way for every
        language (the claim content is already visible in the English attacks)"""
        r = rows[a]
        if r["lang"]:
            wrapper = r["query"][: len(r["query"]) - len(r["text"])].strip()
            what = {"TR>MCQ": "multiple-choice question", "TR>XML": "XML document",
                    "TR>MCQ>JSON": "JSON object containing a multiple-choice question"}.get(a, "text")
            lang = {**NON_LATIN, "fr": "French", "es": "Spanish"}[r["lang"]]
            desc = f"{lang} {what}" if what == "text" else f"{what} in {lang}"
            return esc(wrapper) + f" \\newline \\emph{{[{desc}]}}"
        return esc(shorten(r["query"]))

    name = lambda a: display(a).replace("TR_", "Translate ").replace(">", "$\\to$").replace("SUM", "Summarise")
    single = ["TR_fr", "TR_es", "TR_ar", "TR_hi", "TR_zh", "SUM", "MCQ", "XML", "JSON"]
    chains = [a for a in rows if ">" in a]
    out = []
    for part, atts, lab in (("single", single, "tab:example_single"), ("chained", chains, "tab:example_chain")):
        lines = ["\\begin{table*}[t]\\centering\\footnotesize",
                 "\\begin{tabular}{@{}p{0.13\\textwidth}p{0.84\\textwidth}@{}}", "\\toprule",
                 f"Attack & Text given to the detector \\\\", "\\midrule"]
        if part == "single":
            lines.append(f"Clean & {esc(rows['CLEAN']['query'])} \\\\ \\midrule")
        for a in atts:
            lines.append(f"{name(a)} & {cell(a)} \\\\")
        cap = (f"One {label} PUBHEALTH test claim under the {part} attacks, as given to the detectors (the attack "
               f"text and its task wrapper). Long texts are cut after {MAX_WORDS} words ([...]); texts in other "
               f"languages are described in brackets rather than shown. The judge found "
               f"that every attacked version preserves the claim.")
        if part == "chained":
            cap += f" The chained translations of this claim are into {({'fr': 'French', 'es': 'Spanish'}).get(rows[chains[0]]['lang'], rows[chains[0]]['lang'])}."
        lines += ["\\bottomrule", "\\end{tabular}", f"\\caption{{{cap}}}", f"\\label{{{lab}}}", "\\end{table*}"]
        out.append("\n".join(lines))
    return "\n\n".join(out) + "\n"


def prompt_block(title, text):
    return (f"\\noindent\\textit{{{title}}}\n\\begin{{quote}}\\small\\ttfamily {esc(placeholders(text))}"
            f"\\end{{quote}}\n")


def prompts_tex():
    ga, cz, dt = consts(CODE / "gen_attacks.py"), consts(CODE / "canonicalize.py"), consts(CODE / "detect.py")
    va, ad, ag = consts(CODE / "validate_attacks.py"), consts(CODE / "adaptive_attacks.py"), consts(CODE / "arg_run.py")
    ct = consts(CODE / "contamination.py")
    s = ["\\subsection{Attack generation}",
         "Each attack applies one or more transformations with Qwen2.5-32B-Instruct and then prepends a task "
         "wrapper. Transformation prompts (the text to transform follows after a blank line):\n"]
    names = {"expand_short": "Short expansion", "expand_long": "Long expansion (summarisation attacks)",
             "mcq": "Multiple-choice question", "xml": "XML", "json": "JSON", "translate": "Translation"}
    for k, v in ga["OP_PROMPTS"].items():
        s.append(prompt_block(names[k], v))
    s.append("Task wrappers, one per attack (\\texttt{\\textless{}L\\textgreater{}} is the language name):\n")
    wr = re.findall(r'"([A-Z_>{}a-z]+)":\s*\(\[[^\]]*\],\s*"([^"]+)"\)', (CODE / "gen_attacks.py").read_text(encoding="utf-8"))
    assert len(wr) == 11 and wr[0][0] == "TR_{l}", wr  # one translation template (5 languages) + 10 others
    disp = lambda a: ("Translation (each language)" if a == "TR_{l}" else
                      esc(display(a)).replace("\\textgreater{}", " $\\to$ ").replace("TR", "Translate"))
    s.append("\\begin{quote}\\small " + " \\newline ".join(
        f"{disp(a)}: \\texttt{{{esc(placeholders(w))}}}" for a, w in wr) + "\\end{quote}\n")
    s += ["\\subsection{Preservation judge (Gemma-3-12B)}", prompt_block("Judge", va["PROMPT"]),
          "\\subsection{\\method{}}", prompt_block("Canonicaliser", cz["PROMPTS"]["ours"]),
          prompt_block("Generic claim normalisation (ablation)", cz["PROMPTS"]["norm"]),
          prompt_block("Verifier", dt["VERIFY_PROMPT"]),
          "Each retrieved record is shown as \\texttt{[k] Claim: <claim> / Verdict: TRUE|FALSE / Evidence: <best "
          "passage, at most 80 words>}.\n",
          "\\subsection{Baselines}", prompt_block("Zero-shot LLMs (Llama, Qwen, Claude)", dt["ZS_PROMPT"]),
          prompt_block("Claim retrieval + LLM", dt["RAG_PROMPT"]),
          prompt_block("MiniCheck, system message (official)", dt["MINICHECK_SYSTEM"]),
          prompt_block("MiniCheck, user message (official)", dt["MINICHECK_USER"]),
          prompt_block("ARG rationales (official template; the assistant turn starts with \"Let's think from the "
                       "perspective of textual description / commonsense.\")", ag["Q_TMPL"]),
          "ProgramFC uses the official program-generation prompts of \\citet{pan2023programfc}, which are long "
          "few-shot prompts and are not repeated here.\n",
          "\\subsection{Adaptive attacks}", prompt_block("Injected note (\\texttt{<d>} is a true training claim)", ad["INJ"]),
          "\\subsection{Contamination test}",
          prompt_block("Guided, instruction format", ct["GUIDED"]), prompt_block("General, instruction format", ct["GENERAL"]),
          prompt_block("Guided, pre-filled (Llama; the answer starts with the first piece)", ct["GUIDED_PREFILL"]),
          prompt_block("General, pre-filled", ct["GENERAL_PREFILL"])]
    return "\n".join(s)


def data_table():
    DSN = {"pubhealth": "PUBHEALTH", "climatefever": "CLIMATE-FEVER", "finfact": "Fin-Fact"}
    split = {"pubhealth": "official", "climatefever": "ours, 60/10/30", "finfact": "ours, 70/10/20"}
    lines = ["\\begin{table}[t]\\centering\\small\\setlength{\\tabcolsep}{4pt}",
             "\\begin{tabular}{lrrrr}", "\\toprule", "Dataset & Train & Dev & Test & False (\\%) \\\\", "\\midrule"]
    for ds in DSN:
        cl = [json.loads(l) for l in open(ROOT / f"data/processed/{ds}/claims.jsonl", encoding="utf-8")]
        n = {s: sum(c["split"] == s for c in cl) for s in ("train", "dev", "test")}
        assert sum(n.values()) == len(cl), ds
        false = 100 * sum(c["label"] == 1 for c in cl) / len(cl)
        lines.append(f"{DSN[ds]} & {n['train']:,} & {n['dev']:,} & {n['test']:,} & {false:.0f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}",
              "\\caption{Binary true/false subsets used in this work (claims per split, and share of false claims). "
              "PUBHEALTH uses its official split; CLIMATE-FEVER and Fin-Fact are split by us (stratified, seed 13), "
              "and we release the split identifiers.}", "\\label{tab:data}", "\\end{table}"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    (OUT / "data.tex").write_text(data_table(), encoding="utf-8")
    (OUT / "example.tex").write_text(example_tables(), encoding="utf-8")
    (OUT / "prompts.tex").write_text(prompts_tex(), encoding="utf-8")
    print("written example.tex, prompts.tex")
