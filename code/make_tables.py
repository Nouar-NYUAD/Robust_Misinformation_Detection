"""LaTeX tables + numbers.json for the paper, built only from the analysis outputs:
results/tables/{summary,compare,compare_settingB,compare_ablations,plugin_*,seen_unseen,seen_unseen_B,adaptive,
preservation}.csv and results/contamination/*.summary.json. No number in the paper is typed by hand.

Significance markers: paired bootstrap (compare.py), two-sided p < 0.05.
out: paper/tables/{main,ablation,plugin,seen_unseen,adaptive,contamination,preservation,per_attack_<ds>}.tex,
     paper/numbers.json
"""
import csv, json
import numpy as np
from common import ROOT, DATASETS, ATTACK_ORDER, display

T = ROOT / "results/tables"
C = ROOT / "results/contamination"
OUT = ROOT / "paper/tables"
ALPHA = 0.05
DSN = {"pubhealth": "PUBHEALTH", "climatefever": "CLIMATE-FEVER", "finfact": "Fin-Fact"}
ATTACKS = [a for a in ATTACK_ORDER if a != "CLEAN"]
OURS_A, OURS_B = "ours_vtrained", "ours_air_vtrained_aug"

# Setting A: trained on clean claims only. Setting B: also on transformed training claims of the seen families.
# Setting A has two sub-groups: no parameters learned from the benchmarks (training-free), and fine-tuned on clean
# training claims. Significance markers in Setting A are always against the fine-tuned UNWRAP (OURS_A).
OURS_FREE = "ours"  # UNWRAP with the off-the-shelf verifier
SETTING_A_GROUPS = [
    ("Training-free", [("programfc", "ProgramFC \\citep{pan2023programfc}"),
                       ("minicheck", "MiniCheck \\citep{tang2024minicheck}"),
                       ("rag", "Claim retrieval + LLM \\citep{pikuliak2023multiclaim}"),
                       ("zs", "Llama-3.1-8B zero-shot"), ("zs_qwen", "Qwen2.5-7B zero-shot"),
                       (OURS_FREE, "\\method{}, no training (ours)")]),
    ("Fine-tuned on clean claims", [("kotonya", "SciBERT + evidence \\citep{kotonya2020pubhealth}"),
                                    ("arg", "ARG \\citep{hu2024arg}"), ("clf_xlmr", "XLM-R"),
                                    (OURS_A, "\\method{} (ours)")])]
SETTING_A = [r for _, rows in SETTING_A_GROUPS for r in rows]
SETTING_B = [("kotonya-advtrain", "SciBERT + evidence, adv.\\ training"), ("arg-advtrain", "ARG, adv.\\ training"),
             ("clf_xlmr-advtrain", "XLM-R, adv.\\ training"), (OURS_B, "\\method{} + \\air{} + aug.\\ (ours)")]
# (prediction file, row label, group). Every row changes exactly one thing relative to the full model.
ABL = [("ours_vtrained_nocanon", "Canonicalisation (raw input only)", "Remove"),
       ("ours_vtrained_nosparse", "Sparse retrieval (dense only)", "Remove"),
       ("ours_vtrained_norerank", "Cross-encoder reranker", "Remove"),
       ("ours", "Verifier training (off-the-shelf LLM)", "Remove"),
       ("ours_vtrained_normcanon", "Canonicalisation $\\to$ generic claim normalisation", "Replace"),
       ("ours_vtrained_air_nocanon", "Canonicalisation $\\to$ \\air{} dense encoder", "Replace"),
       ("ours_vtrained_me5", "BGE-M3 $\\to$ mE5 (dense only)", "Replace"),
       ("ours_vtrained_air", "BGE-M3 dense encoder $\\to$ \\air{} encoder", "Replace")]
ABL_GROUP = {"Remove": "Remove one component", "Replace": "Replace one component (original $\\to$ replacement)"}
PLUG = [("kotonya", None, "SciBERT + evidence"), ("clf_xlmr", "clf_xlmr+canon", "XLM-R"),
        ("clf_xlmr-advtrain", "clf_xlmr-advtrain+canon", "XLM-R, adv.\\ training"),
        ("zs", "zs_canon", "Llama-3.1-8B zero-shot"), ("zs_qwen", "zs_qwen+canon", "Qwen2.5-7B zero-shot"),
        ("rag", "rag_canon", "Claim retrieval + LLM"), ("minicheck", "minicheck_canon", "MiniCheck")]
PLUG = [p for p in PLUG if p[1]]
SU_ROWS = [("kotonya", "SciBERT + evidence", "A"), ("arg", "ARG", "A"), ("clf_xlmr", "XLM-R", "A"),
           (OURS_A, "\\method{}", "A"), ("kotonya-advtrain", "SciBERT + evidence", "B"), ("arg-advtrain", "ARG", "B"),
           ("clf_xlmr-advtrain", "XLM-R", "B"), (OURS_B, "\\method{} + \\air{} + aug.", "B")]
ADAPT_ROWS = [("rag", "Claim retrieval + LLM"), ("clf_xlmr-advtrain", "XLM-R, adv.\\ training"),
              (OURS_A, "\\method{}"), (OURS_B, "\\method{} + \\air{} + aug.")]
ADAPT_BASES = ["SUM", "MCQ>JSON"]


def load(name):
    p = T / f"{name}.csv"
    assert p.exists(), f"missing {p} (run aggregate.py / compare.py / seen_unseen.py first)"
    return list(csv.DictReader(open(p, encoding="utf-8")))


pc = lambda x: "--" if x in (None, "", "nan") else f"{100 * float(x):.1f}"
sig = lambda p: float(p) < ALPHA
pfmt = lambda p, B: (f"$<${1 / B:g}" if float(p) < 1 / B else f"{float(p):.4f}" if float(p) < 0.001
                    else f"{float(p):.3f}")  # bootstrap resolution 1/B
nums = {}

S = {(r["dataset"], r["method"]): r for r in load("summary")}
for (d, m), r in S.items():  # attacked_f1 must be the mean of the per-attack scores
    per = [float(r[f"f1_{a}"]) for a in ATTACKS if r.get(f"f1_{a}") not in (None, "", "nan")]
    assert len(per) == len(ATTACKS) and abs(np.mean(per) - float(r["attacked_f1"])) < 1e-9, (d, m)


def comp_index(rows):
    out = {}
    for r in rows:
        key = (r["dataset"], r["a"], r["b"])
        assert key not in out, key
        assert int(r["n_attacks"]) == len(ATTACKS), key
        out[key] = r
    return out


CMP = comp_index(load("compare"))
CMP_B = comp_index(load("compare_settingB"))
CMP_ABL = comp_index(load("compare_ablations"))


def table_star(lines_body, ncols_per_ds, header_cells, caption, label, colspec=None):
    colspec = colspec or ("l" + "c" * ncols_per_ds * 3)
    lines = ["\\begin{table*}[t]\\centering\\small\\setlength{\\tabcolsep}{5pt}", f"\\begin{{tabular}}{{{colspec}}}",
             "\\toprule",
             " & " + " & ".join(f"\\multicolumn{{{ncols_per_ds}}}{{c}}{{{DSN[d]}}}" for d in DATASETS) + " \\\\",
             " ".join(f"\\cmidrule(lr){{{2 + ncols_per_ds * i}-{1 + ncols_per_ds * (i + 1)}}}" for i in range(3)),
             header_cells + " \\\\", "\\midrule"] + lines_body
    lines += ["\\bottomrule", "\\end{tabular}", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\end{table*}"]
    return "\n".join(lines)


# ---------------------------------------------------------------- main table
def main_table():
    body = []
    for block, rows, ours, cmp in (("A", SETTING_A, OURS_A, CMP), ("B", SETTING_B, OURS_B, CMP_B)):
        title = ("Setting A: no attacked or transformed text in training" if block == "A" else
                 "Setting B: also trained on transformed claims of the seen attack families")
        body.append(f"\\multicolumn{{10}}{{l}}{{\\emph{{{title}}}}} \\\\")
        best = {}
        for d in DATASETS:
            for k in ("clean_f1", "attacked_f1"):
                best[d, k] = max(float(S[d, m][k]) for m, _ in rows if (d, m) in S)
        sub = {grp[0][0]: g for g, grp in SETTING_A_GROUPS} if block == "A" else {}
        for m, name in rows:
            if m in sub:  # sub-group header inside Setting A
                body.append(f"\\multicolumn{{10}}{{l}}{{\\quad\\textit{{{sub[m]}}}}} \\\\")
            cells = []
            for d in DATASETS:
                r = S.get((d, m))
                if not r:
                    cells += ["--"] * 3
                    continue
                for k in ("clean_f1", "attacked_f1"):
                    v = pc(r[k])
                    if abs(float(r[k]) - best[d, k]) < 1e-12:
                        v = f"\\textbf{{{v}}}"
                    c = cmp.get((d, ours, m))
                    pkey = "clean_p" if k == "clean_f1" else "attacked_p"
                    dkey = "clean_diff" if k == "clean_f1" else "attacked_diff"
                    if m != ours and c and sig(c[pkey]):
                        v += "$^{\\dagger}$" if float(c[dkey]) > 0 else "$^{\\ddagger}$"
                    cells.append(v)
                cells.append(f"${-100 * float(r['drop_f1']):+.1f}$")
            body.append(("\\quad " if block == "A" else "") + name + " & " + " & ".join(cells) + " \\\\")
        if block == "A":
            body.append("\\midrule")
    cap = ("Macro-F1 (\\%) on clean test claims (Clean), averaged over the fifteen format-shifting attacks (Att.), and "
           "the change from clean to attacked ($\\Delta$). Attacked texts are those the judge found to preserve the claim. "
           "Setting~A: no attacked or transformed text in training, split into training-free detectors (no parameters "
           "learned from the benchmarks) and detectors fine-tuned on clean claims. Fine-tuned models are averaged over "
           "three seeds, except \\method{} + \\air{} + aug.\\ (one seed). Best per setting in bold. "
           "$^{\\dagger}$/$^{\\ddagger}$: significantly worse/better than the fine-tuned \\method{} of the same "
           "setting (paired bootstrap, $p<0.05$).")
    return table_star(body, 3, "Detector & " + " & ".join(["Clean & Att. & $\\Delta$"] * 3), cap, "tab:main")


# ---------------------------------------------------------------- ablations (seed 0 throughout)
def ablation_table():
    full = {}
    for d in DATASETS:  # full model at seed 0 = variant + (full - variant); must agree across all variants
        for k, dk in (("clean_f1", "clean_diff"), ("attacked_f1", "attacked_diff")):
            vals = [float(S[d, m][k]) + float(CMP_ABL[d, f"{OURS_A}@s0", m][dk]) for m, _, _ in ABL]
            assert max(vals) - min(vals) < 1e-9, (d, k, vals)
            full[d, k] = vals[0]
        assert all(S[d, m]["seeds"] == "1" for m, _, _ in ABL)
    nums["ablation_full_s0"] = {f"{d}/{k}": v for (d, k), v in full.items()}
    body = ["Full \\method{} & " + " & ".join(f"{pc(full[d, k])}" for d in DATASETS
                                               for k in ("clean_f1", "attacked_f1")) + " \\\\"]
    for i, (m, name, grp) in enumerate(ABL):
        if i == 0 or ABL[i - 1][2] != grp:
            body += ["\\midrule", f"\\multicolumn{{7}}{{l}}{{\\emph{{{ABL_GROUP[grp]}}}}} \\\\"]
        cells = []
        for d in DATASETS:
            c = CMP_ABL[d, f"{OURS_A}@s0", m]
            for k, pk in (("clean_f1", "clean_p"), ("attacked_f1", "attacked_p")):
                cells.append(pc(S[d, m][k]) + ("$^{*}$" if sig(c[pk]) else ""))
            nums[f"ablation/{d}/{m}"] = {k: c[k] for k in ("clean_diff", "clean_p", "attacked_diff", "attacked_p")}
        body.append("\\quad " + name + " & " + " & ".join(cells) + " \\\\")
    cap = ("Ablations of \\method{}. Each row removes one component of the full model, or replaces it with the "
           "alternative after the arrow, and keeps everything else unchanged. Macro-F1 (\\%) on clean claims and "
           "averaged over the fifteen attacks (Att.); all rows use the first seed. $^{*}$: significantly different "
           "from the full model (paired bootstrap, $p<0.05$).")
    return table_star(body, 2, "Model & " + " & ".join(["Clean & Att."] * 3), cap, "tab:ablation")


# ---------------------------------------------------------------- canonicaliser as a plug-in
def plugin_table():
    body = []
    for a, b, name in PLUG:
        rows = comp_index(load(f"plugin_{a}"))
        cells = []
        for d in DATASETS:
            c = rows[d, b, a]
            assert abs(float(S[d, b]["attacked_f1"]) - float(S[d, a]["attacked_f1"]) - float(c["attacked_diff"])) < 1e-9
            assert abs(float(S[d, b]["clean_f1"]) - float(S[d, a]["clean_f1"]) - float(c["clean_diff"])) < 1e-9
            cells += [pc(S[d, a]["clean_f1"]), pc(S[d, b]["clean_f1"]) + ("$^{*}$" if sig(c["clean_p"]) else ""),
                      pc(S[d, a]["attacked_f1"]), pc(S[d, b]["attacked_f1"]) + ("$^{*}$" if sig(c["attacked_p"]) else "")]
            nums[f"plugin/{d}/{a}"] = {k: c[k] for k in ("clean_diff", "clean_p", "attacked_diff", "attacked_p")}
        body.append(name + " & " + " & ".join(cells) + " \\\\")
    lines = ["\\begin{table*}[t]\\centering\\small\\setlength{\\tabcolsep}{4pt}",
             "\\begin{tabular}{l" + "cccc" * 3 + "}", "\\toprule",
             " & " + " & ".join(f"\\multicolumn{{4}}{{c}}{{{DSN[d]}}}" for d in DATASETS) + " \\\\",
             " ".join(f"\\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}" for i in range(3)),
             " & " + " & ".join(["\\multicolumn{2}{c}{Clean} & \\multicolumn{2}{c}{Att.}"] * 3) + " \\\\",
             "Detector & " + " & ".join(["w/o & with"] * 6) + " \\\\", "\\midrule"] + body + [
             "\\bottomrule", "\\end{tabular}",
             "\\caption{Macro-F1 (\\%) of existing detectors without (w/o) and with \\method{}'s canonicaliser in "
             "front of them, on clean claims and averaged over the fifteen attacks. $^{*}$: significant change "
             "(paired bootstrap, $p<0.05$).}", "\\label{tab:plugin}", "\\end{table*}"]
    return "\n".join(lines)


# ---------------------------------------------------------------- seen vs unseen attack families
def seen_unseen_table():
    su = {(r["dataset"], r["method"]): r for r in load("seen_unseen") + load("seen_unseen_B")}
    groups = [("seen", "Seen"), ("unseen_single", "Unseen"), ("unseen_chain", "Chains")]
    body = []
    for i, (m, name, setting) in enumerate(SU_ROWS):
        if i == 0 or SU_ROWS[i - 1][2] != setting:
            if i:
                body.append("\\midrule")
            body.append(f"\\multicolumn{{10}}{{l}}{{\\emph{{Setting {setting}}}}} \\\\")
        cells = []
        for d in DATASETS:
            r = su[d, m]
            assert abs(float(r["clean"]) - float(S[d, m]["clean_f1"])) < 1e-9, (d, m)
            cells += [pc(r[g]) for g, _ in groups]
            nums[f"seen_unseen/{d}/{m}"] = {g: r[g] for g, _ in groups + [("unseen_all", "")]}
        body.append(f"{name} & " + " & ".join(cells) + " \\\\")
    cap = ("Macro-F1 (\\%) by attack group. Setting~B adds transformed training claims of four attack families only. "
           "Seen: test attacks of these families (French and Spanish translation, summarisation, MCQ); the training "
           "copies were produced by a different model and instructions than the test attacks, so the type is seen "
           "but not the texts. Unseen: single attacks of families absent from training (Arabic, Hindi and Chinese "
           "translation, XML, JSON). Chains: the six chained attacks, never seen as such. Setting~A models saw no "
           "attacks at all; their rows use the same split for comparison.")
    return table_star(body, 3, "Detector & " + " & ".join(["Seen & Unseen & Chains"] * 3), cap, "tab:seen_unseen")


# ---------------------------------------------------------------- adaptive attacks
def adaptive_table():
    A = {(r["dataset"], r["method"], r["attack"]): float(r["macro_f1"]) for r in load("adaptive")}
    body = []
    for m, name in ADAPT_ROWS:
        cells = []
        for d in DATASETS:
            base = np.mean([float(S[d, m][f"f1_{a}"]) for a in ADAPT_BASES])
            inj = np.mean([A[d, m, f"ADAPT_INJ:{a}"] for a in ADAPT_BASES])
            dec = np.mean([A[d, m, f"ADAPT_DECOY:{a}"] for a in ADAPT_BASES])
            cells += [pc(base), pc(inj), pc(dec)]
            nums[f"adaptive/{d}/{m}"] = dict(base=base, inj=inj, decoy=dec)
        body.append(name + " & " + " & ".join(cells) + " \\\\")
    cap = ("Adaptive attacks. Macro-F1 (\\%) averaged over the summarisation and MCQ$\\rightarrow$JSON attacks, "
           "without (Base) and with an injected instruction to the claim extractor (Inj.) or three true decoy claims "
           "around the content (Decoy).")
    return table_star(body, 3, "Detector & " + " & ".join(["Base & Inj. & Decoy"] * 3), cap, "tab:adaptive")


# ---------------------------------------------------------------- contamination
def contamination_table():
    pre = json.load(open(C / "llama_prefill.summary.json"))
    ins = json.load(open(C / "llama.summary.json"))
    cl = json.load(open(C / "claude.summary.json"))
    nums["contamination_llama_prefill"], nums["contamination_llama_instruction"] = pre, ins
    nums["contamination_claude"] = cl
    assert all(pre[d]["n"] == cl[d]["n"] for d in DATASETS)
    dash = lambda r: "--"

    def row(name, f_llama, f_claude, src_llama=pre):
        return (name + " & " + " & ".join(f_llama(src_llama[d]) for d in DATASETS) + " & "
                + " & ".join(f_claude(cl[d]) for d in DATASETS) + " \\\\")
    short = {"pubhealth": "PUBH.", "climatefever": "C-FEVER", "finfact": "Fin-Fact"}
    lines = ["\\begin{table*}[t]\\centering\\small\\setlength{\\tabcolsep}{5pt}",
             "\\begin{tabular}{l" + "c" * 6 + "}", "\\toprule",
             " & \\multicolumn{3}{c}{Llama-3.1-8B-Instruct (answer pre-filled)} & "
             "\\multicolumn{3}{c}{Claude Opus 5.5 (instruction format)} \\\\",
             "\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}",
             " & " + " & ".join([short[d] for d in DATASETS] * 2) + " \\\\", "\\midrule",
             row("Claims tested", lambda r: str(r["n"]), lambda r: str(r["n"])),
             row("ROUGE-L, guided", lambda r: f"{r['rougeL_guided']:.3f}", lambda r: f"{r['rougeL_guided']:.3f}"),
             row("ROUGE-L, general", lambda r: f"{r['rougeL_general']:.3f}", lambda r: f"{r['rougeL_general']:.3f}"),
             row("One-sided $p$ (guided $>$ general)", lambda r: pfmt(r["p_one_sided"], 10000),
                 lambda r: pfmt(r["p_one_sided"], 10000)),
             row("\\quad non-empty pairs only", lambda r: pfmt(r["p_one_sided_nonempty"], 10000), dash),
             row("Exact, guided", lambda r: str(r["exact_guided"]), lambda r: str(r["exact_guided"])),
             row("Near-exact, guided", lambda r: str(r["near_exact_guided"]), lambda r: str(r["near_exact_guided"])),
             row("Exact, general", lambda r: str(r["exact_general"]), lambda r: str(r["exact_general"])),
             row("Guided answers refused or hedged$^{a}$", lambda r: f"{100 * r['refusal_guided']:.0f}\\%",
                 lambda r: f"{100 * r['refusal_guided']:.0f}\\%", src_llama=ins),
             "\\bottomrule", "\\end{tabular}",
             "\\caption{Guided instruction test for test-set contamination \\citep{golchin2024timetravel}: the model "
             "completes the first piece of a test claim once with an instruction naming the dataset and split (guided) "
             "and once without (general). ROUGE-L is computed on the generated second piece; $p$ tests guided $>$ "
             "general (10,000 bootstrap resamples); near-exact means ROUGE-L $\\geq 0.75$. $^{a}$For Llama, in the "
             "instruction format, which it refused almost entirely, so its test uses a pre-filled answer; some of its "
             "guided completions were then empty (scored 0), and the test without them is also given. Claude usually "
             "hedges before giving a guess, which is extracted and scored.}",
             "\\label{tab:contamination}", "\\end{table*}"]
    return "\n".join(lines)


# ---------------------------------------------------------------- Claude on the subsample
CLAUDE_ROWS = [("zs_claude", "Claude Opus 5.5 zero-shot"), ("zs", "Llama-3.1-8B zero-shot"),
               ("rag", "Claim retrieval + LLM"), ("clf_xlmr-advtrain", "XLM-R, adv.\\ training (B)"),
               (OURS_A, "\\method{} (A)"), (OURS_B, "\\method{} + \\air{} + aug.\\ (B)")]


def claude_table():
    Q = {(r["dataset"], r["method"]): r for r in load("claude_subsample")}
    K = {(r["dataset"], r["a"]): r for r in load("compare_claude")}
    nums["claude_subsample"] = {f"{d}/{m}": r for (d, m), r in Q.items()}
    nums["compare_claude"] = {f"{d}/{a}": r for (d, a), r in K.items()}
    body = []
    for m, name in CLAUDE_ROWS:
        cells = []
        for d in DATASETS:
            r = Q[d, m]
            assert r["n_claims"] == "100"
            c = K.get((d, m))
            for k, pk in (("clean", "clean_p"), ("attacked", "attacked_p")):
                cells.append(pc(r[k]) + ("$^{*}$" if c and sig(c[pk]) else ""))
            cells.append(f"${-100 * float(r['drop']):+.1f}$")
        body.append(name + " & " + " & ".join(cells) + " \\\\")
    n_un = {d: int(Q[d, "zs_claude"]["unparsed"]) for d in DATASETS}
    cap = ("Claude Opus 5.5 compared with selected detectors on the same stratified subsample of 100 test claims per "
           "dataset, clean and under the fifteen attacks (macro-F1, \\%). Claude was run on this subsample only, for "
           "cost reasons, so all detectors here are rescored on the same 100 claims; their numbers therefore differ "
           "slightly from Table~\\ref{tab:main}, which uses the full test sets. $^{*}$: significantly different from Claude "
           f"(paired bootstrap, $p<0.05$). Answers that are neither yes nor no ({sum(n_un.values())} of "
           f"{3 * 1600}) count as not flagged.")
    return table_star(body, 3, "Detector & " + " & ".join(["Clean & Att. & $\\Delta$"] * 3), cap, "tab:claude")


# ---------------------------------------------------------------- preservation (appendix)
def preservation_table():
    P = {(r["dataset"], r["attack"]): r for r in load("preservation")}
    overall = {}
    for d in DATASETS:  # pooled over the fifteen attacks (every test claim appears once per attack)
        rs = [P[d, a] for a in ATTACKS]
        assert len({r["n"] for r in rs}) == 1
        overall[d] = float(np.mean([float(r["preserved"]) for r in rs]))
        overall[d + "_false"] = float(np.mean([float(r["preserved_false"]) for r in rs]))
        overall[d + "_true"] = float(np.mean([float(r["preserved_true"]) for r in rs]))
    overall["all"] = float(np.average([overall[d] for d in DATASETS], weights=[int(P[d, ATTACKS[0]]["n"]) for d in DATASETS]))
    nums["preservation"] = overall
    short = {a: display(a).replace("TR_", "").replace(">", "$\\to$") for a in ATTACKS}
    lines = ["\\begin{table}[t]\\color{newtext}\\centering\\small", "\\begin{tabular}{lccc}", "\\toprule",
             "Attack & PUBH. & C-FEVER & Fin-Fact \\\\", "\\midrule"]
    for a in ATTACKS:
        lines.append(short[a] + " & " + " & ".join(pc(P[d, a]["preserved"]) for d in DATASETS) + " \\\\")
    lines += ["\\midrule", "All & " + " & ".join(pc(overall[d]) for d in DATASETS) + " \\\\",
              "\\quad false claims & " + " & ".join(pc(overall[d + "_false"]) for d in DATASETS) + " \\\\",
              "\\quad true claims & " + " & ".join(pc(overall[d + "_true"]) for d in DATASETS) + " \\\\",
              "\\bottomrule", "\\end{tabular}",
              "\\caption{Share (\\%) of attacked test texts that the Gemma-3-12B judge found to preserve the original "
              "claim.}", "\\label{tab:preservation}", "\\end{table}"]
    return "\n".join(lines)


SHORT = {"kotonya": "SciBERT + evidence", "arg": "ARG", "programfc": "ProgramFC", "minicheck": "MiniCheck",
         "rag": "Claim retrieval + LLM", "zs": "Llama-3.1-8B zero-shot", "zs_qwen": "Qwen2.5-7B zero-shot",
         "clf_xlmr": "XLM-R", OURS_FREE: "\\method{}, no training", OURS_A: "\\method{}", "kotonya-advtrain": "SciBERT + evidence, adv.",
         "arg-advtrain": "ARG, adv.", "clf_xlmr-advtrain": "XLM-R, adv.", OURS_B: "\\method{} + \\air{} + aug."}


def per_attack_table(d):
    """two stacked panels so that the table fits the text width at normal size: clean + single attacks, then chains"""
    short = {a: display(a).replace("TR_", "").replace(">", "$\\to$") for a in ATTACKS}
    single = ["CLEAN"] + [a for a in ATTACKS if ">" not in a]
    chains = [a for a in ATTACKS if ">" in a]
    ncol = len(single)
    assert len(single) + len(chains) == len(ATTACKS) + 1 and len(chains) <= ncol
    val = lambda r, a: pc(r["clean_f1"] if a == "CLEAN" else r[f"f1_{a}"])
    lines = ["\\begin{table*}[t]\\centering\\small\\setlength{\\tabcolsep}{4.5pt}",
             "\\begin{tabular}{l" + "c" * ncol + "}", "\\toprule"]
    code = {a: f"C{k + 1}" for k, a in enumerate(chains)}  # chain names are too wide for a column header
    for panel, atts in (("Clean and single attacks", single), ("Chained attacks", chains)):
        if panel != "Clean and single attacks":
            lines.append("\\midrule")
        heads = [("Clean" if a == "CLEAN" else code.get(a, short[a])) for a in atts] + [""] * (ncol - len(atts))
        lines += [f"\\multicolumn{{{ncol + 1}}}{{l}}{{\\emph{{{panel}}}}} \\\\",
                  "Detector & " + " & ".join(heads) + " \\\\", "\\midrule"]
        for i, (m, _) in enumerate(SETTING_A + SETTING_B):
            if i == len(SETTING_A):
                lines.append("\\cmidrule(lr){1-" + str(ncol + 1) + "}")
            r = S[d, m]
            cells = [val(r, a) for a in atts] + [""] * (ncol - len(atts))
            lines.append(SHORT[m] + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}",
              f"\\caption{{Macro-F1 (\\%) per attack on {DSN[d]}. Chained attacks: "
              + ", ".join(f"{code[a]} = {short[a]}" for a in chains)
              + ". Rows above the inner rule are trained on clean claims only (Setting~A), rows below also on "
              "transformed training claims (Setting~B).}",
              f"\\label{{tab:per_{d}}}", "\\end{table*}"]
    return "\n".join(lines)


def fit_width(tex):
    """shrink (never enlarge) the tabular to the text width (table*) or column width (table); needs adjustbox"""
    w = "\\textwidth" if "\\begin{table*}" in tex else "\\columnwidth"
    assert tex.count("\\begin{tabular}") == 1 and tex.count("\\end{tabular}") == 1
    return (tex.replace("\\begin{tabular}", f"\\begin{{adjustbox}}{{max width={w}}}\\begin{{tabular}}")
               .replace("\\end{tabular}", "\\end{tabular}\\end{adjustbox}"))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    out = {"main": main_table(), "ablation": ablation_table(), "plugin": plugin_table(),
           "seen_unseen": seen_unseen_table(), "adaptive": adaptive_table(), "contamination": contamination_table(),
           "preservation": preservation_table(), "claude": claude_table(), **{f"per_attack_{d}": per_attack_table(d) for d in DATASETS}}
    for name, tex in out.items():
        tex = fit_width(tex.replace("\\color{newtext}", ""))
        (OUT / f"{name}.tex").write_text(tex + "\n", encoding="utf-8")
    for (d, m), r in S.items():
        nums[f"{d}/{m}"] = {k: r[k] for k in ("seeds", "clean_f1", "attacked_f1", "drop_f1", "drop_lo", "drop_hi",
                                              "worst_f1", "asr_rel")}
    for name, idx in (("compare", CMP), ("compare_settingB", CMP_B)):
        for (d, a, b), r in idx.items():
            nums[f"{name}/{d}/{a}-{b}"] = {k: r[k] for k in r if k not in ("dataset", "a", "b")}
    json.dump(nums, open(ROOT / "paper/numbers.json", "w"), indent=1)
    print("tables written:", sorted(out))


if __name__ == "__main__":
    main()
