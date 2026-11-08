"""Paper figures, built only from results/tables/summary.csv (aggregate.py output).

fig_clean_vs_attacked.pdf  dumbbell: clean vs attacked macro-F1 per detector, one panel per dataset
fig_per_attack.pdf         macro-F1 per attack (mean over the three datasets) for five detectors,
                           attacks grouped as seen in Setting B training / unseen single / chained
Colours: reference categorical palette (validated with the dataviz validator; contrast WARN relieved by
direct labels and distinct marker shapes, so identity never relies on colour alone).
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import ROOT, DATASETS
from seen_unseen import SEEN, UNSEEN_SINGLE, UNSEEN_CHAIN

OUT = ROOT / "paper/figures"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
DSN = {"pubhealth": "PUBHEALTH", "climatefever": "CLIMATE-FEVER", "finfact": "Fin-Fact"}
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.6})

S = {(r["dataset"], r["method"]): r for r in csv.DictReader(open(ROOT / "results/tables/summary.csv"))}
f1 = lambda d, m, k: 100 * float(S[d, m][k])

ROWS = [("kotonya", "SciBERT + evidence", "A"), ("arg", "ARG", "A"), ("programfc", "ProgramFC", "A"),
        ("minicheck", "MiniCheck", "A"), ("rag", "Claim retrieval + LLM", "A"), ("zs", "Llama-3.1-8B zero-shot", "A"),
        ("zs_qwen", "Qwen2.5-7B zero-shot", "A"), ("clf_xlmr", "XLM-R", "A"), ("ours_vtrained", "UNWRAP", "A"),
        ("kotonya-advtrain", "SciBERT + evidence", "B"), ("arg-advtrain", "ARG", "B"),
        ("clf_xlmr-advtrain", "XLM-R", "B"), ("ours_air_vtrained_aug", "UNWRAP + AIR + aug.", "B")]
OURS = {"ours_vtrained", "ours_air_vtrained_aug"}


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def dumbbell():
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 3.0), sharey=True)
    y = {}
    pos = 0
    for i, (m, _, st) in enumerate(ROWS):
        if i and ROWS[i - 1][2] != st:
            pos += 1.5  # gap between the two settings (room for its label)
        y[m] = pos
        pos += 1
    for ax, d in zip(axes, DATASETS):
        style(ax)
        for m, _, _ in ROWS:
            c, a = f1(d, m, "clean_f1"), f1(d, m, "attacked_f1")
            ax.plot([c, a], [y[m]] * 2, color=GRID if m not in OURS else "#b7d3f6", lw=2.2, solid_capstyle="round",
                    zorder=1)
            ax.scatter([c], [y[m]], s=22, facecolor=SURF, edgecolor=INK2, lw=1.1, zorder=3)
            ax.scatter([a], [y[m]], s=24, facecolor=BLUE, edgecolor=SURF, lw=0.8, zorder=4)
            # attacked score as a number, so that small differences (and overlapping markers) stay readable
            ax.text(99, y[m], f"{a:.1f}", ha="right", va="center", fontsize=6.8, color=INK,
                    fontweight="bold" if m in OURS else "normal")
        ax.text(99, -1.05, "Att.", ha="right", va="center", fontsize=6.8, color=INK2)
        ax.set_xlim(20, 100)
        ax.set_xticks([20, 40, 60, 80])
        ax.set_title(DSN[d], fontsize=8.5, color=INK)
        ax.set_xlabel("Macro-F1 (%)")
    ax0 = axes[0]
    ax0.set_yticks([y[m] for m, _, _ in ROWS])
    ax0.set_yticklabels([n for _, n, _ in ROWS])
    for lab, m in zip(ax0.get_yticklabels(), ROWS):
        if m[0] in OURS:
            lab.set_fontweight("bold")
    ax0.invert_yaxis()
    yb = (y["ours_vtrained"] + y["kotonya-advtrain"]) / 2
    for ax in axes:
        ax.axhline(yb, color=INK2, lw=0.5, ls=(0, (2, 2)))
    ax0.text(-0.02, (y["kotonya"] - 0.9), "Setting A", transform=ax0.get_yaxis_transform(), ha="right",
             va="bottom", fontsize=7.5, style="italic", color=INK2)
    ax0.text(-0.02, (y["kotonya-advtrain"] - 0.9), "Setting B", transform=ax0.get_yaxis_transform(), ha="right",
             va="bottom", fontsize=7.5, style="italic", color=INK2)
    h1 = ax0.scatter([], [], s=22, facecolor=SURF, edgecolor=INK2, lw=1.1, label="Clean")
    h2 = ax0.scatter([], [], s=24, facecolor=BLUE, edgecolor=SURF, lw=0.8, label="Attacked (mean of 15 attacks)")
    fig.legend(handles=[h1, h2], loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.6, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=0.8)
    fig.savefig(OUT / "fig_clean_vs_attacked.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_clean_vs_attacked.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def per_attack():
    # same-setting comparison: every solid line is trained on the same transformed claims (Setting B); UNWRAP
    # trained on clean claims only (Setting A) is a thin dashed reference in the same hue
    series = [("ours_air_vtrained_aug", "UNWRAP (B)", BLUE, "o", "-"),
              ("clf_xlmr-advtrain", "XLM-R adv. (B)", YELLOW, "D", "-"),
              ("arg-advtrain", "ARG adv. (B)", AQUA, "^", "-"),
              ("kotonya-advtrain", "SciBERT+ev. adv. (B)", ORANGE, "s", "-"),
              ("ours_vtrained", "UNWRAP, no attacks (A)", BLUE, "", (0, (3, 2)))]
    groups = [("Clean", ["CLEAN"]), ("Seen in Setting B", SEEN), ("Unseen single", UNSEEN_SINGLE),
              ("Chained (unseen)", UNSEEN_CHAIN)]
    short = lambda a: {"CLEAN": "clean"}.get(a, a.replace("TR_", "").replace(">", "→"))
    xs, labels, spans, x = {}, [], [], 0
    for gname, atts in groups:
        start = x
        for a in atts:
            xs[a] = x; labels.append(short(a)); x += 1
        spans.append((gname, start, x - 1))
        x += 0.6
    fig, ax = plt.subplots(figsize=(6.3, 2.35))
    style(ax)
    ax.grid(axis="x", visible=False); ax.grid(axis="y", color=GRID, lw=0.5)
    for k, (g, s0, s1) in enumerate(spans):
        if k % 2 == 1:
            ax.axvspan(s0 - 0.4, s1 + 0.4, color="#f4f3f0", zorder=0, lw=0)
        ax.text((s0 + s1) / 2, 1.02, g, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=7.5,
                color=INK2)
    mean = lambda m, a: np.mean([100 * float(S[d, m]["clean_f1" if a == "CLEAN" else f"f1_{a}"]) for d in DATASETS])
    for k, (m, name, col, mk, ls) in enumerate(series):
        for _, atts in groups:
            dx = (k - (len(series) - 1) / 2) * 0.12 if atts == ["CLEAN"] else 0.0  # clean scores overlap: dodge
            ax.plot([xs[a] + dx for a in atts], [mean(m, a) for a in atts], color=col, lw=1.4 if ls == "-" else 1.0,
                    ls=ls, marker=mk or ("." if atts == ["CLEAN"] else None), ms=4.2, mec=SURF, mew=0.6,
                    zorder=3 if ls == "-" else 2)
    # direct labels at the right end, spread apart so they never overlap (min gap 4.5 F1 points)
    last = UNSEEN_CHAIN[-1]
    ends = sorted(((mean(m, last), name, m) for m, name, _, _, _ in series), reverse=True)
    ys = []
    for v, _, _ in ends:
        ys.append(min(v, ys[-1] - 4.5) if ys else v)
    for (v, name, _), yl in zip(ends, ys):
        ax.annotate(name, (xs[last], v), xytext=(xs[last] + 0.7, yl), textcoords="data", va="center", fontsize=7,
                    color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.4, shrinkA=0, shrinkB=3))
    handles = [plt.Line2D([], [], color=col, lw=1.4 if ls == "-" else 1.0, ls=ls, marker=mk or None, ms=4.2, mec=SURF,
                          mew=0.6, label=name) for _, name, col, mk, ls in series]
    fig.legend(handles=handles, loc="upper center", ncol=len(series), frameon=False, fontsize=7,
               bbox_to_anchor=(0.5, 1.0), handlelength=2.2, columnspacing=1.2)
    ax.set_xticks(list(xs.values()))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_xlim(-0.6, max(xs.values()) + 4.2)
    ax.set_ylim(30, 85)
    ax.set_ylabel("Macro-F1 (%), mean of 3 datasets")
    ax.spines["left"].set_visible(True)
    fig.tight_layout(rect=(0, 0, 1, 0.9))  # leave the top row for the legend, above the group titles
    fig.savefig(OUT / "fig_per_attack.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_per_attack.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    dumbbell()
    per_attack()
    print("figures written to", OUT)
