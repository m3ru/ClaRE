#!/usr/bin/env python3
"""Two figures for GCG_VOCAB_FINDINGS.md.

  fig_gcg_vocab_rates  two panels sharing a corpus axis: how often each corpus INSERTS a
                       prohibition word, and how often it inserts a danger word. Two panels
                       rather than one grouped chart because the two rates run over very
                       different ranges and a shared x would flatten the smaller one.
  fig_gcg_vocab_z      words separating judge-confirmed from judge-rejected or_loose
                       rewrites, by weighted log-odds z.

Palette is the house one from make_figures.py, which is already validator-checked. Both
figures encode with a chromatic/achromatic pair (orange vs grey), the safest case for
colour-vision deficiency, and neither relies on colour alone: every bar is direct-labelled
and every row is named on the axis.

Run with a matplotlib-bearing python (module load scipy-stack/2024b):
    python make_gcg_vocab_figs.py
"""
import json, os, sys
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_gcg_vocab as A
from compare_gcg_vs_rwr import logodds

# House palette (make_figures.py)
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e6e3"
FLAG, PLAIN = "#eb6834", "#8a8a85"
OUT = "figures"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.linewidth": 1.0, "axes.labelcolor": INK,
    "axes.labelsize": 10, "axes.titlesize": 10.5, "axes.titleweight": "semibold",
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "text.color": INK, "font.family": "DejaVu Sans", "legend.frameon": False,
    "legend.fontsize": 9, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
})


def corpora():
    rows = json.load(open("incoming/qwen_gcg_all.json"))["rows"]
    lg = json.load(open("incoming/sonnet_filtered_strict.json"))["rows"]
    arm = lambda n: [r for r in rows if r["arm"] == n]
    return [
        ("Qwen GCG or_*", arm("or_loose") + arm("or_strict"), "GCG"),
        ("Qwen GCG jb_*", arm("jb_loose") + arm("jb_strict"), "GCG"),
        ("Llama GCG", lg, "GCG"),
        ("RWR llamaAtt\n(confirmed-OR)", A.load_rwr_llama(), "RWR"),
        ("RWR baseQwenAtt\n(unjudged)", A.load_rwr_qwen(), "RWR"),
    ]


def fig_rates():
    data = [(nm, A.cue_rates(rs), fam) for nm, rs, fam in corpora()]
    data.reverse()                      # barh draws bottom-up
    names = [d[0] for d in data]
    cols = [FLAG if d[2] == "GCG" else PLAIN for d in data]
    y = range(len(data))

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5), sharey=True)
    for ax, key, title in ((axes[0], "prohibition", "Inserts a prohibition word\n(no, not, never, cannot, ignore …)"),
                           (axes[1], "danger", "Inserts a danger word\n(exploit, covert, infiltrate, target …)")):
        v = [d[1][key] for d in data]
        ax.barh(list(y), v, height=0.62, color=cols, linewidth=0, zorder=2)
        ax.set_title(title, loc="left", pad=8)
        ax.set_xlabel("% of rewrites")
        ax.set_xlim(0, max(v) * 1.28)
        ax.grid(axis="x", zorder=0)
        ax.set_axisbelow(True)
        for yi, vv in zip(y, v):
            ax.text(vv + max(v) * 0.02, yi, f"{vv:.1f}%", va="center",
                    fontsize=8.8, color=INK)
    axes[0].set_yticks(list(y))
    axes[0].set_yticklabels(names)
    handles = [plt.Rectangle((0, 0), 1, 1, color=FLAG), plt.Rectangle((0, 0), 1, 1, color=PLAIN)]
    axes[1].legend(handles, ["GCG corpora", "RWR corpora"], loc="lower right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/fig_gcg_vocab_rates.{ext}", dpi=200, bbox_inches="tight")
    print("  wrote fig_gcg_vocab_rates")
    return {nm: d for nm, d, _ in [(a, b, c) for a, b, c in data]}


def fig_z(top=12):
    rows = json.load(open("incoming/qwen_gcg_all.json"))["rows"]
    isconf = lambda r: r.get("judge_label") == "REFUSE" and r.get("judge_justified") == "NO"
    conf = [r for r in rows if isconf(r)]
    rej = [r for r in rows if r.get("judge_label") and not isconf(r)]
    ta, da, _ = A.profile(conf, mode="added")
    tb, db, _ = A.profile(rej, mode="added")
    z = logodds(ta, tb)
    cand = [(w, s) for w, s in z.items() if s > 0 and len(da[w]) >= 5]
    cand.sort(key=lambda x: -x[1])
    cand = cand[:top][::-1]

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    y = range(len(cand))
    cols = [FLAG if w in A.PROHIBITION else PLAIN for w, _ in cand]
    ax.barh(list(y), [s for _, s in cand], height=0.66, color=cols, linewidth=0, zorder=2)
    ax.set_yticks(list(y))
    ax.set_yticklabels([w for w, _ in cand])
    ax.set_xlabel("weighted log-odds z  (higher = more specific to judge-confirmed)")
    ax.set_title("Words separating judge-confirmed from judge-rejected or_loose rewrites",
                 loc="left", pad=8)
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)
    mx = max(s for _, s in cand)
    ax.set_xlim(0, mx * 1.22)
    for yi, (w, s) in zip(y, cand):
        ax.text(s + mx * 0.015, yi, f"{s:.2f}   {ta.get(w,0)} vs {tb.get(w,0)}",
                va="center", fontsize=8.4, color=INK)
    handles = [plt.Rectangle((0, 0), 1, 1, color=FLAG), plt.Rectangle((0, 0), 1, 1, color=PLAIN)]
    ax.legend(handles, ["prohibition word", "other"], loc="lower right")
    fig.text(0.005, -0.03, "Labels show z, then raw count in confirmed vs rejected. "
             f"n = {len(conf)} confirmed, {len(rej)} rejected.",
             fontsize=8.2, color=INK2, ha="left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/fig_gcg_vocab_z.{ext}", dpi=200, bbox_inches="tight")
    print("  wrote fig_gcg_vocab_z")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_rates()
    fig_z()
