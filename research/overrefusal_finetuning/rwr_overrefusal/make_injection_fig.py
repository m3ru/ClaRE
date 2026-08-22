#!/usr/bin/env python3
"""Figure for the causal injection result: refusal change per cluster arm, both models.

Colour encodes the target model here, so the house LLAMA/QWEN hues carry their usual
meaning. Error bars are the paired bootstrap 95% intervals from score_injections.py, which
resamples prompts rather than arm-rows because the design is paired.

Run with a matplotlib python (module load scipy-stack/2024b): python make_injection_fig.py
"""
import json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e6e3"
LLAMA, QWEN = "#2a78d6", "#eb6834"
OUT = "figures"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.linewidth": 1.0, "axes.labelcolor": INK,
    "axes.labelsize": 10, "axes.titlesize": 11, "axes.titleweight": "semibold",
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "text.color": INK, "font.family": "DejaVu Sans", "legend.frameon": False,
    "legend.fontsize": 9, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
})

NICE = {"chars": "chars\n(asterisks, slashes)", "negation": "negation\n(no / not / NOT PLEASE)",
        "touchy": "touchy subject\n(AI self, social, privacy)",
        "harmful_phrase": "harmful-sounding phrase\n(not actually harmful)",
        "action": "action words", "chars+harmful_phrase": "chars + harmful phrase",
        "chars+touchy": "chars + touchy", "chars+negation": "chars + negation",
        "all_five": "all five combined"}
ORDER = ["chars", "action", "negation", "harmful_phrase", "touchy",
         "chars+negation", "chars+touchy", "chars+harmful_phrase", "all_five"]

d = {t: json.load(open(f"probe_or/results/injections_{t}_clean_summary.json"))
     for t in ("llama", "qwen")}
y = list(range(len(ORDER)))[::-1]
fig, ax = plt.subplots(figsize=(9.2, 5.6))
h = 0.36
for off, (tag, col, lab) in enumerate((("llama", LLAMA, "Llama-3-8B (baseline 0.7%)"),
                                       ("qwen", QWEN, "Qwen3-32B (baseline 0.7%)"))):
    pos, val, err = [], [], [[], []]
    for i, k in enumerate(ORDER):
        r = d[tag].get(k)
        if not r:
            continue
        p = y[i] + (h / 2 + 0.01 if off == 0 else -h / 2 - 0.01)
        pos.append(p); val.append(r["delta"])
        err[0].append(max(r["delta"] - r["lo"], 0)); err[1].append(max(r["hi"] - r["delta"], 0))
    ax.barh(pos, val, height=h, color=col, linewidth=0, zorder=2, label=lab)
    ax.errorbar(val, pos, xerr=err, fmt="none", ecolor=INK2, elinewidth=1.1,
                capsize=2.5, zorder=3)

ax.axvline(0, color=INK2, lw=1.1, zorder=1)
ax.set_yticks(y)
ax.set_yticklabels([NICE[k] for k in ORDER], fontsize=8.8)
ax.set_xlabel("Change in refusal rate on benign prompts (percentage points)")
ax.set_title("Injecting each cluster into 300 unseen benign prompts (still-benign refusals only)",
             loc="left", pad=10)
ax.grid(axis="x", zorder=0)
ax.set_axisbelow(True)
ax.set_xlim(-2.5, 22)
ax.legend(loc="upper right")
fig.tight_layout()
os.makedirs(OUT, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}/fig_injection_effect.{ext}", dpi=200, bbox_inches="tight")
print("wrote figures/fig_injection_effect.png/.pdf")
