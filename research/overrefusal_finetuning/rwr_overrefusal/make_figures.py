#!/usr/bin/env python3
"""Build the paper figures from the result files, per FIGURE_SPECS.md.

Reads live JSON/CSV so figures regenerate when numbers change. Palette is the validated
categorical instance: Llama #2a78d6, Qwen #eb6834, aqua #1baf7a, neutral #8a8a85 —
checked with the palette validator (adjacent pairs and all-pairs for the scatter).
"""
import csv, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
csv.field_size_limit(sys.maxsize)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

LLAMA, QWEN, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a85"
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e6e3"
OUT = "figures"; os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.linewidth": 1.0, "axes.labelcolor": INK,
    "axes.labelsize": 10, "axes.titlesize": 11, "axes.titleweight": "semibold",
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "text.color": INK, "font.family": "DejaVu Sans", "legend.frameon": False,
    "legend.fontsize": 9, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
})

def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig); print(f"  wrote {OUT}/{name}.png/.pdf")

# ---------------------------------------------------------------- Fig 1
def fig1(shards=6):
    """Distribution of edit distance over ALL GENERATED rewrites — the attacker's behaviour.

    Denominator matters here and is easy to get wrong. Three different rates exist:
      (A) all generated rewrites          Llama 4.1%  Qwen 6.2%  <- plotted here
      (B) confirmed over-refusals, natural Llama 1.5%  Qwen 3.5%
      (C) the final analysis corpus        Llama 8.1%  Qwen 14.8%
    (C) is inflated by the low-bin power-up we performed *because* the bin was empty, so
    plotting it would use the consequence of the intervention to motivate the intervention.
    (A) is the honest statement about what the attacker does; (B) is quoted in the caption as
    the reason a power-up was needed at all."""
    from analyze_edit_distance import pair_metrics
    import glob
    def dists(d, cap_shards):
        fs = sorted(glob.glob(os.path.join(d, "*.json")))[:cap_shards]
        out = []
        for f in fs:
            for rec in json.load(open(f)):
                o = (rec.get("original") or "").strip()
                for w in rec.get("rewrites", []):
                    w = (w or "").strip()
                    if o and w:
                        out.append(pair_metrics(o, w)["wl_dist_content"])
        return np.array(out)

    L = dists("probe_or/results/gen_low_llama_logit", shards)
    Q = dists("probe_or/results/gen_low_qwen_probe", shards)
    CAP = 20
    NAT = {"Llama-3-8B-Instruct": 1.5, "Qwen3-32B": 3.5}
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 6.0), sharex=True)
    for ax, v, col, nm in ((axes[0], L, LLAMA, "Llama-3-8B-Instruct"),
                           (axes[1], Q, QWEN, "Qwen3-32B")):
        if not len(v):
            continue
        vv = np.clip(v, 0, CAP + 1)
        counts, _ = np.histogram(vv, bins=np.arange(0, CAP + 3) - 0.5)
        xs = np.arange(0, CAP + 2)
        ax.bar(xs, counts, width=0.78, color=col, linewidth=0, zorder=2)
        low = (v <= 2).mean() * 100
        ax.axvspan(-0.6, 2.5, color=INK, alpha=0.06, zorder=0)
        ax.axvline(2.5, color=INK2, lw=1.3, ls="--", zorder=3)
        mode = int(xs[np.argmax(counts)])
        ax.annotate(f"mode = {mode} edits", xy=(mode, counts.max()),
                    xytext=(mode + 3.0, counts.max() * 0.90), fontsize=9, color=INK2,
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
        ax.annotate(f"$D_c\\leq2$: {low:.1f}% of generated rewrites\n"
                    f"(only {NAT[nm]}% of confirmed over-refusals)",
                    xy=(1.1, counts.max() * 0.30), xytext=(5.4, counts.max() * 0.52),
                    fontsize=8.8, color=INK,
                    arrowprops=dict(arrowstyle="->", color=INK2, lw=0.9))
        ax.set_title(f"{nm}    (n = {len(v):,} generated rewrites)", loc="left")
        ax.set_ylabel("rewrites")
        ax.grid(axis="y", zorder=0); ax.set_axisbelow(True)
    ticks = list(range(0, CAP + 1, 4)) + [CAP + 1]
    axes[1].set_xticks(ticks)
    axes[1].set_xticklabels([str(t) for t in ticks[:-1]] + [f">{CAP}"])
    axes[1].set_xlabel("content-word edit distance $D_c$")
    axes[0].text(2.35, 0.97, "bin cut $\\tau=2$", transform=axes[0].get_xaxis_transform(),
                 fontsize=8.5, color=INK2, rotation=90, va="top", ha="right")
    fig.suptitle("The attacker rewords wholesale, so minimal-edit pairs are scarce",
                 x=0.012, ha="left", fontsize=13, fontweight="semibold", y=1.03)
    fig.text(0.012, 0.968, "Median normalised distance 0.92 \u2014 a typical rewrite changes "
             "about as many content words as the original contains. This follows from the "
             "reward, which gates on semantic\nsimilarity and charges nothing for lexical "
             "change. Populating the $D_c\\leq2$ bin therefore required generating 32,000 "
             "additional prompts.", fontsize=8.8, color=INK2, ha="left", va="top")
    fig.subplots_adjust(top=0.85, hspace=0.28)
    save(fig, "fig1_edit_distance_distribution")

# ---------------------------------------------------------------- Fig 3
NICE = {"d1_shared": "shared axis $d_1$", "atlas_rhat": "literature $\\hat{r}$"}

def fig3():
    """Grouped horizontal bars, not a scatter.

    A scatter of (over-refusal removed, harmful refusal lost) is the natural encoding, but
    Qwen has a 95pp outlier against a 0-21pp field, which either flattens the rest or needs a
    broken axis, and the near-coincident low-effect points collide when labelled. Bars put the
    direction names on the axis (collisions impossible), keep exact values readable, and let
    the outlier simply be a long bar. The signature of a good direction is a long blue bar
    beside a near-absent orange one."""
    def load(p):
        d = json.load(open(p))["scan"]
        pts = [(x["direction"], x["d_or"], x["d_harm"]) for x in d
               if not x["direction"].startswith("random")]
        rnd = [x["d_or"] for x in d if x["direction"].startswith("random")]
        return pts, (max(rnd) if rnd else 0.0)

    def pretty(n):
        if n in NICE:
            return NICE[n]
        n = n.replace("sym_", "")
        for pre in ("d1_", "d2_", "d3_", "d4_", "d5_", "d6_", "d7_", "d8_"):
            n = n.replace(pre, "")
        return n.replace("_", " ")

    panels = [("Llama-3-8B-Instruct", "probe_or/results/dirsearch_llama_sym.json")]
    if os.path.exists("probe_or/results/dirsearch_qwen_ownframes.json"):
        panels.append(("Qwen3-32B", "probe_or/results/dirsearch_qwen_ownframes.json"))

    fig, axes = plt.subplots(1, len(panels), figsize=(12.6, 5.6))
    if len(panels) == 1:
        axes = [axes]

    for ax, (nm, path) in zip(axes, panels):
        pts, nullmax = load(path)
        pts.sort(key=lambda t: t[1])                      # ascending, best at top after invert
        names = [pretty(n) for n, _, _ in pts]
        dor = np.array([o for _, o, _ in pts])
        dhm = np.array([h for _, _, h in pts])
        y = np.arange(len(pts)); hgt = 0.36

        ax.barh(y + hgt / 2 + 0.02, dor, height=hgt, color=LLAMA, linewidth=0,
                label="over-refusal removed")
        ax.barh(y - hgt / 2 - 0.02, dhm, height=hgt, color=QWEN, linewidth=0,
                label="harmful refusal lost")
        ax.axvspan(0, nullmax, color=GREY, alpha=0.18, zorder=0)
        ax.text(nullmax, len(pts) - 0.35, " random null", fontsize=8.2, color=INK2,
                va="center", ha="left")

        span = max(dor.max(), dhm.max())
        for yi, v in zip(y + hgt / 2 + 0.02, dor):
            ax.text(v + span * 0.015, yi, f"{v:.1f}", va="center", fontsize=8.4, color=INK)
        for yi, v in zip(y - hgt / 2 - 0.02, dhm):
            lab = f"{v:.1f}" + ("  \u2190 costs safety" if v >= 2 else "")
            ax.text(max(v, 0) + span * 0.015, yi, lab, va="center", fontsize=8.4,
                    color=INK if v < 2 else QWEN)
        ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9.5)
        ax.set_xlim(min(-2, dhm.min() * 1.2), span * 1.30)
        ax.axvline(0, color=INK2, lw=1.0)
        ax.set_xlabel("percentage points")
        ax.set_title(nm, loc="left", pad=8)
        ax.grid(axis="x", zorder=0); ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)

    axes[0].legend(loc="lower right", fontsize=9)
    fig.suptitle("A single direction removes over-refusal without costing safety",
                 x=0.012, ha="left", fontsize=13, fontweight="semibold", y=1.04)
    fig.text(0.012, 0.968, "Each direction ablated alone, at every layer. Measured on held-out "
             "originals: 400 confirmed over-refusals and 200 AdvBench harmful prompts. "
             "A good direction has a long blue bar and no orange one.",
             fontsize=8.8, color=INK2, ha="left", va="top")
    fig.subplots_adjust(top=0.84, wspace=0.42)
    save(fig, "fig3_causal_bars")

# ---------------------------------------------------------------- Fig 5
def fig5():
    e = json.load(open("probe_or/results/external_bench.json"))["rates"]
    g = json.load(open("probe_or/results/gcg_transfer.json"))["rates"]
    rnd_g = [v for k, v in g.items()
             if k.startswith("gcg_rewrites__random") and not k.endswith("degen")][0]
    rows = [
        ("our rewrites",  "RWR attacker (ours)",                     74.2, 41.0, 74.0),
        ("GCG corpus",    "different attack method, disjoint vocab", g["gcg_rewrites__baseline"],
                                                                     g["gcg_rewrites__ours_k1"], rnd_g),
        ("XSTest safe",   "hand-written, n = 250",                   e["xstest_safe__baseline"],
                                                                     e["xstest_safe__ablate_d4"],
                                                                     e["xstest_safe__ablate_random"]),
        ("OR-Bench Hard", "auto-generated from toxic seeds, n = 400", e["orbench_hard__baseline"],
                                                                     e["orbench_hard__ablate_d4"],
                                                                     e["orbench_hard__ablate_random"]),
    ]
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ys = np.arange(len(rows))[::-1].astype(float)
    for y, (nm, how, b, a, r) in zip(ys, rows):
        ax.plot([a, b], [y, y], color=LLAMA, lw=3.0, solid_capstyle="round", zorder=2)
        ax.scatter([b], [y], s=110, color=LLAMA, zorder=4, edgecolors=SURF, linewidths=1.8)
        ax.scatter([a], [y], s=110, color=LLAMA, zorder=4, marker="D",
                   edgecolors=SURF, linewidths=1.8)
        # random control: a tick, so coinciding with baseline reads as "no change" not overplot
        ax.plot([r, r], [y - 0.17, y + 0.17], color=INK2, lw=2.0, zorder=5)
        rel = (b - a) / b * 100 if b else 0
        ax.annotate(f"\u2212{b-a:.1f} pp   ({rel:.0f}% relative)",
                    xy=(max(b, r) + 2.2, y + 0.02), fontsize=9, color=INK, va="center")
        ax.annotate(how, xy=(0.6, y - 0.27), fontsize=8.3, color=INK2, va="center")
    ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows], fontsize=10.5)
    ax.set_ylim(-0.62, len(rows) - 0.30)
    ax.set_xlim(-1, 116); ax.set_xlabel("refusal rate (%)")
    ax.grid(axis="x", zorder=0); ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    handles = [Line2D([], [], marker="o", ls="", ms=9.5, color=LLAMA, label="baseline"),
               Line2D([], [], marker="D", ls="", ms=8.5, color=LLAMA,
                      label="after ablating the direction"),
               Line2D([], [], marker="|", ls="", ms=11, mew=2.2, color=INK2,
                      label="random direction (control)")]
    ax.legend(handles=handles, loc="upper right", fontsize=8.8,
              bbox_to_anchor=(1.005, 1.16), ncol=3, columnspacing=1.2, handletextpad=0.5)
    fig.suptitle("The direction generalises to corpora built by other methods",
                 x=0.012, ha="left", fontsize=13, fontweight="semibold", y=1.07)
    fig.text(0.012, 0.995, "Direction fitted only on our own attacker's rewrites (Llama). "
             "The random control lands on the baseline in every corpus. GCG rates measure "
             "refusal, not judge-confirmed over-refusal.",
             fontsize=8.8, color=INK2, ha="left", va="top")
    fig.subplots_adjust(top=0.80)
    save(fig, "fig5_generalisation")

if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "3", "5"]
    if "1" in which: print("Fig 1:"); fig1()
    if "3" in which: print("Fig 3:"); fig3()
    if "5" in which: print("Fig 5:"); fig5()
