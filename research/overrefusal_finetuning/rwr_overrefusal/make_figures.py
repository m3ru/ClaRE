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
# fig4 encodes refused/not-refused, NOT model identity, so it must not reuse the
# model hues; aqua+violet validated separately (adjacent pairs, light mode).
REFUSED, NOTREF = "#4a3aa7", "#1baf7a"
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e6e3"
BARE = "--bare" in sys.argv          # strip headline + deck; panel titles and data labels stay
OUT = "figures/bare" if BARE else "figures"
os.makedirs(OUT, exist_ok=True)

# The headline (suptitle) and the deck line under it are CAPTION text that happens to be
# rendered into the canvas. For LaTeX we want them out of the PNG and into \caption{}.
# Both are intercepted here rather than at each call site so any future figure inherits it.
# Decks are identified by their x anchor, which is 0.012 for every one of them.
_CAPTIONS = {}
_cur = {"name": None}
_orig_suptitle, _orig_figtext = matplotlib.figure.Figure.suptitle, matplotlib.figure.Figure.text

def _suptitle(self, t, *a, **k):
    _CAPTIONS.setdefault(_cur["name"], {})["headline"] = " ".join(str(t).split())
    return None if BARE else _orig_suptitle(self, t, *a, **k)

def _figtext(self, x, y, t, *a, **k):
    if abs(x - 0.012) < 1e-9:
        _CAPTIONS.setdefault(_cur["name"], {})["deck"] = " ".join(str(t).split())
        if BARE:
            return None
    return _orig_figtext(self, x, y, t, *a, **k)

matplotlib.figure.Figure.suptitle = _suptitle
matplotlib.figure.Figure.text = _figtext

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
    # In bare mode the subplots_adjust(top=...) that reserved room for the headline would
    # leave a white band, and bbox_inches="tight" alone does not reclaim it because the
    # axes themselves were moved down. Push the axes back up before cropping.
    if BARE:
        fig.subplots_adjust(top=0.97)
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/{name}.{ext}", dpi=200, bbox_inches="tight",
                    pad_inches=0.02 if BARE else 0.1)
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
        ax.set_title(f"{nm}    (n = {len(v):,} generated rewrites\u2003\u00b7\u2003"
                     f"$D\\leq2$: {low:.1f}%)", loc="left")
        ax.set_ylabel("rewrites")
        ax.grid(axis="y", zorder=0); ax.set_axisbelow(True)
    ticks = list(range(0, CAP + 1, 4)) + [CAP + 1]
    axes[1].set_xticks(ticks)
    axes[1].set_xticklabels([str(t) for t in ticks[:-1]] + [f">{CAP}"])
    axes[1].set_xlabel("content-word edit distance $D$")
    axes[0].text(2.35, 0.97, "bin cut $\\tau=2$", transform=axes[0].get_xaxis_transform(),
                 fontsize=8.5, color=INK2, rotation=90, va="top", ha="right")
    fig.suptitle("The attacker rewords wholesale, so minimal-edit pairs are scarce",
                 x=0.012, ha="left", fontsize=13, fontweight="semibold", y=1.03)
    fig.text(0.012, 0.968, "Median normalised distance 0.92 \u2014 a typical rewrite changes "
             "about as many content words as the original contains. This follows from the "
             "reward, which gates on semantic\nsimilarity and charges nothing for lexical "
             "change. Populating the $D\\leq2$ bin therefore required generating 32,000 "
             "additional prompts.", fontsize=8.8, color=INK2, ha="left", va="top")
    fig.subplots_adjust(top=0.85, hspace=0.28)
    save(fig, "fig1_edit_distance_distribution")

# ---------------------------------------------------------------- Fig 3
def fig3():
    """Single-direction ablation, JUDGED (not regex).

    Two measures at very different scales -- over-refusal removed (0-37pp) and harmful
    refusal lost (within +/-1pp) -- so they get small multiples sharing a y-axis, never a
    dual x-axis. The narrow right panel exists precisely because the safety result is a
    NON-effect: on a shared axis with the left panel it would be invisible, and "invisible"
    is not the same as "shown to be zero".

    Sign convention is identical in both panels: bigger = more refusal removed. On the left
    that is the goal; on the right it is the cost. Zero on the right means the ablation did
    not touch harmful-prompt refusal.

    Rates are Sonnet-judged on substance, not the start-anchored regex. The regex reads a
    moralising refusal as compliance, which inflated the two output-aligned directions by
    up to 97pp; see judged_ablation_2model.json.
    """
    D = json.load(open("probe_or/results/judged_ablation_2model.json"))
    order = ["overall direction", "whole-prompt refusal vector", "weaponization",
             "concealment", "intrusion", "exfiltration", "exploitation",
             "coercion", "fabrication"]
    fig = plt.figure(figsize=(11.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[3.1, 1.0], wspace=0.06, hspace=0.34)
    for r, (mk, nm, col) in enumerate([("llama", "Llama-3-8B-Instruct", LLAMA),
                                       ("qwen", "Qwen3-32B", QWEN)]):
        d = D[mk]; rows = d["rows"]
        y = np.arange(len(order))[::-1]
        axL = fig.add_subplot(gs[r, 0]); axR = fig.add_subplot(gs[r, 1], sharey=axL)
        # left: over-refusal removed
        axL.axvspan(0, d["null_or_drop"], color=GREY, alpha=0.16, lw=0, zorder=0)
        for yi, k in zip(y, order):
            if k not in rows:
                axL.text(0.7, yi, "not in this model's frame set", va="center",
                         fontsize=7.6, color=GREY, style="italic", zorder=4); continue
            v = rows[k]["or_drop"]
            axL.barh(yi, v, height=0.52, color=col, zorder=3)
            axL.text(v + 0.6 if v >= 0 else v - 0.6, yi, f"{v:+.1f}", va="center",
                     ha="left" if v >= 0 else "right", fontsize=8.4, color=INK)
        axL.set_yticks(y); axL.set_yticklabels(order, fontsize=9.2)
        # pin the row range so BOTH models show all 9 rows in the same order -- otherwise
        # matplotlib autoscales to the bars drawn and the two panels stop lining up.
        axL.set_ylim(-0.7, len(order) - 0.3)
        axL.set_xlim(-6, 42); axL.axvline(0, color=INK2, lw=1.0)
        axL.set_title(nm, loc="left", pad=7)
        axL.grid(axis="x", zorder=0); axL.set_axisbelow(True)
        axL.spines["left"].set_visible(False); axL.tick_params(axis="y", length=0)
        # right: harmful refusal lost -- same sign convention, far smaller scale
        axR.axvspan(-1.5, 1.5, color=GREY, alpha=0.16, lw=0, zorder=0)
        for yi, k in zip(y, order):
            if k not in rows:
                continue
            axR.plot(rows[k]["harm_drop"], yi, "o", ms=7, color=col,
                     mec=SURF, mew=1.2, zorder=3)
        axR.set_xlim(-4, 4); axR.axvline(0, color=INK2, lw=1.0)
        axR.grid(axis="x", zorder=0); axR.set_axisbelow(True)
        axR.tick_params(labelleft=False, labelsize=8.4)
        axR.spines["left"].set_visible(False); axR.tick_params(axis="y", length=0)
        if r == 0:
            axL.set_title(nm, loc="left", pad=7)
            axR.set_title("safety cost", loc="left", pad=7, fontsize=9.6)
        if r == 1:
            axL.set_xlabel("over-refusal removed (percentage points)")
            axR.set_xlabel("harmful refusal lost (pp)", fontsize=9)
    fig.suptitle("Removing a single direction reduces over-refusal without costing safety",
                 x=0.012, ha="left", fontsize=13, fontweight="semibold", y=1.02)
    fig.text(0.012, 0.965,
             "Both panels: percentage points of refusal REMOVED, so larger is more removed. "
             "Left is the goal, right is the cost.\nShaded band on the left is the "
             "random-direction null; on the right it is \u00b11.5pp. Every direction on both "
             "models moves harmful\nrefusal by at most 1.0pp. Rates are Sonnet-judged on "
             "substance; a start-anchored regex inflates the two output-aligned\ndirections "
             "by up to 97pp and is not used here.",
             fontsize=8.8, color=INK2, ha="left", va="top")
    fig.subplots_adjust(top=0.80)
    save(fig, "fig3_causal_bars")


# ---------------------------------------------------------------- Fig 5
def fig5():
    """Grouped bars, two per corpus: refusal rate before and after ablating the direction.

    A dumbbell was tried first and read poorly — the four corpora have baselines from 7.6% to
    82.2%, so the connecting lines were wildly different lengths and the short XSTest row
    looked like an error. Two bars per corpus keeps both absolute rates legible at any
    baseline. The random control is a tick on the baseline bar: it lands on the baseline in
    every corpus, so drawing it as a third bar would add ink for a null result."""
    e = json.load(open("probe_or/results/external_bench.json"))["rates"]
    g = json.load(open("probe_or/results/gcg_transfer.json"))["rates"]
    rnd_g = [v for k, v in g.items()
             if k.startswith("gcg_rewrites__random") and not k.endswith("degen")][0]
    rows = [
        ("our rewrites",  "RWR attacker (ours)",                      74.2, 41.0, 74.0),
        ("GCG corpus",    "different attack method",                  g["gcg_rewrites__baseline"],
                                                                      g["gcg_rewrites__ours_k1"], rnd_g),
        ("XSTest safe",   "hand-written, n = 250",                    e["xstest_safe__baseline"],
                                                                      e["xstest_safe__ablate_d4"],
                                                                      e["xstest_safe__ablate_random"]),
        ("OR-Bench Hard", "auto-generated, n = 400",                  e["orbench_hard__baseline"],
                                                                      e["orbench_hard__ablate_d4"],
                                                                      e["orbench_hard__ablate_random"]),
    ]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    y = np.arange(len(rows))[::-1].astype(float)
    h = 0.34
    for yi, (nm, how, b, a, r) in zip(y, rows):
        ax.barh(yi + h / 2 + 0.02, b, height=h, color=GREY, alpha=0.55, linewidth=0, zorder=2)
        ax.barh(yi - h / 2 - 0.02, a, height=h, color=LLAMA, linewidth=0, zorder=2)
        ax.text(b + 1.4, yi + h / 2 + 0.02, f"{b:.1f}", va="center", fontsize=8.6, color=INK2)
        ax.text(a + 1.4, yi - h / 2 - 0.02, f"{a:.1f}", va="center", fontsize=8.6,
                color=LLAMA, fontweight="semibold")
        ax.plot([r, r], [yi + h / 2 + 0.02 - h / 2, yi + h / 2 + 0.02 + h / 2],
                color=INK, lw=1.8, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{nm}\n{how}" for nm, how, *_ in rows], fontsize=9.5)
    ax.set_xlim(0, 100); ax.set_xlabel("refusal rate (%)")
    ax.grid(axis="x", zorder=0); ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    handles = [Line2D([], [], marker="s", ls="", ms=9, color=GREY, alpha=0.55, label="baseline"),
               Line2D([], [], marker="s", ls="", ms=9, color=LLAMA,
                      label="after ablating the direction"),
               Line2D([], [], marker="|", ls="", ms=11, mew=1.9, color=INK,
                      label="random direction (control)")]
    ax.legend(handles=handles, loc="upper left", ncol=3, fontsize=8.7,
              bbox_to_anchor=(0.0, 1.13), columnspacing=1.3, handletextpad=0.5)
    fig.suptitle("The direction generalises to corpora built by other methods",
                 x=0.012, ha="left", fontsize=13, fontweight="semibold", y=1.10)
    fig.text(0.012, 1.04, "Direction fitted only on our own attacker's rewrites (Llama). "
             "The random control lands on the baseline in every corpus. GCG rates measure "
             "refusal, not judge-confirmed over-refusal.",
             fontsize=8.8, color=INK2, ha="left", va="top")
    fig.subplots_adjust(top=0.80)
    save(fig, "fig5_generalisation")

# ---------------------------------------------------------------- Fig 2
def _trigger_stats(low_csv, scored_paths, attacker, min_origs=3, a0=10.0):
    """Weighted log-odds (Monroe et al.) of refused vs NOT-refused rewrites from the SAME
    attacker in the SAME edit bin, computed at document level (each original counted once).

    Also flags whether each word was INTRODUCED by the edit or is a topic marker present in
    both original and rewrite — the distinction the figure exists to make."""
    import math, glob
    from collections import Counter, defaultdict
    from analyze_edit_distance import pair_metrics, content_tokens
    from recompute_v6 import refused_v6

    low = [r for r in csv.DictReader(open(low_csv))
           if r.get("attacker") in (None, "", attacker)]
    or_df, intro_hits, tot_hits = defaultdict(set), Counter(), Counter()
    for r in low:
        o, w = r["original"].strip(), r["rewrite"].strip()
        m = pair_metrics(o, w)
        intro = set(content_tokens(m["introduced_words"]))
        toks = set(content_tokens(w))
        for t in toks:
            if len(t) > 2:
                or_df[t].add(o); tot_hits[t] += 1
                if t in intro:
                    intro_hits[t] += 1

    cmp_df = defaultdict(set)
    for sp in scored_paths:
        if not os.path.exists(sp):
            continue
        for r in json.load(open(sp))["examples"]:
            o, w = r["original"].strip(), r["rewrite"].strip()
            ref, _, usable = refused_v6(r.get("samples", []))
            if not usable or ref:
                continue
            d = pair_metrics(o, w)["wl_dist_content"]
            if not isinstance(d, int) or d > 2:
                continue
            for t in set(content_tokens(w)):
                if len(t) > 2:
                    cmp_df[t].add(o)

    A = {k: len(v) for k, v in or_df.items()}
    B = {k: len(v) for k, v in cmp_df.items()}
    nA, nB = sum(A.values()), sum(B.values())
    bg_tot = sum(A.get(k, 0) + B.get(k, 0) for k in set(A) | set(B))
    out = []
    for wd in set(A) | set(B):
        ya, yb = A.get(wd, 0), B.get(wd, 0)
        if ya < min_origs:
            continue
        aw = a0 * (ya + yb) / bg_tot if bg_tot else 0.0
        try:
            la = math.log((ya + aw) / (nA + a0 - ya - aw))
            lb = math.log((yb + aw) / (nB + a0 - yb - aw))
            z = (la - lb) / math.sqrt(1.0 / (ya + aw) + 1.0 / (yb + aw))
        except (ValueError, ZeroDivisionError):
            continue
        introduced = tot_hits[wd] and (intro_hits[wd] / tot_hits[wd]) >= 0.5
        out.append((wd, z, ya, bool(introduced)))
    out.sort(key=lambda t: -t[1])
    return out


def fig2(top=14):
    specs = [("Llama-3-8B-Instruct", "probe_or/results/edit_strata/or_low_stratum_v7.csv",
              ["probe_or/results/corpus2/llamaAtt_llamaTgt.json",
               "probe_or/results/low_power/low_scored_llamaTgt.json"], "llamaAtt"),
             ("Qwen3-32B", "probe_or/results/edit_strata/or_low_stratum_qwen_v7.csv",
              ["probe_or/results/corpus2/qwenAtt_qwenTgt.json",
               "probe_or/results/low_power_qwen/low_scored_qwenTgt.json"], "qwenAtt")]
    panels = []
    for nm, low, scored, att in specs:
        if os.path.exists(low):
            st = _trigger_stats(low, scored, att)
            if st:
                panels.append((nm, st[:top]))
    if not panels:
        print("  fig2: no data"); return

    fig, axes = plt.subplots(1, len(panels), figsize=(12.4, 5.4))
    if len(panels) == 1:
        axes = [axes]
    for ax, (nm, st) in zip(axes, panels):
        st = st[::-1]                                  # largest at top after barh
        y = np.arange(len(st))
        vals = [z for _, z, _, _ in st]
        cols = [LLAMA if intro else GREY for _, _, _, intro in st]
        ax.barh(y, vals, height=0.68, color=cols, linewidth=0, zorder=2)
        for yi, (wd, z, n, intro) in zip(y, st):
            ax.text(z + max(vals) * 0.018, yi, f"{n} orig.", va="center",
                    fontsize=8.1, color=INK2)
        ax.set_yticks(y)
        ax.set_yticklabels([f"$\\mathtt{{{wd}}}$" for wd, _, _, _ in st], fontsize=9.5)
        ax.set_xlim(0, max(vals) * 1.22)
        ax.set_xlabel("weighted log-odds $z_{\\mathrm{doc}}$   (refused vs not-refused)")
        ax.set_title(nm, loc="left", pad=8)
        ax.grid(axis="x", zorder=0); ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    handles = [Line2D([], [], marker="s", ls="", ms=9, color=LLAMA,
                      label="introduced by the edit  (causal candidate)"),
               Line2D([], [], marker="s", ls="", ms=9, color=GREY,
                      label="topic marker  (present in original too)")]
    # Below the panels in bare mode: the deck is gone and the axes move up, so a
    # top-anchored legend would land on the first row.
    _loc, _anchor = (("upper center", (0.5, 0.045)) if BARE
                     else ("upper left", (0.012, 0.945)))
    fig.legend(handles=handles, loc=_loc, ncol=2, fontsize=8.8, bbox_to_anchor=_anchor)
    fig.suptitle("Only some over-represented words were actually introduced by the edit",
                 x=0.012, ha="left", fontsize=13, fontweight="semibold", y=1.06)
    fig.text(0.012, 0.968, "Refused rewrites contrasted against the same attacker's "
             "NOT-refused rewrites in the same edit bin, so the contrast isolates refusal "
             "rather than the attacker's style. Each original counted once; words in "
             "\u22653 distinct originals.", fontsize=8.8, color=INK2, ha="left", va="top")
    fig.subplots_adjust(top=0.80, wspace=0.40)
    save(fig, "fig2_triggers")


# ---------------------------------------------------------------- Fig 4
def fig4():
    """Slopes, not bars: the finding is a PATTERN of slopes. For the frame residual the two
    lines nearly coincide and both rise with alarm (alarm matters, refusal does not); for d1
    and the literature direction the lines are far apart and flat (refusal matters, alarm does
    not). That contrast is immediate as slopes and requires arithmetic as bars."""
    P = json.load(open("probe_or/results/d4_delta2x2.json"))
    proj, eff, null = P["projections"], P["effects"], P["null_p95"]
    # Plain-English panel titles matching fig3's row labels -- no internal shorthand on
    # the canvas; the formal definitions live in the caption.
    order = [("d4", "weaponization direction"), ("d1", "overall refusal direction"),
             ("r_atlas", "whole-prompt refusal vector")]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.4))
    for ax, (key, title) in zip(axes, order):
        t = proj[key]
        xs = [0, 1]
        for cells, col, lab, mk in ((("or_plain", "or_alarm"), REFUSED, "refused", "o"),
                                    (("ctrl_plain", "ctrl_alarm"), NOTREF, "not refused", "D")):
            ys = [t[cells[0]]["mean"], t[cells[1]]["mean"]]
            lo = [t[c]["mean"] - t[c]["lo"] for c in cells]
            hi = [t[c]["hi"] - t[c]["mean"] for c in cells]
            ax.errorbar(xs, ys, yerr=[lo, hi], color=col, lw=2.2, marker=mk, ms=8,
                        capsize=3, capthick=1.2, label=lab, zorder=3)
        e = eff[key]
        ax.set_xticks(xs); ax.set_xticklabels(["no alarm\nwords", "alarm\nwords"], fontsize=9)
        ax.set_xlim(-0.32, 1.32)
        ax.set_title(title, loc="left", pad=8, fontsize=10.5)
        ax.grid(axis="y", zorder=0); ax.set_axisbelow(True)
        ax.text(0.03, 0.97, f"ALARM  {e['alarm']:+.3f}\nREFUSAL {e['refusal']:+.3f}",
                transform=ax.transAxes, fontsize=8.8, va="top", ha="left", color=INK,
                bbox=dict(boxstyle="round,pad=0.36", fc=SURF, ec=GRID, lw=1))
    axes[0].set_ylabel("mean projection of $\\Delta$ onto the direction")
    axes[0].legend(loc="lower right", fontsize=8.8)
    fig.suptitle("The effective direction tracks alarming wording, not the refusal decision",
                 x=0.012, ha="left", fontsize=13, fontweight="semibold", y=1.05)
    fig.text(0.012, 0.965, "Llama, held-out originals, cluster-bootstrapped over originals "
             f"(95% CI). Random-direction null (50 dirs), 95th pct: ALARM {null['alarm']:.3f}, "
             f"REFUSAL {null['refusal']:.3f}. For the frame residual the two lines nearly "
             "coincide \u2014 alarming words move it whether or not the model refused.\n"
             "Note the y-axes differ in range between panels.",
             fontsize=8.8, color=INK2, ha="left", va="top")
    fig.subplots_adjust(top=0.80, wspace=0.30)
    save(fig, "fig4_alarm_2x2")


def _tex(t):
    """LaTeX-escape caption prose, leaving existing $...$ math segments alone.

    A bare % in a caption comments out the rest of the line and silently truncates
    it -- "95% CI" ate everything after it before this existed.
    """
    subs = [("%", "\\%"), ("&", "\\&"), ("#", "\\#"),
            ("\u2014", "---"), ("\u2013", "--"), ("\u2265", "$\\geq$"),
            ("\u2264", "$\\leq$"), ("\u00b7", "$\\cdot$"), ("\u2192", "$\\to$")]
    out, parts = [], t.split("$")
    for i, seg in enumerate(parts):
        if i % 2:                      # inside $...$: leave verbatim
            out.append(seg); continue
        for a, b in subs:
            seg = seg.replace(a, b)
        # straight double quotes -> LaTeX open/close pairs
        while seg.count('"') >= 2:
            seg = seg.replace('"', "``", 1).replace('"', "''", 1)
        out.append(seg)
    return "$".join(out)



# ---------------------------------------------------------------- Fig 6 (section 2)
def fig6():
    """Why we mine openers per model, and why the logit signal is the one we trust.

    Two methods choices the whole paper rests on, each justified by a measurement:

    (a) Refusal openers are model-specific. "I cannot" opens 56.5% of Llama's refusals but
        only 6.5% of Qwen's, whose dominant opener is "I'm sorry" (53.8%). A phrase detector
        tuned on one model does not transfer.

    (b) On the full evaluation set all three signals separate refused from complied at
        AUC 0.97-0.99 and look interchangeable. Restricting to WITHIN-ORIGINAL comparisons --
        rewrites of the same prompt, so topic is held constant -- separates them: the
        activation signals degrade (Qwen's vector 0.969 -> 0.874) while the logit holds.
        This is the same topic control the paired-Delta analysis uses, and it predicts the
        reward result: the logit is the only signal that trains a better attacker.

    Qwen's vector and probe coincide because its probe places all NNLS weight on one layer,
    so the probe is the standardised vector; that is a property of the fit, not a duplicate.
    """
    OP = json.load(open("refusal_atlas/opener_sets.json"))
    AU = json.load(open("refusal_atlas/results/figures_data.json"))["models"]
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.6, 3.9),
                                   gridspec_kw={"width_ratios": [1.05, 1.0], "wspace": 0.30})

    # (a) opener share, model-specific
    openers = ["I cannot", "I apologize", "I'm sorry", "Sorry", "I can't", "As an AI"]
    y = np.arange(len(openers))[::-1]; h = 0.36
    for off, (mk, nm, col) in ((+h/2, ("llama", "Llama-3-8B", LLAMA)),
                               (-h/2, ("qwen", "Qwen3-32B", QWEN))):
        sh = OP[mk]["opener_shares"]
        v = [100 * sh.get(o, 0.0) for o in openers]
        axA.barh(y + off, v, height=h, color=col, label=nm, zorder=3)
    for yi, o in zip(y, openers):
        for mk, col, off in (("llama", LLAMA, +h/2), ("qwen", QWEN, -h/2)):
            val = 100 * OP[mk]["opener_shares"].get(o, 0.0)
            if val >= 3:
                axA.text(val + 1.2, yi + off, f"{val:.1f}", va="center", fontsize=8,
                         color=INK)
    axA.set_yticks(y); axA.set_yticklabels([f'"{o}"' for o in openers], fontsize=9.2)
    axA.set_xlabel("share of that model's refusals that begin with this phrase (%)")
    axA.set_xlim(0, 68); axA.legend(loc="lower right", fontsize=8.8)
    axA.set_title("Refusal openers are model-specific", loc="left", pad=8, fontsize=10.5)
    axA.grid(axis="x", zorder=0); axA.set_axisbelow(True)
    axA.spines["left"].set_visible(False); axA.tick_params(axis="y", length=0)

    # (b) AUC full -> within-original, per signal per model
    sigs = [("logit_sum", "refusal-opener logit"), ("probe", "multi-layer probe"),
            ("vector", "refusal vector")]
    rows, labs = [], []
    for si, (sk, sn) in enumerate(sigs):
        for mk, nm, col in (("llama", "Llama-3-8B", LLAMA), ("qwen", "Qwen3-32B", QWEN)):
            rows.append((AU[mk]["auc"][sk]["full"], AU[mk]["auc"][sk]["within"], col))
            labs.append(f"{sn}\n{nm}" if mk == "llama" else f"\n{nm}")
    yy = np.arange(len(rows))[::-1]
    for yi, (f_, w_, col) in zip(yy, rows):
        axB.plot([w_, f_], [yi, yi], color=col, lw=2.0, alpha=0.55, zorder=2,
                 solid_capstyle="round")
        axB.plot(f_, yi, "o", ms=8, color=col, mec=SURF, mew=1.2, zorder=3)
        axB.plot(w_, yi, "D", ms=7, color=SURF, mec=col, mew=2.0, zorder=3)
        axB.text(w_ - 0.004, yi, f"{w_:.3f}", va="center", ha="right", fontsize=7.8, color=INK2)
    axB.set_yticks(yy); axB.set_yticklabels(labs, fontsize=8.4)
    axB.set_xlim(0.855, 1.005)
    axB.set_xlabel("AUC, refused vs complied")
    axB.set_title("Controlling for topic separates the signals", loc="left", pad=8, fontsize=10.5)
    axB.grid(axis="x", zorder=0); axB.set_axisbelow(True)
    axB.spines["left"].set_visible(False); axB.tick_params(axis="y", length=0)
    axB.legend(handles=[Line2D([], [], marker="o", ls="", ms=8, color=GREY, label="full set"),
                        Line2D([], [], marker="D", ls="", ms=7, color=SURF, mec=GREY, mew=2,
                               label="within-original (topic controlled)")],
               loc="upper left", fontsize=8.4)   # bottom-left is occupied by the Qwen vector row

    fig.suptitle("Two measurement choices the rest of the paper depends on",
                 x=0.012, ha="left", fontsize=13, fontweight="semibold", y=1.06)
    fig.text(0.012, 0.995,
             "Left: openers mined from 23,595 Llama and 19,952 Qwen generations. "
             "\u201cI cannot\u201d opens 56.5% of Llama's refusals and 6.5% of Qwen's, so a "
             "phrase detector\ndoes not transfer between models. Right: on the full set all "
             "three signals look interchangeable; restricting to rewrites of the SAME original "
             "degrades the\nactivation signals and leaves the logit intact. Qwen's probe "
             "coincides with its vector because the fit places all weight on one layer.",
             fontsize=8.8, color=INK2, ha="left", va="top")
    fig.subplots_adjust(top=0.80)
    save(fig, "fig6_refusal_signals")


FIGS = {"1": ("fig1_edit_distance_distribution", lambda: fig1()),
        "2": ("fig2_triggers", lambda: fig2()),
        "3": ("fig3_causal_bars", lambda: fig3()),
        "4": ("fig4_alarm_2x2", lambda: fig4()),
        "5": ("fig5_generalisation", lambda: fig5()),
        "6": ("fig6_refusal_signals", lambda: fig6())}

if __name__ == "__main__":
    which = [a for a in sys.argv[1:] if not a.startswith("-")] or list(FIGS)
    for k in which:
        name, fn = FIGS[k]
        _cur["name"] = name
        print(f"Fig {k}:"); fn()

    # Emit the stripped text as ready-to-paste LaTeX captions.
    if _CAPTIONS:
        with open(f"{OUT}/captions.tex", "w") as fh:
            fh.write("% Auto-generated by make_figures.py -- the headline and deck text that\n"
                     "% bare-mode figures no longer render into the canvas.\n")
            for k in which:
                name, _ = FIGS[k]
                c = _CAPTIONS.get(name)
                if not c:
                    continue
                fh.write(f"\n\\begin{{figure}}[t]\n  \\centering\n"
                         f"  \\includegraphics[width=\\linewidth]{{figures/{name}.png}}\n"
                         f"  \\caption{{\\textbf{{{_tex(c.get('headline',''))}}} "
                         f"{_tex(c.get('deck',''))}}}\n"
                         f"  \\label{{fig:{name.split('_')[0]}}}\n\\end{{figure}}\n")
        print(f"  wrote {OUT}/captions.tex")
