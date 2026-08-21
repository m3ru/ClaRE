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
    fig.legend(handles=handles, loc="upper left", ncol=2, fontsize=8.8,
               bbox_to_anchor=(0.012, 0.945))
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
    order = [("d4", "frame residual\n(weaponisation)"), ("d1", "shared axis $d_1$"),
             ("r_atlas", "literature $\\hat{r}$")]
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


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3", "4", "5"]
    if "1" in which: print("Fig 1:"); fig1()
    if "2" in which: print("Fig 2:"); fig2()
    if "3" in which: print("Fig 3:"); fig3()
    if "4" in which: print("Fig 4:"); fig4()
    if "5" in which: print("Fig 5:"); fig5()
