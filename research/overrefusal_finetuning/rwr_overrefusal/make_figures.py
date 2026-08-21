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
                     f"$D_c\\leq2$: {low:.1f}%)", loc="left")
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
# Figure labels use no internal shorthand. d1 is literally the average difference between
# rewrites the model refused and rewrites it did not; it correlates 0.78 with the published
# refusal vector, so it IS a refusal direction -- ours, from paired data, versus theirs, from
# unpaired harmful/harmless prompts. The contrast a reader needs is overall vs frame-specific
# and ours vs published. (It is NOT an "alarm" direction: the frame residuals are what load on
# alarming wording, and only after d1 is removed.)
NICE = {"d1_shared": "overall refusal direction", "atlas_rhat": "published refusal vector"}
ROW_ORDER = ["overall refusal direction", "published refusal vector", "weaponization",
             "concealment", "exfiltration", "exploitation", "intrusion", "coercion",
             "fabrication"]

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

    fig, axes = plt.subplots(1, len(panels), figsize=(12.8, 5.6), sharey=True)
    if len(panels) == 1:
        axes = [axes]

    # rows are the union of both models' directions, in one fixed order, so the panels align
    present = set()
    for _, path in panels:
        present |= {pretty(n) for n, _, _ in load(path)[0]}
    rows = [r for r in ROW_ORDER if r in present] + sorted(present - set(ROW_ORDER))
    rows = rows[::-1]                                      # first entry at the top

    for ax, (nm, path) in zip(axes, panels):
        pts, nullmax = load(path)
        lut = {pretty(n): (o, h) for n, o, h in pts}
        names = rows
        dor = np.array([lut.get(r, (np.nan, np.nan))[0] for r in rows], dtype=float)
        dhm = np.array([lut.get(r, (np.nan, np.nan))[1] for r in rows], dtype=float)
        y = np.arange(len(rows)); hgt = 0.36

        ax.barh(y + hgt / 2 + 0.02, np.nan_to_num(dor), height=hgt, color=LLAMA,
                linewidth=0, label="over-refusal  (want a big drop)")
        ax.barh(y - hgt / 2 - 0.02, np.nan_to_num(dhm), height=hgt, color=QWEN,
                linewidth=0, label="harmful-prompt refusal  (want no drop)")
        ax.axvspan(0, nullmax, color=GREY, alpha=0.18, zorder=0)

        span = np.nanmax([np.nanmax(dor), np.nanmax(dhm)])
        for yi, v in zip(y + hgt / 2 + 0.02, dor):
            if np.isfinite(v):
                ax.text(v + span * 0.015, yi, f"{v:.1f}", va="center", fontsize=8.4, color=INK)
        for yi, v in zip(y - hgt / 2 - 0.02, dhm):
            if np.isfinite(v):
                lab = f"{v:.1f}" + ("  \u2190 safety damage" if v >= 2 else "")
                ax.text(max(v, 0) + span * 0.015, yi, lab, va="center", fontsize=8.4,
                        color=INK if v < 2 else QWEN)
        for yi, r in zip(y, rows):
            if r not in lut:
                ax.text(span * 0.02, yi, "not in this model's frame set", va="center",
                        fontsize=7.8, color=GREY, style="italic")
        ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9.5)
        ax.set_xlim(min(-2, np.nanmin(dhm) * 1.2), span * 1.32)
        ax.axvline(0, color=INK2, lw=1.0)
        ax.set_xlabel("percentage-point DROP in refusal after ablation")
        ax.set_title(nm, loc="left", pad=8)
        ax.grid(axis="x", zorder=0); ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)

    from matplotlib.patches import Patch
    hs, ls = axes[0].get_legend_handles_labels()
    hs.append(Patch(facecolor=GREY, alpha=0.18))
    ls.append("random-direction null")
    # Figure-level, below the panels: an in-axes legend collides with the "not in this
    # model's frame set" labels on the empty rows, and which rows are empty depends on
    # the data, so no in-axes corner is reliably safe.
    fig.legend(hs, ls, loc="upper center", bbox_to_anchor=(0.5, 0.035),
               ncol=3, fontsize=8.8)
    fig.suptitle("A single direction removes over-refusal without costing safety",
                 x=0.012, ha="left", fontsize=13, fontweight="semibold", y=1.04)
    fig.text(0.012, 0.968, "\"Overall\" = the average difference between rewrites the model "
             "refused and rewrites it did not. \"Published\" = the refusal vector from prior "
             "work. The rest are\nthat same difference computed within one vocabulary group, "
             "with the overall direction removed. Both bars are DROPS: the model refuses "
             "less of both after ablation. A good direction drops over-refusal a lot and "
             "harmful-prompt refusal not at all.\nEach ablated alone, at every layer; held-out "
             "originals: 400 confirmed over-refusals and 200 AdvBench harmful prompts. "
             "A good direction has a long blue bar and no orange one.",
             fontsize=8.8, color=INK2, ha="left", va="top")
    fig.subplots_adjust(top=0.84, wspace=0.42)
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
             ("r_atlas", "published refusal vector")]
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


FIGS = {"1": ("fig1_edit_distance_distribution", lambda: fig1()),
        "2": ("fig2_triggers", lambda: fig2()),
        "3": ("fig3_causal_bars", lambda: fig3()),
        "4": ("fig4_alarm_2x2", lambda: fig4()),
        "5": ("fig5_generalisation", lambda: fig5())}

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
