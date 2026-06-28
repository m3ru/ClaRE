#!/usr/bin/env python3
"""Render held-out alpaca comparison charts across all five arms (incl. raw Claude
teacher). Produces three figures from the result JSONs:

  rwr_meanOR_alpaca.png       mean OR (broad-consistency view)
  rwr_p90OR_alpaca.png        p90 OR (extreme-tail view; the percentile RWR trains on)
  rwr_mean_vs_p90_alpaca.png  side-by-side panels showing the ranking flip

All numbers: held-out alpaca, n=600, identical 200 prompts, k=5.0/c=0.75/d=100.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

THIS = os.path.dirname(os.path.abspath(__file__))

# label -> (sublabel, is_distilled, color)
META = {
    "claude_rwr":         ("RWR on Claude paraphrases",       True,  "#1f6feb"),
    "claude_paraphrases": ("raw Claude (teacher)",            False, "#79b8ff"),
    "llama_self_rwr":     ("RWR on Llama self-paraphrases",   True,  "#2da44e"),
    "baseline":           ("raw Llama (base policy)",         False, "#8b949e"),
    "rwr_v3":             ("RWR on gpt-oss paraphrases",      True,  "#bc8cff"),
}


def load_stats():
    k5 = json.load(open(os.path.join(THIS, "prompt_iteration_results/held_out_eval/held_out_eval_results_k5.json")))
    ls = json.load(open(os.path.join(THIS, "prompt_iteration_results/held_out_eval_llama_self/held_out_eval_results.json")))
    rc = json.load(open(os.path.join(THIS, "prompt_iteration_results/claude_heldout_alpaca.json")))

    def s(d, lbl):  # alpaca or_score_raw stats
        return d["results"][lbl]["alpaca"]["stats"]["or_score_raw"]
    out = {}
    for lbl in ("claude_rwr", "baseline", "rwr_v3"):
        st = s(k5, lbl)
        out[lbl] = (st["mean"], st["p90"])
    st = s(ls, "llama_self_rwr")
    out["llama_self_rwr"] = (st["mean"], st["p90"])
    st = rc["variants"]["imitation_research_framing"]["stats"]["or_score_raw"]
    out["claude_paraphrases"] = (st["mean"], st["p90"])
    return out


def _style_ax(ax):
    ax.grid(axis="x", color="#e1e4e8", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#d0d7de")
    ax.tick_params(length=0)


def _two_line(lbl):
    return f"{lbl}\n{META[lbl][0]}"


def single_chart(stats, idx, title_metric, fname, subtitle):
    order = sorted(stats, key=lambda k: stats[k][idx])  # ascending -> best on top
    vals = [stats[k][idx] for k in order]
    colors = [META[k][2] for k in order]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, ax = plt.subplots(figsize=(9.6, 5.4), dpi=200)
    fig.patch.set_facecolor("white")
    ax.barh(range(len(order)), vals, color=colors, height=0.62,
            edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([_two_line(k) for k in order])
    xmax = max(vals)
    for i, v in enumerate(vals):
        ax.text(v + xmax * 0.012, i, f"{v:.4f}", va="center", ha="left",
                fontsize=11, fontweight="bold", color="#24292f")
    ax.set_xlim(0, xmax * 1.18)
    ax.set_xlabel(f"{title_metric} over-refusal (OR) score   —   higher = rewrites trigger more over-refusal",
                  fontsize=10.5, color="#57606a")
    ax.set_title(f"Reward-weighted regression vs. raw paraphrasing\n{subtitle}",
                 fontsize=13.5, fontweight="bold", pad=14, loc="left")
    _style_ax(ax)
    ax.legend(handles=[Patch(facecolor="#1f6feb", label="RWR-trained (distilled)"),
                       Patch(facecolor="#8b949e", label="Raw paraphrasing (no training)")],
              loc="lower right", frameon=False, fontsize=9.5)
    fig.text(0.012, 0.012,
             "OR = exp(5·(sim−0.75))·refusal_delta/100, layer-32 refusal-vector activations. "
             "Held-out alpaca (n=600), disjoint from training. Proxy metric, not behavioral refusals.",
             fontsize=7.6, color="#8b949e")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    out = os.path.join(THIS, fname)
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def grouped_chart(stats, fname):
    # Fixed order by p90 desc (best on top) so the mean panel visibly disagrees.
    order = sorted(stats, key=lambda k: -stats[k][1])
    ypos = list(range(len(order)))[::-1]  # top = best
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, (axm, axp) = plt.subplots(1, 2, figsize=(12.4, 5.2), dpi=200, sharey=True)
    fig.patch.set_facecolor("white")

    for ax, idx, name in ((axm, 0, "Mean OR"), (axp, 1, "p90 OR")):
        vals = [stats[k][idx] for k in order]
        colors = [META[k][2] for k in order]
        ax.barh(ypos, vals, color=colors, height=0.6, edgecolor="white", linewidth=0.8, zorder=3)
        xmax = max(vals)
        for y, v in zip(ypos, vals):
            ax.text(v + xmax * 0.015, y, f"{v:.4f}", va="center", ha="left",
                    fontsize=10, fontweight="bold", color="#24292f")
        ax.set_xlim(0, xmax * 1.2)
        ax.set_title(name, fontsize=12.5, fontweight="bold", loc="left", pad=8)
        _style_ax(ax)
    axm.set_yticks(ypos)
    axm.set_yticklabels([_two_line(k) for k in order])
    fig.suptitle("Held-out alpaca: mean vs. p90 OR — the ranking depends on the statistic\n"
                 "(rows ordered by p90; note rwr_v3 & baseline swap rank between panels)",
                 fontsize=13.5, fontweight="bold", x=0.012, ha="left", y=1.02)
    fig.text(0.012, 0.012,
             "n=600, identical 200 prompts, k=5.0. Mean rewards broad consistency; p90 rewards the extreme tail RWR trains on.",
             fontsize=7.8, color="#8b949e")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    out = os.path.join(THIS, fname)
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    stats = load_stats()
    single_chart(stats, 0, "Mean", "rwr_meanOR_alpaca.png",
                 "Mean OR on held-out alpaca (n=600, identical 200 prompts, k=5.0)")
    single_chart(stats, 1, "p90", "rwr_p90OR_alpaca.png",
                 "p90 OR on held-out alpaca (n=600, identical 200 prompts, k=5.0)")
    grouped_chart(stats, "rwr_mean_vs_p90_alpaca.png")


if __name__ == "__main__":
    main()
