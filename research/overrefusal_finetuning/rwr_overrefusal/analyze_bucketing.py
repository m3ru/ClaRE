#!/usr/bin/env python3
"""Diagnostic: did our bucketing waste signal on the Claude data?

Loads two deduped shard directories (Claude + orp3k baseline) through the same
pipeline `train_rwr.py` uses (`load_shards` -> `recompute_or_scores` ->
`filter_and_bin`), then dumps per-bin stats and cross-dataset comparisons.

The single most-load-bearing number this produces:

    Claude bin 2 (currently DROPPED, weight 0) mean OR
        vs
    orp3k bin 4 (currently weight 16) mean OR

If Claude's dropped-bin OR rivals or exceeds orp3k's kept-bin OR, the
weight=[0,0,0,1,16] scheme is throwing away usable training signal.

Outputs:
  prompt_iteration_results/bucketing_analysis.json
  prompt_iteration_results/bucketing_analysis.md
"""
import argparse
import json
import os
import statistics
import sys
from collections import defaultdict

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

from rwr_config import BinningConfig, DataConfig
from rwr_data import filter_and_bin, load_shards, recompute_or_scores


def run_pipeline(shard_dir: str, binning_config: BinningConfig):
    """Match exactly what train_rwr.py does up to (and including) binning.

    Re-runs the digitize after `filter_and_bin` because weight-to-bin recovery
    is ambiguous when bin_weights has duplicates (e.g. [0,0,0,1,16]).
    """
    data_config = DataConfig(shard_dir=shard_dir)
    pairs = load_shards(data_config)
    if binning_config.recompute_or_score:
        recompute_or_scores(pairs, binning_config)
    filtered, _ = filter_and_bin(pairs, binning_config)
    rewards = np.array([p[binning_config.reward_key] for p in filtered])
    percentiles = np.linspace(0, 100, binning_config.num_bins + 1)
    edges = np.percentile(rewards, percentiles)
    bins = np.digitize(rewards, edges[1:-1])
    return filtered, bins


def percentile(arr, q):
    return float(np.percentile(arr, q)) if len(arr) else 0.0


def per_bin_stats(pairs, bins, num_bins):
    ors  = np.array([p["or_score_raw"] for p in pairs])
    sims = np.array([p["similarity"]    for p in pairs])
    origs = [p["original"] for p in pairs]
    rows = []
    for b in range(num_bins):
        mask = bins == b
        n = int(mask.sum())
        if n == 0:
            rows.append({"bin": b, "n": 0})
            continue
        bin_ors  = ors[mask]
        bin_sims = sims[mask]
        bin_origs = [o for o, m in zip(origs, mask) if m]
        rows.append({
            "bin": b,
            "n": n,
            "or_mean":   float(bin_ors.mean()),
            "or_median": float(np.median(bin_ors)),
            "or_p25":    percentile(bin_ors, 25),
            "or_p75":    percentile(bin_ors, 75),
            "or_p90":    percentile(bin_ors, 90),
            "or_p95":    percentile(bin_ors, 95),
            "or_min":    float(bin_ors.min()),
            "or_max":    float(bin_ors.max()),
            "or_std":    float(bin_ors.std()),
            "sim_mean":  float(bin_sims.mean()),
            "n_unique_originals": len(set(bin_origs)),
            "frac_above_or_0.05": float((bin_ors > 0.05).mean()),
            "frac_above_or_0.10": float((bin_ors > 0.10).mean()),
            "frac_above_or_0.20": float((bin_ors > 0.20).mean()),
        })
    return rows


def per_prompt_variance(pairs):
    """Within-prompt vs across-prompt OR variance.

    High within-prompt variance => same prompt's samples spread widely in OR
    (so per-prompt top-1 sampling would lose less than pooled binning loses);
    low within-prompt variance => the bulk of variance is across prompts.
    """
    by_orig = defaultdict(list)
    for p in pairs:
        by_orig[p["original"]].append(p["or_score_raw"])
    multi = [vs for vs in by_orig.values() if len(vs) > 1]
    if not multi:
        return {"prompts_with_multi_samples": 0}
    within = [float(np.var(vs)) for vs in multi]
    means  = [float(np.mean(vs)) for vs in multi]
    return {
        "prompts_with_multi_samples": len(multi),
        "within_prompt_var_mean":    float(np.mean(within)),
        "within_prompt_var_median":  float(np.median(within)),
        "across_prompt_var":         float(np.var(means)),
        "ratio_within_over_across":  (float(np.mean(within)) / float(np.var(means))) if np.var(means) > 0 else None,
    }


def alt_binning_stats(pairs, binning_config, alt_num_bins_list):
    """Show what bin structure would look like with different num_bins on the
    SAME post-filter pool. Just for visibility — does not change training."""
    ors = np.array([p["or_score_raw"] for p in pairs])
    out = {}
    for k in alt_num_bins_list:
        percentiles = np.linspace(0, 100, k + 1)
        edges = np.percentile(ors, percentiles)
        bins = np.digitize(ors, edges[1:-1])
        rows = []
        for b in range(k):
            mask = bins == b
            n = int(mask.sum())
            row = {"bin": b, "n": n}
            if n:
                bo = ors[mask]
                row.update({
                    "or_mean": float(bo.mean()),
                    "or_median": float(np.median(bo)),
                    "or_max": float(bo.max()),
                })
            rows.append(row)
        out[str(k)] = {"edges": [float(x) for x in edges], "rows": rows}
    return out


def cross_compare(stats_by_dataset, num_bins):
    """The headline question: is Claude's DROPPED bin OR > orp3k's KEPT bin OR?

    Compares each non-top dataset's dropped bins (0..num_bins-3, i.e. bins below
    the canonical [0,0,0,1,16] scheme's bin 3) against orp3k's top bin (bin 4)
    mean OR.
    """
    if "orp3k" not in stats_by_dataset:
        return {}
    orp3k_bin4 = next((r for r in stats_by_dataset["orp3k"]["bins"] if r["bin"] == num_bins - 1), None)
    if not orp3k_bin4 or orp3k_bin4["n"] == 0:
        return {}
    threshold = orp3k_bin4["or_mean"]
    median_threshold = orp3k_bin4["or_median"]
    out = {
        "orp3k_bin4_mean_or":   threshold,
        "orp3k_bin4_median_or": median_threshold,
        "datasets": {},
    }
    for name, ds in stats_by_dataset.items():
        if name == "orp3k":
            continue
        comparisons = []
        for row in ds["bins"]:
            if row["n"] == 0:
                continue
            comparisons.append({
                "bin": row["bin"],
                "n": row["n"],
                "or_mean": row["or_mean"],
                "ratio_vs_orp3k_bin4_mean":   (row["or_mean"]   / threshold)        if threshold else None,
                "ratio_vs_orp3k_bin4_median": (row["or_median"] / median_threshold) if median_threshold else None,
                "better_than_orp3k_bin4_mean": row["or_mean"] >= threshold,
            })
        out["datasets"][name] = comparisons
    return out


def render_markdown(report, num_bins, bin_weights):
    lines = []
    P = lines.append
    P("# Bucketing diagnostic")
    P("")
    P(f"BinningConfig: `num_bins={num_bins}`, `bin_weights={bin_weights}`, "
      f"`k={report['binning']['similarity_exponent']}`, `c={report['binning']['similarity_center']}`, "
      f"`d={report['binning']['refusal_divisor']}`, `similarity_floor={report['binning']['similarity_floor']}`")
    P("")
    P("## Per-dataset, per-bin")
    P("")
    P("| dataset | bin | weight | n | unique_origs | OR mean | OR median | OR p90 | OR max | OR std | sim mean | >0.05 | >0.10 | >0.20 |")
    P("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, ds in report["per_dataset"].items():
        for row in ds["bins"]:
            if row["n"] == 0:
                P(f"| {name} | {row['bin']} | {bin_weights[row['bin']]} | 0 | – | – | – | – | – | – | – | – | – | – |")
                continue
            P(f"| {name} | {row['bin']} | {bin_weights[row['bin']]} | {row['n']} | {row['n_unique_originals']} | "
              f"{row['or_mean']:.4f} | {row['or_median']:.4f} | {row['or_p90']:.4f} | "
              f"{row['or_max']:.4f} | {row['or_std']:.4f} | {row['sim_mean']:.3f} | "
              f"{row['frac_above_or_0.05']:.2f} | {row['frac_above_or_0.10']:.2f} | {row['frac_above_or_0.20']:.2f} |")
    P("")

    if report.get("cross"):
        c = report["cross"]
        P("## Headline: are Claude's dropped bins above orp3k's kept-bin OR?")
        P("")
        P(f"orp3k bin {num_bins-1} (the only bin orp3k weights heavily): "
          f"mean OR **{c['orp3k_bin4_mean_or']:.4f}**, median {c['orp3k_bin4_median_or']:.4f}")
        P("")
        P("| dataset | bin | weight | n | OR mean | OR mean ÷ orp3k bin4 mean | beats orp3k bin4? |")
        P("|---|---|---:|---:|---:|---:|---|")
        for name, rows in c["datasets"].items():
            for r in rows:
                w = bin_weights[r["bin"]]
                ratio = f"{r['ratio_vs_orp3k_bin4_mean']:.2f}×" if r["ratio_vs_orp3k_bin4_mean"] else "–"
                mark = "**YES**" if r["better_than_orp3k_bin4_mean"] else "no"
                P(f"| {name} | {r['bin']} | {w} | {r['n']} | {r['or_mean']:.4f} | {ratio} | {mark} |")
        P("")

    P("## Within-prompt vs across-prompt OR variance")
    P("")
    P("If within ≫ across, top-N per prompt loses a lot of signal — argues for keeping bins.")
    P("If within ≪ across, bins on raw pool are fine; you could equivalently take top-K per prompt.")
    P("")
    P("| dataset | prompts w/ multi samples | within-prompt var mean | across-prompt var | within/across |")
    P("|---|---:|---:|---:|---:|")
    for name, ds in report["per_dataset"].items():
        v = ds["variance"]
        if v.get("prompts_with_multi_samples"):
            ratio = f"{v['ratio_within_over_across']:.2f}" if v.get("ratio_within_over_across") else "–"
            P(f"| {name} | {v['prompts_with_multi_samples']} | "
              f"{v['within_prompt_var_mean']:.5f} | {v['across_prompt_var']:.5f} | {ratio} |")
        else:
            P(f"| {name} | 0 | – | – | – |")
    P("")

    if report.get("alt_binning"):
        P("## Alternate num_bins on Claude (visibility only — does not change training)")
        P("")
        for k, info in report["alt_binning"].items():
            P(f"### num_bins={k}")
            P("")
            P("| bin | n | OR mean | OR median | OR max |")
            P("|---|---:|---:|---:|---:|")
            for r in info["rows"]:
                if r["n"] == 0:
                    P(f"| {r['bin']} | 0 | – | – | – |")
                else:
                    P(f"| {r['bin']} | {r['n']} | {r['or_mean']:.4f} | {r['or_median']:.4f} | {r['or_max']:.4f} |")
            P("")

    P("## Decision criteria")
    P("")
    P("- Compare Claude's dropped bins to orp3k bin 4. If any are above, un-dropping is")
    P("  motivated. (On the laptop dry run: Claude bin 2 was only 0.20× orp3k bin 4 — so")
    P("  the original drop scheme is correct and V1 (finer top bins) is the better bet.)")
    P("- Inspect the alternate `num_bins=10` table on Claude. If the top decile (bin 9)")
    P("  has noticeably higher OR than bin 8 (e.g. > 1.4×), the current flat weight-16")
    P("  wastes signal by lumping the top 20% together. → `run_rwr_v1_finer_top.slurm`.")
    P("- If within/across OR variance is high (> ~1.0): per-prompt top-K sampling may")
    P("  beat the pooled binning entirely (not in this batch).")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--shard_dirs",
        default="claude=./prompt_iteration_results/dataset_research_framing_full_shards_deduped,"
                "orp3k=../or_paraphrase_3k_deduped",
        help="Comma-separated name=path pairs.",
    )
    ap.add_argument("--num_bins", type=int, default=5)
    ap.add_argument("--bin_weights", type=str, default="0,0,0,1,16",
                    help="Used only to annotate the report (not to filter).")
    ap.add_argument("--similarity_floor",     type=float, default=0.5)
    ap.add_argument("--similarity_exponent",  type=float, default=5.0)
    ap.add_argument("--similarity_center",    type=float, default=0.75)
    ap.add_argument("--refusal_divisor",      type=float, default=100.0)
    ap.add_argument("--alt_num_bins", type=str, default="10",
                    help="Comma-separated alternate num_bins values to summarize on Claude only.")
    ap.add_argument("--json_out", default="prompt_iteration_results/bucketing_analysis.json")
    ap.add_argument("--md_out",   default="prompt_iteration_results/bucketing_analysis.md")
    args = ap.parse_args()

    bin_weights = [float(x) for x in args.bin_weights.split(",")]
    if len(bin_weights) != args.num_bins:
        raise SystemExit(f"--bin_weights has {len(bin_weights)} values, expected {args.num_bins}")
    alt_num_bins_list = [int(x) for x in args.alt_num_bins.split(",") if x.strip()]

    binning_config = BinningConfig(
        num_bins=args.num_bins,
        bin_weights=bin_weights,
        similarity_floor=args.similarity_floor,
        similarity_exponent=args.similarity_exponent,
        similarity_center=args.similarity_center,
        refusal_divisor=args.refusal_divisor,
        recompute_or_score=True,
    )

    per_dataset = {}
    for entry in args.shard_dirs.split(","):
        if "=" not in entry:
            raise SystemExit(f"--shard_dirs entry must be name=path, got: {entry!r}")
        name, path = entry.split("=", 1)
        name = name.strip()
        path = path.strip()
        print(f"\n=== {name}: {path} ===")
        if not os.path.isdir(path):
            print(f"  SKIP — directory not found")
            continue
        filtered, bins = run_pipeline(path, binning_config)
        per_dataset[name] = {
            "shard_dir": path,
            "n_after_filter": len(filtered),
            "bins": per_bin_stats(filtered, bins, args.num_bins),
            "variance": per_prompt_variance(filtered),
        }
        if name == "claude":
            per_dataset[name]["_alt_binning_pairs"] = filtered

    # Alt binning summaries on Claude (after main per-dataset loop)
    alt = {}
    if "claude" in per_dataset and alt_num_bins_list:
        alt = alt_binning_stats(per_dataset["claude"].pop("_alt_binning_pairs"), binning_config, alt_num_bins_list)

    report = {
        "binning": {
            "num_bins": args.num_bins,
            "bin_weights": bin_weights,
            "similarity_floor": args.similarity_floor,
            "similarity_exponent": args.similarity_exponent,
            "similarity_center": args.similarity_center,
            "refusal_divisor": args.refusal_divisor,
        },
        "per_dataset": per_dataset,
        "cross": cross_compare(per_dataset, args.num_bins),
        "alt_binning": alt,
    }

    os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
    with open(args.json_out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[out] wrote {args.json_out}")

    md = render_markdown(report, args.num_bins, bin_weights)
    with open(args.md_out, "w") as f:
        f.write(md)
    print(f"[out] wrote {args.md_out}")

    # Echo the headline to stdout so it lands in slurm logs.
    if report.get("cross") and "claude" in report["cross"].get("datasets", {}):
        print("\n=== HEADLINE ===")
        print(f"orp3k bin {args.num_bins-1} (kept, weight {bin_weights[-1]}): "
              f"OR mean {report['cross']['orp3k_bin4_mean_or']:.4f}")
        for r in report["cross"]["datasets"]["claude"]:
            mark = "BEATS" if r["better_than_orp3k_bin4_mean"] else "below"
            w = bin_weights[r["bin"]]
            print(f"  claude bin {r['bin']} (weight {w}): OR mean {r['or_mean']:.4f}  "
                  f"({r['ratio_vs_orp3k_bin4_mean']:.2f}× — {mark} orp3k bin {args.num_bins-1})")


if __name__ == "__main__":
    main()
