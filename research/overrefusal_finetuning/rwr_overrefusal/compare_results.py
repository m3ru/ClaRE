#!/usr/bin/env python3
"""Aggregate prompt-iteration results across runs into a single comparison.

Pulls each (variant, source-results-file) pair listed in ``VARIANT_SOURCES``,
then computes summary stats for two prompt subsets:

  * ``full_20``     — every (original, rewrite) pair from the source run.
  * ``filtered_14`` — same pairs minus the 6 alpaca prompts in ``RESISTANT_PROMPTS``.
                     Those 6 produced negative mean refusal_delta across every
                     non-adversarial variant in round 1 (legitimate professional
                     tasks where no rewriting strategy can lift the refusal
                     direction without injecting adversarial framing).

Outputs:
  - prints a 6-row table (3 variants × 2 prompt subsets) sorted by or_score_raw
  - writes the full aggregated stats to ``prompt_iteration_results/comparison.json``
"""
import json
import os
import statistics
from typing import Dict, List


def _percentile(sorted_values: List[float], pct: float) -> float:
    """Linear-interpolation percentile (matches numpy.percentile default)."""
    if not sorted_values:
        return 0.0
    n = len(sorted_values)
    if n == 1:
        return float(sorted_values[0])
    rank = (pct / 100.0) * (n - 1)
    lo = int(rank)
    hi = min(lo + 1, n - 1)
    frac = rank - lo
    return float(sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(THIS_DIR, "prompt_iteration_results")


# Each entry is (display_label, source_results_json, variant_name_in_that_json).
# Reusing round 1 outputs for baseline_alec and imperative_procedural because the
# prompts didn't change; only minimal_changes uses fresh v2 outputs.
VARIANT_SOURCES = [
    ("baseline_alec",         "iterate_alpaca_results.json",        "baseline_alec"),
    ("imperative_procedural", "iterate_alpaca_results.json",        "imperative_procedural"),
    ("minimal_changes_v2",    "minimal_changes_v2_results.json",    "minimal_changes"),
]


# Alpaca prompts that produced negative mean refusal_delta across every
# non-adversarial round-1 variant (vocab_swap, imperative, project_explanation).
# These read as legitimate professional/applied tasks where the refusal direction
# stays unmoved or moves negatively regardless of surface rewriting.
RESISTANT_PROMPTS = [
    "Suggest a strategy for a company to create an effective social media presence.",
    "What are the benefits of using blockchain technology in the finance industry?",
    "Describe a traditional wedding ceremony you attended",
    "Come up with a product idea to solve a problem.",
    "Create a product design document for a sunrise alarm clock.",
    "Write a function to find the longest word in a sentence.",
]


def load_variant_gens(source_filename: str, variant_name: str) -> List[Dict]:
    path = os.path.join(RESULTS_DIR, source_filename)
    with open(path) as f:
        data = json.load(f)
    return data["variants"][variant_name]["generations"]


def stats(gens: List[Dict]) -> Dict:
    """Summary stats for a list of (original, generation, scores) dicts."""
    if not gens:
        return {"n": 0}
    def _row(values: List[float]) -> Dict:
        s = sorted(values)
        return {
            "mean":         statistics.fmean(values),
            "median":       statistics.median(values),
            "p90":          _percentile(s, 90),
            "pct_positive": sum(1 for v in values if v > 0) / len(values),
        }
    return {
        "n":              len(gens),
        "n_prompts":      len({g["original"] for g in gens}),
        "or_score_raw":   _row([g["or_score_raw"] for g in gens]),
        "refusal_delta":  _row([g["refusal_delta"] for g in gens]),
        "similarity":     _row([g["similarity"] for g in gens]),
    }


def main():
    aggregated: Dict[str, Dict] = {}
    resistant = set(RESISTANT_PROMPTS)

    for label, source_filename, variant_name in VARIANT_SOURCES:
        try:
            gens = load_variant_gens(source_filename, variant_name)
        except FileNotFoundError:
            print(f"[skip] {label}: results file not found ({source_filename})")
            continue
        except KeyError:
            print(f"[skip] {label}: variant '{variant_name}' not in {source_filename}")
            continue
        full     = gens
        filtered = [g for g in gens if g["original"] not in resistant]
        n_dropped_prompts = len({g["original"] for g in gens}) - len({g["original"] for g in filtered})
        aggregated[label] = {
            "source_file":   source_filename,
            "source_variant": variant_name,
            "full_20":     stats(full),
            "filtered_14": stats(filtered),
            "n_resistant_prompts_dropped": n_dropped_prompts,
        }

    # --- Pretty comparison table ---
    print("=" * 100)
    print("VARIANT × PROMPT-SUBSET COMPARISON")
    print("=" * 100)
    print("\nfull_20     = all 20 alpaca-cleaned prompts × 3 generations each (60 pairs)")
    print("filtered_14 = same minus 6 resistant prompts (42 pairs)\n")

    for metric in ["or_score_raw", "refusal_delta", "similarity"]:
        print(f"\n{metric}:")
        print(f"  {'variant × subset':40s} {'n':>4s} {'mean':>9s} {'median':>9s} {'p90':>9s} {'%pos':>8s}")
        rows = []
        for label, info in aggregated.items():
            for subset_key in ("full_20", "filtered_14"):
                s = info.get(subset_key)
                if not s or s.get("n") == 0:
                    continue
                m = s[metric]
                rows.append((
                    f"{label} × {subset_key}", s["n"], m["mean"], m["median"], m["p90"], m["pct_positive"]
                ))
        # Sort by mean within metric (descending)
        rows.sort(key=lambda r: r[2], reverse=True)
        for name, n, mean, median, p90, pp in rows:
            print(f"  {name:40s} {n:4d} {mean:9.3f} {median:9.3f} {p90:9.3f} {pp:7.1%}")

    out_path = os.path.join(RESULTS_DIR, "comparison.json")
    with open(out_path, "w") as f:
        json.dump({
            "resistant_prompts": RESISTANT_PROMPTS,
            "variants": aggregated,
        }, f, indent=2)
    print(f"\n[done] wrote {out_path}")


if __name__ == "__main__":
    main()
