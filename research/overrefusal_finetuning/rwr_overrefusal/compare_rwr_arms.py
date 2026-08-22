#!/usr/bin/env python3
"""Assemble every RWR arm's held-out over-refusal rate into one table.

The question: used as the OR reward, does the ABLITERATED refusal vector (the direction
Arditi's ablation criterion selects) train a better attacker than the layers we picked by
other criteria, or than the logit signal?

Each arm's eval reports two numbers on the SAME held-out originals: `rwr` (the trained
attacker's rewrites) and `base` (the untrained model's rewrites). The bar an arm must clear
is its OWN base, because base rates differ slightly between evals.

Reads whatever exists, so it can be run before and after the new arms land.
"""
import glob, json, os, sys
from math import sqrt

ARMS = {
    "llama": [
        ("eval_llama_vector",         "vector @L17  (behavioural corr.)"),
        ("eval_llama_vector_L31",     "vector @L31  (AUC)"),
        ("eval_llama_vector_abl_L12", "vector @L12  (ABLITERATED)"),
        ("eval_llama_probe",          "probe"),
        ("eval_llama_logit",          "logit"),
    ],
    "qwen": [
        ("eval_qwen_3sig_vector",     "vector @L58  (AUC)"),
        ("eval_qwen_vector_abl_L60",  "vector @L60  (ABLITERATED)"),
        ("eval_qwen_3sig_probe",      "probe"),
        ("eval_qwen_3sig_logit",      "logit"),
    ],
}


def wilson(k, n, z=1.96):
    if not n:
        return (float("nan"),) * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def load(tag):
    for p in (f"probe_or/results/{tag}/eval_final.json",):
        if os.path.exists(p):
            return json.load(open(p))
    return None


for model, arms in ARMS.items():
    print(f"\n=== {model.upper()} — held-out over-refusal rate (n=200 originals x 4 rewrites) ===")
    print(f"{'arm':34s} {'trained':>18s} {'untrained base':>18s} {'lift':>8s}  {'sim':>5s}")
    for tag, label in arms:
        d = load(tag)
        if d is None:
            print(f"{label:34s} {'— not run —':>18s}")
            continue
        r, b = d["arms"]["rwr"], d["arms"]["base"]
        nr, nb = r["n_rewrites_total"], b["n_rewrites_total"]
        rr, bb = 100 * r["mean_refuse_rate"], 100 * b["mean_refuse_rate"]
        rl, rh = r.get("refuse_rate_ci95", [float("nan")] * 2)
        bl, bh = b.get("refuse_rate_ci95", [float("nan")] * 2)
        beats = "BEATS" if rl * 100 > bh * 100 else ("loses" if rh * 100 < bl * 100 else "ties")
        print(f"{label:34s} {rr:6.2f} [{100*rl:5.2f},{100*rh:5.2f}] "
              f"{bb:6.2f} [{100*bl:5.2f},{100*bh:5.2f}] {rr-bb:+7.2f}  {r['mean_similarity']:.3f}  {beats}")
    print("  lift = trained minus its own untrained base. 'BEATS'/'loses' = non-overlapping 95% CIs.")

print("\nNote: the trained arms hold semantics far better than base (sim ~0.83 vs ~0.63), so a\n"
      "lower refusal rate is not automatically worse -- base wins some refusals by drifting off\n"
      "topic, which the judge would reject as intent-shifted. Judge-confirmed rates are the\n"
      "final word; these are refusal rates under the eval's own classifier.")
