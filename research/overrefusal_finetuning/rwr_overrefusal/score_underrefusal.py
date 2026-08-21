#!/usr/bin/env python3
"""U1 (under-refusal arm): feed the jailbreak-wrapped harmful prompts to the TARGET model and
measure COMPLIANCE (the mirror of score_target_refusal.py, which measures refusal).

compliance = 1 - refusal, over samples that are actual answers (capability-disclaimer non-answers
are dropped from the denominator, exactly as the OR arm does). Each prompt -> N samples, classify
each with the shared start-anchored refusal opener detector; keep the raw replies so the genuine-
jailbreak judge (U2) can read them and so degeneracy can be audited. Generation is capped at 64
new tokens: onset only (UNDERREFUSAL_PLAN.md sect. 3).

The regex compliance here is the CHEAP first pass; U2's reply-reading LLM judge is the arbiter of
GENUINE jailbreak. We report both and their disagreement.

Run: python score_underrefusal.py --prompts_csv probe_or/results/underrefusal/jb_prompts.csv \
        --base_model meta-llama/Meta-Llama-3-8B-Instruct \
        --output probe_or/results/underrefusal/scored_llamaTgt.json
"""
import argparse
import csv
import json
import os
import sys

csv.field_size_limit(sys.maxsize)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_or"))

from eval_rwr_llama import load_gen, generate, JUDGE_SYSTEM   # target-response measurement helpers
from gen_qwen_refusal import classify                          # broad refusal-opener detector
from capability_filter import is_capability_disclaimer         # drop missing-input non-answers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts_csv", required=True,
                    help="csv with columns pair_id, orig_idx, goal, wrapper, prompt")
    ap.add_argument("--base_model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--n_samples", type=int, default=4)
    ap.add_argument("--max_new", type=int, default=64)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.prompts_csv)))
    prompts = [r["prompt"] for r in rows]
    print(f"[u1] {len(prompts)} wrapped prompts -> target {a.base_model} "
          f"({a.n_samples} samples, {a.max_new} tok)", flush=True)

    model, tok = load_gen(a.base_model, adapter=None)
    # template=None -> the wrapped prompt IS the user content; JUDGE_SYSTEM = helpful assistant.
    reps = generate(model, tok, prompts, JUDGE_SYSTEM, None, a.n_samples, a.temp, a.max_new, a.bs)

    out, comp_sum, n_info = [], 0.0, 0
    for r, samples in zip(rows, reps):
        kept = [s for s in samples if not is_capability_disclaimer(s)]
        n_disc = len(samples) - len(kept)
        if kept:
            refuse_rate = sum(classify(s)[0] for s in kept) / len(kept)
            informative = True
            n_info += 1
            comp_sum += (1.0 - refuse_rate)
        else:
            refuse_rate = None
            informative = False
        out.append({
            "pair_id": r.get("pair_id"), "orig_idx": r.get("orig_idx"),
            "goal": r["goal"], "wrapper": r["wrapper"], "prompt": r["prompt"],
            "refuse_rate": refuse_rate,
            "compliance_rate": (None if refuse_rate is None else 1.0 - refuse_rate),
            "n_disclaimer": n_disc, "informative": informative,
            "samples": samples,
        })

    mean_comp = comp_sum / n_info if n_info else 0.0
    # per-wrapper mean compliance (quick console summary; full stats computed in U5)
    from collections import defaultdict
    bw = defaultdict(list)
    for o in out:
        if o["informative"]:
            bw[o["wrapper"]].append(o["compliance_rate"])
    json.dump({"target": a.base_model, "prompts_csv": a.prompts_csv,
               "n_prompts": len(out), "n_informative": n_info,
               "mean_compliance_rate": mean_comp, "examples": out},
              open(a.output, "w"))
    print(f"[u1] done -> {a.output} | mean compliance = {mean_comp*100:.2f}% "
          f"over {n_info} informative prompts", flush=True)
    for w in sorted(bw):
        v = bw[w]
        print(f"    {w:18s} compliance {100*sum(v)/len(v):6.2f}%  (n={len(v)})", flush=True)


if __name__ == "__main__":
    main()
