#!/usr/bin/env python3
"""Feed a fixed set of rewrites to a TARGET model and measure its refusal rate.

For the attacker x target grid (harder-target vs better-attacker). Reuses the exact
Stage-2 measurement of eval_rwr_llama.py: each rewrite -> target model with the
helpful-assistant system -> n samples -> broad opener classifier. Writes per-rewrite
refuse_rate so refusals can be bootstrapped over originals and (optionally) benign-audited.

Run: python score_target_refusal.py --rewrites_csv <csv w/ orig_idx,original,rewrite> \
        --base_model <target> --n_samples 4 --output <out.json>
"""
import argparse, csv, json, os, sys
csv.field_size_limit(sys.maxsize)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_or"))
from eval_rwr_llama import load_gen, generate, JUDGE_SYSTEM
from gen_qwen_refusal import classify


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rewrites_csv", required=True)
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--n_samples", type=int, default=4)
    ap.add_argument("--max_new", type=int, default=64)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.rewrites_csv)))
    rewrites = [r["rewrite"] for r in rows]
    print(f"[grid] {len(rewrites)} rewrites -> target {a.base_model}", flush=True)

    model, tok = load_gen(a.base_model)
    reps = generate(model, tok, rewrites, JUDGE_SYSTEM, None, a.n_samples, a.temp, a.max_new, a.bs)

    out = []
    tot = 0.0
    for r, samples in zip(rows, reps):
        flags = [classify(s)[0] for s in samples]
        rr = sum(flags) / len(flags)
        tot += rr
        out.append({"orig_idx": r.get("orig_idx"), "original": r["original"],
                    "rewrite": r["rewrite"], "refuse_rate": rr, "samples": samples})
    mean = tot / len(out) if out else 0.0
    json.dump({"target": a.base_model, "rewrites_csv": a.rewrites_csv,
               "n_rewrites": len(out), "mean_refuse_rate": mean, "examples": out},
              open(a.output, "w"))
    print(f"[done] {a.output} | target refuse_rate = {mean*100:.2f}% over {len(out)} rewrites", flush=True)


if __name__ == "__main__":
    main()
