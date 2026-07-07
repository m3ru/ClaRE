#!/usr/bin/env python3
"""Behavioral refusal eval for rewriter models.

For a chunk of held-out alpaca prompts (disjoint from training), generate
n rewrites each from two rewriters -- `baseline` (bare Llama, no adapter) and
`top20` (the RWR top-20%-bin adapter) -- then feed every rewrite to a BARE
Llama-3-8B-Instruct (system="You are a helpful assistant", greedy) and record
whether it refuses. Also runs the untouched originals as a control.

Outputs (in --out_dir):
  responses_all.jsonl   one record per (arm, prompt_idx, sample_idx)
  refusals_only.jsonl   just the regex-refused records, for eyeballing
Prints per-arm refusal rate + delta vs originals.

NOTE: refusal detection here is the start-of-response regex from
test_claude_rewrites_on_llama.py. It OVER-counts (benign "I'm sorry, no passage
was provided" clarifications match). Full responses are saved so a Claude-judge
pass (judge_with_claude.py, needs ANTHROPIC_API_KEY) can refine later.
"""
import argparse
import json
import os
import sys
import time

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, os.path.join(THIS_DIR, "..", "ppo_or"))

from eval_rwr import load_generator, generate_paraphrases            # noqa: E402
from eval_held_out import load_alpaca_held_out                       # noqa: E402
from test_claude_rewrites_on_llama import (                          # noqa: E402
    load_llama, generate_batch, regex_classify,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top20_adapter", required=True)
    ap.add_argument("--base_model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--out_dir", default="prompt_iteration_results/behavioral_rewriters")
    ap.add_argument("--n_prompts", type=int, default=200)
    ap.add_argument("--n_per_prompt", type=int, default=4)
    ap.add_argument("--rewrite_max_new_tokens", type=int, default=64)
    ap.add_argument("--rewrite_temperature", type=float, default=0.7)
    ap.add_argument("--rewrite_batch_size", type=int, default=4)
    ap.add_argument("--resp_max_new_tokens", type=int, default=256)
    ap.add_argument("--resp_batch", type=int, default=16)
    # held-out reconstruction (must match training/eval to stay disjoint)
    ap.add_argument("--eval_seed", type=int, default=99)
    ap.add_argument("--training_seed", type=int, default=42)
    ap.add_argument("--training_n_alpaca", type=int, default=2500)
    ap.add_argument("--iter_seed", type=int, default=42)
    ap.add_argument("--iter_n", type=int, default=20)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # --- held-out prompts ---
    held = load_alpaca_held_out(
        args.n_prompts, args.eval_seed, args.training_seed,
        args.training_n_alpaca, args.iter_seed, args.iter_n,
    )
    prompts = [p for _, p in held]
    idxs = [i for i, _ in held]
    print(f"[behav] {len(prompts)} held-out alpaca prompts")

    # --- Stage 1: generate rewrites from each rewriter arm ---
    items = []  # {arm, prompt_idx, original, sample_idx, text}
    # control: the untouched originals (one per prompt)
    for pi, orig in zip(idxs, prompts):
        items.append({"arm": "original", "prompt_idx": pi, "original": orig,
                      "sample_idx": 0, "text": orig})

    arms = [("baseline", None), ("top20", args.top20_adapter)]
    for arm, adapter in arms:
        print(f"\n[behav] === generating rewrites: {arm} (adapter={adapter}) ===")
        t0 = time.time()
        model, tok = load_generator(args.base_model, adapter_dir=adapter)
        gens = generate_paraphrases(
            model, tok, prompts,
            n_per_prompt=args.n_per_prompt,
            temperature=args.rewrite_temperature,
            max_new_tokens=args.rewrite_max_new_tokens,
            batch_size=args.rewrite_batch_size,
        )
        for pi, orig in zip(idxs, prompts):
            for si, rw in enumerate(gens.get(orig, [])):
                items.append({"arm": arm, "prompt_idx": pi, "original": orig,
                              "sample_idx": si, "text": rw})
        del model
        import torch
        torch.cuda.empty_cache()
        print(f"[behav] {arm} rewrites done in {time.time()-t0:.0f}s")

    print(f"\n[behav] total items to send to bare Llama: {len(items)}")

    # --- Stage 2: bare Llama responds to each text, regex-classify ---
    print("[behav] === loading bare Llama responder ===")
    model, tok = load_llama(args.base_model)
    all_path = os.path.join(args.out_dir, "responses_all.jsonl")
    ref_path = os.path.join(args.out_dir, "refusals_only.jsonl")
    n_ref = {}
    n_tot = {}
    t0 = time.time()
    with open(all_path, "w") as fa, open(ref_path, "w") as fr:
        for s in range(0, len(items), args.resp_batch):
            chunk = items[s:s + args.resp_batch]
            resps = generate_batch(model, tok, [c["text"] for c in chunk], args.resp_max_new_tokens)
            for c, resp in zip(chunk, resps):
                label, pat = regex_classify(resp)
                rec = {**c, "response": resp, "regex_label": label, "regex_pattern": pat}
                fa.write(json.dumps(rec) + "\n")
                n_tot[c["arm"]] = n_tot.get(c["arm"], 0) + 1
                if label == "refused":
                    n_ref[c["arm"]] = n_ref.get(c["arm"], 0) + 1
                    fr.write(json.dumps(rec) + "\n")
            if (s // args.resp_batch) % 10 == 0:
                print(f"  responded {min(s+args.resp_batch, len(items))}/{len(items)}", flush=True)
    print(f"[behav] responses done in {time.time()-t0:.0f}s")

    # --- summary ---
    print("\n================ REFUSAL RATES (regex) ================")
    base_orig = n_ref.get("original", 0) / max(n_tot.get("original", 1), 1)
    for arm in ["original", "baseline", "top20"]:
        r = n_ref.get(arm, 0)
        t = n_tot.get(arm, 0)
        rate = r / max(t, 1)
        delta = "" if arm == "original" else f"   delta vs original = {100*(rate-base_orig):+.2f} pp"
        print(f"  {arm:9s}: {r:4d}/{t:4d} refused = {100*rate:5.2f}%{delta}")
    print(f"\n[behav] wrote {all_path}")
    print(f"[behav] wrote {ref_path}  (refusals only)")


if __name__ == "__main__":
    main()
