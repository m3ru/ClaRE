#!/usr/bin/env python3
"""Behavioral eval of a Llama-3-8B RWR attacker (corrected-signal retrain).

Stage 1: the trained attacker (base Llama + LoRA) rewrites held-out benign prompts.
Stage 2: feed each rewrite to BASE Llama (helpful-assistant system), generate, and
classify refusal with the broad opener detector. Report the actual refusal rate the
attacker induces, vs the base-Llama-as-attacker baseline and the untouched-original
floor, plus MiniLM similarity.

Held-out prompts come from --heldout_json (written by build_llama_shards.py BEFORE
any signal-specific filtering and excluded from every signal's shards), so the
vector/probe/logit attackers are all evaluated on the SAME prompts with no
train/eval overlap in any arm. Mirrors eval_rwr_qwen.py otherwise.
"""
import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_or"))

from rwr_config import SYSTEM_PROMPT, PROMPT_TEMPLATE
from gen_qwen_refusal import classify   # broad refusal-opener detector (same as scoring)

JUDGE_SYSTEM = "You are a helpful assistant."   # system used when measuring Llama's refusal


def load_gen(base, adapter=None):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base, token=os.environ.get("HF_TOKEN"))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        base, token=os.environ.get("HF_TOKEN"), torch_dtype=torch.bfloat16,
        device_map="auto", attn_implementation="eager")
    if adapter:
        print(f"[eval] loading adapter {adapter}", flush=True)
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    return model.eval(), tok


def term_ids(tok):
    term = [tok.eos_token_id]
    for t in ("<|im_end|>", "<|endoftext|>", "<|eot_id|>"):
        tid = tok.convert_tokens_to_ids(t)
        if isinstance(tid, int) and tid >= 0 and tid not in term:
            term.append(tid)
    return term


def bootstrap_ci(orig_idx, values, n_boot=1000, seed=0):
    """95% CI on the mean, resampling ORIGINALS (not rewrites) to respect clustering."""
    orig_idx, values = np.asarray(orig_idx), np.asarray(values)
    uniq = np.unique(orig_idx)
    by_o = {u: values[orig_idx == u] for u in uniq}
    rng = np.random.RandomState(seed)
    means = []
    for _ in range(n_boot):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        means.append(np.concatenate([by_o[u] for u in samp]).mean())
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def generate(model, tok, prompts, system, template, n, temp, max_new, bs):
    """n samples per prompt. template=None => user content is the prompt itself."""
    def fmt(p):
        user = template.format(prompt=p) if template else p
        return tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
    term = term_ids(tok)
    out = [None] * len(prompts)
    for i in range(0, len(prompts), bs):
        chunk = prompts[i:i + bs]
        enc = tok([fmt(p) for p in chunk], return_tensors="pt", padding=True, truncation=True,
                  max_length=512, add_special_tokens=False).to(model.device)
        L = enc["input_ids"].shape[1]
        with torch.no_grad():
            g = model.generate(**enc, do_sample=True, temperature=temp, top_p=0.9,
                               num_return_sequences=n, max_new_tokens=max_new,
                               eos_token_id=term, pad_token_id=tok.pad_token_id)
        dec = tok.batch_decode(g[:, L:], skip_special_tokens=True)
        for b in range(len(chunk)):
            out[i + b] = [d.strip() for d in dec[b * n:(b + 1) * n]]
        if (i // bs) % 20 == 0:
            print(f"  [gen {i + len(chunk)}/{len(prompts)}]", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter_dir", required=True)
    ap.add_argument("--heldout_json", required=True,
                    help="build_llama_shards.py heldout_originals.json (common across signals)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--base_model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--n_rewrites", type=int, default=4, help="rewrites per held-out prompt")
    ap.add_argument("--n_refusal_samples", type=int, default=4, help="base-Llama samples per rewrite")
    ap.add_argument("--max_eval_prompts", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval_base_model", action="store_true", help="also run base-Llama-as-attacker baseline")
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    # --- held-out prompts: the common pre-filter split (same file for all 3 signals) ---
    val_originals = sorted(json.load(open(args.heldout_json)))
    if len(val_originals) > args.max_eval_prompts:
        import random as _r
        _r.seed(args.seed)
        val_originals = sorted(_r.sample(val_originals, args.max_eval_prompts))
    print(f"[eval] {len(val_originals)} held-out prompts", flush=True)

    arms = {}   # arm -> list aligned with val_originals, each a list of n_rewrites strings

    # --- Stage 1: attacker(s) rewrite the held-out prompts ---
    t0 = time.time()
    model, tok = load_gen(args.base_model, args.adapter_dir)
    arms["rwr"] = generate(model, tok, val_originals, SYSTEM_PROMPT, PROMPT_TEMPLATE,
                           args.n_rewrites, args.temperature, 64, bs=8)
    if args.eval_base_model:
        del model; gc.collect(); torch.cuda.empty_cache()
        model, tok = load_gen(args.base_model, adapter=None)
        arms["base"] = generate(model, tok, val_originals, SYSTEM_PROMPT, PROMPT_TEMPLATE,
                                args.n_rewrites, args.temperature, 64, bs=8)
    del model; gc.collect(); torch.cuda.empty_cache()
    # control floor: base-Llama refusal on the UNTOUCHED originals (rewrite == original)
    arms["orig"] = [[o] for o in val_originals]
    print(f"[eval] attacker generation done in {time.time()-t0:.0f}s", flush=True)

    # flatten unique rewrites for the judge pass
    flat = []   # (arm, orig_idx, rewrite)
    for arm, gens in arms.items():
        for oi, rws in enumerate(gens):
            for rw in rws:
                flat.append((arm, oi, rw))
    rewrites = [f[2] for f in flat]

    # --- Stage 2: base Llama (helpful assistant) responds; classify refusal ---
    t0 = time.time()
    judge, tok = load_gen(args.base_model, adapter=None)
    resp = generate(judge, tok, rewrites, JUDGE_SYSTEM, None,
                    args.n_refusal_samples, args.temperature, 64, bs=8)
    del judge; gc.collect(); torch.cuda.empty_cache()
    print(f"[eval] judge generation done in {time.time()-t0:.0f}s", flush=True)

    # --- MiniLM similarity(original, rewrite) ---
    from sentence_transformers import SentenceTransformer
    minilm = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    origs_for_flat = [val_originals[f[1]] for f in flat]
    eo = minilm.encode(origs_for_flat, convert_to_numpy=True, normalize_embeddings=True, batch_size=64)
    er = minilm.encode(rewrites, convert_to_numpy=True, normalize_embeddings=True, batch_size=64)
    sims = (eo * er).sum(1)

    # --- aggregate per arm (full records persisted; bootstrap CI over prompts) ---
    results = {"n_prompts": len(val_originals), "n_rewrites": args.n_rewrites,
               "n_refusal_samples": args.n_refusal_samples, "adapter_dir": args.adapter_dir,
               "arms": {}, "examples": {}}
    for arm in arms:
        idx = [i for i, f in enumerate(flat) if f[0] == arm]
        recs = []
        for i in idx:
            flags = [classify(s) for s in resp[i]]
            recs.append({
                "original": origs_for_flat[i], "rewrite": flat[i][2], "orig_idx": int(flat[i][1]),
                "refuse_rate": float(np.mean([f[0] for f in flags])),
                "icannot_rate": float(np.mean([f[0] and f[1] for f in flags])),
                "similarity": float(sims[i]),
                "rewrite_is_refusal_shaped": bool(classify(flat[i][2])[0]),  # base-attacker degeneracy detector
                "samples": resp[i],
            })
        rr = np.array([r["refuse_rate"] for r in recs])
        lo, hi = bootstrap_ci([r["orig_idx"] for r in recs], rr, seed=args.seed)
        results["arms"][arm] = {
            "n_rewrites_total": len(recs),
            "mean_refuse_rate": float(rr.mean()),
            "refuse_rate_ci95": [round(lo, 4), round(hi, 4)],
            "frac_rewrites_refused_ge1x": float((rr > 0).mean()),
            "mean_icannot_rate": float(np.mean([r["icannot_rate"] for r in recs])),
            "mean_similarity": float(np.mean([r["similarity"] for r in recs])),
            "frac_rewrites_refusal_shaped": float(np.mean([r["rewrite_is_refusal_shaped"] for r in recs])),
        }
        results["examples"][arm] = recs   # FULL per-rewrite records + all judge samples (auditable)

    json.dump(results, open(args.output, "w"), indent=2)
    print("\n" + "=" * 80 + "\nBEHAVIORAL EVAL (Llama-3-8B RWR attacker) -- base-Llama refusal of each arm's rewrites"
          + "\n" + "=" * 80)
    print(f"{'arm':6s} {'n_rw':>5s} {'refuse_rate':>12s} {'95% CI':>17s} {'icannot':>8s} {'sim':>6s} {'rw_refusal':>11s}")
    for arm, a in results["arms"].items():
        ci = a["refuse_rate_ci95"]
        print(f"{arm:6s} {a['n_rewrites_total']:5d} {a['mean_refuse_rate']:11.1%} "
              f"  [{ci[0]:6.1%},{ci[1]:6.1%}] {a['mean_icannot_rate']:7.1%} "
              f"{a['mean_similarity']:6.3f} {a['frac_rewrites_refusal_shaped']:10.1%}")
    print("  orig = base-Llama refusal on untouched held-out originals (the floor). "
          "rw_refusal = fraction of an arm's REWRITES that are themselves refusals (base-attacker degeneracy).")
    print(f"\n[done] wrote {args.output}")


if __name__ == "__main__":
    main()
