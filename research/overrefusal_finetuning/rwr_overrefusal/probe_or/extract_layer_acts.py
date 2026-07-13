#!/usr/bin/env python3
"""Extract per-layer last-token activations for the probe-ensemble experiment.

Two roles (route (b): the direction split is INDEPENDENT of the OR pools):
  1. refusal / benign split  -> builds the per-layer diff-of-means directions
     (same prompt lists that produced the original L32 refusal vector).
  2. eval set  -> pool texts (Claude+orp3k rewrites/originals) with their stored
     P("I cannot") labels, projected later to measure per-layer / ensemble AUC.

For each text: one forward pass, grab the residual-stream hidden state at the
LAST REAL prompt token (padding-agnostic, via the attention mask) at every layer.
Formatting matches score_icannot_or.py / the sweep: system="You are a helpful
assistant.", add_generation_prompt=True — so directions and projections live in
the same activation space the OR scorer reads.

Outputs (--out_dir): acts_ref.npy, acts_ben.npy, acts_eval.npy  (all fp16,
shape [n, nL+1, H]) and eval_meta.json (parallel list of {text, p_icannot}).
"""
import argparse
import csv
import json
import os
import random
import time

import numpy as np
import torch

SYSTEM = "You are a helpful assistant."


def read_prompts(path, cap, rng):
    out = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        col = "prompt" if "prompt" in (r.fieldnames or []) else r.fieldnames[0]
        for row in r:
            p = (row.get(col) or "").strip()
            if p and p.lower() != "prompt":
                out.append(p)
    rng.shuffle(out)
    return out[:cap]


def build_eval_set(pool_csvs, cap, rng):
    """Unique (text, P('I cannot')) from the icannot_or pairs CSVs. Keep the whole
    high-signal tail (P>1e-3) + a random sample of the rest, up to cap."""
    seen = {}
    for path in pool_csvs:
        for row in csv.DictReader(open(path)):
            for text_key, p_key in (("rewrite", "p_icannot_rewrite"), ("original", "p_icannot_orig")):
                t = (row.get(text_key) or "").strip()
                if t and t not in seen:
                    seen[t] = float(row[p_key])
    items = list(seen.items())
    hi = [(t, p) for t, p in items if p > 1e-3]
    lo = [(t, p) for t, p in items if p <= 1e-3]
    rng.shuffle(lo)
    keep = hi + lo[:max(0, cap - len(hi))]
    rng.shuffle(keep)
    return keep


def last_token_all_layers(model, tok, texts, batch_size, max_length):
    """Return fp16 array [n, nL+1, H] of the last real prompt-token hidden state
    at every layer."""
    def fmt(p):
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    assert tok.padding_side == "right", "last-token read assumes right padding"
    nL = model.config.num_hidden_layers
    H = model.config.hidden_size
    out = np.zeros((len(texts), nL + 1, H), dtype=np.float16)
    t0 = time.time()
    for i in range(0, len(texts), batch_size):
        batch = [fmt(t) for t in texts[i:i + batch_size]]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_length, add_special_tokens=False).to(model.device)
        last = enc["attention_mask"].sum(dim=1) - 1          # last real token index per row (right pad)
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states
        # hs: tuple len nL+1 of [B, T, H]
        rows = torch.arange(len(batch), device=model.device)
        for L, h in enumerate(hs):
            out[i:i + len(batch), L, :] = h[rows, last, :].float().cpu().numpy().astype(np.float16)
        if (i // batch_size) % 20 == 0:
            print(f"  [{i + len(batch)}/{len(texts)}] {(i+len(batch))/max(time.time()-t0,1e-3):.1f}/s", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refusal_csv", required=True)
    ap.add_argument("--benign_csv", required=True)
    ap.add_argument("--pool_csvs", nargs="+", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--base_model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--n_per_class", type=int, default=2500)
    ap.add_argument("--max_eval", type=int, default=12000)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf_token = os.environ.get("HF_TOKEN")
    tok = AutoTokenizer.from_pretrained(args.base_model, token=hf_token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"   # REQUIRED: last real token = attention_mask.sum(1)-1 assumes right padding
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, token=hf_token, torch_dtype=torch.bfloat16,
        device_map="auto", attn_implementation="eager").eval()

    ref = read_prompts(args.refusal_csv, args.n_per_class, rng)
    ben = read_prompts(args.benign_csv, args.n_per_class, rng)
    ev = build_eval_set(args.pool_csvs, args.max_eval, rng)
    print(f"[data] refusal={len(ref)} benign={len(ben)} eval={len(ev)} "
          f"(eval P>1e-3: {sum(p>1e-3 for _,p in ev)})", flush=True)

    print("[extract] refusal split...", flush=True)
    np.save(os.path.join(args.out_dir, "acts_ref.npy"),
            last_token_all_layers(model, tok, ref, args.batch_size, args.max_length))
    print("[extract] benign split...", flush=True)
    np.save(os.path.join(args.out_dir, "acts_ben.npy"),
            last_token_all_layers(model, tok, ben, args.batch_size, args.max_length))
    print("[extract] eval pool...", flush=True)
    np.save(os.path.join(args.out_dir, "acts_eval.npy"),
            last_token_all_layers(model, tok, [t for t, _ in ev], args.batch_size, args.max_length))
    json.dump([{"text": t, "p_icannot": p} for t, p in ev],
              open(os.path.join(args.out_dir, "eval_meta.json"), "w"))
    print(f"[done] wrote acts_ref/ben/eval.npy + eval_meta.json -> {args.out_dir}")


if __name__ == "__main__":
    main()
