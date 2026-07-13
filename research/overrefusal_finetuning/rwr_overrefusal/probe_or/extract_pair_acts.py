#!/usr/bin/env python3
"""Paired activation extraction for the DELTA probe experiment.

Reads pairs.csv (build_pair_eval.py) and extracts last-token, all-layer
activations for BOTH the original and the rewrite of each pair, kept aligned by
row. Reuses the verified padding-safe reader from extract_layer_acts.py.

Outputs (--out_dir): acts_orig.npy, acts_rw.npy  (fp16, [n_pairs, nL+1, H])
aligned with pair_meta.json ({p_orig, p_rw, dP, similarity}).
"""
import argparse
import csv
import json
import os

import numpy as np
import torch

from extract_layer_acts import last_token_all_layers   # verified right-pad last-token read


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--base_model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=512)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rows = list(csv.DictReader(open(args.pairs_csv)))
    originals = [r["original"] for r in rows]
    rewrites = [r["rewrite"] for r in rows]
    meta = [{"p_orig": float(r["p_orig"]), "p_rw": float(r["p_rw"]),
             "dP": float(r["dP"]), "similarity": float(r["similarity"])} for r in rows]
    print(f"[data] {len(rows)} pairs -> {2*len(rows)} forward passes")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf_token = os.environ.get("HF_TOKEN")
    tok = AutoTokenizer.from_pretrained(args.base_model, token=hf_token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, token=hf_token, torch_dtype=torch.bfloat16,
        device_map="auto", attn_implementation="eager").eval()

    print("[extract] originals...", flush=True)
    np.save(os.path.join(args.out_dir, "acts_orig.npy"),
            last_token_all_layers(model, tok, originals, args.batch_size, args.max_length))
    print("[extract] rewrites...", flush=True)
    np.save(os.path.join(args.out_dir, "acts_rw.npy"),
            last_token_all_layers(model, tok, rewrites, args.batch_size, args.max_length))
    json.dump(meta, open(os.path.join(args.out_dir, "pair_meta.json"), "w"))
    print(f"[done] acts_orig/acts_rw.npy + pair_meta.json -> {args.out_dir}")


if __name__ == "__main__":
    main()
