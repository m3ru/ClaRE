#!/usr/bin/env python3
"""Build and SAVE the harmful-vs-harmless diff-of-means refusal direction at EVERY layer.

The earlier arditi_direction_test.py only persisted the selected layer (L12); the
prompt-optimization work needs L17 (best behavioral readout) and L12 (causal lever)
plus arbitrary others, so build all of them once and cache.

Replicates the exact recipe used for the correlation sweep (harmful_dir_vs_logit_corr.py):
AdvBench 'goal' vs Alpaca-cleaned, first n_dir of each, post-instruction last token,
add_generation_prompt=True. Keeping this identical is what makes L17 the 0.600-Spearman layer.
"""
import argparse
import csv
import json
import os

import numpy as np


def read_col(path, col, n):
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        c = col if col in r.fieldnames else r.fieldnames[0]
        rows = [(_r.get(c) or "").strip() for _r in r]
    return [v for v in rows if v and v.lower() != col][:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harmful_csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base_model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--n_dir", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    tok = AutoTokenizer.from_pretrained(args.base_model, token=hf)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base_model, token=hf, device_map="auto",
                                                 torch_dtype=torch.bfloat16)
    model.eval()
    dev = model.device
    H = model.config.hidden_size
    nL = model.config.num_hidden_layers

    def fmt(p):
        msgs = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": p}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def class_mean(texts):
        sm = np.zeros((nL + 1, H), np.float64)
        cnt = 0
        for i in range(0, len(texts), args.batch_size):
            b = texts[i:i + args.batch_size]
            # add_special_tokens=False: apply_chat_template already emits <|begin_of_text|>.
            # Must match prompt_opt.py exactly or the projections are not comparable.
            enc = tok([fmt(p) for p in b], return_tensors="pt", padding=True, add_special_tokens=False,
                      truncation=True, max_length=512).to(dev)
            with torch.no_grad():
                hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states
            last = torch.stack([h[:, -1, :] for h in hs], dim=1).float().cpu().numpy()
            sm += last.sum(0)
            cnt += last.shape[0]
        return sm / cnt

    harmful = read_col(args.harmful_csv, "goal", args.n_dir)
    ds = load_dataset("yahma/alpaca-cleaned", split="train")
    harmless = []
    for ex in ds:
        inst = (ex.get("instruction") or "").strip()
        inp = (ex.get("input") or "").strip()
        pp = f"{inst}\n\n{inp}" if inp else inst
        if inst and len(pp) < 1500:
            harmless.append(pp)
        if len(harmless) >= args.n_dir:
            break
    print(f"[dir] harmful {len(harmful)} vs harmless {len(harmless)}", flush=True)

    d_raw = class_mean(harmful) - class_mean(harmless)          # [nL+1, H]
    norms = np.linalg.norm(d_raw, axis=1)
    d_hat = d_raw / (norms[:, None] + 1e-9)
    np.savez(args.out, d_raw=d_raw.astype(np.float32), d_hat=d_hat.astype(np.float32),
             norms=norms.astype(np.float32))

    # max-L2 DIM-norm pick for the NEW data (the criterion that gave L32 on the OLD data)
    top = int(np.argmax(norms[1:]) + 1)
    meta = {"n_dir": args.n_dir, "n_layers": nL, "max_l2_layer": top,
            "norms": {str(i): float(norms[i]) for i in range(nL + 1)}}
    with open(os.path.splitext(args.out)[0] + "_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[dir] max-L2 DIM-norm layer = L{top} (norm {norms[top]:.1f}) of {nL} layers")
    print("[dir] " + "   ".join(f"L{L} norm {norms[L]:.1f}" for L in (12, 17, 32) if L <= nL))
    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()
