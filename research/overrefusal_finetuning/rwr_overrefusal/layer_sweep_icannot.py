#!/usr/bin/env python3
"""Layer sweep: at which Llama-3 layer does the refusal-direction activation delta
best correlate with the behavioral 'I cannot' logprob delta, over the Claude rewrites?

For each layer L (1..32):
  d_L        = mean(last-token hidden @ L | refusal split) - mean(... | benign split)
  ref_delta_L(pair) = proj_L(rewrite) - proj_L(original),  proj_L = <act_L, d_L>/||d_L||
correlate ref_delta_L against icannot_delta (from the existing icannot pairs CSV).

Same prompt formatting everywhere (system='You are a helpful assistant.', add_generation
_prompt=True), left padding, last real token -- matching reward_model / the icannot job.
Layer 32 is the canonical refusal vector, so its correlation should ~match the prior
~0.20 (built-in sanity check).

Outputs: layer_sweep_summary.json (+ prints a per-layer table).
"""
import argparse
import csv
import json
import os
import time
import numpy as np


def read_prompts(path):
    out = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        col = "prompt" if "prompt" in r.fieldnames else r.fieldnames[0]
        for row in r:
            p = (row.get(col) or "").strip()
            if p and p.lower() != "prompt":
                out.append(p)
    return out


def read_pairs(path):
    pairs = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            pairs.append((row["original"], row["rewrite"],
                          float(row["icannot_delta_logprob"]), float(row["similarity"])))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_csv", required=True)
    ap.add_argument("--benign_csv", required=True)
    ap.add_argument("--refusal_csv", required=True)
    ap.add_argument("--base_model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--out_dir", default="prompt_iteration_results/layer_sweep_icannot")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--sim_hi", type=float, default=0.85)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from scipy import stats

    hf = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    tok = AutoTokenizer.from_pretrained(args.base_model, token=hf)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    use_system = True
    try:
        tok.apply_chat_template([{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
                                tokenize=False, add_generation_prompt=True)
    except Exception:
        use_system = False
    model = AutoModelForCausalLM.from_pretrained(args.base_model, token=hf, device_map="auto",
                                                 torch_dtype=torch.bfloat16)
    model.eval()
    nL = model.config.num_hidden_layers  # 32

    def fmt(p):
        msgs = ([{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": p}]
                if use_system else [{"role": "user", "content": p}])
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def last_token_all_layers(texts):
        """Yield per-batch [B, nL+1, H] last-token hidden states (float32, cpu)."""
        for i in range(0, len(texts), args.batch_size):
            batch = texts[i:i + args.batch_size]
            enc = tok([fmt(p) for p in batch], return_tensors="pt", padding=True,
                      truncation=True, max_length=args.max_length).to(model.device)
            with torch.no_grad():
                hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states
            # left padding -> last real token is column -1
            stacked = torch.stack([h[:, -1, :] for h in hs], dim=1).float().cpu()  # [B, nL+1, H]
            yield batch, stacked

    # --- 1. per-layer directions from the split ---
    print("[dir] building per-layer diff-of-means directions from split", flush=True)
    H = model.config.hidden_size
    sum_ref = np.zeros((nL + 1, H), np.float64)
    n_ref = 0
    sum_ben = np.zeros((nL + 1, H), np.float64)
    n_ben = 0
    t0 = time.time()
    for label, path in [("ref", args.refusal_csv), ("ben", args.benign_csv)]:
        prompts = read_prompts(path)
        print(f"[dir] {label}: {len(prompts)} prompts")
        for k, (_, stacked) in enumerate(last_token_all_layers(prompts)):
            arr = stacked.numpy().astype(np.float64).sum(axis=0)  # [nL+1, H]
            if label == "ref":
                sum_ref += arr
                n_ref += stacked.shape[0]
            else:
                sum_ben += arr
                n_ben += stacked.shape[0]
            if k % 40 == 0:
                print(f"  [dir:{label}] {k*args.batch_size} ({time.time()-t0:.0f}s)", flush=True)
    d = (sum_ref / n_ref) - (sum_ben / n_ben)          # [nL+1, H]
    dn = np.linalg.norm(d, axis=1, keepdims=True) + 1e-9

    # --- 2. project Claude unique texts ---
    pairs = read_pairs(args.pairs_csv)
    print(f"[proj] {len(pairs)} pairs; projecting unique texts", flush=True)
    uniq = list(dict.fromkeys([o for o, _, _, _ in pairs] + [r for _, r, _, _ in pairs]))
    proj = {}  # text -> [nL+1]
    t0 = time.time()
    for k, (batch, stacked) in enumerate(last_token_all_layers(uniq)):
        a = stacked.numpy().astype(np.float64)                 # [B, nL+1, H]
        pr = (a * (d / dn)[None, :, :]).sum(axis=2)            # [B, nL+1]
        for b, txt in enumerate(batch):
            proj[txt] = pr[b]
        if k % 40 == 0:
            print(f"  [proj] {k*args.batch_size}/{len(uniq)} ({time.time()-t0:.0f}s)", flush=True)

    # --- 3. per-layer correlation vs icannot delta ---
    ic = np.array([p[2] for p in pairs])
    sim = np.array([p[3] for p in pairs])
    hi = sim >= args.sim_hi
    layers = []
    for L in range(1, nL + 1):
        rd = np.array([proj[r][L] - proj[o][L] for o, r, _, _ in pairs])
        pear = float(np.corrcoef(rd, ic)[0, 1])
        spear = float(stats.spearmanr(rd, ic).statistic)
        pear_hi = float(np.corrcoef(rd[hi], ic[hi])[0, 1]) if hi.sum() > 2 else float("nan")
        spear_hi = float(stats.spearmanr(rd[hi], ic[hi]).statistic) if hi.sum() > 2 else float("nan")
        layers.append({"layer": L, "pearson": pear, "spearman": spear,
                       "pearson_hi": pear_hi, "spearman_hi": spear_hi})

    best = max(layers, key=lambda x: abs(x["pearson"]))
    best_hi = max(layers, key=lambda x: abs(x["pearson_hi"]) if x["pearson_hi"] == x["pearson_hi"] else 0)
    summary = {"n_pairs": len(pairs), "n_high_sim": int(hi.sum()), "sim_threshold": args.sim_hi,
               "n_ref": n_ref, "n_ben": n_ben, "layers": layers,
               "best_layer_pearson": best, "best_layer_pearson_high_sim": best_hi,
               "canonical_layer32": layers[31]}
    json.dump(summary, open(os.path.join(args.out_dir, "layer_sweep_summary.json"), "w"), indent=2)

    print("\n=========== per-layer corr(refusal-dir delta, 'I cannot' logprob delta) ===========")
    print(f"  {'layer':>5s} {'pearson':>9s} {'spearman':>9s} {'pearson_hi':>11s} {'spearman_hi':>12s}")
    for x in layers:
        mark = "  <- canonical(32)" if x["layer"] == 32 else ("  <- best" if x is best else "")
        print(f"  {x['layer']:5d} {x['pearson']:9.4f} {x['spearman']:9.4f} {x['pearson_hi']:11.4f} {x['spearman_hi']:12.4f}{mark}")
    print(f"\n  best layer (|pearson| ALL): L{best['layer']}  pearson {best['pearson']:.4f}")
    print(f"  best layer (|pearson| hi ): L{best_hi['layer']}  pearson_hi {best_hi['pearson_hi']:.4f}")
    print(f"  canonical L32: pearson {layers[31]['pearson']:.4f} (sanity vs prior ~0.20)")
    print("[done]")


if __name__ == "__main__":
    main()
