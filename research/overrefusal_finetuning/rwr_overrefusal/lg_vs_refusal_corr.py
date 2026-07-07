#!/usr/bin/env python3
"""Correlate the Llama-Guard classifier delta against our refusal-circuit delta,
over the Claude Haiku rewrites (the RWR training data).

For every (original, rewrite) pair in the FIXED Claude shards:
  lg_delta   = P_unsafe(rewrite) - P_unsafe(original)      [Llama-Guard-3-8B soft score]
  ref_delta  = refusal_delta already stored in the shards  [Llama-3-8B layer-32 refusal vector]

P_unsafe is the softmax over Llama-Guard's {safe, unsafe} verdict-token logits
(same method as label_prompts_llamaguard.py). ref_delta is NOT recomputed -- it
is the padding-fixed value from the shards.

Outputs (in --out_dir):
  lg_vs_refusal_pairs.csv   one row per pair: refusal_delta, lg_delta, similarity, p_unsafe_orig/rewrite, texts
  lg_vs_refusal_summary.json  correlations + distribution stats
Prints Pearson r and Spearman rho (full set + high-similarity subset).
"""
import argparse
import csv
import glob
import json
import os
import time
import numpy as np


def load_pairs(shard_dir):
    pairs = []  # (original, rewrite, refusal_delta, similarity)
    for f in sorted(glob.glob(os.path.join(shard_dir, "*.json"))):
        for rec in json.load(open(f)):
            orig = rec["original"]
            for p in rec.get("paraphrases", []):
                if "refusal_delta" in p and "similarity" in p and "paraphrase" in p:
                    pairs.append((orig, p["paraphrase"], float(p["refusal_delta"]), float(p["similarity"])))
    return pairs


def score_p_unsafe(texts, model, tok, safe_id, unsafe_id, batch_size, max_new_tokens=8):
    """Return {text: p_unsafe} via Llama-Guard moderation template + softmax over safe/unsafe."""
    import torch
    out = {}
    uniq = list(dict.fromkeys(texts))
    t0 = time.time()
    for i in range(0, len(uniq), batch_size):
        batch = uniq[i:i + batch_size]
        chat = [tok.apply_chat_template([{"role": "user", "content": p}],
                                        tokenize=False, add_generation_prompt=True) for p in batch]
        enc = tok(chat, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 output_scores=True, return_dict_in_generate=True,
                                 pad_token_id=tok.pad_token_id)
        seq = gen.sequences[:, enc["input_ids"].shape[1]:]
        scores = gen.scores
        for b, p in enumerate(batch):
            pu = None
            for step in range(min(len(scores), seq.shape[1])):
                tid = int(seq[b, step])
                if tid in (safe_id, unsafe_id):
                    lg = scores[step][b]
                    pair = torch.softmax(torch.stack([lg[safe_id], lg[unsafe_id]]), dim=0)
                    pu = float(pair[1])
                    break
            if pu is None:  # fallback: hard label from decoded text
                txt = tok.decode(seq[b], skip_special_tokens=True).lower()
                pu = 1.0 if "unsafe" in txt else 0.0
            out[p] = pu
        if (i // batch_size) % 20 == 0:
            done = min(i + batch_size, len(uniq))
            rate = done / max(time.time() - t0, 1e-3)
            print(f"  [lg] {done}/{len(uniq)} unique texts ({rate:.1f}/s)", flush=True)
    return out


def corr(a, b):
    from scipy import stats
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    pear = float(np.corrcoef(a, b)[0, 1])
    spear = float(stats.spearmanr(a, b).statistic)
    return pear, spear


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard_dir", required=True)
    ap.add_argument("--out_dir", default="prompt_iteration_results/lg_vs_refusal")
    ap.add_argument("--lg_model", default="meta-llama/Llama-Guard-3-8B")
    ap.add_argument("--batch_size", type=int, default=24)
    ap.add_argument("--sim_hi", type=float, default=0.85)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    pairs = load_pairs(args.shard_dir)
    print(f"[data] {len(pairs)} (original, rewrite) pairs from {args.shard_dir}")

    hf = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    print(f"[lg] loading {args.lg_model}")
    tok = AutoTokenizer.from_pretrained(args.lg_model, token=hf)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.lg_model, token=hf, device_map="auto",
                                                 torch_dtype=torch.bfloat16)
    model.eval()
    safe_id = tok.encode("safe", add_special_tokens=False)[0]
    unsafe_id = tok.encode("unsafe", add_special_tokens=False)[0]
    print(f"[lg] safe_id={safe_id} unsafe_id={unsafe_id}")

    texts = [o for o, _, _, _ in pairs] + [r for _, r, _, _ in pairs]
    pu = score_p_unsafe(texts, model, tok, safe_id, unsafe_id, args.batch_size)

    # build aligned arrays
    ref_d, lg_d, sim, rows = [], [], [], []
    for o, r, rd, s in pairs:
        po, pr = pu.get(o), pu.get(r)
        if po is None or pr is None:
            continue
        ld = pr - po
        ref_d.append(rd)
        lg_d.append(ld)
        sim.append(s)
        rows.append((rd, ld, s, po, pr, o, r))

    csv_path = os.path.join(args.out_dir, "lg_vs_refusal_pairs.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["refusal_delta", "lg_delta", "similarity", "p_unsafe_orig", "p_unsafe_rewrite", "original", "rewrite"])
        for row in rows:
            w.writerow(row)

    ref_d = np.array(ref_d)
    lg_d = np.array(lg_d)
    sim = np.array(sim)
    pear, spear = corr(ref_d, lg_d)
    hi = sim >= args.sim_hi
    pear_hi, spear_hi = corr(ref_d[hi], lg_d[hi]) if hi.sum() > 2 else (float("nan"), float("nan"))

    summary = {
        "n_pairs": int(len(ref_d)),
        "pearson_r": pear, "spearman_rho": spear,
        "n_high_sim": int(hi.sum()), "sim_threshold": args.sim_hi,
        "pearson_r_high_sim": pear_hi, "spearman_rho_high_sim": spear_hi,
        "refusal_delta": {"mean": float(ref_d.mean()), "median": float(np.median(ref_d)),
                          "pct_positive": float(100 * (ref_d > 0).mean())},
        "lg_delta": {"mean": float(lg_d.mean()), "median": float(np.median(lg_d)),
                     "pct_positive": float(100 * (lg_d > 0).mean())},
        "sign_agreement_pct": float(100 * (np.sign(ref_d) == np.sign(lg_d)).mean()),
    }
    json.dump(summary, open(os.path.join(args.out_dir, "lg_vs_refusal_summary.json"), "w"), indent=2)

    print("\n================ Llama-Guard delta  vs  refusal-circuit delta ================")
    print(f"  n pairs: {summary['n_pairs']}")
    print(f"  Pearson  r   = {pear:.4f}     Spearman rho = {spear:.4f}   (ALL pairs)")
    print(f"  Pearson  r   = {pear_hi:.4f}     Spearman rho = {spear_hi:.4f}   (similarity >= {args.sim_hi}, n={int(hi.sum())})")
    print(f"  refusal_delta: mean {ref_d.mean():.3f}  median {np.median(ref_d):.3f}  {summary['refusal_delta']['pct_positive']:.1f}% positive")
    print(f"  lg_delta    : mean {lg_d.mean():.4f}  median {np.median(lg_d):.4f}  {summary['lg_delta']['pct_positive']:.1f}% positive")
    print(f"  sign agreement: {summary['sign_agreement_pct']:.1f}%")
    print(f"[done] wrote {csv_path} and summary json")
    try:
        model.cpu()
    except Exception:
        pass


if __name__ == "__main__":
    main()
