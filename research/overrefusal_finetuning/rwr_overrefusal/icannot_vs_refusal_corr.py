#!/usr/bin/env python3
"""Correlate, over the Claude Haiku rewrites, two Llama-3-8B signals (both the
DIFFERENCE between rewrite and original):

  ref_delta = refusal_vector_activation(rewrite) - refusal_vector_activation(original)
              (canonical ORRewardModel score: normalized dot of layer-32 last-token
               hidden state with the refusal direction)
  ic_delta  = logP("I cannot" | rewrite) - logP("I cannot" | original)
              (teacher-forced log-prob that the assistant response BEGINS with
               "I cannot" -- the dominant refusal template, ~94% of refusals)

Both computed on the IDENTICAL prompt formatting (system="You are a helpful
assistant.", add_generation_prompt=True) from the same model instance, so the two
deltas are directly comparable. Also reports the probability-space delta.

Outputs (--out_dir):
  icannot_vs_refusal_pairs.csv     per-pair signals
  icannot_vs_refusal_summary.json  correlations + stats
"""
import argparse
import csv
import glob
import json
import os
import sys
import time
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS, "..", "ppo_or"))


def load_pairs(shard_dir):
    pairs = []
    for f in sorted(glob.glob(os.path.join(shard_dir, "*.json"))):
        for rec in json.load(open(f)):
            o = rec["original"]
            for p in rec.get("paraphrases", []):
                if "refusal_delta" in p and "similarity" in p and "paraphrase" in p:
                    pairs.append((o, p["paraphrase"], float(p["refusal_delta"]), float(p["similarity"])))
    return pairs


def format_prompt(rm, p):
    if rm._use_system_role:
        msgs = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": p}]
    else:
        msgs = [{"role": "user", "content": p}]
    return rm._target_tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def icannot_logprob(rm, texts, cont_ids, batch_size, max_prefix=512):
    """Teacher-forced sum log-prob of the continuation cont_ids ('I cannot') for each text."""
    import torch
    tok = rm._target_tokenizer
    model = rm._target_model
    C = len(cont_ids)
    out = {}
    uniq = list(dict.fromkeys(texts))
    t0 = time.time()
    for i in range(0, len(uniq), batch_size):
        batch = uniq[i:i + batch_size]
        prefix_ids = []
        for p in batch:
            ids = tok(format_prompt(rm, p), add_special_tokens=False,
                      truncation=True, max_length=max_prefix)["input_ids"]
            prefix_ids.append(ids)
        seqs = [pi + list(cont_ids) for pi in prefix_ids]
        plen = [len(pi) for pi in prefix_ids]
        L = max(len(s) for s in seqs)
        pad_id = tok.pad_token_id
        input_ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
        attn = torch.zeros((len(seqs), L), dtype=torch.long)
        for b, s in enumerate(seqs):              # RIGHT padding -> arange positions are correct
            input_ids[b, :len(s)] = torch.tensor(s, dtype=torch.long)
            attn[b, :len(s)] = 1
        input_ids = input_ids.to(model.device)
        attn = attn.to(model.device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attn, use_cache=False).logits
        logp = torch.log_softmax(logits.float(), dim=-1)
        for b, p in enumerate(batch):
            P = plen[b]
            tot = 0.0
            for j in range(C):
                tot += float(logp[b, P - 1 + j, cont_ids[j]])
            out[p] = tot
        if rm._hook is not None:
            rm._hook.clear()
        if (i // batch_size) % 20 == 0:
            done = min(i + batch_size, len(uniq))
            print(f"  [ic] {done}/{len(uniq)} ({done/max(time.time()-t0,1e-3):.1f}/s)", flush=True)
    return out


def refusal_scores(rm, texts, batch_size):
    out = {}
    uniq = list(dict.fromkeys(texts))
    t0 = time.time()
    for i in range(0, len(uniq), batch_size):
        batch = uniq[i:i + batch_size]
        sc = rm._compute_refusal_scores(batch)  # [B]
        for b, p in enumerate(batch):
            out[p] = float(sc[b])
        if (i // batch_size) % 20 == 0:
            done = min(i + batch_size, len(uniq))
            print(f"  [ref] {done}/{len(uniq)} ({done/max(time.time()-t0,1e-3):.1f}/s)", flush=True)
    return out


def corr(a, b):
    from scipy import stats
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1]), float(stats.spearmanr(a, b).statistic)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard_dir", required=True)
    ap.add_argument("--vector", required=True)
    ap.add_argument("--out_dir", default="prompt_iteration_results/icannot_vs_refusal")
    ap.add_argument("--base_model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--activation_layer", type=int, default=32)
    ap.add_argument("--batch_size", type=int, default=24)
    ap.add_argument("--sim_hi", type=float, default=0.85)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    from config import ModelConfig, RewardConfig
    from reward_model import ORRewardModel

    pairs = load_pairs(args.shard_dir)
    print(f"[data] {len(pairs)} (original, rewrite) pairs")

    mc = ModelConfig(base_model_id=args.base_model, refusal_vector_path=args.vector,
                     activation_layer=args.activation_layer, quantize_target_model=False)
    rm = ORRewardModel(mc, RewardConfig())
    rm._ensure_target_model()
    rm._ensure_refusal_vector()
    tok = rm._target_tokenizer
    cont_ids = tok.encode("I cannot", add_special_tokens=False)
    print(f"[ic] 'I cannot' -> token ids {cont_ids} -> {[tok.decode([t]) for t in cont_ids]}")

    texts = [o for o, _, _, _ in pairs] + [r for _, r, _, _ in pairs]
    print("[stage] computing refusal-vector scores...")
    ref = refusal_scores(rm, texts, args.batch_size)
    print("[stage] computing 'I cannot' log-probs...")
    ic = icannot_logprob(rm, texts, cont_ids, args.batch_size)

    ref_d, ic_d_lp, ic_d_p, sim, stored_d, rows = [], [], [], [], [], []
    for o, r, srd, s in pairs:
        rd = ref[r] - ref[o]
        ilp = ic[r] - ic[o]                       # log-prob delta
        ip = float(np.exp(ic[r]) - np.exp(ic[o])) # probability delta
        ref_d.append(rd)
        ic_d_lp.append(ilp)
        ic_d_p.append(ip)
        sim.append(s)
        stored_d.append(srd)
        rows.append((rd, srd, ilp, ip, s, ic[o], ic[r], o, r))

    csv_path = os.path.join(args.out_dir, "icannot_vs_refusal_pairs.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ref_delta_recomputed", "ref_delta_stored", "icannot_delta_logprob",
                    "icannot_delta_prob", "similarity", "icannot_logprob_orig",
                    "icannot_logprob_rewrite", "original", "rewrite"])
        for row in rows:
            w.writerow(row)

    ref_d = np.array(ref_d)
    ic_d_lp = np.array(ic_d_lp)
    ic_d_p = np.array(ic_d_p)
    sim = np.array(sim)
    stored_d = np.array(stored_d)
    hi = sim >= args.sim_hi

    p_lp, s_lp = corr(ref_d, ic_d_lp)
    p_p, s_p = corr(ref_d, ic_d_p)
    p_lp_hi, s_lp_hi = corr(ref_d[hi], ic_d_lp[hi]) if hi.sum() > 2 else (float("nan"),) * 2
    p_sanity, s_sanity = corr(ref_d, stored_d)  # recomputed vs stored refusal_delta (should be ~1)

    summary = {
        "n_pairs": int(len(ref_d)),
        "corr_ref_vs_icannot_logprob": {"pearson": p_lp, "spearman": s_lp},
        "corr_ref_vs_icannot_prob":    {"pearson": p_p,  "spearman": s_p},
        "corr_high_sim_logprob": {"pearson": p_lp_hi, "spearman": s_lp_hi,
                                  "n": int(hi.sum()), "sim_threshold": args.sim_hi},
        "sanity_recomputed_vs_stored_refusal_delta": {"pearson": p_sanity, "spearman": s_sanity},
        "icannot_delta_logprob": {"mean": float(ic_d_lp.mean()), "median": float(np.median(ic_d_lp)),
                                  "pct_positive": float(100 * (ic_d_lp > 0).mean())},
        "ref_delta": {"mean": float(ref_d.mean()), "median": float(np.median(ref_d)),
                      "pct_positive": float(100 * (ref_d > 0).mean())},
        "sign_agreement_pct": float(100 * (np.sign(ref_d) == np.sign(ic_d_lp)).mean()),
    }
    json.dump(summary, open(os.path.join(args.out_dir, "icannot_vs_refusal_summary.json"), "w"), indent=2)

    print("\n============ refusal-vector delta  vs  'I cannot' logprob delta ============")
    print(f"  n pairs: {summary['n_pairs']}")
    print(f"  Pearson r = {p_lp:.4f}   Spearman rho = {s_lp:.4f}   (ALL, log-prob delta)")
    print(f"  Pearson r = {p_lp_hi:.4f}   Spearman rho = {s_lp_hi:.4f}   (sim>={args.sim_hi}, n={int(hi.sum())})")
    print(f"  Pearson r = {p_p:.4f}   Spearman rho = {s_p:.4f}   (ALL, probability delta)")
    print(f"  sign agreement: {summary['sign_agreement_pct']:.1f}%")
    print(f"  [sanity] recomputed vs stored refusal_delta: pearson {p_sanity:.4f} (expect ~1.0)")
    print(f"[done] wrote {csv_path}")
    rm.cleanup()


if __name__ == "__main__":
    main()
