#!/usr/bin/env python3
"""Score the generated alpaca OR adaptations with the Llama-3 OR score (fixed
scorer), take the top 10% by OR score, and analyze the most common words.

OR score = exp(k*(sim-c)) * (refusal(rewrite) - refusal(original)) / d, using
Meta-Llama-3-8B-Instruct activations at layer 32 projected on the Llama-3
refusal vector, and MiniLM similarity. k/c/d default to the claude_rwr
training-consistent scale (5.0, 0.75, 100). Uses the padding-fixed reward_model.
"""
import argparse
import glob
import json
import os
import re
import sys
import time
from collections import Counter

import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS, "..", "ppo_or"))

STOP = set("""a an the and or but if then else of to in on for with without as at by from into over under
again further once here there all any both each few more most other some such no nor not only own same so than too very
is are was were be been being have has had do does did doing will would shall should can could may might must this that these those
i you he she it we they them his her its our your their what which who whom whose how when where why me my mine you your yours
about above below up down out off through during before after between within along across also use used using please provide
given following based your you it's i'm can't don't""".split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_dir", required=True)
    ap.add_argument("--vector", required=True)
    ap.add_argument("--base_model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--activation_layer", type=int, default=32)
    ap.add_argument("--k", type=float, default=5.0)
    ap.add_argument("--c", type=float, default=0.75)
    ap.add_argument("--d", type=float, default=100.0)
    ap.add_argument("--top_frac", type=float, default=0.10)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--output", required=True)
    ap.add_argument("--top_words", type=int, default=60)
    args = ap.parse_args()

    # ---- load all completed generation shards ----
    files = sorted(glob.glob(os.path.join(args.gen_dir, "gen_alpaca_or_shard*.json")))
    files = [f for f in files if not f.endswith(".tmp")]
    pairs = []  # (alpaca_idx, original, rewrite)
    for f in files:
        try:
            for rec in json.load(open(f)):
                for rw in rec.get("rewrites", []):
                    if rw and rw.strip():
                        pairs.append((rec["alpaca_idx"], rec["original"], rw.strip()))
        except Exception as e:
            print(f"[warn] skip {f}: {e}")
    print(f"[score] loaded {len(pairs)} (original,rewrite) pairs from {len(files)} shards", flush=True)
    if not pairs:
        print("[score] nothing to score")
        return

    from config import ModelConfig, RewardConfig
    from reward_model import ORRewardModel
    mc = ModelConfig(base_model_id=args.base_model, refusal_vector_path=args.vector,
                     activation_layer=args.activation_layer, quantize_target_model=False)
    rc = RewardConfig(similarity_exponent=args.k, similarity_center=args.c, refusal_divisor=args.d)
    rm = ORRewardModel(mc, rc)
    rm._ensure_target_model()
    rm._ensure_refusal_vector()
    rm._ensure_similarity_model()

    def refusal_scores(texts):
        out = np.empty(len(texts), dtype=np.float64)
        t0 = time.time()
        for i in range(0, len(texts), args.chunk):
            s = rm._compute_refusal_scores(texts[i:i + args.chunk])
            out[i:i + len(s)] = s.detach().float().cpu().numpy()
            if (i // args.chunk) % 50 == 0:
                print(f"[score]  refusal {i}/{len(texts)}  {i/max(time.time()-t0,1e-3):.1f}/s", flush=True)
        return out

    uniq = sorted({o for _, o, _ in pairs})
    uo = {o: k for k, o in enumerate(uniq)}
    print(f"[score] scoring {len(uniq)} unique originals + {len(pairs)} rewrites...", flush=True)
    orig_sc = refusal_scores(uniq)
    rew_sc = refusal_scores([r for _, _, r in pairs])
    # similarity per pair (MiniLM, batched on CPU)
    sims = np.empty(len(pairs), dtype=np.float64)
    for i in range(0, len(pairs), 512):
        b = pairs[i:i + 512]
        s = rm._compute_similarity([o for _, o, _ in b], [r for _, _, r in b])
        sims[i:i + len(s)] = s.detach().float().cpu().numpy()

    o_sc = np.array([orig_sc[uo[o]] for _, o, _ in pairs])
    delta = rew_sc - o_sc
    or_score = np.exp(args.k * (sims - args.c)) * delta / args.d

    order = np.argsort(or_score)[::-1]
    n_top = max(1, int(round(args.top_frac * len(pairs))))
    top = order[:n_top]
    print(f"[score] top {args.top_frac:.0%} = {n_top} rewrites; "
          f"OR cutoff={or_score[top[-1]]:.4f}  max={or_score[top[0]]:.4f}", flush=True)

    # ---- word frequency: top-10% rewrites vs the rest ----
    def words(text):
        return [w for w in re.findall(r"[a-zA-Z][a-zA-Z'\-]+", text.lower())
                if w not in STOP and len(w) > 2]
    top_set = set(top.tolist())
    top_counter, rest_counter = Counter(), Counter()
    for i, (_, _, r) in enumerate(pairs):
        (top_counter if i in top_set else rest_counter).update(set(words(r)))  # per-rewrite presence
    n_rest = len(pairs) - n_top
    # distinctive: overrepresented in top vs rest (smoothed rate ratio, min freq)
    distinctive = []
    for w, ct in top_counter.items():
        if ct < max(5, n_top * 0.01):
            continue
        rate_top = ct / n_top
        rate_rest = (rest_counter.get(w, 0) + 1) / (n_rest + 1)
        distinctive.append((w, round(rate_top / rate_rest, 2), ct, round(100 * rate_top, 1)))
    distinctive.sort(key=lambda x: x[1], reverse=True)

    common = top_counter.most_common(args.top_words)
    res = {
        "n_pairs": len(pairs), "n_shards": len(files), "n_top": n_top,
        "top_frac": args.top_frac, "k": args.k, "c": args.c, "d": args.d,
        "or_score_stats": {"mean": float(or_score.mean()), "median": float(np.median(or_score)),
                           "p90": float(np.percentile(or_score, 90)),
                           "top_cutoff": float(or_score[top[-1]]), "max": float(or_score[top[0]])},
        "pct_positive_OR": float(100 * (or_score > 0).mean()),
        "most_common_words_in_top10pct": [[w, c] for w, c in common],
        "most_distinctive_words_top_vs_rest": [list(x) for x in distinctive[:args.top_words]],
        "sample_top_rewrites": [{"or": round(float(or_score[i]), 4),
                                 "original": pairs[i][1][:160], "rewrite": pairs[i][2][:200]}
                                for i in top[:25].tolist()],
    }
    json.dump(res, open(args.output, "w"), indent=2)
    print(f"\n[score] wrote {args.output}")
    print(f"[score] {len(pairs)} rewrites, {100*(or_score>0).mean():.1f}% positive OR")
    print(f"\n=== MOST COMMON WORDS in top {args.top_frac:.0%} (by # rewrites containing) ===")
    for w, ct in common[:40]:
        print(f"  {ct:6d}  {w}")
    print("\n=== MOST DISTINCTIVE words (top10% vs rest, rate ratio) ===")
    for w, ratio, ct, pct in distinctive[:30]:
        print(f"  {ratio:6.2f}x  {w}  (in {pct}% of top, n={ct})")
    rm.cleanup()


if __name__ == "__main__":
    main()
