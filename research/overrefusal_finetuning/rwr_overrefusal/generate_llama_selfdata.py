#!/usr/bin/env python3
"""Generate Llama-3-8B SELF-paraphrases as RWR training data (self-distillation arm).

Answers: "does reward-weighted self-distillation beat the base policy?" Every prior
RWR dataset came from a *teacher* (gpt-oss-120b / Claude / Gemini / gpt-4o-mini).
This builds the missing arm: the base Llama-3-8B-Instruct policy (with the SAME RWR
system prompt the `baseline` eval arm uses) generates its own candidate paraphrases,
they are reward-scored, and the top reward-bin trains a new RWR adapter. So `baseline`
literally generates this adapter's training data — a true self-distillation loop.

Design is held identical to claude_rwr / rwr_v3 so the 4-way comparison is clean:
the ONLY thing that changes vs those runs is the paraphrase generator (Llama itself).

  - Prompts: the same 2500 dolly + 2500 alpaca TRAINING prompts (seed=42), reconstructed
    with the identical selection logic used by generate_claude_dataset.py — which also
    guarantees they are disjoint from the seed=99 held-out eval set.
  - Generation: rwr_config.SYSTEM_PROMPT + PROMPT_TEMPLATE (same as the baseline arm),
    K samples/prompt, sampled at temperature for candidate diversity.
  - Scoring: ORRewardModel at k=5.0/c=0.75/d=100 (same scale as everything else).
  - Output: shards in the or_susceptibility_rankings_shard*.json schema that
    rwr_data.load_shards() consumes, so train_rwr.py runs unchanged.

GPU job (no internet): see run_llama_selfdata_clusterb.slurm.
"""
import argparse
import csv
import json
import os
import random
import sys
import time
from typing import Dict, List, Tuple

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, os.path.join(THIS_DIR, "..", "ppo_or"))

from eval_rwr import load_generator, generate_paraphrases, score_generations  # noqa: E402


def load_alpaca_training_prompts(training_seed: int, n: int) -> List[Tuple[str, str]]:
    """Mirror generate_claude_dataset.py / eval_held_out._alpaca_training_indices
    (rng = Random(seed+1)). Returns (id, prompt) for the alpaca TRAINING selection."""
    from datasets import load_dataset
    ds = load_dataset("yahma/alpaca-cleaned", split="train")
    rng = random.Random(training_seed + 1)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    out: List[Tuple[str, str]] = []
    seen = set()
    for idx in indices:
        ex = ds[idx]
        inst = (ex.get("instruction") or "").strip()
        inp = (ex.get("input") or "").strip()
        if not inst:
            continue
        prompt = f"{inst}\n\n{inp}" if inp else inst
        if len(prompt) > 1500:
            continue
        if idx in seen:
            continue
        seen.add(idx)
        out.append((f"alpaca:{idx}", prompt))
        if len(out) >= n:
            break
    print(f"[data] alpaca training prompts: {len(out)} (seed={training_seed}+1)")
    return out


def load_dolly_training_prompts(csv_path: str, training_seed: int, n: int) -> List[Tuple[str, str]]:
    """Mirror eval_held_out._dolly_training_ids (rng = Random(seed), shuffle rows, take n)."""
    rows: List[Tuple[int, str]] = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                rid = int(r["id"])
            except (KeyError, ValueError):
                continue
            text = (r.get("prompt") or "").strip()
            if not text or len(text) > 1500:
                continue
            rows.append((rid, text))
    rng = random.Random(training_seed)
    rng.shuffle(rows)
    out = [(f"dolly:{rid}", text) for rid, text in rows[:n]]
    print(f"[data] dolly training prompts: {len(out)} (seed={training_seed})")
    return out


def write_shards(scored: List[Dict], prompt_ids: Dict[str, str], out_dir: str,
                 items_per_shard: int = 50) -> int:
    """Group scored (original, generation) pairs by original prompt and write shards
    in the schema rwr_data.load_shards() expects."""
    os.makedirs(out_dir, exist_ok=True)
    by_orig: Dict[str, List[Dict]] = {}
    for s in scored:
        by_orig.setdefault(s["original"], []).append({
            "paraphrase":    s["generation"],
            "or_score_raw":  s["or_score_raw"],
            "refusal_delta": s["refusal_delta"],
            "similarity":    s["similarity"],
        })

    items = []
    for orig, paras in by_orig.items():
        ors = [p["or_score_raw"] for p in paras]
        sims = [p["similarity"] for p in paras]
        items.append({
            "prompt_idx":   prompt_ids.get(orig, ""),
            "original":     orig,
            "paraphrases":  paras,
            "n_paraphrases": len(paras),
            "n_positive":   sum(1 for o in ors if o > 0),
            "mean_or":      sum(ors) / len(ors),
            "max_or":       max(ors),
            "mean_sim":     sum(sims) / len(sims),
        })

    n_shards = 0
    for i in range(0, len(items), items_per_shard):
        shard = items[i:i + items_per_shard]
        path = os.path.join(out_dir, f"or_susceptibility_rankings_shard{n_shards}.json")
        with open(path, "w") as f:
            json.dump(shard, f, indent=2)
        n_shards += 1
    print(f"[shards] wrote {len(items)} items across {n_shards} shards -> {out_dir}")
    return n_shards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--refusal_vector_path",
                    default="../../refusal_vector/Vector_Extraction/refusal_vector.layer032.npz")
    ap.add_argument("--dolly_csv", default="../../overrefusal_sampling/data/benign_dolly.csv")
    ap.add_argument("--out_dir", default="../llama_selfparaphrase_5k")
    ap.add_argument("--n_alpaca", type=int, default=2500)
    ap.add_argument("--n_dolly", type=int, default=2500)
    ap.add_argument("--n_per_prompt", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="Higher than the 0.7 eval temp: we want diverse candidates so the "
                         "reward-weighted top bin has signal to mine.")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--training_seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="If >0, only first N prompts (smoke test).")
    ap.add_argument("--gen_chunk", type=int, default=500,
                    help="Generate/score in chunks of this many prompts, checkpointing shards "
                         "after each chunk so a timeout doesn't lose everything.")
    args = ap.parse_args()

    prompts: List[Tuple[str, str]] = []
    if args.n_alpaca > 0:
        prompts += load_alpaca_training_prompts(args.training_seed, args.n_alpaca)
    if args.n_dolly > 0:
        prompts += load_dolly_training_prompts(args.dolly_csv, args.training_seed, args.n_dolly)
    if args.limit > 0:
        prompts = prompts[:args.limit]
    prompt_ids = {text: pid for pid, text in prompts}
    print(f"[data] total training prompts: {len(prompts)}  (K={args.n_per_prompt} -> "
          f"{len(prompts)*args.n_per_prompt} candidate generations)")

    print(f"[gen] loading base policy (no adapter): {args.base_model}")
    t0 = time.time()
    model, tokenizer = load_generator(args.base_model, adapter_dir=None)
    print(f"[gen] model loaded in {time.time()-t0:.0f}s")

    # Generate (and score) in chunks, accumulating scored pairs and checkpointing shards.
    all_scored: List[Dict] = []
    prompt_texts = [t for _, t in prompts]
    for ci in range(0, len(prompt_texts), args.gen_chunk):
        chunk = prompt_texts[ci:ci + args.gen_chunk]
        tg = time.time()
        gens = generate_paraphrases(
            model, tokenizer, chunk,
            n_per_prompt=args.n_per_prompt,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
        )
        print(f"[gen] chunk {ci}-{ci+len(chunk)}: generated in {time.time()-tg:.0f}s")
        ts = time.time()
        scored = score_generations(gens, args.refusal_vector_path, args.base_model)
        all_scored.extend(scored)
        print(f"[score] chunk {ci}-{ci+len(chunk)}: scored {len(scored)} pairs in {time.time()-ts:.0f}s "
              f"(total {len(all_scored)})")
        # checkpoint shards after each chunk
        write_shards(all_scored, prompt_ids, args.out_dir)

    # Final write + a tiny summary
    write_shards(all_scored, prompt_ids, args.out_dir)
    ors = [s["or_score_raw"] for s in all_scored]
    pos = sum(1 for o in ors if o > 0)
    print(f"\n[done] {len(all_scored)} scored pairs; {pos} positive ({pos/max(len(ors),1):.1%}); "
          f"mean_or={sum(ors)/max(len(ors),1):.4f}")
    print(f"[done] shards in {args.out_dir} — train with:\n"
          f"  python train_rwr.py --shard_dir {args.out_dir} "
          f"--output_dir <ckpt> --bin_weights 0,0,0,1,16")


if __name__ == "__main__":
    main()
