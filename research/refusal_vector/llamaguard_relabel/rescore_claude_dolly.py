#!/usr/bin/env python3
"""Re-score the Claude Haiku dolly rewrites with Llama-Guard + the relabeled
vector, through the FIXED scorer (left padding + mask-based last-token gather).

Input: claude_dolly_rewrites.json — list of {prompt_idx, original,
paraphrases:[{paraphrase, old_refusal_delta, old_or, similarity}]} (old scores
are the buggy Llama-3 ones). Writes the same structure with lg_refusal_delta /
lg_or_score added, plus a summary vs the old scores.
"""
import argparse, json, os, sys
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS, "..", "..", "overrefusal_finetuning", "ppo_or"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--vector", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--scoring_model", default="meta-llama/Llama-Guard-3-8B")
    ap.add_argument("--layer", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    from config import ModelConfig, RewardConfig
    from reward_model import ORRewardModel

    data = json.load(open(args.input))
    mc = ModelConfig(base_model_id=args.scoring_model, refusal_vector_path=args.vector,
                     activation_layer=args.layer, quantize_target_model=False)  # bf16, match extraction
    rm = ORRewardModel(mc, RewardConfig())
    rc = rm.reward_config

    new_d, old_d = [], []
    for n, e in enumerate(data):
        paras = [p["paraphrase"] for p in e["paraphrases"]]
        for i in range(0, len(paras), args.batch_size):
            b = paras[i:i + args.batch_size]
            out = rm.compute_rewards([e["original"]] * len(b), b)
            for j in range(len(b)):
                p = e["paraphrases"][i + j]
                delta = out["refusal_delta"][j].item()
                p["lg_refusal_delta"] = delta
                p["lg_or_score"] = (out["sim_weight"][j] * out["refusal_delta"][j]
                                    / rc.refusal_divisor).item()
                p["lg_refusal_original"] = out["refusal_scores_original"][j].item()
                p["lg_refusal_paraphrase"] = out["refusal_scores_rewritten"][j].item()
                new_d.append(delta)
                old_d.append(p.get("old_refusal_delta"))
        if (n + 1) % 200 == 0:
            print(f"[rescore] {n+1}/{len(data)} prompts", flush=True)

    json.dump(data, open(args.output, "w"))
    new = np.array(new_d, dtype=np.float64)
    old = np.array([x for x in old_d], dtype=np.float64)
    print("\n==== Claude dolly rewrites re-scored ====")
    print("  n pairs: %d" % len(new))
    print("  Llama-Guard (FIXED) refusal_delta: %.1f%% positive  median %.4f  mean %.4f"
          % (100 * (new > 0).mean(), np.median(new), new.mean()))
    print("  OLD Llama-3 (buggy)  refusal_delta: %.1f%% positive  median %.4f"
          % (100 * (old > 0).mean(), np.median(old)))
    print("  corr(new Llama-Guard delta, old Llama-3 delta): %.3f"
          % float(np.corrcoef(new, old)[0, 1]))
    print("[done] wrote %s" % args.output)
    rm.cleanup()


if __name__ == "__main__":
    main()
