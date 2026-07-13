#!/usr/bin/env python3
"""Score a freshly-generated rewrite pool (e.g. the Sonnet-5 gen_or_sonnet shards)
with icannot-OR. The gen shards carry {source_idx, source, original, rewrites:[str]}
with NO similarity/P yet, so this computes both:

  P("I cannot"|rewrite), P("I cannot"|original)   -- reuses the validated teacher-forcing
  similarity = MiniLM cosine(original, rewrite)
  or_score   = exp(18.4*(sim-0.75)) * P("I cannot"|rewrite)

Writes icannot_or_pairs.csv in the SAME schema as score_icannot_or.py, so it plugs
straight into build_icannot_shards.py and the RWR pipeline.
"""
import argparse
import csv
import glob
import json
import math
import os

from score_icannot_or import compute_icannot_logprobs   # validated 'I cannot' teacher-forcing

OR_K, OR_C = 18.4, 0.75


def load_gen_shards(gen_dir):
    pairs = []
    for f in sorted(glob.glob(os.path.join(gen_dir, "gen_or_sonnet_shard*.json"))):
        for rec in json.load(open(f)):
            o = rec.get("original")
            for rw in rec.get("rewrites", []):
                rw = (rw or "").strip()
                if o and rw:
                    pairs.append({"original": o, "rewrite": rw})
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--base_model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--batch_size", type=int, default=24)
    ap.add_argument("--max_prefix", type=int, default=512)
    ap.add_argument("--orig_flag_p", type=float, default=0.1)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    pairs = load_gen_shards(args.gen_dir)
    print(f"[data] {len(pairs)} (original, rewrite) pairs from {args.gen_dir}")
    if not pairs:
        raise SystemExit("no pairs found")

    # P('I cannot') for every unique original+rewrite (fills lp_orig / lp_rw)
    pairs = compute_icannot_logprobs(pairs, args.base_model, args.batch_size, args.max_prefix)

    # MiniLM cosine similarity per pair
    from sentence_transformers import SentenceTransformer
    minilm = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    origs = [p["original"] for p in pairs]
    rws = [p["rewrite"] for p in pairs]
    eo = minilm.encode(origs, convert_to_numpy=True, normalize_embeddings=True, batch_size=128, show_progress_bar=False)
    er = minilm.encode(rws, convert_to_numpy=True, normalize_embeddings=True, batch_size=128, show_progress_bar=False)
    sims = (eo * er).sum(axis=1)

    rows = []
    for p, s in zip(pairs, sims):
        p_rw = math.exp(p["lp_rw"])
        p_orig = math.exp(p["lp_orig"])
        s = float(s)
        rows.append({"or_score": math.exp(OR_K * (s - OR_C)) * p_rw, "p_rw": p_rw,
                     "p_orig": p_orig, "sim": s, "flag": int(p_orig > args.orig_flag_p),
                     "original": p["original"], "rewrite": p["rewrite"]})

    csv_path = os.path.join(args.out_dir, "icannot_or_pairs.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["or_score", "p_icannot_rewrite", "p_icannot_orig", "similarity",
                    "orig_refusal_prone", "stored_refusal_delta", "original", "rewrite"])
        for r in sorted(rows, key=lambda x: -x["or_score"]):
            w.writerow([f"{r['or_score']:.6g}", f"{r['p_rw']:.6g}", f"{r['p_orig']:.6g}",
                        f"{r['sim']:.4f}", r["flag"], "", r["original"], r["rewrite"]])

    prw = sorted(r["p_rw"] for r in rows)
    summary = {"n_pairs": len(rows),
               "p_icannot_rewrite": {"p90": prw[int(len(prw) * 0.9)], "p99": prw[int(len(prw) * 0.99)],
                                     "max": prw[-1], "n_gt_0.01": sum(x > 1e-2 for x in prw),
                                     "n_gt_0.1": sum(x > 1e-1 for x in prw)},
               "or_max": max(r["or_score"] for r in rows)}
    json.dump(summary, open(os.path.join(args.out_dir, "icannot_or_summary.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))
    print(f"[done] wrote {csv_path}")


if __name__ == "__main__":
    main()
