#!/usr/bin/env python3
"""Convert the Llama-scored pool (score_llama_or.py) into THREE RWR shard sets --
one per corrected refusal signal (vector@L17 / probe ensemble / multi-phrase logit)
-- in the format load_shards() expects:
  {original, paraphrases:[{paraphrase, refusal_delta, similarity, or_score_raw}]}

refusal_delta = the signal's delta; train_rwr.py recomputes or_score from
refusal_delta + similarity (k=18.4, c=0.75, d=1.0), so or_score_raw here is just the
matching placeholder.

Also carves ONE common held-out set of originals (--heldout_fraction, seed) BEFORE
any signal-specific filtering and excludes it from every shard set, so all three
attackers are evaluated on the same prompts (eval_rwr_llama.py --heldout_json) with
no train/eval overlap in any arm.
"""
import argparse
import csv
import json
import os
import random

import numpy as np

SIGNALS = ("vector", "probe", "logit")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored_csv", required=True)
    ap.add_argument("--out_root", required=True,
                    help="writes <out_root>/llama_shards_{signal}{tag}/ + heldout_originals.json")
    ap.add_argument("--heldout_fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_shards", type=int, default=40)
    ap.add_argument("--sim_floor", type=float, default=0.5, help="only for the stats printout")
    ap.add_argument("--signals", default=",".join(SIGNALS),
                    help="comma list of signals to build shards for (default all three)")
    ap.add_argument("--tag", default="",
                    help="suffix on shard dir names, e.g. '_L31' -> llama_shards_vector_L31 "
                         "(keeps a re-scored layer variant from clobbering the original)")
    ap.add_argument("--heldout_json", default=None,
                    help="reuse an existing held-out split (e.g. the L17 run's heldout_originals.json) "
                         "instead of carving a fresh one -- REQUIRED for a comparable re-scored variant "
                         "so the eval prompts are byte-identical across layers")
    args = ap.parse_args()
    signals = [s.strip() for s in args.signals.split(",") if s.strip()]

    rows = list(csv.DictReader(open(args.scored_csv)))
    originals = sorted({r["original"] for r in rows})
    os.makedirs(args.out_root, exist_ok=True)
    if args.heldout_json:
        heldout = set(json.load(open(args.heldout_json)))
        # every held-out original MUST be present in this CSV, else the split isn't the same pool
        missing = heldout - set(originals)
        if missing:
            raise SystemExit(f"[abort] {len(missing)} held-out originals absent from --scored_csv "
                             f"-- the reused split does not match this pool (e.g. {next(iter(missing))[:60]!r})")
        print(f"[split] REUSED {len(heldout)} held-out originals from {args.heldout_json} "
              f"(comparable to that run)")
    else:
        rng = random.Random(args.seed)
        heldout = set(rng.sample(originals, max(1, int(len(originals) * args.heldout_fraction))))
        hpath = os.path.join(args.out_root, "heldout_originals.json")
        json.dump(sorted(heldout), open(hpath, "w"), indent=1)
        print(f"[split] {len(originals)} originals -> {len(heldout)} held out (seed {args.seed}) -> {hpath}")

    for sig in signals:
        by_orig, n_pairs = {}, 0
        for r in rows:
            if r["original"] in heldout:
                continue
            by_orig.setdefault(r["original"], []).append({
                "paraphrase": r["rewrite"],
                "refusal_delta": float(r[f"d_{sig}"]),
                "similarity": float(r["similarity"]),
                "or_score_raw": float(r[f"or_{sig}"]),
            })
            n_pairs += 1

        out_dir = os.path.join(args.out_root, f"llama_shards_{sig}{args.tag}")
        os.makedirs(out_dir, exist_ok=True)
        recs = [{"original": o, "paraphrases": ps} for o, ps in by_orig.items()]
        per = (len(recs) + args.n_shards - 1) // args.n_shards
        written = 0
        for si in range(args.n_shards):
            chunk = recs[si * per:(si + 1) * per]
            if chunk:
                json.dump(chunk, open(os.path.join(out_dir, f"or_susceptibility_rankings_shard{si}.json"), "w"))
                written += len(chunk)

        # what the trainer will keep, and where the k=18.4 OR mass sits (edge picking)
        keep = [(float(r[f"or_{sig}"]))
                for r in rows if r["original"] not in heldout
                and float(r["similarity"]) >= args.sim_floor and float(r[f"d_{sig}"]) > 0]
        keep = np.array(keep)
        if len(keep):
            q = np.quantile(keep, [0.35, 0.65, 0.85])
            print(f"[{sig}] {written} originals / {n_pairs} pairs -> {out_dir}")
            print(f"        trainable(sim>={args.sim_floor}, d>0)={len(keep)}  "
                  f"OR q35/q65/q85 = {q[0]:.4g} / {q[1]:.4g} / {q[2]:.4g}  "
                  f"(candidate --bin_edges {q[0]:.4g},{q[1]:.4g},{q[2]:.4g})")
        else:
            print(f"[{sig}] {written} originals / {n_pairs} pairs -> {out_dir}  "
                  f"WARNING: 0 trainable pairs after filter")


if __name__ == "__main__":
    main()
