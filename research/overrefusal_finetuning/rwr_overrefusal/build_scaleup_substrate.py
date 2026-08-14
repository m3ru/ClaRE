#!/usr/bin/env python3
"""Corpus CSV -> refusal_atlas substrate CSV(s), model-agnostic.

Emits paired rows (original is_rewrite=0, rewrite is_rewrite=1) linked by pair_id, with
source='sonnet_pair' so score_signals.py's pair-delta path fires unmodified. gold_benign is a
PLACEHOLDER (orig=1) here; the real benign label is joined in from the benign-intent filter at
analysis time. Optional --num_shards splits BY PAIR (both sides same shard) to parallelize the
atlas behavioral pass as a SLURM array; concat the per-prompt CSVs afterward.
"""
import argparse, csv, os, sys
csv.field_size_limit(sys.maxsize)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus_csv", required=True)
    ap.add_argument("--out_prefix", required=True, help="e.g. .../substrate_scaleup ; writes _shardNN.csv or .csv")
    ap.add_argument("--source", default="sonnet_pair")
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--max_pairs", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.corpus_csv)))
    if args.max_pairs:
        rows = rows[:args.max_pairs]
    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)
    FIELDS = ["prompt_id", "text", "source", "gold_benign", "native_topic",
              "pair_id", "is_rewrite", "probe_delta", "similarity"]

    writers, files = [], []
    for s in range(args.num_shards):
        path = f"{args.out_prefix}.csv" if args.num_shards == 1 else f"{args.out_prefix}_shard{s:02d}.csv"
        fh = open(path, "w", newline=""); files.append(path)
        w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader()
        writers.append((fh, w))

    # Shard BY ORIGINAL (dense group ids) so an original + all its rewrites land in one shard:
    # avoids scoring the same original redundantly in every shard.
    groups = {}
    def shard_of(oidx):
        if oidx not in groups:
            groups[oidx] = len(groups)
        return groups[oidx] % args.num_shards

    n = 0
    for i, r in enumerate(rows):
        pid = r["pair_id"]; sim = r.get("similarity", "")
        w = writers[shard_of(r.get("orig_idx", pid))][1]
        w.writerow(dict(prompt_id=f"{pid}_o", text=r["original"], source=args.source,
                        gold_benign="1", native_topic="benign_source", pair_id=pid,
                        is_rewrite="0", probe_delta="", similarity=sim))
        w.writerow(dict(prompt_id=f"{pid}_r", text=r["rewrite"], source=args.source,
                        gold_benign="", native_topic="manipulated", pair_id=pid,
                        is_rewrite="1", probe_delta="", similarity=sim))
        n += 1
    for fh, _ in writers:
        fh.close()
    print(f"[done] {n} pairs -> {len(files)} substrate file(s): {files}", flush=True)


if __name__ == "__main__":
    main()
