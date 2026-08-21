#!/usr/bin/env python3
"""Keep only near-minimal-edit rewrites, BEFORE any GPU or API spend.

The LOW (<=2 content-word edits) stratum is the one where causal attribution is direct --
a single word carries the refusal flip -- but it is rare: 4.2% of generated rewrites, and
only ~4% of those survive to a confirmed over-refusal. Powering it by scoring everything
would mean 128k target-model generations for ~200 usable pairs.

Word-level edit distance needs no model, so we compute it first and hand the target model
only the survivors. That is the whole trick: ~5k scored instead of ~128k, same yield.

Dedups on (original, rewrite) because shards can repeat a prompt, and drops rewrites
identical to their original (0 edits is a no-op, not a minimal pair).

Run: python filter_low_edit.py --gen_dir probe_or/results/gen_low_llama_logit \
        --max_edits 2 --out probe_or/results/low_power/low_candidates.csv
"""
import argparse, csv, glob, json, os, sys
csv.field_size_limit(sys.maxsize)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_edit_distance import pair_metrics

FIELDS = ["pair_id", "orig_idx", "original", "rewrite", "wl_dist_content", "norm_dist_content",
          "introduced_words", "removed_words", "n_content_orig"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_dir", required=True)
    ap.add_argument("--max_edits", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude_gen_dirs", nargs="*", default=[],
                    help="generation dirs whose originals must NOT appear here; Alpaca contains "
                         "a few exact-duplicate prompts at different indices, so index-disjoint "
                         "sampling can still repeat a TEXT. Drop those so held-out is exact.")
    a = ap.parse_args()

    excl = set()
    for d in a.exclude_gen_dirs:
        for f in glob.glob(os.path.join(d, "*.json")):
            for rec in json.load(open(f)):
                excl.add((rec.get("original") or "").strip())
    if excl:
        print(f"[filter] excluding {len(excl)} originals seen in prior generations")

    files = sorted(glob.glob(os.path.join(a.gen_dir, "*.json")))
    if not files:
        raise SystemExit(f"no shards in {a.gen_dir}")
    seen, kept, n_all, n_noop, n_excl = set(), [], 0, 0, 0
    hist = {}
    for f in files:
        for rec in json.load(open(f)):
            o = (rec.get("original") or "").strip()
            if o in excl:
                n_excl += 1
                continue
            for rw in rec.get("rewrites", []):
                rw = (rw or "").strip()
                if not rw or (o, rw) in seen:
                    continue
                seen.add((o, rw))
                n_all += 1
                m = pair_metrics(o, rw)
                d = m["wl_dist_content"]
                hist[d] = hist.get(d, 0) + 1
                if d == 0:
                    n_noop += 1
                    continue
                if d <= a.max_edits:
                    kept.append({"pair_id": f"low{len(kept):06d}",
                                 "orig_idx": rec.get("orig_idx", ""), "original": o, "rewrite": rw,
                                 **{k: m[k] for k in FIELDS[4:]}})
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(kept)
    print(f"[filter] {len(files)} shards | {n_all} unique rewrites")
    print(f"[filter] content-edit histogram (first 8): "
          f"{ {k: hist.get(k, 0) for k in range(8)} }")
    print(f"[filter] dropped {n_noop} zero-edit no-ops; {n_excl} originals excluded as already-seen")
    print(f"[filter] kept {len(kept)} with 1..{a.max_edits} content edits "
          f"({100*len(kept)/n_all:.2f}% of unique) over "
          f"{len({r['original'] for r in kept})} distinct originals -> {a.out}")


if __name__ == "__main__":
    main()
