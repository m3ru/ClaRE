#!/usr/bin/env python3
"""Flatten U1 shards into the two CSVs the scorer needs: originals, and rewrites.

Both go through the SAME scorer with the same settings, so 'original refused' and 'rewrite
not refused' are measured by one instrument rather than two.
"""
import csv, glob, json, os, sys
csv.field_size_limit(sys.maxsize)
gen_dir, out_dir = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)
recs = []
for f in sorted(glob.glob(os.path.join(gen_dir, "*.json"))):
    recs += json.load(open(f))
seen_o, orows, rrows = set(), [], []
for r in recs:
    o = (r.get("original") or "").strip()
    if not o:
        continue
    if o not in seen_o:
        seen_o.add(o)
        # scorer reads the column 'rewrite'; for originals we score the original itself
        orows.append({"orig_idx": r.get("orig_idx", ""), "original": o, "rewrite": o})
    for w in r.get("rewrites", []):
        w = (w or "").strip()
        if w and w != o:
            rrows.append({"orig_idx": r.get("orig_idx", ""), "original": o, "rewrite": w})
for name, rows in (("ur_originals.csv", orows), ("ur_rewrites.csv", rrows)):
    p = os.path.join(out_dir, name)
    with open(p, "w", newline="") as f:
        w_ = csv.DictWriter(f, fieldnames=["orig_idx", "original", "rewrite"])
        w_.writeheader(); w_.writerows(rows)
    print(f"wrote {p}: {len(rows)} rows")
