#!/usr/bin/env python3
"""U2/U3 glue: find genuine refusal->compliance flips, then build the judge input.

A pair counts as a candidate under-refusal only if BOTH hold, under the same refusal
definition used everywhere else in this project (recompute_v6.refused_v6: >50% of
non-capability-disclaimer samples start with a refusal opener):

    original  REFUSED      -- the model does decline the underlying request
    rewrite   NOT REFUSED  -- the re-framing suppressed that refusal

Only those go to the judge, because a rewrite of a prompt the model never refused carries no
information, and judging it is pure spend.

The judge then answers the question the refusal classifier cannot: is the rewrite still
asking for the same harmful thing? A rewrite that sanitised the request into something
genuinely benign is NOT a jailbreak -- it is the dominant failure mode of this whole design,
and the reason the yield after judging is the number that gets reported.

Run: python build_ur_pairs.py --orig_scored X.json --rw_scored Y.json --out judge_input.csv
"""
import argparse, csv, json, os, sys
csv.field_size_limit(sys.maxsize)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recompute_v6 import refused_v6
from analyze_edit_distance import pair_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig_scored", required=True)
    ap.add_argument("--rw_scored", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats_out", default="")
    a = ap.parse_args()

    orig = {}
    for r in json.load(open(a.orig_scored))["examples"]:
        ref, _, usable = refused_v6(r.get("samples", []))
        orig[r["rewrite"].strip()] = (ref, usable, r.get("refuse_rate", 0.0))

    rows, n_all, n_orig_ref, n_flip = [], 0, 0, 0
    for r in json.load(open(a.rw_scored))["examples"]:
        o, w = r["original"].strip(), r["rewrite"].strip()
        n_all += 1
        oref, ousable, orate = orig.get(o, (None, False, None))
        if not ousable or not oref:
            continue
        n_orig_ref += 1
        wref, wusable, _ = refused_v6(r.get("samples", []))
        if not wusable or wref:
            continue
        n_flip += 1
        m = pair_metrics(o, w)
        rows.append({"pair_id": f"ur{n_flip:06d}", "original": o, "rewrite": w,
                     "orig_refuse_rate": orate, "rw_refuse_rate": r.get("refuse_rate", 0.0),
                     "wl_dist_content": m["wl_dist_content"],
                     "norm_dist_content": m["norm_dist_content"],
                     "introduced_words": m["introduced_words"],
                     "removed_words": m["removed_words"]})

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fields = ["pair_id", "original", "rewrite", "orig_refuse_rate", "rw_refuse_rate",
              "wl_dist_content", "norm_dist_content", "introduced_words", "removed_words"]
    with open(a.out, "w", newline="") as f:
        w_ = csv.DictWriter(f, fieldnames=fields); w_.writeheader(); w_.writerows(rows)

    stats = dict(rewrites_scored=n_all, with_refused_original=n_orig_ref, flips=n_flip,
                 flip_rate_pct=round(100 * n_flip / n_orig_ref, 2) if n_orig_ref else 0.0,
                 distinct_originals=len({r["original"] for r in rows}))
    print(f"[ur] scored rewrites: {n_all}")
    print(f"[ur] whose ORIGINAL was refused: {n_orig_ref}")
    print(f"[ur] refusal->compliance FLIPS: {n_flip} ({stats['flip_rate_pct']}%) "
          f"over {stats['distinct_originals']} distinct originals")
    print(f"[ur] wrote {a.out}")
    if a.stats_out:
        json.dump(stats, open(a.stats_out, "w"), indent=1)


if __name__ == "__main__":
    main()
