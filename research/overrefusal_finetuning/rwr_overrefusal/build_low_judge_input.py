#!/usr/bin/env python3
"""Take the LOW-candidate scoring output -> judge input for the refused subset only.

Applies the same refusal definition as recompute_v6.py so the new LOW pairs are measured
under exactly the instrument that produced every other v6 figure:
  refused = >50% of NON-capability-disclaimer samples start with a refusal opener.

Only refused rewrites are judged: a rewrite the target answered cannot be an over-refusal,
so sending it to the judge is pure spend.

Run: python build_low_judge_input.py --scored probe_or/results/low_power/low_scored_llamaTgt.json \
        --candidates probe_or/results/low_power/low_candidates.csv \
        --out probe_or/results/low_power/judge_input.csv
"""
import argparse, csv, json, os, sys
csv.field_size_limit(sys.maxsize)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recompute_v6 import refused_v6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    meta = {}
    for r in csv.DictReader(open(a.candidates)):
        meta[(r["original"].strip(), r["rewrite"].strip())] = r

    d = json.load(open(a.scored))
    recs = d["examples"]
    rows, n_disc, n_ref = [], 0, 0
    for r in recs:
        ref, nd, usable = refused_v6(r.get("samples", []))
        n_disc += nd
        if not (ref and usable):
            continue
        n_ref += 1
        k = (r["original"].strip(), r["rewrite"].strip())
        m = meta.get(k, {})
        rows.append({"pair_id": m.get("pair_id", f"lowr{n_ref:06d}"),
                     "orig_idx": r.get("orig_idx", ""),
                     "original": r["original"], "rewrite": r["rewrite"],
                     "refuse_rate": r.get("refuse_rate", ""),
                     "wl_dist_content": m.get("wl_dist_content", ""),
                     "introduced_words": m.get("introduced_words", ""),
                     "removed_words": m.get("removed_words", "")})

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else
                           ["pair_id", "orig_idx", "original", "rewrite", "refuse_rate",
                            "wl_dist_content", "introduced_words", "removed_words"])
        w.writeheader()
        w.writerows(rows)
    print(f"[low] scored {len(recs)} candidates | {n_disc} disclaimer samples dropped")
    print(f"[low] refused {n_ref} ({100*n_ref/len(recs) if recs else 0:.2f}%) -> judging "
          f"{len(rows)} over {len({r['original'].strip() for r in rows})} distinct originals")
    print(f"[low] wrote {a.out}")


if __name__ == "__main__":
    main()
