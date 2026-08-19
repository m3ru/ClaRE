#!/usr/bin/env python3
"""Build the de-duplicated judge input for the 2x2 attacker x target design.

Cost control. A judge verdict depends only on (original, rewrite) -- the judge never sees
which target model refused. So in a 2x2, a rewrite refused by BOTH targets must be judged
once, not twice, and identical rewrites sampled more than once collapse to a single call.

This takes the four scored cells, unions every (original, rewrite) pair that at least one
target refused, de-duplicates, and writes one judge-input CSV. Verdicts are joined back to
all four cells afterwards by build_2x2_report.py.

Run: python build_2x2_judge_input.py --scored_dir <dir> --out <judge_input.csv>
"""
import argparse, csv, json, os, sys
csv.field_size_limit(sys.maxsize)

CELLS = [("llamaAtt", "llamaTgt"), ("llamaAtt", "qwenTgt"),
         ("qwenAtt", "llamaTgt"), ("qwenAtt", "qwenTgt")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--refuse_thresh", type=float, default=0.5)
    args = ap.parse_args()

    uniq = {}          # (original, rewrite) -> pair_id
    refused_by = {}    # (original, rewrite) -> set of cells that refused it
    n_cell = {}
    for att, tgt in CELLS:
        p = os.path.join(args.scored_dir, f"{att}_{tgt}.json")
        if not os.path.exists(p):
            print(f"[warn] missing cell {att}->{tgt} ({p})", file=sys.stderr)
            continue
        recs = json.load(open(p))["examples"]
        hit = 0
        for r in recs:
            if r["refuse_rate"] > args.refuse_thresh:
                k = (r["original"].strip(), r["rewrite"].strip())
                refused_by.setdefault(k, set()).add(f"{att}->{tgt}")
                hit += 1
        n_cell[f"{att}->{tgt}"] = (hit, len(recs))

    for i, k in enumerate(sorted(refused_by)):
        uniq[k] = f"J{i:06d}"

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "original", "rewrite", "refused_by"])
        for k in sorted(refused_by):
            w.writerow([uniq[k], k[0], k[1], ";".join(sorted(refused_by[k]))])

    # Key on the pair explicitly. A " ".join((original, rewrite)) key is lossy: two
    # different (original, rewrite) splits can collapse to the same string.
    json.dump([{"pair_id": v, "original": k[0], "rewrite": k[1]}
               for k, v in sorted(uniq.items(), key=lambda kv: kv[1])],
              open(args.out + ".idmap.json", "w"))

    total_refusals = sum(h for h, _ in n_cell.values())
    print("cell refusal counts:")
    for c, (h, n) in n_cell.items():
        print(f"  {c:22s} {h:6d} / {n:6d}  ({100*h/n:5.2f}%)")
    print(f"\ntotal cell-level refusals : {total_refusals}")
    print(f"unique pairs to judge     : {len(uniq)}")
    if total_refusals:
        print(f"saved by de-duplication   : {total_refusals - len(uniq)} calls "
              f"({100*(total_refusals-len(uniq))/total_refusals:.0f}%)")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
