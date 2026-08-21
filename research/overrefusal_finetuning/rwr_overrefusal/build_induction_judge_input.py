#!/usr/bin/env python3
"""Judge input: did the inserted word keep the ask benign, or did it make it genuinely harmful?

The whole claim is that we INDUCED over-refusal -- a benign request wrongly refused. An
insertion that actually turns the ask harmful ("weaponize the object") is refused CORRECTLY
and is not over-refusal at all; counting it would inflate the headline number by exactly the
amount the rewriter drifted.

So the sample that matters is the REFUSED frame rewrites: every one of those is currently
being counted as induced over-refusal, and the judge's job is to say what fraction really is.
Control arms are sampled too, at a smaller rate, as a check on the judge itself -- ctrl_rare
rewrites should come back overwhelmingly INTENT=SAME / HARM=BENIGN.

Emits the column layout or_judge_v5 expects (pair_id, original, rewrite), so it feeds
judge_direct.py unchanged:

  python build_induction_judge_input.py --generations <..._generations.csv> --out judge_in.csv
  python judge_direct.py --corpus_csv judge_in.csv --out judge_out.csv --model claude-sonnet-5

judge_direct.py writes is_or=1 only when INTENT=SAME and HARM=BENIGN; the mean of that column
over the refused frame rows is the purity of the induced over-refusal count.
"""
import argparse, csv, os, random, sys
from collections import Counter
csv.field_size_limit(sys.maxsize)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", required=True, help="score_induced.py *_generations.csv")
    ap.add_argument("--per_frame", type=int, default=40,
                    help="max refused rewrites sampled per frame arm")
    ap.add_argument("--per_control", type=int, default=15,
                    help="control rows sampled per control arm (judge sanity check)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rnd = random.Random(a.seed)

    rows = list(csv.DictReader(open(a.generations)))
    by = {}
    for r in rows:
        if r["arm"] == "orig":
            continue
        # frame arms: only the rows currently counted as induced over-refusal (refused).
        # control arms: sample regardless, so the judge is exercised on both classes.
        if r["arm"].startswith("frame_") and r["refusal"] != "1":
            continue
        by.setdefault(r["arm"], []).append(r)

    out, stats = [], Counter()
    for arm, rr in sorted(by.items()):
        cap = a.per_frame if arm.startswith("frame_") else a.per_control
        rnd.shuffle(rr)
        for i, r in enumerate(rr[:cap]):
            out.append(dict(pair_id=f"{arm}_{i:03d}", arm=arm, word=r["word"],
                            alpaca_idx=r["alpaca_idx"], original=r["original"],
                            rewrite=r["prompt"], refused=r["refusal"]))
        stats[arm] = (min(cap, len(rr)), len(rr))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    cols = ["pair_id", "arm", "word", "alpaca_idx", "original", "rewrite", "refused"]
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(out)
    print(f"[judge-in] {len(out)} items -> {a.out}")
    for arm, (took, avail) in sorted(stats.items()):
        tag = "refused rewrites" if arm.startswith("frame_") else "control rewrites"
        print(f"   {arm:22s} {took:3d} of {avail:4d} {tag}")


if __name__ == "__main__":
    main()
