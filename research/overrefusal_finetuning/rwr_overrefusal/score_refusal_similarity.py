#!/usr/bin/env python3
"""Score MiniLM similarity for the refusing rewrites from a behavioral eval.

Input : <out_dir>/refusals_only.jsonl (from behavioral_rewriters_eval.py)
Output: <out_dir>/refusals_with_minilm_sim.json  -- each distinct refusing
        (original, rewrite, response) with the MiniLM cosine the OR reward uses.
Prints the sim distribution and the subset above --sim_hi (the "faithful AND
refused" set: candidate genuine over-refusals).
"""
import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--arm", default="top20", help="which rewriter arm to score")
    ap.add_argument("--sim_hi", type=float, default=0.85)
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    rows, seen = [], set()
    with open(os.path.join(args.out_dir, "refusals_only.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if r.get("arm") != args.arm:
                continue
            key = (r["original"].strip(), r["text"].strip())
            if key in seen:
                continue
            seen.add(key)
            rows.append({"o": r["original"], "r": r["text"], "resp": r["response"]})

    m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    oe = m.encode([x["o"] for x in rows], normalize_embeddings=True)
    rw = m.encode([x["r"] for x in rows], normalize_embeddings=True)
    s = (oe * rw).sum(1)
    for x, v in zip(rows, s):
        x["sim"] = float(v)
    out = os.path.join(args.out_dir, "refusals_with_minilm_sim.json")
    json.dump(rows, open(out, "w"), indent=1)

    print("n=%d  mean %.3f  median %.3f  min %.3f  max %.3f"
          % (len(rows), s.mean(), np.median(s), s.min(), s.max()))
    print("  >=0.50: %.0f%%   >=0.75: %.0f%%   >=%.2f: %d of %d"
          % (100 * (s >= 0.5).mean(), 100 * (s >= 0.75).mean(),
             args.sim_hi, int((s >= args.sim_hi).sum()), len(rows)))
    for v, x in sorted(((float(v), x) for v, x in zip(s, rows) if v >= args.sim_hi),
                       key=lambda t: -t[0]):
        print("\nsim=%.3f\n  ORIG   : %s\n  REWRITE: %s\n  LLAMA  : %s"
              % (v, x["o"].strip()[:160], x["r"].strip()[:160], x["resp"].strip()[:160]))
    print("\n[done] wrote %s" % out)


if __name__ == "__main__":
    main()
