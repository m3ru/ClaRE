#!/usr/bin/env python3
"""Clean the extracted spans into injection pools, one per cluster.

cluster_gcg.py's spans come straight from the judge and are verbatim, but they are not yet
fit to inject. Three problems, all visible by eye in the raw pools:

  1. Real profanity and slurs ('salopes', 'beurette', 'ass') land in harmful_phrase. The
     owner ruled that inserted profanity is NOT over-refusal, and injecting a slur into a
     benign prompt would raise refusal for a reason nobody is asking about. Dropped from
     every pool.
  2. The chars pool picks up ordinary words ('How', 'This metaphor represents emotions').
     A character-noise arm has to be character noise, so spans must be mostly
     non-alphanumeric to stay.
  3. Bare function words ('that', 'because', 'fairly') carry no cluster signal. Dropped
     from every pool except negation, where 'no' and 'not' are the entire point.

Everything dropped is written to the report so the decisions are auditable rather than
buried, and the counts are printed per rule.

Run: python build_span_pools.py
"""
import argparse, json, os, re, sys
from collections import Counter

# Real profanity / slurs. Deliberately conservative: only strings that are offensive in
# themselves, not merely alarming-sounding, since alarming-but-benign IS the phenomenon.
PROFANE = re.compile(
    r"\b(f\s*\*+\s*k|fuck|f\W*u\W*c\W*k|shit|bitch|salope|salopes|beurette|whore|slut|"
    r"nigg|n\*+r|cunt|ass\b|asshole|dick|cock|porn|rape|retard)", re.I)
# "F you" and friends, which the owner explicitly excluded.
FU = re.compile(r"\bf+\W*\s*(you|u)\b|\bf\W*\*+", re.I)
FUNCTION = set("""a an the of to in for on with and or but is are was were be been it its
this that these those as at by from into if then than so such do does did have has had i
you he she they we them his her their our your my me us can could will would may might
must about over under again once here there when where why how all any both each because
fairly just very really quite also only more most other some what which who whom""".split())


def is_charnoise(s):
    letters = sum(c.isalnum() for c in s)
    return letters / max(len(s), 1) < 0.4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clusters", default="probe_or/results/gcg_clusters_llama.json")
    ap.add_argument("--out", default="probe_or/results/span_pools_llama.json")
    ap.add_argument("--report", default="SPAN_POOLS.md")
    ap.add_argument("--min_len", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=60)
    a = ap.parse_args()

    rows = json.load(open(a.clusters))
    raw = {}
    for r in rows:
        for k, v in r["spans"].items():
            raw.setdefault(k, []).extend(s.strip() for s in v if s.strip())

    pools, dropped = {}, {}
    for k, v in raw.items():
        keep, cut = [], Counter()
        for s in v:
            if PROFANE.search(s) or FU.search(s):
                cut["profanity or slur"] += 1
            elif not (a.min_len <= len(s) <= a.max_len):
                cut["too short or too long"] += 1
            elif k == "chars" and not is_charnoise(s):
                cut["not character noise"] += 1
            elif k != "negation" and s.lower().strip(".,!?;:'\"") in FUNCTION:
                cut["bare function word"] += 1
            else:
                keep.append(s)
        pools[k] = keep
        dropped[k] = cut

    L = ["# Injection span pools\n",
         "Spans extracted by `cluster_gcg.py` and cleaned by `build_span_pools.py`, one "
         "pool per cluster, for the causal injection test.\n",
         "Pools are disjoint by construction: the judge assigns each span to exactly one "
         "cluster, so combining two arms later is concatenating two non-overlapping sets.\n",
         "\n## What was dropped, and why\n",
         "| cluster | raw | kept | profanity/slur | length | not char noise | function word |",
         "|---|--:|--:|--:|--:|--:|--:|"]
    for k in sorted(raw):
        d = dropped[k]
        L.append(f"| `{k}` | {len(raw[k])} | **{len(pools[k])}** | {d['profanity or slur']} | "
                 f"{d['too short or too long']} | {d['not character noise']} | "
                 f"{d['bare function word']} |")
    L.append("\nProfanity and slurs are dropped from every pool because the owner ruled "
             "that inserted profanity is not over-refusal; injecting one would raise "
             "refusal for a reason that is not in question. Alarming-but-benign phrasing "
             "is exactly what we want to keep, so the profanity list is deliberately "
             "narrow.\n")

    L.append("\n## Pools\n")
    for k in sorted(pools):
        v = pools[k]
        uniq = sorted(set(v), key=lambda s: (-v.count(s), s))
        L.append(f"\n### `{k}` — {len(v)} spans, {len(uniq)} unique\n")
        L.append("Most frequent: " + ", ".join(f"`{s}`" for s in uniq[:12]))
        L.append("\nSample: " + ", ".join(f"`{s}`" for s in uniq[12:30]))
    open(a.report, "w").write("\n".join(L) + "\n")
    json.dump({k: sorted(set(v)) for k, v in pools.items()}, open(a.out, "w"), indent=1)

    print(f"{'cluster':16s}{'raw':>7}{'kept':>7}{'unique':>8}   dropped")
    for k in sorted(raw):
        print(f"{k:16s}{len(raw[k]):7d}{len(pools[k]):7d}{len(set(pools[k])):8d}   "
              + ", ".join(f"{n} {r}" for r, n in dropped[k].most_common() if n))
    print(f"\nwrote {a.out} and {a.report}")


if __name__ == "__main__":
    main()
