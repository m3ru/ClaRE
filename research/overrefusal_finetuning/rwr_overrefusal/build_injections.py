#!/usr/bin/env python3
"""Stage C setup -- build the injected prompt sets for the causal test.

Takes benign prompts and inserts spans from each cluster's pool, so we can measure whether
the material GCG inserts raises refusal on prompts it never saw.

Design decisions and why:

  PAIRED. Every arm uses the SAME benign prompts, so a difference between arms cannot come
  from having sampled different prompts. With refusal rates this low that matters more than
  the sample size does.

  PROMPT POOL. probe_or/results/corpus2/originals.csv, which shares zero originals with
  either GCG corpus -- so it is unseen by the attack, and it is the pool the RWR results
  were measured on, which keeps the numbers comparable.

  PLACEMENT. Each span goes where its cluster's spans actually landed: position is sampled
  from that cluster's empirical distribution of midpoint-fraction in the source rewrites.
  Negation really is end-loaded (71% in the final third, median 0.80) and action really is
  not (49%, median 0.66), so a single append-at-end rule would misrepresent both.

  COUNT. Spans per prompt matches the observed median per cluster: 2 for chars, 1 for the
  rest.

  COMBINATION ARMS are the combinations that actually co-occur in the corpus -- chars +
  harmful_phrase (34.8% of rewrites), chars + touchy (30.6%), chars + negation (18.5%) --
  plus an all-five arm, rather than combinations chosen for neatness.

Note on controls: the owner opted for baseline-only, so there is no length-matched random
arm. The `chars` arm partially covers that gap by accident -- it is meaningless punctuation,
so if it moves refusal as much as the contentful arms do, the effect is about disturbing
the prompt rather than about what was inserted.

Run: python build_injections.py --n 300
"""
import argparse, csv, json, os, random, re, sys
from collections import defaultdict

CLUSTERS = ["chars", "negation", "touchy", "harmful_phrase", "action"]
N_SPANS = {"chars": 2, "negation": 1, "touchy": 1, "harmful_phrase": 1, "action": 1}
COMBOS = {
    "chars+harmful_phrase": ["chars", "harmful_phrase"],
    "chars+touchy": ["chars", "touchy"],
    "chars+negation": ["chars", "negation"],
    "all_five": CLUSTERS,
}


def positions_by_cluster(clusters_path):
    """Empirical midpoint-fraction of each cluster's spans in its source rewrite."""
    pos = defaultdict(list)
    for r in json.load(open(clusters_path)):
        rw = r.get("rewrite") or ""
        if not rw:
            continue
        for k, spans in r.get("spans", {}).items():
            for s in spans:
                i = rw.find(s)
                if i >= 0:
                    pos[k].append((i + len(s) / 2) / len(rw))
    return pos


def insert_at(prompt, span, frac):
    """Insert span at the word boundary nearest to `frac` through the prompt."""
    bounds = [0] + [m.end() for m in re.finditer(r"\s+", prompt)] + [len(prompt)]
    target = frac * len(prompt)
    i = min(bounds, key=lambda b: abs(b - target))
    left, right = prompt[:i].rstrip(), prompt[i:].lstrip()
    parts = [p for p in (left, span.strip(), right) if p]
    return " ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", default="probe_or/results/span_pools_llama.json")
    ap.add_argument("--clusters", default="probe_or/results/gcg_clusters_llama.json")
    ap.add_argument("--prompts", default="probe_or/results/corpus2/originals.csv")
    ap.add_argument("--gcg_originals", default="incoming/sonnet_filtered_strict.json")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", default="probe_or/results/injections.json")
    a = ap.parse_args()
    csv.field_size_limit(sys.maxsize)

    pools = json.load(open(a.pools))
    pos = positions_by_cluster(a.clusters)
    rng = random.Random(a.seed)

    seen = {(r.get("original") or "").strip()
            for r in json.load(open(a.gcg_originals))["rows"]}
    cand = []
    for r in csv.DictReader(open(a.prompts)):
        p = (r.get("original") or "").strip()
        if p and p not in seen and 20 <= len(p) <= 300:
            cand.append(p)
    cand = sorted(set(cand))
    if len(cand) < a.n:
        sys.exit(f"only {len(cand)} usable prompts, need {a.n}")
    prompts = rng.sample(cand, a.n)
    overlap = sum(1 for p in prompts if p in seen)
    print(f"[prompts] {len(cand)} candidates -> {len(prompts)} sampled; "
          f"overlap with GCG originals: {overlap}")

    def inject(prompt, kinds, r):
        out = prompt
        for k in kinds:
            for _ in range(N_SPANS[k]):
                span = r.choice(pools[k])
                frac = r.choice(pos[k]) if pos[k] else 1.0
                out = insert_at(out, span, frac)
        return out

    arms = {"baseline": None}
    arms.update({k: [k] for k in CLUSTERS})
    arms.update(COMBOS)

    data = {"seed": a.seed, "n_prompts": len(prompts), "arms": {}}
    for arm, kinds in arms.items():
        r = random.Random(a.seed + hash(arm) % 10000)
        items = []
        for i, p in enumerate(prompts):
            items.append({"idx": i, "original": p,
                          "prompt": p if kinds is None else inject(p, kinds, r)})
        data["arms"][arm] = items
    json.dump(data, open(a.out, "w"), indent=1)

    print(f"[arms] {len(arms)}: {', '.join(arms)}")
    print(f"[out] {a.out}  ({len(arms)*len(prompts)} prompt instances)\n")
    for arm in arms:
        ex = data["arms"][arm][0]
        print(f"--- {arm} ---")
        print(f"    {ex['prompt'][:150]}")


if __name__ == "__main__":
    main()
