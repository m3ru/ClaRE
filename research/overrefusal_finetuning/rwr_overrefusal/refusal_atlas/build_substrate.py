#!/usr/bin/env python3
"""Assemble the Refusal Atlas prompt substrate (~650 prompts) from staged sources.

Unified schema (one row per prompt):
  prompt_id, text, source, gold_benign, native_topic, pair_id, is_rewrite,
  probe_delta, similarity

Sources & composition (seed 42):
  - orbench_hard : 30/category x 10  = 300   benign-but-scary (gold_benign=1)
  - xstest       : 150 safe + 50 unsafe = 200 (gold_benign from label)
  - sonnet_pair  : 50 original->rewrite pairs = 100 (minimal-edit word carriers)
  - orbench_toxic: 50 genuinely toxic       = 50   (should-refuse anchor, gold_benign=0)

native_topic = each source's own label (orbench 10-cat, xstest type). A unified
OR-Bench-10 topic is assigned to every prompt later (P5) by a classifier.
"""
import argparse, csv, os, random

def load_csv(p):
    with open(p) as f:
        return list(csv.DictReader(f))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--pool_csv",
        default="../probe_or/results/qwen_scored/sonnet_benign_qwen_or.csv")
    ap.add_argument("--out", default="data/substrate.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--per_orbench_cat", type=int, default=30)
    ap.add_argument("--xstest_safe", type=int, default=150)
    ap.add_argument("--xstest_unsafe", type=int, default=50)
    ap.add_argument("--n_pairs", type=int, default=0, help="cap on fresh pairs; 0 = use all")
    ap.add_argument("--n_toxic", type=int, default=50)
    ap.add_argument("--fresh_pairs_csv", default="data/fresh_pairs.csv")
    ap.add_argument("--single_edit_csv", default="data/single_edit_pairs.csv")
    ap.add_argument("--cap_single_edit", type=int, default=0, help="cap single-word pairs; 0 = all")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    rows = []

    def add(text, source, gold_benign, native_topic, pair_id="", is_rewrite="",
            probe_delta="", similarity=""):
        rows.append(dict(prompt_id=f"{source}_{len(rows):04d}", text=text.strip(),
            source=source, gold_benign=gold_benign, native_topic=native_topic,
            pair_id=pair_id, is_rewrite=is_rewrite,
            probe_delta=probe_delta, similarity=similarity))

    # 1) OR-Bench-Hard-1K, stratified by category
    hard = load_csv(f"{a.data_dir}/or-bench-hard-1k.csv")
    by_cat = {}
    for r in hard:
        by_cat.setdefault(r["category"], []).append(r["prompt"])
    for cat in sorted(by_cat):
        pool = by_cat[cat]; rng.shuffle(pool)
        for p in pool[:a.per_orbench_cat]:
            add(p, "orbench_hard", 1, cat)

    # 2) XSTest safe + unsafe
    xs = load_csv(f"{a.data_dir}/xstest.csv")
    safe = [r for r in xs if r["label"] == "safe"]
    unsafe = [r for r in xs if r["label"] != "safe"]
    rng.shuffle(safe); rng.shuffle(unsafe)
    for r in safe[:a.xstest_safe]:
        add(r["prompt"], "xstest", 1, r.get("type", ""))
    for r in unsafe[:a.xstest_unsafe]:
        add(r["prompt"], "xstest", 0, r.get("type", ""))

    # 3) Fresh, leakage-free minimal pairs from gen_fresh_pairs.py (original -> rewrite).
    #    These originals are disjoint from every direction/probe fitting set and the
    #    rewrites are freshly generated (unseen by both models). MiniLM similarity is
    #    filled later in the analysis venv; difflib_ratio is a minimal-edit proxy.
    if os.path.exists(a.fresh_pairs_csv):
        fresh = load_csv(a.fresh_pairs_csv)
        for r in fresh[:a.n_pairs] if a.n_pairs else fresh:
            pid = r.get("pair_id") or f"fresh_{len([x for x in rows if x['source']=='sonnet_pair'])//2:03d}"
            ratio = r.get("difflib_ratio", "")
            add(r["original"], "sonnet_pair", 1, "benign_source", pid, 0, "", ratio)
            add(r["rewrite"], "sonnet_pair", "", "manipulated", pid, 1, "", ratio)
        print(f"[pairs] added {sum(1 for x in rows if x['source']=='sonnet_pair')//2} fresh pairs from {a.fresh_pairs_csv}")
    else:
        print(f"[pairs] {a.fresh_pairs_csv} not found -> building substrate WITHOUT the multi-edit pair slice")

    # 3b) Fresh SINGLE-WORD minimal pairs (clean causal word effects; native_topic = the trigger word)
    if os.path.exists(a.single_edit_csv):
        se = load_csv(a.single_edit_csv)
        if a.cap_single_edit:
            se = se[:a.cap_single_edit]
        for r in se:
            pid = r["pair_id"]
            add(r["original"], "single_edit_pair", 1, "benign_source", pid, 0, "", "")
            add(r["rewrite"], "single_edit_pair", "", f"word:{r['new_word']}", pid, 1, "", "")
        print(f"[single-edit] added {sum(1 for x in rows if x['source']=='single_edit_pair')//2} single-word pairs "
              f"from {a.single_edit_csv}")

    # 4) OR-Bench-Toxic anchor
    tox = load_csv(f"{a.data_dir}/or-bench-toxic.csv")
    rng.shuffle(tox)
    for r in tox[:a.n_toxic]:
        add(r["prompt"], "orbench_toxic", 0, r["category"])

    fields = ["prompt_id","text","source","gold_benign","native_topic",
              "pair_id","is_rewrite","probe_delta","similarity"]
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        w.writerows(rows)

    # report
    from collections import Counter
    print(f"wrote {a.out}: {len(rows)} prompts")
    print("by source:", dict(Counter(r["source"] for r in rows)))
    print("by gold_benign:", dict(Counter(str(r["gold_benign"]) for r in rows)))
    print("orbench_hard by category:",
          dict(Counter(r["native_topic"] for r in rows if r["source"]=="orbench_hard")))
    print(f"sonnet pairs: {sum(1 for r in rows if r['source']=='sonnet_pair')//2}")

if __name__ == "__main__":
    main()
