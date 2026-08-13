#!/usr/bin/env python3
"""Generate FRESH, leakage-free minimal pairs for the Refusal Atlas word-level analysis.

Originals: benign prompts in the scored pool that are DISJOINT from every
direction/probe fitting set (pairs.csv / pairs_sonnet.csv / pairs_qwen.csv) -> no
train/test leakage. Rewrites are generated NOW by Claude Sonnet (reusing the tested
generate_or_sonnet batch flow + minimal-edit template), so neither Llama nor Qwen has
seen them -> no model contamination. Per original we keep the single MOST MINIMAL
rewrite (highest difflib ratio, non-identical) -> a clean 1-2-word-edit minimal pair.

--dry_run builds the batch requests and prints the count WITHOUT submitting (no spend).
"""
import argparse
import csv
import difflib
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root for generate_or_sonnet
import generate_or_sonnet as G   # USER_TEMPLATE, SYSTEM_PROMPT, build_requests, submit/poll/collect, custom_id_for


def load_clean_originals(pool_csv, fit_csvs, max_len):
    fit = set()
    for f in fit_csvs:
        for r in csv.DictReader(open(f)):
            fit.add((r.get("original") or "").strip())
            fit.add((r.get("rewrite") or "").strip())
    seen, clean = set(), []
    for r in csv.DictReader(open(pool_csv)):
        o = (r.get("original") or "").strip()
        if o and o not in fit and o not in seen and 0 < len(o) <= max_len:
            seen.add(o)
            clean.append(o)
    return clean


def pick_minimal(original, rewrites):
    """Most minimal (highest-similarity) non-identical, non-empty rewrite."""
    cands = [rw.strip() for rw in rewrites if rw and rw.strip() and rw.strip() != original]
    if not cands:
        return None
    return max(cands, key=lambda rw: difflib.SequenceMatcher(None, original, rw).ratio())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool_csv", default="../probe_or/results/qwen_scored/sonnet_benign_qwen_or.csv")
    ap.add_argument("--fit_csvs", nargs="+",
                    default=["../probe_or/pairs.csv", "../probe_or/pairs_sonnet.csv", "../probe_or/pairs_qwen.csv"])
    ap.add_argument("--out", default="data/fresh_pairs.csv")
    ap.add_argument("--raw_out", default="data/fresh_pairs_raw.json")
    ap.add_argument("--output_dir", default="data/gen_fresh")   # batch-id persistence
    ap.add_argument("--n_originals", type=int, default=80)
    ap.add_argument("--n_per_prompt", type=int, default=6)
    ap.add_argument("--max_orig_len", type=int, default=800)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--poll_interval", type=int, default=30)
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    clean = load_clean_originals(a.pool_csv, a.fit_csvs, a.max_orig_len)
    rng = random.Random(a.seed)
    rng.shuffle(clean)
    originals = clean[:a.n_originals]
    shard = list(enumerate(originals))   # (idx, original) -> custom_id "p-<idx>"
    requests = G.build_requests(shard, a.model, a.n_per_prompt, a.max_tokens)
    print(f"[select] {len(clean)} clean (leakage-free) originals available; using {len(originals)}")
    print(f"[batch] {len(requests)} requests (1/original), {a.n_per_prompt} rewrites each, model={a.model}")
    print(f"[example originals]")
    for o in originals[:5]:
        print(f"   - {o[:100]!r}")

    if a.dry_run:
        print("\n[dry-run] built requests but did NOT submit -> $0 spent. Re-run without --dry_run to generate.")
        return

    from anthropic import Anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY not set")
    client = Anthropic(api_key=api_key)
    batch_id_path = os.path.join(a.output_dir, "batch_ids.json")
    batch_ids = G.submit_batches(client, requests, batch_id_path, 0)
    for bid in batch_ids:
        G.poll_until_done(client, bid, a.poll_interval, 0)
    rewrites_by_cid, n_fail = G.collect_results(client, batch_ids, a.n_per_prompt)
    print(f"[collect] {len(rewrites_by_cid)} results, {n_fail} failed")

    raw, pairs = [], []
    for idx, original in shard:
        rws = rewrites_by_cid.get(G.custom_id_for(idx), [])
        raw.append({"idx": idx, "original": original, "rewrites": rws})
        rw = pick_minimal(original, rws)
        if rw:
            ratio = difflib.SequenceMatcher(None, original, rw).ratio()
            pairs.append((original, rw, ratio))
    json.dump(raw, open(a.raw_out, "w"), indent=1)
    with open(a.out, "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["pair_id", "original", "rewrite", "difflib_ratio"])
        for k, (o, rw, ratio) in enumerate(pairs):
            wtr.writerow([f"fresh_{k:03d}", o, rw, f"{ratio:.4f}"])
    print(f"[done] {len(pairs)} minimal pairs -> {a.out} (raw -> {a.raw_out})")
    if pairs:
        rr = [p[2] for p in pairs]
        print(f"[minimality] difflib ratio: min={min(rr):.3f} median={sorted(rr)[len(rr)//2]:.3f} max={max(rr):.3f}")
        print("[sample pairs]")
        for o, rw, _ in pairs[:4]:
            print(f"   {o[:60]!r}\n     -> {rw[:70]!r}")


if __name__ == "__main__":
    main()
