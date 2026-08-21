#!/usr/bin/env python3
"""Merge the newly-powered LOW pairs into the v6 LOW stratum, in its exact schema.

The existing LOW stratum (37 llamaAtt pairs / 31 originals) is too small to support the
geometry work or a stable trigger table. This folds in pairs from the dedicated LOW-power
run, which used the SAME attacker (llama-logit), the SAME target (Llama-3-8B-Instruct),
the SAME refusal definition (recompute_v6.refused_v6) and the SAME judge (or_judge_v5),
on held-out originals -- so the merged stratum is one population, not two pooled ones.

Writes or_low_stratum_v7.csv and leaves v6 untouched as the audit trail.

Run: python merge_low_stratum.py --verdicts probe_or/results/low_power/verdicts.csv
"""
import argparse, csv, json, os, sys
csv.field_size_limit(sys.maxsize)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_edit_distance import pair_metrics

SCHEMA = ["attacker", "target", "pair_id", "orig_idx", "stratum", "refuse_rate", "refused",
          "intent", "harm", "is_or", "wl_dist", "wl_dist_content", "norm_dist",
          "norm_dist_content", "n_tok_orig", "n_tok_rw", "n_content_orig", "n_content_rw",
          "introduced_words", "removed_words", "edit_ops", "original", "rewrite"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default="probe_or/results/edit_strata/or_low_stratum_v6.csv")
    ap.add_argument("--judge_input", default="probe_or/results/low_power/judge_input.csv")
    ap.add_argument("--verdicts", default="probe_or/results/low_power/verdicts.csv")
    ap.add_argument("--out", default="probe_or/results/edit_strata/or_low_stratum_v7.csv")
    ap.add_argument("--tag_attacker", default="", help="attacker/target tag for NEW rows. "
                    "Defaults to --attacker_only, because a power-up run scores its own "
                    "attacker against its own target; hardcoding llamaAtt silently mislabels "
                    "a Qwen run and downstream --attacker filters then drop every new pair.")
    ap.add_argument("--attacker_only", default="llamaAtt",
                    help="keep only this attacker from the OLD stratum; over-refusal was "
                         "confirmed per-target, so qwenAtt pairs are not Llama evidence")
    a = ap.parse_args()

    old = [r for r in csv.DictReader(open(a.old))
           if (not a.attacker_only) or r["attacker"] == a.attacker_only]
    print(f"[merge] old LOW ({a.attacker_only}): {len(old)} pairs / "
          f"{len({r['original'].strip() for r in old})} originals")

    ji = {r["pair_id"]: r for r in csv.DictReader(open(a.judge_input))}
    new, n_seen = [], 0
    seen_pairs = {(r["original"].strip(), r["rewrite"].strip()) for r in old}
    for v in csv.DictReader(open(a.verdicts)):
        n_seen += 1
        if v.get("is_or") != "1":
            continue
        src = ji.get(v["pair_id"])
        if not src:
            continue
        key = (src["original"].strip(), src["rewrite"].strip())
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        m = pair_metrics(src["original"], src["rewrite"])
        row = {k: "" for k in SCHEMA}
        row.update(m)
        _att = a.tag_attacker or a.attacker_only or "llamaAtt"
        _tgt = {"llamaAtt": "llamaTgt", "qwenAtt": "qwenTgt"}.get(_att, "llamaTgt")
        row.update(attacker=_att, target=_tgt, pair_id=src["pair_id"],
                   orig_idx=src.get("orig_idx", ""), stratum="LOW",
                   refuse_rate=src.get("refuse_rate", ""), refused="1",
                   intent=v.get("intent", ""), harm=v.get("harm", ""), is_or="1",
                   original=src["original"], rewrite=src["rewrite"])
        new.append(row)

    merged = [{k: r.get(k, "") for k in SCHEMA} for r in old] + new
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SCHEMA)
        w.writeheader()
        w.writerows(merged)
    origs = len({r["original"].strip() for r in merged})
    print(f"[merge] judged {n_seen} | new confirmed-OR added: {len(new)}")
    print(f"[merge] LOW v7: {len(merged)} pairs / {origs} originals "
          f"(was {len(old)}/{len({r['original'].strip() for r in old})}) -> {a.out}")


if __name__ == "__main__":
    main()
