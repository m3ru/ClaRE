#!/usr/bin/env python3
"""Assemble every prompt set the Delta geometry analysis needs, into ONE csv.

Delta = h(rewrite) - h(original). The point of pairing is that subtracting the original
cancels topic, so what remains is what the REWRITE did. To tell "what a refused rewrite did"
apart from "what any rewrite does", every over-refusal pair needs a control pair matched on
the thing we are not interested in: same attacker, same original where possible, same
edit-distance band, but NOT refused.

Sets emitted (column `set`):
  or_high / or_low      confirmed over-refusal, split by content-edit distance (<=2 = low)
  ctrl_high / ctrl_low  NOT refused, same band, preferring rewrites of the SAME original
  adv_harmful/adv_benign  AdvBench + Alpaca-style benign: the POSITIVE CONTROL, a contrast
                        whose dimensionality is known from the literature (~1 direction)

`pair_group` marks originals that have BOTH an OR and a control rewrite in the same band;
those support the within-original estimator Delta' = h(rewrite_OR) - h(rewrite_ctrl), which
cancels topic AND the attacker's house style.

Run: python build_delta_sets.py --out probe_or/results/delta/prompt_sets.csv
"""
import argparse, csv, json, os, random, sys
from collections import defaultdict
csv.field_size_limit(sys.maxsize)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_edit_distance import pair_metrics
from recompute_v6 import refused_v6

FIELDS = ["set", "pair_id", "original", "rewrite", "wl_dist_content", "refuse_rate",
          "refused", "is_or", "pair_group", "introduced_words"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", default="probe_or/results/corpus2/llamaAtt_llamaTgt.json")
    ap.add_argument("--low_scored", default="probe_or/results/low_power/low_scored_llamaTgt.json")
    ap.add_argument("--or_high", default="probe_or/results/edit_strata/or_high_stratum_v6.csv")
    ap.add_argument("--or_low", default="probe_or/results/edit_strata/or_low_stratum_v7.csv")
    ap.add_argument("--advbench", default="probe_or/data/advbench_harmful_behaviors.csv")
    ap.add_argument("--benign", default="probe_or/data/arditi_harmless.csv")
    ap.add_argument("--n_adv", type=int, default=400)
    ap.add_argument("--low_cut", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--attacker", default="llamaAtt",
                    help="which attacker's confirmed-OR pairs to use. Over-refusal was confirmed "
                         "PER TARGET, so llamaAtt pairs are evidence about Llama and qwenAtt "
                         "pairs about Qwen -- never pool them.")
    ap.add_argument("--out", default="probe_or/results/delta/prompt_sets.csv")
    a = ap.parse_args()
    rnd = random.Random(a.seed)

    # ---- confirmed over-refusals (llamaAtt only; qwenAtt OR was established on Qwen) ----
    orset = {}
    for path, tag in ((a.or_high, "or_high"), (a.or_low, "or_low")):
        for r in csv.DictReader(open(path)):
            if r.get("attacker") not in (None, "", a.attacker):
                continue
            key = (r["original"].strip(), r["rewrite"].strip())
            d = int(r["wl_dist_content"]) if str(r.get("wl_dist_content", "")).isdigit() else None
            orset[key] = dict(tag="or_low" if (d is not None and d <= a.low_cut) else "or_high",
                              d=d, refuse_rate=r.get("refuse_rate", ""))
    print(f"[sets] confirmed OR pairs ({a.attacker}): {len(orset)}")

    # ---- every scored rewrite, so controls come from the SAME generation + scoring run ----
    by_orig = defaultdict(list)
    n_scored = 0
    for path in (a.scored, a.low_scored):
        if not os.path.exists(path):
            print(f"[warn] missing {path}"); continue
        for r in json.load(open(path))["examples"]:
            o, rw = r["original"].strip(), r["rewrite"].strip()
            ref, _, usable = refused_v6(r.get("samples", []))
            if not usable:
                continue
            n_scored += 1
            by_orig[o].append(dict(rewrite=rw, refused=ref, rate=r.get("refuse_rate", 0.0)))
    print(f"[sets] scored rewrites available: {n_scored} over {len(by_orig)} originals")

    rows, gid = [], 0
    used_ctrl = set()
    # ---- OR rows + a band-matched, same-original control wherever one exists ----
    for (o, rw), meta in orset.items():
        d = meta["d"]
        if d is None:
            m = pair_metrics(o, rw); d = m["wl_dist_content"]
        band_low = d <= a.low_cut
        # candidate controls: same original, not refused, same band
        cands = []
        for c in by_orig.get(o, []):
            if c["refused"] or (o, c["rewrite"]) in used_ctrl or c["rewrite"] == rw:
                continue
            dc = pair_metrics(o, c["rewrite"])["wl_dist_content"]
            if (dc <= a.low_cut) == band_low:
                cands.append((c, dc))
        g = ""
        if cands:
            c, dc = rnd.choice(cands)
            used_ctrl.add((o, c["rewrite"]))
            gid += 1; g = f"g{gid:05d}"
            rows.append(dict(set="ctrl_low" if band_low else "ctrl_high",
                             pair_id=f"c{gid:05d}", original=o, rewrite=c["rewrite"],
                             wl_dist_content=dc, refuse_rate=c["rate"], refused=0, is_or=0,
                             pair_group=g, introduced_words=""))
        rows.append(dict(set=meta["tag"], pair_id=f"o{len(rows):06d}", original=o, rewrite=rw,
                         wl_dist_content=d, refuse_rate=meta["refuse_rate"], refused=1, is_or=1,
                         pair_group=g, introduced_words=""))

    # ---- top up controls that had no same-original partner, matched on band only ----
    need = defaultdict(int)
    for t in ("or_high", "or_low"):
        have = sum(1 for r in rows if r["set"] == t.replace("or_", "ctrl_"))
        need[t] = max(0, sum(1 for r in rows if r["set"] == t) - have)
    pool = [(o, c) for o, cs in by_orig.items() for c in cs
            if not c["refused"] and (o, c["rewrite"]) not in used_ctrl]
    rnd.shuffle(pool)
    for o, c in pool:
        dc = pair_metrics(o, c["rewrite"])["wl_dist_content"]
        t = "or_low" if dc <= a.low_cut else "or_high"
        if need[t] <= 0:
            continue
        need[t] -= 1
        used_ctrl.add((o, c["rewrite"]))
        rows.append(dict(set="ctrl_low" if dc <= a.low_cut else "ctrl_high",
                         pair_id=f"cx{len(rows):06d}", original=o, rewrite=c["rewrite"],
                         wl_dist_content=dc, refuse_rate=c["rate"], refused=0, is_or=0,
                         pair_group="", introduced_words=""))
        if all(v <= 0 for v in need.values()):
            break

    # ---- positive control: harmful vs benign, dimensionality known from the literature ----
    def col(path, n):
        out = []
        for r in csv.DictReader(open(path)):
            v = (r.get("goal") or r.get("prompt") or r.get("instruction")
                 or list(r.values())[0] or "").strip()
            if v:
                out.append(v)
            if len(out) >= n:
                break
        return out
    for tag, path in (("adv_harmful", a.advbench), ("adv_benign", a.benign)):
        if not os.path.exists(path):
            print(f"[warn] positive control missing {path}"); continue
        for i, p in enumerate(col(path, a.n_adv)):
            rows.append(dict(set=tag, pair_id=f"{tag}{i:05d}", original="", rewrite=p,
                             wl_dist_content="", refuse_rate="", refused="", is_or="",
                             pair_group="", introduced_words=""))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    from collections import Counter
    print("[sets] " + "  ".join(f"{k}={v}" for k, v in sorted(Counter(r["set"] for r in rows).items())))
    print(f"[sets] within-original Delta' groups: {gid}")
    uniq = {r["original"].strip() for r in rows if r["original"].strip()} | \
           {r["rewrite"].strip() for r in rows if r["rewrite"].strip()}
    print(f"[sets] unique prompts to extract: {len(uniq)} -> {a.out}")


if __name__ == "__main__":
    main()
