#!/usr/bin/env python3
"""Phase D: consolidate the causal-rank and cross-attacker results into one report."""
import json, os, math

OUT = "CAUSAL_RESULTS.md"
L = []


def wilson(k, n, z=1.96):
    if not n:
        return (0, 0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


L += ["# Causal results — how many directions carry over-refusal, and do they generalise?\n",
      "All numbers below are measured on **held-out originals**: the directions are fitted on a "
      "train half and never see the prompts they are evaluated on. This closes the leakage in the "
      "earlier frame ablation, which fitted on the same `or_high` pool it evaluated against.\n",
      "The rank question is posed **causally** (how many directions must be ablated before the "
      "behaviour stops), because the correlational version is ill-posed — for a two-class problem "
      "Fisher's LDA has exactly one discriminant by construction, which is why the retracted k\\* "
      "returned 1 for injected ranks 1, 2, 3 and 5.\n"]

p = "probe_or/results/causal_rank.json"
if os.path.exists(p):
    R = json.load(open(p))["rates"]
    K = int(R["_meta"]["K"]); n = int(R["_meta"]["n_eval"]); nh = int(R["_meta"]["n_harmful"])
    b_or, b_adv = R["or_rewrites__baseline"], R["advbench__baseline"]
    L += [f"\n## Causal rank sweep (held-out: {n} originals, {nh} AdvBench)\n",
          "`ours k` ablates the first k directions jointly; `random k` ablates k random "
          "orthonormal directions — the rank-matched control. A drop at k is only a result if "
          "the random control at the same k did not produce it.\n",
          "| k | over-refusal (ours) | over-refusal (random k) | AdvBench | originals | degenerate |",
          "|--:|--:|--:|--:|--:|--:|",
          f"| 0 (baseline) | {b_or:.1f}% | — | {b_adv:.1f}% | {R['originals__baseline']:.1f}% | "
          f"{R['originals__baseline__degen']:.1f}% |"]
    kstar = None
    for k in range(1, K + 1):
        o = R[f"or_rewrites__ours_k{k}"]; rr = R[f"or_rewrites__random_k{k}"]
        ad = R[f"advbench__ours_k{k}"]; og = R[f"originals__ours_k{k}"]
        dg = R[f"originals__ours_k{k}__degen"]
        ok = (o < 0.5 * b_or) and (ad > 0.8 * b_adv) and dg <= 20
        if ok and kstar is None:
            kstar = k
        lo, hi = wilson(round(o * n / 100), n)
        L.append(f"| {k} | **{o:.1f}%** [{lo:.0f}, {hi:.0f}] | {rr:.1f}% | {ad:.1f}% | "
                 f"{og:.1f}% | {dg:.1f}% |")
    L.append(f"\n| reference: atlas r̂ (k=1, literature direction) | "
             f"{R['or_rewrites__atlas_rhat']:.1f}% | | {R['advbench__atlas_rhat']:.1f}% | | |")
    L.append(f"\n**k\\*** (over-refusal below 50% of baseline **and** AdvBench above 80% of "
             f"baseline **and** coherent output) = **{kstar if kstar else 'not reached within K=' + str(K)}**")
    gains = [R[f"or_rewrites__ours_k{k}"] for k in range(1, K + 1)]
    if len(gains) >= 2:
        L.append(f"\nStacking check: k=1 leaves {gains[0]:.1f}% over-refusal, k=2 leaves "
                 f"{gains[1]:.1f}%. A further drop from k=1 to k=2 is the evidence that the "
                 f"second direction carries over-refusal the first does not — i.e. that "
                 f"over-refusal is genuinely more than one-dimensional.")
else:
    L.append("\n_Phase B (causal rank) did not produce output._")

p = "probe_or/results/gcg_transfer.json"
if os.path.exists(p):
    R = json.load(open(p))["rates"]
    b = R["gcg_rewrites__baseline"]
    ks = sorted(int(k.split("ours_k")[1]) for k in R if k.startswith("gcg_rewrites__ours_k")
                and not k.endswith("degen"))
    L += ["\n## Cross-attacker transfer — do RWR-derived directions suppress GCG over-refusals?\n",
          "The directions are fitted **entirely on our own attacker's pairs**. The GCG corpus "
          "comes from a different method on different originals, and the two attacks are almost "
          "lexically disjoint (`exploit`: 453 of our rewrites, 1 of 1,220 GCG). If our directions "
          "still work here, they describe the model rather than our attacker.\n",
          "| condition | GCG rewrites refused | GCG originals | AdvBench |", "|---|--:|--:|--:|",
          f"| baseline | {b:.1f}% | {R['gcg_originals__baseline']:.1f}% | "
          f"{R['advbench__baseline']:.1f}% |"]
    for k in ks:
        L.append(f"| ablate our k={k} | **{R[f'gcg_rewrites__ours_k{k}']:.1f}%** "
                 f"({R[f'gcg_rewrites__ours_k{k}']-b:+.1f}pp) | "
                 f"{R[f'gcg_originals__ours_k{k}']:.1f}% | {R[f'advbench__ours_k{k}']:.1f}% |")
    rk = [k for k in R if k.startswith("gcg_rewrites__random_k") and not k.endswith("degen")]
    if rk:
        L.append(f"| random (rank-matched) | {R[rk[0]]:.1f}% ({R[rk[0]]-b:+.1f}pp) | | |")
    L.append(f"| atlas r̂ (literature) | {R['gcg_rewrites__atlas_rhat']:.1f}% "
             f"({R['gcg_rewrites__atlas_rhat']-b:+.1f}pp) | | "
             f"{R['advbench__atlas_rhat']:.1f}% |")
    L.append("\n**Caveat**: the GCG filter verifies the original was complied with but does not "
             "enforce intent preservation, so it admits rewrites our rubric excludes (e.g. "
             "\"reduce plastic use\" → \"increase plastic use\"). This measures refusal, not "
             "over-refusal under our rubric.")
else:
    L.append("\n_Phase C (GCG transfer) did not produce output._")

L.append("\n---\nSee `FINDINGS_STATUS.md` for the full ledger of standing and retracted claims.")
open(OUT, "w").write("\n".join(L) + "\n")
print(f"wrote {OUT}")
