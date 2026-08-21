#!/usr/bin/env python3
"""Frame analysis: is over-refusal organised into a small set of reusable 'danger frames'?

The LOW bin is not just cleaner data, it is SUPERVISION: for a one/two-word edit we know
exactly which word flipped the model. Those words are semantic, not arbitrary -- weaponize,
exploit, exfiltrate, covert -- which suggests the model has frame-specific detectors rather
than one global danger detector.

The central test is CROSS-BIN and therefore not circular. Estimate each frame's direction
twice on disjoint data: once from LOW pairs (one-word edits) and once from HIGH pairs
(wholesale paraphrases that happen to introduce the same word). Then ask whether

    cos(u_f^LOW, u_f^HIGH)      [same frame, different edit size]

exceeds

    cos(u_f^LOW, u_g^HIGH)      [different frames]

If the diagonal dominates, frames are real objects of the model, recoverable regardless of
how much of the prompt was rewritten -- which is the actual answer to the HIGH-vs-LOW
question, in interpretable units instead of an abstract rank.

Every comparison is reported against a SPLIT-HALF NOISE FLOOR: with n=15-78 per frame in
LOW, a direction estimate is noisy, and a diagonal of 0.6 means nothing if two halves of the
same frame only agree at 0.6 themselves.

Run: python analyze_frames.py --report FRAME_ANALYSIS.md
"""
import argparse, csv, json, os, re, sys
import numpy as np
csv.field_size_limit(sys.maxsize)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_edit_distance import pair_metrics

FRAMES = {
    "exploitation":  r"exploit(s|ed|ing|able|ation)?$",
    "weaponization": r"weaponi[sz](e|ed|ing|ation)?$",
    "exfiltration":  r"(exfiltrat(e|ed|ing|ion)|payload)$",
    "concealment":   r"(covert(ly)?|untraceable|undetect(ed|able)|stealth(y|ily)?|clandestine)$",
    "intrusion":     r"(infiltrat(e|ed|ing|ion)|breach(ed|ing)?|hack(ed|ing)?)$",
    "forgery":       r"(counterfeit|forge(d|ry)?|fraudulent)$",
}
_COMP = {f: re.compile(p) for f, p in FRAMES.items()}


def frames_of(words):
    out = set()
    for w in (words or "").lower().split():
        for f, p in _COMP.items():
            if p.match(w):
                out.add(f)
    return out


def unit(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (n + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta_dir", default="probe_or/results/delta")
    ap.add_argument("--atlas", default="probe_or/results/llama_signals/probe_absolute.npz")
    ap.add_argument("--layer", type=int, default=17)
    ap.add_argument("--min_n", type=int, default=12, help="min pairs to estimate a frame direction")
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report", default="FRAME_ANALYSIS.md")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    L = a.layer

    acts = np.load(os.path.join(a.delta_dir, "acts.npy"), mmap_mode="r")
    idx = json.load(open(os.path.join(a.delta_dir, "prompt_index.json")))
    rows = list(csv.DictReader(open(os.path.join(a.delta_dir, "prompt_sets.csv"))))
    r_atlas = unit(np.load(a.atlas, allow_pickle=True)["d"].astype(np.float64))[L]

    def vec(t):
        return np.asarray(acts[idx[t.strip()], L, :], dtype=np.float64)

    # ---- build Delta + frame labels for every OR / control pair ----
    data = {}          # set -> list of (delta, frames, original)
    for r in rows:
        s = r["set"]
        if s.startswith("adv"):
            continue
        o, w = r["original"].strip(), r["rewrite"].strip()
        if not o or o not in idx or w not in idx:
            continue
        fs = frames_of(pair_metrics(o, w)["introduced_words"])
        data.setdefault(s, []).append((vec(w) - vec(o), fs, o))
    for s in data:
        print(f"[frames] {s}: {len(data[s])} pairs", flush=True)

    ctrl_all = np.array([d for s in ("ctrl_high", "ctrl_low") for d, _, _ in data.get(s, [])])
    mu_ctrl = ctrl_all.mean(0)          # the generic "was rewritten by this attacker" direction

    def frame_dir(setname, f, subset=None):
        """Mean Delta of frame-f pairs minus the generic rewrite direction."""
        D = [d for i, (d, fs, _) in enumerate(data.get(setname, []))
             if f in fs and (subset is None or i in subset)]
        if len(D) < 3:
            return None, 0
        return unit(np.mean(D, axis=0) - mu_ctrl), len(D)

    def frame_idx(setname, f):
        return [i for i, (_, fs, _) in enumerate(data.get(setname, [])) if f in fs]

    out = ["# Frame analysis — is over-refusal organised into reusable danger frames?\n",
           f"Layer {L}. `Δ = h(rewrite) − h(original)` (cancels topic). A frame direction is the "
           "mean Δ of pairs introducing that frame's vocabulary, minus the mean Δ of all "
           "non-refused control rewrites (which removes the generic 'was rewritten' component, "
           "shared by every pair regardless of refusal).\n",
           "The key test is **cross-bin**: each frame is estimated twice on disjoint pairs — from "
           "LOW (one/two-word edits) and from HIGH (wholesale paraphrases introducing the same "
           "word). A high diagonal means the frame is a real object of the model, recoverable "
           "however much of the prompt changed.\n"]

    # ---- inventory ----
    out += ["\n## Frame inventory\n", "| frame | LOW pairs | HIGH pairs |", "|---|--:|--:|"]
    usable = []
    for f in FRAMES:
        nl, nh = len(frame_idx("or_low", f)), len(frame_idx("or_high", f))
        out.append(f"| {f} | {nl} | {nh} |")
        if nl >= a.min_n and nh >= a.min_n:
            usable.append(f)
    out.append(f"\nFrames with ≥{a.min_n} pairs in **both** bins (usable for the cross-bin test): "
               f"**{', '.join(usable) if usable else 'none'}**")

    # ---- split-half noise floor ----
    out += ["\n## Split-half reliability (the noise floor)\n",
            "Two disjoint halves of the SAME frame in the SAME bin. Any cross-bin cosine must be "
            "read against this: a frame cannot agree with itself across bins more than it agrees "
            "with itself within a bin.\n",
            "| frame | bin | n | split-half cos (mean of "
            f"{a.boot} splits) |", "|---|---|--:|--:|"]
    floor = {}
    for f in usable:
        for s, lab in (("or_low", "LOW"), ("or_high", "HIGH")):
            ii = frame_idx(s, f)
            cs = []
            for _ in range(a.boot):
                p = rng.permutation(ii)
                h1, h2 = set(p[:len(p) // 2]), set(p[len(p) // 2:])
                u1, n1 = frame_dir(s, f, h1)
                u2, n2 = frame_dir(s, f, h2)
                if u1 is not None and u2 is not None:
                    cs.append(float(u1 @ u2))
            floor[(f, s)] = float(np.mean(cs)) if cs else float("nan")
            out.append(f"| {f} | {lab} | {len(ii)} | {floor[(f,s)]:+.3f} |")

    # ---- cross-bin frame matrix: THE test ----
    U_low = {f: frame_dir("or_low", f)[0] for f in usable}
    U_high = {f: frame_dir("or_high", f)[0] for f in usable}
    out += ["\n## Cross-bin frame matrix — cos(u_f from LOW, u_g from HIGH)\n",
            "Rows = frame estimated from one-word edits. Columns = frame estimated from wholesale "
            "paraphrases. **Diagonal should dominate its row and column if frames are real.**\n"]
    out.append("| LOW ↓ / HIGH → | " + " | ".join(usable) + " |")
    out.append("|---" * (len(usable) + 1) + "|")
    for f in usable:
        cells = []
        for g in usable:
            c = float(U_low[f] @ U_high[g])
            cells.append(f"**{c:+.3f}**" if f == g else f"{c:+.3f}")
        out.append(f"| **{f}** | " + " | ".join(cells) + " |")

    diag = [float(U_low[f] @ U_high[f]) for f in usable]
    off = [float(U_low[f] @ U_high[g]) for f in usable for g in usable if f != g]
    if diag and off:
        out.append(f"\nmean diagonal **{np.mean(diag):+.3f}** vs mean off-diagonal "
                   f"**{np.mean(off):+.3f}** (gap {np.mean(diag)-np.mean(off):+.3f})")
        nrow = sum(1 for i, f in enumerate(usable)
                   if all(U_low[f] @ U_high[f] >= U_low[f] @ U_high[g] for g in usable))
        out.append(f"\nframes whose diagonal is the largest entry in its row: **{nrow}/{len(usable)}**")

    # ---- is there frame structure BEYOND a shared danger axis? ----
    or_all = np.array([d for s in ("or_low", "or_high") for d, _, _ in data.get(s, [])])
    shared = unit(or_all.mean(0) - mu_ctrl)
    out += ["\n## Beyond a single shared 'danger' axis\n",
            "Every frame may just be the same global over-refusal direction. Removing the shared "
            "component from each frame direction and re-running the matrix tests whether "
            "frame-specific structure survives.\n",
            "| frame | cos(u_f^LOW, shared) | cos(u_f^HIGH, shared) | residual diagonal cos |",
            "|---|--:|--:|--:|"]
    Ul_r = {f: unit(U_low[f] - (U_low[f] @ shared) * shared) for f in usable}
    Uh_r = {f: unit(U_high[f] - (U_high[f] @ shared) * shared) for f in usable}
    for f in usable:
        out.append(f"| {f} | {float(U_low[f]@shared):+.3f} | {float(U_high[f]@shared):+.3f} | "
                   f"**{float(Ul_r[f]@Uh_r[f]):+.3f}** |")
    rd = [float(Ul_r[f] @ Uh_r[f]) for f in usable]
    ro = [float(Ul_r[f] @ Uh_r[g]) for f in usable for g in usable if f != g]
    if rd and ro:
        out.append(f"\nresidual: mean diagonal **{np.mean(rd):+.3f}** vs off-diagonal "
                   f"**{np.mean(ro):+.3f}**")

    # ---- alignment of each frame with the known refusal direction ----
    out += ["\n## Each frame vs the global refusal direction r̂\n",
            "| frame | cos(u_f^LOW, r̂) | cos(u_f^HIGH, r̂) |", "|---|--:|--:|"]
    for f in usable:
        out.append(f"| {f} | {float(U_low[f]@r_atlas):+.3f} | {float(U_high[f]@r_atlas):+.3f} |")
    out.append(f"| _shared OR axis_ | {float(shared@r_atlas):+.3f} | |")

    # ---- do frame-less HIGH pairs still live in the frame span? ----
    B = np.array([U_high[f] for f in usable])
    Q, _ = np.linalg.qr(B.T)                       # orthonormal basis of the frame span
    def span_frac(D):
        D = np.asarray(D)
        if not len(D):
            return float("nan")
        proj = (D - mu_ctrl) @ Q
        return float(np.mean(np.linalg.norm(proj, axis=1) /
                             (np.linalg.norm(D - mu_ctrl, axis=1) + 1e-9)))
    noframe = [d for d, fs, _ in data.get("or_high", []) if not fs]
    withframe = [d for d, fs, _ in data.get("or_high", []) if fs]
    ctrl_h = [d for d, _, _ in data.get("ctrl_high", [])]
    out += ["\n## Do frame-less over-refusals still live in the frame span?\n",
            f"Fraction of ‖Δ − μ_ctrl‖ inside the {len(usable)}-dimensional frame span.\n",
            "| group | n | fraction in frame span |", "|---|--:|--:|",
            f"| HIGH over-refusals **with** a frame word | {len(withframe)} | {span_frac(withframe):.3f} |",
            f"| HIGH over-refusals **without** a frame word | {len(noframe)} | {span_frac(noframe):.3f} |",
            f"| HIGH matched controls (not refused) | {len(ctrl_h)} | {span_frac(ctrl_h):.3f} |"]

    np.savez(os.path.join(a.delta_dir, "frame_directions.npz"),
             layer=L, frames=np.array(usable),
             u_low=np.array([U_low[f] for f in usable]),
             u_high=np.array([U_high[f] for f in usable]),
             shared=shared, mu_ctrl=mu_ctrl)
    out.append(f"\nFrame directions saved to `{a.delta_dir}/frame_directions.npz` for selective "
               "ablation (Phase 4).")

    open(a.report, "w").write("\n".join(out) + "\n")
    print("\n".join(out))
    print(f"\nwrote {a.report}")


if __name__ == "__main__":
    main()
