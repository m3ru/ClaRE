#!/usr/bin/env python3
"""Frame directions orthogonalised against the shared axis ONLY — symmetric across frames.

The sequential basis in build_causal_dirs.py Gram-Schmidts each frame against everything already
placed, in order of frame size. That is required there because those directions get ablated
JOINTLY (rank-k), and the projection h - (h B^T)B is only a projection when B is orthonormal.

But it makes the frames incomparable to each other. Measured fraction of each frame direction
that survives sequential deflation:

    Llama   exploitation 2.3%  concealment 9.3%  weaponization 13.1%  intrusion 9.4%  exfiltration 36.2%
    Qwen    exploitation 2.5%  concealment 4.4%  weaponization  7.1%  intrusion 6.8%  exfiltration 22.4%

Whoever goes first is nearly annihilated by the shared axis; whoever goes last keeps the most.
So "d4 weaponization" actually means "weaponization minus shared, exploitation and concealment",
and a cross-model comparison of basis POSITIONS is not apples-to-apples. On Qwen the frame that
survives deflation best (exfiltration) is also the one that ablates best, so that result is
specifically exposed to the confound.

Here every frame is treated identically:

    r_f = unit( u_f - (u_f . d1) d1 ),   u_f = unit( mean(Delta over frame-f pairs) - mean(Delta over controls) )

Frames are NOT orthogonalised against each other, so they remain mutually correlated — which is
fine and intended, because each is ablated ALONE (rank-1), and a single unit vector is trivially
orthonormal. This isolates "what does frame f add beyond the shared axis", symmetrically, with no
ordering.

Fitted on TRAIN originals; held-out lists are copied through unchanged so evaluation stays clean.

Run: python build_symmetric_frame_dirs.py --delta_dir probe_or/results/delta --layer 17 --out ...
"""
import argparse, csv, json, os, sys
import numpy as np
csv.field_size_limit(sys.maxsize)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_edit_distance import pair_metrics
from analyze_frames import frames_of as _frames_of_default, FRAMES as _FRAMES_DEFAULT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta_dir", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--src_dirs", required=True, help="sequential basis, for held-out lists")
    ap.add_argument("--frames_json", default="",
                    help="per-model frame regex JSON. Empty = the default (Llama-derived) frames. "
                         "For a fair cross-model comparison each model uses frames mined from its "
                         "OWN low-edit triggers.")
    ap.add_argument("--min_n", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import re as _re
    if a.frames_json:
        FR_DEF = json.load(open(a.frames_json))
        _COMP = {f: _re.compile(p) for f, p in FR_DEF.items()}
        def frames_of(words):
            out = set()
            for wd in (words or "").lower().split():
                for f, c in _COMP.items():
                    if c.match(wd):
                        out.add(f)
            return out
        FRAMES = FR_DEF
        print(f"[sym] using {len(FRAMES)} custom frames from {a.frames_json}: {list(FRAMES)}")
    else:
        frames_of, FRAMES = _frames_of_default, _FRAMES_DEFAULT
    acts = np.load(os.path.join(a.delta_dir, "acts.npy"), mmap_mode="r")
    idx = json.load(open(os.path.join(a.delta_dir, "prompt_index.json")))
    rows = list(csv.DictReader(open(os.path.join(a.delta_dir, "prompt_sets.csv"))))
    S = np.load(a.src_dirs, allow_pickle=True)
    train = {str(x) for x in S["train_originals"]}
    A = np.asarray(acts[:, a.layer, :], dtype=np.float64)
    u = lambda x: x / (np.linalg.norm(x) + 1e-9)

    fr = {f: [] for f in FRAMES}; ct = []; allor = []
    for r in rows:
        s = r["set"]; o, w = r["original"].strip(), r["rewrite"].strip()
        if o not in train or o not in idx or w not in idx:
            continue
        d = A[idx[w]] - A[idx[o]]
        if s.startswith("ctrl_"):
            ct.append(d); continue
        if not s.startswith("or_"):
            continue
        allor.append(d)
        for f in frames_of(pair_metrics(o, w)["introduced_words"]):
            fr[f].append(d)

    mu_ct = np.mean(ct, axis=0)
    d1 = u(np.mean(allor, axis=0) - mu_ct)
    keep = [f for f in sorted(fr, key=lambda z: -len(fr[z])) if len(fr[f]) >= a.min_n]
    dirs, labels = [d1], ["d1_shared"]
    print(f"[sym] {len(allor)} OR / {len(ct)} ctrl train pairs @L{a.layer}")
    for f in keep:
        raw = u(np.mean(fr[f], axis=0) - mu_ct)
        res = raw - (raw @ d1) * d1
        n = np.linalg.norm(res)
        dirs.append(u(res)); labels.append(f"sym_{f}")
        print(f"   {f:15s} n={len(fr[f]):4d}  cos(raw,d1)={raw@d1:+.3f}  "
              f"||residual||={n:.3f} ({n*n*100:.1f}% new)")

    D = np.array(dirs)
    print("\n[sym] mutual cosines among frame residuals (NOT orthogonalised to each other):")
    for i in range(1, len(D)):
        print("   " + labels[i][:18].ljust(18) +
              " ".join(f"{float(D[i]@D[j]):+.2f}" for j in range(1, len(D))))
    np.savez(a.out, dirs=D, layer=a.layer, labels=np.array(labels, dtype=object),
             heldout_originals=S["heldout_originals"], heldout_rewrites=S["heldout_rewrites"],
             train_originals=S["train_originals"])
    print(f"\n[sym] wrote {a.out}  ({len(D)} directions)")


if __name__ == "__main__":
    main()
