#!/usr/bin/env python3
"""Is d4 a LEXICAL-ALARM direction or a HARM direction?

d4 (the weaponization residual) removes ~35pp of over-refusal while leaving AdvBench refusal
essentially untouched. Two explanations fit that equally well so far:

  H1 LEXICAL ALARM  d4 tracks alarming SURFACE FORM, not assessed harm. Our over-refusal
                    prompts are benign asks with scary words spliced in, so they load on it;
                    AdvBench is genuinely harmful but plainly phrased, so it does not.
  H2 HARM           d4 is a (category-specific) harm direction after all.

These make OPPOSITE predictions in a 2x2 of {benign, harmful} x {plain, alarming}:

              plain          alarming
  benign      low            HIGH under H1, low under H2
  harmful     HIGH under H2  high under both

So the test is which FACTOR drives the projection: alarming-ness or harmfulness. A design that
only compared over-refusal prompts to AdvBench could not tell them apart, because those differ
on both factors at once.

Cells (all observational -- nothing harmful is generated):
  benign+plain     originals from the low-edit bin, which the model complies with
  benign+alarming  their matched rewrites: SAME prompt, 1-2 alarming words swapped in. This is
                   the cleanest possible manipulation of surface form with content held fixed.
  harmful+alarming AdvBench prompts that contain alarming vocabulary
  harmful+plain    AdvBench prompts that do not

We also fix a measurement error while we are here. Zhao et al. (2025), "LLMs Encode Harmfulness
and Refusal Separately", show harmfulness is encoded at t_inst (last token of the user's
instruction) while refusal lives at t_post-inst (last token of the full prompt, after the
special tokens). Everything in this project so far was read at t_post-inst -- the REFUSAL
position -- which is why our "harm direction" came out at cos 0.78 with the refusal direction:
we had measured two refusal-flavoured things. Here we read BOTH positions, and fit the harm
direction where harmfulness actually lives.

Run: python probe_d4_alarm.py --out probe_or/results/d4_alarm_probe.json
"""
import argparse, csv, json, os, re, sys
import numpy as np
csv.field_size_limit(sys.maxsize)

SYSTEM = "You are a helpful assistant."

# Vocabulary that makes a request SOUND dangerous. Drawn from the trigger words the low-edit
# analysis surfaced, plus the obvious neighbours. Used only to SPLIT existing prompts.
ALARM = re.compile(r"\b(weaponi[sz]\w*|exploit\w*|exfiltrat\w*|covert\w*|untraceable|"
                   r"infiltrat\w*|counterfeit\w*|malicious\w*|illegal\w*|hack\w*|steal\w*|"
                   r"bomb\w*|weapon\w*|attack\w*|destroy\w*|undetect\w*|evade|evading|"
                   r"bypass\w*|smuggl\w*|forge\w*|sabotag\w*|poison\w*|manipulat\w*)\b", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--dirs_npz", default="probe_or/results/delta/causal_dirs.npz")
    ap.add_argument("--atlas", default="probe_or/results/llama_signals/probe_absolute.npz")
    ap.add_argument("--low_bin", default="probe_or/results/edit_strata/or_low_stratum_v7.csv")
    ap.add_argument("--advbench", default="probe_or/data/advbench_harmful_behaviors.csv")
    ap.add_argument("--benign_ref", default="probe_or/data/arditi_harmless.csv")
    ap.add_argument("--d4_idx", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--out", default="probe_or/results/d4_alarm_probe.json")
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    Z = np.load(a.dirs_npz, allow_pickle=True)
    dirs = Z["dirs"].astype(np.float64); L = int(Z["layer"])

    # ---- cells ----
    low = [r for r in csv.DictReader(open(a.low_bin))]
    benign_plain = sorted({r["original"].strip() for r in low if r["original"].strip()})
    benign_alarm = [r["rewrite"].strip() for r in low if r["rewrite"].strip()]
    adv = []
    with open(a.advbench) as f:
        for r in csv.DictReader(f):
            v = (r.get("goal") or "").strip()
            if v:
                adv.append(v)
    harm_alarm = [p for p in adv if ALARM.search(p)]
    harm_plain = [p for p in adv if not ALARM.search(p)]
    ben_ref = []
    with open(a.benign_ref) as f:
        for r in csv.DictReader(f):
            v = (r.get("prompt") or "").strip()
            if v:
                ben_ref.append(v)

    cells = {"benign_plain": benign_plain, "benign_alarm": benign_alarm,
             "harmful_plain": harm_plain, "harmful_alarm": harm_alarm,
             "_benign_ref": ben_ref[:400]}
    for k, v in cells.items():
        print(f"[cells] {k:16s} n={len(v)}", flush=True)
    if min(len(harm_plain), len(harm_alarm)) < 30:
        print("[warn] AdvBench alarm split is lopsided; harmful factor will be weak", flush=True)

    hf = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    tok = AutoTokenizer.from_pretrained(a.base_model, token=hf)
    tok.padding_side = "right"          # index arithmetic below assumes right padding
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.base_model, token=hf, device_map="auto",
                                                 dtype=torch.bfloat16).eval()

    def fmt(p, gen):
        return tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}],
            tokenize=False, add_generation_prompt=gen, enable_thinking=False)

    def acts_two_positions(prompts):
        """-> (A_inst, A_post): [n, H] each, at t_inst and t_post-inst."""
        out_i, out_p = [], []
        for i in range(0, len(prompts), a.batch_size):
            b = prompts[i:i + a.batch_size]
            enc = tok([fmt(p, True) for p in b], return_tensors="pt", padding=True,
                      truncation=True, max_length=512, add_special_tokens=False).to(model.device)
            # length WITHOUT the generation prompt tells us where the instruction ends:
            # [prefix][instruction][<|eot_id|>] -> last content token is len_no_gen - 2
            lens = [len(tok(fmt(p, False), add_special_tokens=False)["input_ids"]) for p in b]
            with torch.no_grad():
                hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states[L]
            am = enc["attention_mask"].to(torch.int)
            post = (am.sum(1) - 1).tolist()
            for j in range(len(b)):
                ti = max(0, min(lens[j] - 2, post[j]))
                out_i.append(hs[j, ti, :].float().cpu().numpy())
                out_p.append(hs[j, post[j], :].float().cpu().numpy())
        return np.array(out_i, dtype=np.float64), np.array(out_p, dtype=np.float64)

    A = {}
    for k, v in cells.items():
        A[k] = acts_two_positions(v)
        print(f"[acts] {k} done ({len(v)})", flush=True)

    u = lambda x: x / (np.linalg.norm(x) + 1e-9)
    d4 = u(dirs[a.d4_idx]); d1 = u(dirs[0])
    r_atlas = u(np.load(a.atlas, allow_pickle=True)["d"].astype(np.float64)[L])

    # harm direction fitted where harmfulness actually lives (t_inst), and -- for contrast --
    # at the refusal position where we had been reading it
    harm_all = np.vstack([A["harmful_plain"][0], A["harmful_alarm"][0]])
    harm_inst = u(harm_all.mean(0) - A["_benign_ref"][0].mean(0))
    harm_post = u(np.vstack([A["harmful_plain"][1], A["harmful_alarm"][1]]).mean(0)
                  - A["_benign_ref"][1].mean(0))

    print("\n=== direction geometry (the measurement-position fix) ===")
    print(f"  cos(harm@t_inst,  harm@t_post-inst) = {harm_inst @ harm_post:+.3f}")
    print(f"  cos(harm@t_inst,  r_atlas)          = {harm_inst @ r_atlas:+.3f}")
    print(f"  cos(harm@t_post,  r_atlas)          = {harm_post @ r_atlas:+.3f}   <- what we had")
    print(f"  cos(d4, harm@t_inst)                = {d4 @ harm_inst:+.3f}   <- the real test")
    print(f"  cos(d4, harm@t_post)                = {d4 @ harm_post:+.3f}")
    print(f"  cos(d4, r_atlas)                    = {d4 @ r_atlas:+.3f}")

    def z(vecs, direction, ref):
        p = vecs @ direction
        return (p - ref.mean()) / (ref.std() + 1e-9)

    ref_post = A["benign_plain"][1] @ d4
    ref_inst = A["benign_plain"][0] @ harm_inst
    print("\n=== 2x2: projection onto each direction (z vs benign_plain) ===")
    hdr = f"{'cell':16s} {'n':>5s} {'d4@post':>9s} {'r_atlas@post':>13s} {'harm@inst':>11s}"
    print(hdr)
    R = {}
    for k in ("benign_plain", "benign_alarm", "harmful_plain", "harmful_alarm"):
        Ai, Ap = A[k]
        zd4 = float(z(Ap, d4, ref_post).mean())
        zra = float(z(Ap, r_atlas, A["benign_plain"][1] @ r_atlas).mean())
        zhi = float(z(Ai, harm_inst, ref_inst).mean())
        R[k] = dict(n=len(Ap), d4=zd4, r_atlas=zra, harm_inst=zhi)
        print(f"{k:16s} {len(Ap):5d} {zd4:9.2f} {zra:13.2f} {zhi:11.2f}")

    def eff(key):
        alarm = ((R["benign_alarm"][key] - R["benign_plain"][key]) +
                 (R["harmful_alarm"][key] - R["harmful_plain"][key])) / 2
        harm = ((R["harmful_plain"][key] - R["benign_plain"][key]) +
                (R["harmful_alarm"][key] - R["benign_alarm"][key])) / 2
        return alarm, harm

    print("\n=== FACTORIAL DECOMPOSITION (main effects, in z units) ===")
    print(f"{'direction':14s} {'ALARM effect':>14s} {'HARM effect':>13s}   verdict")
    for key, lbl in (("d4", "d4"), ("r_atlas", "r_atlas"), ("harm_inst", "harm@t_inst")):
        al, ha = eff(key)
        v = ("driven by ALARM" if abs(al) > 1.5 * abs(ha) else
             "driven by HARM" if abs(ha) > 1.5 * abs(al) else "mixed / neither dominant")
        print(f"{lbl:14s} {al:14.2f} {ha:13.2f}   {v}")
    print("\nH1 (lexical alarm) predicts d4 is ALARM-driven while harm@t_inst is HARM-driven.")
    print("H2 (harm direction) predicts d4 tracks harm@t_inst.")

    json.dump({"cells": R, "geometry": {
        "cos_harm_inst_post": float(harm_inst @ harm_post),
        "cos_harm_inst_ratlas": float(harm_inst @ r_atlas),
        "cos_harm_post_ratlas": float(harm_post @ r_atlas),
        "cos_d4_harm_inst": float(d4 @ harm_inst),
        "cos_d4_harm_post": float(d4 @ harm_post),
        "cos_d4_ratlas": float(d4 @ r_atlas)}},
        open(a.out, "w"), indent=1)
    print(f"\n[done] {a.out}")


if __name__ == "__main__":
    main()
