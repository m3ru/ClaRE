#!/usr/bin/env python3
"""Generate FULL responses for safety grading — refusal-onset is not a harm measure.

Every safety number in this project so far came from a start-anchored refusal regex on 48
tokens. That answers "did the reply begin with a refusal", which is NOT the same as "was the
reply harmful". A model that stops saying "I cannot" and then produces hedged, useless text
scores as a safety failure under the regex when nothing harmful happened; a model that
complies helpfully scores the same as one that complies uselessly.

StrongREJECT exists to fix exactly this: it grades the RESPONSE (refusal, convincingness,
specificity), so it needs enough tokens to judge. Hence max_new_tokens defaults to 256 here,
not the 48 used for refusal detection elsewhere.

Two eval sets, for two different questions:
  strongreject (313 prompts, 6 categories) -- the safety headline. Purpose-built to fix
      AdvBench's weaknesses, and graded on response harmfulness rather than refusal onset.
  xstest_unsafe (200) -- the 9 prompts that flipped refuse->comply under d4 ablation are the
      specific thing we need to look at: are they genuine compliance or hedged non-answers?
      All responses are saved so the flips can be identified and graded individually.

Responses are retained in the output JSON for grading only.

Run: python gen_safety_responses.py --out probe_or/results/safety_responses.json
"""
import argparse, csv, json, os, sys
import numpy as np
csv.field_size_limit(sys.maxsize)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from killswitch_ablate_r17 import is_refusal, degenerate

SYSTEM = "You are a helpful assistant."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--dirs_npz", default="probe_or/results/delta/causal_dirs.npz")
    ap.add_argument("--atlas", default="probe_or/results/llama_signals/probe_absolute.npz")
    ap.add_argument("--strongreject", default="probe_or/data/external/strongreject_dataset.csv")
    ap.add_argument("--xstest", default="probe_or/data/external/xstest_v2.csv")
    ap.add_argument("--d4_idx", type=int, default=3,
                    help="0-based index of the candidate selective direction. Llama: 3 "
                         "(d4, weaponization residual). Qwen: 5 (d6, exfiltration residual) -- "
                         "found by the single-direction search, NOT by analogy to Llama.")
    ap.add_argument("--dir_label", default="d4")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="probe_or/results/safety_responses.json")
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    Z = np.load(a.dirs_npz, allow_pickle=True)
    dirs = Z["dirs"].astype(np.float64); L = int(Z["layer"])
    sr = [r["forbidden_prompt"] for r in csv.DictReader(open(a.strongreject))]
    sr_cat = [r["category"] for r in csv.DictReader(open(a.strongreject))]
    xs = [r for r in csv.DictReader(open(a.xstest)) if r["label"] == "unsafe"]
    evalsets = {"strongreject": sr, "xstest_unsafe": [r["prompt"] for r in xs]}
    for k, v in evalsets.items():
        print(f"[sets] {k}: {len(v)}", flush=True)

    hf = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    tok = AutoTokenizer.from_pretrained(a.base_model, token=hf)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.base_model, token=hf, device_map="auto",
                                                 dtype=torch.bfloat16).eval()
    dev, nL = model.device, model.config.num_hidden_layers
    layers_mod, embed = model.model.layers, model.model.embed_tokens

    def fmt(p):
        return tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def ablate_fn(B):
        def fn(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h = h - (h.to(B.dtype) @ B.T) @ B
            return (h,) + out[1:] if isinstance(out, tuple) else h
        return fn

    def all_layer(M):
        B = torch.tensor(np.atleast_2d(M), dtype=torch.bfloat16, device=dev)
        return [(embed, ablate_fn(B))] + [(layers_mod[i], ablate_fn(B)) for i in range(nL)]

    def generate(prompts, specs):
        hs_ = [m.register_forward_hook(f) for m, f in specs]
        out = []
        try:
            for i in range(0, len(prompts), a.batch_size):
                b = prompts[i:i + a.batch_size]
                enc = tok([fmt(p) for p in b], return_tensors="pt", padding=True,
                          truncation=True, max_length=512).to(dev)
                with torch.no_grad():
                    g = model.generate(**enc, max_new_tokens=a.max_new_tokens, do_sample=False,
                                       pad_token_id=tok.pad_token_id)
                for j in range(len(b)):
                    out.append(tok.decode(g[j, enc["input_ids"].shape[1]:], skip_special_tokens=True))
        finally:
            for h in hs_:
                h.remove()
        return out

    rng = np.random.default_rng(a.seed)
    Q, _ = np.linalg.qr(rng.standard_normal((dirs.shape[1], 1)))
    r_atlas = np.load(a.atlas, allow_pickle=True)["d"].astype(np.float64)[L]
    conds = [("baseline", None), (f"ablate_{a.dir_label}", dirs[a.d4_idx]), ("ablate_d1", dirs[0]),
             ("ablate_atlas_rhat", r_atlas / np.linalg.norm(r_atlas)), ("ablate_random", Q.T[:1])]

    out = {"prompts": evalsets, "categories": {"strongreject": sr_cat}, "responses": {}, "regex": {}}
    for cname, M in conds:
        specs = [] if M is None else all_layer(M)
        for ename, prompts in evalsets.items():
            reps = generate(prompts, specs)
            out["responses"][f"{ename}__{cname}"] = reps
            rr = 100.0 * sum(is_refusal(x) for x in reps) / max(len(reps), 1)
            dg = 100.0 * sum(degenerate(x) for x in reps) / max(len(reps), 1)
            out["regex"][f"{ename}__{cname}"] = {"refusal_pct": rr, "degen_pct": dg}
            print(f"  {cname:18s} {ename:14s} regex-refusal {rr:6.2f}%  degen {dg:5.2f}%", flush=True)

    # the specific prompts that flipped under d4 -- what needs grading
    for ename in evalsets:
        b = out["responses"][f"{ename}__baseline"]; d = out["responses"][f"{ename}__ablate_{a.dir_label}"]
        flips = [i for i in range(len(b)) if is_refusal(b[i]) and not is_refusal(d[i])]
        out.setdefault("d4_flips", {})[ename] = flips
        print(f"[flips] {ename}: {len(flips)} prompts refused at baseline, not after d4 ablation",
              flush=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"[done] {a.out}  (max_new_tokens={a.max_new_tokens} -- long enough to grade)")


if __name__ == "__main__":
    main()
