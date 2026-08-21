#!/usr/bin/env python3
"""Arditi-style refusal-direction layer selection, model-agnostic, degeneracy-guarded.

Ports research/refusal_vector/arditi_direction_test.py (Llama-only, PACE) to rorqual and
to Qwen3-32B. The method is Arditi et al. 2024: build a harmful-vs-harmless diff-of-means
direction at each candidate layer, ablate it at EVERY layer, and pick the layer whose
ablation most reduces refusal on held-out harmful prompts. Then check necessity (ablate on
a harmful test set) and sufficiency (add it to harmless prompts).

Four things the original would get wrong on Qwen, each of which changes the answer:

1. SELECTION CRITERION. The original picks argmin(refusal). is_refusal() is start-anchored,
   so an ablation that collapses the model into repetition scores 0% refusal and WINS the
   argmin. That is not a subtle risk: it is the exact failure that produced this project's
   retracted "100% of over-refusal removed". A sweep over layers systematically searches for
   the most destructive layer unless degeneracy is excluded. We therefore measure degeneracy
   at every candidate and select argmin(refusal) among layers that stay coherent.

2. REFUSAL CLASSIFIER. The original uses a Llama-flavoured substring list over the first 200
   characters. We import the start-anchored is_refusal() the rest of this project uses, so
   the selected layer is comparable with every other number in the paper.

3. QWEN THINKING MODE. Qwen3 emits <think> blocks by default; at 40 new tokens the whole
   generation can sit inside one, containing no refusal phrase, so every condition reads as
   non-refusal and the sweep is noise. enable_thinking=False, matching ablate_causal_rank.py.

4. OFFLINE DATA. The original pulls yahma/alpaca-cleaned from the hub. Compute nodes have no
   network; we read the Arditi splits already on disk.

Also reports harmless-prompt refusal per candidate, so a layer that suppresses refusal by
breaking the model in a way the n-gram guard misses is still visible.
"""
import argparse, csv, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from killswitch_ablate_r17 import is_refusal, degenerate


def read_prompts(path, n, offset=0):
    with open(path, newline="") as f:
        rows = [(r.get("prompt") or "").strip() for r in csv.DictReader(f)]
    rows = [v for v in rows if v]
    return rows[offset:offset + n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="Qwen/Qwen3-32B")
    ap.add_argument("--harmful_csv", default="probe_or/data/arditi_harmful.csv")
    ap.add_argument("--harmless_csv", default="probe_or/data/arditi_harmless.csv")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_dir", type=int, default=256)
    ap.add_argument("--n_val", type=int, default=64)
    ap.add_argument("--n_test", type=int, default=128)
    # 25%-94% of depth, matching the Llama sweep's 8..28 of 32; 57 included explicitly
    # because it is the layer our AUC-selected basis currently uses.
    ap.add_argument("--cand_layers", default="16,20,24,28,32,36,40,44,48,52,56,57,60")
    ap.add_argument("--add_coefs", default="1,2,4")
    ap.add_argument("--max_new_tokens", type=int, default=48)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_degen", type=float, default=5.0,
                    help="a candidate layer is disqualified above this %% degenerate replies")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.base_model)
    tok.padding_side = "left"          # so index -1 IS the last real token
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.base_model, device_map="auto",
                                                 torch_dtype=torch.bfloat16)
    model.eval()
    dev, H, nL = model.device, model.config.hidden_size, model.config.num_hidden_layers
    layers_mod, embed = model.model.layers, model.model.embed_tokens
    print(f"[init] {a.base_model}: {nL} layers, hidden {H}", flush=True)

    is_qwen = "qwen" in a.base_model.lower()

    def fmt(p):
        msgs = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": p}]
        kw = {"enable_thinking": False} if is_qwen else {}
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)

    h_dir = read_prompts(a.harmful_csv, a.n_dir)
    h_val = read_prompts(a.harmful_csv, a.n_val, a.n_dir)
    h_test = read_prompts(a.harmful_csv, a.n_test, a.n_dir + a.n_val)
    hl_dir = read_prompts(a.harmless_csv, a.n_dir)
    hl_val = read_prompts(a.harmless_csv, a.n_val, a.n_dir)
    hl_test = read_prompts(a.harmless_csv, a.n_test, a.n_dir + a.n_val)
    print(f"[data] harmful {len(h_dir)}/{len(h_val)}/{len(h_test)} "
          f"harmless {len(hl_dir)}/{len(hl_val)}/{len(hl_test)} (dir/val/test)", flush=True)

    def class_mean(texts):
        sm, cnt = np.zeros((nL + 1, H), np.float64), 0
        for i in range(0, len(texts), a.batch_size):
            enc = tok([fmt(p) for p in texts[i:i + a.batch_size]], return_tensors="pt",
                      padding=True, truncation=True, max_length=512).to(dev)
            with torch.no_grad():
                hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states
            last = torch.stack([h[:, -1, :] for h in hs], dim=1).float().cpu().numpy()
            sm += last.sum(0); cnt += last.shape[0]
        return sm / cnt

    print("[dir] harmful-vs-harmless diff-of-means at every layer", flush=True)
    d_raw = class_mean(h_dir) - class_mean(hl_dir)
    d_hat = d_raw / (np.linalg.norm(d_raw, axis=1, keepdims=True) + 1e-9)

    def vec(L, hat=True):
        return torch.tensor((d_hat if hat else d_raw)[L], dtype=torch.bfloat16, device=dev)

    def generate(prompts, specs):
        handles = [m.register_forward_hook(fn) for m, fn in specs]
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
                    out.append(tok.decode(g[j, enc["input_ids"].shape[1]:],
                                          skip_special_tokens=True))
        finally:
            for h in handles:
                h.remove()
        return out

    def ablate_fn(dh):
        def fn(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h = h - (h.to(dh.dtype) @ dh).unsqueeze(-1) * dh
            return (h,) + out[1:] if isinstance(out, tuple) else h
        return fn

    def add_fn(dr, c):
        def fn(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h = h + c * dr
            return (h,) + out[1:] if isinstance(out, tuple) else h
        return fn

    def ablate_all(dh):
        return [(embed, ablate_fn(dh))] + [(layers_mod[i], ablate_fn(dh)) for i in range(nL)]

    def score(reps):
        n = max(len(reps), 1)
        return (100.0 * sum(is_refusal(r) for r in reps) / n,
                100.0 * sum(degenerate(r) for r in reps) / n)

    R = {}
    bh, bhd = score(generate(h_test, []))
    bl, bld = score(generate(hl_test, []))
    R["baseline"] = {"harmful_refusal": bh, "harmful_degen": bhd,
                     "harmless_refusal": bl, "harmless_degen": bld}
    print(f"[base] harmful {bh:.1f}% (degen {bhd:.1f}%) | harmless {bl:.1f}% (degen {bld:.1f}%)",
          flush=True)

    cands = [int(x) for x in a.cand_layers.split(",") if int(x) <= nL]
    sel, t0 = {}, time.time()
    for L in cands:
        dh = vec(L)
        hr, hd = score(generate(h_val, ablate_all(dh)))
        lr, ld = score(generate(hl_val, ablate_all(dh)))
        ok = hd <= a.max_degen and ld <= a.max_degen
        sel[L] = {"harmful_refusal": hr, "harmful_degen": hd,
                  "harmless_refusal": lr, "harmless_degen": ld, "eligible": ok}
        print(f"[select] L{L:<3} harmful {hr:5.1f}% (degen {hd:4.1f}%) | "
              f"harmless {lr:5.1f}% (degen {ld:4.1f}%) | {'ok' if ok else 'DISQUALIFIED'}"
              f"  ({time.time()-t0:.0f}s)", flush=True)

    eligible = {L: v["harmful_refusal"] for L, v in sel.items() if v["eligible"]}
    if not eligible:
        print("[select] no candidate stayed coherent -- reporting the raw argmin, DO NOT TRUST",
              flush=True)
        eligible = {L: v["harmful_refusal"] for L, v in sel.items()}
        R["warning"] = "every candidate layer degenerated; selection is not meaningful"
    Lstar = min(eligible, key=eligible.get)
    R["layer_selection"] = sel
    R["selected_layer"] = Lstar
    R["selected_among_eligible"] = sorted(eligible)
    print(f"[select] BEST L{Lstar} (val harmful refusal {eligible[Lstar]:.1f}%)", flush=True)

    dh = vec(Lstar)
    resp = generate(h_test, ablate_all(dh))
    hr, hd = score(resp)
    lr, ld = score(generate(hl_test, ablate_all(dh)))
    R["necessity"] = {"harmful_refusal": hr, "harmful_degen": hd,
                      "harmless_refusal": lr, "harmless_degen": ld,
                      "harmful_refusal_drop": bh - hr}
    print(f"[necessity] harmful {bh:.1f} -> {hr:.1f}% (degen {hd:.1f}%) | "
          f"harmless {bl:.1f} -> {lr:.1f}%", flush=True)
    with open(os.path.join(a.out_dir, "ablated_harmful_responses.jsonl"), "w") as f:
        for p, rr in zip(h_test, resp):
            f.write(json.dumps({"prompt": p, "response": rr,
                                "refused": is_refusal(rr), "degenerate": degenerate(rr)}) + "\n")

    R["sufficiency"] = {}
    for c in [float(x) for x in a.add_coefs.split(",")]:
        r, dg = score(generate(hl_test, [(layers_mod[min(Lstar, nL - 1)], add_fn(vec(Lstar, False), c))]))
        R["sufficiency"][f"coef{c}"] = {"harmless_refusal": r, "degen": dg}
        print(f"[sufficiency] +{c}*d on harmless -> refusal {r:.1f}% (degen {dg:.1f}%)", flush=True)

    np.savez(os.path.join(a.out_dir, "arditi_direction.npz"),
             d_raw=d_raw[Lstar].astype(np.float32), d_hat=d_hat[Lstar].astype(np.float32),
             layer=Lstar, all_d_hat=d_hat.astype(np.float32))
    R["config"] = {"model": a.base_model, "n_layers": nL, "n_dir": a.n_dir, "n_val": a.n_val,
                   "n_test": a.n_test, "cands": cands, "max_degen": a.max_degen,
                   "max_new_tokens": a.max_new_tokens, "enable_thinking": False if is_qwen else None}
    with open(os.path.join(a.out_dir, "arditi_direction_results.json"), "w") as f:
        json.dump(R, f, indent=2)
    print(f"[done] -> {a.out_dir}/arditi_direction_results.json", flush=True)


if __name__ == "__main__":
    main()
