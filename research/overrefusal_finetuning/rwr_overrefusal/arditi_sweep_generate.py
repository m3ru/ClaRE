#!/usr/bin/env python3
"""Phase 1 of a JUDGED Arditi layer sweep: generate, classify nothing.

Why this is split. The selection criterion is "which layer's ablation most reduces refusal
on harmful prompts", so it is only as good as the refusal detector. Our start-anchored
is_refusal() was built to catch over-refusal ONSET on benign rewrites, where a refusal really
does begin "I cannot". Reusing it here asks a different question -- does the model still
refuse? -- and it answers wrongly whenever the model refuses by moralising instead of by
opening with the phrase. On Qwen that mislabelled 53% of still-refusing responses as
compliance and drove the argmin to a near-output phrasing direction (L60, cos 0.19 with the
unembedding refusal direction, vs 0.015 for Llama's genuine L12).

The fix is an LLM judge, but compute nodes have no internet. So: this script only GENERATES
and dumps raw text; arditi_sweep_judge.py scores it on a login node and picks the layer.

Emits one JSONL row per (condition, prompt) with the full response text.
"""
import argparse, csv, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


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
    ap.add_argument("--cand_layers", default="16,20,24,28,32,36,40,44,48,52,56,57,60")
    ap.add_argument("--max_new_tokens", type=int, default=96,
                    help="longer than the regex sweep's 48: a judge needs enough text to tell "
                         "a moralising refusal from a genuine answer")
    ap.add_argument("--batch_size", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.base_model)
    tok.padding_side = "left"                      # index -1 IS the last real token
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.base_model, device_map="auto",
                                                 torch_dtype=torch.bfloat16).eval()
    dev, H, nL = model.device, model.config.hidden_size, model.config.num_hidden_layers
    layers_mod, embed = model.model.layers, model.model.embed_tokens
    is_qwen = "qwen" in a.base_model.lower()
    print(f"[init] {a.base_model}: {nL} layers, hidden {H}", flush=True)

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
    dn = np.linalg.norm(d_raw, axis=1, keepdims=True)
    dn[dn == 0] = 1.0                              # embedding row is exactly 0; avoid /0
    d_hat = d_raw / dn
    np.savez(os.path.join(a.out_dir, "all_directions.npz"),
             d_raw=d_raw.astype(np.float32), all_d_hat=d_hat.astype(np.float32))

    def ablate_fn(dh):
        def fn(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h = h - (h.to(dh.dtype) @ dh).unsqueeze(-1) * dh
            return (h,) + out[1:] if isinstance(out, tuple) else h
        return fn

    def gen(prompts, specs):
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
                    out.append(tok.decode(g[j, enc["input_ids"].shape[1]:], skip_special_tokens=True))
        finally:
            for h in handles:
                h.remove()
        return out

    def all_layer(dh):
        return [(embed, ablate_fn(dh))] + [(layers_mod[i], ablate_fn(dh)) for i in range(nL)]

    fp = open(os.path.join(a.out_dir, "generations.jsonl"), "w")
    def dump(cond, split, prompts, specs):
        t0 = time.time()
        for p, r in zip(prompts, gen(prompts, specs)):
            fp.write(json.dumps({"condition": cond, "split": split,
                                 "prompt": p, "response": r}) + "\n")
        fp.flush()
        print(f"  [{cond}] {split} n={len(prompts)}  ({time.time()-t0:.0f}s)", flush=True)

    dump("baseline", "harmful_test", h_test, [])
    dump("baseline", "harmless_test", hl_test, [])
    for L in [int(x) for x in a.cand_layers.split(",") if int(x) <= nL]:
        v = torch.tensor(d_hat[L], dtype=torch.bfloat16, device=dev)
        dump(f"ablate_L{L}", "harmful_val", h_val, all_layer(v))
        dump(f"ablate_L{L}", "harmless_val", hl_val, all_layer(v))
    fp.close()
    json.dump({"model": a.base_model, "n_layers": nL,
               "cands": [int(x) for x in a.cand_layers.split(",") if int(x) <= nL],
               "max_new_tokens": a.max_new_tokens,
               "n_val": a.n_val, "n_test": a.n_test},
              open(os.path.join(a.out_dir, "meta.json"), "w"), indent=1)
    print(f"[done] -> {a.out_dir}/generations.jsonl", flush=True)


if __name__ == "__main__":
    main()
