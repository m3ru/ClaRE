#!/usr/bin/env python3
"""Generate responses to the injected prompts from build_injections.py.

Deliberately generic: it takes the arms file and a model, and writes one response per
(arm, prompt). All the experimental design lives in build_injections.py, so this stays a
thin generation loop that can be pointed at any target model.

max_new_tokens defaults to 160 -- enough for a judge to tell a refusal from a hedged
compliance, without paying for full answers we are not grading on content.

Run:
  python gen_injections.py --base_model meta-llama/Meta-Llama-3-8B-Instruct \
      --out probe_or/results/injections_llama.json
"""
import argparse, json, os, sys, time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--injections", default="probe_or/results/injections.json")
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=160)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--disable_thinking", action="store_true",
                    help="Qwen3 emits reasoning by default; off for a fair refusal read")
    ap.add_argument("--attn_implementation", default="sdpa")
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    data = json.load(open(a.injections))
    items = [(arm, it) for arm, its in data["arms"].items() for it in its]
    print(f"[data] {len(data['arms'])} arms x {data['n_prompts']} prompts "
          f"= {len(items)} generations", flush=True)

    tok = AutoTokenizer.from_pretrained(a.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        a.base_model, dtype=torch.bfloat16, device_map="auto",
        attn_implementation=a.attn_implementation).eval()
    print(f"[model] {a.base_model} on {model.device}", flush=True)

    kw = {"add_generation_prompt": True, "tokenize": False}
    if a.disable_thinking:
        kw["enable_thinking"] = False

    out, t0 = [], time.time()
    for s in range(0, len(items), a.batch_size):
        chunk = items[s:s + a.batch_size]
        texts = [tok.apply_chat_template([{"role": "user", "content": it["prompt"]}], **kw)
                 for _, it in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=1024, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=a.max_new_tokens, do_sample=False,
                               pad_token_id=tok.pad_token_id)
        for j, (arm, it) in enumerate(chunk):
            resp = tok.decode(g[j, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            out.append({"arm": arm, "idx": it["idx"], "original": it["original"],
                        "prompt": it["prompt"], "response": resp.strip()})
        if s % (a.batch_size * 20) == 0:
            done = s + len(chunk)
            el = time.time() - t0
            print(f"  {done}/{len(items)}  {el:.0f}s  eta {el/max(done,1)*(len(items)-done):.0f}s",
                  flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"base_model": a.base_model, "n": len(out), "rows": out}, open(a.out, "w"), indent=1)
    print(f"[done] wrote {a.out} ({len(out)} responses, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
