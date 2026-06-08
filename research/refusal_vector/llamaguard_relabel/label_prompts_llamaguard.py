#!/usr/bin/env python3
"""Label prompts by Llama-Guard-3-8B's OWN safe/unsafe verdict.

This replaces the original split (which labeled prompts by whether *Llama-3*
refused) with the classifier's actual decision. We read the prompt column of
the existing llama8b_outputs_filtered.csv (the exact pool the original vector
used), run each prompt through Llama-Guard's moderation template (user turn
only), and record:

    prompt, label ("safe"/"unsafe"), p_unsafe (best-effort soft score)

Then we write the two prompt lists the extractor consumes:
    lg_unsafe_prompts.csv  (Llama-Guard says unsafe  -> "refusal" set)
    lg_safe_prompts.csv    (Llama-Guard says safe    -> "benign" set)

Labelling uses short greedy generation + string match (the robust, standard
Llama-Guard usage). Resumable via the incrementally-written labels CSV.
"""
import argparse
import csv
import os
import time
from typing import List

import torch


def read_unique_prompts(path: str, prompt_col: str, max_prompts: int) -> List[str]:
    seen = set()
    prompts = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or prompt_col not in reader.fieldnames:
            raise ValueError(f"column '{prompt_col}' not in {reader.fieldnames}")
        for row in reader:
            p = (row.get(prompt_col) or "").strip()
            if not p or p in seen:
                continue
            seen.add(p)
            prompts.append(p)
            if max_prompts and len(prompts) >= max_prompts:
                break
    return prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--prompt_col", default="prompt")
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--unsafe_out", required=True)
    ap.add_argument("--safe_out", required=True)
    ap.add_argument("--model", default="meta-llama/Llama-Guard-3-8B")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=512,
                    help="Prompt truncation length (matches reward_model scoring).")
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--max_prompts", type=int, default=0, help="0 = all")
    ap.add_argument("--log_every", type=int, default=50)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    prompts = read_unique_prompts(args.input_csv, args.prompt_col, args.max_prompts)
    print(f"[label] {len(prompts)} unique prompts from {args.input_csv}", flush=True)

    # Resume: skip prompts already in output_csv
    done = set()
    if os.path.exists(args.output_csv):
        with open(args.output_csv, newline="") as f:
            for row in csv.DictReader(f):
                done.add(row["prompt"])
        print(f"[label] resuming — {len(done)} already labelled", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, token=hf_token)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, token=hf_token, device_map="auto", torch_dtype=torch.bfloat16,
    )
    model.eval()

    # token ids for the soft score (first content token of the verdict word)
    def first_id(word):
        ids = tok.encode(word, add_special_tokens=False)
        return ids[0] if ids else None
    safe_id, unsafe_id = first_id("safe"), first_id("unsafe")

    todo = [p for p in prompts if p not in done]
    print(f"[label] {len(todo)} to label", flush=True)

    new_file = not os.path.exists(args.output_csv)
    fout = open(args.output_csv, "a", newline="")
    writer = csv.writer(fout)
    if new_file:
        writer.writerow(["prompt", "label", "p_unsafe"])

    t0 = time.time()
    n_unsafe = 0
    for i in range(0, len(todo), args.batch_size):
        batch = todo[i:i + args.batch_size]
        texts = [
            tok.apply_chat_template([{"role": "user", "content": p}],
                                    tokenize=False, add_generation_prompt=True)
            for p in batch
        ]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=args.max_length).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                output_scores=True, return_dict_in_generate=True,
                pad_token_id=tok.pad_token_id,
            )
        gen = out.sequences[:, enc["input_ids"].shape[1]:]
        # soft score: find first step whose argmax decodes to safe/unsafe-ish
        scores = out.scores  # list[max_new_tokens] of [B, V]
        for b in range(len(batch)):
            text = tok.decode(gen[b], skip_special_tokens=True).strip().lower()
            label = "unsafe" if "unsafe" in text else "safe"
            # best-effort p_unsafe: at the first step where the token is safe/unsafe
            p_unsafe = ""
            if safe_id is not None and unsafe_id is not None:
                for step in range(len(scores)):
                    tid = int(gen[b, step]) if step < gen.shape[1] else -1
                    if tid in (safe_id, unsafe_id):
                        logits = scores[step][b]
                        pair = torch.softmax(
                            torch.stack([logits[safe_id], logits[unsafe_id]]), dim=0)
                        p_unsafe = f"{pair[1].item():.4f}"
                        break
            writer.writerow([batch[b], label, p_unsafe])
            if label == "unsafe":
                n_unsafe += 1
        fout.flush()
        done_n = i + len(batch)
        if done_n % args.log_every < args.batch_size:
            rate = done_n / max(time.time() - t0, 1e-3)
            print(f"[label] {done_n}/{len(todo)}  rate={rate:.1f}/s  "
                  f"unsafe so far={n_unsafe}", flush=True)
    fout.close()

    # Build the split files from the full labels CSV
    unsafe, safe = [], []
    with open(args.output_csv, newline="") as f:
        for row in csv.DictReader(f):
            (unsafe if row["label"] == "unsafe" else safe).append(row["prompt"])
    for path, rows in [(args.unsafe_out, unsafe), (args.safe_out, safe)]:
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["prompt"])
            for p in rows:
                w.writerow([p])
    tot = len(unsafe) + len(safe)
    print(f"\n[label] DONE. unsafe={len(unsafe)} ({100*len(unsafe)/max(tot,1):.1f}%)  "
          f"safe={len(safe)} ({100*len(safe)/max(tot,1):.1f}%)", flush=True)
    print(f"[label] wrote {args.unsafe_out} and {args.safe_out}", flush=True)


if __name__ == "__main__":
    main()
