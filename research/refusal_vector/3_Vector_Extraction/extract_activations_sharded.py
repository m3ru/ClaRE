#!/usr/bin/env python3
import os, sys, math, argparse, time, csv
from typing import List, Tuple, Optional


# ---------------------------------------------
# Extract per-layer hidden-state activations for prompts (no generation)
# Sharded via SLURM arrays; each shard writes an .npz with running sums
# across selected prompt token positions. These sums will later be merged
# across shards to compute the mean vector per layer for each dataset
# (refusal vs benign), and then the difference-in-means vector.
# ---------------------------------------------


def read_prompts_robust(path: str) -> List[str]:
    """Read a file containing one prompt per line, or a CSV (first column).
    The function tries a simple line-based read first; if more than 1 line
    contains commas that look like CSV, it will fall back to using csv.reader
    and take the first column.
    """
    # Fast path: treat as raw lines
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [ln.rstrip("\n") for ln in f if ln.rstrip("\n") != ""]
    except FileNotFoundError:
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        sys.exit(2)

    # Heuristic: if lots of lines contain commas, assume CSV
    comma_lines = sum(1 for ln in lines if "," in ln)
    if comma_lines > max(5, len(lines) // 3):
        prompts: List[str] = []
        with open(path, "r", encoding="utf-8", newline="") as f:
            r = csv.reader(f)
            # Try to skip header if present; if first row has more than one column and
            # contains non-textual header-like content, we still just take col 0
            for i, row in enumerate(r):
                if not row:
                    continue
                prompts.append(row[0])
        return [p for p in prompts if p]
    else:
        return lines


def calc_slice(n: int, num_shards: int, shard_idx: int) -> Tuple[int, int]:
    per = math.ceil(n / num_shards) if num_shards > 0 else n
    start = shard_idx * per
    end = min(n, (shard_idx + 1) * per)
    return start, end


def ensure_dir_for_file(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def parse_layers_arg(layers_arg: str, num_layers: int) -> List[int]:
    if layers_arg.lower() == "all":
        return list(range(1, num_layers + 1))  # skip embedding layer 0 by default
    idxs: List[int] = []
    for tok in layers_arg.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            v = int(tok)
        except ValueError:
            raise ValueError(f"Invalid layer index: {tok}")
        if v < 0 or v > num_layers:
            raise ValueError(f"Layer index out of range: {v} (num_layers={num_layers})")
        idxs.append(v)
    # Deduplicate while preserving order
    seen = set()
    out: List[int] = []
    for v in idxs:
        if v not in seen:
            out.append(v)
            seen.add(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to prompts (.txt lines or .csv first column)")
    ap.add_argument("--output_base", required=True, help="Output base path; shard writes <base>.shardNNN.npz")
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_idx", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--last_k", type=int, default=1, help="Aggregate over the last K prompt tokens")
    ap.add_argument("--layers", default="all", help="Comma list of layer indexes, or 'all'")
    ap.add_argument("--test", type=int, default=0, help="If 1, only run 10 prompts in this shard")
    args = ap.parse_args()

    prompts = read_prompts_robust(args.input)
    if len(prompts) == 0:
        print("No prompts found.", file=sys.stderr)
        sys.exit(3)

    start, end = calc_slice(len(prompts), args.num_shards, args.shard_idx)
    shard_prompts = prompts[start:end]
    if args.test:
        shard_prompts = shard_prompts[:10]

    print(f"[shard {args.shard_idx}/{args.num_shards}] prompts [{start}:{end}) => {len(shard_prompts)} (test={args.test})", flush=True)
    if len(shard_prompts) == 0:
        # Write an empty shard with metadata for easier merging
        shard_path = f"{args.output_base}.shard{args.shard_idx:03d}.npz"
        ensure_dir_for_file(shard_path)
        import numpy as np
        np.savez(shard_path, sum_by_layer=np.zeros((0, 0), dtype="float64"), count_tokens=0, hidden_size=0, num_layers=0, model=args.model, last_k=args.last_k, layers=[])
        print(f"[shard {args.shard_idx}] Wrote empty sums → {shard_path}")
        return

    # Lazy heavy imports
    import torch
    import numpy as np
    from transformers import AutoTokenizer, AutoModelForCausalLM

    if args.dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    device_map = "auto"

    hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if not hf_token:
        print("WARNING: HUGGING_FACE_HUB_TOKEN not set; gated models may fail.", file=sys.stderr)

    print(f"Loading {args.model} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, token=hf_token)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch_dtype,
        device_map=device_map,
        token=hf_token,
    )

    # Helper to build consistent chat text from a raw user prompt
    def build_chat(u: str) -> str:
        msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": u},
        ]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    batch_size = max(1, args.batch_size)
    count_tokens: int = 0
    sum_by_layer: Optional[np.ndarray] = None  # shape [L, H]
    layers_indices: Optional[List[int]] = None

    t0 = time.time()
    for i in range(0, len(shard_prompts), batch_size):
        batch_prompts = shard_prompts[i:i+batch_size]
        chats = [build_chat(p) for p in batch_prompts]

        enc = tok(
            chats,
            return_tensors="pt",
            padding=True,
            truncation=False,
        ).to(model.device)

        with torch.no_grad():
            out = model(**enc, output_hidden_states=True, use_cache=False, return_dict=True)

        hidden_states = out.hidden_states  # tuple length num_layers+1 (includes embedding 0)
        num_layers_total = len(hidden_states) - 1

        if layers_indices is None:
            layers_indices = parse_layers_arg(args.layers, num_layers_total)

        # Determine hidden size and allocate sums lazily
        hsize = hidden_states[1].shape[-1] if num_layers_total >= 1 else hidden_states[0].shape[-1]
        if sum_by_layer is None:
            sum_by_layer = np.zeros((len(layers_indices), hsize), dtype=np.float64)

        B, S = enc["input_ids"].shape
        k = max(1, args.last_k)
        if k > S:
            k = S

        # For each selected layer, accumulate the mean over last-k prompt tokens per example
        # We aggregate in float64 on CPU for numerical stability
        # hidden_states[l] shape: [B, S, H]
        for li, l in enumerate(layers_indices):
            hs = hidden_states[l]  # torch.Tensor [B, S, H]
            # select last k tokens along seq dimension, then mean over tokens and batch
            # shape after select: [B, k, H]
            slice_hs = hs[:, S-k:S, :].to(torch.float32).float()  # ensure float32 before cpu
            # mean over token dimension, keep [B, H]
            tok_mean = slice_hs.mean(dim=1)
            # sum over batch, move to cpu numpy float64
            batch_sum = tok_mean.sum(dim=0).cpu().to(torch.float64).numpy()
            sum_by_layer[li] += batch_sum

        count_tokens += B * k

        if (i // batch_size) % 10 == 0:
            done = min(i + batch_size, len(shard_prompts))
            print(f"[shard {args.shard_idx}] {done}/{len(shard_prompts)} prompts processed", flush=True)

    dt = time.time() - t0
    print(f"[shard {args.shard_idx}] Finished in {dt:.1f}s — aggregated {count_tokens} tokens", flush=True)

    # Persist results
    shard_path = f"{args.output_base}.shard{args.shard_idx:03d}.npz"
    ensure_dir_for_file(shard_path)
    import numpy as np
    if sum_by_layer is None:
        sum_by_layer = np.zeros((0, 0), dtype=np.float64)
        layers_indices = []
        hsize = 0
        num_layers_total = 0

    import numpy as np  # ensure available for save
    np.savez(
        shard_path,
        sum_by_layer=sum_by_layer,
        count_tokens=count_tokens,
        hidden_size=sum_by_layer.shape[1] if sum_by_layer.size else 0,
        num_layers=len(layers_indices),
        model=args.model,
        last_k=args.last_k,
        layers=layers_indices,
    )
    print(f"[shard {args.shard_idx}] Wrote sums → {shard_path}")


if __name__ == "__main__":
    main()


