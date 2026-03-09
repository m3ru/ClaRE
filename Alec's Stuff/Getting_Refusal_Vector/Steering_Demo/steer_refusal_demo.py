#!/usr/bin/env python3
import os, sys, argparse, csv, json
from typing import List, Tuple


def read_prompts_robust(path: str, n: int) -> List[str]:
    # Try CSV first (take first column)
    out: List[str] = []
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            r = csv.reader(f)
            for i, row in enumerate(r):
                if not row:
                    continue
                out.append(row[0])
                if len(out) >= n:
                    return out
    except Exception:
        pass
    # Fallback: one prompt per line
    if len(out) < n:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    s = line.rstrip("\n")
                    if s:
                        out.append(s)
                        if len(out) >= n:
                            break
        except FileNotFoundError:
            print(f"ERROR: File not found: {path}", file=sys.stderr)
            sys.exit(2)
    return out[:n]


def parse_scales(scales: str) -> List[float]:
    vals: List[float] = []
    for tok in scales.split(","):
        tok = tok.strip()
        if not tok:
            continue
        vals.append(float(tok))
    # Always include 0.0 (baseline) if not present
    if 0.0 not in vals:
        vals = [0.0] + vals
    return vals


def load_vector(vector_npz_path: str, select_layer: str) -> Tuple[List[int], 'np.ndarray']:
    import numpy as np
    d = np.load(vector_npz_path, allow_pickle=True)
    v = d.get('vector')
    layers = None
    if v is not None:
        # Could be [H] (single) or [L,H] (multi)
        if v.ndim == 1:
            layer = int(d.get('layer', -1))
            if layer < 0:
                raise ValueError("Single vector npz missing 'layer' metadata")
            return [layer], v.astype('float32')
        elif v.ndim == 2:
            layers = d.get('layers')
            if layers is None:
                raise ValueError("Multi-layer vector npz missing 'layers'")
            layers = layers.astype('int32').tolist()
            if select_layer == 'max':
                l2 = d.get('l2_per_layer')
                if l2 is None:
                    l2 = np.linalg.norm(v, axis=1)
                idx = int(np.argmax(l2))
            else:
                idx = int(select_layer)
                if idx < 0 or idx >= v.shape[0]:
                    raise ValueError(f"select_layer index out of range: {idx}")
            return [int(layers[idx])], v[idx].astype('float32')
        else:
            raise ValueError("Unexpected 'vector' shape in npz")
    else:
        # The compute script saved only 'vector' key; older formats not supported
        raise ValueError("npz must contain a 'vector' array")


class SteeringHook:
    def __init__(self, vec, scale: float):
        import torch
        self.vec = vec  # torch tensor [H] on correct device/dtype
        self.scale = float(scale)
        self.step = 0

    def __call__(self, module, input, output):
        import torch
        # Skip the initial prefill pass (prompt only)
        self.step += 1
        if self.scale == 0.0:
            return output

        def add_vec(h):
            # h: [B, T, H]
            if h is None:
                return h
            if h.dim() != 3 or h.shape[-1] != self.vec.shape[0]:
                return h
            add = self.vec * self.scale  # [H]
            # Add only to the last token position (works for both prefill and decoding)
            h = h.clone()
            h[:, -1, :] = h[:, -1, :] + add.view(1, -1)
            return h

        # LlamaDecoderLayer may return tensor or tuple
        if isinstance(output, tuple):
            h = output[0]
            new_h = add_vec(h)
            if new_h is h:
                return output
            return (new_h,) + output[1:]
        else:
            h = output
            new_h = add_vec(h)
            return new_h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--benign_csv', required=True, help='Path to benign prompts CSV or txt')
    ap.add_argument('--vector_npz', required=True, help='Path to vector .npz (single-layer or multi-layer)')
    ap.add_argument('--select_layer', default='max', help="If vector has multiple layers: 'max' or index")
    ap.add_argument('--scales', default='0,0.5,1,2,4', help='Comma-separated scales to apply (0 included automatically)')
    ap.add_argument('--num_prompts', type=int, default=10)
    ap.add_argument('--output_csv', required=True)
    ap.add_argument('--model', default='meta-llama/Meta-Llama-3-8B-Instruct')
    ap.add_argument('--max_new_tokens', type=int, default=200)
    ap.add_argument('--batch_size', type=int, default=1)
    ap.add_argument('--dtype', choices=['bf16','fp16','fp32'], default='bf16')
    ap.add_argument('--do_sample', type=int, default=1, help='1: enable sampling')
    ap.add_argument('--temperature', type=float, default=0.7)
    ap.add_argument('--top_p', type=float, default=0.9)
    ap.add_argument('--calibration_alpha', type=float, default=0.1, help='Steering strength multiplier on base magnitude')
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import numpy as np

    prompts = read_prompts_robust(args.benign_csv, args.num_prompts)
    if len(prompts) == 0:
        print('No prompts loaded.', file=sys.stderr)
        sys.exit(2)

    # Load vector and layer
    layers, vec_np = load_vector(args.vector_npz, args.select_layer)
    if len(layers) != 1:
        print('Expected a single layer vector after selection.', file=sys.stderr)
        sys.exit(3)
    # Saved metadata uses 1-based layer numbering; convert to 0-based for HF modules
    layer_1based = int(layers[0])
    layer_idx = max(0, layer_1based - 1)
    vec_np = vec_np.astype('float32')
    vec_np = vec_np / (np.linalg.norm(vec_np) + 1e-9)

    if args.dtype == 'bf16':
        torch_dtype = torch.bfloat16
    elif args.dtype == 'fp16':
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    device_map = 'auto'

    hf_token = os.environ.get('HUGGING_FACE_HUB_TOKEN') or os.environ.get('HF_TOKEN')
    if not hf_token:
        print('WARNING: HUGGING_FACE_HUB_TOKEN not set; gated models may fail.', file=sys.stderr)

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, token=hf_token)
    tok.padding_side = 'left'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch_dtype, device_map=device_map, token=hf_token
    )

    def build_chat(u: str) -> str:
        msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": u},
        ]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    # Locate the target layer module
    try:
        total_layers = len(model.model.layers)
        if not (0 <= layer_idx < total_layers):
            print(f"Layer index out of range: requested 1-based {layer_1based} (0-based {layer_idx}), model has {total_layers} layers (0..{total_layers-1})", file=sys.stderr)
            sys.exit(4)
        target_mod = model.model.layers[layer_idx]
    except Exception:
        print(f'Could not access model.model.layers[{layer_idx}]', file=sys.stderr)
        sys.exit(4)

    scales = parse_scales(args.scales)
    print(f"Using layer (1-based) {layer_1based} → module index {layer_idx}; scales={scales}")

    # Prepare outputs
    os.makedirs(os.path.dirname(args.output_csv) or '.', exist_ok=True)
    with open(args.output_csv, 'w', newline='', encoding='utf-8') as outf:
        w = csv.writer(outf)
        w.writerow(['prompt','scale','layer_1based','delta_logit_l2','response'])

        for p in prompts:
            chat = build_chat(p)
            enc = tok([chat], return_tensors='pt', padding=True, truncation=False)
            prompt_len = enc['input_ids'].shape[1]

            # Compute per-prompt baseline magnitude at the target layer (last prompt token)
            with torch.no_grad():
                pre = model(**{k:v.to(model.device) for k,v in enc.items()}, output_hidden_states=True, use_cache=False, return_dict=True)
                # hidden_states is 0=embeddings, 1..L=layers → use 1-based index
                hs_pre = pre.hidden_states[layer_1based]  # [1, T, H]
                base_mag = hs_pre[:, -1, :].float().norm(dim=-1).mean().item()
                if not (base_mag > 0):
                    base_mag = 1.0

            for s in scales:
                # Register hook for this scale (0 will no-op inside the hook)
                # Place vector on the exact device of the target layer
                layer_device = next(target_mod.parameters()).device
                vec = torch.from_numpy(vec_np).to(layer_device).to(dtype=model.dtype)
                # Scale by baseline magnitude, calibration alpha, and requested scale; pass scale=1.0 to avoid double scaling
                vec_scaled = (vec / (vec.norm() + 1e-9)) * (base_mag * args.calibration_alpha * float(s))
                hook_obj = SteeringHook(vec_scaled, 1.0)
                handle = target_mod.register_forward_hook(hook_obj)
                try:
                    # Debug: compute next-token logits difference on the prompt with and without hook
                    with torch.no_grad():
                        base = model(**{k:v.to(model.device) for k,v in enc.items()}, output_hidden_states=False, use_cache=False, return_dict=True)
                        logits_base = base.logits[:, -1, :]

                    with torch.no_grad():
                        steered = model(**{k:v.to(model.device) for k,v in enc.items()}, output_hidden_states=False, use_cache=False, return_dict=True)
                        logits_steer = steered.logits[:, -1, :]
                        delta_l2 = (logits_steer - logits_base).float().norm(dim=-1).item()
                        print(f"[debug] scale={s} delta_logit_l2={delta_l2:.4f}")

                    gen = model.generate(
                        **{k:v.to(model.device) for k,v in enc.items()},
                        max_new_tokens=args.max_new_tokens,
                        do_sample=bool(args.do_sample),
                        temperature=max(1e-6, args.temperature),
                        top_p=args.top_p,
                        pad_token_id=tok.eos_token_id,
                    )
                finally:
                    handle.remove()

                # Strip prompt tokens and decode only continuation
                out_ids = gen[0][prompt_len:]
                text = tok.decode(out_ids, skip_special_tokens=True)
                w.writerow([p, s, layer_1based, f"{delta_l2:.6f}", text])
                outf.flush()


if __name__ == '__main__':
    main()


