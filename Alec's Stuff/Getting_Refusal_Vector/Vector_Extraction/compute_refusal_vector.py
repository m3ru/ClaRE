#!/usr/bin/env python3
import os, sys, argparse, glob, json
from typing import List, Tuple


def load_sums(npz_paths: List[str]):
    import numpy as np
    total_sum = None
    total_count = 0
    layers: List[int] = []
    hidden_size = None
    for p in sorted(npz_paths):
        data = np.load(p, allow_pickle=True)
        shard_sum = data["sum_by_layer"]  # [L, H]
        count_tokens = int(data["count_tokens"])  # scalar
        shard_layers = list(data["layers"].tolist()) if "layers" in data else list(range(shard_sum.shape[0]))
        if shard_sum.size == 0 or count_tokens == 0:
            continue
        if total_sum is None:
            total_sum = shard_sum.astype("float64", copy=True)
            total_count = count_tokens
            layers = shard_layers
            hidden_size = shard_sum.shape[1]
        else:
            if (shard_sum.shape != total_sum.shape) or (shard_layers != layers):
                raise ValueError(f"Shard shape/layer mismatch: {p}")
            total_sum += shard_sum
            total_count += count_tokens
    if total_sum is None:
        import numpy as np
        return np.zeros((0, 0), dtype="float64"), 0, []
    return total_sum, total_count, layers


def save_vector(path: str, vector, meta: dict):
    import numpy as np
    d = dict(vector=vector.astype("float32"))
    d.update(meta)
    np.savez(path, **d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benign_dir", required=True, help="Directory of benign shard .npz files")
    ap.add_argument("--refusal_dir", required=True, help="Directory of refusal shard .npz files")
    ap.add_argument("--output", required=True, help="Output .npz path for refusal vector")
    ap.add_argument("--select_layer", default="", help="Optional: select one layer by index to save also")
    args = ap.parse_args()

    import numpy as np

    benign_npzs = sorted(glob.glob(os.path.join(args.benign_dir, "*.npz")))
    refusal_npzs = sorted(glob.glob(os.path.join(args.refusal_dir, "*.npz")))
    if len(benign_npzs) == 0:
        print(f"ERROR: no .npz files in {args.benign_dir}", file=sys.stderr)
        sys.exit(2)
    if len(refusal_npzs) == 0:
        print(f"ERROR: no .npz files in {args.refusal_dir}", file=sys.stderr)
        sys.exit(2)

    benign_sum, benign_count, layers = load_sums(benign_npzs)
    refusal_sum, refusal_count, layers2 = load_sums(refusal_npzs)
    if layers != layers2:
        raise ValueError("Layer index list mismatch between datasets")
    if benign_count == 0 or refusal_count == 0:
        raise ValueError("Empty counts — did extraction run correctly?")

    benign_mean = benign_sum / float(benign_count)  # [L, H]
    refusal_mean = refusal_sum / float(refusal_count)
    diff = refusal_mean - benign_mean  # [L, H]

    meta = dict(
        layers=np.array(layers, dtype="int32"),
        benign_count=int(benign_count),
        refusal_count=int(refusal_count),
        description="Difference-in-means (refusal - benign) per layer",
    )

    # Save full per-layer vectors and also an L2 magnitude per layer to help pick a layer
    l2 = np.linalg.norm(diff, axis=1)
    meta["l2_per_layer"] = l2.astype("float32")
    save_vector(args.output, diff, meta)
    print(f"Saved per-layer difference vectors → {args.output}")

    # Optionally also save a single selected layer vector
    if args.select_layer != "":
        if args.select_layer == "max":
            idx = int(np.argmax(l2))
        else:
            idx = int(args.select_layer)
        if idx < 0 or idx >= diff.shape[0]:
            raise ValueError(f"select_layer index out of range: {idx}")
        single = diff[idx]
        single_out = os.path.splitext(args.output)[0] + f".layer{layers[idx]:03d}.npz"
        save_vector(single_out, single, dict(layer=int(layers[idx]), description="Selected single-layer refusal vector"))
        print(f"Saved single-layer vector → {single_out}")


if __name__ == "__main__":
    main()


