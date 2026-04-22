#!/usr/bin/env python3
"""Download cybermetric, sciq, medqa, pubmedqa, dolly from HuggingFace and save as JSONL.

Each output file has one JSON object per line: {"prompt": "..."}
Compatible with paraphrase_and_score.py's load_prompts().

Usage:
    python prepare_datasets.py --output_dir ./datasets
"""

import argparse
import json
import os


def save_jsonl(records, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  Saved {len(records)} prompts -> {path}")


def prepare_cybermetric(output_dir):
    """CyberMetric from tihanyin/CyberMetric."""
    print("[cybermetric] Loading tihanyin/CyberMetric...")
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id="tihanyin/CyberMetric", filename="CyberMetric-2000-v1.json",
                           repo_type="dataset")
    with open(path, "r") as f:
        data = json.load(f)
    # data is a list of dicts with "question" field
    if isinstance(data, dict):
        data = list(data.values())[0] if len(data) == 1 else [data]
    records = []
    for row in data:
        q = (row.get("question") or "").strip()
        if q and len(q) >= 10:
            records.append({"prompt": q})
    save_jsonl(records, os.path.join(output_dir, "cybermetric.jsonl"))
    return len(records)


def prepare_sciq(output_dir):
    """SciQ from allenai/sciq — train split."""
    from datasets import load_dataset
    print("[sciq] Loading allenai/sciq...")
    ds = load_dataset("allenai/sciq", split="train")
    records = []
    for row in ds:
        q = (row.get("question") or "").strip()
        if q and len(q) >= 10:
            records.append({"prompt": q})
    save_jsonl(records, os.path.join(output_dir, "sciq.jsonl"))
    return len(records)


def prepare_medqa(output_dir):
    """MedQA-USMLE 4-options from GBaker/MedQA-USMLE-4-options — train split."""
    from datasets import load_dataset
    print("[medqa] Loading GBaker/MedQA-USMLE-4-options...")
    ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="train")
    records = []
    for row in ds:
        q = (row.get("question") or "").strip()
        if q and len(q) >= 10:
            records.append({"prompt": q})
    save_jsonl(records, os.path.join(output_dir, "medqa.jsonl"))
    return len(records)


def prepare_pubmedqa(output_dir):
    """PubMedQA from qiaojin/PubMedQA — pqa_labeled subset."""
    from datasets import load_dataset
    print("[pubmedqa] Loading qiaojin/PubMedQA (pqa_labeled)...")
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    records = []
    for row in ds:
        q = (row.get("question") or "").strip()
        if q and len(q) >= 10:
            records.append({"prompt": q})
    save_jsonl(records, os.path.join(output_dir, "pubmedqa.jsonl"))
    return len(records)


def prepare_dolly(output_dir):
    """Dolly 15k from databricks/databricks-dolly-15k — instruction field only."""
    from datasets import load_dataset
    print("[dolly] Loading databricks/databricks-dolly-15k...")
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    records = []
    for row in ds:
        q = (row.get("instruction") or "").strip()
        if q and len(q) >= 10:
            records.append({"prompt": q})
    save_jsonl(records, os.path.join(output_dir, "dolly.jsonl"))
    return len(records)


DATASET_MAP = {
    "cybermetric": prepare_cybermetric,
    "sciq": prepare_sciq,
    "medqa": prepare_medqa,
    "pubmedqa": prepare_pubmedqa,
    "dolly": prepare_dolly,
}


def main():
    parser = argparse.ArgumentParser(description="Download and prepare datasets for paraphrasing")
    parser.add_argument("--output_dir", type=str, default="./datasets")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Prepare a single dataset (cybermetric/sciq/medqa/pubmedqa/dolly). Default: all.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.dataset:
        if args.dataset not in DATASET_MAP:
            raise ValueError(f"Unknown dataset: {args.dataset}. Choose from: {list(DATASET_MAP.keys())}")
        n = DATASET_MAP[args.dataset](args.output_dir)
        print(f"\n[done] {n} prompts for {args.dataset} in {args.output_dir}/")
    else:
        total = 0
        for name, prep_fn in DATASET_MAP.items():
            n = prep_fn(args.output_dir)
            total += n
        print(f"\n[done] {total} total prompts across {len(DATASET_MAP)} datasets in {args.output_dir}/")


if __name__ == "__main__":
    main()
