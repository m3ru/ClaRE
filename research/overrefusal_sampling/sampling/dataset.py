from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional


@dataclass(frozen=True)
class BenignRow:
    id: int
    prompt: str
    source: str = "dolly"
    category: Optional[str] = None


def iter_dolly_rows(dolly_jsonl_path: str | Path) -> Iterator[dict]:
    path = Path(dolly_jsonl_path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def dolly_to_benign_prompt(obj: dict, *, include_context: bool) -> str:
    instruction = (obj.get("instruction") or "").strip()
    context = (obj.get("context") or "").strip()
    if include_context and context:
        return f"{instruction}\n\nContext:\n{context}".strip()
    return instruction


def build_benign_dataset_from_dolly(
    dolly_jsonl_path: str | Path,
    limit: Optional[int] = None,
    min_chars: int = 5,
    max_chars: int = 1500,
    include_context: bool = False,
) -> List[BenignRow]:
    out: List[BenignRow] = []
    for obj in iter_dolly_rows(dolly_jsonl_path):
        prompt = dolly_to_benign_prompt(obj, include_context=include_context).strip()
        if len(prompt) < min_chars:
            continue
        if len(prompt) > max_chars:
            continue
        out.append(
            BenignRow(
                id=len(out),
                prompt=prompt,
                source="dolly",
                category=obj.get("category"),
            )
        )
        if limit is not None and len(out) >= limit:
            break
    return out


def save_benign_csv(rows: Iterable[BenignRow], out_csv: str | Path) -> None:
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "prompt", "source", "category"])
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "id": r.id,
                    "prompt": r.prompt,
                    "source": r.source,
                    "category": r.category if r.category is not None else "",
                }
            )


def load_benign_rows(path: str | Path) -> List[BenignRow]:
    in_path = Path(path)
    with in_path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            raise ValueError(f"Empty CSV: {path}")
        if "id" not in r.fieldnames or "prompt" not in r.fieldnames:
            raise ValueError(f"Expected columns id,prompt in {path}. Got: {r.fieldnames}")

        out: List[BenignRow] = []
        for row in r:
            pid = int(str(row.get("id", "0")).strip())
            prompt = str(row.get("prompt", "") or "").strip()
            if not prompt:
                continue
            out.append(
                BenignRow(
                    id=pid,
                    prompt=prompt,
                    source=str(row.get("source", "") or "dolly"),
                    category=(str(row.get("category", "") or "").strip() or None),
                )
            )
        return out

