from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class GenerationParams:
    temperature: float = 0.9
    top_p: Optional[float] = None
    max_output_tokens: int = 256
    seed: Optional[int] = None


class Provider:
    """
    Minimal provider interface used by run_generate.py.
    """

    name: str
    model: str

    def generate(self, messages: List[dict], params: GenerationParams) -> str:  # pragma: no cover
        raise NotImplementedError

