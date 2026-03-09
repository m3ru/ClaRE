from __future__ import annotations

from typing import List, Optional

from openai import OpenAI

from .. import config
from .base import GenerationParams, Provider


class OpenAIProvider(Provider):
    def __init__(self, model: Optional[str] = None):
        self.name = "openai"
        self.model = model or config.DEFAULT_OPENAI_MODEL
        api_key = config.get_secret("OPENAI_API_KEY", config.OPENAI_API_KEY)
        self._client = OpenAI(api_key=config.require_nonempty("OPENAI_API_KEY", api_key))

    def generate(self, messages: List[dict], params: GenerationParams) -> str:
        # OpenAI SDK expects: [{"role":"system"|"user"|"assistant", "content": "..."}]
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=params.temperature,
            top_p=params.top_p,
            max_tokens=params.max_output_tokens,
        )
        return (resp.choices[0].message.content or "").strip()

