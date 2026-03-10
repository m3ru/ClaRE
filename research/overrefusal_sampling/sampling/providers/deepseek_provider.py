from __future__ import annotations

from typing import List, Optional

from openai import OpenAI

from .. import config
from .base import GenerationParams, Provider


class DeepSeekProvider(Provider):
    """
    DeepSeek is typically OpenAI-compatible (chat.completions) with a custom base_url.
    """

    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None):
        self.name = "deepseek"
        self.model = model or config.DEFAULT_DEEPSEEK_MODEL
        api_key = config.get_secret("DEEPSEEK_API_KEY", config.DEEPSEEK_API_KEY)
        base_url = base_url or config.DEEPSEEK_BASE_URL
        self._client = OpenAI(
            api_key=config.require_nonempty("DEEPSEEK_API_KEY", api_key),
            base_url=base_url,
        )

    def generate(self, messages: List[dict], params: GenerationParams) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=params.temperature,
            top_p=params.top_p,
            max_tokens=params.max_output_tokens,
        )
        return (resp.choices[0].message.content or "").strip()

