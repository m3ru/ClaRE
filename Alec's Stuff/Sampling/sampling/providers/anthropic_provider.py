from __future__ import annotations

from typing import List, Optional, Tuple

from anthropic import Anthropic

from .. import config
from .base import GenerationParams, Provider


def _split_system_and_user(messages: List[dict]) -> Tuple[str, str]:
    system_parts: List[str] = []
    user_parts: List[str] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content") or ""
        if role == "system":
            system_parts.append(str(content))
        elif role == "user":
            user_parts.append(str(content))
        # We ignore assistant messages here since our prompt is 2-turn (system+user).
    return "\n\n".join(system_parts).strip(), "\n\n".join(user_parts).strip()


class AnthropicProvider(Provider):
    def __init__(self, model: Optional[str] = None):
        self.name = "anthropic"
        self.model = model or config.DEFAULT_ANTHROPIC_MODEL
        api_key = config.get_secret("ANTHROPIC_API_KEY", config.ANTHROPIC_API_KEY)
        self._client = Anthropic(api_key=config.require_nonempty("ANTHROPIC_API_KEY", api_key))

    def generate(self, messages: List[dict], params: GenerationParams) -> str:
        system, user = _split_system_and_user(messages)
        kwargs = {
            "model": self.model,
            "system": system if system else None,
            "messages": [{"role": "user", "content": user}],
            "max_tokens": params.max_output_tokens,
            "temperature": params.temperature,
        }
        # Anthropic rejects null for top_p; only include if set.
        if params.top_p is not None:
            kwargs["top_p"] = params.top_p

        resp = self._client.messages.create(**kwargs)
        # anthropic SDK returns content list of blocks (TextBlock)
        out = ""
        for block in resp.content:
            t = getattr(block, "text", None)
            if t:
                out += t
        return out.strip()

