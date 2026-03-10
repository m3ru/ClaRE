from __future__ import annotations

from typing import List, Optional, Tuple

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
    return "\n\n".join(system_parts).strip(), "\n\n".join(user_parts).strip()


class GeminiProvider(Provider):
    def __init__(self, model: Optional[str] = None):
        self.name = "gemini"
        self.model = model or config.DEFAULT_GEMINI_MODEL
        api_key = config.get_secret("GEMINI_API_KEY", config.GEMINI_API_KEY)
        self._api_key = config.require_nonempty("GEMINI_API_KEY", api_key)

    def generate(self, messages: List[dict], params: GenerationParams) -> str:
        system, user = _split_system_and_user(messages)

        # Prefer the newer google-genai package; fallback to google-generativeai if needed.
        try:
            from google import genai  # type: ignore
            from google.genai import types  # type: ignore

            client = genai.Client(api_key=self._api_key)
            prompt = user if not system else f"{system}\n\n{user}"
            resp = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=params.temperature,
                    top_p=params.top_p,
                    max_output_tokens=params.max_output_tokens,
                ),
            )
            return (resp.text or "").strip()
        except Exception:
            import google.generativeai as genai  # type: ignore

            genai.configure(api_key=self._api_key)
            model = genai.GenerativeModel(
                model_name=self.model,
                system_instruction=system if system else None,
            )
            resp = model.generate_content(
                user,
                generation_config=genai.types.GenerationConfig(
                    temperature=params.temperature,
                    top_p=params.top_p,
                    max_output_tokens=params.max_output_tokens,
                ),
            )
            return (getattr(resp, "text", "") or "").strip()

