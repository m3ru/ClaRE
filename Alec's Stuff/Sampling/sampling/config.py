"""
Config for Sampling.

You asked to hard-code API keys. Paste them below.
You can also override any key via environment variable:
  OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, DEEPSEEK_API_KEY
"""

from __future__ import annotations

import os

# =========================
# Paste keys here
# =========================

OPENAI_API_KEY = ""
ANTHROPIC_API_KEY = ""
GEMINI_API_KEY = ""
DEEPSEEK_API_KEY = ""

# DeepSeek is typically OpenAI-compatible with a custom base URL.
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# =========================
# Default model IDs
# =========================

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_ANTHROPIC_MODEL = "claude-3-haiku-20240307"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
# If your DeepSeek account uses different model IDs, override via CLI.
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"


def get_secret(env_name: str, hardcoded_value: str) -> str:
    v = os.environ.get(env_name)
    return v if v is not None and v.strip() != "" else hardcoded_value


def require_nonempty(name: str, value: str) -> str:
    if not value or value.strip() == "":
        raise RuntimeError(
            f"Missing {name}. Paste it into sampling/config.py or set env var {name}."
        )
    return value

