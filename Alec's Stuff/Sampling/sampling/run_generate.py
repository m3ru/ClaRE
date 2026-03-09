from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import traceback

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore

from .dataset import load_benign_rows
from .prompts import get_variant, normalize_variant_list
from .providers import AnthropicProvider, DeepSeekProvider, GeminiProvider, OpenAIProvider, Provider
from .providers.base import GenerationParams
from .score import score_candidate


def _parse_csv_list(s: str) -> List[str]:
    return [p.strip() for p in s.split(",") if p.strip()]


def _parse_model_overrides(s: Optional[str]) -> Dict[str, str]:
    """
    Format: "openai=gpt-4o-mini,anthropic=claude-3-haiku-20240307"
    """
    if not s:
        return {}
    out: Dict[str, str] = {}
    for part in _parse_csv_list(s):
        if "=" not in part:
            raise ValueError(f"Bad --model_overrides entry: {part!r} (expected provider=model)")
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _parse_int_overrides(s: Optional[str]) -> Dict[str, int]:
    """
    Format: "anthropic=1,openai=3"
    """
    if not s:
        return {}
    out: Dict[str, int] = {}
    for part in _parse_csv_list(s):
        if "=" not in part:
            raise ValueError(f"Bad override entry: {part!r} (expected provider=int)")
        k, v = part.split("=", 1)
        out[k.strip()] = int(v.strip())
    return out


def _looks_like_rate_limit(exc: BaseException) -> bool:
    name = exc.__class__.__name__.lower()
    msg = str(exc).lower()
    if "ratelimit" in name or "rate_limit" in name:
        return True
    if "429" in msg:
        return True
    if "rate limit" in msg:
        return True
    return False


def _generate_with_retry(
    provider: Provider,
    messages: List[dict],
    params: GenerationParams,
    retries: int,
    base_sleep_s: float,
) -> str:
    attempt = 0
    while True:
        try:
            return provider.generate(messages, params=params)
        except Exception as e:
            attempt += 1
            if attempt > retries:
                raise
            # Backoff more aggressively for rate limits.
            if _looks_like_rate_limit(e):
                sleep_s = max(5.0, base_sleep_s) * (2 ** (attempt - 1))
                sleep_s = min(sleep_s, 90.0)
            else:
                sleep_s = base_sleep_s * (2 ** (attempt - 1))
                sleep_s = min(sleep_s, 30.0)
            time.sleep(sleep_s)


def _build_providers(
    names: Iterable[str],
    model_overrides: Dict[str, str],
    deepseek_base_url: Optional[str],
) -> List[Provider]:
    out: List[Provider] = []
    for n in names:
        if n == "openai":
            out.append(OpenAIProvider(model=model_overrides.get("openai")))
        elif n == "anthropic":
            out.append(AnthropicProvider(model=model_overrides.get("anthropic")))
        elif n == "gemini":
            out.append(GeminiProvider(model=model_overrides.get("gemini")))
        elif n == "deepseek":
            out.append(
                DeepSeekProvider(
                    model=model_overrides.get("deepseek"),
                    base_url=deepseek_base_url,
                )
            )
        else:
            raise ValueError(f"Unknown provider: {n}. Expected one of: openai,anthropic,gemini,deepseek")
    return out


class _DryRunProvider(Provider):
    def __init__(self, name: str):
        self.name = name
        self.model = "dry_run"

    def generate(self, messages: List[dict], params: GenerationParams) -> str:
        raise RuntimeError("Dry-run provider should never be called.")


def _clamp_temperature(provider_name: str, temperature: float) -> float:
    """
    Providers have slightly different allowed ranges. Anthropic enforces [0, 1].
    """
    t = float(temperature)
    if provider_name == "anthropic":
        return max(0.0, min(1.0, t))
    return max(0.0, t)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate over-refusal prompts from benign prompts (multi-model sampling).")
    ap.add_argument("--benign_csv", required=True, help="CSV with id,prompt columns (see run_prepare_benign)")
    ap.add_argument("--out_jsonl", required=True, help="Output JSONL path (append if exists)")

    ap.add_argument("--providers", default="openai,anthropic,gemini,deepseek")
    ap.add_argument("--prompt_variants", default=None, help="Comma list, e.g. v1,v2,v3. Default: all.")
    ap.add_argument("--model_overrides", default=None, help="Comma list: provider=model (e.g. openai=gpt-4o-mini)")
    ap.add_argument("--deepseek_base_url", default=None, help="Optional DeepSeek base URL override")

    ap.add_argument("--num_samples", type=int, default=3, help="Samples per (provider,variant,benign)")
    ap.add_argument(
        "--num_samples_overrides",
        default=None,
        help='Optional per-provider override, e.g. "anthropic=1,openai=3"',
    )
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_p", type=float, default=None)
    ap.add_argument("--max_output_tokens", type=int, default=256)
    ap.add_argument(
        "--max_output_tokens_overrides",
        default=None,
        help='Optional per-provider override, e.g. "anthropic=192,openai=256"',
    )
    ap.add_argument("--retries", type=int, default=5, help="Retries per request on transient errors (429/5xx/etc).")
    ap.add_argument("--retry_base_sleep_s", type=float, default=2.0, help="Base backoff sleep seconds.")
    ap.add_argument(
        "--skip_provider_errors",
        type=int,
        default=1,
        help="1 to record provider errors and continue; 0 to raise immediately.",
    )

    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)

    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_idx", type=int, default=0)

    ap.add_argument("--dry_run", type=int, default=0, help="1 to avoid API calls (writes dummy outputs)")
    ap.add_argument("--print_samples", type=int, default=0, help="1 to print some sampled outputs to stdout")
    args = ap.parse_args()

    rnd = random.Random(args.seed)

    provider_names = _parse_csv_list(args.providers)
    model_overrides = _parse_model_overrides(args.model_overrides)
    num_samples_overrides = _parse_int_overrides(args.num_samples_overrides)
    max_tokens_overrides = _parse_int_overrides(args.max_output_tokens_overrides)
    if int(args.dry_run) == 1:
        # Don't require API keys in dry-run mode.
        providers = [_DryRunProvider(name=n) for n in provider_names] or [_DryRunProvider(name="dryrun")]
    else:
        providers = _build_providers(provider_names, model_overrides, args.deepseek_base_url)

    variant_ids = normalize_variant_list(_parse_csv_list(args.prompt_variants) if args.prompt_variants else None)

    rows = load_benign_rows(args.benign_csv)
    if args.offset:
        rows = rows[args.offset :]
    if args.limit is not None:
        rows = rows[: args.limit]

    # Shard
    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if not (0 <= args.shard_idx < args.num_shards):
        raise ValueError("--shard_idx must be in [0, num_shards)")
    if args.num_shards > 1:
        rows = [r for r in rows if (int(r.id) % args.num_shards) == args.shard_idx]

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_jobs = len(rows) * len(providers) * len(variant_ids) * args.num_samples
    print(
        f"[run] rows={len(rows)} providers={len(providers)} variants={len(variant_ids)} "
        f"num_samples={args.num_samples} total_generations={total_jobs}"
    )

    printed = 0
    with out_path.open("a", encoding="utf-8") as f:
        it = rows
        if tqdm is not None:
            it = tqdm(rows, total=len(rows))  # type: ignore

        for row in it:  # type: ignore
            benign_id = int(row.id)
            benign_prompt = str(row.prompt)

            for provider in providers:
                for variant_id in variant_ids:
                    variant = get_variant(variant_id)
                    messages = variant.render_messages(benign_prompt)

                    this_num_samples = int(num_samples_overrides.get(provider.name, args.num_samples))
                    this_max_tokens = int(max_tokens_overrides.get(provider.name, args.max_output_tokens))

                    for sample_idx in range(this_num_samples):
                        # Small stochasticity per-sample (provider-level randomness is controlled by temperature).
                        temp = float(args.temperature) + rnd.uniform(-0.15, 0.15)
                        temp = _clamp_temperature(provider.name, temp)
                        params = GenerationParams(
                            temperature=temp,
                            top_p=args.top_p,
                            max_output_tokens=this_max_tokens,
                            seed=None,
                        )

                        error = None
                        try:
                            if int(args.dry_run) == 1:
                                raw = f"{benign_prompt} (rewrite: more refusal-prone)"
                            else:
                                raw = _generate_with_retry(
                                    provider,
                                    messages=messages,
                                    params=params,
                                    retries=int(args.retries),
                                    base_sleep_s=float(args.retry_base_sleep_s),
                                )
                        except Exception as e:
                            if int(args.skip_provider_errors) != 1:
                                raise
                            raw = ""
                            error = {
                                "type": e.__class__.__name__,
                                "message": str(e),
                                "traceback": traceback.format_exc(limit=6),
                            }

                        scored = score_candidate(raw)
                        rec = {
                            "ts": time.time(),
                            "benign_id": benign_id,
                            "benign_prompt": benign_prompt,
                            "provider": provider.name,
                            "model": provider.model,
                            "prompt_variant": variant_id,
                            "sample_idx": sample_idx,
                            "overrefusal_prompt": raw,
                            "normalized_overrefusal_prompt": scored.normalized_text,
                            "score": scored.score,
                            "refused": True if error is not None else scored.refused,
                            "error": error,
                            "gen_params": asdict(params),
                        }
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        f.flush()

                        if int(args.print_samples) == 1 and printed < 10:
                            printed += 1
                            print("\n---")
                            print(f"[benign] {benign_prompt}")
                            print(
                                f"[provider={provider.name} model={provider.model} variant={variant_id} sample={sample_idx}]"
                            )
                            print(f"[out] {scored.normalized_text}")

    print(f"[done] wrote -> {out_path}")


if __name__ == "__main__":
    main()

