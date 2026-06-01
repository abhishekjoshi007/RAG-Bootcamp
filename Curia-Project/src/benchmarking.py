from __future__ import annotations

import os
from dataclasses import dataclass
from math import ceil
from typing import Mapping

from .config import LLM_MAX_TOKENS
from .model_registry import ModelSpec


DEFAULT_UNKNOWN_INPUT_COST_PER_MTOK = 5.0
DEFAULT_UNKNOWN_OUTPUT_COST_PER_MTOK = 15.0


@dataclass(frozen=True)
class CallEstimate:
    input_tokens: int
    output_tokens: int
    cost_usd: float
    pricing_source: str


def estimate_text_tokens(text: str) -> int:
    return max(1, ceil(len(text) / 4))


def estimate_call_cost(
    spec: ModelSpec,
    input_tokens: int,
    output_tokens: int = LLM_MAX_TOKENS,
    env: Mapping[str, str] | None = None,
) -> CallEstimate:
    env = os.environ if env is None else env
    if not spec.is_metered:
        return CallEstimate(input_tokens, output_tokens, 0.0, "free")

    if spec.input_cost_per_million is None or spec.output_cost_per_million is None:
        input_rate = _float_env(env, "CURIA_UNKNOWN_INPUT_COST_PER_MTOK", DEFAULT_UNKNOWN_INPUT_COST_PER_MTOK)
        output_rate = _float_env(env, "CURIA_UNKNOWN_OUTPUT_COST_PER_MTOK", DEFAULT_UNKNOWN_OUTPUT_COST_PER_MTOK)
        pricing_source = "fallback"
    else:
        input_rate = spec.input_cost_per_million
        output_rate = spec.output_cost_per_million
        pricing_source = "registry"

    cost = ((input_tokens / 1_000_000) * input_rate) + ((output_tokens / 1_000_000) * output_rate)
    return CallEstimate(input_tokens, output_tokens, round(cost, 8), pricing_source)


def estimate_prompt_cost(
    spec: ModelSpec,
    prompt: str,
    output_tokens: int = LLM_MAX_TOKENS,
    env: Mapping[str, str] | None = None,
) -> CallEstimate:
    return estimate_call_cost(spec, estimate_text_tokens(prompt), output_tokens, env)


def usage_cost(
    spec: ModelSpec,
    usage: Mapping[str, int | None] | None,
    fallback: CallEstimate,
    env: Mapping[str, str] | None = None,
) -> CallEstimate:
    if not usage:
        return fallback
    input_tokens = int(usage.get("input_tokens") or fallback.input_tokens)
    output_tokens = int(usage.get("output_tokens") or fallback.output_tokens)
    estimate = estimate_call_cost(spec, input_tokens, output_tokens, env)
    return CallEstimate(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=estimate.cost_usd,
        pricing_source=estimate.pricing_source,
    )


def can_afford(spent_usd: float, estimate: CallEstimate, budget_usd: float | None) -> bool:
    if budget_usd is None:
        return True
    return spent_usd + estimate.cost_usd <= budget_usd + 1e-9


def _float_env(env: Mapping[str, str], key: str, default: float) -> float:
    raw = env.get(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default
