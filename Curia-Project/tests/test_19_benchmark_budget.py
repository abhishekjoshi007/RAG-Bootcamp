from __future__ import annotations

from src.benchmarking import can_afford, estimate_call_cost, estimate_text_tokens, usage_cost
from src.model_registry import ModelSpec


def _metered_spec() -> ModelSpec:
    return ModelSpec(
        name="priced",
        provider="openai",
        model_id="priced",
        adapter="openai_compatible",
        api_key_env="OPENAI_API_KEY",
        input_cost_per_million=1.0,
        output_cost_per_million=2.0,
    )


def test_estimate_text_tokens_is_rough_char_quarter():
    assert estimate_text_tokens("abcd") == 1
    assert estimate_text_tokens("abcde") == 2


def test_budget_preflight_uses_fake_registry_prices():
    estimate = estimate_call_cost(_metered_spec(), input_tokens=1_000, output_tokens=500)
    assert estimate.pricing_source == "registry"
    assert estimate.cost_usd == 0.002
    assert can_afford(0.0, estimate, 0.01)
    assert not can_afford(0.009, estimate, 0.01)


def test_unknown_pricing_uses_env_fallback(monkeypatch):
    monkeypatch.setenv("CURIA_UNKNOWN_INPUT_COST_PER_MTOK", "10")
    monkeypatch.setenv("CURIA_UNKNOWN_OUTPUT_COST_PER_MTOK", "20")
    spec = ModelSpec(
        name="unknown",
        provider="openai",
        model_id="unknown",
        adapter="openai_compatible",
        api_key_env="OPENAI_API_KEY",
    )
    estimate = estimate_call_cost(spec, input_tokens=1_000, output_tokens=500)
    assert estimate.pricing_source == "fallback"
    assert estimate.cost_usd == 0.02


def test_usage_cost_prefers_provider_token_counts():
    fallback = estimate_call_cost(_metered_spec(), input_tokens=1_000, output_tokens=500)
    actual = usage_cost(
        _metered_spec(),
        {"input_tokens": 2_000, "output_tokens": 1_000},
        fallback,
    )
    assert actual.input_tokens == 2_000
    assert actual.output_tokens == 1_000
    assert actual.cost_usd == 0.004
