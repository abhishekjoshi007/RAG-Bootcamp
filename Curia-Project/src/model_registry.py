from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Mapping, Sequence


_BASE_URL_ENVS = {
    "openai": ("OPENAI_BASE_URL", "CURIA_OPENAI_BASE_URL"),
    "anthropic": ("ANTHROPIC_BASE_URL", "CURIA_ANTHROPIC_BASE_URL"),
    "google": ("GOOGLE_OPENAI_BASE_URL", "GOOGLE_BASE_URL", "CURIA_GOOGLE_BASE_URL"),
    "xai": ("XAI_BASE_URL", "CURIA_XAI_BASE_URL"),
    "mistral": ("MISTRAL_BASE_URL", "CURIA_MISTRAL_BASE_URL"),
    "deepseek": ("DEEPSEEK_BASE_URL", "CURIA_DEEPSEEK_BASE_URL"),
    "openrouter": ("OPENROUTER_BASE_URL", "CURIA_OPENROUTER_BASE_URL"),
    "together": ("TOGETHER_BASE_URL", "CURIA_TOGETHER_BASE_URL"),
    "ollama": ("OLLAMA_BASE_URL", "CURIA_OLLAMA_BASE_URL"),
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    provider: str
    model_id: str
    adapter: str
    api_key_env: str | None = None
    base_url: str | None = None
    input_cost_per_million: float | None = None
    output_cost_per_million: float | None = None
    max_tokens: int | None = None
    enabled: bool = True
    default: bool = False
    category: str = "closed"
    priority: int = 100
    notes: str = ""

    def has_credentials(self, env: Mapping[str, str] | None = None) -> bool:
        if not self.api_key_env:
            return True
        env = os.environ if env is None else env
        return bool(env.get(self.api_key_env))

    def resolved_base_url(self, env: Mapping[str, str] | None = None) -> str | None:
        env = os.environ if env is None else env
        for key in _BASE_URL_ENVS.get(self.provider, ()):
            value = env.get(key)
            if value:
                return value
        return self.base_url

    @property
    def is_metered(self) -> bool:
        return self.adapter not in {"local", "ollama"}


DEFAULT_MODEL_REGISTRY: tuple[ModelSpec, ...] = (
    ModelSpec(
        name="local",
        provider="local",
        model_id="local",
        adapter="local",
        input_cost_per_million=0.0,
        output_cost_per_million=0.0,
        default=True,
        category="baseline",
        priority=0,
        notes="Deterministic offline grounded generator.",
    ),
    ModelSpec(
        name="gpt-5.4-nano",
        provider="openai",
        model_id="gpt-5.4-nano",
        adapter="openai_compatible",
        api_key_env="OPENAI_API_KEY",
        default=True,
        category="closed",
        priority=10,
    ),
    ModelSpec(
        name="gpt-5.4-mini",
        provider="openai",
        model_id="gpt-5.4-mini",
        adapter="openai_compatible",
        api_key_env="OPENAI_API_KEY",
        default=True,
        category="closed",
        priority=11,
    ),
    ModelSpec(
        name="gpt-5.4",
        provider="openai",
        model_id="gpt-5.4",
        adapter="openai_compatible",
        api_key_env="OPENAI_API_KEY",
        default=True,
        category="closed",
        priority=12,
    ),
    ModelSpec(
        name="gpt-5.5",
        provider="openai",
        model_id="gpt-5.5",
        adapter="openai_compatible",
        api_key_env="OPENAI_API_KEY",
        default=True,
        category="closed",
        priority=20,
    ),
    ModelSpec(
        name="gpt-5.5-pro",
        provider="openai",
        model_id="gpt-5.5-pro",
        adapter="openai_compatible",
        api_key_env="OPENAI_API_KEY",
        default=False,
        category="closed",
        priority=21,
        notes="Not supported by the current Chat Completions adapter; request explicitly only for experiments.",
    ),
    ModelSpec(
        name="claude-haiku-4-5",
        provider="anthropic",
        model_id="claude-haiku-4-5",
        adapter="anthropic",
        api_key_env="ANTHROPIC_API_KEY",
        input_cost_per_million=1.0,
        output_cost_per_million=5.0,
        default=True,
        category="closed",
        priority=30,
    ),
    ModelSpec(
        name="claude-sonnet-4-5",
        provider="anthropic",
        model_id="claude-sonnet-4-5",
        adapter="anthropic",
        api_key_env="ANTHROPIC_API_KEY",
        input_cost_per_million=3.0,
        output_cost_per_million=15.0,
        default=True,
        category="closed",
        priority=31,
    ),
    ModelSpec(
        name="claude-sonnet-4-6",
        provider="anthropic",
        model_id="claude-sonnet-4-6",
        adapter="anthropic",
        api_key_env="ANTHROPIC_API_KEY",
        input_cost_per_million=3.0,
        output_cost_per_million=15.0,
        default=True,
        category="closed",
        priority=32,
    ),
    ModelSpec(
        name="claude-opus-4-6",
        provider="anthropic",
        model_id="claude-opus-4-6",
        adapter="anthropic",
        api_key_env="ANTHROPIC_API_KEY",
        input_cost_per_million=15.0,
        output_cost_per_million=75.0,
        default=True,
        category="closed",
        priority=33,
        notes="Pinned Opus 4.x variant; provider may reject if unavailable on the account.",
    ),
    ModelSpec(
        name="claude-opus-4-7",
        provider="anthropic",
        model_id="claude-opus-4-7",
        adapter="anthropic",
        api_key_env="ANTHROPIC_API_KEY",
        input_cost_per_million=15.0,
        output_cost_per_million=75.0,
        default=True,
        category="closed",
        priority=34,
        notes="Pinned Opus 4.x variant; provider may reject if unavailable on the account.",
    ),
    ModelSpec(
        name="claude-opus-4-8",
        provider="anthropic",
        model_id="claude-opus-4-8",
        adapter="anthropic",
        api_key_env="ANTHROPIC_API_KEY",
        input_cost_per_million=15.0,
        output_cost_per_million=75.0,
        default=True,
        category="closed",
        priority=35,
        notes="Expensive calibration model; budget preflight may skip later items.",
    ),
    ModelSpec(
        name="gemini-3.5-flash-lite-preview",
        provider="google",
        model_id="gemini-3.5-flash-lite-preview",
        adapter="openai_compatible",
        api_key_env="GOOGLE_API_KEY",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        max_tokens=2048,
        default=False,
        category="closed",
        priority=40,
        notes="Not available on the tested Google OpenAI-compatible endpoint.",
    ),
    ModelSpec(
        name="gemini-3.5-flash",
        provider="google",
        model_id="gemini-3.5-flash",
        adapter="openai_compatible",
        api_key_env="GOOGLE_API_KEY",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        max_tokens=2048,
        default=True,
        category="closed",
        priority=41,
    ),
    ModelSpec(
        name="gemini-3.1-pro-preview",
        provider="google",
        model_id="gemini-3.1-pro-preview",
        adapter="openai_compatible",
        api_key_env="GOOGLE_API_KEY",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        max_tokens=2048,
        default=False,
        category="closed",
        priority=42,
        notes="Google quota was unavailable on the tested account; request explicitly after quota/billing is enabled.",
    ),
    ModelSpec(
        name="grok-4.3-fast",
        provider="xai",
        model_id="grok-4.3-fast",
        adapter="openai_compatible",
        api_key_env="XAI_API_KEY",
        base_url="https://api.x.ai/v1",
        default=False,
        category="closed",
        priority=50,
        notes="Not available on the tested xAI endpoint/account.",
    ),
    ModelSpec(
        name="grok-4.3-fast-reasoning",
        provider="xai",
        model_id="grok-4.3-fast-reasoning",
        adapter="openai_compatible",
        api_key_env="XAI_API_KEY",
        base_url="https://api.x.ai/v1",
        default=False,
        category="closed",
        priority=51,
        notes="Not available on the tested xAI endpoint/account.",
    ),
    ModelSpec(
        name="grok-4.3",
        provider="xai",
        model_id="grok-4.3",
        adapter="openai_compatible",
        api_key_env="XAI_API_KEY",
        base_url="https://api.x.ai/v1",
        default=True,
        category="closed",
        priority=52,
    ),
    ModelSpec(
        name="deepseek-chat",
        provider="deepseek",
        model_id="deepseek-chat",
        adapter="openai_compatible",
        api_key_env="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com",
        default=True,
        category="open_weight_hosted",
        priority=60,
    ),
    ModelSpec(
        name="deepseek-reasoner",
        provider="deepseek",
        model_id="deepseek-reasoner",
        adapter="openai_compatible",
        api_key_env="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com",
        default=True,
        category="open_weight_hosted",
        priority=61,
    ),
    ModelSpec(
        name="deepseek-v4-flash",
        provider="deepseek",
        model_id="deepseek-v4-flash",
        adapter="openai_compatible",
        api_key_env="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com",
        default=True,
        category="open_weight_hosted",
        priority=62,
    ),
    ModelSpec(
        name="deepseek-v4-pro",
        provider="deepseek",
        model_id="deepseek-v4-pro",
        adapter="openai_compatible",
        api_key_env="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com",
        default=False,
        category="open_weight_hosted",
        priority=63,
        notes="Provider returned non-JSON output in the tested RAG generation path.",
    ),
    ModelSpec(
        name="mistral-medium-3.5",
        provider="mistral",
        model_id="mistral-medium-latest",
        adapter="openai_compatible",
        api_key_env="MISTRAL_API_KEY",
        base_url="https://api.mistral.ai/v1",
        default=False,
        category="closed",
        priority=70,
        notes="Alias for the current hosted Mistral Medium class.",
    ),
    ModelSpec(
        name="mistral-large-3",
        provider="mistral",
        model_id="mistral-large-latest",
        adapter="openai_compatible",
        api_key_env="MISTRAL_API_KEY",
        base_url="https://api.mistral.ai/v1",
        default=False,
        category="open_weight_hosted",
        priority=71,
        notes="Alias for the current hosted Mistral Large class.",
    ),
    ModelSpec(
        name="ministral-3",
        provider="mistral",
        model_id="ministral-3b-latest",
        adapter="openai_compatible",
        api_key_env="MISTRAL_API_KEY",
        base_url="https://api.mistral.ai/v1",
        default=False,
        category="open_weight_hosted",
        priority=72,
    ),
    ModelSpec(
        name="openrouter-llama-4-maverick",
        provider="openrouter",
        model_id="meta-llama/llama-4-maverick",
        adapter="openai_compatible",
        api_key_env="OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
        default=False,
        category="open_weight_hosted",
        priority=80,
    ),
    ModelSpec(
        name="openrouter-llama-4-scout",
        provider="openrouter",
        model_id="meta-llama/llama-4-scout",
        adapter="openai_compatible",
        api_key_env="OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
        default=False,
        category="open_weight_hosted",
        priority=81,
    ),
    ModelSpec(
        name="openrouter-qwen3.7-max",
        provider="openrouter",
        model_id="qwen/qwen3.7-max",
        adapter="openai_compatible",
        api_key_env="OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
        default=False,
        category="open_weight_hosted",
        priority=82,
    ),
    ModelSpec(
        name="together-llama-4-maverick",
        provider="together",
        model_id="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        adapter="openai_compatible",
        api_key_env="TOGETHER_API_KEY",
        base_url="https://api.together.xyz/v1",
        default=False,
        category="open_weight_hosted",
        priority=88,
    ),
    ModelSpec(
        name="ollama-qwen-small",
        provider="ollama",
        model_id="qwen2.5:7b-instruct",
        adapter="ollama",
        base_url="http://localhost:11434",
        input_cost_per_million=0.0,
        output_cost_per_million=0.0,
        default=False,
        category="local",
        priority=90,
    ),
    ModelSpec(
        name="ollama-llama-small",
        provider="ollama",
        model_id="llama3.1:8b-instruct-q4_K_M",
        adapter="ollama",
        base_url="http://localhost:11434",
        input_cost_per_million=0.0,
        output_cost_per_million=0.0,
        default=False,
        category="local",
        priority=91,
    ),
)


_REGISTRY_BY_NAME = {spec.name: spec for spec in DEFAULT_MODEL_REGISTRY}
_REGISTRY_BY_MODEL_ID = {spec.model_id: spec for spec in DEFAULT_MODEL_REGISTRY}


_PROVIDER_DEFAULTS = {
    "openai": ("openai_compatible", "OPENAI_API_KEY", None, "closed"),
    "anthropic": ("anthropic", "ANTHROPIC_API_KEY", None, "closed"),
    "google": ("openai_compatible", "GOOGLE_API_KEY", "https://generativelanguage.googleapis.com/v1beta/openai/", "closed"),
    "xai": ("openai_compatible", "XAI_API_KEY", "https://api.x.ai/v1", "closed"),
    "mistral": ("openai_compatible", "MISTRAL_API_KEY", "https://api.mistral.ai/v1", "open_weight_hosted"),
    "deepseek": ("openai_compatible", "DEEPSEEK_API_KEY", "https://api.deepseek.com", "open_weight_hosted"),
    "openrouter": ("openai_compatible", "OPENROUTER_API_KEY", "https://openrouter.ai/api/v1", "open_weight_hosted"),
    "together": ("openai_compatible", "TOGETHER_API_KEY", "https://api.together.xyz/v1", "open_weight_hosted"),
}


def registry_by_name() -> dict[str, ModelSpec]:
    return dict(_REGISTRY_BY_NAME)


def parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def select_model_specs(
    requested: Sequence[str] | None = None,
    providers: Sequence[str] | None = None,
) -> list[ModelSpec]:
    requested = list(requested or [])
    provider_set = {provider.strip().lower() for provider in providers or [] if provider.strip()}

    if requested:
        specs = [_resolve_model_spec(name, provider_set) for name in requested]
    else:
        specs = [spec for spec in DEFAULT_MODEL_REGISTRY if spec.default]
        specs.sort(key=lambda spec: spec.priority)

    specs = _dedupe_specs([_REGISTRY_BY_NAME["local"], *specs])
    if provider_set:
        specs = [
            spec for spec in specs
            if spec.provider == "local" or spec.provider in provider_set
        ]
    return [spec for spec in specs if spec.enabled]


def _dedupe_specs(specs: Sequence[ModelSpec]) -> list[ModelSpec]:
    seen: set[str] = set()
    unique: list[ModelSpec] = []
    for spec in specs:
        key = spec.name
        if key in seen:
            continue
        seen.add(key)
        unique.append(spec)
    return unique


def _resolve_model_spec(name: str, providers: set[str] | None = None) -> ModelSpec:
    if name in _REGISTRY_BY_NAME:
        return _REGISTRY_BY_NAME[name]
    if name in _REGISTRY_BY_MODEL_ID:
        return _REGISTRY_BY_MODEL_ID[name]
    if name.startswith("ollama:"):
        model_id = name.split(":", 1)[1]
        return replace(
            _REGISTRY_BY_NAME["ollama-qwen-small"],
            name=name,
            model_id=model_id,
            default=False,
        )

    provider = _infer_provider(name, providers or set())
    adapter, api_key_env, base_url, category = _PROVIDER_DEFAULTS[provider]
    return ModelSpec(
        name=name,
        provider=provider,
        model_id=name,
        adapter=adapter,
        api_key_env=api_key_env,
        base_url=base_url,
        default=False,
        category=category,
        priority=100,
        notes="Ad hoc model inferred from CLI selector.",
    )


def _infer_provider(name: str, providers: set[str]) -> str:
    nonlocal_providers = [provider for provider in providers if provider != "local"]
    if len(nonlocal_providers) == 1 and nonlocal_providers[0] in _PROVIDER_DEFAULTS:
        return nonlocal_providers[0]
    lowered = name.lower()
    if lowered.startswith("claude"):
        return "anthropic"
    if lowered.startswith("gemini"):
        return "google"
    if lowered.startswith("grok"):
        return "xai"
    if lowered.startswith("deepseek"):
        return "deepseek"
    if lowered.startswith(("mistral", "ministral")):
        return "mistral"
    if "/" in lowered:
        return "openrouter"
    return "openai"
