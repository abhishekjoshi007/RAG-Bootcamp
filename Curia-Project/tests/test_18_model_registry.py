from __future__ import annotations

from src.model_registry import ModelSpec, registry_by_name, select_model_specs


def test_registry_contains_required_model_fields():
    spec = registry_by_name()["gpt-5.4-mini"]
    assert spec.provider == "openai"
    assert spec.model_id == "gpt-5.4-mini"
    assert spec.api_key_env == "OPENAI_API_KEY"
    assert spec.category == "closed"


def test_select_default_specs_always_includes_local_first():
    specs = select_model_specs()
    assert specs[0].name == "local"
    assert any(spec.name == "claude-sonnet-4-6" for spec in specs)
    assert any(spec.name == "claude-opus-4-8" for spec in specs)
    assert any(spec.name == "gpt-5.5" for spec in specs)


def test_provider_filter_keeps_local_baseline():
    specs = select_model_specs(providers=["anthropic"])
    assert specs[0].name == "local"
    assert {spec.provider for spec in specs} <= {"local", "anthropic"}


def test_mistral_and_together_are_not_default_providers():
    specs = select_model_specs()
    providers = {spec.provider for spec in specs}
    assert "mistral" not in providers
    assert "together" not in providers


def test_known_unavailable_models_are_not_defaults():
    names = {spec.name for spec in select_model_specs()}
    assert "gpt-5.5-pro" not in names
    assert "gemini-3.5-flash-lite-preview" not in names
    assert "gemini-3.1-pro-preview" not in names
    assert "grok-4.3-fast" not in names
    assert "deepseek-v4-pro" not in names


def test_requested_unknown_gpt_model_is_inferred_as_openai():
    specs = select_model_specs(["gpt-4o-mini"])
    selected = {spec.name: spec for spec in specs}
    assert selected["local"].adapter == "local"
    assert selected["gpt-4o-mini"].provider == "openai"
    assert selected["gpt-4o-mini"].api_key_env == "OPENAI_API_KEY"


def test_ollama_selector_builds_local_runtime_spec():
    specs = select_model_specs(["ollama:qwen2.5:7b"])
    spec = next(item for item in specs if item.name == "ollama:qwen2.5:7b")
    assert spec.provider == "ollama"
    assert spec.adapter == "ollama"
    assert spec.model_id == "qwen2.5:7b"


def test_missing_key_behavior_uses_declared_env_var(monkeypatch):
    monkeypatch.delenv("FAKE_PROVIDER_KEY", raising=False)
    spec = ModelSpec(
        name="fake",
        provider="openai",
        model_id="fake-model",
        adapter="openai_compatible",
        api_key_env="FAKE_PROVIDER_KEY",
    )
    assert not spec.has_credentials()
    monkeypatch.setenv("FAKE_PROVIDER_KEY", "secret")
    assert spec.has_credentials()
