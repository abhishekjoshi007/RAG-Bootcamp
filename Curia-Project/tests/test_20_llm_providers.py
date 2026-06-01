from __future__ import annotations

from src.llm_providers import OpenAICompatibleGenerator
from src.config import LLM_HTTP_TIMEOUT
from src.model_registry import ModelSpec


def test_openai_compatible_request_kwargs_are_chat_completion_shaped():
    spec = ModelSpec(
        name="fake",
        provider="xai",
        model_id="grok-test",
        adapter="openai_compatible",
        api_key_env="XAI_API_KEY",
        base_url="https://api.x.ai/v1",
    )
    gen = OpenAICompatibleGenerator(spec, temperature=0.25, max_tokens=123)
    kwargs = gen.build_chat_completion_kwargs("hello")
    assert kwargs["model"] == "grok-test"
    assert kwargs["temperature"] == 0.25
    assert kwargs["max_tokens"] == 123
    assert kwargs["response_format"] == {"type": "json_object"}
    assert kwargs["messages"] == [{"role": "user", "content": "hello"}]


def test_openai_compatible_client_uses_spec_base_url(monkeypatch):
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    spec = ModelSpec(
        name="fake",
        provider="xai",
        model_id="grok-test",
        adapter="openai_compatible",
        api_key_env="XAI_API_KEY",
        base_url="https://api.x.ai/v1",
    )
    monkeypatch.setenv("XAI_API_KEY", "secret")
    monkeypatch.setattr("src.llm_providers.OpenAI", FakeOpenAI)

    gen = OpenAICompatibleGenerator(spec)
    _ = gen._openai

    assert captured["api_key"] == "secret"
    assert captured["base_url"] == "https://api.x.ai/v1"
    assert captured["timeout"] == LLM_HTTP_TIMEOUT


def test_openai_compatible_client_allows_env_base_url_override(monkeypatch):
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    spec = ModelSpec(
        name="fake",
        provider="deepseek",
        model_id="deepseek-test",
        adapter="openai_compatible",
        api_key_env="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com",
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://example.test/v1")
    monkeypatch.setattr("src.llm_providers.OpenAI", FakeOpenAI)

    gen = OpenAICompatibleGenerator(spec)
    _ = gen._openai

    assert captured["base_url"] == "https://example.test/v1"


def test_openai_compatible_retries_with_max_completion_tokens():
    calls = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            if "max_tokens" in kwargs:
                raise RuntimeError("Use max_completion_tokens instead of max_tokens")
            return object()

    class FakeClient:
        class Chat:
            completions = FakeCompletions()

        chat = Chat()

    spec = ModelSpec(
        name="fake",
        provider="openai",
        model_id="gpt-test",
        adapter="openai_compatible",
        api_key_env="OPENAI_API_KEY",
    )
    gen = OpenAICompatibleGenerator(spec, max_tokens=77)
    gen._client = FakeClient()

    _ = gen._create_chat_completion("hello")

    assert any("max_tokens" in call for call in calls)
    max_completion_calls = [call for call in calls if "max_completion_tokens" in call]
    assert max_completion_calls
    assert max_completion_calls[0]["max_completion_tokens"] == 77


def test_openai_compatible_retries_without_temperature():
    calls = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            if "temperature" in kwargs:
                raise RuntimeError("temperature only supports the default value")
            return object()

    class FakeClient:
        class Chat:
            completions = FakeCompletions()

        chat = Chat()

    spec = ModelSpec(
        name="fake",
        provider="openai",
        model_id="gpt-test",
        adapter="openai_compatible",
        api_key_env="OPENAI_API_KEY",
    )
    gen = OpenAICompatibleGenerator(spec)
    gen._client = FakeClient()

    _ = gen._create_chat_completion("hello")

    assert "temperature" in calls[0]
    assert "temperature" not in calls[1]
