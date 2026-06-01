from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from openai import OpenAI

from .config import LLM_HTTP_TIMEOUT, LLM_MAX_RETRIES, LLM_MAX_TOKENS, LLM_TEMPERATURE
from .llm import LocalGroundedGenerator, parse_json_response
from .model_registry import ModelSpec
from .models import Recommendation, SearchResult
from .prompts import build_recommendation_prompt


class ProviderUnavailableError(RuntimeError):
    pass


class LocalRuntimeUnavailableError(ProviderUnavailableError):
    pass


class OpenAICompatibleGenerator:
    def __init__(
        self,
        spec: ModelSpec,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
    ) -> None:
        self.spec = spec
        self.model = spec.model_id
        self.temperature = temperature
        self.max_tokens = spec.max_tokens or max_tokens
        self.last_usage: dict[str, int | None] | None = None
        self._client: OpenAI | None = None

    @property
    def _openai(self) -> OpenAI:
        if self._client is None:
            if not self.spec.api_key_env:
                raise EnvironmentError(f"{self.spec.name} does not define an API key env var")
            key = os.environ.get(self.spec.api_key_env)
            if not key:
                raise EnvironmentError(f"{self.spec.api_key_env} environment variable is not set")
            kwargs: dict[str, Any] = {"api_key": key, "timeout": LLM_HTTP_TIMEOUT}
            base_url = self.spec.resolved_base_url()
            if base_url:
                kwargs["base_url"] = base_url
            headers = _default_headers(self.spec)
            if headers:
                kwargs["default_headers"] = headers
            self._client = OpenAI(**kwargs)
        return self._client

    def build_chat_completion_kwargs(self, prompt: str) -> dict[str, Any]:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": prompt}],
        }

    def generate(self, unit: dict, evidence: list[SearchResult]) -> Recommendation:
        prompt = build_recommendation_prompt(unit, evidence)
        last_exc: Exception = RuntimeError("No attempts made")
        for _ in range(LLM_MAX_RETRIES):
            try:
                response = self._create_chat_completion(prompt)
                self.last_usage = _openai_usage(response)
                raw = response.choices[0].message.content or ""
                data = parse_json_response(raw)
                return _recommendation_from_json(data)
            except (json.JSONDecodeError, KeyError, IndexError, AttributeError) as exc:
                last_exc = exc
        raise RuntimeError(f"LLM parse failed after {LLM_MAX_RETRIES} retries: {last_exc}") from last_exc

    def _create_chat_completion(self, prompt: str):
        kwargs = self.build_chat_completion_kwargs(prompt)
        candidates = [kwargs]
        if "temperature" in kwargs:
            no_temperature = dict(kwargs)
            no_temperature.pop("temperature", None)
            candidates.append(no_temperature)
        candidates.extend(_max_completion_token_variants(candidates))
        candidates.extend(_without_response_format_variants(candidates))

        last_exc: Exception | None = None
        for candidate in _dedupe_kwargs(candidates):
            try:
                return self._openai.chat.completions.create(**candidate)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if not _should_try_next_chat_variant(exc):
                    raise
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("No chat completion request variants were attempted")


class AnthropicGenerator:
    def __init__(
        self,
        spec: ModelSpec,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
    ) -> None:
        from anthropic import Anthropic

        if not spec.api_key_env:
            raise EnvironmentError(f"{spec.name} does not define an API key env var")
        key = os.environ.get(spec.api_key_env)
        if not key:
            raise EnvironmentError(f"{spec.api_key_env} environment variable is not set")
        self.spec = spec
        self.model = spec.model_id
        self.temperature = temperature
        self.max_tokens = spec.max_tokens or max_tokens
        self.last_usage: dict[str, int | None] | None = None
        self._client = Anthropic(api_key=key, timeout=LLM_HTTP_TIMEOUT)

    def generate(self, unit: dict, evidence: list[SearchResult]) -> Recommendation:
        prompt = build_recommendation_prompt(unit, evidence)
        last_exc: Exception = RuntimeError("No attempts made")
        for _ in range(LLM_MAX_RETRIES):
            try:
                response = self._create_message(prompt)
                self.last_usage = _anthropic_usage(response)
                data = parse_json_response(_anthropic_text(response))
                return _recommendation_from_json(data)
            except (json.JSONDecodeError, KeyError, IndexError, AttributeError) as exc:
                last_exc = exc
        raise RuntimeError(f"LLM parse failed after {LLM_MAX_RETRIES} retries: {last_exc}") from last_exc

    def _create_message(self, prompt: str):
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        candidates = [kwargs]
        no_temperature = dict(kwargs)
        no_temperature.pop("temperature", None)
        candidates.append(no_temperature)

        last_exc: Exception | None = None
        for candidate in _dedupe_kwargs(candidates):
            try:
                return self._client.messages.create(**candidate)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if not _looks_like_temperature_error(exc):
                    raise
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("No Anthropic request variants were attempted")


class OllamaGenerator:
    def __init__(
        self,
        spec: ModelSpec,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
    ) -> None:
        self.spec = spec
        self.model = spec.model_id
        self.temperature = temperature
        self.max_tokens = spec.max_tokens or max_tokens
        self.base_url = (spec.resolved_base_url() or "http://localhost:11434").rstrip("/")
        self.last_usage: dict[str, int | None] | None = None

    def generate(self, unit: dict, evidence: list[SearchResult]) -> Recommendation:
        prompt = build_recommendation_prompt(unit, evidence)
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "format": "json",
            "options": {"temperature": self.temperature, "num_predict": self.max_tokens},
        }
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as exc:
            raise ProviderUnavailableError(f"Ollama request failed: {exc}") from exc
        self.last_usage = {
            "input_tokens": data.get("prompt_eval_count"),
            "output_tokens": data.get("eval_count"),
        }
        raw = data.get("message", {}).get("content", "")
        return _recommendation_from_json(parse_json_response(raw))


def make_generator(spec: ModelSpec, local_runtime: str = "none"):
    if spec.adapter == "local":
        return LocalGroundedGenerator()
    if spec.adapter == "openai_compatible":
        return OpenAICompatibleGenerator(spec)
    if spec.adapter == "anthropic":
        return AnthropicGenerator(spec)
    if spec.adapter == "ollama":
        if local_runtime != "ollama":
            raise LocalRuntimeUnavailableError("Use --local-runtime ollama to enable Ollama-backed models")
        if not ollama_available(spec):
            raise LocalRuntimeUnavailableError("Ollama is not reachable at the configured base URL")
        return OllamaGenerator(spec)
    raise ProviderUnavailableError(f"Unsupported adapter: {spec.adapter}")


def ollama_available(spec: ModelSpec) -> bool:
    base_url = (spec.resolved_base_url() or "http://localhost:11434").rstrip("/")
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=1.5):
            return True
    except (urllib.error.URLError, TimeoutError):
        return False


def _recommendation_from_json(data: dict) -> Recommendation:
    return Recommendation(
        signal_strength=data.get("signal_strength", "low"),
        summary=data.get("summary", ""),
        emerging_topics=list(data.get("emerging_topics", [])),
        evidence_ids=list(data.get("evidence_ids", [])),
    )


def _openai_usage(response) -> dict[str, int | None] | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    return {
        "input_tokens": getattr(usage, "prompt_tokens", None),
        "output_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def _anthropic_usage(response) -> dict[str, int | None] | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    return {
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
    }


def _anthropic_text(response) -> str:
    parts: list[str] = []
    for block in getattr(response, "content", []):
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts)


def _default_headers(spec: ModelSpec) -> dict[str, str]:
    if spec.provider != "openrouter":
        return {}
    headers: dict[str, str] = {}
    referer = os.environ.get("OPENROUTER_SITE_URL")
    title = os.environ.get("OPENROUTER_APP_NAME", "CURIA RAG benchmark")
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title
    return headers


def _looks_like_max_tokens_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "max_tokens" in text and "max_completion_tokens" in text


def _looks_like_temperature_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "temperature" in text and (
        "unsupported" in text or "default" in text or "deprecated" in text
    )


def _looks_like_response_format_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "response_format" in text or "json_object" in text


def _should_try_next_chat_variant(exc: Exception) -> bool:
    return (
        _looks_like_max_tokens_error(exc)
        or _looks_like_temperature_error(exc)
        or _looks_like_response_format_error(exc)
    )


def _max_completion_token_variants(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    for candidate in candidates:
        if "max_tokens" not in candidate:
            continue
        variant = dict(candidate)
        variant["max_completion_tokens"] = variant.pop("max_tokens")
        variants.append(variant)
    return variants


def _without_response_format_variants(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    for candidate in candidates:
        if "response_format" not in candidate:
            continue
        variant = dict(candidate)
        variant.pop("response_format", None)
        variants.append(variant)
    return variants


def _dedupe_kwargs(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = json.dumps(candidate, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique
