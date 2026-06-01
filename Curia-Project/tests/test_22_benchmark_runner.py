from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_runner_module():
    path = Path(__file__).resolve().parents[1] / "eval" / "run_multi_llm_eval.py"
    spec = importlib.util.spec_from_file_location("run_multi_llm_eval", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hard_provider_error_detects_billing_failure():
    runner = _load_runner_module()
    detail = "BadRequestError: Your credit balance is too low to access the API"
    assert runner._is_hard_provider_error(detail)


def test_hard_provider_error_ignores_parse_failure():
    runner = _load_runner_module()
    detail = "RuntimeError: LLM parse failed after 3 retries"
    assert not runner._is_hard_provider_error(detail)
