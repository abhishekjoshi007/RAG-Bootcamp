from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_backtest_module():
    path = Path(__file__).resolve().parents[1] / "eval" / "run_forecast_backtest.py"
    spec = importlib.util.spec_from_file_location("run_forecast_backtest", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_seasonal_naive_repeats_values_from_previous_year():
    backtest = _load_backtest_module()
    train = [float(i) for i in range(1, 25)]
    assert backtest._forecast("seasonal_naive", train, 4) == [13.0, 14.0, 15.0, 16.0]


def test_seasonal_naive_falls_back_to_last_value_when_short():
    backtest = _load_backtest_module()
    assert backtest._forecast("seasonal_naive", [1.0, 2.0, 3.0], 3) == [3.0, 3.0, 3.0]


def test_exp_smoothing_forecast_returns_requested_horizon():
    backtest = _load_backtest_module()
    pred = backtest._forecast(
        "exp_smoothing",
        [0.1, 0.2, 0.25, 0.3],
        3,
        train_months=["2024-01", "2024-02", "2024-03", "2024-04"],
    )
    assert len(pred) == 3
    assert all(value >= 0.0 for value in pred)


def test_model_comparison_reports_non_naive_delta():
    backtest = _load_backtest_module()
    comparison = backtest._model_comparison({
        "naive": {"smape": 0.5},
        "seasonal_naive": {"smape": 0.4},
        "moving_avg": {"smape": 0.6},
        "linear": {"smape": 0.55},
        "exp_smoothing": {"smape": 0.45},
    })
    assert comparison["best_by_smape"] == "seasonal_naive"
    assert comparison["best_non_naive_by_smape"] == "seasonal_naive"
    assert comparison["best_non_naive_beats_naive"] is True
    assert comparison["best_non_naive_smape_delta_vs_naive"] == -0.1
