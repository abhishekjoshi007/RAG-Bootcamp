"""Step 13 — Agent B Forecasting: data generation, forecasting accuracy, backtesting."""

import math
import pytest
from src.forecasting import (
    DataPoint,
    ForecastResult,
    SkillForecaster,
    _generate_historical,
    _linear_forecast,
    _exp_smoothing_forecast,
    _month_sequence,
)


def test_month_sequence_length():
    seq = _month_sequence("2022-01", 12)
    assert len(seq) == 12


def test_month_sequence_format():
    seq = _month_sequence("2022-01", 3)
    assert seq == ["2022-01", "2022-02", "2022-03"]


def test_month_sequence_year_rollover():
    seq = _month_sequence("2022-11", 4)
    assert seq == ["2022-11", "2022-12", "2023-01", "2023-02"]


def test_generate_historical_known_skill():
    history = _generate_historical("machine learning", months=24)
    assert len(history) == 24


def test_generate_historical_unknown_skill():
    history = _generate_historical("nonexistent_skill_xyz", months=12)
    assert history == ()


def test_historical_values_in_range():
    history = _generate_historical("kubernetes", months=40)
    for dp in history:
        assert 0.0 <= dp.frequency <= 1.0, f"Out of range: {dp.frequency}"


def test_historical_datapoints_are_frozen():
    from dataclasses import FrozenInstanceError
    dp = DataPoint(month="2024-01", frequency=0.5)
    with pytest.raises((FrozenInstanceError, AttributeError)):
        dp.frequency = 0.9


def test_historical_deterministic():
    h1 = _generate_historical("devops", months=20)
    h2 = _generate_historical("devops", months=20)
    assert h1 == h2, "Same skill should produce same history"


def test_historical_different_skills_differ():
    h1 = _generate_historical("machine learning", months=12)
    h2 = _generate_historical("kubernetes", months=12)
    freqs1 = [p.frequency for p in h1]
    freqs2 = [p.frequency for p in h2]
    assert freqs1 != freqs2


def test_historical_months_correct():
    history = _generate_historical("python", "2023-06", months=6)
    months = [dp.month for dp in history]
    assert months == ["2023-06", "2023-07", "2023-08", "2023-09", "2023-10", "2023-11"]


def test_rising_skill_ends_higher():
    """LLMs are set to have strong acceleration — last value > first."""
    history = _generate_historical("large language models", months=40)
    first = history[0].frequency
    last  = history[-1].frequency
    assert last > first, f"Expected rising trend: first={first:.3f} last={last:.3f}"


def test_linear_forecast_length():
    history = _generate_historical("machine learning", months=24)
    fc, slope, r2 = _linear_forecast(history, months_ahead=12)
    assert len(fc) == 12


def test_linear_forecast_months_sequence():
    history = _generate_historical("python", months=12)
    fc, _, _ = _linear_forecast(history, months_ahead=3)
    assert fc[0].month > history[-1].month


def test_linear_forecast_values_in_range():
    history = _generate_historical("kubernetes", months=30)
    fc, _, _ = _linear_forecast(history, months_ahead=12)
    for dp in fc:
        assert 0.0 <= dp.frequency <= 1.0


def test_linear_forecast_r_squared_range():
    history = _generate_historical("devops", months=30)
    _, _, r2 = _linear_forecast(history, months_ahead=12)
    assert 0.0 <= r2 <= 1.0


def test_linear_forecast_slope_sign_matches_trend():
    """Machine learning has positive slope — slope should be positive."""
    history = _generate_historical("machine learning", months=40)
    _, slope, _ = _linear_forecast(history, months_ahead=12)
    assert slope > 0, f"Expected positive slope for 'machine learning', got {slope}"


def test_linear_forecast_too_little_data():
    history = _generate_historical("python", months=2)
    fc, slope, r2 = _linear_forecast(history, months_ahead=6)
    assert fc == ()


def test_exp_smoothing_length():
    history = _generate_historical("devops", months=24)
    fc = _exp_smoothing_forecast(history, months_ahead=12)
    assert len(fc) == 12


def test_exp_smoothing_values_in_range():
    history = _generate_historical("cybersecurity", months=30)
    fc = _exp_smoothing_forecast(history, months_ahead=12)
    for dp in fc:
        assert 0.0 <= dp.frequency <= 1.0


@pytest.fixture(scope="module")
def forecaster():
    return SkillForecaster(history_months=40, forecast_months=12)


def test_forecaster_returns_result(forecaster):
    result = forecaster.forecast_skill("machine learning")
    assert result is not None
    assert isinstance(result, ForecastResult)


def test_forecaster_unknown_skill_returns_none(forecaster):
    result = forecaster.forecast_skill("not_a_real_skill_xyz")
    assert result is None


def test_forecast_result_is_frozen(forecaster):
    from dataclasses import FrozenInstanceError
    r = forecaster.forecast_skill("python")
    with pytest.raises((FrozenInstanceError, AttributeError)):
        r.skill = "modified"


def test_forecast_result_fields(forecaster):
    r = forecaster.forecast_skill("kubernetes")
    assert r.skill == "kubernetes"
    assert len(r.historical) == 40
    assert len(r.forecast) == 12
    assert r.trend in ("rising", "stable", "declining")
    assert r.method == "linear"
    assert 0.0 <= r.confidence <= 1.0


def test_llm_trend_is_rising(forecaster):
    r = forecaster.forecast_skill("large language models")
    assert r.trend == "rising", f"Expected rising, got {r.trend}"


def test_forecaster_caches_results(forecaster):
    r1 = forecaster.forecast_skill("python")
    r2 = forecaster.forecast_skill("python")
    assert r1 is r2, "Second call should return cached object"


def test_forecast_field_returns_list(forecaster):
    results = forecaster.forecast_field("Computer Science")
    assert isinstance(results, list)
    assert len(results) > 0


def test_forecast_field_top_n(forecaster):
    results = forecaster.forecast_field("Computer Science", top_n=5)
    assert len(results) <= 5


def test_forecast_field_sorted_descending(forecaster):
    results = forecaster.forecast_field("Computer Science", top_n=8)
    freqs = [r.forecast[-1].frequency for r in results]
    assert freqs == sorted(freqs, reverse=True)


def test_top_rising_are_positive_slope(forecaster):
    rising = forecaster.top_rising("Computer Science")
    for r in rising:
        assert r.slope_per_month > 0
        assert r.trend == "rising"


def test_mape_perfect_forecast(forecaster):
    mape = forecaster.mape([0.5, 0.6, 0.7], [0.5, 0.6, 0.7])
    assert mape == 0.0


def test_mape_bad_forecast(forecaster):
    mape = forecaster.mape([0.5, 0.5, 0.5], [0.1, 0.1, 0.1])
    assert mape > 0.5


def test_mape_mismatched_lengths(forecaster):
    result = forecaster.mape([0.5, 0.6], [0.5])
    assert math.isnan(result)


def test_backtest_returns_dict(forecaster):
    result = forecaster.backtest("machine learning", cutoff_months_ago=6)
    assert "mape" in result
    assert "direction_accuracy" in result
    assert "skill" in result


def test_backtest_mape_is_reasonable(forecaster):
    """Linear trend on a trend-heavy skill should have MAPE < 50%."""
    result = forecaster.backtest("devops", cutoff_months_ago=6)
    assert not math.isnan(result["mape"])
    assert result["mape"] < 0.5, f"MAPE too high: {result['mape']}"


def test_backtest_direction_accuracy_above_chance(forecaster):
    """Should beat random 50% accuracy on a trending skill."""
    result = forecaster.backtest("large language models", cutoff_months_ago=6)
    assert result["direction_accuracy"] > 0.4


def test_different_fields_give_different_skills(forecaster):
    cs_skills  = {r.skill for r in forecaster.forecast_field("Computer Science")}
    mec_skills = {r.skill for r in forecaster.forecast_field("Mechanical Engineering")}
    assert cs_skills != mec_skills
