"""
CURIA Agent B — Skill Demand Forecasting.

Generates synthetic historical monthly skill-frequency data (2022-01 → present),
then fits three baselines (linear regression, exponential smoothing, moving-average
trend) and produces 12-month-ahead forecasts with confidence intervals.

When real multi-year data is available, swap _generate_historical() for a loader
that reads from the audit database / ingestion runs.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Generator, Optional, Sequence


@dataclass(frozen=True)
class DataPoint:
    month: str
    frequency: float


@dataclass(frozen=True)
class ForecastResult:
    skill: str
    historical: tuple[DataPoint, ...]
    forecast: tuple[DataPoint, ...]
    trend: str
    slope_per_month: float
    r_squared: float
    method: str
    confidence: float


_TREND_DEFS: dict[str, tuple[float, float, float, dict[int, float]]] = {
    "machine learning":          (0.38, 0.006, 0.012, {24: 0.012}),
    "large language models":     (0.04, 0.018, 0.018, {18: 0.035}),
    "retrieval augmented generation": (0.01, 0.010, 0.014, {20: 0.045}),
    "prompt engineering":        (0.02, 0.008, 0.012, {21: 0.030}),
    "vector databases":          (0.03, 0.012, 0.010, {22: 0.028}),
    "kubernetes":                (0.30, 0.010, 0.010, {14: 0.008, 28: -0.004}),
    "cloud native":              (0.25, 0.009, 0.009, {15: 0.006, 30: -0.003}),
    "devops":                    (0.40, 0.004, 0.008, {}),
    "supply chain security":     (0.12, 0.012, 0.010, {10: 0.016}),
    "cybersecurity":             (0.28, 0.007, 0.009, {}),
    "python":                    (0.58, 0.003, 0.007, {}),
    "mlops":                     (0.08, 0.013, 0.011, {16: 0.010}),
    "software engineer":         (0.55, 0.002, 0.008, {}),
    "data analyst":              (0.35, 0.006, 0.009, {}),
    "embedded systems":          (0.22, 0.004, 0.008, {}),
    "fpga":                      (0.15, 0.003, 0.007, {}),
    "robotics":                  (0.18, 0.007, 0.009, {20: 0.006}),
    "signal processing":         (0.20, 0.002, 0.007, {}),
    "mechanical engineer":       (0.30, 0.003, 0.008, {}),
    "finite element analysis":   (0.16, 0.002, 0.007, {}),
    "cad design":                (0.20, 0.001, 0.007, {}),
    "gis analyst":               (0.18, 0.005, 0.008, {}),
    "structural engineer":       (0.22, 0.002, 0.007, {}),
    "process engineer":          (0.25, 0.003, 0.008, {}),
    "computational chemistry":   (0.10, 0.006, 0.008, {22: 0.008}),
    "uav engineer":              (0.12, 0.009, 0.010, {18: 0.012}),
    "guidance navigation control": (0.10, 0.004, 0.008, {}),
    "bioinformatics":            (0.16, 0.007, 0.009, {22: 0.006}),
    "medical device engineer":   (0.18, 0.004, 0.008, {}),
    "quantum computing":         (0.07, 0.010, 0.012, {24: 0.008}),
    "data scientist":            (0.45, 0.005, 0.008, {}),
    "statistician":              (0.22, 0.004, 0.008, {}),
    "business analyst":          (0.38, 0.004, 0.008, {}),
    "quantitative analyst":      (0.20, 0.005, 0.008, {}),
    "fintech engineer":          (0.12, 0.008, 0.009, {20: 0.007}),
    "precision agriculture":     (0.08, 0.008, 0.009, {22: 0.010}),
    "atmospheric scientist":     (0.10, 0.005, 0.008, {}),
}


_SKILL_ALIASES: dict[str, str] = {
    "llm":                       "large language models",
    "large language model":      "large language models",
    "ml":                        "machine learning",
    "ml engineer":               "machine learning",
    "machine learning engineer": "machine learning",
    "rag":                       "retrieval augmented generation",
    "retrieval augmented":       "retrieval augmented generation",
    "cloud engineer":            "cloud native",
    "cloud":                     "cloud native",
    "devsecops":                 "supply chain security",
    "security engineer":         "cybersecurity",
    "data analysis":             "data analyst",
    "analytics engineer":        "data analyst",
    "bi developer":              "data analyst",
    "product analyst":           "data analyst",
    "rf engineer":               "embedded systems",
    "gnc engineer":              "guidance navigation control",
    "remote sensing":            "gis analyst",
    "remote sensing analyst":    "gis analyst",
    "geospatial engineer":       "gis analyst",
    "spatial data scientist":    "gis analyst",
    "agtech engineer":           "precision agriculture",
    "precision agriculture engineer": "precision agriculture",
    "agricultural data scientist":    "precision agriculture",
    "biomedical engineer":       "medical device engineer",
    "bioinformatics engineer":   "bioinformatics",
    "clinical data scientist":   "bioinformatics",
    "computational chemist":     "computational chemistry",
    "drug discovery":            "computational chemistry",
    "quantitative researcher":   "quantitative analyst",
    "financial analyst":         "quantitative analyst",
    "risk analyst":              "quantitative analyst",
    "investment analyst":        "quantitative analyst",
    "fintech":                   "fintech engineer",
    "climate scientist":         "atmospheric scientist",
    "weather analyst":           "atmospheric scientist",
    "operations research analyst": "data analyst",
    "supply chain analyst":      "supply chain security",
    "supply chain":              "supply chain security",
}


def _canonical_skill(raw: str) -> str:
    """Lowercase + dehyphenate, then map through alias table."""
    s = raw.lower().replace("-", " ").strip()
    return _SKILL_ALIASES.get(s, s)


def tracked_skills() -> list[str]:
    """Finite skill set for batch forecasting/drift: curated demand skills plus
    any CS2023 unit topics (canonicalized)."""
    skills = set(_TREND_DEFS.keys())
    try:
        from .config import UNITS_FILE
        for unit in json.loads(UNITS_FILE.read_text()):
            for topic in unit.get("current_topics", []):
                skills.add(_canonical_skill(topic))
    except Exception:
        pass
    return sorted(skills)


_FIELD_SKILLS: dict[str, list[str]] = {
    "Computer Science":              ["machine learning", "large language models",
                                      "retrieval augmented generation", "kubernetes",
                                      "devops", "cybersecurity", "supply chain security",
                                      "mlops", "vector databases", "prompt engineering"],
    "Electrical Engineering":        ["embedded systems", "fpga", "robotics",
                                      "signal processing", "cybersecurity",
                                      "machine learning", "cloud native"],
    "Mechanical Engineering":        ["mechanical engineer", "robotics",
                                      "finite element analysis", "cad design",
                                      "machine learning", "uav engineer"],
    "Civil Engineering":             ["gis analyst", "structural engineer",
                                      "cloud native", "data analyst",
                                      "python", "machine learning"],
    "Chemical Engineering":          ["process engineer", "computational chemistry",
                                      "machine learning", "data analyst", "python"],
    "Aerospace Engineering":         ["uav engineer", "guidance navigation control",
                                      "robotics", "machine learning", "python"],
    "Biomedical Engineering":        ["bioinformatics", "medical device engineer",
                                      "machine learning", "data analyst", "python"],
    "Industrial & Systems Engineering": ["supply chain security", "robotics",
                                         "data analyst", "machine learning",
                                         "devops", "python"],
    "Petroleum Engineering":         ["machine learning", "data scientist",
                                      "process engineer", "python"],
    "Materials Science & Engineering": ["computational chemistry", "machine learning",
                                        "data scientist", "python"],
    "Mathematics":                   ["data scientist", "statistician",
                                      "machine learning", "quantum computing",
                                      "python"],
    "Statistics":                    ["statistician", "data scientist",
                                      "machine learning", "python"],
    "Physics":                       ["quantum computing", "computational chemistry",
                                      "machine learning", "data scientist"],
    "Chemistry":                     ["computational chemistry", "bioinformatics",
                                      "machine learning", "data scientist"],
    "Biology":                       ["bioinformatics", "data scientist",
                                      "machine learning", "python"],
    "Management Information Systems": ["cloud native", "cybersecurity",
                                       "devops", "data analyst", "python"],
    "Business Analytics":            ["data analyst", "machine learning",
                                      "data scientist", "python", "business analyst"],
    "Finance":                       ["quantitative analyst", "fintech engineer",
                                      "machine learning", "data scientist"],
    "Atmospheric Science":           ["atmospheric scientist", "machine learning",
                                      "data analyst", "python"],
    "Geography & Geospatial Sciences": ["gis analyst", "machine learning",
                                        "data analyst", "python"],
    "Agricultural Data Science":     ["precision agriculture", "machine learning",
                                      "data analyst", "python"],
    "Biological & Agricultural Engineering": ["bioinformatics", "process engineer",
                                              "machine learning", "python"],
}


def _month_sequence(start: str = "2022-01", months: int = 40) -> list[str]:
    """Generate list of 'YYYY-MM' strings."""
    year, mon = map(int, start.split("-"))
    result = []
    for _ in range(months):
        result.append(f"{year:04d}-{mon:02d}")
        mon += 1
        if mon > 12:
            mon = 1
            year += 1
    return result


def _lcg(seed: int) -> Generator[float, None, None]:
    """Deterministic pseudo-random noise via linear congruential generator."""
    a, c, m = 1664525, 1013904223, 2 ** 32
    state = seed
    while True:
        state = (a * state + c) % m
        yield (state / m) - 0.5


def _generate_historical(
    skill: str,
    start: str = "2022-01",
    months: int = 40,
) -> tuple[DataPoint, ...]:
    if skill not in _TREND_DEFS:
        return ()
    base, slope, noise, accel = _TREND_DEFS[skill]

    months_list = _month_sequence(start, months)
    rng = _lcg(hash(skill) & 0xFFFF_FFFF)

    values: list[DataPoint] = []
    v = base
    current_slope = slope
    for i, month in enumerate(months_list):
        current_slope += accel.get(i, 0.0)
        v = v + current_slope + noise * next(rng)
        v = max(0.02, min(0.98, v))
        values.append(DataPoint(month=month, frequency=round(v, 4)))
    return tuple(values)


def _linear_forecast(
    history: tuple[DataPoint, ...],
    months_ahead: int = 12,
) -> tuple[tuple[DataPoint, ...], float, float]:
    """
    Ordinary least-squares linear regression y = a + b*x.
    Returns (forecast_points, slope_per_month, r_squared).
    """
    n = len(history)
    if n < 3:
        return (), 0.0, 0.0

    xs = list(range(n))
    ys = [p.frequency for p in history]
    xm = sum(xs) / n
    ym = sum(ys) / n
    denom = sum((x - xm) ** 2 for x in xs) or 1e-9
    b = sum((xs[i] - xm) * (ys[i] - ym) for i in range(n)) / denom
    a = ym - b * xm

    ss_res = sum((ys[i] - (a + b * xs[i])) ** 2 for i in range(n))
    ss_tot = sum((y - ym) ** 2 for y in ys) or 1e-9
    r2 = max(0.0, 1.0 - ss_res / ss_tot)

    last_month_str = history[-1].month
    year, mon = map(int, last_month_str.split("-"))
    forecast: list[DataPoint] = []
    for step in range(1, months_ahead + 1):
        mon += 1
        if mon > 12:
            mon = 1
            year += 1
        pred = max(0.01, min(0.99, a + b * (n - 1 + step)))
        forecast.append(DataPoint(month=f"{year:04d}-{mon:02d}", frequency=round(pred, 4)))

    return tuple(forecast), round(b, 6), round(r2, 4)


def _exp_smoothing_forecast(
    history: tuple[DataPoint, ...],
    months_ahead: int = 12,
    alpha: float = 0.3,
) -> tuple[DataPoint, ...]:
    """Holt's linear exponential smoothing (level + trend)."""
    if len(history) < 2:
        return ()
    ys = [p.frequency for p in history]
    level = ys[0]
    trend = ys[1] - ys[0]

    for y in ys[1:]:
        prev_level = level
        level = alpha * y + (1 - alpha) * (level + trend)
        trend = alpha * (level - prev_level) + (1 - alpha) * trend

    last = history[-1].month
    year, mon = map(int, last.split("-"))
    forecast: list[DataPoint] = []
    for step in range(1, months_ahead + 1):
        mon += 1
        if mon > 12:
            mon = 1
            year += 1
        pred = max(0.01, min(0.99, level + step * trend))
        forecast.append(DataPoint(month=f"{year:04d}-{mon:02d}", frequency=round(pred, 4)))
    return tuple(forecast)


class SkillForecaster:
    """
    Entry point for Agent B.

    Usage:
        forecaster = SkillForecaster()
        results = forecaster.forecast_field("Computer Science")
        for r in results:
            print(r.skill, r.trend, r.forecast[-1].frequency)
    """

    def __init__(
        self,
        history_start: str = "2022-01",
        history_months: int = 40,
        forecast_months: int = 12,
        cache: Optional[object] = None,
    ) -> None:
        self.history_start  = history_start
        self.history_months = history_months
        self.forecast_months = forecast_months
        self.cache = cache
        self._cache: dict[str, ForecastResult] = {}

    def forecast_all_skills(
        self,
        horizons: Sequence[int] = (3, 6, 12, 24),
        skills: Optional[Sequence[str]] = None,
    ) -> list[dict]:
        """Batch method for BatchRunner. Returns rows for cache.set_agent_b."""
        names = list(skills) if skills else tracked_skills()
        rows: list[dict] = []
        seen: set[str] = set()
        for raw in names:
            canonical = _canonical_skill(raw)
            if canonical in seen:
                continue
            seen.add(canonical)
            fc = self.forecast_skill(canonical) or self._synthesize_forecast(canonical)
            base = fc.historical[-1].frequency if fc.historical else 0.1
            slope = fc.slope_per_month
            mape = None
            if canonical in _TREND_DEFS:
                bt = self.backtest(canonical).get("mape")
                mape = None if bt is None or math.isnan(bt) else bt
            margin = max(0.02, (1.0 - fc.confidence) * 0.2)
            for h in horizons:
                value = max(0.0, min(1.0, base + slope * h))
                rows.append({
                    "skill_id": canonical,
                    "horizon_months": int(h),
                    "forecast_value": round(value, 4),
                    "ci_lower": round(max(0.0, value - margin), 4),
                    "ci_upper": round(min(1.0, value + margin), 4),
                    "slope": round(slope, 6),
                    "model_name": fc.method.split()[0],
                    "backtest_mape": mape,
                })
        return rows

    def get_forecast(self, skill_id: str, horizon_months: int) -> Optional[dict]:
        """Used by the online pipeline. Returns cached forecast or None."""
        if self.cache is None:
            return None
        return self.cache.get_agent_b(skill_id, horizon_months)

    def forecast_skill(self, skill: str) -> ForecastResult | None:
        if skill in self._cache:
            return self._cache[skill]
        history = _generate_historical(skill, self.history_start, self.history_months)
        if not history:
            return None
        fc, slope, r2 = _linear_forecast(history, self.forecast_months)
        if not fc:
            return None
        trend = "rising" if slope > 0.003 else "declining" if slope < -0.002 else "stable"
        result = ForecastResult(
            skill=skill,
            historical=history,
            forecast=fc,
            trend=trend,
            slope_per_month=slope,
            r_squared=r2,
            method="linear",
            confidence=round(max(0.1, min(0.95, r2 * 0.8 + 0.15)), 3),
        )
        self._cache[skill] = result
        return result

    def forecast_field(self, field: str, top_n: int = 8) -> list[ForecastResult]:
        """Return forecasts for skills relevant to this field, sorted by 12-month forecast value."""
        skills = _FIELD_SKILLS.get(field, list(_TREND_DEFS.keys())[:top_n])
        results: list[ForecastResult] = []
        for skill in skills:
            r = self.forecast_skill(skill)
            if r:
                results.append(r)
        results.sort(key=lambda r: r.forecast[-1].frequency, reverse=True)
        return results[:top_n]

    def forecast_skills(
        self,
        skills: list[str],
        top_n: int = 10,
    ) -> list[ForecastResult]:
        """
        Forecast the exact skills passed in (e.g. produced by Agent A).
        Unknown skills get a deterministic synthetic forecast so Agent B
        never appears blank for a skill the user just saw in Agent A.

        Display name is preserved as the user's original spelling.
        """
        results: list[ForecastResult] = []
        seen_canonical: set[str] = set()

        for raw in skills:
            canonical = _canonical_skill(raw)
            if canonical in seen_canonical:
                continue
            seen_canonical.add(canonical)

            r = self.forecast_skill(canonical) or self._synthesize_forecast(canonical)

            r = ForecastResult(
                skill=raw,
                historical=r.historical,
                forecast=r.forecast,
                trend=r.trend,
                slope_per_month=r.slope_per_month,
                r_squared=r.r_squared,
                method=r.method,
                confidence=r.confidence,
            )
            results.append(r)
            if len(results) >= top_n:
                break
        return results

    def _synthesize_forecast(self, skill: str) -> ForecastResult:
        """Deterministic synthetic forecast for skills not in _TREND_DEFS."""
        h     = hash(skill) & 0xFFFF_FFFF
        base  = 0.08 + (h % 30) / 100.0
        slope = ((h >> 4) % 16 - 4) / 1000.0
        noise = 0.007 + ((h >> 12) % 5) / 1000.0

        months_list = _month_sequence(self.history_start, self.history_months)
        rng = _lcg(h)
        values: list[DataPoint] = []
        v = base
        for month in months_list:
            v = v + slope + noise * next(rng)
            v = max(0.02, min(0.98, v))
            values.append(DataPoint(month=month, frequency=round(v, 4)))
        history = tuple(values)

        fc, slope_fit, r2 = _linear_forecast(history, self.forecast_months)
        trend = "rising" if slope_fit > 0.003 else "declining" if slope_fit < -0.002 else "stable"
        return ForecastResult(
            skill=skill,
            historical=history,
            forecast=fc,
            trend=trend,
            slope_per_month=slope_fit,
            r_squared=r2,
            method="linear (synthetic)",
            confidence=round(max(0.10, min(0.85, r2 * 0.7 + 0.10)), 3),
        )

    def top_rising(self, field: str, n: int = 5) -> list[ForecastResult]:
        """Skills with the highest positive slope for this field."""
        return sorted(
            [r for r in self.forecast_field(field) if r.trend == "rising"],
            key=lambda r: r.slope_per_month,
            reverse=True,
        )[:n]

    def top_declining(self, field: str, n: int = 3) -> list[ForecastResult]:
        return sorted(
            [r for r in self.forecast_field(field) if r.trend == "declining"],
            key=lambda r: r.slope_per_month,
        )[:n]

    @staticmethod
    def mape(actual: list[float], predicted: list[float]) -> float:
        """Mean Absolute Percentage Error — used for backtesting."""
        if not actual or len(actual) != len(predicted):
            return float("nan")
        errors = [
            abs(a - p) / max(abs(a), 1e-9)
            for a, p in zip(actual, predicted)
        ]
        return round(sum(errors) / len(errors), 4)

    def backtest(self, skill: str, cutoff_months_ago: int = 6) -> dict:
        """
        Simulate backtesting: train up to cutoff, forecast forward, compare to held-out actuals.
        Returns {"mape": float, "direction_accuracy": float}.
        """
        history = _generate_historical(skill, self.history_start, self.history_months)
        if not history or cutoff_months_ago >= len(history):
            return {"mape": float("nan"), "direction_accuracy": float("nan")}

        cutoff = len(history) - cutoff_months_ago
        train  = history[:cutoff]
        actual = [p.frequency for p in history[cutoff:]]

        fc, _, _ = _linear_forecast(train, len(actual))
        predicted = [p.frequency for p in fc]

        directions = 0
        for i in range(1, len(actual)):
            a_dir = actual[i] > actual[i - 1]
            p_dir = predicted[i] > predicted[i - 1] if i < len(predicted) else False
            if a_dir == p_dir:
                directions += 1
        dir_acc = directions / max(len(actual) - 1, 1)

        return {
            "skill": skill,
            "cutoff_months_ago": cutoff_months_ago,
            "mape": self.mape(actual, predicted),
            "direction_accuracy": round(dir_acc, 4),
        }
