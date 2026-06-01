"""
Forecast backtest on REAL corpus-derived skill series (RQ2 ground truth).

Retrospective hold-out: for each tracked skill, train on the earlier months,
forecast the held-out final H months, and compare to the realized actuals.
Baselines: naive (last value), moving average, and the repo's linear forecaster
run on the SAME real series. Metrics: MAPE, sMAPE, MASE, directional accuracy.

DATA NOTE: results are only statistically meaningful once the dated corpus is
scaled (the proposal's large-scale ingest). On a small/sparse corpus this is a
SMOKE TEST — the report includes data-sufficiency fields so you know which.

Usage
    python3 eval/run_forecast_backtest.py
    python3 eval/run_forecast_backtest.py --horizon 6 --min-train 6 --out results/
    python3 eval/run_forecast_backtest.py --corpus-dir data/corpus_large --out results/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import CORPUS_DIR
from src.forecasting import DataPoint, _exp_smoothing_forecast, _linear_forecast, tracked_skills
from src.skill_series import monthly_skill_frequency


def _mape(actual: list[float], pred: list[float]) -> float | None:
    pairs = [(a, p) for a, p in zip(actual, pred) if a != 0]
    if not pairs:
        return None
    return round(sum(abs(a - p) / abs(a) for a, p in pairs) / len(pairs), 4)


def _smape(actual: list[float], pred: list[float]) -> float:
    vals = [2 * abs(a - p) / (abs(a) + abs(p)) for a, p in zip(actual, pred) if (abs(a) + abs(p)) > 0]
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def _mase(actual: list[float], pred: list[float], train: list[float]) -> float | None:
    if len(train) < 2:
        return None
    denom = sum(abs(train[i] - train[i - 1]) for i in range(1, len(train))) / (len(train) - 1)
    if denom == 0:
        return None
    mae = sum(abs(a - p) for a, p in zip(actual, pred)) / len(actual)
    return round(mae / denom, 4)


def _directional_accuracy(actual: list[float], pred: list[float], last_train: float) -> float | None:
    a_prev = p_prev = last_train
    hits = total = 0
    for a, p in zip(actual, pred):
        if (a - a_prev) * (p - p_prev) > 0 or (a == a_prev and p == p_prev):
            hits += 1
        total += 1
        a_prev, p_prev = a, p
    return round(hits / total, 4) if total else None


def _forecast(model: str, train: list[float], h: int,
              train_months: list[str] | None = None, window: int = 3) -> list[float]:
    if not train:
        return [0.0] * h
    if model == "naive":
        return [train[-1]] * h
    if model == "seasonal_naive":
        season = 12
        if len(train) < season:
            return [train[-1]] * h
        return [train[-season + (step % season)] for step in range(h)]
    if model == "moving_avg":
        w = train[-window:]
        return [sum(w) / len(w)] * h
    if model == "linear":
        labels = train_months or [f"2000-{(i % 12) + 1:02d}" for i in range(len(train))]
        history = tuple(DataPoint(month=labels[i], frequency=v) for i, v in enumerate(train))
        fc, _slope, _r2 = _linear_forecast(history, h)
        return [max(0.0, p.frequency) for p in fc][:h] or [train[-1]] * h
    if model == "exp_smoothing":
        labels = train_months or [f"2000-{(i % 12) + 1:02d}" for i in range(len(train))]
        history = tuple(DataPoint(month=labels[i], frequency=v) for i, v in enumerate(train))
        fc = _exp_smoothing_forecast(history, h)
        return [max(0.0, p.frequency) for p in fc][:h] or [train[-1]] * h
    raise ValueError(model)


MODELS = ("naive", "seasonal_naive", "moving_avg", "linear", "exp_smoothing")


def _model_comparison(per_model: dict[str, dict]) -> dict:
    naive_smape = per_model.get("naive", {}).get("smape")
    candidates = [m for m in MODELS if per_model.get(m, {}).get("smape") is not None]
    best = min(candidates, key=lambda m: per_model[m]["smape"], default=None)
    best_non_naive = min(
        (m for m in candidates if m != "naive"),
        key=lambda m: per_model[m]["smape"],
        default=None,
    )
    comparison = {
        "best_by_smape": best,
        "best_non_naive_by_smape": best_non_naive,
        "naive_smape": naive_smape,
    }
    if naive_smape is not None and best_non_naive is not None:
        best_smape = per_model[best_non_naive]["smape"]
        comparison["best_non_naive_smape"] = best_smape
        comparison["best_non_naive_beats_naive"] = best_smape < naive_smape
        comparison["best_non_naive_smape_delta_vs_naive"] = round(best_smape - naive_smape, 4)
    return comparison


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--min-train", type=int, default=6)
    parser.add_argument("--corpus-dir", default=str(CORPUS_DIR),
                        help="Directory of dated corpus JSON files")
    parser.add_argument("--out", help="Directory to also write a JSON report into")
    args = parser.parse_args()

    skills = tracked_skills()
    corpus_dir = Path(args.corpus_dir)
    series, months, month_counts = monthly_skill_frequency(corpus_dir=corpus_dir, skills=skills)

    agg: dict[str, dict[str, list[float]]] = {m: {"mape": [], "smape": [], "mase": [], "dir_acc": []} for m in MODELS}
    evaluated = 0
    skipped_sparse = 0

    for skill in skills:
        pairs = series[skill]
        values = [v for _m, v in pairs]
        month_labels = [m for m, _v in pairs]
        nonzero = sum(1 for v in values if v > 0)
        if len(values) < args.min_train + args.horizon or nonzero < 2:
            skipped_sparse += 1
            continue
        train = values[: -args.horizon]
        train_months = month_labels[: -args.horizon]
        actual = values[-args.horizon:]
        if len(train) < args.min_train:
            skipped_sparse += 1
            continue
        evaluated += 1
        for model in MODELS:
            pred = _forecast(model, train, args.horizon, train_months=train_months)
            mape = _mape(actual, pred)
            if mape is not None:
                agg[model]["mape"].append(mape)
            agg[model]["smape"].append(_smape(actual, pred))
            mase = _mase(actual, pred, train)
            if mase is not None:
                agg[model]["mase"].append(mase)
            da = _directional_accuracy(actual, pred, train[-1])
            if da is not None:
                agg[model]["dir_acc"].append(da)

    def _mean(xs: list[float]) -> float | None:
        return round(sum(xs) / len(xs), 4) if xs else None

    per_model = {
        m: {
            "mape": _mean(agg[m]["mape"]),
            "smape": _mean(agg[m]["smape"]),
            "mase": _mean(agg[m]["mase"]),
            "directional_accuracy": _mean(agg[m]["dir_acc"]),
            "n_skills_scored": len(agg[m]["smape"]),
        }
        for m in MODELS
    }

    nonzero_months = sum(1 for v in month_counts.values() if v > 0)
    sufficient = evaluated >= 30 and nonzero_months >= 24
    comparison = _model_comparison(per_model)
    report = {
        "corpus_dir": str(corpus_dir),
        "horizon_months": args.horizon,
        "data_sufficiency": {
            "skills_evaluated": evaluated,
            "skills_skipped_sparse": skipped_sparse,
            "total_months_in_span": len(months),
            "months_with_documents": len(month_counts),
            "statistically_meaningful": sufficient,
            "note": ("OK" if sufficient else
                     "SMOKE TEST — corpus too sparse for meaningful backtest; "
                     "scale the dated ingest then re-run"),
        },
        "per_model": per_model,
        **comparison,
    }
    print(json.dumps(report, indent=2))

    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        import time
        path = out_dir / f"forecast_backtest_{time.strftime('%Y%m%d_%H%M%S')}.json"
        path.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
