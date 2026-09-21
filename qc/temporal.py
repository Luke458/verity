"""Temporal intelligence: latest-week forecasting, robust statistics and
forecast calibration (Milestone C).

The engine consumes forecast quantiles, residual percentile and independent
statistical detectors as evidence. Chronos is an optional adapter; the default
seasonal-difference baseline keeps tests and local runs dependency-free and
doubles as the champion/challenger reference. Calibration is fitted on rolling
origins from the previous version only, so the target week is never used to
score itself.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any, Protocol

import numpy as np
import pandas as pd

from .calendar import build_calendar, calendar_identity
from .config import DatasetConfig
from .conformal import minimum_samples
from .hierarchy import national_share
from .lifecycle import NEW_BACKFILL, NEW_RECENT
from .versions import VersionPair

_NORMAL = NormalDist()


# ---------------------------------------------------------------------------
# Forecasters
# ---------------------------------------------------------------------------


class Forecaster(Protocol):
    @property
    def name(self) -> str: ...

    def predict(
        self,
        history: Sequence[float],
        horizon: int,
        quantiles: Sequence[float],
    ) -> np.ndarray: ...


def _finite_values(values: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def _difference_scale(values: np.ndarray, season: int) -> float:
    finite = _finite_values(values)
    if len(finite) > season:
        diffs = finite[season:] - finite[:-season]
    else:
        diffs = np.diff(finite) if len(finite) > 1 else np.array([0.0])
    if len(diffs) == 0:
        return 1e-9
    median = float(np.median(diffs))
    scale = 1.4826 * float(np.median(np.abs(diffs - median)))
    level = max(abs(float(np.median(finite))) if len(finite) else 0.0, 1.0)
    # A zero-MAD difference sequence (e.g. deterministic cycles) must not
    # produce unbounded z-scores; floor the scale relative to the series level.
    return max(scale, 1e-4 * level, 1e-9)


@dataclass(frozen=True)
class BaselineForecaster:
    """Seasonal-naive median with residual-scale normal quantiles."""

    season: int = 52
    name: str = "seasonal_baseline"

    def predict(
        self,
        history: Sequence[float],
        horizon: int,
        quantiles: Sequence[float],
    ) -> np.ndarray:
        values = np.asarray(history, dtype=float)
        levels = np.asarray(quantiles, dtype=float)
        predictions = np.zeros((horizon, len(levels)))
        scale = _difference_scale(values, self.season)
        z = np.array([_NORMAL.inv_cdf(float(level)) for level in levels])
        finite = _finite_values(values)
        fallback = float(finite[-1]) if len(finite) else 0.0
        past: list[float] = []
        for step in range(horizon):
            index = len(values) - self.season + step
            # Missing weeks stay in place: a gap is not compressed into a
            # consecutive observation, so the seasonal position is preserved.
            if 0 <= index < len(values) and np.isfinite(values[index]):
                median = float(values[index])
            elif past:
                median = past[-1]
            elif len(finite):
                median = fallback
            else:
                median = 0.0
            predictions[step] = median + scale * z
            past.append(median)
        return predictions


class ChronosForecaster:
    """Optional Chronos adapter; imports and weights load lazily."""

    name = "chronos"

    def __init__(self, model_id: str = "amazon/chronos-2", device: str = "cpu"):
        self.model_id = model_id
        self.device = device
        self._pipeline = None

    def _load(self):
        if self._pipeline is None:
            # BaseChronosPipeline dispatches to the checkpoint architecture
            # (Chronos-2, Chronos-Bolt, Chronos-T5).
            from chronos import BaseChronosPipeline  # type: ignore

            self._pipeline = BaseChronosPipeline.from_pretrained(
                self.model_id, device_map=self.device
            )
        return self._pipeline

    def predict(
        self,
        history: Sequence[float],
        horizon: int,
        quantiles: Sequence[float],
    ) -> np.ndarray:
        import torch  # type: ignore

        pipeline = self._load()
        context = torch.tensor(np.asarray(history, dtype=np.float32))
        predictions, _ = pipeline.predict_quantiles(
            context,
            prediction_length=horizon,
            quantile_levels=list(quantiles),
        )
        return np.asarray(predictions[0].cpu().numpy(), dtype=float)


def get_forecaster(config: DatasetConfig) -> Forecaster:
    if config.forecaster == "baseline":
        return BaselineForecaster(season=config.temporal_season)
    if config.forecaster == "chronos":
        return ChronosForecaster(config.chronos_model)
    if config.forecaster == "sarimax":
        return SarimaxForecaster(
            order=tuple(config.sarimax_order),
            seasonal_order=tuple(config.sarimax_seasonal_order),
        )
    raise ValueError(f"unknown forecaster: {config.forecaster!r}")


class SarimaxForecaster:
    """Optional classical challenger; imports statsmodels lazily.

    ARIMA/seasonal ARIMA with covariate support is a useful independent
    comparator for stable, low-count series where a foundation model may
    overfit. It is not a default: order selection is per series and fits are
    slow, so it belongs behind the same Forecaster protocol.
    """

    name = "sarimax"

    def __init__(
        self,
        order: tuple[int, ...] = (1, 0, 1),
        seasonal_order: tuple[int, ...] = (1, 0, 1, 52),
        trend: str = "c",
        max_history: int = 520,
    ):
        self.order = tuple(int(value) for value in order)
        self.seasonal_order = tuple(int(value) for value in seasonal_order)
        self.trend = trend
        self.max_history = max_history

    def predict(
        self,
        history: Sequence[float],
        horizon: int,
        quantiles: Sequence[float],
    ) -> np.ndarray:
        from statsmodels.tsa.statespace.sarimax import SARIMAX  # type: ignore

        values = np.asarray(history, dtype=float)
        if len(values) > self.max_history:
            values = values[-self.max_history :]
        model = SARIMAX(
            values,
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend=self.trend,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted = model.fit(disp=False)
        forecast = fitted.get_forecast(steps=horizon)
        mean = np.asarray(forecast.predicted_mean, dtype=float)
        standard_error = np.sqrt(
            np.asarray(forecast.var_pred_mean, dtype=float)
        )
        z = np.array([_NORMAL.inv_cdf(float(level)) for level in quantiles])
        return mean[:, None] + standard_error[:, None] * z


# ---------------------------------------------------------------------------
# Series construction and robust statistics
# ---------------------------------------------------------------------------


def build_temporal_series(fact: pd.DataFrame, config: DatasetConfig) -> pd.DataFrame:
    """Dense, explicit series over the complete observed week span.

    Every series carries a row for every business week in the combined span;
    missing weeks are ``value = NaN`` with ``missing = True`` instead of being
    compressed into consecutive observations. The explicit ``missing`` marker
    is what lets later evidence say "this series is sparse" rather than invent
    an observation that never existed.
    """
    week = config.week_column
    metric = config.primary_metric
    if fact.empty or week not in fact.columns or metric not in fact.columns:
        return pd.DataFrame(columns=[week, "series_id", "value", "missing"])
    frames: list[pd.DataFrame] = []
    national = fact.groupby(week, dropna=False)[metric].sum().reset_index()
    national["series_id"] = "national"
    frames.append(
        national[[week, "series_id", metric]].rename(columns={metric: "value"})
    )
    for column in config.temporal_entity_columns:
        if column not in fact.columns:
            continue
        grouped = (
            fact.groupby([week, column], dropna=False)[metric].sum().reset_index()
        )
        grouped["series_id"] = column + ":" + grouped[column].astype(str)
        frames.append(
            grouped[[week, "series_id", metric]].rename(columns={metric: "value"})
        )
    series = pd.concat(frames, ignore_index=True)
    if series.empty:
        return pd.DataFrame(columns=[week, "series_id", "value", "missing"])
    series[week] = series[week].astype(int)
    weeks = range(int(series[week].min()), int(series[week].max()) + 1)
    series_ids = sorted(str(value) for value in series["series_id"].unique())
    full = pd.MultiIndex.from_product(
        [series_ids, list(weeks)], names=["series_id", week]
    )
    dense = (
        series.set_index(["series_id", week])
        .reindex(full)
        .reset_index()
    )
    dense["missing"] = dense["value"].isna()
    return dense.sort_values(["series_id", week], kind="stable").reset_index(drop=True)


def _mad(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    median = float(np.median(values))
    return float(np.median(np.abs(values - median)))


def robust_z(values: Sequence[float], target: float, window: int) -> float:
    recent = _finite_values(values[-window:])
    if len(recent) == 0:
        return 0.0
    scale = 1.4826 * _mad(recent)
    if scale <= 1e-12:
        return 0.0
    return (target - float(np.median(recent))) / scale


def seasonal_z(
    weeks: Sequence[int],
    values: Sequence[float],
    target_week: int,
    target: float,
    season: int,
) -> float:
    same = [
        float(value)
        for week, value in zip(weeks, values)
        if math.isfinite(float(value))
        and (target_week - int(week)) % season == 0
    ]
    if len(same) < 2:
        return 0.0
    array = np.asarray(same, dtype=float)
    scale = 1.4826 * _mad(array)
    if scale <= 1e-12:
        return 0.0
    return (target - float(np.median(array))) / scale


def ewma_z(values: Sequence[float], target: float, span: int) -> float:
    finite = _finite_values(values)
    series = pd.Series(finite)
    if len(series) < 2:
        return 0.0
    ewma = series.ewm(span=span, adjust=False).mean()
    errors = (series - ewma.shift(1)).dropna().to_numpy()
    scale = 1.4826 * _mad(errors)
    if scale <= 1e-12:
        return 0.0
    return (target - float(ewma.iloc[-1])) / scale


def change_point_score(values: Sequence[float], window: int) -> float:
    recent = _finite_values(values[-window:])
    if len(recent) < 8:
        return 0.0
    best = 0.0
    for split in range(3, len(recent) - 2):
        left, right = recent[:split], recent[split:]
        denominator = (
            math.sqrt(
                float(np.var(left)) / len(left) + float(np.var(right)) / len(right)
            )
            + 1e-12
        )
        statistic = abs(float(np.mean(right)) - float(np.mean(left))) / denominator
        best = max(best, statistic)
    return best


# ---------------------------------------------------------------------------
# Forecast calibration
# ---------------------------------------------------------------------------


def _forecast_scale(predictions: np.ndarray, levels: Sequence[float]) -> float:
    level = max(abs(float(np.median(predictions))), 1.0)
    for upper, lower in ((0.95, 0.5), (0.75, 0.5)):
        if upper in levels and lower in levels:
            width = float(predictions[list(levels).index(upper)]) - float(
                predictions[list(levels).index(lower)]
            )
            return max(width, 1e-4 * level, 1e-9)
    return max(float(np.std(predictions)), 1e-4 * level, 1e-9)


@dataclass
class CalibrationMap:
    residuals: list[float] = field(default_factory=list)
    coverage: dict[str, float] = field(default_factory=dict)
    n: int = 0

    def percentile(self, z: float) -> float | None:
        if not self.residuals:
            return None
        array = np.sort(np.asarray(self.residuals, dtype=float))
        # Left-continuous empirical CDF: a value below every observed residual
        # maps to 0.0 so tail thresholds remain reachable with small samples.
        return float(np.searchsorted(array, z, side="left") / len(array))

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "coverage": dict(self.coverage),
            "residuals": [round(value, 6) for value in self.residuals],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalibrationMap:
        return cls(
            residuals=[float(v) for v in data.get("residuals", [])],
            coverage={str(k): float(v) for k, v in data.get("coverage", {}).items()},
            n=int(data.get("n", 0)),
        )


def backtest_forecaster(
    series: pd.DataFrame,
    config: DatasetConfig,
    forecaster: Forecaster,
    origins: int | None = None,
) -> CalibrationMap:
    origins = origins or config.temporal_backtest_origins
    levels = list(config.forecast_quantiles)
    median_index = levels.index(0.5) if 0.5 in levels else len(levels) // 2
    residuals: list[float] = []
    hits = {str(level): 0 for level in levels}
    count = 0

    for _, group in series.groupby("series_id", sort=True):
        group = group.sort_values(config.week_column)
        values = group["value"].to_numpy(dtype=float)
        if len(values) < config.temporal_min_history + 1:
            continue
        start = max(config.temporal_min_history, len(values) - origins)
        for index in range(start, len(values)):
            if not math.isfinite(values[index]):
                continue
            history = values[:index]
            finite = history[np.isfinite(history)]
            if len(finite) < 2:
                continue
            actual = float(values[index])
            predictions = forecaster.predict(finite.tolist(), 1, levels)[0]
            median = float(predictions[median_index])
            scale = _forecast_scale(predictions, levels)
            residuals.append((actual - median) / scale)
            for level, prediction in zip(levels, predictions):
                if actual <= float(prediction):
                    hits[str(level)] += 1
            count += 1

    coverage = {
        level: (hits[level] / count if count else 0.0) for level in hits
    }
    return CalibrationMap(residuals=residuals, coverage=coverage, n=count)


# ---------------------------------------------------------------------------
# Candidate models, frozen selection and residual pools
# ---------------------------------------------------------------------------

NAIVE_KIND = "seasonal_naive"
RIDGE_KIND = "ridge"
FIXED_KIND = "fixed"
RIDGE_PENALTIES: tuple[float, ...] = (0.1, 1.0, 10.0)
FOURIER_ORDERS: tuple[int, ...] = (1, 2, 3)
MIN_SELECTION_ORIGINS = 2


@dataclass(frozen=True)
class CandidateSpec:
    """A frozen lightweight CPU candidate.

    ``seasonal_naive`` is the existing dependency-free baseline. ``ridge`` is
    trend plus annual Fourier terms plus declared-calendar event indicators,
    fitted with a standardized-feature ridge penalty. ``fixed`` preserves a
    configured external forecaster (Chronos/SARIMAX) that has no selectable
    grid and therefore cannot be model-selected.
    """

    kind: str
    fourier_order: int = 0
    penalty: float = 1.0
    reduced: bool = False
    fixed: str = ""

    @property
    def revision(self) -> str:
        if self.kind == FIXED_KIND:
            return f"fixed:{self.fixed or 'configured'}:v1"
        if self.kind == NAIVE_KIND:
            return "seasonal_naive:v1"
        if self.fourier_order == 0:
            return f"ridge:trend_only:p{self.penalty:g}"
        return f"ridge:fourier{self.fourier_order}:p{self.penalty:g}"

    @property
    def annual(self) -> bool:
        return self.kind == RIDGE_KIND and self.fourier_order > 0

    @property
    def complexity(self) -> tuple[int, int, float]:
        rank = {NAIVE_KIND: 0, FIXED_KIND: 0, RIDGE_KIND: 1}[self.kind]
        return (rank, self.fourier_order, -self.penalty)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "fourier_order": self.fourier_order,
            "penalty": self.penalty,
            "reduced": self.reduced,
            "revision": self.revision,
        }


def candidate_grid(config: DatasetConfig, observed: int) -> list[CandidateSpec]:
    """Fixed candidate grid; annual candidates require two retail years."""
    specs = [CandidateSpec(NAIVE_KIND)]
    annual = observed >= config.temporal_min_annual_history
    specs.append(CandidateSpec(RIDGE_KIND, 0, 1.0, reduced=not annual))
    if annual:
        for order in FOURIER_ORDERS:
            for penalty in RIDGE_PENALTIES:
                specs.append(CandidateSpec(RIDGE_KIND, order, penalty))
    return specs


def _phase(week: int, calendar) -> float:
    if calendar is None:
        return ((int(week) - 1) % 52) / 52.0
    year, retail_week = calendar.retail_year_week(int(week))
    return (retail_week - 1) / calendar.weeks_in_retail_year(year)


def _design_matrix(
    weeks: Sequence[int],
    spec: CandidateSpec,
    calendar,
    event_windows: dict[str, set[int]],
) -> tuple[np.ndarray, list[str]]:
    columns: list[np.ndarray] = [np.asarray(weeks, dtype=float)]
    names = ["trend"]
    if spec.kind == RIDGE_KIND and spec.fourier_order:
        phase = np.asarray([_phase(week, calendar) for week in weeks])
        for order in range(1, spec.fourier_order + 1):
            columns.append(np.sin(2.0 * np.pi * order * phase))
            names.append(f"sin{order}")
            columns.append(np.cos(2.0 * np.pi * order * phase))
            names.append(f"cos{order}")
        if calendar is not None:
            for name in sorted(event_windows):
                columns.append(
                    np.asarray(
                        [
                            1.0 if int(week) in event_windows[name] else 0.0
                            for week in weeks
                        ]
                    )
                )
                names.append(f"event:{name}")
    return np.column_stack(columns), names


def _fit_ridge(
    matrix: np.ndarray, targets: np.ndarray, penalty: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
    finite = np.isfinite(targets)
    if int(finite.sum()) < 2:
        return None
    observations = matrix[finite]
    values = targets[finite]
    means = observations.mean(axis=0)
    scales = observations.std(axis=0)
    scales[scales <= 1e-12] = 1.0
    standardized = (observations - means) / scales
    centre = float(values.mean())
    centred = values - centre
    normal = standardized.T @ standardized + penalty * np.eye(standardized.shape[1])
    try:
        weights = np.linalg.solve(normal, standardized.T @ centred)
    except np.linalg.LinAlgError:
        weights, *_ = np.linalg.lstsq(normal, standardized.T @ centred, rcond=None)
    return means, scales, weights, centre


@dataclass
class _NaiveModel:
    by_week: dict[int, float]
    last: float | None
    season: int

    def predict(self, future_weeks: Sequence[int]) -> np.ndarray:
        output = np.empty(len(future_weeks), dtype=float)
        for index, week in enumerate(future_weeks):
            value = None
            for lag in (1, 2, 3):
                candidate = int(week) - self.season * lag
                if candidate in self.by_week:
                    value = self.by_week[candidate]
                    break
            output[index] = value if value is not None else (self.last or 0.0)
        return output


@dataclass
class _RidgeModel:
    spec: CandidateSpec
    coefficients: tuple[np.ndarray, np.ndarray, np.ndarray, float]
    calendar: Any
    event_windows: dict[str, set[int]]

    def predict(self, future_weeks: Sequence[int]) -> np.ndarray:
        means, scales, weights, centre = self.coefficients
        matrix, _ = _design_matrix(
            future_weeks, self.spec, self.calendar, self.event_windows
        )
        standardized = (matrix - means) / scales
        return standardized @ weights + centre


@dataclass
class _FixedModel:
    forecaster: Forecaster
    values: np.ndarray
    levels: tuple[float, ...]
    median_index: int
    last_week: int

    def predict(self, future_weeks: Sequence[int]) -> np.ndarray:
        horizons = [max(1, int(week) - self.last_week) for week in future_weeks]
        predictions = self.forecaster.predict(self.values.tolist(), max(horizons), self.levels)
        return np.asarray(
            [float(predictions[horizon - 1][self.median_index]) for horizon in horizons]
        )


def fit_model(
    spec: CandidateSpec,
    weeks: Sequence[int],
    values: Sequence[float],
    future_weeks: Sequence[int],
    calendar,
    config: DatasetConfig,
    forecaster: Forecaster | None = None,
):
    pairs = [
        (int(week), float(value))
        for week, value in zip(weeks, values)
        if math.isfinite(float(value))
    ]
    observed_weeks = [week for week, _ in pairs]
    observed_values = np.asarray([value for _, value in pairs], dtype=float)
    if spec.kind == FIXED_KIND:
        if forecaster is None:
            raise ValueError("fixed candidate requires a configured forecaster")
        levels = tuple(config.forecast_quantiles)
        median_index = (
            levels.index(0.5) if 0.5 in levels else len(levels) // 2
        )
        return _FixedModel(
            forecaster,
            observed_values,
            levels,
            median_index,
            observed_weeks[-1] if observed_weeks else 0,
        )
    if spec.kind == NAIVE_KIND or not observed_weeks:
        return _NaiveModel(
            {week: value for week, value in pairs},
            float(observed_values[-1]) if len(observed_values) else None,
            config.temporal_season,
        )
    windows = (
        calendar.event_weeks_for_range(observed_weeks + list(future_weeks))
        if calendar is not None
        else {}
    )
    matrix, _ = _design_matrix(observed_weeks, spec, calendar, windows)
    fitted = _fit_ridge(matrix, observed_values, spec.penalty)
    if fitted is None:
        return _NaiveModel(
            {week: value for week, value in pairs},
            float(observed_values[-1]) if len(observed_values) else None,
            config.temporal_season,
        )
    return _RidgeModel(spec, fitted, calendar, windows)


def rolling_origin_errors(
    weeks: Sequence[int],
    values: Sequence[float],
    spec: CandidateSpec,
    horizon: int,
    calendar,
    config: DatasetConfig,
    origins: int | None = None,
    forecaster: Forecaster | None = None,
) -> list[tuple[float, float]]:
    """One (standardized, raw) error per rolling origin.

    Every forecast precedes the observation it is evaluated against. Errors are
    standardized by a scale estimated only from that origin's history so that
    series with different volumes can share a pool without one dominating it.
    """
    pairs = [
        (int(week), float(value))
        for week, value in zip(weeks, values)
        if math.isfinite(float(value))
    ]
    origins = origins or config.temporal_backtest_origins
    if len(pairs) <= horizon:
        return []
    levels = list(config.forecast_quantiles)
    median_index = levels.index(0.5) if 0.5 in levels else len(levels) // 2
    start = max(0, len(pairs) - origins)
    errors: list[tuple[float, float]] = []
    for index in range(start, len(pairs)):
        origin = index - horizon
        if origin < 2:
            continue
        training = pairs[:origin]
        if len(training) < config.temporal_min_history:
            continue
        target_week, actual = pairs[index]
        train_weeks = [week for week, _ in training]
        train_values = [value for _, value in training]
        if spec.kind == FIXED_KIND:
            if forecaster is None:
                raise ValueError("fixed candidate requires a configured forecaster")
            predictions = forecaster.predict(train_values, horizon, levels)[-1]
            predicted = float(predictions[median_index])
            scale = _forecast_scale(predictions, levels)
        else:
            model = fit_model(
                spec, train_weeks, train_values, [target_week], calendar, config
            )
            predicted = float(model.predict([target_week])[0])
            scale = _difference_scale(
                np.asarray(train_values, dtype=float), config.temporal_season
            )
        raw = float(actual) - predicted
        errors.append((raw / scale, raw))
    return errors


def _residual_scale(residuals: Sequence[float], level: float) -> float:
    finite = _finite_values(residuals)
    if len(finite) == 0:
        return max(1e-4 * abs(level), 1e-9)
    median = float(np.median(finite))
    scale = 1.4826 * float(np.median(np.abs(finite - median)))
    return max(scale, 1e-4 * abs(level), 1e-9)


def _interval_score(errors: Sequence[float], alpha: float) -> float | None:
    finite = _finite_values(errors)
    if len(finite) == 0:
        return None
    lower = float(np.quantile(finite, alpha / 2.0))
    upper = float(np.quantile(finite, 1.0 - alpha / 2.0))
    penalty = np.maximum(lower - finite, 0.0) + np.maximum(finite - upper, 0.0)
    return float(np.mean((upper - lower) + (2.0 / alpha) * penalty))


@dataclass
class CandidateEvaluation:
    spec: CandidateSpec
    eligible: bool
    reason: str
    origins: int
    mae: float | None
    interval_score: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "eligible": self.eligible,
            "reason": self.reason,
            "origins": self.origins,
            "mae": self.mae,
            "interval_score": self.interval_score,
        }


@dataclass
class SelectionOutcome:
    spec: CandidateSpec
    mode: str
    frozen: bool
    horizon: int
    history_observed: int
    history_weeks: int
    evaluations: list[CandidateEvaluation] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "frozen": self.frozen,
            "horizon": self.horizon,
            "history_observed": self.history_observed,
            "history_weeks": self.history_weeks,
            "selected": self.spec.to_dict(),
            "evaluations": [item.to_dict() for item in self.evaluations],
        }

    def compact_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "frozen": self.frozen,
            "horizon": self.horizon,
            "history_observed": self.history_observed,
            "history_weeks": self.history_weeks,
            "selected": self.spec.revision,
            "candidates": [
                {
                    "revision": item.spec.revision,
                    "eligible": item.eligible,
                    "reason": item.reason,
                    "origins": item.origins,
                    "mae": item.mae,
                    "interval_score": item.interval_score,
                }
                for item in self.evaluations
            ],
        }


def select_candidate(
    weeks: Sequence[int],
    values: Sequence[float],
    horizon: int,
    calendar,
    config: DatasetConfig,
    forecaster: Forecaster | None = None,
) -> SelectionOutcome:
    """Select a candidate on pre-target rolling-origin error.

    The selection is frozen before interval calibration: the returned spec is
    the only model fitted on the full pre-target history and used to score the
    assessment target. Ties break on interval score, then on the simpler model.
    """
    observed = [
        (int(week), float(value))
        for week, value in zip(weeks, values)
        if math.isfinite(float(value))
    ]
    history_weeks = (
        int(observed[-1][0]) - int(observed[0][0]) + 1 if observed else 0
    )
    if config.forecaster != "auto":
        spec = CandidateSpec(FIXED_KIND, fixed=config.forecaster)
        return SelectionOutcome(
            spec=spec,
            mode="fixed",
            frozen=True,
            horizon=horizon,
            history_observed=len(observed),
            history_weeks=history_weeks,
        )
    evaluations: list[CandidateEvaluation] = []
    for spec in candidate_grid(config, len(observed)):
        errors = rolling_origin_errors(
            weeks, values, spec, horizon, calendar, config, forecaster=forecaster
        )
        if len(errors) < MIN_SELECTION_ORIGINS:
            evaluations.append(
                CandidateEvaluation(
                    spec,
                    False,
                    f"{len(errors)} rolling origins; need {MIN_SELECTION_ORIGINS}",
                    len(errors),
                    None,
                    None,
                )
            )
            continue
        raw = [raw for _, raw in errors]
        mae = float(np.mean(np.abs(np.asarray(raw, dtype=float))))
        score = _interval_score(raw, config.temporal_interval_alpha)
        evaluations.append(
            CandidateEvaluation(spec, True, "", len(errors), mae, score)
        )
    eligible = [item for item in evaluations if item.eligible]
    if not eligible:
        fallback = CandidateSpec(NAIVE_KIND)
        return SelectionOutcome(
            spec=fallback,
            mode="auto",
            frozen=True,
            horizon=horizon,
            history_observed=len(observed),
            history_weeks=history_weeks,
            evaluations=evaluations,
        )
    eligible.sort(
        key=lambda item: (
            item.mae if item.mae is not None else math.inf,
            item.interval_score if item.interval_score is not None else math.inf,
            item.spec.complexity,
        )
    )
    return SelectionOutcome(
        spec=eligible[0].spec,
        mode="auto",
        frozen=True,
        horizon=horizon,
        history_observed=len(observed),
        history_weeks=history_weeks,
        evaluations=evaluations,
    )


@dataclass
class ResidualPool:
    """Errors for one metric/level/revision/horizon combination.

    ``residuals`` are standardized so different-volume series can share a pool;
    ``raw_residuals`` keep the original scale for intervals and error reports.
    """

    metric: str
    level: str
    revision: str
    horizon: int
    residuals: list[float] = field(default_factory=list)
    raw_residuals: list[float] = field(default_factory=list)
    coverage: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.metric}|{self.level}|{self.revision}|h{self.horizon}"

    @property
    def n(self) -> int:
        return len(self.residuals)

    @property
    def mean_absolute_error(self) -> float | None:
        if not self.raw_residuals:
            return None
        return float(np.mean(np.abs(np.asarray(self.raw_residuals, dtype=float))))

    def percentile(self, z: float) -> float | None:
        if not self.residuals:
            return None
        array = np.sort(np.asarray(self.residuals, dtype=float))
        return float(np.searchsorted(array, z, side="left") / len(array))

    def conformal(self, center: float, alpha: float):
        from .conformal import conformal_interval, leave_one_out_coverage

        raw = self.raw_residuals or self.residuals
        interval = conformal_interval(raw, center, alpha)
        coverage = leave_one_out_coverage(raw, alpha)
        self.coverage = {
            "alpha": alpha,
            "coverage": coverage["coverage"],
            "evaluated": coverage["evaluated"],
            "n": coverage["n"],
        }
        return interval

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "metric": self.metric,
            "level": self.level,
            "revision": self.revision,
            "horizon": self.horizon,
            "n": self.n,
            "mean_absolute_error": self.mean_absolute_error,
            "coverage": dict(self.coverage),
            "residuals": [round(value, 6) for value in self.residuals],
            "raw_residuals": [round(value, 6) for value in self.raw_residuals],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResidualPool:
        return cls(
            metric=str(data.get("metric", "")),
            level=str(data.get("level", "")),
            revision=str(data.get("revision", "")),
            horizon=int(data.get("horizon", 0)),
            residuals=[float(value) for value in data.get("residuals", [])],
            raw_residuals=[float(value) for value in data.get("raw_residuals", [])],
            coverage=dict(data.get("coverage", {})),
        )


@dataclass
class CalibrationPools:
    metric: str
    horizon: int
    alpha: float
    pools: dict[str, ResidualPool] = field(default_factory=dict)

    def add(
        self, level: str, revision: str, errors: Sequence[tuple[float, float]]
    ) -> ResidualPool:
        key = f"{self.metric}|{level}|{revision}|h{self.horizon}"
        pool = self.pools.setdefault(
            key, ResidualPool(self.metric, level, revision, self.horizon)
        )
        for standardized, raw in errors:
            pool.residuals.append(float(standardized))
            pool.raw_residuals.append(float(raw))
        return pool

    def get(self, level: str, revision: str) -> ResidualPool | None:
        return self.pools.get(f"{self.metric}|{level}|{revision}|h{self.horizon}")

    def aggregate(self, revision: str | None = None) -> CalibrationMap:
        residuals: list[float] = []
        for pool in self.pools.values():
            if revision is not None and pool.revision != revision:
                continue
            residuals.extend(pool.raw_residuals)
        return CalibrationMap(residuals=residuals, coverage={}, n=len(residuals))

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "horizon": self.horizon,
            "alpha": self.alpha,
            "pools": [pool.to_dict() for pool in self.pools.values()],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalibrationPools:
        pools = cls(
            metric=str(data.get("metric", "")),
            horizon=int(data.get("horizon", 0)),
            alpha=float(data.get("alpha", 0.1)),
        )
        for item in data.get("pools", []):
            pool = ResidualPool.from_dict(item)
            pools.pools[pool.key] = pool
        return pools


def series_level(series_id: str) -> str:
    return "national" if ":" not in series_id else series_id.split(":", 1)[0]


def _two_sided_percentile(item: SeriesTemporalEvidence) -> float:
    percentile = (
        item.calibrated_percentile
        if item.calibrated_percentile is not None
        else item.nominal_percentile
    )
    return min(1.0, max(0.0, 2.0 * min(percentile, 1.0 - percentile)))


def coordinated_groups(
    evidence: Sequence[SeriesTemporalEvidence],
    config: DatasetConfig,
) -> list[dict[str, Any]]:
    """Combine same-direction leaf residuals within a level before materiality.

    Only groups where the combined movement crosses the materiality threshold
    and at least two contributors individually carry at least half of it are
    reported; groups already captured by an individual anomaly are labelled as
    such so they add no duplicate escalation.
    """
    floor = config.temporal_min_relative_residual / 2.0
    national = next(
        (item for item in evidence if item.series_id == "national"), None
    )
    parent_value = abs(national.actual) if national is not None else 0.0
    groups: dict[tuple[str, int], list[SeriesTemporalEvidence]] = {}
    for item in evidence:
        if item.level == "national" or item.residual == 0.0:
            continue
        direction = 1 if item.residual > 0 else -1
        if abs(item.relative_residual) < floor:
            continue
        groups.setdefault((item.level, direction), []).append(item)
    coordinated: list[dict[str, Any]] = []
    for (level, direction), items in sorted(groups.items()):
        if len(items) < 2:
            continue
        combined = float(sum(item.residual for item in items))
        ratio = abs(combined) / parent_value if parent_value else 0.0
        if ratio < config.temporal_min_relative_residual:
            continue
        coordinated.append(
            {
                "level": level,
                "direction": "increase" if direction > 0 else "decrease",
                "series_ids": [item.series_id for item in items],
                "combined_residual": combined,
                "combined_ratio": ratio,
                "individually_flagged": any(item.anomaly for item in items),
            }
        )
    return coordinated


# ---------------------------------------------------------------------------
# Latest-week evidence
# ---------------------------------------------------------------------------


@dataclass
class SeriesTemporalEvidence:
    series_id: str
    target_week: int
    actual: float
    adjusted_actual: float
    adjustment: float
    forecast_median: float
    forecast_quantiles: dict[str, float]
    residual: float
    relative_residual: float
    nominal_percentile: float
    calibrated_percentile: float | None
    standardized_residual: float
    robust_z: float
    seasonal_z: float
    ewma_z: float
    change_point_score: float
    anomaly: bool
    flags: list[str]
    # Evidence-complete additions: history, calibration and interval support.
    metric: str = ""
    level: str = ""
    support: str = "model"
    parent_series: str = ""
    share: float | None = None
    selected_model: str = ""
    history_weeks: int = 0
    history_observed: int = 0
    history_missing: int = 0
    missing_weeks: tuple[int, ...] = ()
    forecast_error: float | None = None
    interval_lower: float | None = None
    interval_upper: float | None = None
    interval_alpha: float | None = None
    interval_width: float | None = None
    interval_coverage: float | None = None
    calibration_status: str = "UNAVAILABLE"
    calibration_pool: str = ""
    calibration_n: int = 0
    adjusted_p: float | None = None
    fdr_significant: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "series_id": self.series_id,
            "metric": self.metric,
            "level": self.level,
            "support": self.support,
            "parent_series": self.parent_series,
            "share": self.share,
            "target_week": self.target_week,
            "actual": self.actual,
            "adjusted_actual": self.adjusted_actual,
            "adjustment": self.adjustment,
            "forecast_median": self.forecast_median,
            "forecast_quantiles": self.forecast_quantiles,
            "residual": self.residual,
            "relative_residual": self.relative_residual,
            "nominal_percentile": self.nominal_percentile,
            "calibrated_percentile": self.calibrated_percentile,
            "standardized_residual": self.standardized_residual,
            "robust_z": self.robust_z,
            "seasonal_z": self.seasonal_z,
            "ewma_z": self.ewma_z,
            "change_point_score": self.change_point_score,
            "anomaly": self.anomaly,
            "flags": list(self.flags),
            "selected_model": self.selected_model,
            "history_weeks": self.history_weeks,
            "history_observed": self.history_observed,
            "history_missing": self.history_missing,
            "missing_weeks": list(self.missing_weeks),
            "forecast_error": self.forecast_error,
            "interval_lower": self.interval_lower,
            "interval_upper": self.interval_upper,
            "interval_alpha": self.interval_alpha,
            "interval_width": self.interval_width,
            "interval_coverage": self.interval_coverage,
            "calibration_status": self.calibration_status,
            "calibration_pool": self.calibration_pool,
            "calibration_n": self.calibration_n,
            "adjusted_p": self.adjusted_p,
            "fdr_significant": self.fdr_significant,
        }


@dataclass
class TemporalResult:
    target_week: int
    anomaly: bool
    flags: list[str]
    calibration: dict[str, Any]
    series: list[SeriesTemporalEvidence]
    unavailable_series: list[str] = field(default_factory=list)
    calendar: dict[str, Any] = field(default_factory=dict)
    coordinated: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_week": self.target_week,
            "anomaly": self.anomaly,
            "flags": list(self.flags),
            "calibration": dict(self.calibration),
            "series": [item.to_dict() for item in self.series],
            "unavailable_series": self.unavailable_series,
            "calendar": dict(self.calendar),
            "coordinated": [dict(item) for item in self.coordinated],
        }


# Flags that describe the target week itself. ``change_point`` is reported as
# series history evidence but does not by itself make the new week anomalous.
# A forecast extreme or seasonal deviation is self-sufficient; robust z and
# EWMA must corroborate each other so low-variance noise is not flagged.
STRONG_TARGET_FLAGS = frozenset({"forecast_lower", "forecast_upper", "seasonal_z"})
CONFIRMING_TARGET_FLAGS = frozenset({"robust_z", "ewma"})


def new_period_adjustments(
    current: pd.DataFrame,
    events: Sequence[Any],
    pair: VersionPair,
    config: DatasetConfig,
) -> dict[str, float]:
    """Expected new-period contributions of newly appearing entities.

    A historical backfill also adds rows for the new period; those rows explain
    part of the latest-week movement and must not be counted as anomalies.
    """
    week = config.week_column
    metric = config.primary_metric
    target = pair.current_max_week
    adjustments: dict[str, float] = {}
    if week not in current.columns or metric not in current.columns:
        return adjustments

    new_entity_events = [
        event
        for event in events
        if getattr(event, "classification", None) in (NEW_BACKFILL, NEW_RECENT)
    ]
    # Adjust at one entity level only (store before product) so child events
    # cannot subtract the same rows twice.
    entity_type = next(
        (
            candidate
            for candidate in config.explanation_entity_types
            if any(
                getattr(event, "entity_type", None) == candidate
                for event in new_entity_events
            )
        ),
        None,
    )
    if entity_type is None:
        return adjustments

    for event in new_entity_events:
        if getattr(event, "entity_type", None) != entity_type:
            continue
        entity_id = getattr(event, "entity_id", None)
        column = f"{entity_type}_id"
        rows = current.loc[
            (current[column] == entity_id) & (current[week] == target)
        ]
        if rows.empty:
            continue
        adjustments["national"] = adjustments.get("national", 0.0) + float(
            rows[metric].sum()
        )
        for series_column in config.temporal_entity_columns:
            if series_column not in rows.columns:
                continue
            for value, group in rows.groupby(series_column, dropna=False):
                series_id = f"{series_column}:{value}"
                adjustments[series_id] = adjustments.get(series_id, 0.0) + float(
                    group[metric].sum()
                )
    return adjustments


def run_temporal_qc(
    previous: pd.DataFrame,
    current: pd.DataFrame,
    pair: VersionPair,
    config: DatasetConfig,
    events: Sequence[Any] = (),
) -> TemporalResult | None:
    if not config.temporal_enabled or not pair.new_periods:
        return None
    target = pair.current_max_week
    week = config.week_column
    # Train on the previous version only; new periods come from the current
    # version, so backfilled overlap history cannot inflate the baseline.
    current_new = current.loc[current[week].isin(list(pair.new_periods))]
    combined = pd.concat([previous, current_new], ignore_index=True, sort=False)
    series = build_temporal_series(combined, config)
    if series.empty:
        return None
    calendar = build_calendar(config)
    adjustments = new_period_adjustments(current, events, pair, config)
    forecaster = None if config.forecaster == "auto" else get_forecaster(config)
    horizon = max(1, target - pair.previous_max_week)
    levels = list(config.forecast_quantiles)
    median_index = levels.index(0.5) if 0.5 in levels else len(levels) // 2
    alpha = config.temporal_interval_alpha
    pools = CalibrationPools(config.primary_metric, horizon, alpha)
    pending: list[dict[str, Any]] = []
    selections: dict[str, Any] = {}
    unavailable: list[str] = []

    for series_id, group in series.groupby("series_id", sort=True):
        group = group.sort_values(week)
        weeks = [int(value) for value in group[week]]
        values = [float(value) for value in group["value"]]
        observed = [
            (week_id, value)
            for week_id, value in zip(weeks, values)
            if math.isfinite(value)
        ]
        # Every pre-target observation, and nothing at or after the target, may
        # feed model selection, fitting and residual calibration.
        training = [
            (week_id, value)
            for week_id, value in observed
            if week_id <= pair.previous_max_week
        ]
        actuals = [
            value
            for week_id, value in zip(weeks, values)
            if week_id == target and math.isfinite(value)
        ]
        if (
            len(observed) < config.temporal_min_history
            or len(training) < config.temporal_min_history
            or not actuals
        ):
            unavailable.append(str(series_id))
            continue
        train_weeks = [week_id for week_id, _ in training]
        train_values = [value for _, value in training]
        level = series_level(str(series_id))
        selection = select_candidate(
            train_weeks, train_values, horizon, calendar, config, forecaster
        )
        model = fit_model(
            selection.spec,
            train_weeks,
            train_values,
            [target],
            calendar,
            config,
            forecaster,
        )
        point = float(model.predict([target])[0])
        quantiles: dict[str, float] = {}
        scale: float | None = None
        if selection.spec.kind == FIXED_KIND:
            if forecaster is None:
                raise ValueError("fixed candidate requires a configured forecaster")
            predictions = forecaster.predict(train_values, horizon, levels)[-1]
            point = float(predictions[median_index])
            scale = _forecast_scale(predictions, levels)
            quantiles = {
                str(level): float(prediction)
                for level, prediction in zip(levels, predictions)
            }
        residuals = rolling_origin_errors(
            train_weeks,
            train_values,
            selection.spec,
            horizon,
            calendar,
            config,
            forecaster=forecaster,
        )
        pool = pools.add(level, selection.spec.revision, residuals)
        selections[str(series_id)] = selection.compact_dict()
        history_weeks = target - min(weeks) + 1
        missing_weeks = tuple(
            int(week_id)
            for week_id, value in zip(weeks, values)
            if week_id <= target and not math.isfinite(value)
        )
        pending.append(
            {
                "series_id": str(series_id),
                "level": level,
                "actual": actuals[0],
                "adjustment": float(adjustments.get(series_id, 0.0)),
                "train_weeks": train_weeks,
                "train_values": train_values,
                "selection": selection,
                "point": point,
                "quantiles": quantiles,
                "scale": scale,
                "pool": pool,
                "history_observed": len(training),
                "history_weeks": history_weeks,
                "missing_weeks": missing_weeks,
            }
        )

    if config.temporal_sparse_fallback and pending:
        # Sparse leaves use the parent expectation scaled by their historical
        # share. The fallback is labelled and never yields calibration support.
        points = {item["series_id"]: item["point"] for item in pending}
        for series_id in list(unavailable):
            sid = str(series_id)
            level = series_level(sid)
            if level == "national" or ":" not in sid or "national" not in points:
                continue
            child_value = sid.split(":", 1)[1]
            share = national_share(
                combined, level, child_value, config, through_week=pair.previous_max_week
            )
            if share is None or share <= 0.0:
                continue
            group = series.loc[series["series_id"] == sid].sort_values(week)
            weeks = [int(entry) for entry in group[week]]
            values = [float(entry) for entry in group["value"]]
            observed = [
                (week_id, value_id)
                for week_id, value_id in zip(weeks, values)
                if math.isfinite(value_id)
            ]
            training = [
                (week_id, value_id)
                for week_id, value_id in observed
                if week_id <= pair.previous_max_week
            ]
            actuals = [
                value_id
                for week_id, value_id in zip(weeks, values)
                if week_id == target and math.isfinite(value_id)
            ]
            if len(observed) < 4 or len(training) < 4 or not actuals:
                continue
            unavailable.remove(series_id)
            pending.append(
                {
                    "series_id": sid,
                    "level": level,
                    "actual": actuals[0],
                    "adjustment": float(adjustments.get(series_id, 0.0)),
                    "train_weeks": [week_id for week_id, _ in training],
                    "train_values": [value_id for _, value_id in training],
                    "selection": None,
                    "point": float(points["national"]) * float(share),
                    "quantiles": {},
                    "scale": None,
                    "pool": None,
                    "fallback": True,
                    "parent_series": "national",
                    "share": float(share),
                    "history_observed": len(training),
                    "history_weeks": target - min(weeks) + 1,
                    "missing_weeks": tuple(
                        int(week_id)
                        for week_id, value_id in zip(weeks, values)
                        if week_id <= target and not math.isfinite(value_id)
                    ),
                }
            )

    evidence: list[SeriesTemporalEvidence] = []
    for item in pending:
        adjusted_actual = item["actual"] - item["adjustment"]
        point = item["point"]
        if item.get("fallback"):
            scale = _difference_scale(
                np.asarray(item["train_values"], dtype=float),
                config.temporal_season,
            )
            calibrated = None
            interval_lower = interval_upper = interval_width = None
            interval_coverage = None
            calibration_status = "PARENT_SHARE_FALLBACK"
            calibration_pool = ""
            calibration_n = 0
            forecast_error = None
            selected_model = "parent_share:v1"
            support = "parent_share"
        else:
            pool = item["pool"]
            selection = item["selection"]
            if selection.spec.kind != FIXED_KIND:
                if pool.residuals:
                    item["quantiles"] = {
                        str(level): point + float(np.quantile(pool.residuals, level))
                        for level in levels
                    }
                if not item["quantiles"]:
                    item["quantiles"] = {
                        str(0.5): point,
                    }
                scale = _difference_scale(
                    np.asarray(item["train_values"], dtype=float),
                    config.temporal_season,
                )
            else:
                scale = item["scale"] or _residual_scale(pool.raw_residuals, point)
            # An empirical percentile at a 1% tail needs enough held-out errors
            # to be meaningful; with too few samples every extreme z maps to 0/1
            # and would flag by construction. Insufficient calibration falls
            # back to the nominal percentile and is reported as such.
            tail_alpha = min(
                config.temporal_lower_percentile,
                1.0 - config.temporal_upper_percentile,
            )
            z = (adjusted_actual - point) / scale
            calibrated = (
                pool.percentile(z)
                if pool.n >= minimum_samples(tail_alpha)
                else None
            )
            interval = pool.conformal(point, alpha)
            interval_lower = interval.lower
            interval_upper = interval.upper
            interval_width = interval.width
            interval_coverage = pool.coverage.get("coverage")
            calibration_status = (
                "OK" if interval.status == "OK" else "INSUFFICIENT_CALIBRATION"
            )
            calibration_pool = pool.key
            calibration_n = pool.n
            forecast_error = pool.mean_absolute_error
            selected_model = selection.spec.revision
            support = "model"
        z = (adjusted_actual - point) / scale
        nominal = _NORMAL.cdf(z)

        rz = robust_z(
            item["train_values"], adjusted_actual, config.temporal_window
        )
        sz = seasonal_z(
            item["train_weeks"],
            item["train_values"],
            target,
            adjusted_actual,
            config.temporal_season,
        )
        ez = ewma_z(
            item["train_values"], adjusted_actual, config.temporal_window
        )
        cp = change_point_score(
            item["train_values"], config.temporal_window * 2
        )

        flags: list[str] = []
        percentile = calibrated if calibrated is not None else nominal
        # Statistical significance is not enough: a target week must also move
        # materially versus the forecast median before it can raise an anomaly.
        relative_residual = (
            (adjusted_actual - point) / point if point else 0.0
        )
        material = (
            point == 0.0
            or abs(relative_residual) >= config.temporal_min_relative_residual
        )
        if material:
            if percentile <= config.temporal_lower_percentile:
                flags.append("forecast_lower")
            if percentile >= config.temporal_upper_percentile:
                flags.append("forecast_upper")
            if abs(rz) >= config.temporal_z_threshold:
                flags.append("robust_z")
            if abs(sz) >= config.temporal_z_threshold:
                flags.append("seasonal_z")
            if abs(ez) >= config.temporal_z_threshold:
                flags.append("ewma")
        if cp >= config.temporal_change_point_threshold:
            flags.append("change_point")

        flag_set = set(flags)
        anomaly = bool(flag_set & STRONG_TARGET_FLAGS) or (
            len(flag_set & CONFIRMING_TARGET_FLAGS) >= 2
        )

        evidence.append(
            SeriesTemporalEvidence(
                series_id=item["series_id"],
                target_week=target,
                actual=item["actual"],
                adjusted_actual=adjusted_actual,
                adjustment=item["adjustment"],
                forecast_median=point,
                forecast_quantiles=item["quantiles"],
                residual=adjusted_actual - point,
                relative_residual=relative_residual,
                nominal_percentile=nominal,
                calibrated_percentile=calibrated,
                standardized_residual=z,
                robust_z=rz,
                seasonal_z=sz,
                ewma_z=ez,
                change_point_score=cp,
                anomaly=anomaly,
                flags=flags,
                metric=config.primary_metric,
                level=item["level"],
                support=support,
                parent_series=item.get("parent_series", ""),
                share=item.get("share"),
                selected_model=selected_model,
                history_weeks=item["history_weeks"],
                history_observed=item["history_observed"],
                history_missing=len(item["missing_weeks"]),
                missing_weeks=item["missing_weeks"],
                forecast_error=forecast_error,
                interval_lower=interval_lower,
                interval_upper=interval_upper,
                interval_alpha=alpha,
                interval_width=interval_width,
                interval_coverage=interval_coverage,
                calibration_status=calibration_status,
                calibration_pool=calibration_pool,
                calibration_n=calibration_n,
            )
        )
    if config.temporal_fdr_enabled:
        from .conformal import benjamini_hochberg

        leaves = [item for item in evidence if item.level != "national"]
        p_values = [_two_sided_percentile(item) for item in leaves]
        adjusted, significant = benjamini_hochberg(p_values, config.temporal_fdr_q)
        for leaf, adjusted_p, is_significant in zip(leaves, adjusted, significant):
            leaf.adjusted_p = float(adjusted_p)
            leaf.fdr_significant = bool(is_significant)
            if is_significant:
                continue
            if not {"forecast_lower", "forecast_upper"} & set(leaf.flags):
                continue
            leaf.flags = [
                flag
                for flag in leaf.flags
                if flag not in ("forecast_lower", "forecast_upper")
            ]
            flag_set = set(leaf.flags)
            leaf.anomaly = bool(flag_set & STRONG_TARGET_FLAGS) or (
                len(flag_set & CONFIRMING_TARGET_FLAGS) >= 2
            )
    coordinated = coordinated_groups(evidence, config)
    anomalies = [item for item in evidence if item.anomaly]
    overall_flags = sorted({flag for item in anomalies for flag in item.flags})
    coverage: dict[str, float] = {}
    for pool in pools.pools.values():
        coverage_value = pool.coverage.get("coverage")
        if coverage_value is not None:
            coverage.setdefault(
                str(round(1.0 - alpha, 4)), float(coverage_value)
            )
    calibration = {
        "n": sum(pool.n for pool in pools.pools.values()),
        "coverage": coverage,
        "horizon": horizon,
        "alpha": alpha,
        "calendar": calendar_identity(config),
        "models": sorted({pool.revision for pool in pools.pools.values()}),
        "pools": [pool.to_dict() for pool in pools.pools.values()],
        "selection": selections,
        "parent_share": [
            {
                "series_id": item["series_id"],
                "parent_series": item["parent_series"],
                "share": item["share"],
                "basis": "historical_share",
            }
            for item in pending
            if item.get("fallback")
        ],
    }
    coordinated_material = [
        group for group in coordinated if not group["individually_flagged"]
    ]
    return TemporalResult(
        target_week=target,
        anomaly=bool(anomalies) or bool(coordinated_material),
        flags=overall_flags,
        calibration=calibration,
        series=evidence,
        unavailable_series=unavailable,
        calendar=calendar_identity(config),
        coordinated=coordinated,
    )
