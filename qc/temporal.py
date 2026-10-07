"""Latest-week temporal QC: forecasting, calibration and anomaly decisions.

Each configured measure and appended period is forecast from the previous
version only (``forecaster: auto`` selects from a fixed grid; ``baseline`` is
seasonal-naive), intervals are calibrated on reserved rolling origins, and
``decide_anomalies`` decides every series of a refresh in one BH family: the
forecast p-value for national series, the share-of-parent p-value for leaves.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any, NamedTuple, Protocol

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


def get_forecaster(config: DatasetConfig) -> Forecaster:
    if config.forecaster == "baseline":
        return BaselineForecaster(season=config.temporal_season)
    raise ValueError(f"unknown forecaster: {config.forecaster!r}")


# ---------------------------------------------------------------------------
# Series construction and robust statistics
# ---------------------------------------------------------------------------


def build_temporal_series(
    fact: pd.DataFrame, config: DatasetConfig, metric: str | None = None
) -> pd.DataFrame:
    """Dense, explicit series over the complete observed week span.

    Every series carries a row for every business week in the combined span;
    missing weeks are ``value = NaN`` with ``missing = True`` instead of being
    compressed into consecutive observations. The explicit ``missing`` marker
    is what lets later evidence say "this series is sparse" rather than invent
    an observation that never existed. One frame is built per measure, so a
    series identifier always means the same metric/scope in every layer.
    """
    week = config.week_column
    metric = metric or config.primary_metric
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
        for week, value in zip(weeks, values, strict=True)
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
            for level, prediction in zip(levels, predictions, strict=True):
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
    fitted with a standardized-feature ridge penalty. ``fixed`` wraps the
    configured forecaster (``baseline``), which has no selectable
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
        for week, value in zip(weeks, values, strict=True)
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


def _rolling_evaluations(
    pairs: Sequence[tuple[int, float]],
    horizon: int,
    config: DatasetConfig,
    origins: int | None,
) -> list[tuple[int, int]]:
    """Eligible (target index, training endpoint index) rolling origins.

    The endpoint is the last observed week at or before ``target - horizon``,
    so missing weeks never compress into a shorter forecast step and the
    evaluated step gap is exactly ``horizon`` business weeks.
    """
    origins = origins or config.temporal_backtest_origins
    eligible: list[tuple[int, int]] = []
    for index, (target_week, _) in enumerate(pairs):
        endpoint = None
        for position in range(index - 1, -1, -1):
            if pairs[position][0] <= target_week - horizon:
                endpoint = position
                break
        if endpoint is None:
            continue
        training = pairs[: endpoint + 1]
        if len(training) < config.temporal_min_history:
            continue
        eligible.append((index, endpoint))
    if origins and len(eligible) > origins:
        eligible = eligible[-origins:]
    return eligible


def rolling_origin_errors(
    weeks: Sequence[int],
    values: Sequence[float],
    spec: CandidateSpec,
    horizon: int,
    calendar,
    config: DatasetConfig,
    origins: int | None = None,
    forecaster: Forecaster | None = None,
    reserved_tail: int = 0,
    evaluation: str = "selection",
) -> list[tuple[float, float]]:
    """One (standardized, raw) error per rolling origin.

    Every forecast precedes the observation it is evaluated against by exactly
    ``horizon`` business weeks. Errors are standardized by a scale estimated
    only from that origin's history so that series with different volumes can
    share a pool without one dominating it.

    ``reserved_tail`` holds back that many latest eligible origins exclusively
    for calibration; ``evaluation`` selects which disjoint partition to return.
    Selection and calibration observations are never reused across partitions.
    """
    pairs = [
        (int(week), float(value))
        for week, value in zip(weeks, values, strict=True)
        if math.isfinite(float(value))
    ]
    eligible = _rolling_evaluations(pairs, horizon, config, origins)
    if reserved_tail > 0:
        if evaluation == "calibration":
            eligible = eligible[-reserved_tail:]
        else:
            eligible = eligible[:-reserved_tail]
    if evaluation == "calibration" and len(eligible) < reserved_tail:
        return []
    levels = list(config.forecast_quantiles)
    median_index = levels.index(0.5) if 0.5 in levels else len(levels) // 2
    errors: list[tuple[float, float]] = []
    for index, endpoint in eligible:
        training = pairs[: endpoint + 1]
        target_week, actual = pairs[index]
        train_weeks = [week for week, _ in training]
        train_values = [value for _, value in training]
        step = max(1, int(target_week) - int(training[-1][0]))
        if spec.kind == FIXED_KIND:
            if forecaster is None:
                raise ValueError("fixed candidate requires a configured forecaster")
            predictions = forecaster.predict(train_values, step, levels)[-1]
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
    calibration_origins: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "frozen": self.frozen,
            "horizon": self.horizon,
            "history_observed": self.history_observed,
            "history_weeks": self.history_weeks,
            "calibration_origins": self.calibration_origins,
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
            "calibration_origins": self.calibration_origins,
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

    The latest configured calibration origins are withheld from selection and
    reserved for interval calibration. The selection is frozen before interval
    calibration: the returned spec is the only model fitted on the full
    pre-target history and used to score the assessment target. Ties break on
    interval score, then on the simpler model. When the required partitions
    cannot be formed the outcome is explicitly insufficient rather than
    reusing selection observations as calibration.
    """
    observed = [
        (int(week), float(value))
        for week, value in zip(weeks, values, strict=True)
        if math.isfinite(float(value))
    ]
    history_weeks = (
        int(observed[-1][0]) - int(observed[0][0]) + 1 if observed else 0
    )
    reserved = max(0, int(getattr(config, "temporal_calibration_origins", 0)))
    if config.forecaster != "auto":
        spec = CandidateSpec(FIXED_KIND, fixed=config.forecaster)
        return SelectionOutcome(
            spec=spec,
            mode="fixed",
            frozen=True,
            horizon=horizon,
            history_observed=len(observed),
            history_weeks=history_weeks,
            calibration_origins=reserved,
        )
    evaluations: list[CandidateEvaluation] = []
    for spec in candidate_grid(config, len(observed)):
        errors = rolling_origin_errors(
            weeks,
            values,
            spec,
            horizon,
            calendar,
            config,
            forecaster=forecaster,
            reserved_tail=reserved,
            evaluation="selection",
        )
        if len(errors) < MIN_SELECTION_ORIGINS:
            reason = f"{len(errors)} selection origins; need {MIN_SELECTION_ORIGINS}"
            if reserved:
                reason += f" after reserving {reserved} calibration origins"
            evaluations.append(
                CandidateEvaluation(
                    spec,
                    False,
                    reason,
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
            mode="insufficient",
            frozen=True,
            horizon=horizon,
            history_observed=len(observed),
            history_weeks=history_weeks,
            evaluations=evaluations,
            calibration_origins=reserved,
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
        calibration_origins=reserved,
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

    def conformal(
        self,
        center: float,
        alpha: float,
        scale: float = 1.0,
        series_errors: Sequence[float] | None = None,
    ):
        """Interval in target units from pooled standardized residuals.

        The pooled standardized errors estimate the shape of the error
        distribution; the target's own historical scale converts the radius
        back into that target's units, so a large sibling cannot inflate a
        small series' interval. Coverage is measured on the target's own
        held-out errors, never on pooled in-sample counts.
        """
        from .conformal import ConformalInterval, conformal_interval

        standardized = [float(value) for value in (self.residuals or self.raw_residuals)]
        interval = conformal_interval(standardized, 0.0, alpha)
        if interval.status != "OK" or interval.upper is None:
            self.coverage = {
                "alpha": alpha,
                "coverage": None,
                "evaluated": False,
                "n": interval.n,
            }
            return ConformalInterval(
                alpha,
                None,
                None,
                interval.n,
                interval.rank,
                interval.status,
                interval.detail,
            )
        radius = float(interval.upper)
        coverage: float | None = None
        evaluated = False
        finite = [
            float(value)
            for value in (series_errors or ())
            if math.isfinite(float(value))
        ]
        if finite:
            coverage = sum(1 for value in finite if abs(value) <= radius) / len(finite)
            evaluated = True
        self.coverage = {
            "alpha": alpha,
            "coverage": coverage,
            "evaluated": evaluated,
            "n": interval.n,
            "series_n": len(finite),
        }
        return ConformalInterval(
            alpha,
            center - scale * radius,
            center + scale * radius,
            interval.n,
            interval.rank,
            "OK",
        )

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
        self,
        level: str,
        revision: str,
        errors: Sequence[tuple[float, float]],
        horizon: int | None = None,
    ) -> ResidualPool:
        resolved_horizon = self.horizon if horizon is None else int(horizon)
        key = f"{self.metric}|{level}|{revision}|h{resolved_horizon}"
        pool = self.pools.setdefault(
            key, ResidualPool(self.metric, level, revision, resolved_horizon)
        )
        for standardized, raw in errors:
            pool.residuals.append(float(standardized))
            pool.raw_residuals.append(float(raw))
        return pool

    def get(
        self, level: str, revision: str, horizon: int | None = None
    ) -> ResidualPool | None:
        resolved_horizon = self.horizon if horizon is None else int(horizon)
        return self.pools.get(f"{self.metric}|{level}|{revision}|h{resolved_horizon}")

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
    horizon: int = 0
    materiality: float = 0.0
    training_endpoint_week: int = 0
    forecast_origin_week: int = 0
    aggregation: str = "flow"
    # Share-of-parent shift test (leaves only): the leaf's share of its parent
    # this period against its own trailing share distribution.
    share_t: float | None = None
    share_p: float | None = None
    share_impact: float | None = None
    share_n: int = 0
    share_method: str = ""
    share_spread: float | None = None
    share_level: float | None = None
    # Smallest relative movement of this leaf's share that the refresh would
    # flag on its own, with 50% / 80% power (``decide_anomalies``).
    detectable_change: float | None = None
    detectable_change_80: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "series_id": self.series_id,
            "metric": self.metric,
            "level": self.level,
            "aggregation": self.aggregation,
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
            "horizon": self.horizon,
            "materiality": self.materiality,
            "training_endpoint_week": self.training_endpoint_week,
            "forecast_origin_week": self.forecast_origin_week,
            "share_t": self.share_t,
            "share_p": self.share_p,
            "share_impact": self.share_impact,
            "share_n": self.share_n,
            "share_method": self.share_method,
            "share_spread": self.share_spread,
            "share_level": self.share_level,
            "detectable_change": self.detectable_change,
            "detectable_change_80": self.detectable_change_80,
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
    targets: list[int] = field(default_factory=list)
    unavailable: list[dict[str, Any]] = field(default_factory=list)
    metric_status: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_week": self.target_week,
            "targets": list(self.targets),
            "anomaly": self.anomaly,
            "flags": list(self.flags),
            "calibration": dict(self.calibration),
            "series": [item.to_dict() for item in self.series],
            "unavailable_series": self.unavailable_series,
            "unavailable": [dict(item) for item in self.unavailable],
            "calendar": dict(self.calendar),
            "metric_status": dict(self.metric_status),
            "sensitivity": self.sensitivity(),
        }

    def sensitivity(self) -> dict[str, Any]:
        """Median detectable movement per level and measure (share-tested leaves)."""
        groups: dict[tuple[str, str], list[SeriesTemporalEvidence]] = {}
        for item in self.series:
            if item.detectable_change is not None:
                groups.setdefault((item.level, item.metric), []).append(item)
        out: dict[str, Any] = {}
        for (level, metric), items in sorted(groups.items()):
            changes = sorted(float(i.detectable_change) for i in items if i.detectable_change is not None)
            strong = sorted(float(i.detectable_change_80) for i in items if i.detectable_change_80 is not None)
            out[f"{level}:{metric}"] = {
                "leaves": len(items),
                "median_detectable_change": float(np.median(changes)),
                "median_detectable_change_80": float(np.median(strong)) if strong else None,
                "least_sensitive": max(items, key=lambda i: float(i.detectable_change or 0.0)).series_id,
            }
        return out


# Flags that record a decision (see ``decide_anomalies``): a significant,
# material forecast residual (national series, and leaves without a share test)
# or share-of-parent shift (leaves). Robust, seasonal and EWMA z-scores and
# change points ignore trend, seasonality and multiplicity; they are reported
# as evidence only.
DECISION_FLAGS = frozenset({"forecast_lower", "forecast_upper", "share_shift"})
# A share shift needs this many trailing aligned observations to estimate the
# leaf's own share variability.
MIN_SHARE_HISTORY = 8


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the regularized incomplete beta function."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    return h


def student_t_quantile(probability: float, df: int) -> float:
    """Upper Student-t quantile: t with P(T <= t) = ``probability`` (> 0.5)."""
    if not 0.5 < probability < 1.0:
        raise ValueError("probability must be in (0.5, 1)")
    target = 2.0 * (1.0 - probability)
    low, high = 0.0, 1.0
    while student_t_two_sided_p(high, df) > target:
        high *= 2.0
    for _ in range(80):
        mid = (low + high) / 2.0
        if student_t_two_sided_p(mid, df) > target:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def share_drop(leaf_drop: float, share: float) -> float:
    """Relative share drop when a leaf holding ``share`` of its parent drops by ``leaf_drop``.

    The parent drops too, by ``share * leaf_drop``, so the share moves less
    than the leaf: (1 - d) / (1 - d c).
    """
    return 1.0 - (1.0 - leaf_drop) / (1.0 - leaf_drop * share)


def detectable_change(
    spread: float, df: int, tests: int, q: float, power: float, method: str, share: float = 0.0
) -> float:
    """Smallest drop of the leaf's own value flagged with ``power`` as the refresh's only anomaly.

    A lone anomaly among ``tests`` hypotheses is rejected by BH at level ``q``
    when its two-sided p is at most ``q / tests``. The share shift must reach
    that critical t plus the t quantile of ``power`` (0 for 50%); the share
    drop is then converted to the leaf's own drop (others unchanged) using its
    ``share`` of the parent.
    """
    critical = student_t_quantile(1.0 - q / (2.0 * tests), df)
    shift = critical + (student_t_quantile(power, df) if power > 0.5 else 0.0)
    drop = 1.0 - math.exp(-shift * spread) if method == "seasonal" else min(1.0, shift * spread)
    share = min(max(share, 0.0), 1.0)
    return min(1.0, drop / (1.0 - share + share * drop)) if drop < 1.0 else 1.0


def student_t_two_sided_p(t: float, df: int) -> float:
    """Two-sided Student-t p-value (exact, via the incomplete beta function)."""
    if not math.isfinite(t):
        return 0.0
    if df < 1:
        raise ValueError("degrees of freedom must be positive")
    x = df / (df + t * t)
    a, b = df / 2.0, 0.5
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    ) if 0.0 < x < 1.0 else 0.0
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    if x < (a + 1.0) / (a + b + 2.0):
        value = front * _betacf(a, b, x) / a
    else:
        value = 1.0 - front * _betacf(b, a, 1.0 - x) / b
    return min(1.0, max(0.0, value))


class ShareTest(NamedTuple):
    t: float
    p: float
    impact: float
    n: int
    method: str
    # Predictive spread on a relative scale (log share for ``seasonal``,
    # spread over the trailing mean for ``trailing``): what one t unit is worth.
    spread: float = math.nan
    # The share the baseline expects this period (the leaf's weight in its parent).
    level: float = math.nan


def _trailing_baseline(values: np.ndarray) -> tuple[float, float, int] | None:
    """Mean, predictive sd and n of a trailing sample, or None if degenerate."""
    n = len(values)
    if n < MIN_SHARE_HISTORY:
        return None
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    if not math.isfinite(sd) or sd <= 1e-12 * max(abs(mean), 1e-12):
        return None
    # Prediction interval for one new observation: the trailing mean is itself
    # estimated, so the spread is sd * sqrt(1 + 1/n) with n - 1 df.
    return mean, sd * math.sqrt(1.0 + 1.0 / n), n


def share_shift_test(
    child_weeks: Sequence[int],
    child_values: Sequence[float],
    parent_weeks: Sequence[int],
    parent_values: Sequence[float],
    child_actual: float,
    parent_actual: float,
    window: int,
    target_week: int | None = None,
    season: int = 52,
) -> ShareTest | None:
    """Test the leaf's share of its parent against its own history.

    Two baselines are built from the training history only:

    - ``trailing``: the share against its trailing mean (stable shares);
    - ``seasonal``: the year-over-year log change of the share against recent
      year-over-year changes (shares that move with their own annual cycle,
      such as cough & cold in winter). Last year's share is the mean over the
      three weeks centred ``season`` weeks earlier, which halves the noise the
      single comparison week would add. Needs that share and a full window of
      year-over-year history.

    The baseline with the smaller relative predictive spread is used. A common
    (market-wide) movement leaves shares unchanged and is not a leaf anomaly.
    ``impact`` is the leaf movement attributable to the share change at the
    parent's current level. Returns ``None`` when no baseline is available.
    """
    if not (math.isfinite(parent_actual) and parent_actual > 0.0):
        return None
    if not math.isfinite(child_actual):
        return None
    parent = {
        int(week_id): float(value)
        for week_id, value in zip(parent_weeks, parent_values, strict=True)
        if math.isfinite(float(value)) and float(value) > 0.0
    }
    share_by_week = {
        int(week_id): float(value) / parent[int(week_id)]
        for week_id, value in zip(child_weeks, child_values, strict=True)
        if int(week_id) in parent and math.isfinite(float(value))
    }
    weeks = sorted(share_by_week)
    observed = child_actual / parent_actual
    size = max(int(window), MIN_SHARE_HISTORY)
    candidates: list[tuple[float, ShareTest]] = []

    trailing = _trailing_baseline(
        np.asarray([share_by_week[week] for week in weeks][-size:], dtype=float)
    )
    if trailing is not None:
        mean, spread, n = trailing
        t = (observed - mean) / spread
        candidates.append(
            (
                spread / abs(mean) if mean else math.inf,
                ShareTest(
                    t, student_t_two_sided_p(t, n - 1), (observed - mean) * parent_actual, n, "trailing",
                    spread / abs(mean) if mean else math.inf, mean,
                ),
            )
        )

    def year_ago(week: int) -> float | None:
        values = [
            share_by_week[other]
            for other in (week - season - 1, week - season, week - season + 1)
            if other in share_by_week and share_by_week[other] > 0.0
        ]
        return sum(values) / len(values) if values else None

    last_year = year_ago(int(target_week)) if target_week is not None else None
    if last_year is not None and observed > 0.0:
        pairs = [
            (share_by_week[week], year_ago(week))
            for week in weeks
            if share_by_week[week] > 0.0
        ]
        changes = np.asarray(
            [math.log(now / then) for now, then in pairs if then is not None][-size:],
            dtype=float,
        )
        seasonal = _trailing_baseline(changes)
        if seasonal is not None:
            mean, spread, n = seasonal
            t = (math.log(observed / last_year) - mean) / spread
            expected = last_year * math.exp(mean)
            # Log units are already relative; compare spreads on one scale.
            candidates.append(
                (
                    spread,
                    ShareTest(
                        t, student_t_two_sided_p(t, n - 1), (observed - expected) * parent_actual, n, "seasonal", spread,
                        expected,
                    ),
                )
            )
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])[1]


def _attach_share_tests(
    evidence: list[SeriesTemporalEvidence],
    pending: Sequence[dict[str, Any]],
    config: DatasetConfig,
) -> None:
    """Run the share-of-parent test for every leaf of one measure and period."""
    items = {item["series_id"]: item for item in pending}
    national = next((entry for entry in evidence if entry.series_id == "national"), None)
    parent_item = items.get("national")
    if national is None or parent_item is None:
        return
    window = max(config.temporal_window * 2, MIN_SHARE_HISTORY)
    for leaf in evidence:
        item = items.get(leaf.series_id)
        if leaf.level == "national" or item is None:
            continue
        tested = share_shift_test(
            item["train_weeks"],
            item["train_values"],
            parent_item["train_weeks"],
            parent_item["train_values"],
            leaf.adjusted_actual,
            national.adjusted_actual,
            window,
            target_week=int(leaf.target_week),
            season=config.temporal_season,
        )
        if tested is not None:
            leaf.share_t, leaf.share_p, leaf.share_impact, leaf.share_n = tested[:4]
            leaf.share_method = tested.method
            leaf.share_spread = float(tested.spread)
            leaf.share_level = float(tested.level)


def decide_anomalies(
    evidence: Sequence[SeriesTemporalEvidence], config: DatasetConfig
) -> None:
    """Decide every series of one assessment in a single BH-FDR family.

    Each national series contributes its forecast p-value; each leaf its
    share-of-parent p-value (or its forecast p-value when no share test is
    possible). All measures and periods of one refresh are one family because
    they move together: separate families per measure or per level would let
    one ordinary wobble page several times, and under the global null the
    chance of any false page per refresh is then about ``temporal_fdr_q``.
    A significant series is an anomaly only when its movement is also
    material (the share-attributable movement for a share-tested leaf).
    """
    from .conformal import benjamini_hochberg

    series = list(evidence)
    if not series:
        return
    share_tested = [
        entry.level != "national" and entry.share_p is not None for entry in series
    ]
    p_values = [
        float(entry.share_p)
        if tested and entry.share_p is not None
        else _two_sided_percentile(entry)
        for entry, tested in zip(series, share_tested, strict=True)
    ]
    if config.temporal_fdr_enabled:
        adjusted, significant = benjamini_hochberg(p_values, config.temporal_fdr_q)
    else:
        adjusted = list(p_values)
        significant = [value <= config.temporal_fdr_q for value in p_values]
    for entry, adjusted_p, is_significant, tested in zip(
        series, adjusted, significant, share_tested, strict=True
    ):
        entry.adjusted_p = float(adjusted_p)
        entry.fdr_significant = bool(is_significant)
        entry.flags = [flag for flag in entry.flags if flag not in DECISION_FLAGS]
        if tested:
            material = abs(float(entry.share_impact or 0.0)) >= entry.materiality
            decided = "share_shift"
        else:
            material = (
                entry.forecast_median == 0.0
                or abs(float(entry.residual)) >= entry.materiality
            )
            decided = "forecast_lower" if entry.residual < 0 else "forecast_upper"
        entry.anomaly = bool(is_significant and material)
        if entry.anomaly:
            entry.flags.append(decided)
        if tested and entry.share_spread is not None and math.isfinite(entry.share_spread) and entry.share_n > 1:
            # What this refresh could see: the larger of the statistical limit
            # (lone anomaly, this family's size and q) and the materiality floor.
            floor = entry.materiality / abs(entry.forecast_median) if entry.forecast_median else 0.0
            q = config.temporal_fdr_q
            family = len(series) if config.temporal_fdr_enabled else 1
            entry.detectable_change, entry.detectable_change_80 = (
                max(floor, detectable_change(
                    entry.share_spread, entry.share_n - 1, family, q, power, entry.share_method,
                    entry.share_level if entry.share_level is not None and math.isfinite(entry.share_level) else 0.0,
                ))
                for power in (0.5, 0.8)
            )


def new_period_adjustments(
    current: pd.DataFrame,
    events: Sequence[Any],
    pair: VersionPair,
    config: DatasetConfig,
    target: int | None = None,
    metric: str | None = None,
) -> dict[str, float]:
    """Expected new-period contributions of newly appearing entities.

    A historical backfill also adds rows for the new period; those rows explain
    part of the latest-week movement and must not be counted as anomalies.
    """
    week = config.week_column
    metric = metric or config.primary_metric
    target = int(target if target is not None else pair.current_max_week)
    adjustments: dict[str, float] = {}
    if week not in current.columns or metric not in current.columns:
        return adjustments

    new_entity_events = [
        event
        for event in events
        if getattr(event, "classification", None) in (NEW_BACKFILL, NEW_RECENT)
    ]
    if not new_entity_events:
        return adjustments
    # Select the union of rows belonging to any newly appearing entity, once:
    # a product inside a new store matches both events but its movement is
    # explained a single time.
    mask = pd.Series(False, index=current.index)
    matched = False
    for event in new_entity_events:
        entity_type = getattr(event, "entity_type", None)
        entity_id = getattr(event, "entity_id", None)
        column = f"{entity_type}_id"
        if not entity_type or column not in current.columns:
            continue
        matched = True
        mask |= current[column] == entity_id
    if not matched:
        return adjustments
    rows = current.loc[mask & (current[week] == target)]
    if rows.empty:
        return adjustments
    adjustments["national"] = float(rows[metric].sum())
    for series_column in config.temporal_entity_columns:
        if series_column not in rows.columns:
            continue
        for value, group in rows.groupby(series_column, dropna=False):
            series_id = f"{series_column}:{value}"
            adjustments[series_id] = float(group[metric].sum())
    return adjustments


def _run_single_target(
    previous: pd.DataFrame,
    current: pd.DataFrame,
    pair: VersionPair,
    config: DatasetConfig,
    events: Sequence[Any],
    target: int,
    metric: str | None = None,
) -> TemporalResult | None:
    if not config.temporal_enabled or not pair.new_periods:
        return None
    metric = metric or config.primary_metric
    week = config.week_column
    # Train on the previous version only; new periods come from the current
    # version, so backfilled overlap history cannot inflate the baseline.
    current_new = current.loc[current[week].isin(list(pair.new_periods))]
    combined = pd.concat([previous, current_new], ignore_index=True, sort=False)
    series = build_temporal_series(combined, config, metric)
    if series.empty:
        return None
    aggregation = "snapshot" if metric in config.snapshot_metrics else "flow"
    calendar = build_calendar(config)
    adjustments = new_period_adjustments(
        current, events, pair, config, target=target, metric=metric
    )
    forecaster = None if config.forecaster == "auto" else get_forecaster(config)
    horizon = max(1, target - pair.previous_max_week)
    levels = list(config.forecast_quantiles)
    median_index = levels.index(0.5) if 0.5 in levels else len(levels) // 2
    alpha = config.temporal_interval_alpha
    reserved = max(0, int(getattr(config, "temporal_calibration_origins", 0)))
    pools = CalibrationPools(metric, horizon, alpha)
    pending: list[dict[str, Any]] = []
    selections: dict[str, Any] = {}
    unavailable_series: list[str] = []
    unavailable: list[dict[str, Any]] = []

    for series_id, group in series.groupby("series_id", sort=True):
        group = group.sort_values(week)
        weeks = [int(value) for value in group[week]]
        values = [float(value) for value in group["value"]]
        observed = [
            (week_id, value)
            for week_id, value in zip(weeks, values, strict=True)
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
            for week_id, value in zip(weeks, values, strict=True)
            if week_id == target and math.isfinite(value)
        ]
        series_key = str(series_id)
        if (
            len(observed) < config.temporal_min_history
            or len(training) < config.temporal_min_history
        ):
            unavailable_series.append(series_key)
            unavailable.append(
                {
                    "series_id": series_key,
                    "metric": metric,
                    "target_week": target,
                    "reason": "insufficient observed history",
                }
            )
            continue
        if not actuals:
            unavailable_series.append(series_key)
            unavailable.append(
                {
                    "series_id": series_key,
                    "metric": metric,
                    "target_week": target,
                    "reason": "missing target observation",
                }
            )
            continue
        train_weeks = [week_id for week_id, _ in training]
        train_values = [value for _, value in training]
        level = series_level(series_key)
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
                for level, prediction in zip(levels, predictions, strict=True)
            }
        # The latest configured origins are reserved exclusively for
        # calibration; selection never reuses them.
        calibration_errors = (
            rolling_origin_errors(
                train_weeks,
                train_values,
                selection.spec,
                horizon,
                calendar,
                config,
                forecaster=forecaster,
                reserved_tail=reserved,
                evaluation="calibration",
            )
            if reserved > 0
            else []
        )
        pool = pools.add(level, selection.spec.revision, calibration_errors)
        selections[f"{metric}|{series_key}"] = selection.compact_dict()
        history_weeks = target - min(weeks) + 1
        missing_weeks = tuple(
            int(week_id)
            for week_id, value in zip(weeks, values, strict=True)
            if week_id <= target and not math.isfinite(value)
        )
        pending.append(
            {
                "series_id": series_key,
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
                "horizon": horizon,
                "forecast_origin_week": pair.previous_max_week,
                "training_endpoint_week": train_weeks[-1],
                "calibration_z": [z for z, _ in calibration_errors],
                "calibration_raw": [raw for _, raw in calibration_errors],
                "history_observed": len(training),
                "history_weeks": history_weeks,
                "missing_weeks": missing_weeks,
            }
        )

    if config.temporal_sparse_fallback and pending:
        # Sparse leaves use the parent expectation scaled by their historical
        # share. The fallback is labelled and never yields calibration support.
        points = {item["series_id"]: item["point"] for item in pending}
        remaining_unavailable: list[dict[str, Any]] = []
        for entry in unavailable:
            sid = str(entry["series_id"])
            level = series_level(sid)
            if level == "national" or ":" not in sid or "national" not in points:
                remaining_unavailable.append(entry)
                continue
            child_value = sid.split(":", 1)[1]
            share = national_share(
                combined, level, child_value, config, through_week=pair.previous_max_week
            )
            if share is None or share <= 0.0:
                remaining_unavailable.append(entry)
                continue
            group = series.loc[series["series_id"] == sid].sort_values(week)
            weeks = [int(value) for value in group[week]]
            values = [float(value) for value in group["value"]]
            observed = [
                (week_id, value_id)
                for week_id, value_id in zip(weeks, values, strict=True)
                if math.isfinite(value_id)
            ]
            training = [
                (week_id, value_id)
                for week_id, value_id in observed
                if week_id <= pair.previous_max_week
            ]
            actuals = [
                value_id
                for week_id, value_id in zip(weeks, values, strict=True)
                if week_id == target and math.isfinite(value_id)
            ]
            if len(observed) < 4 or len(training) < 4 or not actuals:
                remaining_unavailable.append(entry)
                continue
            pending.append(
                {
                    "series_id": sid,
                    "level": level,
                    "actual": actuals[0],
                    "adjustment": float(adjustments.get(sid, 0.0)),
                    "train_weeks": [week_id for week_id, _ in training],
                    "train_values": [value_id for _, value_id in training],
                    "selection": None,
                    "point": float(points["national"]) * float(share),
                    "quantiles": {},
                    "scale": None,
                    "pool": None,
                    "horizon": horizon,
                    "forecast_origin_week": pair.previous_max_week,
                    "training_endpoint_week": [week_id for week_id, _ in training][-1],
                    "calibration_z": [],
                    "calibration_raw": [],
                    "fallback": True,
                    "parent_series": "national",
                    "share": float(share),
                    "history_observed": len(training),
                    "history_weeks": target - min(weeks) + 1,
                    "missing_weeks": tuple(
                        int(week_id)
                        for week_id, value_id in zip(weeks, values, strict=True)
                        if week_id <= target and not math.isfinite(value_id)
                    ),
                }
            )
        unavailable = remaining_unavailable
        unavailable_series = sorted(
            {str(entry["series_id"]) for entry in unavailable}
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
            calibration_series = list(item.get("calibration_z", ()))
            if selection.spec.kind != FIXED_KIND:
                scale = _difference_scale(
                    np.asarray(item["train_values"], dtype=float),
                    config.temporal_season,
                )
                if pool.residuals:
                    item["quantiles"] = {
                        str(level): point
                        + scale * float(np.quantile(pool.residuals, level))
                        for level in levels
                    }
                if not item["quantiles"]:
                    item["quantiles"] = {
                        str(0.5): point,
                    }
            else:
                scale = item["scale"] or _residual_scale(pool.raw_residuals, point)
            interval = pool.conformal(
                point, alpha, scale, series_errors=calibration_series
            )
            interval_lower = interval.lower
            interval_upper = interval.upper
            interval_width = interval.width
            interval_coverage = pool.coverage.get("coverage")
            calibration_status = (
                "OK"
                if (
                    interval.status == "OK"
                    and calibration_series
                    and selection.mode != "insufficient"
                )
                else "INSUFFICIENT_CALIBRATION"
            )
            calibration_pool = pool.key
            calibration_n = len(calibration_series)
            raw_calibration = item.get("calibration_raw", [])
            forecast_error = (
                float(np.mean(np.abs(np.asarray(raw_calibration, dtype=float))))
                if raw_calibration
                else None
            )
            selected_model = selection.spec.revision
            support = "model"
            calibrated = None
        z = (adjusted_actual - point) / scale
        nominal = _NORMAL.cdf(z)
        if not item.get("fallback"):
            # An empirical percentile at a 1% tail needs enough held-out errors
            # to be meaningful; with too few samples every extreme z maps to 0/1
            # and would flag by construction. Insufficient calibration falls
            # back to the nominal percentile and is reported as such.
            tail_alpha = min(
                config.temporal_lower_percentile,
                1.0 - config.temporal_upper_percentile,
            )
            if calibration_n >= minimum_samples(tail_alpha):
                calibrated = item["pool"].percentile(z)

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
        # Statistical significance is not enough: a target week must also move
        # materially versus the same metric, period and scope's expected
        # magnitude before it can raise an anomaly. The materiality carries a
        # configured absolute floor and never uses an accumulated total.
        relative_residual = (
            (adjusted_actual - point) / abs(point) if point else 0.0
        )
        materiality = max(
            float(getattr(config, "temporal_materiality_abs", 0.0)),
            config.temporal_min_relative_residual * abs(point),
        )
        material = point == 0.0 or abs(adjusted_actual - point) >= materiality
        if material:
            if abs(rz) >= config.temporal_z_threshold:
                flags.append("robust_z")
            if abs(sz) >= config.temporal_z_threshold:
                flags.append("seasonal_z")
            if abs(ez) >= config.temporal_z_threshold:
                flags.append("ewma")
        if cp >= config.temporal_change_point_threshold:
            flags.append("change_point")

        # Decided once per assessment across every series (decide_anomalies).
        anomaly = False

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
                metric=metric,
                aggregation=aggregation,
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
                horizon=int(item["horizon"]),
                materiality=float(materiality),
                training_endpoint_week=int(item["training_endpoint_week"]),
                forecast_origin_week=int(item["forecast_origin_week"]),
            )
        )
    # Anomalies are decided once per assessment in ``run_temporal_qc``.
    _attach_share_tests(evidence, pending, config)
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
        "metric": metric,
        "n": sum(pool.n for pool in pools.pools.values()),
        "coverage": coverage,
        "horizon": horizon,
        "horizons": sorted({pool.horizon for pool in pools.pools.values()}),
        "targets": [target],
        "calibration_origins": reserved,
        "alpha": alpha,
        "calendar": calendar_identity(config),
        "models": sorted({pool.revision for pool in pools.pools.values()}),
        "pools": [pool.to_dict() for pool in pools.pools.values()],
        "selection": selections,
        "parent_share": [
            {
                "metric": metric,
                "series_id": item["series_id"],
                "parent_series": item["parent_series"],
                "share": item["share"],
                "basis": "historical_share",
            }
            for item in pending
            if item.get("fallback")
        ],
    }
    return TemporalResult(
        target_week=target,
        targets=[target],
        anomaly=bool(anomalies),
        flags=overall_flags,
        calibration=calibration,
        series=evidence,
        unavailable_series=unavailable_series,
        unavailable=unavailable,
        calendar=calendar_identity(config),
        metric_status={metric: "ASSESSED"},
    )


def run_temporal_qc(
    previous: pd.DataFrame,
    current: pd.DataFrame,
    pair: VersionPair,
    config: DatasetConfig,
    events: Sequence[Any] = (),
    targets: Sequence[int] | None = None,
) -> TemporalResult | None:
    """Assess every configured measure and appended period.

    Each measure/period gets its own period-specific forecasts, calibration
    pool and evidence; snapshot measures are compared within a period and
    never summed across time. Training for every target uses only
    previous-snapshot observations: earlier newly appended periods never enter
    training for later targets in the same assessment. Absent configured
    measures are reported explicitly so required measures can make the
    assessment incomplete instead of silently passing.
    """
    if not config.temporal_enabled or not pair.new_periods:
        return None
    requested = sorted(
        {
            int(value)
            for value in (targets if targets is not None else pair.new_periods)
        }
    )
    valid_targets = [
        value for value in requested if value > pair.previous_max_week
    ]
    if not valid_targets:
        return None
    metrics = config.temporal_metrics() or (config.primary_metric,)
    results: list[TemporalResult] = []
    metric_status: dict[str, str] = {}
    for metric in metrics:
        metric_results = [
            item
            for item in (
                _run_single_target(
                    previous, current, pair, config, events, target, metric
                )
                for target in valid_targets
            )
            if item is not None
        ]
        metric_status[metric] = "ASSESSED" if metric_results else "ABSENT"
        results.extend(metric_results)
    if not results:
        target = max(valid_targets)
        return TemporalResult(
            target_week=target,
            targets=list(valid_targets),
            anomaly=False,
            flags=[],
            calibration={
                "unavailable_reason": "no configured measure is present",
                "metric_status": dict(metric_status),
            },
            series=[],
            unavailable_series=[],
            unavailable=[
                {
                    "series_id": "national",
                    "metric": metric,
                    "target_week": target,
                    "reason": "configured measure is absent",
                }
                for metric in metrics
            ],
            calendar=calendar_identity(config),
            metric_status=metric_status,
        )
    series = [entry for result in results for entry in result.series]
    # Every series of every measure and period is one BH family.
    decide_anomalies(series, config)
    if len(results) == 1:
        result = results[0]
        result.metric_status = dict(metric_status)
        anomalous = [entry for entry in result.series if entry.anomaly]
        result.anomaly = bool(anomalous)
        result.flags = sorted({flag for entry in anomalous for flag in entry.flags})
        return result
    unavailable = [entry for result in results for entry in result.unavailable]
    unavailable_series = sorted(
        {entry for result in results for entry in result.unavailable_series}
    )
    pools = [
        pool for result in results for pool in result.calibration.get("pools", [])
    ]
    selections: dict[str, Any] = {}
    parent_share: list[dict[str, Any]] = []
    coverage: dict[str, Any] = {}
    for result in results:
        selections.update(result.calibration.get("selection", {}))
        parent_share.extend(result.calibration.get("parent_share", []))
        # Same key as a single-measure result: the nominal coverage level.
        for level_key, value in (result.calibration.get("coverage") or {}).items():
            coverage.setdefault(level_key, value)
    calibration = {
        "n": sum(int(pool.get("n", 0)) for pool in pools),
        "coverage": coverage,
        "horizon": max(int(result.calibration.get("horizon", 0)) for result in results),
        "horizons": sorted(
            {
                int(result.calibration.get("horizon", 0))
                for result in results
            }
        ),
        "targets": list(valid_targets),
        "calibration_origins": max(
            int(result.calibration.get("calibration_origins", 0))
            for result in results
        ),
        "alpha": config.temporal_interval_alpha,
        "calendar": calendar_identity(config),
        "models": sorted(
            {
                str(pool.get("revision", ""))
                for pool in pools
                if pool.get("revision")
            }
        ),
        "pools": pools,
        "selection": selections,
        "parent_share": parent_share,
    }
    anomalies = [entry for entry in series if entry.anomaly]
    flags = sorted({flag for entry in anomalies for flag in entry.flags})
    return TemporalResult(
        target_week=max(valid_targets),
        targets=list(valid_targets),
        anomaly=bool(anomalies),
        flags=flags,
        calibration=calibration,
        series=series,
        unavailable_series=unavailable_series,
        unavailable=unavailable,
        calendar=calendar_identity(config),
        metric_status=dict(metric_status),
    )
