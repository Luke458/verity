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

from .config import DatasetConfig
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


def _difference_scale(values: np.ndarray, season: int) -> float:
    if len(values) > season:
        diffs = values[season:] - values[:-season]
    else:
        diffs = np.diff(values) if len(values) > 1 else np.array([0.0])
    if len(diffs) == 0:
        return 1e-9
    median = float(np.median(diffs))
    scale = 1.4826 * float(np.median(np.abs(diffs - median)))
    level = max(abs(float(np.median(values))) if len(values) else 0.0, 1.0)
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
        past: list[float] = []
        for step in range(horizon):
            index = len(values) - self.season + step
            if 0 <= index < len(values):
                median = float(values[index])
            elif past:
                median = past[-1]
            elif len(values):
                median = float(values[-1])
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
    week = config.week_column
    metric = config.primary_metric
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
    return series.sort_values(["series_id", week], kind="stable").reset_index(drop=True)


def _mad(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    median = float(np.median(values))
    return float(np.median(np.abs(values - median)))


def robust_z(values: Sequence[float], target: float, window: int) -> float:
    recent = np.asarray(values[-window:], dtype=float)
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
        if (target_week - int(week)) % season == 0
    ]
    if len(same) < 2:
        return 0.0
    array = np.asarray(same, dtype=float)
    scale = 1.4826 * _mad(array)
    if scale <= 1e-12:
        return 0.0
    return (target - float(np.median(array))) / scale


def ewma_z(values: Sequence[float], target: float, span: int) -> float:
    series = pd.Series(np.asarray(values, dtype=float))
    if len(series) < 2:
        return 0.0
    ewma = series.ewm(span=span, adjust=False).mean()
    errors = (series - ewma.shift(1)).dropna().to_numpy()
    scale = 1.4826 * _mad(errors)
    if scale <= 1e-12:
        return 0.0
    return (target - float(ewma.iloc[-1])) / scale


def change_point_score(values: Sequence[float], window: int) -> float:
    recent = np.asarray(values[-window:], dtype=float)
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
            history = values[:index]
            actual = float(values[index])
            predictions = forecaster.predict(history, 1, levels)[0]
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "series_id": self.series_id,
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
        }


@dataclass
class TemporalResult:
    target_week: int
    anomaly: bool
    flags: list[str]
    calibration: dict[str, Any]
    series: list[SeriesTemporalEvidence]

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_week": self.target_week,
            "anomaly": self.anomaly,
            "flags": list(self.flags),
            "calibration": dict(self.calibration),
            "series": [item.to_dict() for item in self.series],
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
    adjustments = new_period_adjustments(current, events, pair, config)
    forecaster = get_forecaster(config)
    calibration = backtest_forecaster(series, config, forecaster)
    levels = list(config.forecast_quantiles)
    median_index = levels.index(0.5) if 0.5 in levels else len(levels) // 2
    evidence: list[SeriesTemporalEvidence] = []

    for series_id, group in series.groupby("series_id", sort=True):
        group = group.sort_values(config.week_column)
        weeks = [int(value) for value in group[config.week_column]]
        values = [float(value) for value in group["value"]]
        training = [
            (week, value)
            for week, value in zip(weeks, values)
            if week <= pair.previous_max_week
        ]
        actuals = [
            value for week, value in zip(weeks, values) if week == target
        ]
        if len(training) < config.temporal_min_history or not actuals:
            continue
        train_weeks = [week for week, _ in training]
        train_values = [value for _, value in training]
        actual = actuals[0]
        adjustment = float(adjustments.get(series_id, 0.0))
        adjusted_actual = actual - adjustment

        horizon = target - pair.previous_max_week
        predictions = forecaster.predict(train_values, horizon, levels)[-1]
        median = float(predictions[median_index])
        scale = _forecast_scale(predictions, levels)
        z = (adjusted_actual - median) / scale
        nominal = _NORMAL.cdf(z)
        calibrated = calibration.percentile(z)

        rz = robust_z(train_values, adjusted_actual, config.temporal_window)
        sz = seasonal_z(
            train_weeks, train_values, target, adjusted_actual, config.temporal_season
        )
        ez = ewma_z(train_values, adjusted_actual, config.temporal_window)
        cp = change_point_score(train_values, config.temporal_window * 2)

        flags: list[str] = []
        percentile = calibrated if calibrated is not None else nominal
        # Statistical significance is not enough: a target week must also move
        # materially versus the forecast median before it can raise an anomaly.
        relative_residual = (
            (adjusted_actual - median) / median if median else 0.0
        )
        material = (
            median == 0.0
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
                series_id=series_id,
                target_week=target,
                actual=actual,
                adjusted_actual=adjusted_actual,
                adjustment=adjustment,
                forecast_median=median,
                forecast_quantiles={
                    str(level): float(prediction)
                    for level, prediction in zip(levels, predictions)
                },
                residual=adjusted_actual - median,
                relative_residual=(
                    (adjusted_actual - median) / median if median else 0.0
                ),
                nominal_percentile=nominal,
                calibrated_percentile=calibrated,
                standardized_residual=z,
                robust_z=rz,
                seasonal_z=sz,
                ewma_z=ez,
                change_point_score=cp,
                anomaly=anomaly,
                flags=flags,
            )
        )
    anomalies = [item for item in evidence if item.anomaly]
    overall_flags = sorted({flag for item in anomalies for flag in item.flags})
    return TemporalResult(
        target_week=target,
        anomaly=bool(anomalies),
        flags=overall_flags,
        calibration={"n": calibration.n, "coverage": calibration.coverage},
        series=evidence,
    )
