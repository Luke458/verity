"""Prequential (across-load) calibration with leakage rules.

Calibration for a target week may only use residuals that were *available*
before the target run was scored: ``available_on < as_of`` and
``target_week < target_week``. Records are deduplicated per target so a
repeated or partially recomputed load cannot stack evidence for itself, and
the pool is bounded. This is the across-load counterpart to the within-version
calibration in ``qc/temporal.py``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .config import DatasetConfig
from .conformal import ConformalInterval, conformal_interval, minimum_samples
from .jsonutil import dumps as json_dumps
from .temporal import _forecast_scale, build_temporal_series, get_forecaster


@dataclass(frozen=True)
class CalibrationRecord:
    series_id: str
    target_week: int
    available_on: int
    residual: float
    run_id: str | None = None
    scope: str = "default"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalibrationRecord:
        residual = float(data["residual"])
        if residual != residual or residual in (float("inf"), float("-inf")):
            raise ValueError("residual must be finite")
        return cls(
            series_id=str(data["series_id"]),
            target_week=int(data["target_week"]),
            available_on=int(data["available_on"]),
            residual=residual,
            run_id=data.get("run_id"),
            scope=str(data.get("scope", "default")),
        )


class CalibrationPool:
    def __init__(self, max_records: int = 100, min_samples: int = 9) -> None:
        if max_records < 1:
            raise ValueError("max_records must be positive")
        self.max_records = max_records
        self.min_samples = min_samples
        self.records: list[CalibrationRecord] = []

    def add(self, record: CalibrationRecord) -> bool:
        """Add a record; the latest record for a key supersedes older ones.

        A repeated or partially recomputed load still contributes one record
        per (scope, series, target), but a corrected residual replaces the
        stale value instead of being rejected forever.
        """
        if not math.isfinite(float(record.residual)):
            raise ValueError("residual must be finite")
        if record in self.records:
            return False
        self.records.append(record)
        return True

    def usable(
        self, as_of: int, target_week: int, scope: str | None = None
    ) -> list[CalibrationRecord]:
        latest: dict[tuple[str, str, int], CalibrationRecord] = {}
        for record in sorted(self.records, key=lambda item: item.available_on):
            if (record.available_on < as_of and record.target_week < target_week
                    and (scope is None or record.scope == scope)):
                latest[(record.scope, record.series_id, record.target_week)] = record
        return list(latest.values())[-self.max_records:]

    def percentile(
        self,
        z: float,
        as_of: int,
        target_week: int,
        scope: str | None = None,
    ) -> float | None:
        """Empirical CDF fraction of residuals at or below ``z``."""
        residuals = sorted(
            record.residual
            for record in self.usable(as_of, target_week, scope=scope)
        )
        if len(residuals) < self.min_samples:
            return None
        import numpy as np

        array = np.asarray(residuals, dtype=float)
        return float(np.searchsorted(array, z, side="right") / len(array))

    def p_value(
        self,
        z: float,
        as_of: int,
        target_week: int,
        tail: str = "lower",
        scope: str | None = None,
    ) -> float | None:
        """Conformal p-value for ``z`` against the usable pool.

        ``(1 + #{residuals at least as extreme}) / (n + 1)``: small values mean
        ``z`` is unusually low (``tail="lower"``) or high (``tail="upper"``)
        relative to the pool. Finite-sample valid under exchangeability.
        """
        residuals = [
            record.residual
            for record in self.usable(as_of, target_week, scope=scope)
        ]
        n = len(residuals)
        if n == 0:
            return None
        if tail == "lower":
            extreme = sum(1 for value in residuals if value <= z)
        elif tail == "upper":
            extreme = sum(1 for value in residuals if value >= z)
        else:
            raise ValueError("tail must be 'lower' or 'upper'")
        return (1 + extreme) / (n + 1)

    def interval(
        self,
        center: float,
        as_of: int,
        target_week: int,
        alpha: float = 0.05,
        scope: str | None = None,
    ) -> ConformalInterval:
        """Conformal interval in residual units around ``center``.

        Residuals are standardized residuals, so callers pass ``center=0.0``
        to obtain an interval in z-space.
        """
        residuals = [
            record.residual
            for record in self.usable(as_of, target_week, scope=scope)
        ]
        if len(residuals) < max(self.min_samples, minimum_samples(alpha)):
            return ConformalInterval(
                alpha,
                None,
                None,
                len(residuals),
                None,
                "INSUFFICIENT_CALIBRATION",
                f"need {max(self.min_samples, minimum_samples(alpha))} usable "
                f"residuals, have {len(residuals)}",
            )
        return conformal_interval(residuals, center=center, alpha=alpha)

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_records": self.max_records,
            "min_samples": self.min_samples,
            "records": [record.to_dict() for record in self.records],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalibrationPool:
        pool = cls(
            max_records=int(data.get("max_records", 100)),
            min_samples=int(data.get("min_samples", 9)),
        )
        for payload in data.get("records", []):
            pool.records.append(CalibrationRecord.from_dict(payload))
        return pool


class PrequentialStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def add(self, records: Sequence[CalibrationRecord]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as handle:
            for record in records:
                if not math.isfinite(float(record.residual)):
                    raise ValueError(
                        f"residual for {record.series_id}@{record.target_week} "
                        "must be finite"
                    )
                handle.write(json_dumps(record.to_dict(), sort_keys=True) + "\n")

    def pool(self, max_records: int = 100, min_samples: int = 9) -> CalibrationPool:
        pool = CalibrationPool(max_records=max_records, min_samples=min_samples)
        if not self.path.exists():
            return pool
        for line in self.path.read_text().splitlines():
            if line.strip():
                pool.add(CalibrationRecord.from_dict(json.loads(line)))
        return pool


# ---------------------------------------------------------------------------
# Point-in-time forecasting across loads
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoadSnapshot:
    """One weekly load: the frame it delivered and when it became observable.

    ``available_on`` is the week in which this load's outcome became
    observable. It is used as the point-in-time boundary: records from a load
    with ``available_on >= as_of`` can never enter the calibration for the load
    being scored, including same-week loads.
    """

    load_id: str
    target_week: int
    frame: pd.DataFrame
    available_on: int
    version: str | None = None


@dataclass
class PrequentialSeriesEvidence:
    load_id: str
    target_week: int
    series_id: str
    status: str
    actual: float | None = None
    forecast_median: float | None = None
    relative_residual: float | None = None
    standardized_residual: float | None = None
    pool_percentile: float | None = None
    pool_p_value: float | None = None
    pool_records: int = 0
    flag: str | None = None
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PrequentialRunResult:
    scope: str
    loads: int
    records_committed: int
    evidence: list[PrequentialSeriesEvidence]
    pool: CalibrationPool
    limitations: list[str] = field(default_factory=list)

    @property
    def flagged(self) -> list[PrequentialSeriesEvidence]:
        return [item for item in self.evidence if item.flag]

    def to_dict(self) -> dict[str, Any]:
        loads: dict[str, dict[str, Any]] = {}
        for item in self.evidence:
            entry = loads.setdefault(
                item.load_id,
                {
                    "target_week": item.target_week,
                    "evaluated": 0,
                    "insufficient_history": 0,
                    "flagged": [],
                },
            )
            if item.status == "EVALUATED":
                entry["evaluated"] += 1
            else:
                entry["insufficient_history"] += 1
            if item.flag:
                entry["flagged"].append(
                    {"series_id": item.series_id, "flag": item.flag}
                )
        return {
            "scope": self.scope,
            "loads": self.loads,
            "records_committed": self.records_committed,
            "pool_size": len(self.pool.records),
            "by_load": loads,
            "evidence": [item.to_dict() for item in self.evidence],
            "limitations": list(self.limitations),
        }


def forecast_across_loads(
    loads: Sequence[LoadSnapshot],
    config: DatasetConfig | None = None,
    scope: str = "default",
    min_samples: int = 9,
) -> PrequentialRunResult:
    """Forecast each load from its own history, calibrated only on prior loads.

    For every load the training history is that load's own frame up to
    ``target_week - 1`` (causal by construction), and the calibration pool is
    queried with ``as_of = available_on`` before any of the current load's
    residuals are committed. Records are committed after all series of a load
    are scored, so a corrupted load can never widen its own interval.
    """
    if not loads:
        raise ValueError("at least one load is required")
    config = config or DatasetConfig()
    week = config.week_column
    metric = config.primary_metric
    forecaster = get_forecaster(config)
    levels = list(config.forecast_quantiles)
    median_index = levels.index(0.5) if 0.5 in levels else len(levels) // 2

    last_target: int | None = None
    last_available: int | None = None
    for load in loads:
        if week not in load.frame.columns or metric not in load.frame.columns:
            raise ValueError(
                f"{load.load_id}: frame must contain {week!r} and {metric!r}"
            )
        if last_target is not None and load.target_week <= last_target:
            raise ValueError(
                f"target weeks must strictly increase: {load.load_id} "
                f"repeats or moves backwards"
            )
        if last_available is not None and load.available_on < last_available:
            raise ValueError("available_on must not move backwards")
        last_target = load.target_week
        last_available = load.available_on

    pool = CalibrationPool(max_records=100, min_samples=min_samples)
    lower_alpha = config.temporal_lower_percentile
    upper_alpha = 1.0 - config.temporal_upper_percentile
    required_samples = max(
        min_samples, minimum_samples(min(lower_alpha, upper_alpha))
    )
    evidence: list[PrequentialSeriesEvidence] = []
    committed = 0

    for load in loads:
        series = build_temporal_series(load.frame, config)
        pending: list[CalibrationRecord] = []
        for series_id, group in series.groupby("series_id", sort=True):
            group = group.sort_values(week)
            weeks = [int(value) for value in group[week]]
            values = [float(value) for value in group["value"]]
            training = [
                (week_id, value)
                for week_id, value in zip(weeks, values)
                if week_id <= load.target_week - 1
            ]
            actuals = [
                value for week_id, value in zip(weeks, values) if week_id == load.target_week
            ]
            pool_records = len(
                pool.usable(load.available_on, load.target_week, scope=scope)
            )

            if not actuals or len(training) < config.temporal_min_history:
                evidence.append(
                    PrequentialSeriesEvidence(
                        load_id=load.load_id,
                        target_week=load.target_week,
                        series_id=series_id,
                        status="INSUFFICIENT_HISTORY",
                        pool_records=pool_records,
                        note=(
                            f"{len(training)} training weeks; "
                            f"need {config.temporal_min_history}"
                        ),
                    )
                )
                continue

            training_values = [value for _, value in training]
            actual = actuals[0]
            horizon = load.target_week - max(week_id for week_id, _ in training)
            predictions = forecaster.predict(training_values, horizon, levels)[-1]
            median = float(predictions[median_index])
            scale = _forecast_scale(predictions, levels)
            standardized = (actual - median) / scale
            relative = (actual - median) / median if median else 0.0
            percentile = pool.percentile(
                standardized, load.available_on, load.target_week, scope=scope
            )
            p_value = None
            flag = None
            if pool_records >= required_samples:
                lower_p = pool.p_value(
                    standardized,
                    load.available_on,
                    load.target_week,
                    tail="lower",
                    scope=scope,
                )
                upper_p = pool.p_value(
                    standardized,
                    load.available_on,
                    load.target_week,
                    tail="upper",
                    scope=scope,
                )
                if lower_p is not None and lower_p <= lower_alpha:
                    p_value = lower_p
                    flag = "prequential_lower"
                elif upper_p is not None and upper_p <= upper_alpha:
                    p_value = upper_p
                    flag = "prequential_upper"

            evidence.append(
                PrequentialSeriesEvidence(
                    load_id=load.load_id,
                    target_week=load.target_week,
                    series_id=series_id,
                    status="EVALUATED",
                    actual=actual,
                    forecast_median=median,
                    relative_residual=relative,
                    standardized_residual=standardized,
                    pool_percentile=percentile,
                    pool_p_value=p_value,
                    pool_records=pool_records,
                    flag=flag,
                    note=(
                        ""
                        if pool_records >= required_samples
                        else (
                            f"p-values need {required_samples} usable residuals "
                            f"for alpha={min(lower_alpha, upper_alpha)}; "
                            f"have {pool_records}"
                        )
                    ),
                )
            )
            pending.append(
                CalibrationRecord(
                    series_id=series_id,
                    target_week=load.target_week,
                    available_on=load.available_on,
                    residual=standardized,
                    run_id=load.load_id,
                    scope=scope,
                )
            )
        for record in pending:
            if pool.add(record):
                committed += 1

    return PrequentialRunResult(
        scope=scope,
        loads=len(loads),
        records_committed=committed,
        evidence=evidence,
        pool=pool,
        limitations=[
            "Prequential p-values need "
            f"{required_samples} usable residuals for alpha="
            f"{min(lower_alpha, upper_alpha)}; early loads are evaluated "
            "without flags.",
            "The pool mixes series of different scales within a scope; "
            "p-values are computed per residual, not per independent weekly "
            "shock.",
            "available_on is the caller's observation week, not an "
            "authenticated arrival time.",
            "Synthetic residuals inherit the generator's behaviour; real "
            "calibration needs real loads.",
        ],
    )


def prequential_sequence(
    source: Any,
    version_ids: Sequence[str],
    config: DatasetConfig | None = None,
    scope: str | None = None,
    min_samples: int = 9,
    stage: str | None = None,
) -> PrequentialRunResult:
    """Build loads from consecutive pinned versions of a VersionSource.

    Each version is treated as one weekly load: the target is its maximum
    week and ``available_on`` is that same week. Versions are read in the
    order given and must deliver strictly increasing target weeks.
    """
    if len(version_ids) < 1:
        raise ValueError("at least one version is required")
    config = config or DatasetConfig()
    scope = scope or config.name
    stage = stage or source.resolve_stage(version_ids[0], config.analysis_stages())
    week = config.week_column
    loads: list[LoadSnapshot] = []
    for version in version_ids:
        frame = source.read_fact(version, stage)
        if week not in frame.columns:
            raise ValueError(f"version {version!r} has no {week!r} column")
        target = int(frame[week].max())
        loads.append(
            LoadSnapshot(
                load_id=f"{scope}:{version}",
                target_week=target,
                frame=frame,
                available_on=target,
                version=str(version),
            )
        )
    return forecast_across_loads(
        loads, config, scope=scope, min_samples=min_samples
    )


def records_from_result(
    result: Any,
    scope: str = "default",
    available_on: int | None = None,
) -> list[CalibrationRecord]:
    """Calibration records from one run's temporal evidence.

    One record per series, using the standardized residual that was computed
    before the target was scored. ``available_on`` defaults to the target week
    (the week the outcome became observable).
    """
    temporal = getattr(result, "temporal", None)
    if temporal is None:
        return []
    run_id = str(getattr(result, "run_id", ""))
    records: list[CalibrationRecord] = []
    for item in temporal.series:
        # Multiple measures share scope identifiers; calibration identity is
        # metric-qualified so records never collide across measures.
        records.append(
            CalibrationRecord(
                series_id=(
                    f"{item.metric}:{item.series_id}"
                    if getattr(item, "metric", "")
                    else item.series_id
                ),
                target_week=int(item.target_week),
                available_on=(
                    int(item.target_week) if available_on is None else available_on
                ),
                residual=float(item.standardized_residual),
                run_id=run_id,
                scope=scope,
            )
        )
    return records


def append_run_calibration(
    result: Any,
    store_path: str | Path,
    scope: str = "default",
    available_on: int | None = None,
) -> int:
    records = records_from_result(result, scope=scope, available_on=available_on)
    if records:
        PrequentialStore(store_path).add(records)
    return len(records)
