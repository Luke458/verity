"""Counterfactual reconstruction.

Rebuild the current overlap revision as if the structural lifecycle events had
not happened: remove the value of added entity-weeks, restore the value of
removed entity-weeks, and compare the reconstructed weekly totals against the
previous version. The values are recomputed from the frames, not taken from
the events, so a lifecycle classification that points at the wrong entity or
week cannot produce a high score.

A score is only produced when there is at least one reconstructable or
net-conserving structural event; otherwise ``reconciliation_score`` is ``None``
and the result must not be treated as evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .config import DatasetConfig
from .lifecycle import (
    EXPLAINABLE_CLASSES,
    EXTENDED,
    NEW_BACKFILL,
    NEW_RECENT,
    RECLASSIFIED,
    REMOVED,
    TRUNCATED,
    LifecycleEvent,
)


@dataclass
class CounterfactualResult:
    metric: str
    raw_delta: float = 0.0
    explained_delta: float = 0.0
    reconstructed_delta: float = 0.0
    previous_total: float = 0.0
    reconciliation_score: float | None = None
    evaluated: bool = False
    absolute_residual: float = 0.0
    by_week: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "raw_delta": self.raw_delta,
            "explained_delta": self.explained_delta,
            "reconstructed_delta": self.reconstructed_delta,
            "previous_total": self.previous_total,
            "reconciliation_score": self.reconciliation_score,
            "evaluated": self.evaluated,
            "absolute_residual": self.absolute_residual,
            "by_week": list(self.by_week),
        }


def _entity_column(entity_type: str, frame: pd.DataFrame) -> str | None:
    column = f"{entity_type}_id"
    return column if column in frame.columns else None


def _week_sum(frame: pd.DataFrame, column: str, entity_id: str, weeks: list[int], config: DatasetConfig) -> pd.Series:
    week = config.week_column
    metric = config.primary_metric
    rows = frame.loc[
        (frame[column].astype(str) == entity_id) & frame[week].isin(weeks)
    ]
    if rows.empty:
        return pd.Series(dtype=float)
    return rows.groupby(week)[metric].sum()


def reconstruct_counterfactual(
    previous: pd.DataFrame,
    current: pd.DataFrame,
    events: list[LifecycleEvent],
    config: DatasetConfig,
) -> CounterfactualResult:
    metric = config.primary_metric
    week, metric = config.week_column, config.primary_metric
    result = CounterfactualResult(metric=metric)
    keys = [week, *dict(config.stage_keys).get(config.analysis_stage, config.entity_key_columns)]
    if any(key not in previous or key not in current for key in keys) or metric not in previous or metric not in current:
        return result
    overlap = sorted(set(previous[week]) & set(current[week]))
    if not overlap:
        return result
    remove = pd.Series(False, index=current.index)
    restore = pd.Series(False, index=previous.index)
    reconstructable = False
    for event in events:
        if event.classification not in EXPLAINABLE_CLASSES:
            continue
        column = _entity_column(event.entity_type, current)
        if column is None or column not in previous:
            continue
        if event.classification in (NEW_BACKFILL, NEW_RECENT, EXTENDED) and event.historical_weeks_added:
            remove |= current[column].astype(str).eq(event.entity_id) & current[week].isin(event.historical_weeks_added)
            reconstructable = True
        elif event.classification in (TRUNCATED, REMOVED) and event.historical_weeks_removed:
            restore |= previous[column].astype(str).eq(event.entity_id) & previous[week].isin(event.historical_weeks_removed)
            reconstructable = True
        elif event.classification == RECLASSIFIED:
            reconstructable = True
    if not reconstructable:
        return result
    adjusted = pd.concat([current.loc[~remove], previous.loc[restore]], ignore_index=True)
    def grouped(frame):
        return frame.loc[frame[week].isin(overlap)].groupby(keys, dropna=False)[metric].sum(min_count=1)
    aligned = pd.concat([grouped(previous).rename("previous"), grouped(current).rename("current"),
                         grouped(adjusted).rename("adjusted")], axis=1)
    if config.absent_entity_policy != "zero" and aligned.isna().any().any():
        return result
    aligned = aligned.fillna(0.0)
    raw = aligned.current - aligned.previous
    residual = aligned.adjusted - aligned.previous
    numerator = float(residual.abs().sum())
    denominator = float(aligned.previous.abs().sum())
    score = max(0., 1. - numerator / denominator) if denominator > 1e-9 else float(numerator <= 1e-9)
    weekly_raw = raw.groupby(level=week).sum()
    weekly_residual = residual.groupby(level=week).sum()
    return CounterfactualResult(metric=metric, raw_delta=float(raw.sum()),
        explained_delta=float((aligned.current - aligned.adjusted).sum()),
        reconstructed_delta=float(residual.sum()), previous_total=float(aligned.previous.sum()),
        reconciliation_score=score, evaluated=True, absolute_residual=numerator,
        by_week=[{"week": int(value), "raw_delta": float(weekly_raw.get(value, 0.)),
                  "reconstructed_delta": float(weekly_residual.get(value, 0.))} for value in overlap])
