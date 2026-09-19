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
    week = config.week_column
    result = CounterfactualResult(metric=metric)
    if metric not in previous.columns or metric not in current.columns:
        return result
    if week not in previous.columns or week not in current.columns:
        return result

    overlap = sorted(
        {int(value) for value in previous[week].unique()}
        & {int(value) for value in current[week].unique()}
    )
    if not overlap:
        return result

    previous_week = previous.groupby(week)[metric].sum().astype(float)
    current_week = current.groupby(week)[metric].sum().astype(float)
    adjusted = current_week.copy()
    relevant = [
        event for event in events if event.classification in EXPLAINABLE_CLASSES
    ]
    if not relevant:
        return result

    reconstructable = 0
    for event in relevant:
        if event.classification in (NEW_BACKFILL, NEW_RECENT, EXTENDED):
            column = _entity_column(event.entity_type, current)
            weeks = list(event.historical_weeks_added)
            if column is None or not weeks:
                continue
            contribution = _week_sum(current, column, event.entity_id, weeks, config)
            adjusted = adjusted.subtract(contribution, fill_value=0.0)
            reconstructable += 1
        elif event.classification in (TRUNCATED, REMOVED):
            column = _entity_column(event.entity_type, previous)
            weeks = list(event.historical_weeks_removed)
            if column is None or not weeks:
                continue
            contribution = _week_sum(previous, column, event.entity_id, weeks, config)
            adjusted = adjusted.add(contribution, fill_value=0.0)
            reconstructable += 1
        elif event.classification == RECLASSIFIED:
            # Net-conserving by definition; the score tests that conservation.
            reconstructable += 1

    if reconstructable == 0:
        return result

    raw = current_week - previous_week
    reconstructed = adjusted - previous_week
    overlap_index = [
        value
        for value in overlap
        if value in adjusted.index and value in previous_week.index
    ]
    numerator = float(
        sum(
            abs(
                float(adjusted.get(value, 0.0))
                - float(previous_week.get(value, 0.0))
            )
            for value in overlap_index
        )
    )
    denominator = float(
        sum(abs(float(previous_week.get(value, 0.0))) for value in overlap_index)
    )
    if denominator <= 1e-9:
        score = 1.0 if numerator <= 1e-9 else 0.0
    else:
        score = max(0.0, 1.0 - numerator / denominator)

    return CounterfactualResult(
        metric=metric,
        raw_delta=float(raw.reindex(overlap_index).fillna(0.0).sum()),
        explained_delta=float(
            (current_week - adjusted).reindex(overlap_index).fillna(0.0).sum()
        ),
        reconstructed_delta=float(
            reconstructed.reindex(overlap_index).fillna(0.0).sum()
        ),
        previous_total=float(
            previous_week.reindex(overlap_index).fillna(0.0).sum()
        ),
        reconciliation_score=score,
        evaluated=True,
        by_week=[
            {
                "week": int(value),
                "raw_delta": float(raw.get(value, 0.0)),
                "reconstructed_delta": float(reconstructed.get(value, 0.0)),
            }
            for value in overlap_index
        ],
    )
