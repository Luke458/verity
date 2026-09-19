"""Counterfactual reconstruction.

Remove the explained structural changes from the current overlap revision and
compare the reconstructed current against the previous version. A reconstruction
that closely matches the previous version is strong evidence that the
identified lifecycle events genuinely explain the change (architecture section
17).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .config import DatasetConfig
from .lifecycle import (
    EXPLAINABLE_CLASSES,
    EXTENDED,
    LifecycleEvent,
    NEW_BACKFILL,
    NEW_RECENT,
    RECLASSIFIED,
    REMOVED,
    TRUNCATED,
)


@dataclass
class CounterfactualResult:
    metric: str
    raw_delta: float = 0.0
    explained_delta: float = 0.0
    reconstructed_delta: float = 0.0
    previous_total: float = 0.0
    reconciliation_score: float = 1.0
    by_week: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "raw_delta": self.raw_delta,
            "explained_delta": self.explained_delta,
            "reconstructed_delta": self.reconstructed_delta,
            "previous_total": self.previous_total,
            "reconciliation_score": self.reconciliation_score,
            "by_week": list(self.by_week),
        }


def _explanation_type(
    events: list[LifecycleEvent], config: DatasetConfig
) -> str | None:
    explainable = [
        event for event in events if event.classification in EXPLAINABLE_CLASSES
    ]
    for entity_type in config.explanation_entity_types:
        if any(event.entity_type == entity_type for event in explainable):
            return entity_type
    return explainable[0].entity_type if explainable else None


def reconstruct_counterfactual(
    base_cube: pd.DataFrame,
    events: list[LifecycleEvent],
    config: DatasetConfig,
) -> CounterfactualResult:
    metric = config.primary_metric
    delta_column = f"{metric}_delta"
    overlap = (
        base_cube.loc[base_cube["period"] == "overlap"]
        if len(base_cube)
        else base_cube
    )
    result = CounterfactualResult(metric=metric)
    if delta_column not in overlap.columns or not len(overlap):
        return result

    raw_by_week = (
        overlap.groupby(config.week_column)[delta_column].sum().sort_index()
    )
    explanation_type = _explanation_type(events, config)
    explained_by_week = pd.Series(0.0, index=raw_by_week.index)
    for event in events:
        if event.entity_type != explanation_type:
            continue
        if event.classification in (NEW_BACKFILL, NEW_RECENT, EXTENDED):
            for week, value in event.value_added_by_week.items():
                if week in explained_by_week.index:
                    explained_by_week.loc[week] += value
        elif event.classification in (TRUNCATED, REMOVED):
            for week, value in event.value_removed_by_week.items():
                if week in explained_by_week.index:
                    explained_by_week.loc[week] -= value

    adjusted = raw_by_week - explained_by_week
    raw_total = float(raw_by_week.sum())
    explained_total = float(explained_by_week.sum())
    reconstructed = float(adjusted.sum())
    if abs(raw_total) > 1e-12:
        score = 1.0 - abs(reconstructed) / abs(raw_total)
    else:
        score = 1.0 if abs(reconstructed) <= 1e-9 else 0.0
    score = max(0.0, min(1.0, score))

    return CounterfactualResult(
        metric=metric,
        raw_delta=raw_total,
        explained_delta=explained_total,
        reconstructed_delta=reconstructed,
        previous_total=float(
            overlap[f"{metric}_previous"].sum()
            if f"{metric}_previous" in overlap.columns
            else 0.0
        ),
        reconciliation_score=score,
        by_week=[
            {
                "week": int(week),
                "raw_delta": float(raw_by_week.loc[week]),
                "explained_delta": float(explained_by_week.loc[week]),
                "reconstructed_delta": float(adjusted.loc[week]),
            }
            for week in raw_by_week.index
        ],
    )
