"""Pipeline lineage: stage summaries and first-divergence detection.

For each captured stage the overlap revision magnitude is computed against the
previous version. The first stage where the divergence becomes material (or
where row counts change) is the likely origin of the change. Latest-period
faults that do not alter overlap history are surfaced by lifecycle instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .config import DatasetConfig
from .fingerprints import structural_fingerprint
from .source import VersionSource
from .versions import VersionPair


@dataclass
class LineageResult:
    status: str = "PASS"
    first_divergence: str | None = None
    stages: list[str] = field(default_factory=list)
    summaries: dict[str, dict[str, Any]] = field(default_factory=dict)
    divergences: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "first_divergence": self.first_divergence,
            "stages": list(self.stages),
            "summaries": self.summaries,
            "divergences": list(self.divergences),
        }


def _overlap_relative_divergence(
    previous: pd.DataFrame,
    current: pd.DataFrame,
    overlap: list[int],
    config: DatasetConfig,
) -> tuple[float, int]:
    week = config.week_column
    metric = config.primary_metric
    if week not in previous.columns or week not in current.columns:
        return 0.0, 0
    previous_overlap = previous.loc[previous[week].isin(overlap)]
    current_overlap = current.loc[current[week].isin(overlap)]
    row_delta = int(len(current_overlap) - len(previous_overlap))

    if metric not in previous_overlap.columns or metric not in current_overlap.columns:
        return 0.0, row_delta
    previous_by_week = previous_overlap.groupby(week)[metric].sum()
    current_by_week = current_overlap.groupby(week)[metric].sum()
    aligned = pd.concat(
        [previous_by_week.rename("previous"), current_by_week.rename("current")],
        axis=1,
    ).fillna(0.0)
    absolute = float((aligned["current"] - aligned["previous"]).abs().sum())
    previous_total = float(aligned["previous"].sum())
    if abs(previous_total) > 1e-12:
        relative = absolute / abs(previous_total)
    else:
        relative = 0.0 if absolute <= 1e-9 else float("inf")
    return relative, row_delta


def analyze_lineage(
    source: VersionSource,
    previous_id: str,
    current_id: str,
    stages: list[str],
    config: DatasetConfig,
    pair: VersionPair,
) -> LineageResult:
    ordered = [stage for stage in config.pipeline_order if stage in stages]
    result = LineageResult(stages=ordered)
    if not ordered:
        return result

    overlap = list(pair.overlap_weeks)
    for stage in ordered:
        previous = source.read_fact(previous_id, stage)
        current = source.read_fact(current_id, stage)
        relative, row_delta = _overlap_relative_divergence(
            previous, current, overlap, config
        )
        diverged = (
            relative > config.lineage_materiality_ratio or row_delta != 0
        )
        result.summaries[stage] = {
            "rows": int(len(current)),
            "fingerprint": structural_fingerprint(current, config),
        }
        result.divergences.append(
            {
                "stage": stage,
                "row_count_delta": row_delta,
                "relative_divergence": relative,
                "diverged": diverged,
            }
        )
        if result.first_divergence is None and diverged:
            result.first_divergence = stage

    result.status = (
        "FIRST_DIVERGENCE" if result.first_divergence else "PASS"
    )
    return result
