"""Pipeline lineage: stage summaries and first-divergence detection.

For each captured stage the overlap revision is fingerprinted for *both*
versions. The first stage where the fingerprints differ (row/entity/metric
structure) is the likely origin of the change. Stages outside the configured
pipeline order are reported as unmapped; when no stage can be mapped the
status is ``UNKNOWN`` rather than ``PASS``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import pandas as pd

from .config import DatasetConfig
from .fingerprints import fingerprint_deltas, structural_fingerprint
from .source import VersionSource
from .versions import VersionPair


@dataclass
class LineageResult:
    status: str = "PASS"
    first_divergence: str | None = None
    stages: list[str] = field(default_factory=list)
    unmapped: list[str] = field(default_factory=list)
    summaries: dict[str, dict[str, Any]] = field(default_factory=dict)
    divergences: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "first_divergence": self.first_divergence,
            "stages": list(self.stages),
            "unmapped": list(self.unmapped),
            "summaries": self.summaries,
            "divergences": list(self.divergences),
        }


def _overlap_relative_divergence(
    previous: pd.DataFrame,
    current: pd.DataFrame,
    overlap: list[int],
    config: DatasetConfig,
    grain: tuple[str, ...] | None = None,
) -> tuple[float | None, int]:
    week = config.week_column
    metric = config.primary_metric
    if week not in previous.columns or week not in current.columns:
        return None, 0
    previous_overlap = previous.loc[previous[week].isin(overlap)]
    current_overlap = current.loc[current[week].isin(overlap)]
    row_delta = int(len(current_overlap) - len(previous_overlap))

    if metric not in previous_overlap.columns or metric not in current_overlap.columns:
        return None, row_delta
    keys = [week, *(grain if grain is not None else config.entity_key_columns)]
    if any(key not in previous or key not in current for key in keys):
        return None, row_delta
    previous_by_week = previous_overlap.groupby(keys, dropna=False)[metric].sum(min_count=1)
    current_by_week = current_overlap.groupby(keys, dropna=False)[metric].sum(min_count=1)
    aligned = pd.concat(
        [previous_by_week.rename("previous"), current_by_week.rename("current")],
        axis=1,
    ).fillna(0.0)
    absolute = float((aligned["current"] - aligned["previous"]).abs().sum())
    previous_total = float(aligned["previous"].sum())
    if abs(previous_total) > 1e-12:
        return absolute / abs(previous_total), row_delta
    return (0.0 if absolute <= 1e-9 else None), row_delta


def _fingerprint_changed(
    previous: pd.DataFrame,
    current: pd.DataFrame,
    overlap: list[int],
    config: DatasetConfig,
) -> tuple[bool, dict[str, float]]:
    week = config.week_column
    previous_overlap = previous.loc[previous[week].isin(overlap)]
    current_overlap = current.loc[current[week].isin(overlap)]
    deltas = fingerprint_deltas(
        structural_fingerprint(previous_overlap, config),
        structural_fingerprint(current_overlap, config),
    )
    changed = any(abs(value) > 1e-9 for value in deltas.values())
    return changed, deltas


def analyze_lineage(
    source: VersionSource,
    previous_id: str,
    current_id: str,
    stages: list[str],
    config: DatasetConfig,
    pair: VersionPair,
) -> LineageResult:
    ordered = [stage for stage in config.pipeline_order if stage in stages]
    unmapped = [stage for stage in stages if stage not in ordered]
    result = LineageResult(stages=ordered, unmapped=unmapped)
    if not ordered:
        result.status = "UNKNOWN"
        return result

    overlap = list(pair.overlap_weeks)
    for stage in ordered:
        previous = source.read_fact(previous_id, stage)
        current = source.read_fact(current_id, stage)
        relative, row_delta = _overlap_relative_divergence(
            previous, current, overlap, config,
            dict(config.stage_keys).get(stage, config.report_grain if stage == config.contract_stage else config.entity_key_columns)
        )
        stage_config = replace(config, entity_key_columns=dict(config.stage_keys).get(stage, config.report_grain if stage == config.contract_stage else config.entity_key_columns))
        fingerprint_changed, deltas = _fingerprint_changed(
            previous, current, overlap, stage_config
        )
        diverged = (
            fingerprint_changed
            or row_delta != 0
            or relative is None
            or relative > config.lineage_materiality_ratio
        )
        result.summaries[stage] = {
            "rows": int(len(current)),
            "fingerprint": structural_fingerprint(current, stage_config),
            "previous_fingerprint": structural_fingerprint(previous, stage_config),
            "fingerprint_deltas": deltas,
        }
        result.divergences.append(
            {
                "stage": stage,
                "row_count_delta": row_delta,
                "relative_divergence": relative,
                "fingerprint_changed": fingerprint_changed,
                "diverged": diverged,
            }
        )
        if result.first_divergence is None and diverged:
            result.first_divergence = stage

    if result.first_divergence is not None:
        result.status = "FIRST_DIVERGENCE"
    elif unmapped:
        result.status = "UNKNOWN"
    else:
        result.status = "PASS"
    return result
