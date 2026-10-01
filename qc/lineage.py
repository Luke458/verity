"""Pipeline lineage: stage summaries and origin attribution.

For each captured stage the overlap revision (current minus previous, per
week) is measured, and each stage's *increment* over the stage upstream of it
is computed. A value-preserving pipeline passes an upstream revision through
unchanged, so only the stage that introduced a change shows an increment. The
origin (``first_divergence``, kept for compatibility) is the stage with the
largest material increment. This matters when the source legitimately
restates recent weeks (late-arriving data): the source then always diverges
first, and "first divergent stage" would blame it for every downstream fault.

When no stage adds a material value increment, the first stage whose
structure changed (rows, entities, fingerprints) is reported, as before.
``lineage_restatement_weeks`` excludes a declared late-arrival window from the
comparison. Stages outside the configured pipeline order are reported as
unmapped; when no stage can be mapped the status is ``UNKNOWN``.
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


def _week_revision(
    previous: pd.DataFrame,
    current: pd.DataFrame,
    weeks: list[int],
    config: DatasetConfig,
) -> tuple[pd.Series | None, float]:
    """Per-week revision of the primary metric and the previous total."""
    week = config.week_column
    metric = config.primary_metric
    if any(
        column not in frame.columns
        for frame in (previous, current)
        for column in (week, metric)
    ):
        return None, 0.0
    before = previous.loc[previous[week].isin(weeks)].groupby(week)[metric].sum()
    after = current.loc[current[week].isin(weeks)].groupby(week)[metric].sum()
    index = pd.Index(sorted(weeks))
    before = before.reindex(index, fill_value=0.0).astype(float)
    after = after.reindex(index, fill_value=0.0).astype(float)
    return after - before, float(before.abs().sum())


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
    window = max(0, int(config.lineage_restatement_weeks))
    compared = sorted(overlap)[:-window] if window else list(overlap)
    upstream_revision: pd.Series | None = None
    increments: dict[str, float] = {}
    for stage in ordered:
        previous = source.read_fact(previous_id, stage)
        current = source.read_fact(current_id, stage)
        revision, scale = _week_revision(previous, current, compared, config)
        increment: float | None = None
        if revision is not None and scale > 1e-12:
            inherited = (
                upstream_revision.reindex(revision.index, fill_value=0.0)
                if upstream_revision is not None
                else 0.0 * revision
            )
            increment = float((revision - inherited).abs().sum()) / scale
            increments[stage] = increment
            upstream_revision = revision
        relative, row_delta = _overlap_relative_divergence(
            previous, current, compared, config,
            dict(config.stage_keys).get(stage, config.report_grain if stage == config.contract_stage else config.entity_key_columns)
        )
        stage_config = replace(config, entity_key_columns=dict(config.stage_keys).get(stage, config.report_grain if stage == config.contract_stage else config.entity_key_columns))
        fingerprint_changed, deltas = _fingerprint_changed(
            previous, current, compared, stage_config
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
                "increment": increment,
                "fingerprint_changed": fingerprint_changed,
                "diverged": diverged,
            }
        )

    material = {
        stage: value
        for stage, value in increments.items()
        if value > config.lineage_materiality_ratio
    }
    if material:
        result.first_divergence = max(material, key=lambda stage: material[stage])
    else:
        result.first_divergence = next(
            (item["stage"] for item in result.divergences if item["diverged"]), None
        )

    if result.first_divergence is not None:
        result.status = "FIRST_DIVERGENCE"
    elif unmapped:
        result.status = "UNKNOWN"
    else:
        result.status = "PASS"
    return result
