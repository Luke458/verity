"""Declared hierarchy aggregates, coverage checks and sparse parent support.

Every declared view is built from the same analysis frame, so national, banner,
state, commodity, product and store totals stay internally reconcilable.
Geographic and commercial breakdowns are alternative views of the same total,
never additive explanations of each other. Declared intersections
(commodity-by-banner, commodity-by-state) are checked against their additive
parent rather than against the national total.

Sparse temporal leaves use the parent expectation scaled by a historically
estimated child share. The fallback is labelled explicitly and cannot support
an automatic statistical explanation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import DatasetConfig

HIERARCHY_SCHEMA = 1

_ROLE_BY_COLUMN = {
    "banner_id": "commercial",
    "state_id": "geographic",
    "commodity_id": "commercial",
    "product_id": "product",
    "store_id": "store",
}


@dataclass(frozen=True)
class HierarchyLevel:
    name: str
    keys: tuple[str, ...]
    role: str
    parent: str | None = None
    additive_to_total: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "keys": list(self.keys),
            "role": self.role,
            "parent": self.parent,
            "additive_to_total": self.additive_to_total,
        }


def default_levels(config: DatasetConfig) -> list[HierarchyLevel]:
    levels = [HierarchyLevel("national", (), "total")]
    for column in config.entity_columns:
        levels.append(
            HierarchyLevel(
                column,
                (column,),
                _ROLE_BY_COLUMN.get(column, "other"),
            )
        )
    for keys in config.hierarchy_intersections:
        if not keys:
            continue
        levels.append(
            HierarchyLevel(
                "_x_".join(keys),
                tuple(keys),
                "intersection",
                parent=keys[0],
                additive_to_total=False,
            )
        )
    return levels


def _measure_columns(frame: pd.DataFrame, config: DatasetConfig) -> list[str]:
    measures = [metric for metric in config.metric_columns if metric in frame.columns]
    for count, entity in config.count_entity_map:
        if count in frame.columns or entity in frame.columns:
            measures.append(count)
    return list(dict.fromkeys(measures))


def build_aggregate(
    frame: pd.DataFrame, keys: list[str], config: DatasetConfig
) -> pd.DataFrame:
    """Aggregate measures to ``keys``; snapshot metrics need the period key."""
    if any(key not in frame.columns for key in keys):
        raise ValueError("declared aggregation key is missing")
    week = config.week_column
    if week not in keys and any(
        metric in config.snapshot_metrics for metric in config.metric_columns
    ):
        raise ValueError("snapshot metrics cannot be aggregated across time")
    if not keys:
        return pd.DataFrame()
    aggs: dict[str, pd.NamedAgg] = {}
    for measure in _measure_columns(frame, config):
        entity = next(
            (name for count, name in config.count_entity_map if count == measure), None
        )
        if entity is not None and entity in frame.columns:
            aggs[measure] = pd.NamedAgg(column=entity, aggfunc="nunique")
        elif measure in frame.columns:
            aggs[measure] = pd.NamedAgg(column=measure, aggfunc="sum")
    if not aggs:
        return pd.DataFrame(columns=keys)
    return frame.groupby(keys, dropna=False).agg(**aggs).reset_index()


class HierarchyCache:
    """Per-run aggregate cache; frames are held so their identity stays stable."""

    def __init__(self, config: DatasetConfig) -> None:
        self.config = config
        self._hierarchies: dict[int, tuple[pd.DataFrame, dict[str, pd.DataFrame]]] = {}

    def get(
        self, frame: pd.DataFrame, levels: list[HierarchyLevel] | None = None
    ) -> dict[str, pd.DataFrame]:
        key = id(frame)
        cached = self._hierarchies.get(key)
        if cached is None or cached[0] is not frame:
            built = build_hierarchy(
                frame, self.config, levels or default_levels(self.config)
            )
            self._hierarchies[key] = (frame, built)
            return built
        return cached[1]


def build_hierarchy(
    frame: pd.DataFrame,
    config: DatasetConfig,
    levels: list[HierarchyLevel] | None = None,
) -> dict[str, pd.DataFrame]:
    week = config.week_column
    built: dict[str, pd.DataFrame] = {}
    for level in levels or default_levels(config):
        if any(key not in frame.columns for key in level.keys):
            continue
        keys = [week, *level.keys]
        try:
            built[level.name] = build_aggregate(frame, keys, config)
        except ValueError:
            continue
    return built


@dataclass
class HierarchyCheck:
    name: str
    level: str
    status: str  # PASS | FAIL | UNSUPPORTED
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "level": self.level,
            "status": self.status,
            "detail": self.detail,
        }


def hierarchy_checks(
    current: pd.DataFrame,
    config: DatasetConfig,
    levels: list[HierarchyLevel] | None = None,
) -> list[HierarchyCheck]:
    """Inexpensive coverage and conservation checks at every declared level.

    Checks are coverage-only: they confirm each declared view reconciles with
    its additive parent and that keys are fully populated. They do not fit
    models and never treat one breakdown as an explanation of another.
    """
    week = config.week_column
    levels = levels or default_levels(config)
    checks: list[HierarchyCheck] = []
    available = [level for level in levels if all(key in current.columns for key in level.keys)]
    national = build_aggregate(current, [week], config)
    for level in available:
        if level.name == "national":
            continue
        if not level.keys:
            continue
        frame = build_aggregate(current, [week, *level.keys], config)
        if frame.empty:
            checks.append(
                HierarchyCheck(
                    f"hierarchy_total:{level.name}",
                    level.name,
                    "UNSUPPORTED",
                    "no rows at the declared level",
                )
            )
            continue
        null_keys = int(frame[list(level.keys)].isna().any(axis=1).sum())
        if null_keys:
            checks.append(
                HierarchyCheck(
                    f"hierarchy_keys:{level.name}",
                    level.name,
                    "FAIL",
                    f"{null_keys} rows have missing hierarchy keys",
                )
            )
        tolerance = config.reconciliation_tolerance
        for metric in config.metric_columns:
            if metric not in frame.columns or metric not in national.columns:
                continue
            level_totals = frame.groupby(week, dropna=False)[metric].sum()
            national_totals = national.set_index(week)[metric]
            aligned = level_totals.reindex(national_totals.index).fillna(0.0)
            scale = national_totals.abs().clip(lower=1.0)
            worst = float(((aligned - national_totals).abs() / scale).max())
            status = "PASS" if worst <= tolerance else "FAIL"
            checks.append(
                HierarchyCheck(
                    f"hierarchy_total:{level.name}:{metric}",
                    level.name,
                    status,
                    f"max relative deviation {worst:.3g}",
                )
            )
        if level.parent:
            parent = next(
                (item for item in available if item.name == level.parent), None
            )
            if parent is not None and parent.name != "national":
                parent_frame = build_aggregate(current, [week, *parent.keys], config)
                for metric in config.metric_columns:
                    if metric not in frame.columns or metric not in parent_frame.columns:
                        continue
                    child_totals = frame.groupby(week, dropna=False)[metric].sum()
                    parent_totals = parent_frame.groupby(week, dropna=False)[metric].sum()
                    aligned = child_totals.reindex(parent_totals.index).fillna(0.0)
                    scale = parent_totals.abs().clip(lower=1.0)
                    worst = float(((aligned - parent_totals).abs() / scale).max())
                    checks.append(
                        HierarchyCheck(
                            f"hierarchy_total:{level.name}->{level.parent}:{metric}",
                            level.name,
                            "PASS" if worst <= tolerance else "FAIL",
                            f"max relative deviation {worst:.3g}",
                        )
                    )
    return checks


def national_share(
    frame: pd.DataFrame,
    column: str,
    value: Any,
    config: DatasetConfig,
    through_week: int,
) -> float | None:
    """Historically estimated share of one child series in the national total.

    The share is arithmetic support only; it is never a statistical
    explanation by itself.
    """
    week = config.week_column
    metric = config.primary_metric
    if column not in frame.columns or metric not in frame.columns:
        return None
    scoped = frame.loc[frame[week] <= through_week]
    child = (
        scoped.loc[scoped[column].astype(str) == str(value)]
        .groupby(week, dropna=False)[metric]
        .sum()
    )
    total = scoped.groupby(week, dropna=False)[metric].sum()
    if child.empty or total.empty:
        return None
    aligned, totals = child.align(total, join="inner")
    if float(totals.sum()) <= 0.0 or not np.isfinite(float(totals.sum())):
        return None
    share = float(aligned.sum()) / float(totals.sum())
    if not np.isfinite(share):
        return None
    return min(max(share, 0.0), 1.0)
