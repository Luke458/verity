"""Revision cube construction.

For every metric and grain, join the previous and current snapshots and record
previous value, current value, absolute delta and relative delta. Rows whose
week is a new period are labelled ``new``; the rest are ``overlap``. Revision
QC compares overlap weeks only, because new periods have no previous value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DatasetConfig


def _aggregate(
    frame: pd.DataFrame,
    keys: list[str],
    metrics: list[str],
    config: DatasetConfig,
) -> pd.DataFrame:
    if any(key not in frame.columns for key in keys):
        raise ValueError("declared aggregation grain is missing")
    if config.week_column not in keys and any(metric in config.snapshot_metrics for metric in metrics):
        raise ValueError("snapshot metrics cannot be aggregated across time")
    if not keys:
        return pd.DataFrame()

    aggs: dict[str, pd.NamedAgg] = {
        metric: pd.NamedAgg(column=metric, aggfunc="sum")
        for metric in metrics
        if metric in frame.columns
    }
    for count, entity in config.count_entity_map:
        if count not in frame.columns:
            continue
        if entity in frame.columns and entity not in keys:
            aggs[count] = pd.NamedAgg(column=entity, aggfunc="nunique")
        elif count not in aggs:
            aggs[count] = pd.NamedAgg(column=count, aggfunc="sum")
    if not aggs:
        return pd.DataFrame(columns=keys)

    grouped = (
        frame.groupby(keys, dropna=False)
        .agg(**aggs)
        .reset_index()
    )
    return grouped


def base_keys(frame: pd.DataFrame, config: DatasetConfig) -> list[str]:
    keys = [config.week_column]
    keys += list(dict(config.stage_keys).get(config.analysis_stage, config.entity_key_columns))
    return keys


def headline_keys(frame: pd.DataFrame, config: DatasetConfig) -> list[str]:
    keys = [config.week_column]
    keys += list(config.report_grain)
    return keys


def build_revision_cube(
    previous: pd.DataFrame,
    current: pd.DataFrame,
    keys: list[str],
    config: DatasetConfig,
    new_periods: tuple[int, ...],
) -> pd.DataFrame:
    week = config.week_column
    if any(key not in previous.columns or key not in current.columns for key in keys):
        raise ValueError("declared revision grain is missing")
    if not keys:
        raise ValueError("no shared grain keys between versions")

    metrics = [
        metric
        for metric in config.metric_columns
        if metric in previous.columns or metric in current.columns
    ]
    counts = [
        count
        for count, _ in config.count_entity_map
        if count in previous.columns or count in current.columns
    ]
    measures = metrics + counts

    previous_agg = _aggregate(previous, keys, metrics, config)
    current_agg = _aggregate(current, keys, metrics, config)
    merged = previous_agg.merge(
        current_agg, on=keys, how="outer", suffixes=("_previous", "_current"), indicator=True
    )

    for measure in measures:
        previous_column = f"{measure}_previous"
        current_column = f"{measure}_current"
        if previous_column not in merged.columns:
            merged[previous_column] = np.nan
        if current_column not in merged.columns:
            merged[current_column] = np.nan
        if config.absent_entity_policy == "zero":
            merged.loc[merged["_merge"] == "right_only", previous_column] = 0.0
            merged.loc[merged["_merge"] == "left_only", current_column] = 0.0
        delta = merged[current_column] - merged[previous_column]
        merged[f"{measure}_delta"] = delta
        previous_values = merged[previous_column].to_numpy(dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            merged[f"{measure}_relative"] = np.where(
                previous_values != 0.0,
                delta.to_numpy(dtype=np.float64) / previous_values,
                np.nan,
            )

    merged["period"] = np.where(
        merged[week].isin(list(new_periods)), "new", "overlap"
    )
    return merged.drop(columns="_merge").sort_values(keys).reset_index(drop=True)


def cube_summary(cube: pd.DataFrame, metric: str) -> dict[str, float]:
    """Total previous/current/delta for a metric, ignoring new periods."""
    overlap = cube.loc[cube["period"] == "overlap"]
    previous_column = f"{metric}_previous"
    current_column = f"{metric}_current"
    delta_column = f"{metric}_delta"
    if delta_column not in overlap.columns:
        return {"previous": 0.0, "current": 0.0, "delta": 0.0}
    return {
        "previous": float(overlap[previous_column].sum()),
        "current": float(overlap[current_column].sum()),
        "delta": float(overlap[delta_column].sum()),
    }
