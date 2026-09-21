"""Structural fingerprints.

Lightweight per-stage/version summaries used for lineage and reconciliation:
row and null counts, duplicate keys, metric sums, distinct entity counts and an
order-insensitive entity-set hash. Fingerprints are storage-independent and can
be persisted next to each Delta version.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from .config import DatasetConfig
from .jsonutil import dumps as json_dumps


def manifest_data_fingerprint(manifest: dict) -> str:
    """Content identity of a version pair from its stage fingerprints."""
    versions = manifest.get("versions", {})
    payload = {
        name: info.get("fingerprints", {}) for name, info in versions.items()
    }
    return hashlib.sha256(
        json_dumps(payload, sort_keys=True).encode()
    ).hexdigest()[:16]


def _entity_set_hash(frame: pd.DataFrame, columns: list[str]) -> str:
    if not columns or len(frame) == 0:
        return hashlib.sha256(b"").hexdigest()[:16]
    # Hash the *unique* entity tuples with a stable string cast, so duplicate
    # rows or object/categorical dtype drift do not change the entity set.
    unique = frame[columns].astype(str).drop_duplicates()
    hashed = (
        pd.util.hash_pandas_object(unique, index=False)
        .sort_values(kind="stable")
        .to_numpy(dtype=np.uint64)
    )
    return hashlib.sha256(hashed.tobytes()).hexdigest()[:16]


def structural_fingerprint(frame: pd.DataFrame, config: DatasetConfig) -> dict:
    week = config.week_column
    metrics = [column for column in config.metric_columns if column in frame.columns]
    entities = [column for column in config.entity_columns if column in frame.columns]
    key_columns = [week, *config.entity_key_columns]
    missing_keys = [key for key in key_columns if key not in frame]
    return {
        "rows": int(len(frame)),
        "weeks": int(frame[week].nunique()) if week in frame.columns else 0,
        "null_cells": (
            int(frame[metrics].isna().sum().sum()) if metrics else 0
        ),
        "duplicate_keys": (
            int(frame.duplicated(subset=key_columns).sum()) if not missing_keys else None
        ),
        "missing_grain_columns": missing_keys,
        "entity_set_hash": _entity_set_hash(frame, entities),
        "metric_sums": {
            metric: float((frame.loc[frame[week] == frame[week].max(), metric]
                           if metric in config.snapshot_metrics and week in frame else frame[metric]).sum(min_count=1))
            for metric in metrics
        },
        "distinct_counts": {
            entity: int(frame[entity].nunique()) for entity in entities
        },
    }


def fingerprint_deltas(previous: dict, current: dict) -> dict[str, float]:
    deltas: dict[str, float] = {}
    for key in ("rows", "null_cells", "duplicate_keys", "weeks"):
        if current.get(key) is not None and previous.get(key) is not None:
            deltas[key] = float(current[key] - previous[key])
    for metric, value in current.get("metric_sums", {}).items():
        deltas[f"metric:{metric}"] = float(
            value - previous.get("metric_sums", {}).get(metric, 0.0)
        )
    for entity, value in current.get("distinct_counts", {}).items():
        deltas[f"distinct:{entity}"] = float(
            value - previous.get("distinct_counts", {}).get(entity, 0)
        )
    if current.get("entity_set_hash") != previous.get("entity_set_hash"):
        deltas["entity_set_changed"] = 1.0
    return deltas
