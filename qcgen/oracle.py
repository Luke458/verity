"""Ground-truth oracle: case records, entity masks and effect measurement.

Every injected fault produces exactly one ``GroundTruthCase``. Effects are
measured by comparing the clean pipeline against the faulty pipeline stage by
stage, restricted to the affected entities and weeks. Because the pipeline
transforms are deterministic, the effect of a fault at its injection stage is
exact, and rows outside the affected set cancel between clean and faulty.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

ENTITY_COLUMNS: tuple[tuple[str, str], ...] = (
    ("store_id", "stores"),
    ("product_id", "products"),
    ("commodity_id", "commodities"),
    ("banner_id", "banners"),
    ("state_id", "states"),
)

# Week indices are keys, not measures.
EXCLUDED_NUMERIC: frozenset[str] = frozenset({"week"})


@dataclass
class GroundTruthCase:
    case_id: str
    family: str
    kind: str  # "fault" | "control" | "expected_event"
    injection_stage: str
    expected_status: str
    expected_class: str
    expected_origin: str | None = None
    affected: dict[str, list[Any]] = field(default_factory=dict)
    weeks: list[int] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    effects: dict[str, dict[str, float]] = field(default_factory=dict)
    injected_effect: dict[str, float] = field(default_factory=dict)
    event_registry: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def affected_mask(
    fact: pd.DataFrame,
    affected: dict[str, list[Any]] | None,
    weeks: list[int] | None = None,
) -> pd.Series:
    """Rows matching the affected entities, narrowed to the affected weeks.

    Entity types are OR-ed; on grains where a type is absent it is skipped.
    Rows matched but not actually injected cancel between clean and faulty.
    """
    mask = pd.Series(False, index=fact.index)
    used = False
    for column, key in ENTITY_COLUMNS:
        ids = affected.get(key) if affected else None
        if ids and column in fact.columns:
            mask |= fact[column].isin(list(ids))
            used = True
    if not used:
        mask = pd.Series(True, index=fact.index)
    if weeks and "week" in fact.columns:
        mask &= fact["week"].isin(list(weeks))
    return mask


def summarize(
    fact: pd.DataFrame | None,
    affected: dict[str, list[Any]] | None,
    weeks: list[int] | None = None,
) -> dict[str, float]:
    if fact is None or len(fact) == 0:
        return {}
    subset = fact.loc[affected_mask(fact, affected, weeks)]
    if len(subset) == 0:
        return {}
    numeric = [
        column
        for column in subset.columns
        if column not in EXCLUDED_NUMERIC
        and pd.api.types.is_numeric_dtype(subset[column])
        and not pd.api.types.is_bool_dtype(subset[column])
    ]
    return {column: float(subset[column].sum(skipna=True)) for column in numeric}


def effect_delta(
    clean_fact: pd.DataFrame | None,
    faulty_fact: pd.DataFrame | None,
    affected: dict[str, list[Any]] | None,
    weeks: list[int] | None = None,
) -> dict[str, float]:
    before = summarize(clean_fact, affected, weeks)
    after = summarize(faulty_fact, affected, weeks)
    keys = sorted(set(before) | set(after))
    return {key: round(after.get(key, 0.0) - before.get(key, 0.0), 6) for key in keys}
