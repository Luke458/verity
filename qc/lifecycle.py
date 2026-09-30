"""Entity lifecycle classification.

Compares entity-week presence and value between versions and classifies each
changed entity: new recent, historical backfill, removed, history extended or
truncated, a gap in the latest period, or a possible reclassification.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from .config import DatasetConfig
from .versions import VersionPair

UNCHANGED = "UNCHANGED_ENTITY"
NEW_RECENT = "NEW_ENTITY_RECENT"
NEW_BACKFILL = "NEW_ENTITY_HISTORICAL_BACKFILL"
REMOVED = "ENTITY_REMOVED"
EXTENDED = "ENTITY_HISTORY_EXTENDED"
TRUNCATED = "ENTITY_HISTORY_TRUNCATED"
LATEST_MISSING = "LATEST_WEEK_MISSING"
RECLASSIFIED = "POSSIBLE_RECLASSIFICATION"

CHANGED_CLASSES = frozenset(
    {NEW_RECENT, NEW_BACKFILL, REMOVED, EXTENDED, TRUNCATED, LATEST_MISSING, RECLASSIFIED}
)
# NEW_RECENT is an entity appearing only with the new period; it has no
# historical impact and does not need registry confirmation.
EXPLAINABLE_CLASSES = frozenset(
    {NEW_BACKFILL, REMOVED, TRUNCATED, RECLASSIFIED}
)


@dataclass
class LifecycleEvent:
    entity_type: str
    entity_id: str
    classification: str
    classifications: tuple[str, ...] = ()
    first_week_previous: int | None = None
    last_week_previous: int | None = None
    first_week_current: int | None = None
    last_week_current: int | None = None
    week_count_previous: int = 0
    week_count_current: int = 0
    rows_previous: int = 0
    rows_current: int = 0
    historical_weeks_added: tuple[int, ...] = ()
    historical_weeks_removed: tuple[int, ...] = ()
    historical_value_added: float = 0.0
    historical_value_removed: float = 0.0
    value_added_by_week: dict[int, float] = field(default_factory=dict)
    value_removed_by_week: dict[int, float] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["historical_weeks_added"] = list(self.historical_weeks_added)
        data["historical_weeks_removed"] = list(self.historical_weeks_removed)
        data["classifications"] = list(self.classifications)
        return data


def _entity_type(column: str) -> str:
    return column[:-3] if column.endswith("_id") else column


def _presence(
    frame: pd.DataFrame,
    entity_column: str,
    week_column: str,
    metric: str,
) -> tuple[dict[str, set[int]], dict[tuple[str, int], float], dict[str, int]]:
    # min_count=1 makes an all-null group NaN, so a row that carries no metric
    # value does not count as presence.
    grouped = frame.groupby([entity_column, week_column], dropna=False)[metric].sum(
        min_count=1
    )
    weeks: dict[str, set[int]] = {}
    values: dict[tuple[str, int], float] = {}
    for (entity, week), value in grouped.items():
        if pd.isna(value):
            continue
        entity_id = str(entity)
        week_id = int(week)
        weeks.setdefault(entity_id, set()).add(week_id)
        values[(entity_id, week_id)] = float(value)
    rows = {
        str(entity): int(count)
        for entity, count in frame.groupby(entity_column, dropna=False).size().items()
    }
    return weeks, values, rows


def _value_for_weeks(
    values: dict[tuple[str, int], float], entity_id: str, weeks: set[int]
) -> float:
    return float(sum(values.get((entity_id, week), 0.0) for week in weeks))


def _values_by_week(
    values: dict[tuple[str, int], float], entity_id: str, weeks: set[int]
) -> dict[int, float]:
    return {
        int(week): float(values.get((entity_id, week), 0.0))
        for week in sorted(weeks)
    }


def classify_entity_changes(
    previous: pd.DataFrame,
    current: pd.DataFrame,
    pair: VersionPair,
    config: DatasetConfig,
) -> list[LifecycleEvent]:
    week = config.week_column
    metric = config.primary_metric
    if metric not in previous.columns or metric not in current.columns:
        return []
    overlap = set(pair.overlap_weeks)
    events: list[LifecycleEvent] = []

    for entity_column in config.entity_columns:
        if entity_column not in previous.columns or entity_column not in current.columns:
            continue
        entity_type = _entity_type(entity_column)
        previous_weeks, previous_values, previous_rows = _presence(
            previous, entity_column, week, metric
        )
        current_weeks, current_values, current_rows = _presence(
            current, entity_column, week, metric
        )

        for entity_id in sorted(set(previous_weeks) | set(current_weeks)):
            in_previous = entity_id in previous_weeks
            in_current = entity_id in current_weeks
            previous_set = previous_weeks.get(entity_id, set())
            current_set = current_weeks.get(entity_id, set())

            event = LifecycleEvent(
                entity_type=entity_type,
                entity_id=entity_id,
                classification=UNCHANGED,
                first_week_previous=min(previous_set) if previous_set else None,
                last_week_previous=max(previous_set) if previous_set else None,
                first_week_current=min(current_set) if current_set else None,
                last_week_current=max(current_set) if current_set else None,
                week_count_previous=len(previous_set),
                week_count_current=len(current_set),
                rows_previous=previous_rows.get(entity_id, 0),
                rows_current=current_rows.get(entity_id, 0),
            )

            added_weeks = current_set - previous_set
            removed_weeks = previous_set - current_set
            historical_added = added_weeks & overlap
            historical_removed = removed_weeks & overlap
            # An entity that traded last week and not this week is only evidence
            # of a coverage regression if that entity is *reliably* present. Real
            # transactional data is sparse: a product that trades in roughly one
            # week in five is expected to be absent in the next one. Without an
            # entity-specific reliability baseline, absence degenerates into
            # "always fires" on any naturally sparse source, which on real data
            # is every refresh.
            candidates = [
                week_id
                for week_id in range(
                    pair.overlap_start, pair.previous_max_week + 1
                )
            ][-config.entity_presence_window :]
            presence_rate = (
                sum(1 for week_id in candidates if week_id in previous_set)
                / len(candidates)
                if candidates
                else 0.0
            )
            latest_missing = (
                pair.previous_max_week in previous_set
                and pair.current_max_week in pair.new_periods
                and pair.current_max_week not in current_set
                and presence_rate >= config.entity_presence_threshold
            )

            classifications: list[str] = []
            if in_previous and not in_current:
                if not historical_removed:
                    # Only non-overlap weeks disappeared (e.g. a shortened
                    # history): there is no structural change in the overlap.
                    continue
                classifications.append(REMOVED)
            elif in_current and not in_previous:
                classifications.append(
                    NEW_BACKFILL if historical_added else NEW_RECENT
                )
            else:
                if latest_missing:
                    classifications.append(LATEST_MISSING)
                if historical_removed:
                    classifications.append(TRUNCATED)
                if historical_added:
                    classifications.append(EXTENDED)

            if not classifications:
                continue

            if historical_added:
                event.historical_weeks_added = tuple(sorted(historical_added))
                event.historical_value_added = _value_for_weeks(
                    current_values, entity_id, historical_added
                )
                event.value_added_by_week = _values_by_week(
                    current_values, entity_id, historical_added
                )
            if historical_removed:
                event.historical_weeks_removed = tuple(sorted(historical_removed))
                event.historical_value_removed = _value_for_weeks(
                    previous_values, entity_id, historical_removed
                )
                event.value_removed_by_week = _values_by_week(
                    previous_values, entity_id, historical_removed
                )
            event.classifications = tuple(classifications)
            event.classification = classifications[0]
            events.append(event)

    return events


def _product_commodity_map(
    frame: pd.DataFrame, dim: pd.DataFrame | None
) -> dict[str, str]:
    if dim is not None and {"product_id", "commodity_id"} <= set(dim.columns):
        return {
            str(product): str(commodity)
            for product, commodity in zip(dim["product_id"], dim["commodity_id"])
        }
    if "product_id" in frame.columns and "commodity_id" in frame.columns:
        grouped = frame.groupby("product_id")["commodity_id"].agg(
            lambda values: sorted({str(value) for value in values})
        )
        return {
            str(product): commodities[0]
            for product, commodities in grouped.items()
            if len(commodities) == 1
        }
    return {}


def detect_reclassification(
    previous: pd.DataFrame,
    current: pd.DataFrame,
    previous_dim: pd.DataFrame | None,
    current_dim: pd.DataFrame | None,
    pair: VersionPair,
    config: DatasetConfig,
) -> list[LifecycleEvent]:
    """Detect products whose commodity mapping changed between versions."""
    previous_map = _product_commodity_map(previous, previous_dim)
    current_map = _product_commodity_map(current, current_dim)
    if not previous_map or not current_map:
        return []

    pairs: dict[tuple[str, str], list[str]] = {}
    for product, previous_commodity in previous_map.items():
        current_commodity = current_map.get(product)
        if current_commodity is not None and current_commodity != previous_commodity:
            pairs.setdefault((previous_commodity, current_commodity), []).append(product)

    week = config.week_column
    metric = config.primary_metric
    overlap = list(pair.overlap_weeks)
    events: list[LifecycleEvent] = []
    for (from_commodity, to_commodity), products in sorted(pairs.items()):
        product_set = {str(product) for product in products}
        moved_previous = float(
            previous.loc[
                previous["product_id"].astype(str).isin(product_set)
                & previous[week].isin(overlap),
                metric,
            ].sum()
        )
        moved_current = float(
            current.loc[
                current["product_id"].astype(str).isin(product_set)
                & current[week].isin(overlap),
                metric,
            ].sum()
        )
        previous_from = float(
            previous.loc[
                (previous["commodity_id"].astype(str) == from_commodity)
                & previous[week].isin(overlap),
                metric,
            ].sum()
        )
        current_from = float(
            current.loc[
                (current["commodity_id"].astype(str) == from_commodity)
                & current[week].isin(overlap),
                metric,
            ].sum()
        )
        previous_to = float(
            previous.loc[
                (previous["commodity_id"].astype(str) == to_commodity)
                & previous[week].isin(overlap),
                metric,
            ].sum()
        )
        current_to = float(
            current.loc[
                (current["commodity_id"].astype(str) == to_commodity)
                & current[week].isin(overlap),
                metric,
            ].sum()
        )
        from_delta = current_from - previous_from
        to_delta = current_to - previous_to
        # Conservation is measured on the value that actually moved with the
        # products, not on whole-commodity totals that other changes can move.
        gross = max(abs(moved_previous), abs(moved_current))
        conservation = (
            1.0 - abs(moved_current - moved_previous) / gross
            if gross > 1e-9
            else 1.0
        )

        events.append(
            LifecycleEvent(
                entity_type="commodity",
                entity_id=f"{from_commodity}->{to_commodity}",
                classification=RECLASSIFIED,
                historical_value_added=moved_current,
                historical_value_removed=moved_previous,
                details={
                    "from_commodity": from_commodity,
                    "to_commodity": to_commodity,
                    "products": products,
                    "product_count": len(products),
                    "from_delta": from_delta,
                    "to_delta": to_delta,
                    "moved_previous": moved_previous,
                    "moved_current": moved_current,
                    "gross_moved": gross,
                    "conservation_ratio": conservation,
                },
            )
        )
    return events
