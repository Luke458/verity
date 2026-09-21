"""Quantified explanation ledger with one allocation per contribution.

Arithmetic attribution is kept separate from support for a legitimate
explanation. Matched-item quantity and unit-value effects use a symmetric
(Bennet-style) decomposition so the two effects sum exactly to the matched
movement, whichever direction the quantity and price move. Entry, exit and
new-period movement stay distinct categories, and the unexplained remainder is
recorded explicitly with its sign so offsetting errors cannot disappear.

Support labels ("approved_event", "statistical") never change the arithmetic:
they record which evidence would authorize a clearance, and only the review
policy may act on them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
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
)

LEDGER_SCHEMA = 1

ARITHMETIC = "arithmetic"
STATISTICAL = "statistical"
APPROVED_EVENT = "approved_event"

MATCHED_QUANTITY = "matched_quantity"
MATCHED_UNIT_VALUE = "matched_unit_value"
MATCHED_VOLUME = "matched_volume"
ENTITY_ENTRY = "entity_entry"
ENTITY_EXIT = "entity_exit"
CLASSIFICATION_CHANGE = "classification_change"
NEW_PERIOD_EXPECTED = "new_period_expected"
NEW_PERIOD_ENTITY_ENTRY = "new_period_entity_entry"
NEW_PERIOD_UNEXPLAINED = "new_period_unexplained"
UNEXPLAINED = "unexplained"


@dataclass(frozen=True)
class LedgerContribution:
    category: str
    scope: str
    metric: str
    value: float
    support: str = ARITHMETIC
    evidence_ids: tuple[str, ...] = ()
    approved_by: tuple[str, ...] = ()
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "scope": self.scope,
            "metric": self.metric,
            "value": self.value,
            "support": self.support,
            "evidence_ids": list(self.evidence_ids),
            "approved_by": list(self.approved_by),
            "detail": dict(self.detail),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LedgerContribution:
        return cls(
            category=str(data.get("category", "")),
            scope=str(data.get("scope", "")),
            metric=str(data.get("metric", "")),
            value=float(data.get("value", 0.0)),
            support=str(data.get("support", ARITHMETIC)),
            evidence_ids=tuple(data.get("evidence_ids", [])),
            approved_by=tuple(data.get("approved_by", [])),
            detail=dict(data.get("detail", {})),
        )


@dataclass
class ExplanationLedger:
    metric: str
    overlap_delta: float
    new_period_movement: float
    raw_movement: float
    contributions: list[LedgerContribution] = field(default_factory=list)
    conservation: dict[str, float] = field(default_factory=dict)
    coverage: dict[str, Any] = field(default_factory=dict)
    support_level: str = "unsupported"
    schema_version: int = LEDGER_SCHEMA

    @property
    def net_explained(self) -> float:
        return float(
            sum(
                item.value
                for item in self.contributions
                if item.category != UNEXPLAINED
                and item.category != NEW_PERIOD_UNEXPLAINED
            )
        )

    @property
    def gross_explained(self) -> float:
        return float(
            sum(
                abs(item.value)
                for item in self.contributions
                if item.category != UNEXPLAINED
                and item.category != NEW_PERIOD_UNEXPLAINED
            )
        )

    @property
    def net_unexplained(self) -> float:
        return float(
            sum(
                item.value
                for item in self.contributions
                if item.category in (UNEXPLAINED, NEW_PERIOD_UNEXPLAINED)
            )
        )

    @property
    def gross_unexplained(self) -> float:
        return float(
            sum(
                abs(item.value)
                for item in self.contributions
                if item.category in (UNEXPLAINED, NEW_PERIOD_UNEXPLAINED)
            )
        )

    @property
    def explained_fraction(self) -> float:
        if abs(self.raw_movement) < 1e-12:
            return 1.0 if abs(self.net_unexplained) < 1e-12 else 0.0
        return max(0.0, min(1.0, self.net_explained / self.raw_movement))

    def by_category(self) -> dict[str, float]:
        totals: dict[str, float] = {}
        for item in self.contributions:
            totals[item.category] = totals.get(item.category, 0.0) + item.value
        return totals

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "metric": self.metric,
            "overlap_delta": self.overlap_delta,
            "new_period_movement": self.new_period_movement,
            "raw_movement": self.raw_movement,
            "contributions": [item.to_dict() for item in self.contributions],
            "net_explained": self.net_explained,
            "gross_explained": self.gross_explained,
            "net_unexplained": self.net_unexplained,
            "gross_unexplained": self.gross_unexplained,
            "explained_fraction": self.explained_fraction,
            "conservation": dict(self.conservation),
            "coverage": dict(self.coverage),
            "support_level": self.support_level,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExplanationLedger:
        return cls(
            metric=str(data.get("metric", "")),
            overlap_delta=float(data.get("overlap_delta", 0.0)),
            new_period_movement=float(data.get("new_period_movement", 0.0)),
            raw_movement=float(data.get("raw_movement", 0.0)),
            contributions=[
                LedgerContribution.from_dict(item)
                for item in data.get("contributions", [])
            ],
            conservation=dict(data.get("conservation", {})),
            coverage=dict(data.get("coverage", {})),
            support_level=str(data.get("support_level", "unsupported")),
            schema_version=int(data.get("schema_version", 1)),
        )


def _entity_keys(config: DatasetConfig) -> list[str]:
    return list(
        dict(config.stage_keys).get(config.analysis_stage, config.entity_key_columns)
    )


def _overlap_aggregates(
    previous: pd.DataFrame,
    current: pd.DataFrame,
    keys: list[str],
    metric: str,
    quantity_metric: str,
    week: str,
) -> pd.DataFrame:
    group_keys = [*keys, week]

    def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
        measures = list(
            dict.fromkeys(
                column
                for column in (metric, quantity_metric)
                if column and column in frame.columns
            )
        )
        return frame.groupby(group_keys, dropna=False)[measures].sum().reset_index()

    previous_agg = aggregate(previous)
    current_agg = aggregate(current)
    return previous_agg.merge(
        current_agg,
        on=group_keys,
        how="outer",
        suffixes=("_previous", "_current"),
        indicator=True,
    )


def _top_entities(
    frame: pd.DataFrame,
    keys: list[str],
    value_column: str,
    limit: int,
) -> list[dict[str, Any]]:
    if value_column not in frame.columns:
        return []
    grouped = frame.groupby(keys, dropna=False)[value_column].sum()
    grouped = grouped.loc[grouped.abs() > 0].sort_values(key=abs, ascending=False)
    top: list[dict[str, Any]] = []
    for index, value in grouped.head(limit).items():
        scope = index if isinstance(index, tuple) else (index,)
        top.append(
            {
                "scope": "|".join(str(part) for part in scope),
                "value": float(value),
            }
        )
    return top


def _event_support(
    events: list[Any],
    config: DatasetConfig,
    *,
    classifications: tuple[str, ...],
    approved_event_ids: tuple[str, ...],
) -> tuple[list[Any], list[str]]:
    applicable = [
        event
        for event in events
        if event.classification in classifications
        and event.entity_type in config.explanation_entity_types
    ]
    approved = [
        event
        for event in applicable
        if event.classification in EXPLAINABLE_CLASSES and approved_event_ids
    ]
    return approved, [f"event:{event.entity_type}:{event.entity_id}" for event in approved]


def build_explanation_ledger(
    previous: pd.DataFrame,
    current: pd.DataFrame,
    pair: Any,
    config: DatasetConfig,
    events: list[Any] | None = None,
    temporal: Any | None = None,
    calendar: Any | None = None,
    approved_event_ids: tuple[str, ...] = (),
    new_periods: Sequence[int] | None = None,
    metric: str | None = None,
) -> ExplanationLedger:
    """Decompose the movement between two versions into one allocation each.

    ``new_periods`` restricts the new-period movement to an explicit period set
    (one call per appended period) while overlap contributions still describe
    the whole revision pair. ``metric`` selects the assessed measure; the
    resulting ledger is keyed by that measure.
    """
    week = config.week_column
    metric = metric or config.primary_metric
    keys = _entity_keys(config)
    missing = [
        key
        for key in (*keys, week, metric)
        if key not in previous.columns or key not in current.columns
    ]
    if missing:
        raise ValueError(f"ledger grain missing: {sorted(set(missing))}")
    events = list(events or ())
    overlap_weeks = set(pair.overlap_weeks)
    periods = set(
        pair.new_periods if new_periods is None else (int(value) for value in new_periods)
    )

    previous_overlap = previous.loc[previous[week].isin(overlap_weeks)]
    current_overlap = current.loc[current[week].isin(overlap_weeks)]
    merged = _overlap_aggregates(
        previous_overlap,
        current_overlap,
        keys,
        metric,
        config.unit_value_quantity,
        week,
    )
    delta_column = f"{metric}_delta"
    merged[delta_column] = (
        merged[f"{metric}_current"].fillna(0.0)
        - merged[f"{metric}_previous"].fillna(0.0)
    )
    overlap_delta = float(merged[delta_column].sum())
    entered = merged.loc[merged["_merge"] == "right_only"]
    exited = merged.loc[merged["_merge"] == "left_only"]
    matched = merged.loc[merged["_merge"] == "both"]

    contributions: list[LedgerContribution] = []
    approved, evidence_ids = _event_support(
        events,
        config,
        classifications=(NEW_BACKFILL, NEW_RECENT, EXTENDED),
        approved_event_ids=approved_event_ids,
    )
    entry_support = APPROVED_EVENT if approved else ARITHMETIC
    entry_approved = tuple(approved_event_ids) if approved else ()
    if len(entered):
        contributions.append(
            LedgerContribution(
                category=ENTITY_ENTRY,
                scope="overlap",
                metric=metric,
                value=float(entered[f"{metric}_current"].fillna(0.0).sum()),
                support=entry_support,
                evidence_ids=tuple(evidence_ids),
                approved_by=entry_approved,
                detail={
                    "entities": _top_entities(
                        entered, keys, f"{metric}_current", config.top_contributors
                    )
                },
            )
        )
    removed, removed_evidence = _event_support(
        events,
        config,
        classifications=(TRUNCATED, REMOVED),
        approved_event_ids=approved_event_ids,
    )
    exit_support = APPROVED_EVENT if removed else ARITHMETIC
    if len(exited):
        contributions.append(
            LedgerContribution(
                category=ENTITY_EXIT,
                scope="overlap",
                metric=metric,
                value=-float(exited[f"{metric}_previous"].fillna(0.0).sum()),
                support=exit_support,
                evidence_ids=tuple(removed_evidence),
                approved_by=tuple(approved_event_ids) if removed else (),
                detail={
                    "entities": _top_entities(
                        exited, keys, f"{metric}_previous", config.top_contributors
                    )
                },
            )
        )

    if len(matched):
        quantity_metric = config.unit_value_quantity
        decomposable_columns = {
            f"{quantity_metric}_previous",
            f"{quantity_metric}_current",
        }
        if (
            metric == config.unit_value_metric
            and quantity_metric
            and decomposable_columns.issubset(matched.columns)
        ):
            previous_units = matched[
                f"{quantity_metric}_previous"
            ].fillna(0.0).to_numpy(dtype=float)
            current_units = matched[
                f"{quantity_metric}_current"
            ].fillna(0.0).to_numpy(dtype=float)
            previous_dollars = matched[f"{metric}_previous"].fillna(0.0).to_numpy(
                dtype=float
            )
            current_dollars = matched[f"{metric}_current"].fillna(0.0).to_numpy(
                dtype=float
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                previous_price = np.where(
                    previous_units != 0.0, previous_dollars / previous_units, np.nan
                )
                current_price = np.where(
                    current_units != 0.0, current_dollars / current_units, np.nan
                )
            decomposable = np.isfinite(previous_price) & np.isfinite(current_price)
            quantity_effect = np.where(
                decomposable,
                (current_units - previous_units)
                * (current_price + previous_price)
                / 2.0,
                0.0,
            )
            unit_value_effect = np.where(
                decomposable,
                (current_price - previous_price)
                * (current_units + previous_units)
                / 2.0,
                0.0,
            )
            matched_delta = current_dollars - previous_dollars
            volume_effect = np.where(decomposable, 0.0, matched_delta)
            contributions.append(
                LedgerContribution(
                    category=MATCHED_QUANTITY,
                    scope="overlap",
                    metric=metric,
                    value=float(quantity_effect.sum()),
                    support=ARITHMETIC,
                    detail={
                        "entities": _top_entities(
                            matched.assign(__effect=quantity_effect),
                            keys,
                            "__effect",
                            config.top_contributors,
                        )
                    },
                )
            )
            contributions.append(
                LedgerContribution(
                    category=MATCHED_UNIT_VALUE,
                    scope="overlap",
                    metric=metric,
                    value=float(unit_value_effect.sum()),
                    support=ARITHMETIC,
                    detail={
                        "entities": _top_entities(
                            matched.assign(__effect=unit_value_effect),
                            keys,
                            "__effect",
                            config.top_contributors,
                        )
                    },
                )
            )
            if float(np.abs(volume_effect).sum()) > 1e-9:
                contributions.append(
                    LedgerContribution(
                        category=MATCHED_VOLUME,
                        scope="overlap",
                        metric=metric,
                        value=float(volume_effect.sum()),
                        support=ARITHMETIC,
                    )
                )
            expected_matched = float(
                matched_delta.sum()
            )
            allocated = (
                float(quantity_effect.sum())
                + float(unit_value_effect.sum())
                + float(volume_effect.sum())
            )
            matched_remainder = expected_matched - allocated
        else:
            contributions.append(
                LedgerContribution(
                    category=MATCHED_VOLUME,
                    scope="overlap",
                    metric=metric,
                    value=float(matched[delta_column].sum()),
                    support=ARITHMETIC,
                    detail={
                        "entities": _top_entities(
                            matched, keys, delta_column, config.top_contributors
                        )
                    },
                )
            )
            matched_remainder = 0.0
    else:
        matched_remainder = 0.0

    reclass_events = [
        event for event in events if event.classification == RECLASSIFIED
    ]
    if reclass_events:
        moved = float(
            sum(
                float(event.details.get("gross_moved", 0.0))
                for event in reclass_events
            )
        )
        contributions.append(
            LedgerContribution(
                category=CLASSIFICATION_CHANGE,
                scope="overlap",
                metric=metric,
                value=0.0,
                support=ARITHMETIC,
                evidence_ids=tuple(
                    f"event:{event.entity_type}:{event.entity_id}"
                    for event in reclass_events
                ),
                detail={
                    "gross_moved": moved,
                    "conservation": {
                        event.entity_id: float(
                            event.details.get("conservation_ratio", 1.0)
                        )
                        for event in reclass_events
                    },
                    "note": "classification moves are already allocated to entry/exit",
                },
            )
        )

    if abs(matched_remainder) > 1e-9:
        contributions.append(
            LedgerContribution(
                category=UNEXPLAINED,
                scope="overlap",
                metric=metric,
                value=float(matched_remainder),
                support="unknown",
                detail={"reason": "decomposition rounding remainder"},
            )
        )

    current_new = current.loc[current[week].isin(periods)]
    new_period_movement = float(current_new[metric].sum()) if len(current_new) else 0.0
    new_period_expected = 0.0
    new_period_entry = 0.0
    new_period_unexplained = new_period_movement
    expected_support = "unknown"
    expected_basis = "unavailable"
    if temporal is not None:
        national = next(
            (item for item in temporal.series if item.series_id == "national"), None
        )
        if national is not None:
            new_period_expected = float(national.forecast_median)
            new_period_entry = float(national.adjustment)
            new_period_unexplained = (
                new_period_movement - new_period_expected - new_period_entry
            )
            expected_support = STATISTICAL
            target = getattr(temporal, "target_week", None)
            if (
                calendar is not None
                and target is not None
                and any(calendar.event_windows(target).values())
            ):
                expected_basis = "calendar"
            else:
                expected_basis = "trend"
    contributions.append(
        LedgerContribution(
            category=NEW_PERIOD_EXPECTED,
            scope="new_period",
            metric=metric,
            value=new_period_expected,
            support=expected_support,
            detail={"basis": expected_basis},
        )
    )
    if abs(new_period_entry) > 1e-12:
        contributions.append(
            LedgerContribution(
                category=NEW_PERIOD_ENTITY_ENTRY,
                scope="new_period",
                metric=metric,
                value=new_period_entry,
                support=ARITHMETIC,
            )
        )
    contributions.append(
        LedgerContribution(
            category=NEW_PERIOD_UNEXPLAINED,
            scope="new_period",
            metric=metric,
            value=float(new_period_unexplained),
            support="unknown",
        )
    )

    ledger = ExplanationLedger(
        metric=metric,
        overlap_delta=overlap_delta,
        new_period_movement=new_period_movement,
        raw_movement=overlap_delta + new_period_movement,
        contributions=contributions,
        coverage={
            "matched_entities": int(len(matched[keys].drop_duplicates())),
            "entered_entities": int(len(entered[keys].drop_duplicates())),
            "exited_entities": int(len(exited[keys].drop_duplicates())),
            "overlap_weeks": len(overlap_weeks),
            "new_period_weeks": len(periods),
        },
        support_level=(
            "verified"
            if expected_support == STATISTICAL
            else ("arithmetic_only" if contributions else "unsupported")
        ),
    )
    allocated = float(sum(item.value for item in contributions))
    ledger.conservation = {
        "raw_movement": ledger.raw_movement,
        "contribution_sum": allocated,
        "difference": ledger.raw_movement - allocated,
    }
    return ledger
