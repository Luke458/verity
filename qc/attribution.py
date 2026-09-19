"""Attribution: contributors, explained delta, unexplained residual and status.

The residual is the part of the historical revision that no structural event
explains. A revision is only reported as ``PASS_WITH_EXPLANATION`` when the
explaining events match the expected-event registry; unregistered structural
changes remain ``INVESTIGATE`` by design.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from .config import DatasetConfig
from .lifecycle import (
    EXPLAINABLE_CLASSES,
    EXTENDED,
    LATEST_MISSING,
    NEW_BACKFILL,
    NEW_RECENT,
    RECLASSIFIED,
    REMOVED,
    TRUNCATED,
    LifecycleEvent,
    _entity_type,
)


@dataclass(frozen=True)
class Contributor:
    entity_type: str
    entity_id: str
    delta: float
    abs_delta: float


@dataclass
class AttributionResult:
    raw_delta: float = 0.0
    explained_delta: float = 0.0
    explained_added: float = 0.0
    explained_removed: float = 0.0
    reclassified_moved: float = 0.0
    unexplained_delta: float = 0.0
    explained_fraction: float = 1.0
    previous_total: float = 0.0
    material: bool = False
    breadth: float = 0.0
    over_explained: bool = False
    offsetting: bool = False
    contributors: list[Contributor] = field(default_factory=list)
    explanations: list[str] = field(default_factory=list)
    structural_events: list[LifecycleEvent] = field(default_factory=list)
    unmatched_events: list[LifecycleEvent] = field(default_factory=list)
    matched_event_ids: list[str] = field(default_factory=list)
    per_metric: dict[str, float] = field(default_factory=dict)
    cross_metric_flags: list[str] = field(default_factory=list)
    conservation: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_delta": self.raw_delta,
            "explained_delta": self.explained_delta,
            "explained_added": self.explained_added,
            "explained_removed": self.explained_removed,
            "reclassified_moved": self.reclassified_moved,
            "unexplained_delta": self.unexplained_delta,
            "explained_fraction": self.explained_fraction,
            "previous_total": self.previous_total,
            "material": self.material,
            "breadth": self.breadth,
            "over_explained": self.over_explained,
            "offsetting": self.offsetting,
            "contributors": [asdict(contributor) for contributor in self.contributors],
            "explanations": list(self.explanations),
            "structural_events": [event.to_dict() for event in self.structural_events],
            "unmatched_events": [event.to_dict() for event in self.unmatched_events],
            "matched_event_ids": list(self.matched_event_ids),
            "per_metric": dict(self.per_metric),
            "cross_metric_flags": list(self.cross_metric_flags),
            "conservation": dict(self.conservation),
        }


def _match_expected(
    event: LifecycleEvent, registry: list[dict[str, Any]]
) -> str | None:
    for item in registry:
        entity_ids = [str(value) for value in item.get("entity_ids", [])]
        if event.entity_type == str(item.get("entity_type")) and event.entity_id in entity_ids:
            return str(item.get("event_id"))
    return None


def _contributors(
    overlap: pd.DataFrame,
    delta_column: str,
    config: DatasetConfig,
    entity_type: str | None = None,
) -> list[Contributor]:
    if delta_column not in overlap.columns:
        return []
    columns = list(config.entity_key_columns)
    if entity_type is not None:
        matching = [
            column for column in columns if _entity_type(column) == entity_type
        ]
        if matching:
            columns = matching
    contributors: list[Contributor] = []
    for column in columns:
        if column not in overlap.columns:
            continue
        grouped = overlap.groupby(column, dropna=False)[delta_column].sum()
        for entity_id, value in grouped.items():
            delta = float(value)
            contributors.append(
                Contributor(_entity_type(column), str(entity_id), delta, abs(delta))
            )
    contributors.sort(key=lambda contributor: contributor.abs_delta, reverse=True)
    return contributors[: config.top_contributors]


def _cross_metric_flags(
    overlap: pd.DataFrame,
    per_metric: dict[str, float],
    previous_total: float,
    config: DatasetConfig,
) -> list[str]:
    flags: list[str] = []
    primary = config.primary_metric
    if primary not in per_metric or "units" not in per_metric:
        return flags
    primary_rate = abs(per_metric[primary]) / abs(previous_total) if previous_total else 0.0
    units_previous_column = "units_previous"
    units_previous = (
        float(overlap[units_previous_column].sum())
        if units_previous_column in overlap.columns
        else 0.0
    )
    units_rate = abs(per_metric["units"]) / abs(units_previous) if units_previous else 0.0
    if primary_rate > config.materiality_ratio and units_rate < 0.1 * primary_rate:
        flags.append(f"{primary}_change_without_units")
    return flags


def explain_revision(
    base_cube: pd.DataFrame,
    events: list[LifecycleEvent],
    expected_events: list[dict[str, Any]],
    config: DatasetConfig,
) -> AttributionResult:
    metric = config.primary_metric
    delta_column = f"{metric}_delta"
    previous_column = f"{metric}_previous"
    overlap = (
        base_cube.loc[base_cube["period"] == "overlap"]
        if len(base_cube)
        else base_cube
    )
    metrics_present = [
        column
        for column in config.metric_columns
        if f"{column}_delta" in overlap.columns
    ]

    if delta_column in overlap.columns and len(overlap):
        raw_delta = float(overlap[delta_column].sum())
        previous_total = float(overlap[previous_column].sum())
    else:
        raw_delta = 0.0
        previous_total = 0.0
    per_metric = {
        column: float(overlap[f"{column}_delta"].sum()) for column in metrics_present
    }
    if metrics_present and len(overlap):
        changed = (
            overlap[[f"{column}_delta" for column in metrics_present]]
            .abs()
            .sum(axis=1)
            > 1e-9
        )
        breadth = float(changed.mean())
    else:
        breadth = 0.0
    material = abs(raw_delta) > max(
        config.materiality_abs, config.materiality_ratio * abs(previous_total)
    )

    structural = [event for event in events if event.classification in EXPLAINABLE_CLASSES]
    explanation_type = next(
        (
            entity_type
            for entity_type in config.explanation_entity_types
            if any(event.entity_type == entity_type for event in structural)
        ),
        structural[0].entity_type if structural else None,
    )
    explained_added = 0.0
    explained_removed = 0.0
    reclassified_moved = 0.0
    explanations: list[str] = []
    unmatched: list[LifecycleEvent] = []
    matched: list[str] = []
    conservation: dict[str, float] = {}

    for event in structural:
        # Every structural event must be registry-matched, whatever grain it
        # sits at; only the primary grain contributes to the explained sum.
        event_id = _match_expected(event, expected_events or [])
        if event_id:
            matched.append(event_id)
        else:
            unmatched.append(event)

        if event.classification == RECLASSIFIED:
            conservation[event.entity_id] = float(
                event.details.get("conservation_ratio", 1.0)
            )
            reclassified_moved += float(event.details.get("gross_moved", 0.0))
            explanations.append(f"possible_reclassification:{event.entity_id}")
            continue

        if event.entity_type != explanation_type:
            continue
        if event.classification in (NEW_BACKFILL, NEW_RECENT, EXTENDED):
            explained_added += event.historical_value_added
            explanations.append(
                f"{event.classification.lower()}:{event.entity_type}:{event.entity_id}"
            )
        elif event.classification in (TRUNCATED, REMOVED):
            explained_removed += event.historical_value_removed
            explanations.append(
                f"{event.classification.lower()}:{event.entity_type}:{event.entity_id}"
            )

    explained = explained_added - explained_removed
    unexplained = raw_delta - explained
    epsilon = 1e-12
    if abs(raw_delta) > epsilon:
        if explained * raw_delta < 0:
            explained_fraction = 0.0
        else:
            explained_fraction = min(1.0, abs(explained) / abs(raw_delta))
    else:
        # A near-zero revision can only be called explained when the events
        # themselves net to near-zero.
        explained_fraction = 1.0 if abs(explained) <= epsilon else 0.0
    over_explained = (
        abs(explained) > abs(raw_delta) + max(epsilon, 1e-6 * abs(raw_delta))
        and explained * raw_delta > 0
    )
    offsetting = (
        min(explained_added, explained_removed) > 0.1 * max(abs(raw_delta), epsilon)
        or (abs(raw_delta) <= epsilon and explained_added + explained_removed > epsilon)
    )

    return AttributionResult(
        raw_delta=raw_delta,
        explained_delta=explained,
        explained_added=explained_added,
        explained_removed=explained_removed,
        reclassified_moved=reclassified_moved,
        unexplained_delta=unexplained,
        explained_fraction=explained_fraction,
        previous_total=previous_total,
        material=material,
        breadth=breadth,
        over_explained=over_explained,
        offsetting=offsetting,
        contributors=_contributors(overlap, delta_column, config, explanation_type),
        explanations=explanations,
        structural_events=structural,
        unmatched_events=unmatched,
        matched_event_ids=matched,
        per_metric=per_metric,
        cross_metric_flags=_cross_metric_flags(
            overlap, per_metric, previous_total, config
        ),
        conservation=conservation,
    )


def classify_run(
    contract_status: str,
    events: list[LifecycleEvent],
    attribution: AttributionResult,
    config: DatasetConfig,
    reconstruction_score: float | None = None,
) -> str:
    if contract_status == "DATA_CONTRACT_FAILURE":
        return "DATA_CONTRACT_FAILURE"
    if any(event.classification == LATEST_MISSING for event in events):
        return "INVESTIGATE"
    if attribution.structural_events:
        explained_enough = (
            attribution.explained_fraction >= config.explained_fraction_threshold
            and not attribution.over_explained
        )
        all_matched = bool(attribution.matched_event_ids) and not attribution.unmatched_events
        reconstructed_enough = (
            reconstruction_score is not None
            and reconstruction_score >= config.reconstruction_score_threshold
        )
        if all_matched and explained_enough and reconstructed_enough:
            return "PASS_WITH_EXPLANATION"
        return "INVESTIGATE"
    if attribution.material and attribution.explained_fraction < config.explained_fraction_threshold:
        return "INVESTIGATE"
    if not attribution.material and attribution.breadth > config.broad_recalculation_breadth:
        return "INVESTIGATE"
    return "PASS"
