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
    missing_entity_impact,
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
    approval_coverage: list[dict[str, Any]] = field(default_factory=list)
    # Latest-week absences explained by an approved closure: "type:id" -> approval.
    matched_absences: dict[str, str] = field(default_factory=dict)
    # Category moves explained by an approved entry: "from->to" -> approval.
    matched_moves: dict[str, str] = field(default_factory=dict)
    # Categories emptied or created by an approved move: "commodity:id" -> approval.
    move_consequences: dict[str, str] = field(default_factory=dict)
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
            "matched_absences": dict(self.matched_absences),
            "matched_moves": dict(self.matched_moves),
            "move_consequences": dict(self.move_consequences),
            "per_metric": dict(self.per_metric),
            "cross_metric_flags": list(self.cross_metric_flags),
            "conservation": dict(self.conservation),
        }


def _usable(item: dict[str, Any], config: DatasetConfig) -> bool:
    return bool(item.get("approved_by")) and item.get("confirmed") is True and item.get("dataset") == config.name


def _in_effect(item: dict[str, Any], week: int | None) -> bool:
    start, end = item.get("effective_from_week"), item.get("effective_to_week")
    return week is not None and type(start) is int and type(end) is int and start <= week <= end


def _match_expected(
    event: LifecycleEvent,
    registry: list[dict[str, Any]],
    config: DatasetConfig,
    latest_week: int | None = None,
) -> str | None:
    """The approved registry entry that explains ``event``, if exactly scoped.

    Historical changes (backfill, removal, truncation) must fall inside the
    entry's ``expected_history_start``..``expected_history_end``. A closure
    (``LATEST_WEEK_MISSING``) and a category move (``POSSIBLE_RECLASSIFICATION``)
    have no historical weeks; the refresh's latest week must fall inside the
    entry's ``effective_from_week``..``effective_to_week``. A move must also
    conserve the moved value and, when the entry lists ``product_ids``, move
    only those products.
    """
    for item in registry:
        if not _usable(item, config):
            continue
        expected_class = item.get("classification")
        if expected_class is None and item.get("event_type") in ("new_store_historical_backfill", "new_entity_historical_backfill"):
            expected_class = NEW_BACKFILL
        entity_ids = [str(value) for value in item.get("entity_ids", [])]
        if event.entity_type != str(item.get("entity_type")) or event.entity_id not in entity_ids:
            continue
        if expected_class == LATEST_MISSING:
            if LATEST_MISSING in (event.classifications or (event.classification,)) and _in_effect(item, latest_week):
                return str(item.get("event_id"))
            continue
        if expected_class != event.classification:
            continue
        if expected_class == RECLASSIFIED:
            products = {str(value) for value in event.details.get("products", [])}
            listed = item.get("product_ids")
            conserved = float(event.details.get("conservation_ratio", 0.0)) >= config.explained_fraction_threshold
            if _in_effect(item, latest_week) and conserved and (listed is None or products <= {str(v) for v in listed}):
                return str(item.get("event_id"))
            continue
        weeks = (*event.historical_weeks_added, *event.historical_weeks_removed)
        start, end = item.get("expected_history_start"), item.get("expected_history_end")
        if weeks and type(start) is int and type(end) is int and all(start <= w <= end for w in weeks):
            return str(item.get("event_id"))
    return None


def _move_consequences(
    structural: list[LifecycleEvent],
    registry: list[dict[str, Any]],
    config: DatasetConfig,
    latest_week: int | None,
) -> dict[str, str]:
    """Categories that approved moves emptied (removed) or created (backfilled).

    A move that takes every product out of a category also shows as that
    category's removal (and a move into a new category as its backfill). The
    event is the move's consequence only if its value is the value the
    approved moves carried, within the conservation tolerance.
    """
    tolerance = 1.0 - config.explained_fraction_threshold
    moved_out: dict[str, float] = {}
    moved_in: dict[str, float] = {}
    approval: dict[str, str] = {}
    for event in structural:
        if event.classification != RECLASSIFIED:
            continue
        event_id = _match_expected(event, registry, config, latest_week)
        if not event_id:
            continue
        source, target = str(event.details.get("from_commodity")), str(event.details.get("to_commodity"))
        moved_out[source] = moved_out.get(source, 0.0) + float(event.details.get("moved_previous", 0.0))
        moved_in[target] = moved_in.get(target, 0.0) + float(event.details.get("moved_current", 0.0))
        approval.setdefault(source, event_id)
        approval.setdefault(target, event_id)
    found: dict[str, str] = {}
    for event in structural:
        if event.entity_type != "commodity":
            continue
        if event.classification == REMOVED and event.entity_id in moved_out:
            value, moved = event.historical_value_removed, moved_out[event.entity_id]
        elif event.classification == NEW_BACKFILL and event.entity_id in moved_in:
            value, moved = event.historical_value_added, moved_in[event.entity_id]
        else:
            continue
        if abs(value - moved) <= tolerance * max(abs(value), abs(moved), 1e-9):
            found[f"commodity:{event.entity_id}"] = approval[event.entity_id]
    return found


def match_absences(
    events: list[LifecycleEvent],
    registry: list[dict[str, Any]],
    config: DatasetConfig,
    latest_week: int | None,
) -> dict[tuple[str, str], str]:
    """Latest-week absences an approved closure explains: (type, id) -> approval."""
    matched: dict[tuple[str, str], str] = {}
    for event in events:
        if LATEST_MISSING not in (event.classifications or (event.classification,)):
            continue
        event_id = _match_expected(event, registry, config, latest_week)
        if event_id:
            matched[(event.entity_type, event.entity_id)] = event_id
    return matched


def _contributors(
    overlap: pd.DataFrame,
    delta_column: str,
    config: DatasetConfig,
    entity_type: str | None = None,
) -> list[Contributor]:
    if delta_column not in overlap.columns:
        return []
    columns = list(dict(config.stage_keys).get(config.analysis_stage, config.entity_key_columns))
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


@dataclass(frozen=True)
class WeekRevision:
    """One overlap week's revision that no structural event explains."""

    week: int
    previous: float
    delta: float
    explained: float
    unexplained: float
    relative: float
    tolerance: float
    in_restatement_window: bool
    material: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def week_revisions(
    base_cube: pd.DataFrame,
    events: list[LifecycleEvent],
    config: DatasetConfig,
) -> list[WeekRevision]:
    """Per-week unexplained revision of the primary metric.

    Whole-history materiality cannot see a restatement confined to one week:
    it must exceed ``materiality_ratio`` of *all* overlap weeks. Each overlap
    week is therefore also judged on its own: its revision minus the part its
    structural events (backfills, extensions, truncations, removals) account
    for, as a fraction of that week's previous value. Weeks inside the
    declared ``restatement_weeks`` window (late-arriving data) use the looser
    ``restatement_tolerance``. Every week is returned; ``material`` marks the
    ones that escalate.
    """
    metric = config.primary_metric
    week = config.week_column
    delta_column, previous_column = f"{metric}_delta", f"{metric}_previous"
    if not len(base_cube) or delta_column not in base_cube.columns:
        return []
    overlap = base_cube.loc[base_cube["period"] == "overlap"]
    if not len(overlap):
        return []
    per_week = overlap.groupby(week)[[delta_column, previous_column]].sum()
    structural = [
        event
        for event in events
        if event.classification in EXPLAINABLE_CLASSES and event.classification != RECLASSIFIED
    ]
    explanation_type = next(
        (
            entity_type
            for entity_type in config.explanation_entity_types
            if any(event.entity_type == entity_type for event in structural)
        ),
        structural[0].entity_type if structural else None,
    )
    explained: dict[int, float] = {}
    for event in structural:
        if event.entity_type != explanation_type:
            continue
        for week_id, value in event.value_added_by_week.items():
            explained[int(week_id)] = explained.get(int(week_id), 0.0) + float(value)
        for week_id, value in event.value_removed_by_week.items():
            explained[int(week_id)] = explained.get(int(week_id), 0.0) - float(value)
    weeks = sorted(int(value) for value in per_week.index)
    window = set(weeks[-config.restatement_weeks:]) if config.restatement_weeks else set()
    revisions: list[WeekRevision] = []
    for week_id in weeks:
        delta = float(per_week.at[week_id, delta_column])
        previous = float(per_week.at[week_id, previous_column])
        accounted = explained.get(week_id, 0.0)
        unexplained = delta - accounted
        relative = abs(unexplained) / abs(previous) if abs(previous) > 1e-12 else (
            0.0 if abs(unexplained) <= 1e-9 else float("inf")
        )
        in_window = week_id in window
        tolerance = config.restatement_tolerance if in_window else config.week_revision_ratio
        revisions.append(
            WeekRevision(
                week=week_id,
                previous=previous,
                delta=delta,
                explained=accounted,
                unexplained=unexplained,
                relative=relative,
                tolerance=tolerance,
                in_restatement_window=in_window,
                material=relative > tolerance and abs(unexplained) > config.materiality_abs,
            )
        )
    return revisions


def explain_revision(
    base_cube: pd.DataFrame,
    events: list[LifecycleEvent],
    expected_events: list[dict[str, Any]],
    config: DatasetConfig,
    latest_week: int | None = None,
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
        column: float((overlap.loc[overlap[config.week_column] == overlap[config.week_column].max()]
                       if column in config.snapshot_metrics and len(overlap) else overlap)[f"{column}_delta"].sum())
        for column in metrics_present
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
    gross_delta = float(overlap[delta_column].abs().sum()) if delta_column in overlap else 0.0
    material = gross_delta > max(
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
    coverage: list[dict[str, Any]] = []
    moves: dict[str, str] = {}
    conservation: dict[str, float] = {}
    consequences = _move_consequences(structural, expected_events or [], config, latest_week)

    for event in structural:
        key = f"{event.entity_type}:{event.entity_id}"
        if key in consequences:
            # A category emptied or created by an approved move: the move's
            # approval covers it, and its value moved rather than changed.
            matched.append(consequences[key])
            coverage.append({"check": "historical_event", "scope": f"{key}:{event.classification}",
                             "approval_id": consequences[key], "weeks": []})
            explanations.append(f"moved_category:{event.entity_id}")
            continue
        # Every structural event must be registry-matched, whatever grain it
        # sits at; only the primary grain contributes to the explained sum.
        event_id = _match_expected(event, expected_events or [], config, latest_week)
        if event_id:
            matched.append(event_id)
            coverage.append({"check": "historical_event", "scope": f"{event.entity_type}:{event.entity_id}:{event.classification}",
                             "approval_id": event_id, "weeks": sorted(set((*event.historical_weeks_added, *event.historical_weeks_removed)))})
        else:
            unmatched.append(event)

        if event.classification == RECLASSIFIED:
            if event_id:
                moves[event.entity_id] = event_id
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

    absences = match_absences(events, expected_events or [], config, latest_week)
    for (entity_type, entity_id), event_id in sorted(absences.items()):
        matched.append(event_id)
        coverage.append({"check": "absence_event", "scope": f"{entity_type}:{entity_id}:{LATEST_MISSING}",
                         "approval_id": event_id, "weeks": [latest_week]})

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
    if material and not structural:
        explained_fraction = 0.0
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
        approval_coverage=coverage,
        matched_absences={f"{t}:{i}": a for (t, i), a in sorted(absences.items())},
        matched_moves=moves,
        move_consequences=dict(consequences),
        per_metric=per_metric,
        cross_metric_flags=_cross_metric_flags(
            overlap, per_metric, previous_total, config
        ),
        conservation=conservation,
    )


def like_for_like(
    previous: pd.DataFrame,
    current: pd.DataFrame,
    events: list[LifecycleEvent],
    attribution: AttributionResult,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Restate both versions so approved changes do not move the latest week.

    An approved closure removes the closed entities from every week, so the
    latest week is compared with the history of the entities still trading.
    An approved category move assigns the moved products' previous rows to
    their new category, so both categories are compared on the current
    mapping. Only changes the registry matched in this refresh are applied;
    with no approvals the frames are returned unchanged.
    """
    excluded: list[dict[str, str]] = []
    restated: list[dict[str, Any]] = []
    for key, approval in attribution.matched_absences.items():
        entity_type, entity_id = key.split(":", 1)
        column = f"{entity_type}_id"
        if column in previous.columns and column in current.columns:
            previous = previous.loc[previous[column].astype(str) != entity_id]
            current = current.loc[current[column].astype(str) != entity_id]
            excluded.append({"entity": key, "approval_id": approval})
    if {"product_id", "commodity_id"} <= set(previous.columns):
        for event in events:
            moved_by = attribution.matched_moves.get(event.entity_id) if event.classification == RECLASSIFIED else None
            if moved_by is None:
                continue
            products = {str(p) for p in event.details.get("products", [])}
            rows = previous["product_id"].astype(str).isin(products) & (
                previous["commodity_id"].astype(str) == str(event.details["from_commodity"])
            )
            previous = previous.copy()
            previous.loc[rows, "commodity_id"] = event.details["to_commodity"]
            restated.append({"move": event.entity_id, "products": len(products), "approval_id": moved_by})
    return previous, current, {"excluded": excluded, "restated": restated}


def classify_run(
    contract_status: str,
    events: list[LifecycleEvent],
    attribution: AttributionResult,
    config: DatasetConfig,
    reconstruction_score: float | None = None,
    absences: list[LifecycleEvent] | None = None,
) -> str:
    """Historical status. ``absences`` replaces ``events`` for the missing-entity
    check when approved closures restated the data (see ``like_for_like``)."""
    if contract_status == "DATA_CONTRACT_FAILURE":
        return "DATA_CONTRACT_FAILURE"
    unexplained_absences = [
        event for event in (events if absences is None else absences)
        if f"{event.entity_type}:{event.entity_id}" not in attribution.matched_absences
    ]
    if any(entry["material"] for entry in missing_entity_impact(unexplained_absences, config).values()):
        return "INVESTIGATE"
    if attribution.structural_events:
        all_matched = not attribution.unmatched_events
        reconstructed_enough = (
            reconstruction_score is not None
            and reconstruction_score >= config.reconstruction_score_threshold
        )
        if all(
            event.classification == RECLASSIFIED
            or f"{event.entity_type}:{event.entity_id}" in attribution.move_consequences
            for event in attribution.structural_events
        ):
            # A category move (and a category it empties or creates) relabels
            # value without changing it, so it explains none of the revision;
            # whatever revision remains is judged as if there were no
            # structural change.
            explained_enough = not _residual_escalates(attribution, config)
        else:
            explained_enough = (
                attribution.explained_fraction >= config.explained_fraction_threshold
                and not attribution.over_explained
            )
        if all_matched and explained_enough and reconstructed_enough:
            return "PASS_WITH_EXPLANATION"
        return "INVESTIGATE"
    return "INVESTIGATE" if _residual_escalates(attribution, config) else "PASS"


def _residual_escalates(attribution: AttributionResult, config: DatasetConfig) -> bool:
    if attribution.material and attribution.explained_fraction < config.explained_fraction_threshold:
        return True
    return not attribution.material and attribution.breadth > config.broad_recalculation_breadth
