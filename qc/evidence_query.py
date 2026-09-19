"""Bounded evidence queries for the investigation agent.

An allowlisted, read-only view over a completed run's evidence. Queries never
touch source tables and always report ``total_rows`` and ``truncated`` so a
consumer can tell when it is looking at a limited answer. Rows are collected
lazily with the limit applied during iteration, the limit is capped, and
unknown filter keys raise rather than silently matching nothing.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

ALLOWED_QUERIES: tuple[str, ...] = (
    "lifecycle_changes",
    "historical_residuals",
    "temporal_anomalies",
    "first_divergence",
    "contract_failures",
    "contributors",
    "relationships",
)
MAX_LIMIT = 500

FILTER_KEYS: dict[str, frozenset[str]] = {
    "lifecycle_changes": frozenset(
        {"entity_type", "entity_id", "classification"}
    ),
    "historical_residuals": frozenset({"entity_type", "entity_id"}),
    "temporal_anomalies": frozenset({"series_id"}),
    "first_divergence": frozenset({"stage"}),
    "contract_failures": frozenset({"name"}),
    "contributors": frozenset({"entity_type", "entity_id"}),
    "relationships": frozenset(
        {"source_id", "target_id", "entity_type", "relationship"}
    ),
}


@dataclass
class EvidenceQueryResult:
    query: str
    rows: list[dict[str, Any]]
    total_rows: int
    truncated: bool
    filters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "rows": self.rows,
            "total_rows": self.total_rows,
            "truncated": self.truncated,
            "filters": self.filters,
        }


def _lifecycle_changes(result: Any) -> Iterator[dict[str, Any]]:
    for event in getattr(result, "events", ()) or ():
        yield {
            "entity_type": event.entity_type,
            "entity_id": event.entity_id,
            "classification": event.classification,
            "weeks_added": list(event.historical_weeks_added),
            "weeks_removed": list(event.historical_weeks_removed),
        }


def _historical_residuals(result: Any) -> Iterator[dict[str, Any]]:
    attribution = getattr(result, "attribution", None)
    if attribution is None:
        return
    yield {
        "entity_type": "total",
        "entity_id": "national",
        "delta": attribution.raw_delta,
        "explained_delta": attribution.explained_delta,
        "unexplained_delta": attribution.unexplained_delta,
        "explained_fraction": attribution.explained_fraction,
    }
    for contributor in attribution.contributors:
        yield {
            "entity_type": contributor.entity_type,
            "entity_id": contributor.entity_id,
            "delta": contributor.delta,
        }


def _temporal_anomalies(result: Any) -> Iterator[dict[str, Any]]:
    temporal = getattr(result, "temporal", None)
    if temporal is None:
        return
    for item in temporal.series:
        if not item.anomaly:
            continue
        yield {
            "series_id": item.series_id,
            "week": item.target_week,
            "actual": item.actual,
            "adjusted_actual": item.adjusted_actual,
            "forecast_median": item.forecast_median,
            "relative_residual": item.relative_residual,
            "calibrated_percentile": item.calibrated_percentile,
            "flags": item.flags,
        }


def _first_divergence(result: Any) -> Iterator[dict[str, Any]]:
    lineage = getattr(result, "lineage", None)
    if lineage is None:
        return
    for entry in lineage.divergences:
        yield {
            "stage": entry["stage"],
            "row_count_delta": entry["row_count_delta"],
            "relative_divergence": entry["relative_divergence"],
            "is_first": entry["stage"] == lineage.first_divergence,
        }


def _contract_failures(result: Any) -> Iterator[dict[str, Any]]:
    contracts = getattr(result, "contracts", None)
    if contracts is None:
        return
    for check in contracts.failed:
        yield {"name": check.name, "detail": check.detail}


def _contributors(result: Any) -> Iterator[dict[str, Any]]:
    attribution = getattr(result, "attribution", None)
    if attribution is None:
        return
    for contributor in attribution.contributors:
        yield {
            "entity_type": contributor.entity_type,
            "entity_id": contributor.entity_id,
            "delta": contributor.delta,
            "abs_delta": contributor.abs_delta,
        }


def _relationships(result: Any) -> Iterator[dict[str, Any]]:
    for relationship in getattr(result, "relationships", ()) or ():
        yield relationship.to_dict()


_BUILDERS: dict[str, Callable[[Any], Iterable[dict[str, Any]]]] = {
    "lifecycle_changes": _lifecycle_changes,
    "historical_residuals": _historical_residuals,
    "temporal_anomalies": _temporal_anomalies,
    "first_divergence": _first_divergence,
    "contract_failures": _contract_failures,
    "contributors": _contributors,
    "relationships": _relationships,
}


def _collect(
    rows: Iterable[dict[str, Any]], limit: int, filters: dict[str, Any]
) -> tuple[list[dict[str, Any]], int]:
    collected: list[dict[str, Any]] = []
    total = 0
    for row in rows:
        if any(str(row.get(key)) != str(value) for key, value in filters.items()):
            continue
        total += 1
        if len(collected) < limit:
            collected.append(row)
    return collected, total


def query_evidence(
    result: Any,
    query: str,
    limit: int = 50,
    **filters: Any,
) -> EvidenceQueryResult:
    if limit < 1:
        raise ValueError("limit must be >= 1")
    if query not in _BUILDERS:
        raise ValueError(
            f"unknown evidence query {query!r}; allowed: {sorted(_BUILDERS)}"
        )
    limit = min(limit, MAX_LIMIT)
    unknown_filters = set(filters) - FILTER_KEYS[query]
    if unknown_filters:
        raise ValueError(
            f"unknown filter keys for {query!r}: {sorted(unknown_filters)}; "
            f"allowed: {sorted(FILTER_KEYS[query])}"
        )
    rows, total = _collect(_BUILDERS[query](result), limit, dict(filters))
    return EvidenceQueryResult(
        query=query,
        rows=rows,
        total_rows=total,
        truncated=total > limit,
        filters=dict(filters),
    )
