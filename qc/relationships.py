"""Entity relationship graph (architecture section 12).

Known identifier relationships - superseded, remapped, merged, split, alias,
replaced - explain why an entity disappears while another appears. Confirmed
relationships are loaded from a store; unconfirmed candidates are detected
conservatively from correlated weekly series between a removed entity and a
newly appearing entity with comparable volume.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import DatasetConfig
from .jsonutil import dumps as json_dumps
from .versions import VersionPair

RELATIONSHIP_TYPES: tuple[str, ...] = (
    "superseded_by",
    "remapped_to",
    "merged_into",
    "split_into",
    "alias_of",
    "replaced_by",
)


@dataclass
class EntityRelationship:
    source_id: str
    target_id: str
    entity_type: str
    relationship: str
    effective_week: int | None = None
    confidence: float | None = None
    confirmed: bool = False
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EntityRelationship:
        relationship = str(data["relationship"])
        if relationship not in RELATIONSHIP_TYPES:
            raise ValueError(f"unknown relationship {relationship!r}")
        return cls(
            source_id=str(data["source_id"]),
            target_id=str(data["target_id"]),
            entity_type=str(data["entity_type"]),
            relationship=relationship,
            effective_week=(
                int(data["effective_week"])
                if data.get("effective_week") is not None
                else None
            ),
            confidence=(
                float(data["confidence"])
                if data.get("confidence") is not None
                else None
            ),
            confirmed=bool(data.get("confirmed", False)),
            evidence=dict(data.get("evidence", {})),
        )


class RelationshipStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def add(self, relationship: EntityRelationship) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as handle:
            handle.write(
                json_dumps(relationship.to_dict(), sort_keys=True) + "\n"
            )

    def load(self) -> list[EntityRelationship]:
        if not self.path.exists():
            return []
        return [
            EntityRelationship.from_dict(json.loads(line))
            for line in self.path.read_text().splitlines()
            if line.strip()
        ]

    def for_entity(
        self, entity_type: str, entity_ids: Sequence[str]
    ) -> list[EntityRelationship]:
        wanted = {str(entity) for entity in entity_ids}
        return [
            relationship
            for relationship in self.load()
            if relationship.entity_type == entity_type
            and (
                relationship.source_id in wanted
                or relationship.target_id in wanted
            )
        ]


def _correlation(left: Sequence[float], right: Sequence[float]) -> float:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return 0.0
    if float(np.std(a)) <= 1e-12 or float(np.std(b)) <= 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _fisher_p_value(correlation: float, n: int) -> float:
    """Two-sided p-value for a Pearson correlation via Fisher's z.

    Uses the normal approximation, which is adequate for the small candidate
    sets here and avoids a scipy dependency.
    """
    import math

    if n < 4:
        return 1.0
    if abs(correlation) >= 1.0:
        return 0.0
    z = math.atanh(correlation) * math.sqrt(n - 3)
    return math.erfc(abs(z) / math.sqrt(2.0))


def _benjamini_hochberg(p_values: Sequence[float], q: float = 0.05) -> list[bool]:
    from .conformal import benjamini_hochberg

    return benjamini_hochberg(p_values, q)[1]


def _weekly_series(
    frame: pd.DataFrame,
    entity_column: str,
    entity_id: str,
    week: str,
    metric: str,
    weeks: list[int],
) -> list[float]:
    subset = frame.loc[
        (frame[entity_column].astype(str) == entity_id)
        & (frame[week].isin(weeks))
    ]
    grouped = subset.groupby(week)[metric].sum()
    return [float(grouped.get(week_id, 0.0)) for week_id in weeks]


def detect_relationships(
    previous: pd.DataFrame,
    current: pd.DataFrame,
    pair: VersionPair,
    config: DatasetConfig,
) -> list[EntityRelationship]:
    """Conservative replacement candidates between removed and new entities.

    Every candidate pair is tested for correlation significance and the whole
    candidate set is corrected with Benjamini-Hochberg, so scanning many pairs
    cannot manufacture a replacement by chance.
    """
    week = config.week_column
    metric = config.primary_metric
    overlap = list(pair.overlap_weeks)
    low, high = config.relationship_ratio_bounds
    if len(overlap) < config.relationship_min_weeks:
        return []
    candidates: list[dict[str, Any]] = []

    for entity_column in config.entity_columns:
        if entity_column not in previous.columns or entity_column not in current.columns:
            continue
        entity_type = (
            entity_column[:-3] if entity_column.endswith("_id") else entity_column
        )
        previous_ids = {
            str(value)
            for value in previous[entity_column].dropna().unique()
        }
        current_ids = {
            str(value)
            for value in current[entity_column].dropna().unique()
        }
        removed = sorted(previous_ids - current_ids)
        added = sorted(current_ids - previous_ids)
        if not removed or not added:
            continue

        previous_series = {
            entity_id: _weekly_series(
                previous, entity_column, entity_id, week, metric, overlap
            )
            for entity_id in removed
        }
        current_series = {
            entity_id: _weekly_series(
                current, entity_column, entity_id, week, metric, overlap
            )
            for entity_id in added
        }

        for source_id, source_values in previous_series.items():
            source_total = sum(source_values)
            active_source = sum(1 for value in source_values if abs(value) > 0.0)
            if active_source < config.relationship_min_weeks:
                continue
            for target_id, target_values in current_series.items():
                active_target = sum(
                    1 for value in target_values if abs(value) > 0.0
                )
                if active_target < config.relationship_min_weeks:
                    continue
                correlation = _correlation(source_values, target_values)
                ratio = (
                    sum(target_values) / source_total
                    if abs(source_total) > 1e-12
                    else 0.0
                )
                if (
                    correlation >= config.relationship_correlation_threshold
                    and low <= ratio <= high
                ):
                    candidates.append(
                        {
                            "source_id": source_id,
                            "target_id": target_id,
                            "entity_type": entity_type,
                            "correlation": correlation,
                            "ratio": ratio,
                            "active_weeks": min(active_source, active_target),
                            "p_value": _fisher_p_value(correlation, len(overlap)),
                        }
                    )

    keep = _benjamini_hochberg([candidate["p_value"] for candidate in candidates])
    relationships: list[EntityRelationship] = []
    for candidate, significant in zip(candidates, keep):
        if not significant:
            continue
        relationships.append(
            EntityRelationship(
                source_id=candidate["source_id"],
                target_id=candidate["target_id"],
                entity_type=candidate["entity_type"],
                relationship="replaced_by",
                effective_week=pair.current_max_week,
                confidence=candidate["correlation"],
                confirmed=False,
                evidence={
                    "weeks": len(overlap),
                    "active_weeks": candidate["active_weeks"],
                    "correlation": candidate["correlation"],
                    "volume_ratio": candidate["ratio"],
                    "p_value": candidate["p_value"],
                    "multiple_comparison": "benjamini_hochberg",
                },
            )
        )
    return relationships
