from __future__ import annotations

import pytest

from qc.config import DatasetConfig
from qc.relationships import (
    EntityRelationship,
    RelationshipStore,
    detect_relationships,
)
from qc.versions import build_version_pair

CONFIG = DatasetConfig()


def _frame(rows):
    import pandas as pd

    return pd.DataFrame(rows)


def _pair(previous, current):
    return build_version_pair(
        previous, current, "V1", "V2", "warehouse", "warehouse", CONFIG
    )


def _series(entity, values):
    return [
        {"week": week, "store_id": entity, "dollar": value, "units": 1}
        for week, value in values.items()
    ]


def test_detect_replacement_candidate():
    previous = _frame(
        _series("A", {week: float(week) for week in range(1, 7)})
        + _series("B", {week: 100.0 for week in range(1, 7)})
    )
    current = _frame(
        _series("B", {week: 100.0 for week in range(1, 7)})
        + _series("C", {week: float(week) for week in range(1, 7)})
    )
    candidates = detect_relationships(previous, current, _pair(previous, current), CONFIG)
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.entity_type == "store"
    assert candidate.source_id == "A"
    assert candidate.target_id == "C"
    assert candidate.relationship == "replaced_by"
    assert candidate.confidence == pytest.approx(1.0)
    assert candidate.confirmed is False


def test_detect_ignores_uncorrelated_new_entity():
    previous = _frame(
        _series("A", {week: float(week) for week in range(1, 7)})
    )
    current = _frame(
        _series("C", {week: float((week * 7) % 3) for week in range(1, 7)})
    )
    assert detect_relationships(previous, current, _pair(previous, current), CONFIG) == []


def test_relationship_store_roundtrip_and_lookup(tmp_path):
    store = RelationshipStore(tmp_path / "relationships.jsonl")
    relationship = EntityRelationship(
        source_id="A",
        target_id="C",
        entity_type="store",
        relationship="replaced_by",
        effective_week=30,
        confidence=0.95,
        confirmed=True,
        evidence={"weeks": 6},
    )
    store.add(relationship)
    loaded = store.load()
    assert loaded == [relationship]
    assert store.for_entity("store", ["A"]) == [relationship]
    assert store.for_entity("store", ["Z"]) == []


def test_unknown_relationship_type_rejected():
    with pytest.raises(ValueError):
        EntityRelationship.from_dict(
            {
                "source_id": "A",
                "target_id": "C",
                "entity_type": "store",
                "relationship": "teleported_to",
            }
        )
