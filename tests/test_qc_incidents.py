from __future__ import annotations

import numpy as np
import pytest

from qc.decisions import FeatureEncoder, RuleDecisionProvider
from qc.incidents import (
    IncidentRecord,
    IncidentStore,
    build_incident_record,
    symptom_tags_for,
)
from qc.run import run_qc
from qcgen.config import suite_config
from qcgen.scenarios import build_scenario
from qcgen.sources import ScenarioSource


def _record(identifier: str, features: list[float], tags: list[str]) -> IncidentRecord:
    return IncidentRecord(
        incident_id=identifier,
        run_id=f"run-{identifier}",
        root_cause="MISSING_STORES",
        likely_origin="SOURCE",
        resolution="Supplier extract omitted stores.",
        analyst_summary="Similar store deficit.",
        symptom_tags=tags,
        features=features,
        feature_version=1,
        confirmed=True,
    )


def test_incident_store_retrieves_nearest(tmp_path):
    store = IncidentStore(tmp_path / "incidents.jsonl")
    store.add(_record("near", [1.0, 0.1, 0.0], ["MISSING_STORES"]))
    store.add(_record("far", [0.0, 0.0, 1.0], ["WAREHOUSE"]))
    hits = store.retrieve([1.0, 0.0, 0.0], k=2, tags=["MISSING_STORES"])
    assert [hit[0].incident_id for hit in hits] == ["near", "far"]
    assert hits[0][1] > hits[1][1]


def test_incident_store_ignores_shape_mismatch(tmp_path):
    store = IncidentStore(tmp_path / "incidents.jsonl")
    store.add(_record("mismatch", [1.0, 0.0], ["MISSING_STORES"]))
    assert store.retrieve([1.0, 0.0, 0.0]) == []


def test_build_incident_record_from_run(tmp_path):
    built = build_scenario(
        suite_config("tiny"),
        0,
        tmp_path,
        "missing_stores",
        ("source", "coded", "warehouse", "report"),
    )
    result = run_qc(ScenarioSource(built.directory), "V0002", "V0001")
    decisions = result.decisions
    tags = symptom_tags_for(result, decisions)
    assert "MISSING_STORES" in tags

    record = build_incident_record(
        result,
        decisions,
        feedback={"resolution": "Fixed supplier extract.", "confirmed": True},
    )
    assert record.confirmed is True
    assert record.root_cause == "MISSING_STORES"
    assert len(record.features) == len(FeatureEncoder().feature_names)
    assert any(entity.startswith("store:") for entity in record.affected_entities)
