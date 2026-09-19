from __future__ import annotations

import json

import pytest

from qc.decisions import (
    FEATURE_VERSION,
    SEVERITY_VALUES,
    FeatureEncoder,
    feature_version,
    field_index,
)
from qc.labels import CAUSE_BY_ORACLE
from qc.run import run_qc
from qcgen.config import suite_config
from qcgen.scenarios import build_scenario
from qcgen.sources import ScenarioSource

FAMILIES = (
    "missing_stores",
    "new_store_backfill",
    "expected_event",
    "history_truncation",
    "commodity_remap",
    "coding_error",
    "warehouse_transform_error",
    "recalculation",
    "schema_failure",
    "null_duplicate_storm",
    "market_movement",
    "missing_products",
    "entity_merge",
)

EXPECTED = {
    "missing_stores": ("MISSING_STORES", True),
    "new_store_backfill": ("BACKFILL", True),
    "expected_event": ("BACKFILL", False),
    "history_truncation": ("HISTORICAL_CORRECTION", True),
    "commodity_remap": ("RECLASSIFICATION", True),
    "coding_error": ("CODING", True),
    "warehouse_transform_error": ("WAREHOUSE", True),
    "recalculation": ("HISTORICAL_CORRECTION", True),
    "schema_failure": ("SCHEMA_FAILURE", True),
    "null_duplicate_storm": ("SCHEMA_FAILURE", True),
    "market_movement": ("MARKET_MOVEMENT", False),
    "missing_products": ("MISSING_PRODUCTS", True),
    "entity_merge": ("ENTITY_MERGE", True),
}


@pytest.fixture(scope="module")
def family_results(tmp_path_factory):
    config = suite_config("tiny")
    root = tmp_path_factory.mktemp("decisions")
    results = {}
    for index, family in enumerate(FAMILIES):
        built = build_scenario(
            config,
            index,
            root,
            family,
            ("source", "coded", "warehouse", "report"),
        )
        manifest = built.manifest
        results[family] = run_qc(
            ScenarioSource(built.directory),
            "V0002",
            "V0001",
            expected_events=manifest.get("expected_events", []),
        )
    return results


def test_feature_encoder_shape_and_version():
    encoder = FeatureEncoder()
    assert feature_version() == FEATURE_VERSION
    names = encoder.feature_names
    assert len(names) == len(set(names))
    assert names[0] == "contract_failed"


@pytest.mark.parametrize("family", FAMILIES)
def test_rule_decision_matches_family(family, family_results):
    result = family_results[family]
    assert result.decisions is not None
    cause, requires = EXPECTED[family]
    assert result.decisions.get("likely_cause").value == cause
    assert result.decisions.requires_investigation is requires

    encoder = FeatureEncoder()
    vector = encoder.encode(result)
    assert vector.shape == (len(encoder.feature_names),)
    assert all(isinstance(value, float) for value in vector.tolist())

    payload = result.decisions.to_dict()
    json.dumps(payload)


def test_severity_is_ordered_score(family_results):
    severity = family_results["missing_stores"].decisions.get("severity")
    assert field_index()["severity"].ordered is True
    assert severity.kind == "score"
    assert severity.value in SEVERITY_VALUES
    assert severity.index is not None
    assert 0.0 <= severity.index <= len(SEVERITY_VALUES) - 1
    assert sum(severity.probabilities.values()) == pytest.approx(1.0)
    assert set(severity.probabilities) == set(SEVERITY_VALUES)


def test_rule_decisions_carry_evidence(family_results):
    result = family_results["missing_stores"]
    cause = result.decisions.get("likely_cause")
    assert cause.evidence
    assert any("store" in item for item in cause.evidence)
    assert cause.probability_kind == "heuristic"
    probabilities = cause.probabilities
    assert probabilities[cause.value] == max(probabilities.values())


def test_oracle_cause_vocabulary_is_complete():
    from qc.decisions import CAUSE_VALUES

    for mapped in CAUSE_BY_ORACLE.values():
        assert mapped in CAUSE_VALUES
