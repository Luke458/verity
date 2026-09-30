from __future__ import annotations

import json

import pytest

from qc.decisions import SEVERITY_VALUES, field_index
from qc.registry import StaticRegistry
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
    "clean",
)

EXPECTED = {
    "missing_stores": ("MISSING_STORES", True),
    "new_store_backfill": ("BACKFILL", True),
    # The registered backfill is explained; nothing else requires review.
    "expected_event": ("BACKFILL", False),
    "history_truncation": ("HISTORICAL_CORRECTION", True),
    "commodity_remap": ("RECLASSIFICATION", True),
    "coding_error": ("CODING", True),
    "warehouse_transform_error": ("WAREHOUSE", True),
    "recalculation": ("SOURCE_INGESTION", True),
    "schema_failure": ("SCHEMA_FAILURE", True),
    "null_duplicate_storm": ("SCHEMA_FAILURE", True),
    "market_movement": ("UNKNOWN", True),
    "missing_products": ("MISSING_PRODUCTS", True),
    "entity_merge": ("ENTITY_MERGE", True),
    "clean": ("UNKNOWN", False),
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
        results[family] = run_qc(
            ScenarioSource(built.directory),
            "V0002",
            "V0001",
            registry=StaticRegistry(
                list(built.oracle.get("expected_events", []))
            ),
        )
    return results


@pytest.mark.parametrize("family", FAMILIES)
def test_rule_decision_matches_family(family, family_results):
    result = family_results[family]
    assert result.decisions is not None
    cause, requires = EXPECTED[family]
    assert result.decisions.get("likely_cause").value == cause
    assert result.decisions.requires_investigation is requires
    # Review follows the policy status; a label can neither clear nor escalate.
    assert requires is (result.status not in ("PASS", "PASS_WITH_EXPLANATION"))

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
