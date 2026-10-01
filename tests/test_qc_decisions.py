from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from qc.decisions import SEVERITY_VALUES, RuleDecisionProvider, field_index
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
    "recalculation": ("HISTORICAL_CORRECTION", True),
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
    # A single fault yields exactly its own cause in the set.
    assert result.decisions.get("likely_causes").value == ([] if cause == "UNKNOWN" else [cause])
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


def test_cause_set_names_both_faults_of_a_pair(tmp_path):
    built = build_scenario(
        suite_config("tiny"),
        0,
        tmp_path,
        "missing_stores",
        ("source", "coded", "warehouse", "report"),
        magnitude=0.2,
        also=("coding_error",),
    )
    result = run_qc(ScenarioSource(built.directory), "V0002", "V0001")
    assert result.decisions.get("likely_cause").value == "MISSING_STORES"
    assert set(result.decisions.get("likely_causes").value) == {"MISSING_STORES", "CODING"}


def _stub(first_divergence, events=(), status="PASS", attribution=None, week_revisions=()):
    return SimpleNamespace(
        run_id="stub",
        status=status,
        contracts=None,
        events=list(events),
        attribution=attribution,
        lineage=SimpleNamespace(first_divergence=first_divergence),
        temporal=None,
        relationships=[],
        week_revisions=list(week_revisions),
    )


def _remap():
    return SimpleNamespace(
        classification="POSSIBLE_RECLASSIFICATION", classifications=(), entity_type="product", entity_id="P1"
    )


@pytest.mark.parametrize(("stage", "cause"), [("coded", "CODING"), ("warehouse", "WAREHOUSE")])
def test_immaterial_lineage_names_the_stage_without_review(stage, cause):
    decisions = RuleDecisionProvider().decide(_stub(stage))
    assert decisions.get("likely_cause").value == cause
    assert decisions.get("likely_causes").value == [cause]
    assert decisions.get("severity").value == "LOW"
    # The label never escalates: review still follows the (PASS) status.
    assert decisions.requires_investigation is False


def test_immaterial_lineage_defers_to_a_structural_explanation():
    decisions = RuleDecisionProvider().decide(_stub("coded", [_remap()], status="INVESTIGATE"))
    assert decisions.get("likely_causes").value == ["RECLASSIFICATION"]


def test_source_divergence_alone_names_nothing():
    decisions = RuleDecisionProvider().decide(_stub("source"))
    assert decisions.get("likely_cause").value == "UNKNOWN"
    assert decisions.get("likely_causes").value == []


def test_history_rewriting_event_explains_source_revisions():
    unexplained = SimpleNamespace(
        previous_total=100.0, unexplained_delta=5.0, material=True,
        explained_fraction=0.5, breadth=0.0, unmatched_events=[],
    )
    restated = [SimpleNamespace(week=1, relative=0.2, material=True)]
    provider = RuleDecisionProvider()
    for stub in (
        _stub("source", [_remap()], "INVESTIGATE", attribution=unexplained),
        _stub("coded", [_remap()], "INVESTIGATE", week_revisions=restated),
    ):
        assert provider.decide(stub).get("likely_causes").value == ["RECLASSIFICATION"]
    # Without the event, the same evidence is a historical correction.
    alone = provider.decide(_stub("source", (), "INVESTIGATE", attribution=unexplained))
    assert alone.get("likely_causes").value == ["HISTORICAL_CORRECTION"]


def test_category_emptied_by_a_remap_is_not_a_separate_cause():
    emptied = SimpleNamespace(
        classification="ENTITY_REMOVED", classifications=("ENTITY_REMOVED",),
        entity_type="commodity", entity_id="C01",
    )
    remap = SimpleNamespace(
        classification="POSSIBLE_RECLASSIFICATION", classifications=(),
        entity_type="commodity", entity_id="C01->C07",
    )
    decisions = RuleDecisionProvider().decide(_stub("coded", [remap, emptied], "INVESTIGATE"))
    assert decisions.get("likely_causes").value == ["RECLASSIFICATION"]
