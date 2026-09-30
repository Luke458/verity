from __future__ import annotations

import json

import pytest

from qc.cli import _json_default
from qc.registry import StaticRegistry
from qc.run import run_qc
from qcgen.config import suite_config
from qcgen.scenarios import build_scenario
from qcgen.sources import ScenarioSource
from qcgen.spec import FAMILY_SPECS

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

EXPECTATIONS = {
    "missing_stores": {
        "status": "INVESTIGATE",
        "reason_prefix": "latest_week_missing",
        "classification": "LATEST_WEEK_MISSING",
        "latest_week_anomaly": True,
    },
    "new_store_backfill": {
        "status": "INVESTIGATE",
        "classification": "NEW_ENTITY_HISTORICAL_BACKFILL",
        "fraction_min": 0.99,
    },
    "expected_event": {
        # The registered event explains the revision and nothing else moved.
        "status": "PASS_WITH_EXPLANATION",
        "classification": "NEW_ENTITY_HISTORICAL_BACKFILL",
        "fraction_min": 0.99,
        "latest_week_anomaly": False,
    },
    "history_truncation": {
        "status": "INVESTIGATE",
        "classification": "ENTITY_HISTORY_TRUNCATED",
        "fraction_min": 0.99,
    },
    "commodity_remap": {
        "status": "INVESTIGATE",
        "classification": "POSSIBLE_RECLASSIFICATION",
        "conservation_min": 0.99,
    },
    "coding_error": {
        "status": "INVESTIGATE",
        "reason_prefix": "unexplained_value_change",
        "cross_metric": "dollar_change_without_units",
    },
    "warehouse_transform_error": {
        "status": "INVESTIGATE",
        "reason_prefix": "unexplained_value_change",
    },
    "recalculation": {
        "status": "INVESTIGATE",
        "reason_prefix": "broad_historical_recalculation",
    },
    "schema_failure": {"status": "DATA_CONTRACT_FAILURE"},
    "null_duplicate_storm": {"status": "DATA_CONTRACT_FAILURE"},
    "market_movement": {
        "status": "INVESTIGATE",
        "historical_status": "PASS",
        "latest_week_anomaly": True,
    },
    "missing_products": {
        "status": "INVESTIGATE",
        "reason_prefix": "latest_week_missing",
        "classification": "LATEST_WEEK_MISSING",
    },
    "entity_merge": {
        "status": "INVESTIGATE",
        "reason_prefix": "possible_replacement",
        "classification": "ENTITY_REMOVED",
        "fraction_min": 0.99,
    },
    # The negative control: an untouched refresh must PASS end to end.
    "clean": {
        "status": "PASS",
        "historical_status": "PASS",
        "latest_week_anomaly": False,
    },
}


@pytest.fixture(scope="module")
def engine_runs(tmp_path_factory):
    config = suite_config("tiny")
    root = tmp_path_factory.mktemp("qc-families")
    runs = {}
    for index, family in enumerate(FAMILIES):
        built = build_scenario(
            config,
            index,
            root,
            family,
            ("source", "coded", "warehouse", "report"),
        )
        runs[family] = run_qc(
            ScenarioSource(built.directory),
            "V0002",
            "V0001",
            registry=StaticRegistry(
                list(built.oracle.get("expected_events", []))
            ),
        )
    return runs


@pytest.mark.parametrize("family", FAMILIES)
def test_family_outcome(family, engine_runs):
    assert EXPECTATIONS[family]["status"] == FAMILY_SPECS[family].expected_status
    run = engine_runs[family]
    expected = EXPECTATIONS[family]

    assert run.status == expected["status"]

    if "reason_prefix" in expected:
        assert any(
            reason.startswith(expected["reason_prefix"]) for reason in run.reasons
        ), run.reasons

    if "classification" in expected:
        classifications = {event.classification for event in run.events}
        assert expected["classification"] in classifications, classifications

    if "fraction_min" in expected:
        assert run.attribution is not None
        assert run.attribution.explained_fraction >= expected["fraction_min"]

    if "conservation_min" in expected:
        ratios = list(run.attribution.conservation.values())
        assert ratios and min(ratios) >= expected["conservation_min"]

    if "cross_metric" in expected:
        assert expected["cross_metric"] in run.attribution.cross_metric_flags

    if "historical_status" in expected:
        assert (
            run.machine["historical_revision"]["status"]
            == expected["historical_status"]
        )

    if "latest_week_anomaly" in expected:
        assert run.temporal is not None
        assert run.temporal.anomaly is expected["latest_week_anomaly"]

    json.dumps(run.machine, default=_json_default)


def test_contract_failure_runs_stop_before_revision(engine_runs):
    for family in ("schema_failure", "null_duplicate_storm"):
        run = engine_runs[family]
        assert run.attribution is None
        assert run.version_pair is None
        assert any(reason.startswith("contract_failure") for reason in run.reasons)


def test_expected_event_is_registry_matched(engine_runs):
    run = engine_runs["expected_event"]
    assert run.attribution.matched_event_ids
    assert "expected_event_matched" in run.reasons


def test_reclassification_details(engine_runs):
    run = engine_runs["commodity_remap"]
    event = next(
        event
        for event in run.events
        if event.classification == "POSSIBLE_RECLASSIFICATION"
    )
    assert event.details["product_count"] >= 1
    assert event.details["gross_moved"] > 0
    assert event.details["conservation_ratio"] >= 0.99


def test_machine_output_shape(engine_runs):
    run = engine_runs["market_movement"]
    machine = run.machine
    assert machine["status"] == "INVESTIGATE"
    assert machine["historical_revision"]["status"] == "PASS"
    assert machine["latest_week"]["anomaly"] is True
    assert machine["contracts"]["status"] == "PASS"
    assert machine["version_pair"]["shape"] == "NORMAL"
    assert machine["counterfactual"]["reconciliation_score"] is None
    assert machine["counterfactual"]["evaluated"] is False
    assert machine["reconciliation"]["status"] == "PASS"
    assert machine["lineage"]["status"] == "PASS"
    assert machine["lineage"]["first_divergence"] is None
    assert any(
        reason.startswith("latest_week_anomaly") for reason in machine["evidence"]
    )


LINEAGE_EXPECTATIONS = {
    "missing_stores": None,
    "new_store_backfill": "source",
    "expected_event": "source",
    "history_truncation": "source",
    "commodity_remap": "coded",
    "coding_error": "coded",
    "warehouse_transform_error": "warehouse",
    "recalculation": "source",
    "missing_products": None,
    "entity_merge": "source",
}


@pytest.mark.parametrize("family", list(LINEAGE_EXPECTATIONS))
def test_lineage_first_divergence(family, engine_runs):
    run = engine_runs[family]
    assert run.lineage is not None
    assert run.lineage.first_divergence == LINEAGE_EXPECTATIONS[family]


@pytest.mark.parametrize(
    "family",
    ["new_store_backfill", "expected_event", "history_truncation", "commodity_remap"],
)
def test_counterfactual_explains_structural_changes(family, engine_runs):
    run = engine_runs[family]
    assert run.counterfactual is not None
    assert run.counterfactual.reconciliation_score >= 0.99


def test_reconciliation_passes_where_evaluated(engine_runs):
    for family, run in engine_runs.items():
        if run.reconciliation is not None:
            assert run.reconciliation.status == "PASS", family
