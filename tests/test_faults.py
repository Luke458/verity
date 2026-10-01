from __future__ import annotations

import pandas as pd
import pytest

from qcgen.config import ALL_FAMILIES
from qcgen.scenarios import build_scenario
from qcgen.spec import FAMILY_SPECS
from qcgen.stages import STAGE_ORDER


@pytest.mark.parametrize("family", ALL_FAMILIES)
def test_family_ground_truth(family, tiny_config, tmp_path):
    stages = ("source", "coded", "warehouse", "report")
    result = build_scenario(tiny_config, 0, tmp_path, family, stages)
    manifest = result.manifest
    oracle = result.oracle
    case = oracle["cases"][0]
    effect = case["injected_effect"]

    assert case["family"] == family
    assert case["injection_stage"] == oracle["fault"]["stage"]
    spec = FAMILY_SPECS[family]
    assert case["kind"] == spec.kind
    assert case["expected_status"] == spec.expected_status
    assert case["expected_class"] == spec.expected_class
    assert case["expected_origin"] == spec.expected_origin
    assert case["expected_status"] in {
        "INVESTIGATE",
        "PASS",
        "PASS_WITH_EXPLANATION",
        "DATA_CONTRACT_FAILURE",
    }

    # No divergence may be recorded before the injection stage.
    for stage in STAGE_ORDER:
        if STAGE_ORDER.index(stage) < STAGE_ORDER.index(case["injection_stage"]):
            assert case["effects"][stage] == {}

    if family == "missing_stores":
        assert effect["dollar"] < 0
        assert effect["units"] < 0
        assert case["affected"]["stores"]
        assert case["expected_class"] == "missing_stores"
        assert case["expected_origin"] == "source"
    elif family == "missing_products":
        assert effect["dollar"] < 0
        assert effect["units"] < 0
        assert case["affected"]["products"]
        assert case["expected_class"] == "missing_products"
        assert case["expected_origin"] == "source"
    elif family == "entity_merge":
        assert case["expected_class"] == "entity_merge"
        stores = case["affected"]["stores"]
        assert len(stores) == 2
        assert case["details"]["source_store"] in stores
        assert case["details"]["target_store"] in stores
        assert abs(effect.get("dollar", 0.0)) > 0
    elif family == "new_store_backfill":
        assert effect["dollar"] > 0
        assert effect["units"] > 0
        assert all(store.startswith("S9") for store in case["affected"]["stores"])
        assert case["kind"] == "fault"
        assert case["expected_class"] == "backfill"
        assert oracle["expected_events"] == []
    elif family == "expected_event":
        assert case["kind"] == "expected_event"
        assert case["expected_status"] == "PASS_WITH_EXPLANATION"
        events = oracle["expected_events"]
        assert len(events) == 1
        assert events[0]["entity_ids"] == case["affected"]["stores"]
        assert events[0]["event_type"] == "new_store_historical_backfill"
    elif family == "history_truncation":
        assert effect["dollar"] < 0
        assert case["weeks"]
        assert case["expected_class"] == "truncation"
    elif family == "commodity_remap":
        per_value = case["details"]["per_value"]
        values = list(per_value.values())
        assert len(values) == 2
        assert values[0] * values[1] < 0
        assert abs(sum(values)) <= 0.01 * max(abs(v) for v in values)
        assert case["details"]["effect_stage"] == "coded"
        assert case["injected_effect"]["dollar_moved"] > 0
    elif family == "coding_error":
        assert abs(effect["units"]) <= 1e-9
        assert abs(effect["dollar"]) > 0
        assert case["injection_stage"] == "coded"
        assert case["expected_origin"] == "coded"
    elif family == "warehouse_transform_error":
        assert abs(effect["units"]) <= 1e-9
        assert abs(effect["dollar"]) > 0
        assert case["injection_stage"] == "warehouse"
        assert case["expected_origin"] == "warehouse"
    elif family == "recalculation":
        assert abs(effect["dollar"]) > 0
        assert case["weeks"]
        assert max(case["weeks"]) < manifest["n_current_weeks"]
        assert case["expected_class"] == "historical_correction"
    elif family == "schema_failure":
        assert case["expected_status"] == "DATA_CONTRACT_FAILURE"
        report = pd.read_parquet(
            result.directory / "versions" / "V0002" / "report" / "fact.parquet"
        )
        assert case["details"]["column"] not in report.columns
    elif family == "null_duplicate_storm":
        assert case["details"]["duplicated_rows"] > 0
        assert effect["units"] > 0
    elif family == "market_movement":
        assert case["kind"] == "movement"
        assert case["expected_status"] == "INVESTIGATE"
        assert effect["dollar"] < 0
        assert effect["units"] < 0
    elif family == "week_restatement":
        assert effect["dollar"] < 0
        assert case["weeks"] == [case["details"]["week"]]
        assert case["details"]["week"] < manifest["n_current_weeks"]
        assert case["expected_class"] == "historical_correction"
    elif family == "clean":
        assert case["kind"] == "control"
        assert case["expected_status"] == "PASS"
        assert all(abs(value) <= 1e-9 for value in effect.values())
    else:
        pytest.fail(f"unhandled family {family}")


def test_source_fault_propagates_to_every_stage(tiny_config, tmp_path):
    stages = ("source", "coded", "warehouse", "report")
    result = build_scenario(tiny_config, 0, tmp_path, "missing_stores", stages)
    case = result.oracle["cases"][0]
    for stage in stages:
        assert case["effects"][stage].get("dollar", 0.0) < 0, stage
