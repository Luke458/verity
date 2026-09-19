from __future__ import annotations

import pandas as pd
import pytest

from qc.attribution import (
    AttributionResult,
    Contributor,
    classify_run,
    explain_revision,
)
from qc.config import DatasetConfig
from qc.contracts import validate_contracts
from qc.lifecycle import (
    EXTENDED,
    LATEST_MISSING,
    NEW_BACKFILL,
    NEW_RECENT,
    RECLASSIFIED,
    REMOVED,
    TRUNCATED,
    classify_entity_changes,
    detect_reclassification,
)
from qc.revision import build_revision_cube, cube_summary
from qc.versions import build_version_pair

CONFIG = DatasetConfig()


def _fact(rows):
    return pd.DataFrame(rows)


def _pair(previous, current):
    return build_version_pair(previous, current, "V0001", "V0002", "wh", "wh", CONFIG)


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------


def _clean_facts():
    previous = _fact(
        [
            {"week": 1, "store_id": "S1", "dollar": 10.0, "units": 1},
            {"week": 2, "store_id": "S1", "dollar": 11.0, "units": 1},
        ]
    )
    current = previous.copy()
    return previous, current


def test_contracts_pass_on_clean_data():
    previous, current = _clean_facts()
    result = validate_contracts(current, previous, CONFIG)
    assert result.status == "PASS"
    assert not result.failed


def test_contracts_fail_on_missing_column():
    previous, current = _clean_facts()
    result = validate_contracts(current.drop(columns=["dollar"]), previous, CONFIG)
    assert result.status == "DATA_CONTRACT_FAILURE"
    assert any(check.name == "required_columns" for check in result.failed)


def test_contracts_fail_on_non_numeric_metric():
    previous, current = _clean_facts()
    current["dollar"] = current["dollar"].astype(str)
    result = validate_contracts(current, previous, CONFIG)
    assert any(check.name == "metric_dtypes" for check in result.failed)


def test_contracts_fail_on_week_gap():
    previous, current = _clean_facts()
    current.loc[1, "week"] = 3
    result = validate_contracts(current, previous, CONFIG)
    assert any(check.name == "week_progression" for check in result.failed)


def test_contracts_fail_on_null_and_duplicate_storm():
    previous, current = _clean_facts()
    storm = pd.concat([current, current], ignore_index=True)
    storm.loc[0, "dollar"] = None
    result = validate_contracts(storm, previous, CONFIG)
    names = {check.name for check in result.failed}
    assert "null_fraction" in names
    assert "duplicate_fraction" in names


# ---------------------------------------------------------------------------
# Version shapes
# ---------------------------------------------------------------------------


def _weeks_frame(weeks, store="S1"):
    return _fact(
        [{"week": w, "store_id": store, "dollar": 1.0, "units": 1} for w in weeks]
    )


def test_version_shape_normal():
    pair = _pair(_weeks_frame([1, 2, 3]), _weeks_frame([1, 2, 3, 4]))
    assert pair.shape == "NORMAL"
    assert pair.new_periods == (4,)
    assert pair.overlap_weeks == (1, 2, 3)


def test_version_shape_no_new_week():
    pair = _pair(_weeks_frame([1, 2, 3]), _weeks_frame([1, 2, 3]))
    assert pair.shape == "NO_NEW_WEEK"
    assert pair.new_periods == ()


def test_version_shape_multiple_new_weeks():
    pair = _pair(_weeks_frame([1, 2]), _weeks_frame([1, 2, 3, 4]))
    assert pair.shape == "MULTIPLE_NEW_WEEKS"
    assert pair.new_periods == (3, 4)


def test_version_shape_shortened_and_extended():
    shortened = _pair(_weeks_frame([1, 2, 3, 4]), _weeks_frame([2, 3, 4, 5]))
    assert shortened.shape == "HISTORY_SHORTENED"
    extended = _pair(_weeks_frame([2, 3]), _weeks_frame([1, 2, 3, 4]))
    assert extended.shape == "HISTORY_EXTENDED"


# ---------------------------------------------------------------------------
# Revision cube
# ---------------------------------------------------------------------------


def _cube_frames():
    previous = _fact(
        [
            {"week": 1, "store_id": "S1", "product_id": "P1", "dollar": 10.0, "units": 2},
            {"week": 1, "store_id": "S1", "product_id": "P2", "dollar": 5.0, "units": 1},
            {"week": 2, "store_id": "S1", "product_id": "P1", "dollar": 10.0, "units": 2},
            {"week": 2, "store_id": "S2", "product_id": "P1", "dollar": 7.0, "units": 1},
        ]
    )
    current = _fact(
        [
            {"week": 1, "store_id": "S1", "product_id": "P1", "dollar": 12.0, "units": 2},
            {"week": 1, "store_id": "S1", "product_id": "P2", "dollar": 0.0, "units": 0},
            {"week": 2, "store_id": "S1", "product_id": "P1", "dollar": 10.0, "units": 2},
            {"week": 3, "store_id": "S1", "product_id": "P1", "dollar": 4.0, "units": 1},
        ]
    )
    return previous, current


def test_revision_cube_exact_deltas_and_periods():
    previous, current = _cube_frames()
    cube = build_revision_cube(
        previous, current, ["week", "store_id", "product_id"], CONFIG, (3,)
    )
    assert set(cube["period"]) == {"overlap", "new"}

    week1_s1_p1 = cube[
        (cube["week"] == 1) & (cube["store_id"] == "S1") & (cube["product_id"] == "P1")
    ].iloc[0]
    assert week1_s1_p1["dollar_previous"] == 10.0
    assert week1_s1_p1["dollar_current"] == 12.0
    assert week1_s1_p1["dollar_delta"] == 2.0
    assert week1_s1_p1["dollar_relative"] == pytest.approx(0.2)
    assert week1_s1_p1["period"] == "overlap"

    week3 = cube[cube["week"] == 3].iloc[0]
    assert week3["period"] == "new"
    assert week3["dollar_previous"] == 0.0
    assert week3["dollar_current"] == 4.0

    summary = cube_summary(cube, "dollar")
    assert summary["previous"] == pytest.approx(32.0)
    assert summary["current"] == pytest.approx(22.0)
    assert summary["delta"] == pytest.approx(-10.0)


def test_revision_cube_aggregates_to_coarse_grain():
    previous, current = _cube_frames()
    cube = build_revision_cube(previous, current, ["week", "store_id"], CONFIG, (3,))
    week1 = cube[cube["week"] == 1].iloc[0]
    assert week1["dollar_previous"] == 15.0
    assert week1["dollar_current"] == 12.0
    assert week1["dollar_delta"] == -3.0


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def _lifecycle_frames():
    rows_prev = []
    rows_curr = []
    for week in (1, 2, 3):
        rows_prev.append({"week": week, "store_id": "A", "dollar": 1.0, "units": 1})
        rows_curr.append({"week": week, "store_id": "A", "dollar": 1.0, "units": 1})
        rows_curr.append({"week": week, "store_id": "B", "dollar": 2.0, "units": 1})
        rows_prev.append({"week": week, "store_id": "C", "dollar": 3.0, "units": 1})
        rows_prev.append({"week": week, "store_id": "F", "dollar": 5.0, "units": 1})
        rows_curr.append({"week": week, "store_id": "F", "dollar": 5.0, "units": 1})
    # D gains historical week 3 in the current version: history extension.
    for week in (1, 2):
        rows_prev.append({"week": week, "store_id": "D", "dollar": 4.0, "units": 1})
    for week in (1, 2, 3, 4):
        rows_curr.append({"week": week, "store_id": "D", "dollar": 4.0, "units": 1})
        rows_curr.append({"week": week, "store_id": "E", "dollar": 6.0, "units": 1})
    rows_curr.append({"week": 4, "store_id": "A", "dollar": 1.0, "units": 1})
    rows_curr.append({"week": 4, "store_id": "G", "dollar": 7.0, "units": 1})
    return _fact(rows_prev), _fact(rows_curr)


def test_lifecycle_classification():
    previous, current = _lifecycle_frames()
    pair = _pair(previous, current)
    events = {
        event.entity_id: event
        for event in classify_entity_changes(previous, current, pair, CONFIG)
        if event.entity_type == "store"
    }
    assert "A" not in events  # unchanged store is omitted
    assert events["B"].classification == NEW_BACKFILL
    assert events["B"].historical_value_added == pytest.approx(6.0)
    assert events["C"].classification == REMOVED
    assert events["C"].historical_value_removed == pytest.approx(9.0)
    assert events["D"].classification == EXTENDED
    assert events["D"].historical_value_added == pytest.approx(4.0)
    assert events["E"].classification == NEW_BACKFILL
    # F is absent only from the new week 4: a latest-period gap, not truncation.
    assert events["F"].classification == LATEST_MISSING
    assert events["G"].classification == NEW_RECENT


def test_lifecycle_truncation_is_distinct_from_latest_gap():
    previous = _fact(
        [
            {"week": 1, "store_id": "S1", "dollar": 1.0, "units": 1},
            {"week": 2, "store_id": "S1", "dollar": 1.0, "units": 1},
            {"week": 3, "store_id": "S1", "dollar": 1.0, "units": 1},
        ]
    )
    current = _fact(
        [
            {"week": 1, "store_id": "S1", "dollar": 1.0, "units": 1},
            {"week": 3, "store_id": "S1", "dollar": 1.0, "units": 1},
            {"week": 4, "store_id": "S1", "dollar": 1.0, "units": 1},
        ]
    )
    pair = _pair(previous, current)
    events = classify_entity_changes(previous, current, pair, CONFIG)
    store_event = next(event for event in events if event.entity_type == "store")
    assert store_event.classification == TRUNCATED
    assert store_event.historical_weeks_removed == (2,)


def test_reclassification_conservation():
    products_previous = pd.DataFrame(
        {"product_id": ["P1", "P2"], "commodity_id": ["C1", "C1"]}
    )
    products_current = pd.DataFrame(
        {"product_id": ["P1", "P2"], "commodity_id": ["C2", "C1"]}
    )
    previous = _fact(
        [
            {"week": 1, "product_id": "P1", "commodity_id": "C1", "dollar": 10.0, "units": 1},
            {"week": 1, "product_id": "P2", "commodity_id": "C1", "dollar": 4.0, "units": 1},
            {"week": 2, "product_id": "P1", "commodity_id": "C1", "dollar": 10.0, "units": 1},
        ]
    )
    current = _fact(
        [
            {"week": 1, "product_id": "P1", "commodity_id": "C2", "dollar": 10.0, "units": 1},
            {"week": 1, "product_id": "P2", "commodity_id": "C1", "dollar": 4.0, "units": 1},
            {"week": 2, "product_id": "P1", "commodity_id": "C2", "dollar": 10.0, "units": 1},
        ]
    )
    pair = _pair(previous, current)
    events = detect_reclassification(
        previous, current, products_previous, products_current, pair, CONFIG
    )
    assert len(events) == 1
    event = events[0]
    assert event.classification == RECLASSIFIED
    assert event.details["product_count"] == 1
    assert event.details["conservation_ratio"] == pytest.approx(1.0)
    assert event.details["from_delta"] == pytest.approx(-20.0)
    assert event.details["to_delta"] == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------


def test_attribution_deduplicates_entity_hierarchy():
    from qc.lifecycle import LifecycleEvent

    previous = _fact(
        [
            {"week": 1, "store_id": "S1", "product_id": "P1", "dollar": 10.0, "units": 1},
            {"week": 2, "store_id": "S1", "product_id": "P1", "dollar": 10.0, "units": 1},
        ]
    )
    current = pd.concat(
        [
            previous,
            _fact(
                [
                    {"week": 1, "store_id": "S9", "product_id": "P1", "dollar": 5.0, "units": 1},
                    {"week": 2, "store_id": "S9", "product_id": "P1", "dollar": 5.0, "units": 1},
                ]
            ),
        ],
        ignore_index=True,
    )
    cube = build_revision_cube(
        previous, current, ["week", "store_id", "product_id"], CONFIG, ()
    )
    events = [
        LifecycleEvent("store", "S9", NEW_BACKFILL, historical_value_added=10.0),
        LifecycleEvent("product", "P1", NEW_BACKFILL, historical_value_added=10.0),
    ]
    result = explain_revision(cube, events, [], CONFIG)
    assert result.raw_delta == pytest.approx(10.0)
    # The store event explains the rows; the product event must not double count.
    assert result.explained_delta == pytest.approx(10.0)
    assert result.explained_fraction == pytest.approx(1.0)


def test_attribution_matches_expected_events_and_status():
    previous = _fact(
        [
            {"week": 1, "store_id": "S1", "product_id": "P1", "dollar": 10.0, "units": 1},
            {"week": 2, "store_id": "S1", "product_id": "P1", "dollar": 10.0, "units": 1},
        ]
    )
    current = pd.concat(
        [
            previous,
            _fact(
                [
                    {"week": 1, "store_id": "S9", "product_id": "P1", "dollar": 5.0, "units": 1},
                    {"week": 2, "store_id": "S9", "product_id": "P1", "dollar": 5.0, "units": 1},
                ]
            ),
        ],
        ignore_index=True,
    )
    pair = _pair(previous, current)
    cube = build_revision_cube(previous, current, ["week", "store_id"], CONFIG, ())
    events = classify_entity_changes(previous, current, pair, CONFIG)
    registry = [
        {
            "event_id": "evt-1",
            "event_type": "new_store_historical_backfill",
            "entity_type": "store",
            "entity_ids": ["S9"],
        }
    ]
    result = explain_revision(cube, events, registry, CONFIG)
    assert result.matched_event_ids == ["evt-1"]
    assert not result.unmatched_events
    assert classify_run("PASS", events, result, CONFIG) == "PASS_WITH_EXPLANATION"


def test_status_machine():
    attribution = AttributionResult(
        raw_delta=100.0,
        explained_delta=0.0,
        unexplained_delta=100.0,
        explained_fraction=0.0,
        material=True,
        breadth=0.1,
    )
    assert classify_run("PASS", [], attribution, CONFIG) == "INVESTIGATE"
    assert classify_run("DATA_CONTRACT_FAILURE", [], attribution, CONFIG) == "DATA_CONTRACT_FAILURE"

    borderline = AttributionResult(
        raw_delta=0.0,
        explained_delta=0.0,
        unexplained_delta=0.0,
        explained_fraction=1.0,
        material=False,
        breadth=0.8,
    )
    assert classify_run("PASS", [], borderline, CONFIG) == "INVESTIGATE"

    quiet = AttributionResult(
        raw_delta=0.0,
        explained_delta=0.0,
        unexplained_delta=0.0,
        explained_fraction=1.0,
        material=False,
        breadth=0.0,
    )
    assert classify_run("PASS", [], quiet, CONFIG) == "PASS"


def test_cross_metric_flag():
    events = []
    attribution = AttributionResult(
        raw_delta=-5.0,
        explained_delta=0.0,
        unexplained_delta=-5.0,
        explained_fraction=0.0,
        material=True,
        breadth=0.1,
        contributors=[Contributor("store", "S1", -5.0, 5.0)],
        cross_metric_flags=["dollar_change_without_units"],
    )
    assert classify_run("PASS", events, attribution, CONFIG) == "INVESTIGATE"
