from __future__ import annotations

import pandas as pd
import pytest

from qc.config import DatasetConfig
from qc.ledger import (
    CLASSIFICATION_CHANGE,
    ENTITY_ENTRY,
    ENTITY_EXIT,
    MATCHED_QUANTITY,
    MATCHED_UNIT_VALUE,
    ExplanationLedger,
    build_explanation_ledger,
)
from qc.lifecycle import NEW_BACKFILL, RECLASSIFIED, LifecycleEvent
from qc.versions import build_version_pair

CONFIG = DatasetConfig(report_grain=())


def _fact(rows):
    return pd.DataFrame(rows)


def _row(week, store, product, units, dollar):
    return {
        "week": week,
        "store_id": store,
        "product_id": product,
        "units": units,
        "dollar": dollar,
    }


def _pair(previous_rows, current_rows):
    previous = _fact(previous_rows)
    current = _fact(current_rows)
    pair = build_version_pair(
        previous, current, "v1", "v2", "warehouse", "warehouse", CONFIG
    )
    return previous, current, pair


def test_symmetric_decomposition_sums_exactly():
    previous, current, pair = _pair(
        [
            _row(1, "s1", "p1", 10.0, 100.0),
            _row(2, "s1", "p1", 10.0, 100.0),
        ],
        [
            _row(1, "s1", "p1", 12.0, 180.0),
            _row(2, "s1", "p1", 10.0, 100.0),
        ],
    )
    ledger = build_explanation_ledger(previous, current, pair, CONFIG)

    assert ledger.overlap_delta == pytest.approx(80.0)
    assert ledger.raw_movement == pytest.approx(80.0)
    categories = ledger.by_category()
    # quantity: (12-10) * (15+10)/2 = 25; unit value: (15-10) * (12+10)/2 = 55.
    assert categories[MATCHED_QUANTITY] == pytest.approx(25.0)
    assert categories[MATCHED_UNIT_VALUE] == pytest.approx(55.0)
    assert ledger.net_unexplained == pytest.approx(0.0)
    assert ledger.conservation["difference"] == pytest.approx(0.0, abs=1e-9)
    assert ledger.explained_fraction == pytest.approx(1.0)
    assert ledger.to_dict()["schema_version"] == 1
    assert ExplanationLedger.from_dict(ledger.to_dict()).by_category() == pytest.approx(
        categories
    )


def test_entry_exit_and_new_period_stay_distinct_categories():
    previous, current, pair = _pair(
        [
            _row(1, "s1", "p1", 10.0, 100.0),
            _row(2, "s1", "p1", 10.0, 100.0),
        ],
        [
            _row(1, "s1", "p1", 10.0, 100.0),
            _row(2, "s1", "p1", 10.0, 100.0),
            _row(1, "s2", "p2", 5.0, 50.0),
            _row(2, "s2", "p2", 5.0, 50.0),
            _row(3, "s1", "p1", 10.0, 100.0),
            _row(3, "s2", "p2", 5.0, 50.0),
        ],
    )
    ledger = build_explanation_ledger(previous, current, pair, CONFIG)
    categories = ledger.by_category()

    assert categories[ENTITY_ENTRY] == pytest.approx(100.0)
    assert ENTITY_EXIT not in categories
    assert ledger.overlap_delta == pytest.approx(100.0)
    assert ledger.new_period_movement == pytest.approx(150.0)
    assert ledger.raw_movement == pytest.approx(250.0)
    assert ledger.conservation["difference"] == pytest.approx(0.0, abs=1e-9)
    assert ledger.coverage["entered_entities"] == 1
    assert ledger.coverage["matched_entities"] == 1
    assert ledger.to_dict()["coverage"]["new_period_weeks"] == 1


def test_exit_contribution_is_negative():
    previous, current, pair = _pair(
        [
            _row(1, "s1", "p1", 10.0, 100.0),
            _row(2, "s1", "p1", 10.0, 100.0),
            _row(1, "s9", "p9", 5.0, 50.0),
            _row(2, "s9", "p9", 5.0, 50.0),
        ],
        [
            _row(1, "s1", "p1", 10.0, 100.0),
            _row(2, "s1", "p1", 10.0, 100.0),
        ],
    )
    ledger = build_explanation_ledger(previous, current, pair, CONFIG)
    categories = ledger.by_category()
    assert categories[ENTITY_EXIT] == pytest.approx(-100.0)
    assert ledger.raw_movement == pytest.approx(-100.0)
    assert ledger.conservation["difference"] == pytest.approx(0.0, abs=1e-9)


def test_approved_event_labels_entry_support_without_changing_arithmetic():
    previous, current, pair = _pair(
        [
            _row(1, "s1", "p1", 10.0, 100.0),
            _row(2, "s1", "p1", 10.0, 100.0),
        ],
        [
            _row(1, "s1", "p1", 10.0, 100.0),
            _row(2, "s1", "p1", 10.0, 100.0),
            _row(1, "s2", "p2", 5.0, 50.0),
            _row(2, "s2", "p2", 5.0, 50.0),
        ],
    )
    event = LifecycleEvent(
        "store", "s2", NEW_BACKFILL, historical_value_added=100.0
    )
    unapproved = build_explanation_ledger(previous, current, pair, CONFIG, events=[event])
    approved = build_explanation_ledger(
        previous,
        current,
        pair,
        CONFIG,
        events=[event],
        approved_event_ids=("evt-1",),
    )
    plain_entry = next(
        item for item in unapproved.contributions if item.category == ENTITY_ENTRY
    )
    approved_entry = next(
        item for item in approved.contributions if item.category == ENTITY_ENTRY
    )
    assert plain_entry.value == pytest.approx(approved_entry.value)
    assert plain_entry.support == "arithmetic"
    assert approved_entry.support == "approved_event"
    assert approved_entry.approved_by == ("evt-1",)


def test_reclassification_is_annotated_once_without_double_counting():
    previous, current, pair = _pair(
        [
            _row(1, "s1", "p1", 10.0, 100.0),
            _row(1, "s1", "p2", 5.0, 50.0),
            _row(2, "s1", "p1", 10.0, 100.0),
            _row(2, "s1", "p2", 5.0, 50.0),
        ],
        [
            _row(1, "s1", "p1", 15.0, 150.0),
            _row(2, "s1", "p1", 15.0, 150.0),
        ],
    )
    event = LifecycleEvent(
        "commodity",
        "c1",
        RECLASSIFIED,
        details={"gross_moved": 100.0, "conservation_ratio": 1.0},
    )
    ledger = build_explanation_ledger(previous, current, pair, CONFIG, events=[event])
    annotation = next(
        item for item in ledger.contributions if item.category == CLASSIFICATION_CHANGE
    )
    assert annotation.value == 0.0
    assert annotation.detail["gross_moved"] == pytest.approx(100.0)
    assert ledger.conservation["difference"] == pytest.approx(0.0, abs=1e-9)


def test_decomposition_without_quantity_falls_back_to_matched_volume():
    config = DatasetConfig(report_grain=(), unit_value_quantity="")
    previous, current, pair = _pair(
        [_row(1, "s1", "p1", 10.0, 100.0), _row(2, "s1", "p1", 10.0, 100.0)],
        [_row(1, "s1", "p1", 12.0, 180.0), _row(2, "s1", "p1", 10.0, 100.0)],
    )
    ledger = build_explanation_ledger(previous, current, pair, config)
    categories = ledger.by_category()
    assert "matched_volume" in categories
    assert ledger.conservation["difference"] == pytest.approx(0.0, abs=1e-9)
