"""Registry scoping for closures and category moves (qc.attribution)."""

from __future__ import annotations

import pandas as pd
import pytest

from qc.attribution import (
    AttributionResult,
    _match_expected,
    classify_run,
    like_for_like,
    match_absences,
)
from qc.config import DatasetConfig
from qc.lifecycle import LATEST_MISSING, RECLASSIFIED, LifecycleEvent

CONFIG = DatasetConfig()
APPROVED = {"dataset": CONFIG.name, "confirmed": True, "approved_by": "analyst"}


def _closure(**extra):
    return {**APPROVED, "event_id": "close", "classification": LATEST_MISSING, "entity_type": "store",
            "entity_ids": ["S001"], "effective_from_week": 10, "effective_to_week": 12, **extra}


def _move(**extra):
    return {**APPROVED, "event_id": "move", "classification": RECLASSIFIED, "entity_type": "commodity",
            "entity_ids": ["C01->C02"], "effective_from_week": 10, "effective_to_week": 10, **extra}


def _absent(store, expected=50.0):
    return LifecycleEvent("store", store, LATEST_MISSING, (LATEST_MISSING,), expected_value=expected,
                          expected_period_total=100.0)


def _moved(conservation=1.0, products=("P1", "P2")):
    return LifecycleEvent("commodity", "C01->C02", RECLASSIFIED, details={
        "from_commodity": "C01", "to_commodity": "C02", "products": list(products),
        "conservation_ratio": conservation})


@pytest.mark.parametrize(
    ("entry", "week", "matched"),
    [
        (_closure(), 11, True),
        (_closure(), 13, False),  # outside the weeks it is in effect
        (_closure(entity_ids=["S002"]), 11, False),
        (_closure(confirmed=False), 11, False),
        (_closure(approved_by=None), 11, False),
        (_closure(dataset="other"), 11, False),
        (_closure(effective_to_week=None), 11, False),
        (_closure(classification="ENTITY_REMOVED"), 11, False),
    ],
)
def test_closure_matches_only_its_entities_and_weeks(entry, week, matched):
    assert (_match_expected(_absent("S001"), [entry], CONFIG, week) == "close") is matched


@pytest.mark.parametrize(
    ("entry", "event", "matched"),
    [
        (_move(), _moved(), True),
        (_move(product_ids=["P1", "P2", "P3"]), _moved(), True),
        (_move(product_ids=["P1"]), _moved(), False),  # a product the approval does not list moved
        (_move(), _moved(conservation=0.5), False),  # the moved value changed as well
        (_move(entity_ids=["C02->C01"]), _moved(), False),
        (_move(effective_from_week=11, effective_to_week=12), _moved(), False),
    ],
)
def test_move_matches_only_a_conserving_move_of_listed_products(entry, event, matched):
    assert (_match_expected(event, [entry], CONFIG, 10) == "move") is matched


def test_unapproved_absences_still_escalate():
    events = [_absent("S001"), _absent("S002")]
    matched = match_absences(events, [_closure()], CONFIG, 11)
    assert matched == {("store", "S001"): "close"}
    attribution = AttributionResult(matched_absences={"store:S001": "close"})
    assert classify_run("PASS", events, attribution, CONFIG) == "INVESTIGATE"
    attribution = AttributionResult(matched_absences={"store:S001": "close", "store:S002": "close"})
    assert classify_run("PASS", events, attribution, CONFIG) == "PASS"


def test_like_for_like_restates_only_matched_changes():
    previous = pd.DataFrame({
        "store_id": ["S001", "S002", "S001", "S002"], "product_id": ["P1", "P1", "P3", "P3"],
        "commodity_id": ["C01", "C01", "C01", "C01"], "week": [9, 9, 9, 9], "dollar": [1.0, 2.0, 3.0, 4.0],
    })
    current = previous.loc[previous["store_id"] == "S002"]
    unchanged = like_for_like(previous, current, [_moved()], AttributionResult())
    assert unchanged[0] is previous and unchanged[1] is current
    assert unchanged[2] == {"excluded": [], "restated": []}

    attribution = AttributionResult(matched_absences={"store:S001": "close"}, matched_moves={"C01->C02": "move"})
    restated_previous, restated_current, record = like_for_like(previous, current, [_moved()], attribution)
    assert set(restated_previous["store_id"]) == {"S002"} and set(restated_current["store_id"]) == {"S002"}
    # Only the observed products move; P3 stays in its category.
    assert dict(zip(restated_previous["product_id"], restated_previous["commodity_id"], strict=True)) == {
        "P1": "C02", "P3": "C01"}
    assert record["excluded"] == [{"entity": "store:S001", "approval_id": "close"}]
    assert record["restated"] == [{"move": "C01->C02", "products": 2, "approval_id": "move"}]
