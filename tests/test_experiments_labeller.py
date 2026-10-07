"""Cause-labelling sandbox: scoring, the paired test and the model interface."""

from __future__ import annotations

import pytest

from experiments.labeller.evaluate import mcnemar, score
from experiments.labeller.models import RuleLabeller


def _row(kind, causes, rule, features=None):
    return {"kind": kind, "causes": causes, "rule_causes": rule, "features": features or {"x": 0.0}}


def test_score_separates_exact_sets_and_pair_coverage():
    rows = [
        _row("single", ["CODING"], ["CODING"]),
        _row("pair", ["CODING", "MARKET_MOVEMENT"], ["CODING"]),
        _row("clean", [], []),
    ]
    report = score(rows, RuleLabeller().predict(rows))
    assert report["exact"]["hits"] == 2
    pair = report["by_kind"]["pair"]
    assert pair["names_both"]["hits"] == 0 and pair["names_one_or_more"]["hits"] == 1
    assert report["per_cause"]["MARKET_MOVEMENT"]["fn"] == 1


def test_mcnemar_is_exact_and_symmetric():
    assert mcnemar([True] * 5, [True] * 5)["p_value"] == 1.0
    lopsided = mcnemar([True] * 10, [False] * 10)
    assert lopsided["only_first"] == 10 and lopsided["p_value"] == pytest.approx(2 / 2**10)
    assert mcnemar([True, False], [False, True])["p_value"] == 1.0


def test_tree_labeller_learns_a_separable_cause():
    pytest.importorskip("sklearn")
    from experiments.labeller.models import TreeLabeller

    rows = [
        _row("single", ["CODING"] if i % 2 else ["WAREHOUSE"], [], {"x": float(i % 2), "y": 0.0})
        for i in range(200)
    ]
    model = TreeLabeller()
    model.fit(rows[:150], rows[150:])
    assert model.predict(rows[150:]) == [set(row["causes"]) for row in rows[150:]]


def test_service_state_is_readable_and_omits_empty_evidence():
    from experiments.labeller.service import QUESTIONS, render_state

    features = {
        "status:INVESTIGATE": 1.0, "status:PASS": 0.0, "events:LATEST_WEEK_MISSING": 3.0,
        "missing_share:store": 0.12, "missing_material:store": 1.0, "origin:coded": 1.0,
        "increment_log:coded": -1.5, "increment_log:source": -12.0, "min_share_p_log": 0.0,
    }
    state = render_state(features)
    assert state["refresh status"] == "INVESTIGATE"
    assert state["entity lifecycle events"] == {"latest week missing": 3}
    assert state["missing stores material"] is True
    assert state["pipeline lineage"]["revision added at each stage, relative"] == {"coded": 0.03162}
    assert "strongest category share shift" not in state
    assert set(QUESTIONS) == {"BACKFILL", "CODING", "ENTITY_MERGE", "HISTORICAL_CORRECTION", "MARKET_MOVEMENT",
                              "MISSING_PRODUCTS", "MISSING_STORES", "RECLASSIFICATION", "SCHEMA_FAILURE", "WAREHOUSE"}


def test_service_labeller_thresholds():
    from experiments.labeller.models import CAUSES
    from experiments.labeller.service import ServiceLabeller

    def row(i, causes):
        return {"split": "dev", "seed": 1, "index": i, "causes": causes, "kind": "single", "features": {}}

    rows = [row(i, ["CODING"] if i % 2 else []) for i in range(20)]
    # P(CODING) is 0.4 on true rows and 0.2 otherwise: below 0.5, but separable.
    probabilities = {
        f"dev/1/{i}": {cause: (0.4 if (cause == "CODING" and i % 2) else 0.2) for cause in CAUSES}
        for i in range(20)
    }
    zero_shot = ServiceLabeller("svc", probabilities, tuned=False)
    zero_shot.fit([], rows)
    assert zero_shot.predict(rows) == [set()] * 20
    tuned = ServiceLabeller("svc-tuned", probabilities, tuned=True)
    tuned.fit([], rows)
    assert tuned.predict(rows) == [set(r["causes"]) for r in rows]
