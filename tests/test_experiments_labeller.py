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
