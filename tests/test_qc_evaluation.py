from __future__ import annotations

import pytest

from qc.evaluation import (
    EvaluationCase,
    evaluation_gates,
    evaluation_metrics,
    false_clearance_upper_bound,
)


def _case(index: int, *, actionable: bool, review: bool, challenger: bool | None = None):
    return EvaluationCase(
        case_id=f"case-{index}",
        family="family",
        group=f"group-{index}",
        actionable=actionable,
        verifier_status="INVESTIGATE" if review else "PASS",
        challenger_status="INVESTIGATE" if (challenger if challenger is not None else review) else "PASS",
        effective_status="INVESTIGATE" if review else "PASS",
        verifier_review=review,
        challenger_review=challenger if challenger is not None else review,
        effective_review=review,
        expected_review=actionable,
    )


def test_metrics_separate_raw_challenger_from_effective_decisions():
    cases = [
        _case(1, actionable=True, review=True, challenger=False),
        _case(2, actionable=True, review=True, challenger=True),
        _case(3, actionable=False, review=False, challenger=False),
    ]
    metrics = evaluation_metrics(cases)
    assert metrics["detection_rate"] == pytest.approx(1.0)
    assert metrics["raw_challenger_detection_rate"] == pytest.approx(0.5)
    assert metrics["raw_policy_overrides"] == 1
    assert metrics["review_rate"] == pytest.approx(2 / 3)


def test_false_clearance_bound_requires_enough_incidents():
    insufficient = false_clearance_upper_bound(
        [_case(index, actionable=True, review=True) for index in range(5)]
    )
    assert insufficient["status"] == "INSUFFICIENT_EVIDENCE"
    assert insufficient["passes"] is False

    clean = false_clearance_upper_bound(
        [_case(index, actionable=True, review=True) for index in range(400)]
    )
    assert clean["status"] == "EVALUATED"
    assert clean["cleared"] == 0
    assert clean["passes"] is True

    small_clean = false_clearance_upper_bound(
        [_case(index, actionable=True, review=True) for index in range(20)]
    )
    assert small_clean["status"] == "EVALUATED"
    assert small_clean["passes"] is False


def test_gate_fails_on_false_clearance_and_deterministic_errors():
    cases = [_case(index, actionable=True, review=True) for index in range(12)]
    cases += [_case(500 + index, actionable=False, review=False) for index in range(4)]
    cases[0] = _case(99, actionable=True, review=False)
    gates = evaluation_gates(cases)
    assert gates["status"] == "FAIL"
    assert gates["gates"]["false_clearance"]["status"] == "FAIL"
    assert gates["gates"]["deterministic_fixture_errors"]["status"] == "FAIL"
    assert "case-99" in gates["gates"]["deterministic_fixture_errors"]["errors"]

    deterministic = [_case(index, actionable=True, review=True) for index in range(12)]
    deterministic += [_case(500 + index, actionable=False, review=False) for index in range(4)]
    deterministic[0] = EvaluationCase(
        **{
            **_case(0, actionable=True, review=True).to_dict(),
            "effective_review": False,
        }
    )
    gates = evaluation_gates(deterministic)
    assert gates["gates"]["deterministic_fixture_errors"]["status"] == "FAIL"


def test_gates_are_insufficient_without_controls_or_labels():
    actionable_only = [
        _case(index, actionable=True, review=True) for index in range(12)
    ]
    gates = evaluation_gates(actionable_only)
    assert gates["status"] == "INSUFFICIENT_EVIDENCE"
    assert gates["gates"]["false_positive_rate"]["status"] == "INSUFFICIENT_EVIDENCE"

    unlabelled = [
        EvaluationCase(
            **{**_case(index, actionable=True, review=True).to_dict(), "expected_review": None}
        )
        for index in range(12)
    ]
    gates = evaluation_gates(unlabelled)
    assert gates["gates"]["deterministic_fixture_errors"]["status"] == "INSUFFICIENT_EVIDENCE"
    assert gates["status"] == "INSUFFICIENT_EVIDENCE"
