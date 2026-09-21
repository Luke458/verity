from __future__ import annotations

import pytest

from qc.evaluation import (
    EvaluationCase,
    ablate_evidence,
    evaluate_materiality_candidates,
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
    cases[0] = _case(99, actionable=True, review=False)
    gates = evaluation_gates(cases)
    assert gates["status"] == "FAIL"
    assert gates["gates"]["false_clearance"]["status"] == "FAIL"

    deterministic = [_case(index, actionable=True, review=True) for index in range(12)]
    deterministic[0] = EvaluationCase(
        **{
            **_case(0, actionable=True, review=True).to_dict(),
            "effective_review": False,
        }
    )
    gates = evaluation_gates(deterministic)
    assert gates["gates"]["deterministic_fixture_errors"]["status"] == "FAIL"


def test_ablation_reports_review_volume_per_level():
    cases = [
        {
            "case_id": "a",
            "actionable": True,
            "levels": {
                "summary": False,
                "forecast": True,
                "hierarchy": True,
                "complete": True,
            },
        },
        {
            "case_id": "b",
            "actionable": False,
            "levels": {
                "summary": False,
                "forecast": False,
                "hierarchy": True,
                "complete": True,
            },
        },
    ]
    ablation = ablate_evidence(cases)
    assert ablation["results"]["summary"]["review_rate"] == 0.0
    assert ablation["results"]["forecast"]["review_rate"] == 0.5
    assert ablation["results"]["forecast"]["false_clearance_rate"] == 0.0


def test_materiality_selection_prefers_low_review_volume_when_safe():
    def outcomes(clear_actionable: bool):
        cases = [_case(index, actionable=True, review=True) for index in range(400)]
        if clear_actionable:
            cases[0] = _case(0, actionable=True, review=False)
        cases += [_case(500 + index, actionable=False, review=False) for index in range(8)]
        return cases

    selection = evaluate_materiality_candidates(
        {0.0005: outcomes(False), 0.001: outcomes(False), 0.002: outcomes(True)},
        (0.0005, 0.001, 0.002),
    )
    assert selection["selected"]["materiality_ratio"] == 0.0005
    assert selection["candidates"][2]["safe"] is False
