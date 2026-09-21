"""Evidence-quality evaluation, ablation and statistical gates.

Model quality and evidence quality are measured separately. Raw challenger
decisions are scored as model predictions; deterministic verifier decisions and
effective operational decisions are scored separately, and a policy override is
never counted as a correct model prediction. Final decisions, approval outcomes
and oracle labels are excluded from challenger inputs.

Statistical qualification keeps the detection/false-positive confidence-bound
gates and adds a 95% upper confidence bound of 1% on false clearance of
actionable incidents. Too few independent incidents stays insufficient
evidence; synthetic success never grants production eligibility.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import median
from typing import Any

from .conformal import wilson_interval

EVALUATION_SCHEMA = 1
FALSE_CLEARANCE_UPPER_BOUND = 0.01
MINIMUM_ACTIONABLE_INCIDENTS = 10
NEEDS_REVIEW = ("INVESTIGATE", "DATA_CONTRACT_FAILURE", "INCOMPLETE")


@dataclass
class EvaluationCase:
    """One incident-grouped assessment with its oracle label and decisions."""

    case_id: str
    family: str
    group: str
    actionable: bool
    verifier_status: str
    challenger_status: str
    effective_status: str
    verifier_review: bool
    challenger_review: bool
    effective_review: bool
    recorded: bool = True
    abstained: bool = False
    cause: str | None = None
    expected_cause: str | None = None
    severity: str | None = None
    expected_severity: str | None = None
    explanation_covered: bool = False
    forecast_coverage: float | None = None
    interval_width: float | None = None
    calibrated_percentile: float | None = None
    latency_ms: float | None = None
    peak_memory_mb: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "family": self.family,
            "group": self.group,
            "actionable": self.actionable,
            "verifier_status": self.verifier_status,
            "challenger_status": self.challenger_status,
            "effective_status": self.effective_status,
            "verifier_review": self.verifier_review,
            "challenger_review": self.challenger_review,
            "effective_review": self.effective_review,
            "recorded": self.recorded,
            "abstained": self.abstained,
            "cause": self.cause,
            "expected_cause": self.expected_cause,
            "severity": self.severity,
            "expected_severity": self.expected_severity,
            "explanation_covered": self.explanation_covered,
            "forecast_coverage": self.forecast_coverage,
            "interval_width": self.interval_width,
            "calibrated_percentile": self.calibrated_percentile,
            "latency_ms": self.latency_ms,
            "peak_memory_mb": self.peak_memory_mb,
        }


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return float(ordered[index])


def evaluation_metrics(cases: Sequence[EvaluationCase]) -> dict[str, Any]:
    """Detection, clearance, review-volume and quality metrics."""
    actionable = [case for case in cases if case.actionable]
    quiet = [case for case in cases if not case.actionable]
    detected = [case for case in actionable if case.effective_review]
    cleared_actionable = [case for case in actionable if not case.effective_review]
    false_positive = [case for case in quiet if case.effective_review]
    raw_detected = [case for case in actionable if case.challenger_review]
    raw_false_positive = [case for case in quiet if case.challenger_review]
    cause_matches = [
        case
        for case in cases
        if case.expected_cause is not None
        and case.cause is not None
        and case.cause == case.expected_cause
    ]
    cause_evaluated = [
        case for case in cases if case.expected_cause is not None and case.cause is not None
    ]
    severity_matches = [
        case
        for case in cases
        if case.expected_severity is not None
        and case.severity is not None
        and case.severity == case.expected_severity
    ]
    severity_evaluated = [
        case
        for case in cases
        if case.expected_severity is not None and case.severity is not None
    ]
    coverage = [
        case.forecast_coverage
        for case in cases
        if case.forecast_coverage is not None
    ]
    widths = [case.interval_width for case in cases if case.interval_width is not None]
    latencies = [case.latency_ms for case in cases if case.latency_ms is not None]
    memories = [
        case.peak_memory_mb for case in cases if case.peak_memory_mb is not None
    ]
    return {
        "schema_version": EVALUATION_SCHEMA,
        "cases": len(cases),
        "actionable_cases": len(actionable),
        "quiet_cases": len(quiet),
        "detection_rate": _rate(len(detected), len(actionable)),
        "false_clearance_rate": _rate(len(cleared_actionable), len(actionable)),
        "false_positive_rate": _rate(len(false_positive), len(quiet)),
        "review_rate": _rate(sum(1 for case in cases if case.effective_review), len(cases)),
        "raw_challenger_detection_rate": _rate(len(raw_detected), len(actionable)),
        "raw_challenger_false_positive_rate": _rate(
            len(raw_false_positive), len(quiet)
        ),
        "raw_policy_overrides": sum(
            1
            for case in cases
            if case.challenger_review != case.effective_review
        ),
        "abstention_rate": _rate(sum(1 for case in cases if case.abstained), len(cases)),
        "explanation_coverage": _rate(
            sum(1 for case in cases if case.explanation_covered), len(cases)
        ),
        "cause_accuracy": _rate(len(cause_matches), len(cause_evaluated)),
        "severity_accuracy": _rate(len(severity_matches), len(severity_evaluated)),
        "forecast_coverage_mean": (
            sum(coverage) / len(coverage) if coverage else None
        ),
        "interval_width_mean": sum(widths) / len(widths) if widths else None,
        "interval_width_median": median(widths) if widths else None,
        "latency_ms_p50": _percentile(latencies, 0.5),
        "latency_ms_p95": _percentile(latencies, 0.95),
        "peak_memory_mb_max": max(memories) if memories else None,
        "false_clearance_upper_bound": false_clearance_upper_bound(actionable),
    }


def false_clearance_upper_bound(
    actionable: Sequence[EvaluationCase],
) -> dict[str, Any]:
    """95% upper confidence bound on false clearance of actionable incidents."""
    total = len(actionable)
    cleared = sum(1 for case in actionable if not case.effective_review)
    if total < MINIMUM_ACTIONABLE_INCIDENTS:
        return {
            "status": "INSUFFICIENT_EVIDENCE",
            "incidents": total,
            "cleared": cleared,
            "minimum_incidents": MINIMUM_ACTIONABLE_INCIDENTS,
            "upper_bound": None,
            "passes": False,
        }
    _, upper = wilson_interval(cleared, total)
    return {
        "status": "EVALUATED",
        "incidents": total,
        "cleared": cleared,
        "minimum_incidents": MINIMUM_ACTIONABLE_INCIDENTS,
        "upper_bound": upper,
        "bound": FALSE_CLEARANCE_UPPER_BOUND,
        "passes": upper <= FALSE_CLEARANCE_UPPER_BOUND,
    }


def evaluation_gates(
    cases: Sequence[EvaluationCase],
    *,
    minimum_detection_rate: float = 0.9,
    maximum_false_positive_rate: float = 0.1,
    require_false_clearance_bound: bool = True,
) -> dict[str, Any]:
    """Engineering and statistical gates for a frozen evaluation."""
    metrics = evaluation_metrics(cases)
    deterministic_errors = [
        case.case_id
        for case in cases
        if case.recorded and case.effective_review != case.verifier_review
    ]
    gates: dict[str, Any] = {
        "deterministic_fixture_errors": {
            "status": "PASS" if not deterministic_errors else "FAIL",
            "errors": deterministic_errors,
        },
        "detection_rate": {
            "status": (
                "PASS"
                if (metrics["detection_rate"] or 0.0) >= minimum_detection_rate
                else "FAIL"
            ),
            "value": metrics["detection_rate"],
            "minimum": minimum_detection_rate,
        },
        "false_positive_rate": {
            "status": (
                "PASS"
                if (metrics["false_positive_rate"] or 0.0)
                <= maximum_false_positive_rate
                else "FAIL"
            ),
            "value": metrics["false_positive_rate"],
            "maximum": maximum_false_positive_rate,
        },
        "false_clearance": {
            "status": (
                "PASS"
                if metrics["false_clearance_upper_bound"]["passes"]
                else "FAIL"
            ),
            "bound": metrics["false_clearance_upper_bound"],
        },
    }
    if not require_false_clearance_bound:
        gates["false_clearance"]["status"] = "SKIPPED"
    overall = all(
        gate["status"] in ("PASS", "SKIPPED") for gate in gates.values()
    )
    return {
        "schema_version": EVALUATION_SCHEMA,
        "status": "PASS" if overall else "FAIL",
        "gates": gates,
        "metrics": metrics,
    }


ABLATION_LEVELS = ("summary", "forecast", "hierarchy", "complete")


def ablate_evidence(
    cases: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Compare review decisions under progressively complete evidence.

    Each case provides booleans for the evidence available at each level; the
    comparison reports review volume and false clearance per ablation so the
    evaluation can see what each evidence layer adds.
    """
    results: dict[str, Any] = {}
    for level in ABLATION_LEVELS:
        selected = [case for case in cases if case.get("levels", {}).get(level) is not None]
        actionable = [case for case in selected if case.get("actionable")]
        reviews = sum(
            1 for case in selected if case["levels"][level]
        )
        cleared = sum(
            1 for case in actionable if not case["levels"][level]
        )
        results[level] = {
            "cases": len(selected),
            "review_rate": _rate(reviews, len(selected)),
            "false_clearance_rate": _rate(cleared, len(actionable)),
        }
    return {
        "schema_version": EVALUATION_SCHEMA,
        "levels": list(ABLATION_LEVELS),
        "results": results,
    }


def evaluate_materiality_candidates(
    outcomes: Mapping[float, Sequence[EvaluationCase]],
    candidates: Sequence[float],
) -> dict[str, Any]:
    """Choose the candidate minimizing review volume while passing safety gates.

    ``outcomes`` maps each candidate ratio to the evaluated cases under that
    frozen configuration. A candidate is safe only when it clears no actionable
    incident and honours the false-clearance confidence bound.
    """
    evaluated: list[dict[str, Any]] = []
    for candidate in candidates:
        cases = outcomes.get(candidate, ())
        metrics = evaluation_metrics(cases)
        bound = metrics["false_clearance_upper_bound"]
        safe = not any(
            case.actionable and not case.effective_review for case in cases
        ) and bound["passes"]
        evaluated.append(
            {
                "materiality_ratio": candidate,
                "safe": safe,
                "review_rate": metrics["review_rate"],
                "detection_rate": metrics["detection_rate"],
                "false_clearance_rate": metrics["false_clearance_rate"],
                "false_clearance_upper_bound": bound,
            }
        )
    safe = [item for item in evaluated if item["safe"]]
    selected = (
        min(
            safe,
            key=lambda item: (
                item["review_rate"] if item["review_rate"] is not None else math.inf,
                item["materiality_ratio"],
            ),
        )
        if safe
        else None
    )
    return {
        "schema_version": EVALUATION_SCHEMA,
        "candidates": evaluated,
        "selected": selected,
    }
