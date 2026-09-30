"""Evidence-quality evaluation, ablation and statistical gates (schema 2).

Model quality and evidence quality are measured separately. Raw challenger
decisions are scored as model predictions; deterministic verifier decisions and
effective operational decisions are scored separately, and a policy override is
never counted as a correct model prediction. Final decisions, approval outcomes
and oracle labels are excluded from challenger inputs.

Statistical qualification computes confidence bounds over independent incident
groups: detection uses the 95% lower bound, false-positive rate the 95% upper
bound and false clearance the agreed 95% upper bound of 1% (a group fails when
any actionable member is falsely cleared). Missing controls, required classes
or sufficient independent groups yield ``INSUFFICIENT_EVIDENCE``, never a pass.
Deterministic fixtures are compared against independent oracle labels, not
agreement between two implementation outputs.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from statistics import median
from typing import Any

from .conformal import wilson_interval

EVALUATION_SCHEMA = 2
FALSE_CLEARANCE_UPPER_BOUND = 0.01
MINIMUM_ACTIONABLE_INCIDENTS = 10
MINIMUM_CONTROL_GROUPS = 2
MINIMUM_DETECTION_GROUPS = 2
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
    expected_review: bool | None = None
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
            "expected_review": self.expected_review,
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


def groups_of(cases: Sequence[EvaluationCase]) -> dict[str, list[EvaluationCase]]:
    """Independent incident groups; repeated snapshots stay together."""
    grouped: dict[str, list[EvaluationCase]] = {}
    for case in cases:
        grouped.setdefault(case.group or case.case_id, []).append(case)
    return grouped


def group_metrics(cases: Sequence[EvaluationCase]) -> dict[str, Any]:
    grouped = groups_of(cases)
    actionable_groups = [
        members
        for members in grouped.values()
        if any(case.actionable for case in members)
    ]
    control_groups = [
        members
        for members in grouped.values()
        if not any(case.actionable for case in members)
    ]
    detected_groups = sum(
        1
        for members in actionable_groups
        if all(case.effective_review for case in members if case.actionable)
    )
    cleared_groups = sum(
        1
        for members in actionable_groups
        if any(
            case.actionable and not case.effective_review for case in members
        )
    )
    false_positive_groups = sum(
        1
        for members in control_groups
        if any(case.effective_review for case in members)
    )
    return {
        "groups": len(grouped),
        "actionable_groups": len(actionable_groups),
        "control_groups": len(control_groups),
        "detected_groups": detected_groups,
        "cleared_groups": cleared_groups,
        "false_positive_groups": false_positive_groups,
        "detection_group_rate": _rate(detected_groups, len(actionable_groups)),
        "false_clearance_group_rate": _rate(cleared_groups, len(actionable_groups)),
        "false_positive_group_rate": _rate(false_positive_groups, len(control_groups)),
    }


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
    metrics = {
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
    metrics.update(group_metrics(cases))
    return metrics


def false_clearance_upper_bound(
    actionable: Sequence[EvaluationCase],
) -> dict[str, Any]:
    """95% upper confidence bound on false clearance over independent groups.

    A group fails when any actionable member is falsely cleared.
    """
    grouped = groups_of(actionable)
    total = len(grouped)
    cleared = sum(
        1
        for members in grouped.values()
        if any(case.actionable and not case.effective_review for case in members)
    )
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


def _bound_gate(
    *,
    status: str,
    reason: str,
    value: float | None,
    bound: float | None,
    direction: str,
    groups: int,
) -> dict[str, Any]:
    return {
        "status": status,
        "reason": reason,
        "value": value,
        "bound": bound,
        "direction": direction,
        "groups": groups,
    }


def confidence_gates(
    cases: Sequence[EvaluationCase],
    *,
    minimum_detection_rate: float = 0.9,
    maximum_false_positive_rate: float = 0.1,
) -> dict[str, Any]:
    """Group-level confidence-bound gates for detection, FPR and clearance."""
    metrics = group_metrics(cases)
    actionable_groups = int(metrics["actionable_groups"])
    control_groups = int(metrics["control_groups"])
    detection_gate: dict[str, Any]
    if actionable_groups < MINIMUM_DETECTION_GROUPS:
        detection_gate = _bound_gate(
            status="INSUFFICIENT_EVIDENCE",
            reason="too few independent actionable incident groups",
            value=metrics["detection_group_rate"],
            bound=None,
            direction="lower",
            groups=actionable_groups,
        )
    else:
        lower, _ = wilson_interval(
            int(metrics["detected_groups"]), actionable_groups
        )
        detection_gate = _bound_gate(
            status="PASS" if lower >= minimum_detection_rate else "FAIL",
            reason="",
            value=lower,
            bound=minimum_detection_rate,
            direction="lower",
            groups=actionable_groups,
        )
    fpr_gate: dict[str, Any]
    if control_groups < MINIMUM_CONTROL_GROUPS:
        fpr_gate = _bound_gate(
            status="INSUFFICIENT_EVIDENCE",
            reason="too few independent control groups",
            value=metrics["false_positive_group_rate"],
            bound=None,
            direction="upper",
            groups=control_groups,
        )
    else:
        _, upper = wilson_interval(
            int(metrics["false_positive_groups"]), control_groups
        )
        fpr_gate = _bound_gate(
            status="PASS" if upper <= maximum_false_positive_rate else "FAIL",
            reason="",
            value=upper,
            bound=maximum_false_positive_rate,
            direction="upper",
            groups=control_groups,
        )
    bound = false_clearance_upper_bound(
        [case for case in cases if case.actionable]
    )
    clearance_gate = {
        "status": (
            "PASS"
            if bound["passes"]
            else ("INSUFFICIENT_EVIDENCE" if bound["status"] == "INSUFFICIENT_EVIDENCE" else "FAIL")
        ),
        "bound": bound,
    }
    return {
        "detection_rate": detection_gate,
        "false_positive_rate": fpr_gate,
        "false_clearance": clearance_gate,
    }


def independent_label_errors(
    cases: Sequence[EvaluationCase],
) -> dict[str, Any]:
    """Effective review decisions that contradict independent oracle labels."""
    labelled = [
        case
        for case in cases
        if case.expected_review is not None and case.recorded
    ]
    errors = [
        case.case_id
        for case in labelled
        if bool(case.effective_review) != bool(case.expected_review)
    ]
    if not labelled:
        return {
            "status": "INSUFFICIENT_EVIDENCE",
            "reason": "no independent oracle labels",
            "errors": [],
            "labelled": 0,
        }
    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "labelled": len(labelled),
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
    bounds = confidence_gates(
        cases,
        minimum_detection_rate=minimum_detection_rate,
        maximum_false_positive_rate=maximum_false_positive_rate,
    )
    gates: dict[str, Any] = {
        "deterministic_fixture_errors": independent_label_errors(cases),
        "detection_rate": bounds["detection_rate"],
        "false_positive_rate": bounds["false_positive_rate"],
        "false_clearance": bounds["false_clearance"],
    }
    if not require_false_clearance_bound:
        gates["false_clearance"]["status"] = "SKIPPED"
    statuses = [gate["status"] for gate in gates.values()]
    overall = all(status in ("PASS", "SKIPPED") for status in statuses)
    insufficient = any(status == "INSUFFICIENT_EVIDENCE" for status in statuses)
    return {
        "schema_version": EVALUATION_SCHEMA,
        "status": "PASS" if overall else ("INSUFFICIENT_EVIDENCE" if insufficient else "FAIL"),
        "gates": gates,
        "metrics": metrics,
    }


ABLATION_LEVELS = ("summary", "forecast", "hierarchy", "complete")


def _ablation_summary(
    cases: Sequence[dict[str, Any]],
    level: str,
    decide: Callable[[Any], bool],
) -> dict[str, Any] | None:
    selected: list[dict[str, Any]] = []
    review_flags: list[bool] = []
    for case in cases:
        views = case.get("views") or {}
        if level not in views:
            continue
        selected.append(case)
        review_flags.append(bool(decide(views[level])))
    if not selected:
        return None
    actionable = [case for case in selected if case.get("actionable")]
    reviews = sum(1 for flag in review_flags if flag)
    cleared = sum(
        1
        for case, flag in zip(selected, review_flags, strict=True)
        if case.get("actionable") and not flag
    )
    return {
        "cases": len(selected),
        "review_rate": _rate(reviews, len(selected)),
        "false_clearance_rate": _rate(cleared, len(actionable)),
    }


def ablate_evidence(
    cases: Sequence[dict[str, Any]],
    decide: Callable[[Any], bool] | None = None,
    baseline_decide: Callable[[Any], bool] | None = None,
) -> dict[str, Any]:
    """Compare review decisions under independently constructed evidence views.

    With ``decide`` the cases carry a ``views`` mapping for each level and the
    actual provider adapter is rerun on the view; otherwise precomputed boolean
    levels are summarized. A ``baseline_decide`` is reported separately and
    explicitly labelled ``rule_baseline``: a bespoke rule is never presented as
    a provider result. Case membership, actionable labels and measurement
    conditions stay fixed across levels.
    """
    results: dict[str, Any] = {}
    baseline: dict[str, Any] = {}
    for level in ABLATION_LEVELS:
        if decide is not None:
            summary = _ablation_summary(cases, level, decide)
            if summary is not None:
                results[level] = summary
        else:
            selected: list[dict[str, Any]] = []
            review_flags: list[bool] = []
            for case in cases:
                levels = case.get("levels") or {}
                if levels.get(level) is None:
                    continue
                selected.append(case)
                review_flags.append(bool(levels[level]))
            if selected:
                actionable = [
                    case for case in selected if case.get("actionable")
                ]
                results[level] = {
                    "cases": len(selected),
                    "review_rate": _rate(
                        sum(1 for flag in review_flags if flag), len(selected)
                    ),
                    "false_clearance_rate": _rate(
                        sum(
                            1
                            for case, flag in zip(selected, review_flags, strict=True)
                            if case.get("actionable") and not flag
                        ),
                        len(actionable),
                    ),
                }
        if baseline_decide is not None:
            baseline_summary = _ablation_summary(
                cases, level, baseline_decide
            )
            if baseline_summary is not None:
                baseline[level] = baseline_summary
    payload: dict[str, Any] = {
        "schema_version": EVALUATION_SCHEMA,
        "levels": list(ABLATION_LEVELS),
        "results": results,
        "mode": "provider_rerun" if decide is not None else "precomputed",
    }
    if baseline_decide is not None:
        payload["baseline"] = {
            "label": "rule_baseline",
            "results": baseline,
        }
    return payload


def evaluate_materiality_candidates(
    outcomes: Mapping[float, Sequence[EvaluationCase]],
    candidates: Sequence[float],
) -> dict[str, Any]:
    """Choose the candidate minimizing review volume while passing safety gates.

    ``outcomes`` maps each candidate ratio to the evaluated cases under that
    frozen configuration. A candidate is safe only when no actionable incident
    is cleared, the false-clearance confidence bound passes, and detection
    confidence bounds are not insufficient.
    """
    evaluated: list[dict[str, Any]] = []
    for candidate in candidates:
        cases = outcomes.get(candidate, ())
        metrics = evaluation_metrics(cases)
        bounds = confidence_gates(cases)
        bound = metrics["false_clearance_upper_bound"]
        detection_ok = bounds["detection_rate"]["status"] == "PASS"
        safe = (
            not any(
                case.actionable and not case.effective_review for case in cases
            )
            and bound["passes"]
            and detection_ok
        )
        evaluated.append(
            {
                "materiality_ratio": candidate,
                "safe": safe,
                "detection_gate": bounds["detection_rate"],
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
