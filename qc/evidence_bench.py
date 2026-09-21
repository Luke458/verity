"""Frozen evidence benchmark: threshold selection and evidence ablation.

The benchmark runs one frozen configuration per materiality candidate over
pre-registered scenarios, scores raw challenger decisions separately from
deterministic verifier and effective operational decisions, and reports the
false-clearance confidence bound, review volume and an evidence ablation.
The plan hash pins the frozen benchmark; later runs must repeat it unchanged.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .config import DatasetConfig
from .evaluation import (
    EvaluationCase,
    ablate_evidence,
    evaluate_materiality_candidates,
    evaluation_gates,
    evaluation_metrics,
)

BENCH_SCHEMA = 1
DEFAULT_THRESHOLDS: tuple[float, ...] = (0.0005, 0.001, 0.002)
NEEDS_REVIEW = ("INVESTIGATE", "DATA_CONTRACT_FAILURE", "INCOMPLETE")


@dataclass
class EvidenceBenchResult:
    suite: str
    plan_sha256: str
    thresholds: list[float]
    selection: dict[str, Any]
    gates: dict[str, Any]
    ablation: dict[str, Any]
    metrics: dict[str, Any]
    cases: list[dict[str, Any]] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    schema_version: int = BENCH_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "suite": self.suite,
            "plan_sha256": self.plan_sha256,
            "thresholds": list(self.thresholds),
            "selection": self.selection,
            "gates": self.gates,
            "ablation": self.ablation,
            "metrics": self.metrics,
            "cases": self.cases,
            "config": self.config,
        }


def _actions(result: Any) -> tuple[bool, bool, bool]:
    machine = result.machine
    policy_review = bool(
        machine.get(
            "policy_required_review",
            result.status not in ("PASS", "PASS_WITH_EXPLANATION"),
        )
    )
    effective_review = bool(
        machine.get(
            "requires_investigation",
            result.status not in ("PASS", "PASS_WITH_EXPLANATION"),
        )
    )
    recommendation = machine.get("provider_recommendation") or {}
    challenger_review = bool(recommendation.get("requires_investigation", effective_review))
    return policy_review, challenger_review, effective_review


def _ablation_levels(result: Any, config: DatasetConfig) -> dict[str, bool]:
    machine = result.machine
    historical = (machine.get("historical_revision") or {}).get("status")
    summary = historical == "INVESTIGATE" or result.contracts.status == "DATA_CONTRACT_FAILURE"
    forecast = summary or bool(result.temporal is not None and result.temporal.anomaly)
    hierarchy_fail = any(
        check.status == "FAIL" for check in getattr(result, "hierarchy", ())
    )
    unsupported_material = False
    ledger = getattr(result, "ledger", None)
    if ledger is not None:
        supported = sum(
            item.value
            for item in ledger.contributions
            if item.scope == "overlap" and item.support in ("statistical", "approved_event")
        )
        unsupported_material = (
            abs(ledger.overlap_delta - supported)
            > machine.get("materiality_threshold", 0.0)
        )
    hierarchy = forecast or hierarchy_fail or unsupported_material
    complete = bool(machine.get("requires_investigation"))
    return {
        "summary": summary,
        "forecast": forecast,
        "hierarchy": hierarchy,
        "complete": complete,
    }


def _case(
    result: Any,
    config: DatasetConfig,
    *,
    case_id: str,
    family: str,
    group: str,
    oracle: dict[str, Any],
    latency_ms: float | None,
) -> EvaluationCase:
    from .labels import oracle_labels_for_result

    labels = oracle_labels_for_result(result, oracle).labels
    expected_status = str(oracle.get("cases", [{}])[0].get("expected_status", "PASS"))
    actionable = expected_status in ("INVESTIGATE", "DATA_CONTRACT_FAILURE")
    policy_review, challenger_review, effective_review = _actions(result)
    national = next(
        (
            item
            for item in (result.temporal.series if result.temporal else ())
            if item.series_id == "national"
        ),
        None,
    )
    recommendation = result.machine.get("provider_recommendation") or {}
    cause = None
    severity = None
    if result.decisions is not None:
        cause_value = result.decisions.get("likely_cause")
        severity_value = result.decisions.get("severity")
        cause = cause_value.value if cause_value else None
        severity = severity_value.value if severity_value else None
    return EvaluationCase(
        case_id=case_id,
        family=family,
        group=group,
        actionable=actionable,
        verifier_status=result.status,
        challenger_status=(
            "INVESTIGATE" if challenger_review else "PASS"
        ),
        effective_status=result.status,
        verifier_review=policy_review,
        challenger_review=challenger_review,
        effective_review=effective_review,
        abstained=bool(recommendation.get("abstained", False)),
        cause=cause,
        expected_cause=labels.get("likely_cause"),
        severity=severity,
        expected_severity=labels.get("severity"),
        explanation_covered=bool(result.machine.get("clearance"))
        or result.status == "PASS_WITH_EXPLANATION",
        forecast_coverage=national.interval_coverage if national else None,
        interval_width=national.interval_width if national else None,
        calibrated_percentile=(
            national.calibrated_percentile if national else None
        ),
        latency_ms=latency_ms,
    )


def run_evidence_bench(
    suite_dir: str | Path,
    *,
    config: DatasetConfig | None = None,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    minimum_detection_rate: float = 0.9,
    maximum_false_positive_rate: float = 0.1,
    scenario_ids: Sequence[str] | None = None,
) -> EvidenceBenchResult:
    import time

    from qcgen.oracle_vault import OracleVault, default_oracle_root
    from qcgen.sources import ScenarioSource

    from .run import run_qc
    from .versions import select_versions

    config = config or DatasetConfig()
    suite_dir = Path(suite_dir)
    suite_text = (suite_dir / "suite.json").read_text()
    suite = json.loads(suite_text)
    if not config.calendar_anchor_date and suite.get("profile"):
        from qcgen.config import dataset_calendar

        config = replace(config, **dataset_calendar(str(suite["profile"])))
    plan_sha256 = hashlib.sha256(suite_text.encode()).hexdigest()
    vault = OracleVault(default_oracle_root(suite_dir))
    selected = set(scenario_ids) if scenario_ids else None

    outcomes: dict[float, list[EvaluationCase]] = {ratio: [] for ratio in thresholds}
    ablation_cases: list[dict[str, Any]] = []
    case_records: list[dict[str, Any]] = []
    for entry in suite.get("scenarios", []):
        scenario_id = str(entry["scenario_id"])
        if selected is not None and scenario_id not in selected:
            continue
        scenario_dir = suite_dir / scenario_id
        oracle = vault.require(scenario_id)
        family = str(oracle.get("family", entry.get("family", "unknown")))
        for ratio in thresholds:
            candidate_config = replace(config, materiality_ratio=ratio)
            source = ScenarioSource(scenario_dir)
            current, previous = select_versions(
                source.list_versions(), None, None
            )
            started = time.perf_counter()
            result = run_qc(source, current, previous, candidate_config)
            latency_ms = (time.perf_counter() - started) * 1000.0
            evaluation = _case(
                result,
                candidate_config,
                case_id=f"{scenario_id}:{ratio}",
                family=family,
                group=scenario_id,
                oracle=oracle,
                latency_ms=latency_ms,
            )
            outcomes[ratio].append(evaluation)
            case_records.append(evaluation.to_dict())
            if ratio == thresholds[0]:
                ablation_cases.append(
                    {
                        "case_id": scenario_id,
                        "actionable": evaluation.actionable,
                        "levels": _ablation_levels(result, candidate_config),
                    }
                )

    selection = evaluate_materiality_candidates(outcomes, thresholds)
    primary = selection["selected"]["materiality_ratio"] if selection["selected"] else thresholds[0]
    gates = evaluation_gates(
        outcomes[primary],
        minimum_detection_rate=minimum_detection_rate,
        maximum_false_positive_rate=maximum_false_positive_rate,
    )
    return EvidenceBenchResult(
        suite=str(suite_dir),
        plan_sha256=plan_sha256,
        thresholds=list(thresholds),
        selection=selection,
        gates=gates,
        ablation=ablate_evidence(ablation_cases),
        metrics=evaluation_metrics(outcomes[primary]),
        cases=case_records,
        config=config.to_dict(),
    )
