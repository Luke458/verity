"""Frozen evidence benchmark: threshold selection and evidence ablation.

The benchmark runs the frozen candidate over development scenarios, selects a
materiality candidate on development evidence only, then runs exactly one final
configuration over untouched test scenarios and evaluates the registered
detection, false-positive and false-clearance gates. Qualification is built
from all measures and periods with source identities and the gate outcome; a
selection, gate or qualification failure stays visible and is never papered
over by a candidate that merely appears selected.

Evidence ablation reruns the actual deterministic provider adapter on each
independently constructed view and reports a bespoke rule baseline separately
under the explicit ``rule_baseline`` label.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .config import DatasetConfig
from .evaluation import (
    ABLATION_LEVELS,
    EvaluationCase,
    ablate_evidence,
    evaluate_materiality_candidates,
    evaluation_gates,
    evaluation_metrics,
)

BENCH_SCHEMA = 3
DEFAULT_THRESHOLDS: tuple[float, ...] = (0.0005, 0.001, 0.002)
NEEDS_REVIEW = ("INVESTIGATE", "DATA_CONTRACT_FAILURE", "INCOMPLETE")
VIEW_BUDGET_CHARS = 200_000


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
    qualification: dict[str, Any] = field(default_factory=dict)
    plan_manifest: dict[str, Any] = field(default_factory=dict)
    report: dict[str, Any] = field(default_factory=dict)
    schema_version: int = BENCH_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "suite": self.suite,
            "plan_sha256": self.plan_sha256,
            "plan_manifest": self.plan_manifest,
            "thresholds": list(self.thresholds),
            "selection": self.selection,
            "gates": self.gates,
            "ablation": self.ablation,
            "metrics": self.metrics,
            "cases": self.cases,
            "config": self.config,
            "qualification": self.qualification,
            "report": self.report,
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


def _national(result: Any, metric: str | None = None) -> Any | None:
    for item in getattr(result.temporal, "series", ()) if result.temporal else ():
        if item.series_id != "national":
            continue
        if metric is None or item.metric == metric:
            return item
    return None


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
    actionable = expected_status in NEEDS_REVIEW
    policy_review, challenger_review, effective_review = _actions(result)
    national = _national(result, config.primary_metric) or _national(result)
    recommendation = result.machine.get("provider_recommendation") or {}
    availability = result.machine.get("provider_availability") or {}
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
        challenger_status=("INVESTIGATE" if challenger_review else "PASS"),
        effective_status=result.status,
        verifier_review=policy_review,
        challenger_review=challenger_review,
        effective_review=effective_review,
        abstained=bool(recommendation.get("abstained", False))
        or str(availability.get("status", "")) in ("UNAVAILABLE", "ABSTAINED"),
        expected_review=actionable,
        cause=cause,
        expected_cause=labels.get("likely_cause"),
        severity=severity,
        expected_severity=labels.get("severity"),
        explanation_covered=bool(result.machine.get("clearance"))
        or result.status == "PASS_WITH_EXPLANATION",
        forecast_coverage=national.interval_coverage if national else None,
        interval_width=national.interval_width if national else None,
        calibrated_percentile=(national.calibrated_percentile if national else None),
        latency_ms=latency_ms,
    )


def _packages(result: Any) -> list[Any]:
    packages = list(getattr(result, "evidence_packages", ()) or ())
    if not packages and getattr(result, "evidence_package", None) is not None:
        packages = [result.evidence_package]
    return packages


def _combined_view(result: Any, level: str) -> dict[str, Any]:
    """One independently constructed view across every measure and period."""
    packages = _packages(result)
    if not packages:
        return {"abstain": True, "reason": "no evidence package recorded"}
    views = [package.provider_view(VIEW_BUDGET_CHARS, level=level) for package in packages]
    for view in views:
        if view.get("abstain"):
            return view
    merged = dict(views[0])
    merged["predictions"] = [
        prediction for view in views for prediction in view.get("predictions", ())
    ]
    merged["contracts"] = [
        check for view in views for check in view.get("contracts", ())
    ]
    merged["failed_checks"] = [
        check for view in views for check in view.get("failed_checks", ())
    ]
    merged["missing_required"] = sorted(
        {
            value
            for view in views
            for value in view.get("missing_required", ())
        }
    )
    merged["contributions"] = [
        item for view in views for item in view.get("contributions", ())
    ]
    merged["net_unexplained"] = sum(
        float(view.get("net_unexplained", 0.0)) for view in views
    )
    merged["gross_unexplained"] = sum(
        float(view.get("gross_unexplained", 0.0)) for view in views
    )
    merged["materiality_threshold"] = max(
        (float(view.get("materiality_threshold", 0.0)) for view in views),
        default=0.0,
    )
    return merged


def _qualification_entries(
    result: Any, *, scenario_id: str, split: str
) -> list[dict[str, Any]]:
    """Independent held-out coverage entries for every measure and period."""
    entries: list[dict[str, Any]] = []
    for index, package in enumerate(_packages(result)):
        for prediction in package.predictions:
            if prediction.interval_lower is None or prediction.interval_upper is None:
                continue
            if not (
                float("-inf")
                < float(prediction.interval_lower)
                <= float(prediction.interval_upper)
                < float("inf")
            ):
                continue
            covered = (
                prediction.interval_lower
                <= prediction.actual
                <= prediction.interval_upper
            )
            expected = abs(float(prediction.expected))
            width_ratio = (
                float(prediction.interval_width) / expected
                if prediction.interval_width is not None and expected > 1e-12
                else float("inf")
            )
            model_identity = json.dumps(
                package.model_identity, sort_keys=True, default=str
            )
            entries.append(
                {
                    "assessment_id": package.assessment_id,
                    "group": scenario_id,
                    "cutoff": str(package.observation_cutoff or ""),
                    "split": split,
                    "basis": "held_out_target",
                    "package": index,
                    "model": prediction.selected_model,
                    "metric": prediction.metric,
                    "level": prediction.level,
                    "horizon": int(prediction.horizon),
                    "covered": bool(covered),
                    "width_ratio": float(width_ratio),
                    "model_identity": hashlib.sha256(
                        model_identity.encode()
                    ).hexdigest()[:24],
                }
            )
    return entries


def _view_result(view: dict[str, Any], run_id: str) -> Any:
    """Reconstruct the fields the actual rule provider reads from a view.

    Derived verdicts (final status, anomaly flags, certificates) are absent by
    construction, so the provider decides only from structured evidence that a
    provider is allowed to see.
    """
    failed_checks = list(view.get("failed_checks", ()) or ())
    contract_failures = [
        check
        for check in failed_checks
        if str(check.get("outcome", "")) == "CONTRACT_FAILURE"
        or str(check.get("scope_type", "")) == "contract"
    ]
    contracts = SimpleNamespace(
        status="DATA_CONTRACT_FAILURE" if contract_failures else "PASS",
        failed=[SimpleNamespace(name=str(check.get("check", ""))) for check in contract_failures],
    )
    ledger = view.get("ledger") or {}
    net = float(ledger.get("net_unexplained", view.get("net_unexplained", 0.0)) or 0.0)
    raw = float(ledger.get("raw_movement", 0.0) or 0.0)
    materiality = float(view.get("materiality_threshold", 0.0) or 0.0)
    explained = float(view.get("explained_fraction", 1.0) or 1.0)
    attribution = SimpleNamespace(
        previous_total=abs(raw) if raw else abs(net),
        raw_delta=raw,
        unexplained_delta=net,
        explained_fraction=explained,
        material=abs(net) >= materiality,
        breadth=0.0,
        over_explained=net < -materiality if materiality else False,
        offsetting=False,
        unmatched_events=[],
        cross_metric_flags=list(view.get("cross_metric", ()) or ()),
    )
    return SimpleNamespace(
        run_id=run_id,
        contracts=contracts,
        events=[],
        attribution=attribution,
        lineage=None,
        temporal=None,
        relationships=[],
    )


def _provider_decision(view: dict[str, Any], config: DatasetConfig) -> bool:
    from .decisions import RuleDecisionProvider

    if view.get("abstain"):
        return True
    decision = RuleDecisionProvider(config).decide(_view_result(view, "ablation"))
    return bool(decision.requires_investigation)


def _review_from_view(view: dict[str, Any]) -> bool:
    """Bespoke deterministic rule baseline, explicitly labelled as such."""
    if view.get("abstain"):
        return True
    for check in list(view.get("failed_checks", ()) or ()):
        if str(check.get("outcome", "")) != "PASS":
            return True
    for check in view.get("contracts", ()) or ():
        if str(check.get("status", "")) != "PASS":
            return True
    materiality = float(view.get("materiality_threshold", 0.0) or 0.0)
    for prediction in view.get("predictions", ()) or ():
        if prediction.get("support") != "model":
            return True
        if prediction.get("calibration_status") != "OK":
            return True
        if abs(float(prediction.get("residual", 0.0))) > float(
            prediction.get("materiality", materiality)
        ):
            return True
    if abs(float(view.get("net_unexplained", 0.0))) > materiality:
        return True
    ledger = view.get("ledger") or {}
    conservation = ledger.get("conservation") or {}
    if abs(float(conservation.get("difference", 0.0))) > 1e-6 * max(
        1.0, abs(float(ledger.get("raw_movement", 0.0)))
    ):
        return True
    return False


def _split_entries(
    scenario_id: str, entries: Sequence[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    development = _is_development(scenario_id)
    return (
        list(entries) if development else [],
        [] if development else list(entries),
    )


def _is_development(scenario_id: str) -> bool:
    """Stable pre-registered split: development selects, test is untouched."""
    digest = hashlib.sha256(str(scenario_id).encode()).hexdigest()
    return int(digest[:8], 16) % 2 == 0


def _run_scenario(
    scenario_dir: Path,
    config: DatasetConfig,
) -> tuple[Any, float]:
    import time

    from qcgen.sources import ScenarioSource

    from .run import run_qc
    from .versions import select_versions

    source = ScenarioSource(scenario_dir)
    current, previous = select_versions(source.list_versions(), None, None)
    started = time.perf_counter()
    result = run_qc(source, current, previous, config)
    return result, (time.perf_counter() - started) * 1000.0


def run_evidence_bench(
    suite_dir: str | Path,
    *,
    config: DatasetConfig | None = None,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    minimum_detection_rate: float = 0.9,
    maximum_false_positive_rate: float = 0.1,
    scenario_ids: Sequence[str] | None = None,
) -> EvidenceBenchResult:
    from qcgen.oracle_vault import OracleVault, default_oracle_root

    from .cohort import code_sha256
    from .qualification import build_qualification

    config = config or DatasetConfig()
    suite_dir = Path(suite_dir)
    suite_text = (suite_dir / "suite.json").read_text()
    suite = json.loads(suite_text)
    if not config.calendar_anchor_date and suite.get("profile"):
        from qcgen.config import dataset_calendar

        config = replace(config, **dataset_calendar(str(suite["profile"])))
    vault = OracleVault(default_oracle_root(suite_dir))
    selected = set(scenario_ids) if scenario_ids else None

    scenarios: list[tuple[str, Path, dict[str, Any]]] = []
    for entry in suite.get("scenarios", []):
        scenario_id = str(entry["scenario_id"])
        if selected is not None and scenario_id not in selected:
            continue
        scenarios.append(
            (scenario_id, suite_dir / scenario_id, vault.require(scenario_id))
        )
    development = [
        item for item in scenarios if _is_development(item[0])
    ]
    test = [item for item in scenarios if not _is_development(item[0])]

    dev_outcomes: dict[float, list[EvaluationCase]] = {
        ratio: [] for ratio in thresholds
    }
    dev_entries: dict[float, list[dict[str, Any]]] = {
        ratio: [] for ratio in thresholds
    }
    case_records: list[dict[str, Any]] = []
    oracles: dict[str, str] = {}
    for scenario_id, scenario_dir, oracle in development:
        family = str(oracle.get("family", "unknown"))
        oracles[scenario_id] = str(
            oracle.get("cases", [{}])[0].get("expected_status", "PASS")
        )
        for ratio in thresholds:
            candidate_config = replace(config, materiality_ratio=ratio)
            result, latency_ms = _run_scenario(scenario_dir, candidate_config)
            evaluation = _case(
                result,
                candidate_config,
                case_id=f"{scenario_id}:{ratio}",
                family=family,
                group=scenario_id,
                oracle=oracle,
                latency_ms=latency_ms,
            )
            dev_outcomes[ratio].append(evaluation)
            dev_entries[ratio].extend(
                _qualification_entries(
                    result, scenario_id=scenario_id, split="development"
                )
            )
            case_records.append(evaluation.to_dict())

    selection = evaluate_materiality_candidates(dev_outcomes, thresholds)
    selected_candidate = selection.get("selected")
    primary = (
        float(selected_candidate["materiality_ratio"])
        if selected_candidate is not None
        else None
    )

    test_cases: list[EvaluationCase] = []
    test_entries: list[dict[str, Any]] = []
    ablation_cases: list[dict[str, Any]] = []
    if primary is not None:
        primary_config = replace(config, materiality_ratio=primary)
        for scenario_id, scenario_dir, oracle in test:
            family = str(oracle.get("family", "unknown"))
            oracles[scenario_id] = str(
                oracle.get("cases", [{}])[0].get("expected_status", "PASS")
            )
            result, latency_ms = _run_scenario(scenario_dir, primary_config)
            evaluation = _case(
                result,
                primary_config,
                case_id=f"{scenario_id}:{primary}",
                family=family,
                group=scenario_id,
                oracle=oracle,
                latency_ms=latency_ms,
            )
            test_cases.append(evaluation)
            test_entries.extend(
                _qualification_entries(result, scenario_id=scenario_id, split="test")
            )
            case_records.append(evaluation.to_dict())
        for scenario_id, scenario_dir, _ in development + test:
            result, _ = _run_scenario(scenario_dir, primary_config)
            ablation_cases.append(
                {
                    "case_id": scenario_id,
                    "actionable": str(
                        oracles.get(scenario_id, "PASS")
                    )
                    in NEEDS_REVIEW,
                    "views": {
                        level: _combined_view(result, level)
                        for level in ABLATION_LEVELS
                    },
                }
            )

    if primary is not None:
        gates = evaluation_gates(
            test_cases,
            minimum_detection_rate=minimum_detection_rate,
            maximum_false_positive_rate=maximum_false_positive_rate,
        )
        metrics = evaluation_metrics(test_cases)
    else:
        gates = evaluation_gates(
            (),
            minimum_detection_rate=minimum_detection_rate,
            maximum_false_positive_rate=maximum_false_positive_rate,
        )
        metrics = evaluation_metrics(())
    qualification_entries = (
        dev_entries.get(primary, []) if primary is not None else []
    )
    from .calendar import calendar_identity
    from .evidence_package import EVIDENCE_POLICY_VERSION

    qualification = build_qualification(
        qualification_entries,
        test_entries,
        provenance="synthetic",
        gates=gates,
        engine=code_sha256(),
        config_hash=hashlib.sha256(
            json.dumps(config.to_dict(), sort_keys=True, default=str).encode()
        ).hexdigest(),
        calendar_hash=hashlib.sha256(
            json.dumps(calendar_identity(config), sort_keys=True, default=str).encode()
        ).hexdigest(),
        evidence_policy_version=EVIDENCE_POLICY_VERSION,
    )
    ablation = ablate_evidence(
        ablation_cases,
        decide=lambda view: _provider_decision(view, config),
        baseline_decide=_review_from_view,
    )
    report = {
        "selection_status": (
            "FROZEN_CANDIDATE"
            if primary is not None
            else "NO_SAFE_CANDIDATE"
        ),
        "selected_materiality_ratio": primary,
        "detection": {
            "rate": metrics.get("detection_rate"),
            "group_rate": metrics.get("detection_group_rate"),
            "groups": metrics.get("actionable_groups"),
        },
        "review": {
            "rate": metrics.get("review_rate"),
            "false_positive_rate": metrics.get("false_positive_rate"),
            "false_clearance_rate": metrics.get("false_clearance_rate"),
            "false_clearance_upper_bound": metrics.get(
                "false_clearance_upper_bound"
            ),
        },
        "abstention": {"rate": metrics.get("abstention_rate")},
        "attribution": {
            "cause_accuracy": metrics.get("cause_accuracy"),
            "severity_accuracy": metrics.get("severity_accuracy"),
        },
        "resources": {
            "latency_ms_p50": metrics.get("latency_ms_p50"),
            "latency_ms_p95": metrics.get("latency_ms_p95"),
            "peak_memory_mb_max": metrics.get("peak_memory_mb_max"),
        },
        "gates_status": gates["status"],
        "qualification_status": qualification.status,
        "dev_scenarios": sorted(item[0] for item in development),
        "test_scenarios": sorted(item[0] for item in test),
    }
    plan_manifest = {
        "suite_sha256": hashlib.sha256(suite_text.encode()).hexdigest(),
        "engine": code_sha256(),
        "config": config.to_dict(),
        "thresholds": list(thresholds),
        "scenario_ids": sorted(oracles),
        "split": {
            scenario_id: (
                "development" if _is_development(scenario_id) else "test"
            )
            for scenario_id in sorted(oracles)
        },
        "oracle_labels": oracles,
        "evidence_schema": 3,
        "selection": {
            "selected": primary,
            "frozen": primary is not None,
        },
        "provider": "rule_adapter",
        "baseline": "rule_baseline",
    }
    plan_sha256 = hashlib.sha256(
        json.dumps(plan_manifest, sort_keys=True, default=str).encode()
    ).hexdigest()
    return EvidenceBenchResult(
        suite=str(suite_dir),
        plan_sha256=plan_sha256,
        plan_manifest=plan_manifest,
        thresholds=list(thresholds),
        selection=selection,
        gates=gates,
        ablation=ablation,
        metrics=metrics,
        cases=case_records,
        config=config.to_dict(),
        qualification=qualification.to_dict(),
        report=report,
    )
