"""QC run orchestration.

``run_qc`` ties the deterministic layers together:

    contracts -> version pair -> revision cubes -> lifecycle -> attribution

and returns a machine-readable result following section 77 of the architecture
document. It never falls back to a model: semantic decisions are Milestone D.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from .attribution import AttributionResult, classify_run, explain_revision
from .config import DatasetConfig
from .contracts import ContractCheck, ContractResult, validate_contracts
from .counterfactual import CounterfactualResult, reconstruct_counterfactual
from .decisions import DecisionProvider, DecisionSet, RuleDecisionProvider
from .evidence import EvidenceGraph, build_evidence_graph
from .hierarchy import HierarchyCheck, hierarchy_checks
from .ledger import ExplanationLedger, build_explanation_ledger
from .lifecycle import (
    LifecycleEvent,
    classify_entity_changes,
    detect_reclassification,
)
from .lineage import LineageResult, analyze_lineage
from .reconciliation import ReconciliationResult, run_reconciliation
from .reference import ReferenceSpec, compare_reference, reference_mismatches
from .registry import RegistryStore, as_registry
from .relationships import EntityRelationship, detect_relationships
from .revision import base_keys, build_revision_cube, headline_keys
from .source import VersionSource
from .temporal import TemporalResult, run_temporal_qc
from .versions import VersionPair, build_version_pair

_EVENT_REASONS = {
    "LATEST_WEEK_MISSING": "latest_week_missing",
    "ENTITY_HISTORY_TRUNCATED": "history_truncated",
    "NEW_ENTITY_HISTORICAL_BACKFILL": "historical_backfill",
    "ENTITY_REMOVED": "entity_removed",
    "ENTITY_HISTORY_EXTENDED": "history_extended",
    "POSSIBLE_RECLASSIFICATION": "possible_reclassification",
    "NEW_ENTITY_RECENT": "new_entity",
}


@dataclass
class QCRunResult:
    run_id: str
    dataset: str
    status: str
    contracts: ContractResult
    version_pair: VersionPair | None
    cubes: dict[str, pd.DataFrame] = field(default_factory=dict)
    events: list[LifecycleEvent] = field(default_factory=list)
    attribution: AttributionResult | None = None
    counterfactual: CounterfactualResult | None = None
    reconciliation: ReconciliationResult | None = None
    lineage: LineageResult | None = None
    temporal: TemporalResult | None = None
    evidence: EvidenceGraph | None = None
    decisions: DecisionSet | None = None
    relationships: list[EntityRelationship] = field(default_factory=list)
    reference: dict[str, Any] | None = None
    expectations: Sequence[Any] = ()
    observed_at: str | None = None
    temporal_required: bool = False
    input_findings: list[dict[str, Any]] = field(default_factory=list)
    snapshot_manifest: dict[str, Any] | None = None
    reasons: list[str] = field(default_factory=list)
    hierarchy: list[HierarchyCheck] = field(default_factory=list)
    ledger: ExplanationLedger | None = None
    period_ledgers: dict[int, ExplanationLedger] = field(default_factory=dict)
    metric_ledgers: dict[str, ExplanationLedger] = field(default_factory=dict)
    period_metric_ledgers: dict[str, ExplanationLedger] = field(
        default_factory=dict
    )
    evidence_package: Any | None = None
    evidence_packages: list[Any] = field(default_factory=list)
    certificates: tuple[Any, ...] = ()
    assessment_id: str | None = None
    machine: dict[str, Any] = field(default_factory=dict)


def _read_dim(
    source: VersionSource, version: str, name: str
) -> pd.DataFrame | None:
    reader = getattr(source, "read_dim", None)
    if reader is None:
        return None
    return reader(version, name)


class _TemporalPeriod:
    """Minimal single-metric/period view for the period explanation ledger."""

    def __init__(self, temporal: Any, target: int, metric: str | None = None):
        self.target_week = int(target)
        self.series = [
            item
            for item in getattr(temporal, "series", ())
            if int(item.target_week) == int(target)
            and (metric is None or getattr(item, "metric", "") == metric)
        ]


def _source_provenance(source: VersionSource) -> str:
    """Real Delta tables are real-data assessments; fixture sources are not."""
    from .delta import DeltaSource

    raw: Any = source
    for _ in range(4):
        inner = getattr(raw, "source", None) or getattr(raw, "inner", None)
        if inner is None:
            break
        raw = inner
    return "real" if isinstance(raw, DeltaSource) else "synthetic"


def _resolve_common_stage(
    source: VersionSource,
    previous_id: str,
    current_id: str,
    preferred: tuple[str, ...],
) -> str:
    """Resolve one stage available in both versions, in preference order."""
    previous_available = set(source.available_stages(previous_id))
    current_available = set(source.available_stages(current_id))
    for stage in preferred:
        if stage in previous_available and stage in current_available:
            return stage
    common = sorted(previous_available & current_available)
    if not common:
        raise ValueError(
            f"no common stage between {previous_id!r} and {current_id!r}"
        )
    return common[-1]


def _reasons(
    contracts: ContractResult,
    events: list[LifecycleEvent],
    attribution: AttributionResult | None,
    config: DatasetConfig,
    status: str,
    relationships: Sequence[EntityRelationship] = (),
) -> list[str]:
    reasons: list[str] = []
    for check in contracts.failed:
        reasons.append(f"contract_failure:{check.name}")
    if attribution is None:
        return reasons

    for event in events:
        prefix = _EVENT_REASONS.get(event.classification)
        if prefix:
            suffix = event.entity_id if event.entity_type == "commodity" else f"{event.entity_type}:{event.entity_id}"
            reasons.append(f"{prefix}:{suffix}")

    if (
        attribution.material
        and attribution.explained_fraction < config.explained_fraction_threshold
        and not attribution.structural_events
    ):
        reasons.append("unexplained_value_change")
    if attribution.over_explained:
        reasons.append("over_explained_revision")
    if attribution.offsetting:
        reasons.append("offsetting_explanations")
    if (
        abs(attribution.raw_delta) <= max(config.materiality_abs, config.materiality_ratio * abs(attribution.previous_total))
        and attribution.breadth > config.broad_recalculation_breadth
    ):
        reasons.append("broad_historical_recalculation")
    reasons.extend(attribution.cross_metric_flags)
    if status == "PASS_WITH_EXPLANATION":
        reasons.append("expected_event_matched")
    for relationship in relationships:
        reasons.append(
            f"possible_replacement:{relationship.source_id}->{relationship.target_id}"
        )

    seen: set[str] = set()
    ordered: list[str] = []
    for reason in reasons:
        if reason not in seen:
            seen.add(reason)
            ordered.append(reason)
    return ordered


def _machine(
    run_id: str,
    dataset: str,
    status: str,
    contracts: ContractResult,
    pair: VersionPair | None,
    events: list[LifecycleEvent],
    attribution: AttributionResult | None,
    reasons: list[str],
    counterfactual: CounterfactualResult | None = None,
    reconciliation: ReconciliationResult | None = None,
    lineage: LineageResult | None = None,
    historical_status: str | None = None,
    temporal: TemporalResult | None = None,
    evidence: EvidenceGraph | None = None,
    relationships: Sequence[EntityRelationship] = (),
    reference: dict[str, Any] | None = None,
) -> dict[str, Any]:
    historical: dict[str, Any] | None = None
    if attribution is not None:
        historical = {
            "status": historical_status or status,
            "raw_delta": attribution.raw_delta,
            "explained_delta": attribution.explained_delta,
            "unexplained_delta": attribution.unexplained_delta,
            "explained_fraction": attribution.explained_fraction,
            "material": attribution.material,
            "breadth": attribution.breadth,
            "over_explained": attribution.over_explained,
            "offsetting": attribution.offsetting,
            "matched_expected_events": list(attribution.matched_event_ids),
            "conservation": dict(attribution.conservation),
        }
    return {
        "run_id": run_id,
        "dataset": dataset,
        "status": status,
        "historical_revision": historical,
        "latest_week": temporal.to_dict() if temporal is not None else None,
        "counterfactual": (
            counterfactual.to_dict() if counterfactual is not None else None
        ),
        "reconciliation": (
            reconciliation.to_dict() if reconciliation is not None else None
        ),
        "lineage": lineage.to_dict() if lineage is not None else None,
        "evidence_graph": evidence.to_dict() if evidence is not None else None,
        "relationships": [
            relationship.to_dict() for relationship in relationships
        ],
        "reference": reference,
        "decision": None,
        "contracts": contracts.to_dict(),
        "version_pair": pair.to_dict() if pair is not None else None,
        "events": [event.to_dict() for event in events],
        "evidence": list(reasons),
    }


def _certificate_periods(
    result: Any, findings: list[Any]
) -> Sequence[int | None]:
    """Periods that need their own evidence package and certificates."""
    from .policy import CLEARABLE_CHECKS

    targets: Sequence[int | None] = sorted(
        {
            int(item.period)
            for item in findings
            if item.period is not None
            and item.check in CLEARABLE_CHECKS
            and item.outcome == "FAIL"
        }
    )
    if targets:
        return targets
    temporal = getattr(result, "temporal", None)
    if temporal is None:
        return [None]
    temporal_targets = getattr(temporal, "targets", None)
    if temporal_targets:
        return sorted({int(value) for value in temporal_targets})
    return [int(temporal.target_week)]


def _attach_decisions(
    result: QCRunResult,
    provider: DecisionProvider | None,
    config: DatasetConfig,
    prior_refreshes: Sequence[dict[str, Any]] = (),
    assessment_id: str | None = None,
    provenance: str = "synthetic",
    qualification: Any | None = None,
) -> QCRunResult:
    if result.snapshot_manifest is not None:
        result.machine["snapshot_manifest"] = result.snapshot_manifest
        result.machine["observed_at"] = result.observed_at
    from .calendar import calendar_identity
    result.machine["calendar"] = calendar_identity(config)
    result.machine["hierarchy"] = {
        "schema_version": 1,
        "checks": [check.to_dict() for check in result.hierarchy],
    }
    result.machine["ledger"] = result.ledger.to_dict() if result.ledger else None
    result.machine["statistical"] = {
        "forecaster": config.forecaster,
        "selection": "frozen_grid_v2",
        "min_annual_history": config.temporal_min_annual_history,
        "interval_alpha": config.temporal_interval_alpha,
        "calibration_origins": config.temporal_calibration_origins,
    }
    result.machine["materiality_threshold"] = max(
        config.materiality_abs,
        config.materiality_ratio * abs(result.attribution.previous_total),
    ) if result.attribution else config.materiality_abs
    result.machine["assessment_provenance"] = provenance
    result.assessment_id = assessment_id or result.run_id

    from .policy import apply_policy, collect_findings
    findings = collect_findings(result, config, tuple(prior_refreshes))
    if getattr(config, "recurrence_enabled", True):
        # An initial assessment with no eligible predecessor history is a
        # recorded cold start, not an implicit clean recurrence history.
        result.machine["recurrence_status"] = (
            "ASSESSED" if prior_refreshes else "COLD_START"
        )

    qualification_error = ""
    qualification_path = str(getattr(config, "qualification_path", "") or "")
    if qualification is None and qualification_path:
        # Direct callers may load the artifact here; orchestrated weekly runs
        # resolve and validate it once, before assessment identity, and pass
        # the frozen artifact through so a replaced path cannot change replay.
        from .qualification import load_qualification
        try:
            qualification = load_qualification(qualification_path)
        except Exception as error:  # noqa: BLE001 - clearance stays disabled
            qualification_error = f"{type(error).__name__}: {error}"
    if qualification is not None:
        result.machine["qualification"] = {
            "status": qualification.status,
            "provenance": qualification.provenance,
            "digest": qualification.digest,
        }
    elif qualification_error:
        result.machine["qualification"] = {
            "status": "UNAVAILABLE",
            "error": qualification_error,
        }
    else:
        result.machine["qualification"] = {"status": "NOT_PINNED"}

    packages: list[Any] = []
    certificates: tuple[Any, ...] = ()
    if getattr(config, "decision_enabled", True) or result.temporal is not None:
        from .evidence_package import (
            build_assessment_evidence,
            verify_explanations,
        )

        for target in _certificate_periods(result, findings):
            period_findings = [
                item
                for item in findings
                if item.period is None or target is None or item.period == target
            ]
            ledger_override = (
                result.period_ledgers.get(int(target))
                if target is not None
                else None
            )
            metric_ledger_override: dict[str, Any] = {}
            if target is not None:
                for metric in config.temporal_metrics() or (
                    config.primary_metric,
                ):
                    built = result.period_metric_ledgers.get(
                        f"{metric}|{int(target)}"
                    )
                    if built is not None:
                        metric_ledger_override[metric] = built
            try:
                package = build_assessment_evidence(
                    result,
                    config,
                    assessment_id=result.assessment_id,
                    observation_cutoff=result.observed_at,
                    findings=findings,
                    target_week=target,
                    ledger_override=ledger_override,
                    metric_ledgers=metric_ledger_override,
                    qualification=qualification,
                )
            except Exception as error:  # noqa: BLE001 - keep deterministic results
                result.machine["certificate_availability"] = {
                    "status": "UNAVAILABLE",
                    "error": f"{type(error).__name__}: {error}",
                }
                packages = []
                certificates = ()
                break
            packages.append(package)
            if getattr(config, "statistical_clearance_enabled", True):
                certificates = certificates + verify_explanations(
                    package,
                    config,
                    materiality_threshold=result.machine["materiality_threshold"],
                    findings=period_findings,
                    qualification=qualification,
                    provenance=provenance,
                )
    result.evidence_packages = packages
    result.evidence_package = packages[0] if packages else None
    result.certificates = certificates
    result.machine["evidence_digests"] = [item.digest for item in packages]
    if "certificate_availability" not in result.machine:
        result.machine["certificate_availability"] = {
            "status": "AVAILABLE",
            "verified": sum(
                1 for item in certificates if item.status == "VERIFIED"
            ),
            "rejected": sum(
                1 for item in certificates if item.status == "REJECTED"
            ),
        }
    apply_policy(result, findings, certificates, config.optional_checks)
    from dataclasses import asdict
    result.machine["rule_evidence"] = {
        "contracts": {"status": result.contracts.status, "failed": [asdict(c) for c in result.contracts.failed]},
        "events": [asdict(e) for e in result.events],
        "attribution": asdict(result.attribution) if result.attribution else None,
        "lineage": asdict(result.lineage) if result.lineage else None,
        "temporal": asdict(result.temporal) if result.temporal else None,
        "relationships": [asdict(r) for r in result.relationships],
    }
    from .systemone import build_evidence_state
    result.machine["recorded_evidence_state"] = build_evidence_state(result, 128 * 1024)
    if not config.decision_enabled:
        return result
    active = provider or RuleDecisionProvider(config)
    try:
        result.decisions = active.decide(result)
        result.machine["provider_availability"] = {"status": "AVAILABLE", "provider": active.name}
    except Exception as error:
        if provider is None:
            raise
        from .systemone import ProviderAbstention
        abstained = isinstance(error, ProviderAbstention)
        result.machine["provider_availability"] = {
            "status": "ABSTAINED" if abstained else "UNAVAILABLE",
            "provider": active.name,
            "error": str(error),
        }
        result.decisions = RuleDecisionProvider(config).decide(result)
    result.machine["provider_recommendation"] = result.decisions.to_dict()
    result.machine["policy_required_review"] = result.machine["requires_investigation"]
    if result.machine["policy_required_review"]:
        from .decisions import DecisionValue
        result.decisions.values["requires_investigation"] = DecisionValue(
            field="requires_investigation", kind="boolean", value=True,
            probabilities={"False": 0.0, "True": 1.0}, strategy="deterministic_policy",
            probability_kind="deterministic_policy", evidence=[f["finding_id"] for f in result.machine["findings"]
                if f["outcome"] in ("FAIL", "CONTRACT_FAILURE", "UNAVAILABLE") and f["required"]])
    result.decisions.requires_investigation = (
        result.decisions.requires_investigation or result.machine["requires_investigation"]
    )
    result.machine["requires_investigation"] = result.decisions.requires_investigation
    result.machine["decision"] = result.decisions.to_dict()
    return result


def run_qc(
    source: VersionSource,
    current_id: str,
    previous_id: str,
    config: DatasetConfig | None = None,
    registry: RegistryStore | list[dict[str, Any]] | None = None,
    run_id: str | None = None,
    decision_provider: DecisionProvider | None = None,
    reference_frame: pd.DataFrame | None = None,
    reference_spec: ReferenceSpec | None = None,
    expectations: Sequence[Any] = (),
    observed_at: str | None = None,
    prior_refreshes: Sequence[dict[str, Any]] = (),
    assessment_id: str | None = None,
    qualification: Any | None = None,
) -> QCRunResult:
    config = config or DatasetConfig()
    from .store import observation_time
    observed_at = observation_time(observed_at)
    from .source import CachedSource
    if not isinstance(source, CachedSource):
        source = CachedSource(source, config)
    provenance = _source_provenance(source)
    from .versions import select_versions
    current_id, previous_id = select_versions(source.list_versions(), current_id, previous_id)
    if hasattr(source, "snapshot_metadata"):
        cutoff_ms = datetime.fromisoformat(observed_at).timestamp() * 1000
        for selected in (previous_id, current_id):
            commit = source.snapshot_metadata(selected).get("committed_at_ms")
            if commit is not None and commit > cutoff_ms + 0.001:
                raise ValueError(f"snapshot {selected} was unavailable at observation cutoff")
    registry = as_registry(registry)
    run_id = run_id or f"{previous_id}->{current_id}"

    from .assessment import snapshot_manifest
    raw_source = getattr(source, "source", source)
    raw_source = getattr(raw_source, "inner", raw_source)
    source_identity = str(getattr(raw_source, "source_id", getattr(raw_source, "uri", getattr(raw_source, "scenario_dir", "in_memory"))))
    snapshots = snapshot_manifest(source, previous_id, current_id, config, source_identity)
    previous_available = set(source.available_stages(previous_id))
    current_available = set(source.available_stages(current_id))
    common_stages = sorted(previous_available & current_available)
    input_findings = [{"check": f"input:{stage}", "scope": config.name, "outcome": "UNAVAILABLE", "required": True}
                      for stage in (config.analysis_stage, config.contract_stage) if stage not in common_stages]
    for name in sorted(set(("products", "stores", *config.required_dimensions))):
        for version in (previous_id, current_id):
            frame = _read_dim(source, version, name)
            if frame is None:
                input_findings.append({"check": f"dimension:{name}", "scope": version,
                                       "outcome": "UNAVAILABLE", "required": name in config.required_dimensions})
    if config.analysis_stage not in common_stages:
        contracts = ContractResult(status="PASS", checks=[])
        result = QCRunResult(run_id=run_id, dataset=config.name, status="INCOMPLETE", contracts=contracts,
                             version_pair=None, input_findings=input_findings, snapshot_manifest=snapshots, observed_at=observed_at,
                             machine=_machine(run_id, config.name, "INCOMPLETE", contracts, None, [], None, []))
        return _attach_decisions(
            result,
            decision_provider,
            config,
            prior_refreshes,
            assessment_id=assessment_id,
            provenance=provenance,
            qualification=qualification,
        )
    stages_to_check: list[str] = []
    for stage in common_stages:
        if stage in common_stages and stage not in stages_to_check:
            stages_to_check.append(stage)
    if not stages_to_check:
        stages_to_check = [common_stages[-1]]

    checks: list[ContractCheck] = []
    contract_current_fact: pd.DataFrame | None = None
    for stage in stages_to_check:
        current_fact = source.read_fact(current_id, stage)
        previous_fact = source.read_fact(previous_id, stage)
        stage_result = validate_contracts(current_fact, previous_fact, config, stage)
        checks.extend(
            ContractCheck(f"{stage}:{check.name}", check.passed, check.detail)
            for check in stage_result.checks
        )
        if stage == config.contract_stage:
            contract_current_fact = current_fact
    contracts = ContractResult(
        status=(
            "DATA_CONTRACT_FAILURE"
            if any(not check.passed for check in checks)
            else "PASS"
        ),
        checks=checks,
    )

    if contracts.status == "DATA_CONTRACT_FAILURE":
        reasons = _reasons(contracts, [], None, config, contracts.status)
        result = QCRunResult(
            run_id=run_id,
            dataset=config.name,
            status=contracts.status,
            contracts=contracts,
            version_pair=None,
            snapshot_manifest=snapshots, observed_at=observed_at,
            reasons=reasons,
            machine=_machine(
                run_id, config.name, contracts.status, contracts, None, [], None, reasons
            ),
        )
        return _attach_decisions(
            result,
            decision_provider,
            config,
            prior_refreshes,
            assessment_id=assessment_id,
            provenance=provenance,
            qualification=qualification,
        )

    analysis_stage = _resolve_common_stage(
        source, previous_id, current_id, config.analysis_stages()
    )
    previous = source.read_fact(previous_id, analysis_stage)
    current = source.read_fact(current_id, analysis_stage)
    pair = build_version_pair(
        previous,
        current,
        previous_id,
        current_id,
        analysis_stage,
        analysis_stage,
        config,
    )

    cubes: dict[str, pd.DataFrame] = {}
    base_grain = base_keys(current, config)
    cubes["base"] = build_revision_cube(
        previous, current, base_grain, config, pair.new_periods
    )
    headline_grain = headline_keys(current, config)
    if any(key not in current or key not in previous for key in headline_grain):
        input_findings.append({"check": "revision:headline_grain", "scope": config.name, "outcome": "UNAVAILABLE", "required": True})
    elif headline_grain != base_grain:
        cubes["headline"] = build_revision_cube(
            previous, current, headline_grain, config, pair.new_periods
        )
    week_grain = [config.week_column]
    if week_grain not in (base_grain, headline_grain):
        cubes["week"] = build_revision_cube(
            previous, current, week_grain, config, pair.new_periods
        )

    events = classify_entity_changes(previous, current, pair, config)
    events += detect_reclassification(
        previous,
        current,
        _read_dim(source, previous_id, "products"),
        _read_dim(source, current_id, "products"),
        pair,
        config,
    )
    relationships = detect_relationships(previous, current, pair, config)

    def approval_available(item):
        try:
            approved = datetime.fromisoformat(item["approved_at"])
            cutoff = datetime.fromisoformat(observed_at)
            if approved.tzinfo is None:
                approved = approved.replace(tzinfo=UTC)
            if cutoff.tzinfo is None:
                cutoff = cutoff.replace(tzinfo=UTC)
            return approved <= cutoff
        except (KeyError, ValueError, TypeError):
            return False
    approved_events = [item for item in registry.events() if approval_available(item)]
    attribution = explain_revision(cubes["base"], events, approved_events, config)
    counterfactual = reconstruct_counterfactual(previous, current, events, config)
    reconciliation = run_reconciliation(current, contract_current_fact if config.analysis_stage != config.contract_stage else None, config)
    lineage = analyze_lineage(
        source, previous_id, current_id, common_stages, config, pair
    )
    if reference_frame is not None and reference_spec is None:
        raise ValueError("reference_frame requires reference_spec")
    if reference_spec is not None and reference_frame is None:
        reference_frame = pd.DataFrame()
    reference_report = (
        compare_reference(current, reference_frame, reference_spec, config)
        if reference_frame is not None and reference_spec is not None
        else None
    )
    historical_status = classify_run(
        contracts.status,
        events,
        attribution,
        config,
        reconstruction_score=counterfactual.reconciliation_score,
    )
    try:
        temporal = run_temporal_qc(previous, current, pair, config, events)
    except Exception as exc:  # noqa: BLE001 - preserve completed deterministic checks
        temporal = TemporalResult(
            target_week=pair.current_max_week, anomaly=False, flags=[],
            calibration={"unavailable_reason": f"{type(exc).__name__}: {exc}"},
            series=[], unavailable_series=["provider_execution"],
        )
    status = historical_status
    if (
        temporal is not None
        and temporal.anomaly
        and status in ("PASS", "PASS_WITH_EXPLANATION")
    ):
        status = "INVESTIGATE"
    if (
        reference_report is not None
        and reference_report["status"] == "MISMATCH"
    ):
        status = "INVESTIGATE"
    from .calendar import build_calendar
    calendar = build_calendar(config)
    try:
        hierarchy = hierarchy_checks(current, config)
    except ValueError:
        hierarchy = []
    configured_metrics = config.temporal_metrics() or (config.primary_metric,)
    metric_ledgers: dict[str, ExplanationLedger] = {}
    for metric in configured_metrics:
        try:
            metric_ledgers[metric] = build_explanation_ledger(
                previous,
                current,
                pair,
                config,
                events=events,
                temporal=temporal,
                calendar=calendar,
                approved_event_ids=tuple(attribution.matched_event_ids),
                metric=metric,
            )
        except ValueError:
            continue
    ledger = metric_ledgers.get(config.primary_metric)
    period_ledgers: dict[int, ExplanationLedger] = {}
    period_metric_ledgers: dict[str, ExplanationLedger] = {}
    if temporal is not None:
        for target in getattr(temporal, "targets", ()) or (temporal.target_week,):
            for metric in configured_metrics:
                try:
                    built = build_explanation_ledger(
                        previous,
                        current,
                        pair,
                        config,
                        events=events,
                        temporal=_TemporalPeriod(temporal, int(target), metric),
                        calendar=calendar,
                        approved_event_ids=tuple(attribution.matched_event_ids),
                        new_periods=(int(target),),
                        metric=metric,
                    )
                except ValueError:
                    continue
                period_metric_ledgers[f"{metric}|{int(target)}"] = built
                if metric == config.primary_metric:
                    period_ledgers[int(target)] = built
    reasons = _reasons(
        contracts, events, attribution, config, historical_status, relationships
    )
    reasons.extend(
        f"reference_mismatch:{metric}"
        for metric in reference_mismatches(reference_report)
    )
    if temporal is not None:
        for item in temporal.series:
            if item.anomaly:
                reasons.append(f"latest_week_anomaly:{item.series_id}")
    evidence = build_evidence_graph(
        run_id=run_id,
        pair=pair,
        contracts=contracts,
        events=events,
        attribution=attribution,
        counterfactual=counterfactual,
        reconciliation=reconciliation,
        lineage=lineage,
        temporal=temporal,
        relationships=relationships,
    )

    return _attach_decisions(
        QCRunResult(
            run_id=run_id,
            dataset=config.name,
            status=status,
            contracts=contracts,
            version_pair=pair,
            snapshot_manifest=snapshots,
            cubes=cubes,
            events=events,
            attribution=attribution,
            counterfactual=counterfactual,
            reconciliation=reconciliation,
            lineage=lineage,
            temporal=temporal,
            evidence=evidence,
            relationships=relationships,
            reference=reference_report,
            expectations=expectations,
            hierarchy=hierarchy,
            ledger=ledger,
            period_ledgers=period_ledgers,
            metric_ledgers=metric_ledgers,
            period_metric_ledgers=period_metric_ledgers,
            observed_at=observed_at,
            temporal_required=config.temporal_enabled and config.temporal_required and bool(pair.new_periods),
            input_findings=input_findings,
            reasons=reasons,
            machine=_machine(
                run_id,
                config.name,
                status,
                contracts,
                pair,
                events,
                attribution,
                reasons,
                counterfactual,
                reconciliation,
                lineage,
                historical_status,
                temporal,
                evidence,
                relationships,
                reference_report,
            ),
        ),
        decision_provider,
        config,
        prior_refreshes,
        assessment_id=assessment_id,
        provenance=provenance,
        qualification=qualification,
    )
