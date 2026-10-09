"""QC run orchestration.

``run_qc`` ties the layers together:

    contracts -> version pair -> revision cubes -> lifecycle -> attribution
    -> counterfactual / reconciliation / lineage / reference -> temporal

then materializes findings and computes the final status exactly once
(``policy.apply_policy``). Cause labels from ``RuleDecisionProvider`` are
attached afterwards and cannot change the status.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from .attribution import (
    AttributionResult,
    WeekRevision,
    classify_run,
    explain_revision,
    like_for_like,
    week_revisions,
)
from .config import DatasetConfig
from .contracts import ContractCheck, ContractResult, validate_contracts
from .counterfactual import CounterfactualResult, reconstruct_counterfactual
from .decisions import DecisionSet, RuleDecisionProvider
from .hierarchy import HierarchyCheck, hierarchy_checks
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
    decisions: DecisionSet | None = None
    relationships: list[EntityRelationship] = field(default_factory=list)
    week_revisions: list[WeekRevision] = field(default_factory=list)
    reference: dict[str, Any] | None = None
    expectations: Sequence[Any] = ()
    observed_at: str | None = None
    temporal_required: bool = False
    input_findings: list[dict[str, Any]] = field(default_factory=list)
    snapshot_manifest: dict[str, Any] | None = None
    reasons: list[str] = field(default_factory=list)
    hierarchy: list[HierarchyCheck] = field(default_factory=list)
    assessment_id: str | None = None
    machine: dict[str, Any] = field(default_factory=dict)


def _read_dim(
    source: VersionSource, version: str, name: str
) -> pd.DataFrame | None:
    reader = getattr(source, "read_dim", None)
    if reader is None:
        return None
    return reader(version, name)


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


def _finalize(
    result: QCRunResult,
    config: DatasetConfig,
    assessment_id: str | None = None,
) -> QCRunResult:
    """Materialize findings, compute the one final status, attach labels."""
    from .calendar import calendar_identity
    from .policy import apply_policy, collect_findings

    if result.snapshot_manifest is not None:
        result.machine["snapshot_manifest"] = result.snapshot_manifest
        result.machine["observed_at"] = result.observed_at
    result.machine["calendar"] = calendar_identity(config)
    result.machine["hierarchy"] = {
        "schema_version": 1,
        "checks": [check.to_dict() for check in result.hierarchy],
    }
    result.machine["statistical"] = {
        "forecaster": config.forecaster,
        "min_annual_history": config.temporal_min_annual_history,
        "interval_alpha": config.temporal_interval_alpha,
        "calibration_origins": config.temporal_calibration_origins,
        "fdr_q": config.temporal_fdr_q,
    }
    result.machine["materiality_threshold"] = max(
        config.materiality_abs,
        config.materiality_ratio * abs(result.attribution.previous_total),
    ) if result.attribution else config.materiality_abs
    result.assessment_id = assessment_id or result.run_id
    result.machine["week_revisions"] = [
        item.to_dict() for item in result.week_revisions if item.material
    ]

    findings = collect_findings(result, config)
    apply_policy(result, findings, config.optional_checks)
    if config.decision_enabled:
        result.decisions = RuleDecisionProvider(config).decide(result)
        result.machine["decision"] = result.decisions.to_dict()
    return result


def run_qc(
    source: VersionSource,
    current_id: str,
    previous_id: str,
    config: DatasetConfig | None = None,
    registry: RegistryStore | list[dict[str, Any]] | None = None,
    run_id: str | None = None,
    reference_frame: pd.DataFrame | None = None,
    reference_spec: ReferenceSpec | None = None,
    expectations: Sequence[Any] = (),
    observed_at: str | None = None,
    assessment_id: str | None = None,
) -> QCRunResult:
    config = config or DatasetConfig()
    from .store import observation_time
    observed_at = observation_time(observed_at)
    from .source import CachedSource
    if not isinstance(source, CachedSource):
        source = CachedSource(source, config)
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
        return _finalize(result, config, assessment_id)
    checks: list[ContractCheck] = []
    contract_current_fact: pd.DataFrame | None = None
    for stage in common_stages:
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
        return _finalize(result, config, assessment_id)

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
    attribution = explain_revision(cubes["base"], events, approved_events, config, pair.current_max_week)
    # A category emptied or created by an approved move changed no value.
    valued_events = [e for e in events if f"{e.entity_type}:{e.entity_id}" not in attribution.move_consequences]
    revisions_by_week = week_revisions(cubes["base"], valued_events, config)
    counterfactual = reconstruct_counterfactual(previous, current, valued_events, config)
    reconciliation = run_reconciliation(
        current,
        contract_current_fact if config.analysis_stage != config.contract_stage else None,
        config,
        periods=pair.new_periods,
    )
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
    # Approved closures and moves restate both versions like-for-like. An
    # absence that only follows from an approved closure (a product sold only
    # in the closed store) is not a further absence like-for-like.
    temporal_previous, temporal_current, restatement = like_for_like(previous, current, events, attribution)
    absences = (
        classify_entity_changes(temporal_previous, temporal_current, pair, config)
        if restatement["excluded"]
        else None
    )
    historical_status = classify_run(
        contracts.status,
        events,
        attribution,
        config,
        reconstruction_score=counterfactual.reconciliation_score,
        absences=absences,
    )
    try:
        temporal = run_temporal_qc(temporal_previous, temporal_current, pair, config, events)
    except Exception as exc:  # noqa: BLE001 - preserve completed deterministic checks
        temporal = TemporalResult(
            target_week=pair.current_max_week, anomaly=False, flags=[],
            calibration={"unavailable_reason": f"{type(exc).__name__}: {exc}"},
            series=[], unavailable_series=["provider_execution"],
        )
    # The final status is computed exactly once, from findings, by
    # ``apply_policy``. Every check that can escalate must emit a finding; the
    # historical status here is only the provisional value until then.
    status = historical_status
    try:
        hierarchy = hierarchy_checks(current, config)
    except ValueError:
        input_findings.append({"check": "hierarchy", "scope": config.name,
                               "outcome": "UNAVAILABLE", "required": False})
        hierarchy = []
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
    return _finalize(
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
            relationships=relationships,
            week_revisions=revisions_by_week,
            reference=reference_report,
            expectations=expectations,
            hierarchy=hierarchy,
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
                relationships,
                reference_report,
            ) | {"like_for_like": restatement},
        ),
        config,
        assessment_id,
    )
