"""QC run orchestration.

``run_qc`` ties the deterministic layers together:

    contracts -> version pair -> revision cubes -> lifecycle -> attribution

and returns a machine-readable result following section 77 of the architecture
document. It never falls back to a model: semantic decisions are Milestone D.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .attribution import AttributionResult, classify_run, explain_revision
from .config import DatasetConfig
from .contracts import ContractCheck, ContractResult, validate_contracts
from .counterfactual import CounterfactualResult, reconstruct_counterfactual
from .decisions import DecisionProvider, DecisionSet, RuleDecisionProvider
from .evidence import EvidenceGraph, build_evidence_graph
from .lifecycle import (
    LifecycleEvent,
    classify_entity_changes,
    detect_reclassification,
)
from .lineage import LineageResult, analyze_lineage
from .reconciliation import ReconciliationResult, run_reconciliation
from .reference import ReferenceSpec, compare_reference, reference_mismatches
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
    reasons: list[str] = field(default_factory=list)
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
    if (
        not attribution.material
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


def _attach_decisions(
    result: QCRunResult,
    provider: DecisionProvider | None,
    config: DatasetConfig,
) -> QCRunResult:
    if not config.decision_enabled:
        return result
    active = provider or RuleDecisionProvider(config)
    result.decisions = active.decide(result)
    result.machine["decision"] = result.decisions.to_dict()
    return result


def run_qc(
    source: VersionSource,
    current_id: str,
    previous_id: str,
    config: DatasetConfig | None = None,
    expected_events: list[dict[str, Any]] | None = None,
    run_id: str | None = None,
    decision_provider: DecisionProvider | None = None,
    reference_frame: pd.DataFrame | None = None,
    reference_spec: ReferenceSpec | None = None,
) -> QCRunResult:
    config = config or DatasetConfig()
    expected_events = expected_events or []
    run_id = run_id or f"{previous_id}->{current_id}"

    previous_available = set(source.available_stages(previous_id))
    current_available = set(source.available_stages(current_id))
    common_stages = sorted(previous_available & current_available)
    if not common_stages:
        raise ValueError(f"no common stage between {previous_id!r} and {current_id!r}")
    stages_to_check: list[str] = []
    for stage in (config.contract_stage, config.analysis_stage):
        if stage in common_stages and stage not in stages_to_check:
            stages_to_check.append(stage)
    if not stages_to_check:
        stages_to_check = [common_stages[-1]]

    checks: list[ContractCheck] = []
    contract_current_fact: pd.DataFrame | None = None
    for stage in stages_to_check:
        current_fact = source.read_fact(current_id, stage)
        previous_fact = source.read_fact(previous_id, stage)
        stage_result = validate_contracts(current_fact, previous_fact, config)
        checks.extend(
            ContractCheck(f"{stage}:{check.name}", check.passed, check.detail)
            for check in stage_result.checks
        )
        if stage == config.contract_stage or contract_current_fact is None:
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
            reasons=reasons,
            machine=_machine(
                run_id, config.name, contracts.status, contracts, None, [], None, reasons
            ),
        )
        return _attach_decisions(result, decision_provider, config)

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
    if headline_grain != base_grain:
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

    attribution = explain_revision(cubes["base"], events, expected_events, config)
    counterfactual = reconstruct_counterfactual(cubes["base"], events, config)
    reconciliation = run_reconciliation(current, contract_current_fact, config)
    lineage = analyze_lineage(
        source, previous_id, current_id, common_stages, config, pair
    )
    if (reference_frame is None) != (reference_spec is None):
        raise ValueError(
            "reference_frame and reference_spec must be provided together"
        )
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
    temporal = run_temporal_qc(previous, current, pair, config, events)
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
    )
