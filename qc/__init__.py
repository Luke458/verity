"""Retail refresh QC engine.

Deterministic checks (contracts, version pairing, revision cubes, lifecycle,
attribution, counterfactual, reconciliation, lineage, reference controls),
latest-week temporal QC, versioned findings with one final status, rule-based
cause labels, the weekly orchestrator and the synthetic evaluation harnesses.
"""

from .attribution import AttributionResult, Contributor, classify_run, explain_revision
from .cohort import CohortCase, CohortPlan, CohortResult, code_sha256, run_cohort
from .config import DatasetConfig, load_dataset_config
from .conformal import ConformalInterval, conformal_interval, leave_one_out_coverage
from .contracts import ContractCheck, ContractResult, validate_contracts
from .counterfactual import CounterfactualResult, reconstruct_counterfactual
from .decisions import (
    DecisionProvider,
    DecisionSet,
    DecisionValue,
    FieldSpec,
    RuleDecisionProvider,
    default_fields,
    score_index,
)
from .delta import DeltaSource, describe_delta_table
from .events import load_registry, propose_expected_events, save_registry
from .expectations import (
    RatioExpectation,
    apply_expectations,
    load_expectations,
    save_expectations,
)
from .fingerprints import fingerprint_deltas, structural_fingerprint
from .lifecycle import (
    LifecycleEvent,
    classify_entity_changes,
    detect_reclassification,
    missing_entity_impact,
)
from .lineage import LineageResult, analyze_lineage
from .notify import (
    DEFAULT_NOTIFY_STATUSES,
    NotificationResult,
    NotificationSpec,
    load_notification_specs,
)
from .notify import build_payload as build_notification_payload
from .notify import dispatch_all as dispatch_notifications
from .onboard import (
    assess_versions,
    config_from_proposal,
    profile_table,
    propose_config,
)
from .policy import Finding, apply_policy, collect_findings, status_for
from .reconciliation import (
    ReconciliationCheck,
    ReconciliationResult,
    run_reconciliation,
)
from .reference import ReferenceSpec, compare_reference
from .relationships import EntityRelationship, detect_relationships
from .reporting import render_markdown, write_report
from .revision import build_revision_cube
from .run import QCRunResult, run_qc
from .shadow import ShadowRecord, run_shadow
from .store import SqliteStore
from .temporal import (
    BaselineForecaster,
    SeriesTemporalEvidence,
    TemporalResult,
    backtest_forecaster,
    build_temporal_series,
    decide_anomalies,
    get_forecaster,
    run_temporal_qc,
    share_shift_test,
)
from .versions import VersionPair, resolve_versions
from .weekly import WeeklyResult, run_weekly

__all__ = [
    "DEFAULT_NOTIFY_STATUSES",
    "AttributionResult",
    "BaselineForecaster",
    "CohortCase",
    "CohortPlan",
    "CohortResult",
    "ConformalInterval",
    "ContractCheck",
    "ContractResult",
    "Contributor",
    "CounterfactualResult",
    "DatasetConfig",
    "DecisionProvider",
    "DecisionSet",
    "DecisionValue",
    "DeltaSource",
    "EntityRelationship",
    "FieldSpec",
    "Finding",
    "LifecycleEvent",
    "LineageResult",
    "NotificationResult",
    "NotificationSpec",
    "QCRunResult",
    "RatioExpectation",
    "ReconciliationCheck",
    "ReconciliationResult",
    "ReferenceSpec",
    "RuleDecisionProvider",
    "SeriesTemporalEvidence",
    "ShadowRecord",
    "SqliteStore",
    "TemporalResult",
    "VersionPair",
    "WeeklyResult",
    "analyze_lineage",
    "apply_expectations",
    "apply_policy",
    "assess_versions",
    "backtest_forecaster",
    "build_notification_payload",
    "build_revision_cube",
    "build_temporal_series",
    "classify_entity_changes",
    "classify_run",
    "code_sha256",
    "collect_findings",
    "compare_reference",
    "config_from_proposal",
    "conformal_interval",
    "decide_anomalies",
    "default_fields",
    "describe_delta_table",
    "detect_reclassification",
    "detect_relationships",
    "dispatch_notifications",
    "explain_revision",
    "fingerprint_deltas",
    "get_forecaster",
    "leave_one_out_coverage",
    "load_dataset_config",
    "load_expectations",
    "load_notification_specs",
    "load_registry",
    "missing_entity_impact",
    "profile_table",
    "propose_config",
    "propose_expected_events",
    "reconstruct_counterfactual",
    "render_markdown",
    "resolve_versions",
    "run_cohort",
    "run_qc",
    "run_reconciliation",
    "run_shadow",
    "run_temporal_qc",
    "run_weekly",
    "save_expectations",
    "save_registry",
    "score_index",
    "share_shift_test",
    "status_for",
    "structural_fingerprint",
    "validate_contracts",
    "write_report",
]
