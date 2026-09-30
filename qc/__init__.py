"""Deterministic QC engine.

Implements the early milestones of section 73 of ``docs/architecture.md``:
data contracts, version resolution, the revision cube, entity lifecycle,
attribution and the explained/unexplained residual. No model dependencies are
permitted in this package.
"""

from .agent import (
    CommandAgent,
    InvestigationAgent,
    InvestigationBrief,
    InvestigationResult,
    NullAgent,
    build_investigation_brief,
    parse_investigation_result,
)
from .attribution import AttributionResult, Contributor, classify_run, explain_revision
from .calibration import (
    CALIBRATOR_VERSION,
    IsotonicCalibrator,
    PlattCalibrator,
    TemperatureCalibrator,
    calibrator_report,
    fit_isotonic,
    fit_platt,
    negative_log_likelihood,
    reliability_bins,
    select_calibrator,
)
from .champion import (
    DEFAULT_CHAMPION_GATES,
    ChampionResult,
    EvalItem,
    ProviderScore,
    eval_cases_from_suite,
    render_champion_markdown,
    run_champion,
    write_champion_report,
)
from .cohort import CohortCase, CohortPlan, CohortResult, code_sha256, run_cohort
from .config import DatasetConfig, load_dataset_config
from .conformal import ConformalInterval, conformal_interval, leave_one_out_coverage
from .contracts import ContractCheck, ContractResult, validate_contracts
from .counterfactual import CounterfactualResult, reconstruct_counterfactual
from .decisions import (
    DecisionProvider,
    DecisionSet,
    DecisionValue,
    FeatureEncoder,
    FieldSpec,
    LinearHead,
    RuleDecisionProvider,
    TrainedDecisionProvider,
    default_fields,
    score_index,
)
from .decomposition import (
    DECOMPOSITION_VERSION,
    BinaryQuestion,
    DecomposedProvider,
    decomposition_questions,
    evaluate_decomposed,
    question_targets,
    reconstruct_field,
    train_decomposed_provider,
)
from .delta import DeltaSource, describe_delta_table
from .distribution_drift import (
    DRIFT_CHECK_VERSION,
    DistributionDriftResult,
    DriftScope,
    assess_distribution_drift,
    population_stability_index,
)
from .drift import DriftReport, monitor_drift
from .events import load_registry, propose_expected_events, save_registry
from .evidence import EvidenceGraph, EvidenceNode, build_evidence_graph
from .evidence_query import ALLOWED_QUERIES, EvidenceQueryResult, query_evidence
from .evidence_text import EVIDENCE_TEXT_VERSION, evidence_text
from .expectations import (
    RatioExpectation,
    apply_expectations,
    load_expectations,
    save_expectations,
)
from .fingerprints import fingerprint_deltas, structural_fingerprint
from .incidents import (
    IncidentRecord,
    IncidentStore,
    build_incident_record,
    symptom_tags_for,
)
from .labels import (
    LabelRecord,
    LabelStore,
    build_oracle_labels,
    records_from_store,
)
from .lifecycle import LifecycleEvent, classify_entity_changes, detect_reclassification
from .lineage import LineageResult, analyze_lineage
from .notify import (
    DEFAULT_NOTIFY_STATUSES,
    NotificationResult,
    NotificationSpec,
    load_notification_specs,
)
from .notify import (
    build_payload as build_notification_payload,
)
from .notify import (
    dispatch_all as dispatch_notifications,
)
from .onboard import (
    assess_versions,
    config_from_proposal,
    profile_table,
    propose_config,
)
from .policy import apply_policy, collect_findings, finalize_policy
from .prequential import (
    CalibrationPool,
    CalibrationRecord,
    LoadSnapshot,
    PrequentialRunResult,
    PrequentialSeriesEvidence,
    PrequentialStore,
    append_run_calibration,
    forecast_across_loads,
    prequential_sequence,
    records_from_result,
)
from .qualification import (
    QualificationArtifact,
    build_qualification,
    load_qualification,
    pin_qualification,
)
from .rca import RCAResult, RCAStep, investigate
from .reconciliation import (
    ReconciliationCheck,
    ReconciliationResult,
    run_reconciliation,
)
from .reference import ReferenceSpec, compare_reference
from .relationships import (
    EntityRelationship,
    RelationshipStore,
    detect_relationships,
)
from .replay import ReplayCase, ReplayPlan, ReplayReport, replay
from .reporting import render_markdown, write_report
from .revision import build_revision_cube
from .run import QCRunResult, run_qc
from .shadow import ShadowRecord, run_shadow
from .store import SqliteStore, import_outcomes
from .synthetic_analyst import (
    PROFILES as ANALYST_PROFILES,
)
from .synthetic_analyst import (
    AnalystProfile,
    simulate_analyst,
)
from .systemone import (
    FallbackDecisionProvider,
    ProviderAbstention,
    SystemOneDecisionProvider,
    answers_to_decisions,
    build_evidence_state,
    build_provider_state,
    decision_to_systemone_answers,
)
from .temporal import (
    BaselineForecaster,
    CalibrationMap,
    ChronosForecaster,
    SarimaxForecaster,
    SeriesTemporalEvidence,
    TemporalResult,
    backtest_forecaster,
    build_temporal_series,
    get_forecaster,
    run_temporal_qc,
)
from .text_provider import (
    ModernBertEmbedder,
    TextDecisionProvider,
    evaluate_text_provider,
    load_decision_provider,
    train_text_provider,
)
from .training import (
    cost_matrix,
    evaluate_decision_provider,
    evaluate_field,
    label_metrics,
    pair_cost,
    save_training_run,
    train_decision_provider,
)
from .tspulse import (
    AnomalyResult,
    EmbeddingResult,
    TSPulseResearch,
)
from .tspulse import (
    cosine as tspulse_cosine,
)
from .tspulse_benchmark import (
    BenchmarkResult,
    RevisionSeries,
    build_revision_series,
    run_tspulse_benchmark,
)
from .versions import VersionPair, resolve_versions
from .weekly import WeeklyResult, run_weekly

__all__ = [
    "ALLOWED_QUERIES",
    "ANALYST_PROFILES",
    "AnalystProfile",
    "AnomalyResult",
    "AttributionResult",
    "BaselineForecaster",
    "BenchmarkResult",
    "BinaryQuestion",
    "CALIBRATOR_VERSION",
    "CalibrationMap",
    "CalibrationPool",
    "CalibrationRecord",
    "ChampionResult",
    "ChronosForecaster",
    "CohortCase",
    "CohortPlan",
    "CohortResult",
    "CommandAgent",
    "ConformalInterval",
    "Contributor",
    "ContractCheck",
    "ContractResult",
    "CounterfactualResult",
    "DEFAULT_CHAMPION_GATES",
    "DatasetConfig",
    "DECOMPOSITION_VERSION",
    "DecomposedProvider",
    "DecisionProvider",
    "DecisionSet",
    "DecisionValue",
    "DeltaSource",
    "DRIFT_CHECK_VERSION",
    "DistributionDriftResult",
    "DriftReport",
    "DriftScope",
    "EVIDENCE_TEXT_VERSION",
    "EmbeddingResult",
    "EntityRelationship",
    "EvalItem",
    "EvidenceGraph",
    "EvidenceNode",
    "EvidenceQueryResult",
    "FallbackDecisionProvider",
    "FeatureEncoder",
    "FieldSpec",
    "IncidentRecord",
    "IncidentStore",
    "DEFAULT_NOTIFY_STATUSES",
    "InvestigationAgent",
    "InvestigationBrief",
    "InvestigationResult",
    "IsotonicCalibrator",
    "LabelRecord",
    "LabelStore",
    "LifecycleEvent",
    "LinearHead",
    "LineageResult",
    "LoadSnapshot",
    "ModernBertEmbedder",
    "NotificationResult",
    "NotificationSpec",
    "NullAgent",
    "PlattCalibrator",
    "PrequentialRunResult",
    "PrequentialSeriesEvidence",
    "PrequentialStore",
    "ProviderAbstention",
    "ProviderScore",
    "QCRunResult",
    "QualificationArtifact",
    "RCAResult",
    "RCAStep",
    "RatioExpectation",
    "ReconciliationCheck",
    "ReconciliationResult",
    "ReferenceSpec",
    "RelationshipStore",
    "ReplayCase",
    "ReplayPlan",
    "ReplayReport",
    "RevisionSeries",
    "RuleDecisionProvider",
    "SarimaxForecaster",
    "SeriesTemporalEvidence",
    "ShadowRecord",
    "SqliteStore",
    "SystemOneDecisionProvider",
    "TSPulseResearch",
    "TemperatureCalibrator",
    "TemporalResult",
    "TextDecisionProvider",
    "TrainedDecisionProvider",
    "VersionPair",
    "WeeklyResult",
    "analyze_lineage",
    "answers_to_decisions",
    "append_run_calibration",
    "apply_expectations",
    "apply_policy",
    "assess_distribution_drift",
    "assess_versions",
    "backtest_forecaster",
    "build_evidence_graph",
    "build_evidence_state",
    "build_incident_record",
    "build_investigation_brief",
    "build_notification_payload",
    "build_oracle_labels",
    "build_provider_state",
    "build_qualification",
    "build_revision_cube",
    "build_revision_series",
    "build_temporal_series",
    "calibrator_report",
    "classify_entity_changes",
    "classify_run",
    "code_sha256",
    "collect_findings",
    "conformal_interval",
    "config_from_proposal",
    "compare_reference",
    "cost_matrix",
    "decision_to_systemone_answers",
    "decomposition_questions",
    "default_fields",
    "describe_delta_table",
    "detect_reclassification",
    "detect_relationships",
    "dispatch_notifications",
    "evaluate_decision_provider",
    "evaluate_decomposed",
    "evaluate_field",
    "evaluate_text_provider",
    "evidence_text",
    "eval_cases_from_suite",
    "explain_revision",
    "fit_isotonic",
    "fit_platt",
    "fingerprint_deltas",
    "finalize_policy",
    "forecast_across_loads",
    "get_forecaster",
    "import_outcomes",
    "investigate",
    "label_metrics",
    "leave_one_out_coverage",
    "load_dataset_config",
    "load_decision_provider",
    "load_qualification",
    "load_expectations",
    "load_registry",
    "load_notification_specs",
    "monitor_drift",
    "pair_cost",
    "parse_investigation_result",
    "population_stability_index",
    "pin_qualification",
    "profile_table",
    "propose_config",
    "propose_expected_events",
    "prequential_sequence",
    "negative_log_likelihood",
    "query_evidence",
    "question_targets",
    "reconstruct_counterfactual",
    "reconstruct_field",
    "records_from_result",
    "records_from_store",
    "reliability_bins",
    "render_champion_markdown",
    "render_markdown",
    "replay",
    "resolve_versions",
    "run_champion",
    "run_cohort",
    "run_qc",
    "run_reconciliation",
    "run_shadow",
    "run_temporal_qc",
    "run_tspulse_benchmark",
    "run_weekly",
    "save_registry",
    "save_expectations",
    "save_training_run",
    "select_calibrator",
    "score_index",
    "simulate_analyst",
    "structural_fingerprint",
    "symptom_tags_for",
    "train_decision_provider",
    "train_decomposed_provider",
    "train_text_provider",
    "tspulse_cosine",
    "validate_contracts",
    "write_champion_report",
    "write_report",
]
