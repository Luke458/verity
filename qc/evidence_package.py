"""Versioned assessment evidence and explanation certificates (schema 2).

``AssessmentEvidence`` is the complete, serializable record behind a review
decision: predictions with calibration and intervals, integrity checks,
contributions, contradictions, net and gross unexplained residuals and the
identities (snapshot, calendar, model, configuration, cutoff) that pin it.
``ExplanationCertificate`` names the exact findings a scoped clearance covers,
the support it rests on, the assessment identity, evidence digest, policy
version and pinned qualification it was verified against.

Providers receive a compressed view built independently of whether automatic
clearance is enabled. Compression may drop detail but never mandatory evidence:
when the budget cannot carry the mandatory fields the view abstains instead.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from .conformal import minimum_samples
from .jsonutil import dumps as json_dumps

EVIDENCE_PACKAGE_SCHEMA = 3
EVIDENCE_POLICY_VERSION = "3"
CERTIFICATE_SCHEMA = 3

_FAILED_STATUSES = frozenset(
    {"CONTRACT_FAILURE", "DATA_CONTRACT_FAILURE", "FAIL"}
)


def _contract_failed(status: Any) -> bool:
    return str(status or "").upper() in _FAILED_STATUSES


def evidence_digest(value: Any) -> str:
    return hashlib.sha256(
        json_dumps(value, sort_keys=True, default=str).encode()
    ).hexdigest()


_MAX_DETAIL_CHARS = 160


def _compact_text(value: str) -> str:
    """Bound a verbose detail into a stable reference, never discard it."""
    if len(value) <= _MAX_DETAIL_CHARS:
        return value
    digest = hashlib.sha256(value.encode()).hexdigest()[:12]
    return f"{value[:96]}...<sha256:{digest}>"


def _compact_check(check: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(check)
    detail = payload.get("detail")
    if isinstance(detail, str):
        payload["detail"] = _compact_text(detail)
    return payload


def _bounded_omissions(omissions: list[str]) -> list[str]:
    """Record every omission as a count plus a bounded sample of references."""
    counts: dict[str, int] = {}
    samples: dict[str, list[str]] = {}
    for entry in omissions:
        section = entry.split(":", 1)[0]
        counts[section] = counts.get(section, 0) + 1
        if len(samples.setdefault(section, [])) < 5:
            samples[section].append(entry)
    bounded: list[str] = []
    for section in sorted(counts):
        bounded.append(f"{section}:{counts[section]} omitted")
        bounded.extend(samples[section])
    return bounded


def _compress_view(value: Any) -> Any:
    """Recursively bound long strings while preserving structure and identity."""
    if isinstance(value, str):
        return _compact_text(value)
    if isinstance(value, dict):
        return {key: _compress_view(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_compress_view(item) for item in value]
    return value


@dataclass(frozen=True)
class PredictionEvidence:
    series_id: str
    metric: str
    level: str
    target_week: int
    actual: float
    expected: float
    residual: float
    relative_residual: float
    standardized_residual: float
    nominal_percentile: float
    calibrated_percentile: float | None
    interval_lower: float | None
    interval_upper: float | None
    interval_alpha: float | None
    interval_width: float | None
    interval_coverage: float | None
    calibration_status: str
    calibration_pool: str
    calibration_n: int
    forecast_error: float | None
    history_weeks: int
    history_observed: int
    history_missing: int
    missing_weeks: tuple[int, ...]
    selected_model: str
    anomaly: bool
    flags: tuple[str, ...]
    horizon: int = 0
    support: str = "model"
    materiality: float = 0.0
    training_endpoint_week: int = 0
    forecast_origin_week: int = 0
    aggregation: str = "flow"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["missing_weeks"] = list(self.missing_weeks)
        payload["flags"] = list(self.flags)
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PredictionEvidence:
        fields: dict[str, Any] = {
            "series_id": str(data.get("series_id", "")),
            "metric": str(data.get("metric", "")),
            "level": str(data.get("level", "")),
            "target_week": int(data.get("target_week", 0)),
            "actual": float(data.get("actual", 0.0)),
            "expected": float(data.get("expected", 0.0)),
            "residual": float(data.get("residual", 0.0)),
            "relative_residual": float(data.get("relative_residual", 0.0)),
            "standardized_residual": float(data.get("standardized_residual", 0.0)),
            "nominal_percentile": float(data.get("nominal_percentile", 0.0)),
            "calibrated_percentile": data.get("calibrated_percentile"),
            "interval_lower": data.get("interval_lower"),
            "interval_upper": data.get("interval_upper"),
            "interval_alpha": data.get("interval_alpha"),
            "interval_width": data.get("interval_width"),
            "interval_coverage": data.get("interval_coverage"),
            "calibration_status": str(data.get("calibration_status", "UNAVAILABLE")),
            "calibration_pool": str(data.get("calibration_pool", "")),
            "calibration_n": int(data.get("calibration_n", 0)),
            "forecast_error": data.get("forecast_error"),
            "history_weeks": int(data.get("history_weeks", 0)),
            "history_observed": int(data.get("history_observed", 0)),
            "history_missing": int(data.get("history_missing", 0)),
            "missing_weeks": tuple(int(value) for value in data.get("missing_weeks", [])),
            "selected_model": str(data.get("selected_model", "")),
            "anomaly": bool(data.get("anomaly", False)),
            "flags": tuple(str(value) for value in data.get("flags", [])),
            "horizon": int(data.get("horizon", 0)),
            "support": str(data.get("support", "model")),
            "materiality": float(data.get("materiality", 0.0)),
            "training_endpoint_week": int(data.get("training_endpoint_week", 0)),
            "forecast_origin_week": int(data.get("forecast_origin_week", 0)),
            "aggregation": str(data.get("aggregation", "flow")),
        }
        return cls(**fields)


@dataclass(frozen=True)
class ContributionEvidence:
    entity_type: str
    entity_id: str
    delta: float
    evidence_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["evidence_ids"] = list(self.evidence_ids)
        return payload


@dataclass(frozen=True)
class AssessmentEvidence:
    assessment_id: str
    run_id: str
    dataset: str
    observation_cutoff: str | None
    snapshot_identity: dict[str, Any]
    calendar_identity: dict[str, Any]
    model_identity: dict[str, Any]
    config_identity: dict[str, Any]
    predictions: tuple[PredictionEvidence, ...] = ()
    contracts: tuple[dict[str, Any], ...] = ()
    reconciliation: dict[str, Any] | None = None
    lineage: dict[str, Any] | None = None
    contributions: tuple[ContributionEvidence, ...] = ()
    ledger: dict[str, Any] | None = None
    metric_ledgers: dict[str, Any] = field(default_factory=dict)
    cross_metric: tuple[str, ...] = ()
    contradictions: tuple[str, ...] = ()
    net_unexplained: float = 0.0
    gross_unexplained: float = 0.0
    explained_fraction: float = 1.0
    materiality_threshold: float = 0.0
    evidence_ids: tuple[str, ...] = ()
    omitted: tuple[str, ...] = ()
    approval_ids: tuple[str, ...] = ()
    provenance: str = "synthetic"
    target_week: int | None = None
    failed_checks: tuple[dict[str, Any], ...] = ()
    missing_required: tuple[str, ...] = ()
    qualification_digest: str = ""
    qualification_provenance: str = ""
    schema_version: int = EVIDENCE_PACKAGE_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "assessment_id": self.assessment_id,
            "run_id": self.run_id,
            "dataset": self.dataset,
            "observation_cutoff": self.observation_cutoff,
            "snapshot_identity": self.snapshot_identity,
            "calendar_identity": self.calendar_identity,
            "model_identity": self.model_identity,
            "config_identity": self.config_identity,
            "predictions": [item.to_dict() for item in self.predictions],
            "contracts": list(self.contracts),
            "reconciliation": self.reconciliation,
            "lineage": self.lineage,
            "contributions": [item.to_dict() for item in self.contributions],
            "ledger": self.ledger,
            "metric_ledgers": dict(self.metric_ledgers),
            "cross_metric": list(self.cross_metric),
            "contradictions": list(self.contradictions),
            "net_unexplained": self.net_unexplained,
            "gross_unexplained": self.gross_unexplained,
            "explained_fraction": self.explained_fraction,
            "materiality_threshold": self.materiality_threshold,
            "evidence_ids": list(self.evidence_ids),
            "omitted": list(self.omitted),
            "approval_ids": list(self.approval_ids),
            "provenance": self.provenance,
            "target_week": self.target_week,
            "failed_checks": [dict(item) for item in self.failed_checks],
            "missing_required": list(self.missing_required),
            "qualification_digest": self.qualification_digest,
            "qualification_provenance": self.qualification_provenance,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AssessmentEvidence:
        return cls(
            assessment_id=str(data.get("assessment_id", "")),
            run_id=str(data.get("run_id", "")),
            dataset=str(data.get("dataset", "")),
            observation_cutoff=data.get("observation_cutoff"),
            snapshot_identity=dict(data.get("snapshot_identity", {})),
            calendar_identity=dict(data.get("calendar_identity", {})),
            model_identity=dict(data.get("model_identity", {})),
            config_identity=dict(data.get("config_identity", {})),
            predictions=tuple(
                PredictionEvidence.from_dict(item)
                for item in data.get("predictions", [])
            ),
            contracts=tuple(dict(item) for item in data.get("contracts", [])),
            reconciliation=data.get("reconciliation"),
            lineage=data.get("lineage"),
            contributions=tuple(
                ContributionEvidence(
                    entity_type=str(item.get("entity_type", "")),
                    entity_id=str(item.get("entity_id", "")),
                    delta=float(item.get("delta", 0.0)),
                    evidence_ids=tuple(item.get("evidence_ids", [])),
                )
                for item in data.get("contributions", [])
            ),
            ledger=data.get("ledger"),
            metric_ledgers={
                str(key): value
                for key, value in dict(data.get("metric_ledgers", {})).items()
            },
            cross_metric=tuple(str(value) for value in data.get("cross_metric", [])),
            contradictions=tuple(
                str(value) for value in data.get("contradictions", [])
            ),
            net_unexplained=float(data.get("net_unexplained", 0.0)),
            gross_unexplained=float(data.get("gross_unexplained", 0.0)),
            explained_fraction=float(data.get("explained_fraction", 1.0)),
            materiality_threshold=float(data.get("materiality_threshold", 0.0)),
            evidence_ids=tuple(str(value) for value in data.get("evidence_ids", [])),
            omitted=tuple(str(value) for value in data.get("omitted", [])),
            approval_ids=tuple(str(value) for value in data.get("approval_ids", [])),
            provenance=str(data.get("provenance", "synthetic")),
            target_week=data.get("target_week"),
            failed_checks=tuple(
                dict(item) for item in data.get("failed_checks", [])
            ),
            missing_required=tuple(
                str(value) for value in data.get("missing_required", [])
            ),
            qualification_digest=str(data.get("qualification_digest", "")),
            qualification_provenance=str(
                data.get("qualification_provenance", "")
            ),
            schema_version=int(data.get("schema_version", 0)),
        )

    @property
    def digest(self) -> str:
        return evidence_digest(self.to_dict())

    def mandatory_payload(self) -> dict[str, Any]:
        """Evidence a provider view must retain to be usable at all.

        Failed checks and their compact structured evidence, missing required
        inputs, uncertainty and identity are mandatory by decision dependency
        and are never dropped, only compressed.
        """
        return {
            "schema_version": self.schema_version,
            "assessment_id": self.assessment_id,
            "run_id": self.run_id,
            "dataset": self.dataset,
            "observation_cutoff": self.observation_cutoff,
            "snapshot_identity": self.snapshot_identity,
            "calendar_identity": self.calendar_identity,
            "model_identity": self.model_identity,
            "net_unexplained": self.net_unexplained,
            "gross_unexplained": self.gross_unexplained,
            "materiality_threshold": self.materiality_threshold,
            "contradictions": list(self.contradictions),
            "failed_checks": [dict(item) for item in self.failed_checks],
            "missing_required": list(self.missing_required),
            "omitted": list(self.omitted),
        }

    def provider_predictions(self) -> list[dict[str, Any]]:
        """Underlying prediction evidence with derived verdicts removed.

        Final status, anomaly verdicts, flags and certificate outcomes are
        policy outputs and never enter a provider input. Forecast uncertainty,
        residuals, calibration support and provenance remain.
        """
        output: list[dict[str, Any]] = []
        for item in self.predictions:
            payload = item.to_dict()
            for key in ("anomaly", "flags"):
                payload.pop(key, None)
            output.append(payload)
        return output

    def provider_view(
        self, budget_chars: int, level: str = "complete"
    ) -> dict[str, Any]:
        """Compress for a provider, or abstain when mandatory evidence will not fit.

        ``level`` constructs an independent evidence view for ablation:
        ``summary`` retains mandatory identity, uncertainty, integrity results
        and failed-check references only, ``forecast`` adds predictions,
        ``hierarchy`` adds ledger and contributions, and ``complete`` adds
        cross-metric and relationship evidence. Verbose details are compressed
        into bounded references before optional sections are dropped, and every
        omission is recorded after packing completes. Mandatory evidence is
        never dropped: the view abstains instead.
        """
        mandatory = _compress_view(self.mandatory_payload())
        if len(json_dumps(mandatory)) >= budget_chars:
            return {
                "abstain": True,
                "reason": "mandatory evidence exceeds provider budget",
                "schema_version": self.schema_version,
                "assessment_id": self.assessment_id,
            }
        view: dict[str, Any] = dict(mandatory)
        view["contracts"] = [
            _compact_check(check) for check in self.contracts
        ]
        view["provenance"] = {
            "assessment": self.provenance,
            "dataset": self.dataset,
            "run_id": self.run_id,
        }
        omissions: list[str] = []
        if level in ("forecast", "hierarchy", "complete"):
            view["predictions"] = _compress_view(self.provider_predictions())
        if level in ("hierarchy", "complete"):
            view["ledger"] = self.ledger
            view["metric_ledgers"] = dict(self.metric_ledgers)
            view["contributions"] = [item.to_dict() for item in self.contributions]
            view["reconciliation"] = self.reconciliation
            view["lineage"] = self.lineage
            view["explained_fraction"] = self.explained_fraction
        if level == "complete":
            view["cross_metric"] = list(self.cross_metric)
            view["evidence_ids"] = list(self.evidence_ids)
        view = _compress_view(view)

        def packed_size() -> int:
            view["omitted"] = [
                *self.omitted,
                *_bounded_omissions(omissions),
            ]
            return len(json_dumps(view))

        drop_order = (
            "cross_metric",
            "evidence_ids",
            "lineage",
            "reconciliation",
            "contributions",
            "ledger",
            "contracts",
            "predictions",
        )
        for section in drop_order:
            while packed_size() > budget_chars and view.get(section):
                container = view[section]
                if not isinstance(container, list):
                    view.pop(section, None)
                    omissions.append(section)
                    break
                popped = container.pop()
                if isinstance(popped, dict):
                    label = (
                        popped.get("series_id")
                        or popped.get("entity_id")
                        or len(omissions)
                    )
                else:
                    label = str(popped)
                omissions.append(f"{section}:{label}")
            if section in view and not view[section]:
                view.pop(section, None)
            if packed_size() <= budget_chars:
                break
        if packed_size() > budget_chars:
            return {
                "abstain": True,
                "reason": "mandatory evidence exceeds provider budget after compression",
                "schema_version": self.schema_version,
                "assessment_id": self.assessment_id,
            }
        return view


@dataclass(frozen=True)
class ExplanationCertificate:
    certificate_id: str
    assessment_id: str
    scope: str
    finding_ids: tuple[str, ...]
    bases: tuple[str, ...]
    status: str  # VERIFIED | REJECTED
    reasons: tuple[str, ...]
    support: dict[str, Any]
    coverage: dict[str, Any]
    net_unexplained: float
    gross_unexplained: float
    materiality_threshold: float
    evidence_ids: tuple[str, ...]
    approval_ids: tuple[str, ...]
    clearance_basis: str  # statistical | human_approval | informational
    evidence_digest: str = ""
    qualification_digest: str = ""
    scope_type: str = ""
    period: int | None = None
    metric: str = ""
    level: str = ""
    policy_version: str = EVIDENCE_POLICY_VERSION
    schema_version: int = CERTIFICATE_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "certificate_id": self.certificate_id,
            "assessment_id": self.assessment_id,
            "scope": self.scope,
            "finding_ids": list(self.finding_ids),
            "bases": list(self.bases),
            "status": self.status,
            "reasons": list(self.reasons),
            "support": self.support,
            "coverage": self.coverage,
            "net_unexplained": self.net_unexplained,
            "gross_unexplained": self.gross_unexplained,
            "materiality_threshold": self.materiality_threshold,
            "evidence_ids": list(self.evidence_ids),
            "approval_ids": list(self.approval_ids),
            "clearance_basis": self.clearance_basis,
            "evidence_digest": self.evidence_digest,
            "qualification_digest": self.qualification_digest,
            "scope_type": self.scope_type,
            "period": self.period,
            "metric": self.metric,
            "level": self.level,
            "policy_version": self.policy_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExplanationCertificate:
        return cls(
            certificate_id=str(data.get("certificate_id", "")),
            assessment_id=str(data.get("assessment_id", "")),
            scope=str(data.get("scope", "")),
            finding_ids=tuple(data.get("finding_ids", [])),
            bases=tuple(data.get("bases", [])),
            status=str(data.get("status", "REJECTED")),
            reasons=tuple(data.get("reasons", [])),
            support=dict(data.get("support", {})),
            coverage=dict(data.get("coverage", {})),
            net_unexplained=float(data.get("net_unexplained", 0.0)),
            gross_unexplained=float(data.get("gross_unexplained", 0.0)),
            materiality_threshold=float(data.get("materiality_threshold", 0.0)),
            evidence_ids=tuple(data.get("evidence_ids", [])),
            approval_ids=tuple(data.get("approval_ids", [])),
            clearance_basis=str(data.get("clearance_basis", "")),
            evidence_digest=str(data.get("evidence_digest", "")),
            qualification_digest=str(data.get("qualification_digest", "")),
            scope_type=str(data.get("scope_type", "")),
            period=data.get("period"),
            metric=str(data.get("metric", "")),
            level=str(data.get("level", "")),
            policy_version=str(data.get("policy_version", "")),
            schema_version=int(data.get("schema_version", 0)),
        )


def _certificate_id(
    assessment_id: str,
    scope: str,
    basis: str,
    finding_ids: tuple[str, ...],
    evidence_digest: str,
    qualification_digest: str = "",
) -> str:
    identity = json_dumps(
        [
            CERTIFICATE_SCHEMA,
            assessment_id,
            scope,
            basis,
            list(finding_ids),
            evidence_digest,
            qualification_digest,
        ],
        separators=(",", ":"),
    )
    return hashlib.sha256(identity.encode()).hexdigest()[:24]


def _annual_model(selected_model: str) -> bool:
    return "fourier" in str(selected_model).lower()


def _qualification_reasons(
    qualification: Any,
    provenance: str,
    prediction: PredictionEvidence,
    evidence_digest_value: str,
) -> list[str]:
    reasons: list[str] = []
    if qualification is None:
        return ["no pinned qualification artifact"]
    if getattr(qualification, "status", "") != "QUALIFIED":
        reasons.append("qualification artifact is unqualified")
    if getattr(qualification, "provenance", "") != provenance:
        reasons.append(
            "qualification provenance does not match the assessment provenance"
        )
    digest = str(getattr(qualification, "digest", "") or "")
    if not digest:
        reasons.append("qualification artifact has no identity digest")
    elif evidence_digest_value and digest != evidence_digest_value:
        reasons.append("qualification identity does not match this evidence")
    if not qualification.supports(
        prediction.selected_model,
        prediction.metric,
        prediction.level,
        prediction.horizon,
    ):
        reasons.append("model/metric/level/horizon combination is not qualified")
    required_coverage = float(getattr(qualification, "required_coverage", 1.0))
    coverage = prediction.interval_coverage
    if coverage is None:
        reasons.append("independent held-out forecast coverage is unavailable")
    elif float(coverage) < required_coverage:
        reasons.append("held-out forecast coverage is below qualification")
    max_width_ratio = float(getattr(qualification, "max_width_ratio", 0.0))
    expected = abs(float(prediction.expected))
    if max_width_ratio <= 0.0 or not math.isfinite(max_width_ratio):
        reasons.append("qualification width policy is not registered")
    elif prediction.interval_width is not None and math.isfinite(
        float(prediction.interval_width)
    ):
        if float(prediction.interval_width) > max_width_ratio * expected:
            reasons.append("interval width exceeds the qualification limit")
    elif prediction.interval_width is None:
        reasons.append("interval width is unavailable")
    return reasons


def _blocking_findings(findings: Any) -> list[Any]:
    """Applicable failures that can never be cleared by a certificate."""
    from .policy import BLOCKING_SCOPE_KINDS

    blocking: list[Any] = []
    for item in findings:
        outcome = str(getattr(item, "outcome", ""))
        scope_type = str(getattr(item, "scope_type", ""))
        required = bool(getattr(item, "required", False))
        if outcome == "CONTRACT_FAILURE" or scope_type == "contract":
            blocking.append(item)
        elif scope_type in BLOCKING_SCOPE_KINDS and (
            outcome == "FAIL"
            or (outcome in ("UNAVAILABLE", "SKIPPED") and required)
        ):
            blocking.append(item)
    return blocking


def _contradiction_reasons(
    evidence: AssessmentEvidence, prediction: PredictionEvidence
) -> list[str]:
    """Structured contradictions that apply to this prediction's metric."""
    metric = str(prediction.metric or "")
    applicable: list[str] = []
    for entry in evidence.contradictions:
        text = str(entry)
        if text in ("over_explained", "offsetting_explanations"):
            applicable.append(text)
        elif metric and (
            text.startswith(f"{metric}_") or text.startswith(f"{metric}:")
        ):
            applicable.append(text)
    return applicable


def _dataset_certificate(
    evidence: AssessmentEvidence,
    errors: tuple[Any, ...],
) -> ExplanationCertificate:
    """Assessment-scope record; it never authorizes clearing a finding."""
    temporal_findings = tuple(
        str(item.finding_id)
        for item in errors
        if getattr(item, "finding_id", None)
    )
    reasons = ["assessment-scope certificate does not authorize clearance"]
    if not temporal_findings:
        reasons.append("no eligible findings covered")
    return ExplanationCertificate(
        certificate_id=_certificate_id(
            evidence.assessment_id, "dataset", "informational", (), evidence.digest
        ),
        assessment_id=evidence.assessment_id,
        scope="dataset",
        finding_ids=temporal_findings,
        bases=(),
        status="REJECTED",
        reasons=tuple(reasons),
        support={},
        coverage={
            "finding_ids": list(temporal_findings),
            "contracts": len(evidence.contracts),
        },
        net_unexplained=evidence.net_unexplained,
        gross_unexplained=evidence.gross_unexplained,
        materiality_threshold=evidence.materiality_threshold,
        evidence_ids=evidence.evidence_ids,
        approval_ids=evidence.approval_ids,
        clearance_basis="informational",
        evidence_digest=evidence.digest,
        scope_type="dataset",
    )


def verify_explanations(
    evidence: AssessmentEvidence,
    config: Any,
    materiality_threshold: float,
    findings: Any = (),
    qualification: Any = None,
    provenance: str | None = None,
) -> tuple[ExplanationCertificate, ...]:
    """Verify one statistical clearance certificate per eligible finding.

    Clearance is scoped to a single temporal series and period and requires
    successful applicable integrity checks, complete mandatory evidence,
    supported history, a pinned qualification covering the exact
    model/metric/level/horizon, held-out forecast coverage, a finite interval
    within the qualification width limit, no contradictory evidence for that
    scope and a residual below the scoped materiality. The explained fraction
    alone is never sufficient, and findings on contracts, reconciliation,
    references, lineage, hierarchy integrity, required evidence or historical
    revisions cannot receive statistical clearance at all.
    """
    from .policy import CLEARABLE_CHECKS, SCOPE_TEMPORAL

    provenance = provenance or evidence.provenance
    required_n = minimum_samples(config.temporal_interval_alpha)
    integrity_ok = bool(evidence.contracts) and not any(
        _contract_failed(item.get("status")) for item in evidence.contracts
    )
    finding_list = list(findings)
    blocking = _blocking_findings(finding_list)
    certificates: list[ExplanationCertificate] = [
        _dataset_certificate(
            evidence,
            tuple(
                item
                for item in finding_list
                if getattr(item, "period", None) is not None
            ),
        )
    ]
    for item in finding_list:
        if getattr(item, "outcome", None) != "FAIL":
            continue
        if getattr(item, "check", None) not in CLEARABLE_CHECKS:
            continue
        if getattr(item, "scope_type", "") != SCOPE_TEMPORAL:
            continue
        if getattr(item, "approval_ids", ()):
            continue
        period = getattr(item, "period", None)
        prediction = next(
            (
                entry
                for entry in evidence.predictions
                if entry.series_id == item.scope
                and (period is None or entry.target_week == period)
                and (not item.metric or entry.metric == item.metric)
            ),
            None,
        )
        reasons: list[str] = []
        bases: list[str] = []
        if prediction is None:
            reasons.append("no matching period-specific prediction")
        if not integrity_ok:
            reasons.append("integrity check failed")
        if not evidence.contracts:
            reasons.append("mandatory integrity evidence is missing")
        if blocking:
            reasons.append(
                "applicable integrity failures present: "
                + ", ".join(
                    sorted(
                        {
                            f"{getattr(entry, 'check', '')}:{getattr(entry, 'scope', '')}"
                            for entry in blocking
                        }
                    )
                )
            )
        contradictions: list[str] = []
        if prediction is not None:
            contradictions = _contradiction_reasons(evidence, prediction)
            if not all(
                math.isfinite(float(value))
                for value in (
                    prediction.actual,
                    prediction.expected,
                    prediction.residual,
                    prediction.relative_residual,
                )
            ):
                reasons.append("nonfinite prediction evidence")
            if prediction.support != "model":
                reasons.append("sparse or parent-share support is informational")
            if prediction.calibration_status != "OK":
                reasons.append("uncalibrated interval")
            if prediction.calibration_n < required_n:
                reasons.append(
                    f"insufficient calibration: {prediction.calibration_n} < {required_n}"
                )
            if (
                prediction.interval_lower is None
                or prediction.interval_upper is None
                or not (
                    float("-inf") < prediction.interval_lower <= prediction.interval_upper < float("inf")
                )
            ):
                reasons.append("no finite interval")
            elif not (
                prediction.interval_lower
                <= prediction.actual
                <= prediction.interval_upper
            ):
                reasons.append("observation outside the calibrated interval")
            minimum_history = (
                config.temporal_min_annual_history
                if _annual_model(prediction.selected_model)
                else min(config.temporal_min_history, 8)
            )
            if prediction.history_observed < minimum_history:
                reasons.append(
                    f"insufficient observed history: {prediction.history_observed} < {minimum_history}"
                )
            reasons.extend(
                _qualification_reasons(
                    qualification,
                    provenance,
                    prediction,
                    evidence.qualification_digest,
                )
            )
        if contradictions:
            reasons.append("contradictory evidence: " + ", ".join(contradictions))
        if prediction is not None:
            # Zero scoped materiality stays zero: never silently fall back to a
            # broader dataset threshold.
            limit = float(getattr(item, "materiality", 0.0))
            if not math.isfinite(limit) or limit < 0.0:
                reasons.append("invalid scoped materiality")
            elif abs(float(prediction.residual)) > limit:
                reasons.append("residual impact exceeds scoped materiality")
        if not math.isfinite(float(materiality_threshold)) or materiality_threshold < 0.0:
            reasons.append("invalid materiality threshold")
        if not reasons and prediction is not None:
            bases = ["historically_calibrated_interval"]
            if prediction.interval_coverage is not None:
                bases.append("held_out_forecast_coverage")
            if prediction.horizon:
                bases.append(f"horizon_{prediction.horizon}")
            ledger = (
                evidence.metric_ledgers.get(prediction.metric)
                or evidence.ledger
                or {}
            )
            for contribution in ledger.get("contributions", []):
                if contribution.get("category") != "new_period_expected":
                    continue
                basis = (contribution.get("detail") or {}).get("basis")
                if basis == "calendar":
                    bases.append("calendar_expectation")
                elif basis == "trend":
                    bases.append("trend_expectation")
                break
            if ledger.get("support_level"):
                bases.append(f"ledger_{ledger['support_level']}")
            if evidence.explained_fraction >= getattr(
                config, "explained_fraction_threshold", 0.9
            ):
                bases.append("explained_movement")
        status = "VERIFIED" if not reasons else "REJECTED"
        support_payload: dict[str, Any] = {}
        if prediction is not None:
            support_payload = {
                "calibration_pool": prediction.calibration_pool,
                "calibration_n": prediction.calibration_n,
                "interval_alpha": prediction.interval_alpha,
                "interval_width": prediction.interval_width,
                "interval_coverage": prediction.interval_coverage,
                "forecast_error": prediction.forecast_error,
                "selected_model": prediction.selected_model,
                "horizon": prediction.horizon,
                "history_observed": prediction.history_observed,
                "history_missing": prediction.history_missing,
                "materiality": float(getattr(item, "materiality", 0.0)),
            }
        finding_ids = (
            (item.finding_id,)
            if getattr(item, "finding_id", None)
            else ()
        )
        qualification_digest = (
            str(getattr(qualification, "digest", "") or "")
            if qualification is not None
            else ""
        )
        certificates.append(
            ExplanationCertificate(
                certificate_id=_certificate_id(
                    evidence.assessment_id,
                    str(getattr(item, "scope", "")),
                    "statistical",
                    finding_ids,
                    evidence.digest,
                    qualification_digest,
                ),
                assessment_id=evidence.assessment_id,
                scope=str(getattr(item, "scope", "")),
                finding_ids=finding_ids,
                bases=tuple(bases),
                status=status,
                reasons=tuple(reasons),
                support=support_payload,
                coverage={
                    "finding_ids": list(finding_ids),
                    "contracts": len(evidence.contracts),
                    "contradictions": len(contradictions),
                    "history_missing": (
                        prediction.history_missing if prediction else 0
                    ),
                },
                net_unexplained=evidence.net_unexplained,
                gross_unexplained=evidence.gross_unexplained,
                materiality_threshold=materiality_threshold,
                evidence_ids=evidence.evidence_ids,
                approval_ids=evidence.approval_ids,
                clearance_basis="statistical",
                evidence_digest=evidence.digest,
                qualification_digest=qualification_digest,
                scope_type=getattr(item, "scope_type", ""),
                period=getattr(item, "period", None),
                metric=getattr(item, "metric", ""),
                level=getattr(item, "level", ""),
            )
        )
    return tuple(certificates)


def build_assessment_evidence(
    result: Any,
    config: Any,
    *,
    assessment_id: str,
    observation_cutoff: str | None = None,
    findings: Any = (),
    target_week: int | None = None,
    ledger_override: Any | None = None,
    metric_ledgers: Mapping[str, Any] | None = None,
    qualification: Any = None,
) -> AssessmentEvidence:
    """Build the complete evidence package from a finished QC result.

    ``target_week`` builds the period-specific package (predictions and ledger
    for one appended period); ``ledger_override`` supplies that period's
    explanation ledger.
    """
    machine = getattr(result, "machine", {}) or {}
    temporal = getattr(result, "temporal", None)
    predictions: list[PredictionEvidence] = []
    if temporal is not None:
        for item in temporal.series:
            if target_week is not None and int(item.target_week) != int(target_week):
                continue
            predictions.append(
                PredictionEvidence(
                    series_id=item.series_id,
                    metric=item.metric or config.primary_metric,
                    level=item.level or ("national" if item.series_id == "national" else ""),
                    target_week=item.target_week,
                    actual=item.actual,
                    expected=item.forecast_median,
                    residual=item.residual,
                    relative_residual=item.relative_residual,
                    standardized_residual=item.standardized_residual,
                    nominal_percentile=item.nominal_percentile,
                    calibrated_percentile=item.calibrated_percentile,
                    interval_lower=item.interval_lower,
                    interval_upper=item.interval_upper,
                    interval_alpha=item.interval_alpha,
                    interval_width=item.interval_width,
                    interval_coverage=item.interval_coverage,
                    calibration_status=item.calibration_status,
                    calibration_pool=item.calibration_pool,
                    calibration_n=item.calibration_n,
                    forecast_error=item.forecast_error,
                    history_weeks=item.history_weeks,
                    history_observed=item.history_observed,
                    history_missing=item.history_missing,
                    missing_weeks=tuple(item.missing_weeks),
                    selected_model=item.selected_model,
                    anomaly=item.anomaly,
                    flags=tuple(item.flags),
                    horizon=int(getattr(item, "horizon", 0) or 0),
                    support=item.support,
                    materiality=max(
                        float(getattr(config, "temporal_materiality_abs", 0.0)),
                        float(getattr(config, "temporal_min_relative_residual", 0.02))
                        * abs(float(item.forecast_median)),
                    ),
                    training_endpoint_week=int(
                        getattr(item, "training_endpoint_week", 0) or 0
                    ),
                    forecast_origin_week=int(
                        getattr(item, "forecast_origin_week", 0) or 0
                    ),
                    aggregation=str(
                        getattr(item, "aggregation", "")
                        or (
                            "snapshot"
                            if item.metric in config.snapshot_metrics
                            else "flow"
                        )
                    ),
                )
            )
    contracts = tuple(
        {
            "name": check.name,
            "status": "PASS" if check.passed else "CONTRACT_FAILURE",
            "detail": check.detail,
        }
        for check in getattr(result.contracts, "checks", ())
    )
    reconciliation = (
        result.reconciliation.to_dict() if getattr(result, "reconciliation", None) else None
    )
    lineage = result.lineage.to_dict() if getattr(result, "lineage", None) else None
    attribution = getattr(result, "attribution", None)
    contributions = tuple(
        ContributionEvidence(
            entity_type=contributor.entity_type,
            entity_id=contributor.entity_id,
            delta=contributor.delta,
        )
        for contributor in (attribution.contributors if attribution else ())
    )
    ledger = (
        ledger_override
        if ledger_override is not None
        else getattr(result, "ledger", None)
    )
    contradictions: list[str] = []
    if attribution is not None:
        contradictions.extend(attribution.cross_metric_flags)
        if attribution.over_explained:
            contradictions.append("over_explained")
        if attribution.offsetting:
            contradictions.append("offsetting_explanations")
    failed_checks: list[dict[str, Any]] = []
    missing_required: list[str] = []
    for item in findings:
        outcome = str(getattr(item, "outcome", ""))
        required = bool(getattr(item, "required", False))
        if outcome not in ("FAIL", "CONTRACT_FAILURE", "UNAVAILABLE", "SKIPPED"):
            continue
        if outcome in ("UNAVAILABLE", "SKIPPED") and not required:
            continue
        failed_checks.append(
            {
                "check": str(getattr(item, "check", "")),
                "scope": str(getattr(item, "scope", "")),
                "scope_type": str(getattr(item, "scope_type", "")),
                "metric": str(getattr(item, "metric", "")),
                "level": str(getattr(item, "level", "")),
                "period": getattr(item, "period", None),
                "outcome": outcome,
                "impact": float(getattr(item, "impact", 0.0)),
                "materiality": float(getattr(item, "materiality", 0.0)),
                "finding_id": str(getattr(item, "finding_id", "")),
            }
        )
        if outcome in ("UNAVAILABLE", "SKIPPED"):
            missing_required.append(
                f"{getattr(item, 'check', '')}:{getattr(item, 'scope', '')}"
            )
    evidence_ids: list[str] = []
    graph = machine.get("evidence_graph") or {}
    for node in graph.get("nodes", []):
        if node.get("evidence_id"):
            evidence_ids.append(str(node["evidence_id"]))
    for item in findings:
        finding_id = getattr(item, "finding_id", None)
        if finding_id:
            evidence_ids.append(str(finding_id))
    for finding in machine.get("findings", []):
        if finding.get("finding_id"):
            evidence_ids.append(str(finding["finding_id"]))
    if ledger is not None:
        net_unexplained = float(ledger.net_unexplained)
        gross_unexplained = float(ledger.gross_unexplained)
    else:
        net_unexplained = getattr(attribution, "unexplained_delta", 0.0)
        offset = (
            min(attribution.explained_added, attribution.explained_removed)
            if attribution is not None
            else 0.0
        )
        gross_unexplained = abs(float(net_unexplained)) + float(offset)
    return AssessmentEvidence(
        assessment_id=assessment_id,
        run_id=str(machine.get("run_id", "")),
        dataset=str(machine.get("dataset", "")),
        observation_cutoff=observation_cutoff or machine.get("observed_at"),
        snapshot_identity=dict(machine.get("snapshot_manifest") or {}),
        calendar_identity=dict(
            machine.get("calendar")
            or (temporal.calendar if temporal is not None else {"declared": False})
        ),
        model_identity={
            "engine": "0.22.0",
            "forecaster": config.forecaster,
            "calibration_pools": [
                {
                    "key": entry.get("key"),
                    "n": entry.get("n"),
                    "mean_absolute_error": entry.get("mean_absolute_error"),
                }
                for entry in (
                    temporal.calibration.get("pools", []) if temporal is not None else []
                )
            ],
        },
        config_identity=config.to_dict(),
        predictions=tuple(predictions),
        contracts=contracts,
        reconciliation=reconciliation,
        lineage=lineage,
        contributions=contributions,
        ledger=ledger.to_dict() if ledger is not None else None,
        metric_ledgers={
            str(metric): value.to_dict()
            for metric, value in (metric_ledgers or {}).items()
        },
        cross_metric=tuple(attribution.cross_metric_flags) if attribution else (),
        contradictions=tuple(contradictions),
        net_unexplained=float(net_unexplained),
        gross_unexplained=gross_unexplained,
        explained_fraction=(
            float(attribution.explained_fraction) if attribution else 1.0
        ),
        materiality_threshold=float(machine.get("materiality_threshold", 0.0)),
        evidence_ids=tuple(dict.fromkeys(evidence_ids)),
        approval_ids=tuple(
            dict.fromkeys(
                str(value)
                for item in machine.get("approval_coverage", [])
                for value in item.get("approval_ids", [])
            )
        ),
        provenance=str(machine.get("assessment_provenance", "synthetic")),
        target_week=(
            int(target_week)
            if target_week is not None
            else (int(temporal.target_week) if temporal is not None else None)
        ),
        failed_checks=tuple(failed_checks),
        missing_required=tuple(dict.fromkeys(missing_required)),
        qualification_digest=(
            str(getattr(qualification, "digest", "") or "")
            if qualification is not None
            else ""
        ),
        qualification_provenance=(
            str(getattr(qualification, "provenance", "") or "")
            if qualification is not None
            else ""
        ),
    )
