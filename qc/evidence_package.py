"""Versioned assessment evidence and explanation certificates (schema 1).

``AssessmentEvidence`` is the complete, serializable record behind a review
decision: predictions with calibration and intervals, integrity checks,
contributions, contradictions, net and gross unexplained residuals and the
identities (snapshot, calendar, model, configuration, cutoff) that pin it.
``ExplanationCertificate`` names the exact findings a scoped clearance covers,
the support it rests on and the remainder that stays unexplained.

Providers receive a compressed view. The compression is allowed to drop detail
but never mandatory evidence: when the budget cannot carry the mandatory
fields the view abstains instead.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any

from .conformal import minimum_samples
from .jsonutil import dumps as json_dumps

EVIDENCE_PACKAGE_SCHEMA = 1
EVIDENCE_POLICY_VERSION = "1"
CERTIFICATE_SCHEMA = 1


def evidence_digest(value: Any) -> str:
    return hashlib.sha256(
        json_dumps(value, sort_keys=True, default=str).encode()
    ).hexdigest()


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
    cross_metric: tuple[str, ...] = ()
    contradictions: tuple[str, ...] = ()
    net_unexplained: float = 0.0
    gross_unexplained: float = 0.0
    explained_fraction: float = 1.0
    materiality_threshold: float = 0.0
    evidence_ids: tuple[str, ...] = ()
    omitted: tuple[str, ...] = ()
    approval_ids: tuple[str, ...] = ()
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
            "cross_metric": list(self.cross_metric),
            "contradictions": list(self.contradictions),
            "net_unexplained": self.net_unexplained,
            "gross_unexplained": self.gross_unexplained,
            "explained_fraction": self.explained_fraction,
            "materiality_threshold": self.materiality_threshold,
            "evidence_ids": list(self.evidence_ids),
            "omitted": list(self.omitted),
            "approval_ids": list(self.approval_ids),
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
            schema_version=int(data.get("schema_version", 1)),
        )

    @property
    def digest(self) -> str:
        return evidence_digest(self.to_dict())

    def mandatory_payload(self) -> dict[str, Any]:
        """Evidence a provider view must retain to be usable at all."""
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
            "omitted": list(self.omitted),
        }

    def provider_view(self, budget_chars: int) -> dict[str, Any]:
        """Compress for a provider, or abstain when mandatory evidence will not fit."""
        mandatory = self.mandatory_payload()
        if len(json_dumps(mandatory)) >= budget_chars:
            return {
                "abstain": True,
                "reason": "mandatory evidence exceeds provider budget",
                "schema_version": self.schema_version,
                "assessment_id": self.assessment_id,
            }
        view: dict[str, Any] = dict(mandatory)
        view["contracts"] = list(self.contracts)
        view["reconciliation"] = self.reconciliation
        view["lineage"] = self.lineage
        view["predictions"] = [item.to_dict() for item in self.predictions]
        omitted: list[str] = []
        for section in ("predictions", "contributions", "contracts"):
            while len(json_dumps(view)) > budget_chars and view.get(section):
                popped = view[section].pop()
                omitted.append(
                    f"{section}:{popped.get('series_id') or popped.get('entity_id') or len(omitted)}"
                )
        view["omitted"] = [*self.omitted, *omitted]
        view["cross_metric"] = list(self.cross_metric)
        view["evidence_ids"] = list(self.evidence_ids)
        if len(json_dumps(view)) > budget_chars:
            return {
                "abstain": True,
                "reason": "evidence exceeds provider budget after compression",
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
    clearance_basis: str  # statistical | human_approval
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
            clearance_basis=str(data.get("clearance_basis", "statistical")),
            policy_version=str(data.get("policy_version", EVIDENCE_POLICY_VERSION)),
            schema_version=int(data.get("schema_version", 1)),
        )


def _certificate_id(assessment_id: str, scope: str, basis: str) -> str:
    identity = json_dumps(
        [CERTIFICATE_SCHEMA, assessment_id, scope, basis], separators=(",", ":")
    )
    return hashlib.sha256(identity.encode()).hexdigest()[:24]


def _dataset_certificate(
    evidence: AssessmentEvidence,
    config: Any,
    materiality_threshold: float,
    required_n: int,
    integrity_ok: bool,
    finding_ids: tuple[str, ...],
) -> ExplanationCertificate:
    """One dataset-scope certificate for a historical-revision clearance."""
    reasons: list[str] = []
    bases: list[str] = []
    ledger = evidence.ledger or {}
    if not integrity_ok:
        reasons.append("integrity check failed")
    if not ledger:
        reasons.append("no explanation ledger")
    else:
        conservation = ledger.get("conservation") or {}
        difference = abs(float(conservation.get("difference", 0.0)))
        scale = max(1.0, abs(float(ledger.get("raw_movement", 0.0))))
        if difference > 1e-6 * scale:
            reasons.append("ledger does not conserve")
        overlap_delta = float(ledger.get("overlap_delta", 0.0))
        supported_overlap = float(
            sum(
                float(item.get("value", 0.0))
                for item in ledger.get("contributions", [])
                if item.get("scope") == "overlap"
                and item.get("support") in ("statistical", "approved_event")
            )
        )
        unsupported = overlap_delta - supported_overlap
        if abs(unsupported) > materiality_threshold:
            reasons.append(
                "revision lacks statistical or approved-event support"
            )
        else:
            bases.append("supported_movement")
        if ledger.get("support_level") == "verified":
            bases.append("ledger_verified")
        else:
            reasons.append("ledger support is not verified")
    if evidence.contradictions:
        reasons.append(
            "contradictory evidence: " + ", ".join(evidence.contradictions)
        )
    national = next(
        (item for item in evidence.predictions if item.series_id == "national"),
        None,
    )
    if (
        national is None
        or national.calibration_status != "OK"
        or national.calibration_n < required_n
    ):
        reasons.append("no calibrated national support")
    else:
        bases.append("calibrated_national_support")
    overlap_delta = float(ledger.get("overlap_delta", 0.0)) if ledger else 0.0
    supported_overlap = (
        sum(
            float(item.get("value", 0.0))
            for item in ledger.get("contributions", [])
            if item.get("scope") == "overlap"
            and item.get("support") in ("statistical", "approved_event")
        )
        if ledger
        else 0.0
    )
    support = {
        "ledger_support_level": ledger.get("support_level"),
        "overlap_delta": overlap_delta,
        "supported_overlap": supported_overlap,
        "unsupported_overlap": overlap_delta - supported_overlap,
        "materiality_threshold": materiality_threshold,
        "national_calibration_n": national.calibration_n if national else 0,
    }
    return ExplanationCertificate(
        certificate_id=_certificate_id(
            evidence.assessment_id, "dataset", "statistical"
        ),
        assessment_id=evidence.assessment_id,
        scope="dataset",
        finding_ids=finding_ids,
        bases=tuple(bases),
        status="VERIFIED" if not reasons else "REJECTED",
        reasons=tuple(reasons),
        support=support,
        coverage={"contradictions": len(evidence.contradictions)},
        net_unexplained=evidence.net_unexplained,
        gross_unexplained=evidence.gross_unexplained,
        materiality_threshold=materiality_threshold,
        evidence_ids=evidence.evidence_ids,
        approval_ids=evidence.approval_ids,
        clearance_basis="statistical",
    )


def verify_explanations(
    evidence: AssessmentEvidence,
    config: Any,
    materiality_threshold: float,
    finding_ids: tuple[str, ...] = (),
) -> tuple[ExplanationCertificate, ...]:
    """Verify one statistical clearance certificate per assessed series.

    Clearance is scoped to a single series and requires historically calibrated
    support, an interval that contains the observation, no contradictory
    evidence for that scope and a residual below the frozen materiality
    threshold. A wide or unqualified interval cannot justify clearance, and the
    explained fraction alone is never sufficient.
    """
    certificates: list[ExplanationCertificate] = []
    required_n = minimum_samples(config.temporal_interval_alpha)
    integrity_ok = not any(
        item.get("status") == "DATA_CONTRACT_FAILURE" for item in evidence.contracts
    )
    certificates.append(
        _dataset_certificate(
            evidence,
            config,
            materiality_threshold,
            required_n,
            integrity_ok,
            finding_ids,
        )
    )
    for prediction in evidence.predictions:
        scope = prediction.series_id
        reasons: list[str] = []
        bases: list[str] = []
        if not integrity_ok:
            reasons.append("integrity check failed")
        if prediction.calibration_status != "OK":
            reasons.append("uncalibrated interval")
        if prediction.calibration_n < required_n:
            reasons.append(
                f"insufficient calibration: {prediction.calibration_n} < {required_n}"
            )
        if prediction.interval_lower is None or prediction.interval_upper is None:
            reasons.append("no interval")
        elif not (
            prediction.interval_lower
            <= prediction.actual
            <= prediction.interval_upper
        ):
            reasons.append("observation outside the calibrated interval")
        contradictions = [
            item
            for item in evidence.contradictions
            if item == scope or item.endswith(f":{scope}")
        ]
        if contradictions:
            reasons.append("contradictory evidence: " + ", ".join(contradictions))
        ledger = evidence.ledger or {}
        conservation = ledger.get("conservation") or {}
        difference = abs(float(conservation.get("difference", 0.0)))
        scale = max(1.0, abs(float(ledger.get("raw_movement", 0.0))))
        if ledger and difference > 1e-6 * scale:
            reasons.append("ledger does not conserve")
        if abs(prediction.residual) > materiality_threshold:
            reasons.append("residual impact exceeds materiality")
        if not reasons:
            bases = ["historically_calibrated_interval"]
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
            if prediction.relative_residual:
                bases.append("expected_behaviour")
            if evidence.explained_fraction >= getattr(
                config, "explained_fraction_threshold", 0.9
            ):
                bases.append("explained_movement")
        status = "VERIFIED" if not reasons else "REJECTED"
        support = {
            "calibration_pool": prediction.calibration_pool,
            "calibration_n": prediction.calibration_n,
            "interval_alpha": prediction.interval_alpha,
            "interval_width": prediction.interval_width,
            "interval_coverage": prediction.interval_coverage,
            "forecast_error": prediction.forecast_error,
            "selected_model": prediction.selected_model,
            "history_observed": prediction.history_observed,
            "history_missing": prediction.history_missing,
        }
        certificates.append(
            ExplanationCertificate(
                certificate_id=_certificate_id(
                    evidence.assessment_id, scope, "statistical"
                ),
                assessment_id=evidence.assessment_id,
                scope=scope,
                finding_ids=finding_ids,
                bases=tuple(bases),
                status=status,
                reasons=tuple(reasons),
                support=support,
                coverage={
                    "contracts": len(evidence.contracts),
                    "contradictions": len(contradictions),
                    "history_missing": prediction.history_missing,
                },
                net_unexplained=evidence.net_unexplained,
                gross_unexplained=evidence.gross_unexplained,
                materiality_threshold=materiality_threshold,
                evidence_ids=evidence.evidence_ids,
                approval_ids=evidence.approval_ids,
                clearance_basis="statistical",
            )
        )
    return tuple(certificates)


def build_assessment_evidence(
    result: Any,
    config: Any,
    *,
    assessment_id: str,
    observation_cutoff: str | None = None,
) -> AssessmentEvidence:
    """Build the complete evidence package from a finished QC result."""
    machine = getattr(result, "machine", {}) or {}
    temporal = getattr(result, "temporal", None)
    predictions: list[PredictionEvidence] = []
    if temporal is not None:
        for item in temporal.series:
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
    ledger = getattr(result, "ledger", None)
    contradictions: list[str] = []
    if attribution is not None:
        contradictions.extend(attribution.cross_metric_flags)
        if attribution.over_explained:
            contradictions.append("over_explained")
        if attribution.offsetting:
            contradictions.append("offsetting_explanations")
    evidence_ids: list[str] = []
    graph = machine.get("evidence_graph") or {}
    for node in graph.get("nodes", []):
        if node.get("evidence_id"):
            evidence_ids.append(str(node["evidence_id"]))
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
            "engine": "0.20.0",
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
    )
