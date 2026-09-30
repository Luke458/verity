"""Versioned deterministic findings and the effective analyst review policy.

Findings carry an explicit disposition so the review decision can distinguish a
hard failure, missing required evidence, an unexplained actionable anomaly, a
statistically explained movement, a human-approved exception and an
informational finding below escalation thresholds. Statistical clearance is
granted only through a verified explanation certificate bound to the exact
finding, assessment identity, evidence digest and policy version; no model can
override the verifier.

The policy runs in two phases so certificates can bind to immutable findings:

1. ``collect_findings`` runs every check and materializes findings (including
   period-specific temporal findings and recurrence escalations).
2. ``apply_policy`` verifies certificate binding and computes the final status
   exactly once.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, replace
from typing import Any

from .jsonutil import dumps as json_dumps

FINDING_SCHEMA_VERSION = 4
MACHINE_SCHEMA_VERSION = 5

HARD_FAILURE = "HARD_FAILURE"
UNAVAILABLE_EVIDENCE = "UNAVAILABLE_EVIDENCE"
UNEXPLAINED_ANOMALY = "UNEXPLAINED_ANOMALY"
STATISTICALLY_EXPLAINED = "STATISTICALLY_EXPLAINED"
HUMAN_APPROVED = "HUMAN_APPROVED"
INFORMATIONAL = "INFORMATIONAL"
PASSING = "PASS"

# Only these checks may ever receive automatic statistical clearance: an
# explicitly eligible temporal anomaly scoped to one series and period.
CLEARABLE_CHECKS = frozenset({"temporal"})

# Outcomes a check may report. Unknown outcomes are rejected when findings are
# materialized and treated as failures if they somehow reach the policy.
KNOWN_OUTCOMES = frozenset(
    {"PASS", "FAIL", "UNAVAILABLE", "CONTRACT_FAILURE", "SKIPPED"}
)
# Scope kinds whose failures can never be cleared by a certificate.
BLOCKING_SCOPE_KINDS = frozenset(
    {
        "contract",
        "input",
        "reference",
        "reconciliation",
        "lineage",
        "hierarchy",
        "historical_revision",
        "revision",
        "approval",
        "counterfactual",
    }
)

# Structured scope kinds. Reference identifiers are never interchangeable with
# hierarchy scopes and each finding declares which namespace its scope uses.
SCOPE_CONTRACT = "contract"
SCOPE_INPUT = "input"
SCOPE_HISTORICAL_REVISION = "historical_revision"
SCOPE_RECONCILIATION = "reconciliation"
SCOPE_COUNTERFACTUAL = "counterfactual"
SCOPE_REVISION = "revision"
SCOPE_REFERENCE = "reference"
SCOPE_APPROVAL = "approval"
SCOPE_TEMPORAL = "temporal_series"
SCOPE_RECURRENCE = "recurrence"
SCOPE_HIERARCHY = "hierarchy"
SCOPE_LINEAGE = "lineage"

_CHECK_SCOPE_KINDS = {
    "historical_revision": SCOPE_HISTORICAL_REVISION,
    "counterfactual_residual": SCOPE_COUNTERFACTUAL,
    "revision_missing_evidence": SCOPE_REVISION,
    "reference": SCOPE_REFERENCE,
    "ratio:dollar_per_unit": SCOPE_APPROVAL,
    "recurrence": SCOPE_RECURRENCE,
    "temporal": SCOPE_TEMPORAL,
}


def scope_kind_for(check: str) -> str:
    if check.startswith("contracts:"):
        return SCOPE_CONTRACT
    if check.startswith("input:") or check.startswith("dimension:"):
        return SCOPE_INPUT
    if check.startswith("reconciliation:"):
        return SCOPE_RECONCILIATION
    if check.startswith("hierarchy"):
        return SCOPE_HIERARCHY
    if check.startswith("lineage"):
        return SCOPE_LINEAGE
    if check.startswith("historical_event"):
        return SCOPE_APPROVAL
    return _CHECK_SCOPE_KINDS.get(check, check.split(":", 1)[0] or "unknown")


@dataclass(frozen=True)
class Finding:
    finding_id: str
    check: str
    scope: str
    outcome: str
    required: bool
    evidence_refs: tuple[str, ...]
    approval_ids: tuple[str, ...] = ()
    disposition: str = ""
    clearance_basis: str = ""
    certificate_id: str = ""
    scope_type: str = ""
    metric: str = ""
    level: str = ""
    period: int | None = None
    impact: float = 0.0
    materiality: float = 0.0
    stable_key: str = ""
    schema_version: int = FINDING_SCHEMA_VERSION


def stable_key_for(
    check: str, scope_type: str, metric: str, level: str, scope: str
) -> str:
    """Serialized, period-independent key shared by recurrence and storage.

    One function derives the key so collection, serialization and matching can
    never disagree about which findings belong to the same scope.
    """
    return "|".join((check, scope_type, metric, level, scope))


def disposition_for(
    outcome: str, required: bool, approvals: tuple[str, ...]
) -> str:
    if outcome == "CONTRACT_FAILURE":
        return HARD_FAILURE
    if outcome == "FAIL":
        return HUMAN_APPROVED if approvals else UNEXPLAINED_ANOMALY
    if outcome in ("UNAVAILABLE", "SKIPPED"):
        return UNAVAILABLE_EVIDENCE if required else INFORMATIONAL
    if outcome == "PASS":
        return PASSING if required else INFORMATIONAL
    return HARD_FAILURE


def finding(
    check: str,
    scope: str,
    outcome: str,
    required: bool = True,
    approvals: tuple[str, ...] = (),
    disposition: str | None = None,
    clearance_basis: str = "",
    certificate_id: str = "",
    scope_type: str | None = None,
    metric: str = "",
    level: str = "",
    period: int | None = None,
    impact: float = 0.0,
    materiality: float = 0.0,
) -> Finding:
    if outcome not in KNOWN_OUTCOMES:
        raise ValueError(f"unknown check outcome {outcome!r}")
    resolved_kind = scope_type or scope_kind_for(check)
    identity = json_dumps(
        [
            FINDING_SCHEMA_VERSION,
            check,
            resolved_kind,
            scope,
            metric,
            level,
            period,
        ],
        separators=(",", ":"),
    )
    resolved = disposition or disposition_for(outcome, required, approvals)
    return Finding(
        hashlib.sha256(identity.encode()).hexdigest()[:24],
        check,
        scope,
        outcome,
        required,
        (check,),
        approvals,
        resolved,
        clearance_basis,
        certificate_id,
        resolved_kind,
        metric,
        level,
        period,
        float(impact),
        float(materiality),
        stable_key_for(check, resolved_kind, metric, level, scope),
    )


def _explained(item: Finding) -> bool:
    return bool(item.approval_ids) or item.clearance_basis == "statistical"


def status_for(findings: list[Finding]) -> str:
    if any(item.outcome == "CONTRACT_FAILURE" for item in findings):
        return "DATA_CONTRACT_FAILURE"
    if any(item.outcome == "FAIL" and not _explained(item) for item in findings):
        return "INVESTIGATE"
    if any(item.required and item.outcome == "UNAVAILABLE" for item in findings):
        return "INCOMPLETE"
    if any(item.outcome == "FAIL" and _explained(item) for item in findings):
        return "PASS_WITH_EXPLANATION"
    return "PASS"


def certificate_covers(certificate: Any, item: Finding) -> bool:
    """Exact certificate binding.

    The certificate must be verified, statistical, carry nonempty assessment,
    evidence and qualification identities, cover the exact finding ID in both
    its coverage record and its explicit finding list, report no rejection
    reasons and match the finding's scope kind, metric, level, series and
    period exactly. Missing fields are never treated as current.
    """
    if getattr(certificate, "status", None) != "VERIFIED":
        return False
    if getattr(certificate, "clearance_basis", "") != "statistical":
        return False
    if not getattr(certificate, "assessment_id", ""):
        return False
    if not getattr(certificate, "evidence_digest", ""):
        return False
    if not getattr(certificate, "qualification_digest", ""):
        return False
    finding_ids = getattr(certificate, "finding_ids", ())
    if not finding_ids or item.finding_id not in finding_ids:
        return False
    coverage = getattr(certificate, "coverage", None) or {}
    covered = coverage.get("finding_ids")
    if not covered or item.finding_id not in covered:
        return False
    if getattr(certificate, "reasons", ()):
        return False
    if getattr(certificate, "scope", "") != item.scope:
        return False
    if getattr(certificate, "scope_type", "") != item.scope_type:
        return False
    if getattr(certificate, "metric", "") != item.metric:
        return False
    if getattr(certificate, "level", "") != item.level:
        return False
    if getattr(certificate, "period", None) != item.period:
        return False
    return True


def _clearance_for_finding(
    item: Finding, certificates: tuple[Any, ...]
) -> Any | None:
    if item.outcome != "FAIL" or item.approval_ids or item.clearance_basis:
        return None
    if item.check not in CLEARABLE_CHECKS:
        return None
    if item.scope_type != SCOPE_TEMPORAL:
        return None
    for certificate in certificates:
        if certificate_covers(certificate, item):
            return certificate
    return None


def _temporal_materiality(series: Any, config: Any) -> float:
    relative = float(getattr(config, "temporal_min_relative_residual", 0.02))
    absolute = float(getattr(config, "temporal_materiality_abs", 0.0))
    expected = abs(float(getattr(series, "forecast_median", 0.0) or 0.0))
    return max(absolute, relative * expected)


def collect_findings(
    result: Any,
    config: Any | None = None,
    prior_refreshes: tuple[dict[str, Any], ...] = (),
) -> list[Finding]:
    """Run checks into immutable findings; never grants clearance here."""
    findings = [
        finding(
            f"contracts:{c.name}",
            result.dataset,
            "PASS" if c.passed else "CONTRACT_FAILURE",
        )
        for c in result.contracts.checks
    ]
    findings.extend(finding(**item) for item in result.input_findings)
    historical = (result.machine.get("historical_revision") or {}).get("status")
    if historical in ("INVESTIGATE", "PASS_WITH_EXPLANATION"):
        approvals = (
            tuple(result.attribution.matched_event_ids)
            if (historical == "PASS_WITH_EXPLANATION" and result.attribution)
            else ()
        )
        findings.append(
            finding("historical_revision", result.dataset, "FAIL", approvals=approvals)
        )
    if result.reconciliation is not None:
        for check in result.reconciliation.checks:
            findings.append(
                finding(
                    f"reconciliation:{check.name}",
                    result.dataset,
                    {"PASS": "PASS", "FAIL": "FAIL", "SKIPPED": "UNAVAILABLE"}[
                        check.status
                    ],
                )
            )
    if result.counterfactual is not None and result.counterfactual.evaluated:
        threshold = result.machine.get("materiality_threshold", 0.0)
        findings.append(
            finding(
                "counterfactual_residual",
                result.dataset,
                "FAIL"
                if result.counterfactual.absolute_residual > threshold
                else "PASS",
                impact=abs(float(result.counterfactual.absolute_residual)),
                materiality=float(threshold),
            )
        )
    for name, cube in result.cubes.items():
        measure_columns = [
            column for column in cube if column.endswith(("_previous", "_current"))
        ]
        if cube[measure_columns].isna().any().any():
            findings.append(finding("revision_missing_evidence", name, "UNAVAILABLE"))
    if result.reference is not None:
        findings.append(
            finding(
                "reference",
                result.reference["reference_id"],
                {"MATCH": "PASS", "MISMATCH": "FAIL", "INCOMPLETE": "UNAVAILABLE"}[
                    result.reference["status"]
                ],
            )
        )
    for check in getattr(result, "hierarchy", ()) or ():
        outcome = {"PASS": "PASS", "FAIL": "FAIL", "UNSUPPORTED": "UNAVAILABLE"}.get(
            str(check.status)
        )
        if outcome is None:
            raise ValueError(f"unknown hierarchy status {check.status!r}")
        findings.append(finding(check.name, check.level, outcome))
    drift = getattr(result, "distribution_drift", None)
    if drift is not None:
        # Opt-in check: when it ran, each drifted scope escalates; a scope with
        # too little history to estimate its own threshold is informational.
        for scope in drift.scopes:
            findings.append(
                finding(
                    "distribution_drift",
                    scope.scope,
                    "FAIL" if scope.drifted else ("PASS" if scope.psi is not None else "UNAVAILABLE"),
                    required=False,
                    metric=scope.metric,
                    level=scope.scope_type,
                    period=int(scope.period),
                    impact=float(scope.psi or 0.0),
                    materiality=float(scope.threshold or 0.0),
                )
            )
    lineage = getattr(result, "lineage", None)
    if lineage is not None:
        # A first divergence is expected refresh evidence (attribution explains
        # it), so it is recorded as passing rather than blocking clearance.
        outcome = {
            "PASS": "PASS",
            "FIRST_DIVERGENCE": "PASS",
            "FAIL": "FAIL",
            "SKIPPED": "UNAVAILABLE",
            "UNKNOWN": "UNAVAILABLE",
        }.get(str(getattr(lineage, "status", "PASS")))
        if outcome is None:
            raise ValueError(
                f"unknown lineage status {lineage.status!r}"
            )
        findings.append(
            finding("lineage", result.dataset, outcome, required=False)
        )
    if result.reconciliation is not None:
        from .expectations import apply_expectations

        flags = [
            {
                "metric": "dollar_per_unit",
                "week": f["week"],
                "ratio": f["dollar_per_unit"],
            }
            for f in result.reconciliation.ratio_flags
        ]
        audit = apply_expectations(
            flags,
            result.expectations,
            {"dataset": result.dataset, "as_of": result.observed_at},
        )
        result.machine["expectations"] = audit
        coverage = []
        for item in result.attribution.approval_coverage if result.attribution else []:
            event_finding = finding(
                item["check"], item["scope"], "FAIL", approvals=(item["approval_id"],)
            )
            findings.append(event_finding)
            coverage.append(
                {
                    "finding_id": event_finding.finding_id,
                    "approval_ids": event_finding.approval_ids,
                    "weeks": item["weeks"],
                }
            )
        for flag in audit["expected"] + audit["unexpected"]:
            approvals = (flag["explained_by"],) if "explained_by" in flag else ()
            item = finding(
                "ratio:dollar_per_unit",
                str(flag["week"]),
                "FAIL",
                approvals=approvals,
                period=int(flag["week"]),
            )
            findings.append(item)
            coverage.append({"finding_id": item.finding_id, "approval_ids": approvals})
        result.machine["approval_coverage"] = coverage
    configured_metrics: tuple[str, ...] = ()
    required_metrics: tuple[str, ...] = ()
    if config is not None and hasattr(config, "temporal_metrics"):
        configured_metrics = tuple(config.temporal_metrics())
        required_metrics = tuple(config.required_temporal_metrics())
    if result.temporal_required and (
        result.temporal is None or not result.temporal.series
    ):
        for metric in configured_metrics or ("",):
            findings.append(
                finding(
                    "temporal",
                    result.dataset,
                    "UNAVAILABLE",
                    metric=metric,
                    required=(not configured_metrics)
                    or metric in required_metrics,
                )
            )
    elif result.temporal is not None:
        for metric, state in (
            getattr(result.temporal, "metric_status", {}) or {}
        ).items():
            if state != "ABSENT":
                continue
            findings.append(
                finding(
                    f"input:metric:{metric}",
                    result.dataset,
                    "UNAVAILABLE",
                    metric=metric,
                    required=metric in required_metrics,
                )
            )
    if result.temporal is not None:
        period_unavailable = list(getattr(result.temporal, "unavailable", ()) or ())
        if period_unavailable:
            for entry in period_unavailable:
                findings.append(
                    finding(
                        "temporal",
                        str(entry["series_id"]),
                        "UNAVAILABLE",
                        required=result.temporal_required,
                        metric=str(entry.get("metric", "")),
                        level=_series_level(str(entry["series_id"])),
                        period=entry.get("target_week"),
                    )
                )
        else:
            for series_id in result.temporal.unavailable_series:
                findings.append(
                    finding(
                        "temporal",
                        series_id,
                        "UNAVAILABLE",
                        required=result.temporal_required,
                        level=_series_level(str(series_id)),
                    )
                )
        for series in result.temporal.series:
            findings.append(
                finding(
                    "temporal",
                    series.series_id,
                    "FAIL" if series.anomaly else "PASS",
                    metric=series.metric,
                    level=series.level,
                    period=int(series.target_week),
                    # A share-tested leaf's impact is the movement attributable
                    # to its share change, not the parent-driven residual.
                    impact=float(
                        series.share_impact
                        if getattr(series, "share_impact", None) is not None
                        else series.residual
                    ),
                    materiality=_temporal_materiality(series, config),
                )
            )

    recurrence: list[dict[str, Any]] = []
    if config is not None and prior_refreshes:
        from .recurrence import assess_recurrence

        for assessment in assess_recurrence(
            [asdict(item) for item in findings],
            list(prior_refreshes),
            config,
        ):
            recurrence.append(assessment.to_dict())
            if not assessment.material:
                continue
            findings.append(
                finding(
                    "recurrence",
                    f"{assessment.check}:{assessment.scope}",
                    "FAIL",
                    metric=assessment.metric,
                    level=assessment.level,
                    impact=float(assessment.cumulative_impact),
                    materiality=float(assessment.cumulative_threshold),
                )
            )
        result.machine["recurrence"] = recurrence
    return findings


def _series_level(series_id: str) -> str:
    return "national" if ":" not in series_id else series_id.split(":", 1)[0]


def apply_policy(
    result: Any,
    findings: list[Finding],
    certificates: tuple[Any, ...] = (),
    optional_checks: tuple[str, ...] = (),
) -> None:
    """Resolve certificate clearances once and compute the final status.

    Only certificates bound to this assessment identity, one of its evidence
    digests, the supported certificate schema and the current policy version
    can clear anything; any other artifact is ignored.
    """
    from .evidence_package import (
        CERTIFICATE_SCHEMA,
        EVIDENCE_POLICY_VERSION,
    )

    expected_id = getattr(result, "assessment_id", None) or result.run_id
    digests = set(getattr(result, "machine", {}).get("evidence_digests", ()) or ())
    compatible = tuple(
        certificate
        for certificate in certificates
        if expected_id
        and getattr(certificate, "assessment_id", "") == expected_id
        and getattr(certificate, "evidence_digest", "") in digests
        and getattr(certificate, "schema_version", 0) == CERTIFICATE_SCHEMA
        and getattr(certificate, "policy_version", "") == EVIDENCE_POLICY_VERSION
        and getattr(certificate, "qualification_digest", "")
    )
    findings = [
        replace(f, required=False)
        if f.check in optional_checks and not f.check.startswith("contracts:")
        else f
        for f in findings
    ]
    cleared: list[dict[str, Any]] = []
    resolved: list[Finding] = []
    for item in findings:
        certificate = _clearance_for_finding(item, compatible)
        if certificate is None:
            resolved.append(item)
            continue
        item = replace(
            item,
            disposition=STATISTICALLY_EXPLAINED,
            clearance_basis="statistical",
            certificate_id=certificate.certificate_id,
        )
        resolved.append(item)
        cleared.append(
            {
                "finding_id": item.finding_id,
                "scope": item.scope,
                "period": item.period,
                "certificate_id": certificate.certificate_id,
                "basis": "statistical",
                "reasons": list(certificate.reasons),
            }
        )
    findings = resolved
    result.status = status_for(findings)
    clearance_bases = {
        f.finding_id: (
            f.clearance_basis or ("human_approval" if f.approval_ids else "")
        )
        for f in findings
        if f.outcome == "FAIL"
    }
    result.machine.update(
        schema_version=MACHINE_SCHEMA_VERSION,
        findings=[asdict(f) for f in findings],
        status=result.status,
        requires_investigation=result.status not in ("PASS", "PASS_WITH_EXPLANATION"),
        clearance=cleared,
        clearance_bases=clearance_bases,
    )


def finalize_policy(
    result: Any,
    optional_checks: tuple[str, ...] = (),
    certificates: tuple[Any, ...] = (),
    config: Any | None = None,
    prior_refreshes: tuple[dict[str, Any], ...] = (),
    findings: list[Finding] | None = None,
) -> None:
    """Never let an integrity failure disappear behind attribution or a model."""
    collected = (
        findings
        if findings is not None
        else collect_findings(result, config, prior_refreshes)
    )
    apply_policy(result, collected, certificates, optional_checks)
