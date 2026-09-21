"""Versioned deterministic findings and the effective analyst review policy.

Findings carry an explicit disposition so the review decision can distinguish a
hard failure, missing required evidence, an unexplained actionable anomaly, a
statistically explained movement, a human-approved exception and an
informational finding below escalation thresholds. Statistical clearance is
granted only through a verified explanation certificate; no model can override
the verifier.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, replace
from typing import Any

from .jsonutil import dumps as json_dumps

FINDING_SCHEMA_VERSION = 2

HARD_FAILURE = "HARD_FAILURE"
UNAVAILABLE_EVIDENCE = "UNAVAILABLE_EVIDENCE"
UNEXPLAINED_ANOMALY = "UNEXPLAINED_ANOMALY"
STATISTICALLY_EXPLAINED = "STATISTICALLY_EXPLAINED"
HUMAN_APPROVED = "HUMAN_APPROVED"
INFORMATIONAL = "INFORMATIONAL"
PASSING = "PASS"


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
    schema_version: int = FINDING_SCHEMA_VERSION


def disposition_for(
    outcome: str, required: bool, approvals: tuple[str, ...]
) -> str:
    if outcome == "CONTRACT_FAILURE":
        return HARD_FAILURE
    if outcome == "FAIL":
        return HUMAN_APPROVED if approvals else UNEXPLAINED_ANOMALY
    if outcome == "UNAVAILABLE":
        return UNAVAILABLE_EVIDENCE if required else INFORMATIONAL
    if outcome == "PASS":
        return PASSING if required else INFORMATIONAL
    return INFORMATIONAL


def finding(
    check: str,
    scope: str,
    outcome: str,
    required: bool = True,
    approvals: tuple[str, ...] = (),
    disposition: str | None = None,
    clearance_basis: str = "",
    certificate_id: str = "",
) -> Finding:
    identity = json_dumps(
        [FINDING_SCHEMA_VERSION, check, scope], separators=(",", ":")
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


def _clearance_for_finding(
    item: Finding, certificates: tuple[Any, ...]
) -> Any | None:
    if item.outcome != "FAIL" or item.approval_ids:
        return None
    for certificate in certificates:
        if getattr(certificate, "status", None) != "VERIFIED":
            continue
        if certificate.scope == item.scope:
            return certificate
        if item.check == "historical_revision" and certificate.scope == "dataset":
            return certificate
    return None


def finalize_policy(
    result: Any,
    optional_checks: tuple[str, ...] = (),
    certificates: tuple[Any, ...] = (),
    config: Any | None = None,
    prior_refreshes: tuple[dict[str, Any], ...] = (),
) -> None:
    """Never let an integrity failure disappear behind attribution or a model."""
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
        findings.append(finding("counterfactual_residual", result.dataset,
                                "FAIL" if result.counterfactual.absolute_residual > threshold else "PASS"))
    for name, cube in result.cubes.items():
        measure_columns = [column for column in cube if column.endswith(("_previous", "_current"))]
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
                "ratio:dollar_per_unit", str(flag["week"]), "FAIL", approvals=approvals
            )
            findings.append(item)
            coverage.append({"finding_id": item.finding_id, "approval_ids": approvals})
        result.machine["approval_coverage"] = coverage
    if result.temporal_required and (
        result.temporal is None or not result.temporal.series
    ):
        findings.append(finding("temporal", result.dataset, "UNAVAILABLE"))
    if result.temporal is not None:
        for series_id in result.temporal.unavailable_series:
            findings.append(finding("temporal", series_id, "UNAVAILABLE", required=result.temporal_required))
        for series in result.temporal.series:
            findings.append(
                finding(
                    "temporal", series.series_id, "FAIL" if series.anomaly else "PASS"
                )
            )
        # Coordinated same-direction residuals are combined before materiality;
        # groups already captured by an individual anomaly add no duplicate.
        for group in getattr(result.temporal, "coordinated", ()):
            if group.get("individually_flagged"):
                continue
            findings.append(
                finding(
                    "coordinated_residual",
                    f"{group['level']}:{group['direction']}",
                    "FAIL",
                )
            )

    findings = [
        replace(f, required=False)
        if f.check in optional_checks and not f.check.startswith("contracts:")
        else f
        for f in findings
    ]
    recurrence: list[dict[str, Any]] = []
    if config is not None and prior_refreshes:
        from .recurrence import assess_recurrence

        net_impact = 0.0
        if getattr(result, "ledger", None) is not None:
            net_impact = float(result.ledger.net_unexplained)
        elif result.attribution is not None:
            net_impact = float(result.attribution.unexplained_delta)
        for assessment in assess_recurrence(
            [asdict(item) for item in findings],
            list(prior_refreshes),
            config,
            current_impact=net_impact,
            current_threshold=result.machine.get("materiality_threshold", 0.0),
        ):
            recurrence.append(assessment.to_dict())
            if not assessment.material:
                continue
            findings.append(
                finding(
                    "recurrence",
                    f"{assessment.check}:{assessment.scope}",
                    "FAIL",
                )
            )
        result.machine["recurrence"] = recurrence

    cleared: list[dict[str, Any]] = []
    resolved: list[Finding] = []
    for item in findings:
        certificate = _clearance_for_finding(item, certificates)
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
        schema_version=3,
        findings=[asdict(f) for f in findings],
        status=result.status,
        requires_investigation=result.status not in ("PASS", "PASS_WITH_EXPLANATION"),
        clearance=cleared,
        clearance_bases=clearance_bases,
    )
