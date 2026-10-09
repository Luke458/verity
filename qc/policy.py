"""Versioned findings and the one final review status.

Findings carry an explicit disposition so the review decision can distinguish a
hard failure, missing required evidence, an unexplained actionable anomaly, a
human-approved exception and an informational finding below escalation
thresholds. Only a registered, approved expectation or expected event can
explain a failure; nothing is cleared automatically.

1. ``collect_findings`` runs every check and materializes findings (including
   period-specific temporal findings).
2. ``apply_policy`` computes the final status exactly once from them.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, replace
from typing import Any

from .jsonutil import dumps as json_dumps

FINDING_SCHEMA_VERSION = 5
MACHINE_SCHEMA_VERSION = 7

HARD_FAILURE = "HARD_FAILURE"
UNAVAILABLE_EVIDENCE = "UNAVAILABLE_EVIDENCE"
UNEXPLAINED_ANOMALY = "UNEXPLAINED_ANOMALY"
HUMAN_APPROVED = "HUMAN_APPROVED"
INFORMATIONAL = "INFORMATIONAL"
PASSING = "PASS"

# Outcomes a check may report. Unknown outcomes are rejected when findings are
# materialized and treated as failures if they somehow reach the policy.
KNOWN_OUTCOMES = frozenset(
    {"PASS", "FAIL", "UNAVAILABLE", "CONTRACT_FAILURE", "SKIPPED"}
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
SCOPE_HIERARCHY = "hierarchy"
SCOPE_LINEAGE = "lineage"

_CHECK_SCOPE_KINDS = {
    "historical_revision": SCOPE_HISTORICAL_REVISION,
    "counterfactual_residual": SCOPE_COUNTERFACTUAL,
    "revision_missing_evidence": SCOPE_REVISION,
    "revision_week": SCOPE_REVISION,
    "reference": SCOPE_REFERENCE,
    "ratio:dollar_per_unit": SCOPE_APPROVAL,
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
    if check.startswith(("historical_event", "absence_event")):
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
    """Serialized, period-independent key for a finding's scope.

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
        resolved_kind,
        metric,
        level,
        period,
        float(impact),
        float(materiality),
        stable_key_for(check, resolved_kind, metric, level, scope),
    )


def _explained(item: Finding) -> bool:
    return bool(item.approval_ids)


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


def _temporal_materiality(series: Any, config: Any) -> float:
    relative = float(getattr(config, "temporal_min_relative_residual", 0.02))
    absolute = float(getattr(config, "temporal_materiality_abs", 0.0))
    expected = abs(float(getattr(series, "forecast_median", 0.0) or 0.0))
    return max(absolute, relative * expected)


def collect_findings(
    result: Any,
    config: Any | None = None,
) -> list[Finding]:
    """Run every check into immutable findings."""
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
    # A restatement confined to one overlap week, judged against that week.
    for item in getattr(result, "week_revisions", ()) or ():
        if item.material:
            findings.append(
                finding(
                    "revision_week",
                    str(item.week),
                    "FAIL",
                    metric=getattr(config, "primary_metric", ""),
                    period=int(item.week),
                    impact=abs(float(item.unexplained)),
                    materiality=float(item.tolerance) * abs(float(item.previous)),
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
    lineage = getattr(result, "lineage", None)
    if lineage is not None:
        # A first divergence is expected refresh evidence (attribution explains
        # it), so it is recorded as passing.
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

    return findings


def _series_level(series_id: str) -> str:
    return "national" if ":" not in series_id else series_id.split(":", 1)[0]


def apply_policy(
    result: Any,
    findings: list[Finding],
    optional_checks: tuple[str, ...] = (),
) -> None:
    """Compute the final status once from the findings."""
    findings = [
        replace(f, required=False)
        if f.check in optional_checks and not f.check.startswith("contracts:")
        else f
        for f in findings
    ]
    result.status = status_for(findings)
    result.machine.update(
        schema_version=MACHINE_SCHEMA_VERSION,
        findings=[asdict(f) for f in findings],
        status=result.status,
        requires_investigation=result.status not in ("PASS", "PASS_WITH_EXPLANATION"),
        explained_by={
            f.finding_id: list(f.approval_ids)
            for f in findings
            if f.outcome == "FAIL" and f.approval_ids
        },
    )
