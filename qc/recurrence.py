"""Cross-refresh recurrence escalation.

A finding recurring in at least ``recurrence_minimum`` of the last
``recurrence_window`` refreshes escalates to review only when the cumulative
unexplained impact across those refreshes is material relative to their frozen
materiality thresholds. Single small fluctuations therefore stay informational.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .config import DatasetConfig

RECURRENCE_SCHEMA = 1
ESCALATE_DISPOSITIONS = ("", "UNEXPLAINED_ANOMALY")


@dataclass(frozen=True)
class RecurrenceAssessment:
    finding_id: str
    check: str
    scope: str
    occurrences: int
    window: int
    cumulative_impact: float
    cumulative_threshold: float
    repeated: bool
    material: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RECURRENCE_SCHEMA,
            "finding_id": self.finding_id,
            "check": self.check,
            "scope": self.scope,
            "occurrences": self.occurrences,
            "window": self.window,
            "cumulative_impact": self.cumulative_impact,
            "cumulative_threshold": self.cumulative_threshold,
            "repeated": self.repeated,
            "material": self.material,
        }


def _prior_unexplained(refresh: dict[str, Any]) -> float:
    ledger = refresh.get("ledger") or {}
    if ledger:
        return abs(float(ledger.get("net_unexplained", 0.0)))
    return abs(float(refresh.get("unexplained_delta", 0.0)))


def assess_recurrence(
    current_findings: Sequence[dict[str, Any]],
    prior_refreshes: Sequence[dict[str, Any]],
    config: DatasetConfig,
    *,
    current_impact: float = 0.0,
    current_threshold: float = 0.0,
) -> list[RecurrenceAssessment]:
    """Assess repeated unexplained findings over the recent refresh window."""
    if not config.recurrence_enabled:
        return []
    window = config.recurrence_window
    minimum = config.recurrence_minimum
    prior = list(prior_refreshes)[: max(0, window - 1)]
    assessments: list[RecurrenceAssessment] = []
    for finding in current_findings:
        if finding.get("outcome") != "FAIL":
            continue
        if finding.get("disposition") not in ESCALATE_DISPOSITIONS:
            continue
        if finding.get("approval_ids") or finding.get("clearance_basis") == "statistical":
            continue
        matches = [
            refresh
            for refresh in prior
            if any(
                item.get("finding_id") == finding.get("finding_id")
                for item in refresh.get("findings", [])
            )
        ]
        occurrences = 1 + len(matches)
        repeated = occurrences >= minimum
        if not repeated:
            continue
        cumulative_impact = abs(current_impact) + sum(
            _prior_unexplained(refresh) for refresh in matches
        )
        cumulative_threshold = current_threshold + sum(
            abs(float(refresh.get("materiality_threshold", 0.0)))
            for refresh in matches
        )
        material = cumulative_impact >= cumulative_threshold
        assessments.append(
            RecurrenceAssessment(
                finding_id=str(finding.get("finding_id", "")),
                check=str(finding.get("check", "")),
                scope=str(finding.get("scope", "")),
                occurrences=occurrences,
                window=window,
                cumulative_impact=cumulative_impact,
                cumulative_threshold=cumulative_threshold,
                repeated=repeated,
                material=material,
            )
        )
    return assessments
