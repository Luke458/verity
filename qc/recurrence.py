"""Cross-refresh recurrence escalation.

A finding recurring in at least ``recurrence_minimum`` of the last
``recurrence_window`` distinct logical refreshes escalates to review only when
the signed cumulative unexplained impact of the matching findings is material
relative to a registered cumulative budget.

Findings are matched on a serialized, period-independent stable key
(check, scope kind, metric, hierarchy level and scope) produced by
``qc.policy.stable_key_for``; the key is stored explicitly with every finding,
so matching survives serialization, storage and replay. Only eligible
unexplained predecessors (failed, not approved, not statistically cleared)
participate; multiple periods inside one logical refresh count as one
occurrence and their impacts are combined. A legitimate zero scoped impact is
never replaced by dataset-wide movement.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .config import DatasetConfig

RECURRENCE_SCHEMA = 3
RECURRENCE_INPUT_SCHEMA = 2
ESCALATE_DISPOSITIONS = ("", "UNEXPLAINED_ANOMALY", "INFORMATIONAL")


@dataclass(frozen=True)
class RecurrenceAssessment:
    finding_id: str
    check: str
    scope: str
    metric: str
    level: str
    occurrences: int
    window: int
    cumulative_impact: float
    cumulative_gross_impact: float
    cumulative_threshold: float
    repeated: bool
    material: bool
    stable_key: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RECURRENCE_SCHEMA,
            "finding_id": self.finding_id,
            "stable_key": self.stable_key,
            "check": self.check,
            "scope": self.scope,
            "metric": self.metric,
            "level": self.level,
            "occurrences": self.occurrences,
            "window": self.window,
            "cumulative_impact": self.cumulative_impact,
            "cumulative_gross_impact": self.cumulative_gross_impact,
            "cumulative_threshold": self.cumulative_threshold,
            "repeated": self.repeated,
            "material": self.material,
        }


def finding_key(item: dict[str, Any]) -> str:
    """Stable match key shared with ``qc.policy``; never the period-specific ID."""
    recorded = str(item.get("stable_key") or "")
    if recorded:
        return recorded
    from .policy import stable_key_for

    return stable_key_for(
        str(item.get("check", "")),
        str(item.get("scope_type", "")),
        str(item.get("metric", "")),
        str(item.get("level", "")),
        str(item.get("scope", "")),
    )


def eligible_unexplained(item: dict[str, Any]) -> bool:
    """Only failed, unapproved, uncleared findings may recur."""
    if str(item.get("outcome", "")) != "FAIL":
        return False
    if item.get("approval_ids"):
        return False
    if item.get("clearance_basis") == "statistical":
        return False
    return str(item.get("disposition", "")) in ESCALATE_DISPOSITIONS


def _finding_impact(item: dict[str, Any]) -> float:
    return float(item.get("impact", 0.0))


def _finding_materiality(item: dict[str, Any]) -> float:
    return abs(float(item.get("materiality", 0.0)))


def _current_groups(
    current_findings: Sequence[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for finding in current_findings:
        if not eligible_unexplained(finding):
            continue
        groups.setdefault(finding_key(finding), []).append(finding)
    return groups


def assess_recurrence(
    current_findings: Sequence[dict[str, Any]],
    prior_refreshes: Sequence[dict[str, Any]],
    config: DatasetConfig,
) -> list[RecurrenceAssessment]:
    """Assess repeated unexplained findings over the recent refresh window."""
    if not config.recurrence_enabled:
        return []
    window = config.recurrence_window
    minimum = config.recurrence_minimum
    budget_ratio = float(getattr(config, "recurrence_budget_ratio", 1.0))
    prior = list(prior_refreshes)[: max(0, window - 1)]
    assessments: list[RecurrenceAssessment] = []
    for key, current_group in _current_groups(current_findings).items():
        matches: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
        for refresh in prior:
            repeated_findings = [
                item
                for item in refresh.get("findings", [])
                if eligible_unexplained(item) and finding_key(item) == key
            ]
            if repeated_findings:
                matches.append((refresh, repeated_findings))
        occurrences = 1 + len(matches)
        if occurrences < minimum:
            continue
        impact = sum(_finding_impact(item) for item in current_group)
        gross_impact = sum(
            abs(_finding_impact(item)) for item in current_group
        )
        materialities = [_finding_materiality(item) for item in current_group]
        for _, repeated_findings in matches:
            impact += sum(_finding_impact(item) for item in repeated_findings)
            gross_impact += sum(
                abs(_finding_impact(item)) for item in repeated_findings
            )
            materialities.extend(
                _finding_materiality(item) for item in repeated_findings
            )
        scoped = max(materialities, default=0.0)
        cumulative_threshold = budget_ratio * scoped
        first = current_group[0]
        assessments.append(
            RecurrenceAssessment(
                finding_id=str(first.get("finding_id", "")),
                check=str(first.get("check", "")),
                scope=str(first.get("scope", "")),
                metric=str(first.get("metric", "")),
                level=str(first.get("level", "")),
                occurrences=occurrences,
                window=window,
                cumulative_impact=impact,
                cumulative_gross_impact=gross_impact,
                cumulative_threshold=cumulative_threshold,
                repeated=True,
                # A registered budget of zero never escalates: a legitimate
                # zero scoped impact must not fall back to dataset movement.
                # Gross persistence is preserved so alternating deviations
                # that net to zero still escalate.
                material=cumulative_threshold > 0.0
                and max(abs(impact), gross_impact) >= cumulative_threshold,
                stable_key=key,
            )
        )
    return assessments
