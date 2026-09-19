"""Human-readable reports (architecture section 78).

Markdown is the human view; the machine JSON written beside it remains the
contract for downstream code and scheduled agents. Report directories are
created exclusively, so an existing run is never overwritten.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .agent import InvestigationBrief


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def _number(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:,.{digits}f}"


def render_markdown(
    result: Any,
    brief: InvestigationBrief | None = None,
) -> str:
    pair = result.version_pair
    lines: list[str] = []
    lines.append(f"# QC RUN {result.run_id}")
    lines.append("")
    lines.append(f"**STATUS {result.status}**")
    lines.append("")

    lines.append("## Versions")
    if pair is not None:
        lines.append(
            f"- {pair.previous_id} -> {pair.current_id} ({pair.shape}); "
            f"overlap {pair.overlap_start}..{pair.overlap_end}; "
            f"new periods {list(pair.new_periods)}"
        )
    else:
        lines.append("- unavailable (contract failure)")
    lines.append("")

    attribution = result.attribution
    if attribution is not None:
        lines.append("## Historical revision")
        lines.append(
            _table(
                ["raw", "explained", "unexplained", "explained fraction", "breadth", "material"],
                [
                    [
                        _number(attribution.raw_delta),
                        _number(attribution.explained_delta),
                        _number(attribution.unexplained_delta),
                        f"{attribution.explained_fraction:.4f}",
                        f"{attribution.breadth:.4f}",
                        attribution.material,
                    ]
                ],
            )
        )
        if attribution.contributors:
            lines.append("")
            lines.append("### Top contributors")
            lines.append(
                _table(
                    ["entity", "id", "delta"],
                    [
                        [
                            contributor.entity_type,
                            contributor.entity_id,
                            _number(contributor.delta),
                        ]
                        for contributor in attribution.contributors[:10]
                    ],
                )
            )
    lines.append("")

    if result.counterfactual is not None:
        lines.append("## Counterfactual")
        lines.append(
            f"- reconstructed delta {_number(result.counterfactual.reconstructed_delta)}; "
            "reconciliation score "
            + (
                f"{result.counterfactual.reconciliation_score:.4f}"
                if result.counterfactual.reconciliation_score is not None
                else "not evaluated"
            )
        )
        lines.append("")
    if result.reconciliation is not None:
        lines.append(f"## Reconciliation: {result.reconciliation.status}")
        for check in result.reconciliation.checks:
            if not check.passed:
                lines.append(f"- FAILED {check.name}: {check.detail}")
        if result.reconciliation.ratio_flags:
            lines.append(
                f"- ratio outliers on {len(result.reconciliation.ratio_flags)} week(s)"
            )
        lines.append("")
    if result.lineage is not None:
        lines.append("## Lineage")
        lines.append(
            f"- first divergence: {result.lineage.first_divergence or 'none'}"
        )
        lines.append("")

    if result.events:
        lines.append("## Lifecycle events")
        lines.append(
            _table(
                ["entity", "classification", "weeks added", "weeks removed"],
                [
                    [
                        f"{event.entity_type}:{event.entity_id}",
                        event.classification,
                        list(event.historical_weeks_added),
                        list(event.historical_weeks_removed),
                    ]
                    for event in result.events
                ],
            )
        )
        lines.append("")

    if result.temporal is not None:
        lines.append(f"## Latest week {result.temporal.target_week}")
        lines.append(
            f"- anomaly: {result.temporal.anomaly}; flags: {result.temporal.flags or 'none'}"
        )
        anomalous = [item for item in result.temporal.series if item.anomaly]
        if anomalous:
            lines.append(
                _table(
                    ["series", "actual", "adjusted", "forecast", "residual", "percentile", "flags"],
                    [
                        [
                            item.series_id,
                            _number(item.actual, 0),
                            _number(item.adjusted_actual, 0),
                            _number(item.forecast_median, 0),
                            _number(item.relative_residual, 4),
                            (
                                f"{item.calibrated_percentile:.3f}"
                                if item.calibrated_percentile is not None
                                else "n/a"
                            ),
                            ", ".join(item.flags),
                        ]
                        for item in anomalous
                    ],
                )
            )
        coverage = result.temporal.calibration.get("coverage", {})
        if coverage:
            lines.append("")
            lines.append("- calibration coverage: " + ", ".join(
                f"q{level}={value:.2f}" for level, value in sorted(coverage.items())
            ))
        lines.append("")

    if result.relationships:
        lines.append("## Entity relationships (candidates)")
        lines.append(
            _table(
                ["source", "target", "type", "confidence"],
                [
                    [
                        relationship.source_id,
                        relationship.target_id,
                        relationship.relationship,
                        (
                            f"{relationship.confidence:.3f}"
                            if relationship.confidence is not None
                            else "n/a"
                        ),
                    ]
                    for relationship in result.relationships
                ],
            )
        )
        lines.append("")

    decisions = getattr(result, "decisions", None)
    if decisions is not None:
        lines.append(f"## Decision ({decisions.provider})")
        lines.append(
            _table(
                ["field", "value", "probability", "index"],
                [
                    [
                        name,
                        decision.value,
                        (
                            f"{max(decision.probabilities.values()):.3f}"
                            if decision.probabilities
                            else "n/a"
                        ),
                        f"{decision.index:.2f}" if decision.index is not None else "",
                    ]
                    for name, decision in decisions.values.items()
                ],
            )
        )
        lines.append("")

    if result.reasons:
        lines.append("## Reasons")
        for reason in result.reasons:
            lines.append(f"- {reason}")
        lines.append("")

    if brief is not None:
        lines.append("## Investigation brief")
        if brief.open_questions:
            lines.append("### Open questions")
            for question in brief.open_questions:
                lines.append(f"- {question}")
        if brief.recommended_queries:
            lines.append("")
            lines.append("### Recommended first queries")
            lines.append("```sql")
            lines.extend(brief.recommended_queries)
            lines.append("```")
        if brief.similar_incidents:
            lines.append("")
            lines.append("### Similar incidents")
            lines.append(
                _table(
                    ["incident", "root cause", "resolution", "similarity"],
                    [
                        [
                            incident["incident_id"],
                            incident["root_cause"],
                            incident["resolution"],
                            f"{incident['similarity']:.3f}",
                        ]
                        for incident in brief.similar_incidents
                    ],
                )
            )
        lines.append("")

    if result.evidence is not None:
        node_types: dict[str, int] = {}
        for node in result.evidence.nodes:
            node_types[node.type] = node_types.get(node.type, 0) + 1
        lines.append("## Evidence graph")
        lines.append(
            f"- nodes: {len(result.evidence.nodes)}; edges: {len(result.evidence.edges)}"
        )
        for node_type, count in sorted(node_types.items()):
            lines.append(f"- {node_type}: {count}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_report(
    result: Any,
    directory: str | Path,
    brief: InvestigationBrief | None = None,
) -> dict[str, Path]:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=False)
    markdown_path = path / "report.md"
    machine_path = path / "report.json"
    markdown_path.write_text(render_markdown(result, brief))
    machine_path.write_text(
        json.dumps(result.machine, indent=2, sort_keys=True, default=str)
    )
    return {"markdown": markdown_path, "machine": machine_path}
