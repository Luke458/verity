"""Canonical evidence text for encoder-based providers.

A deterministic, bounded, human-readable serialization of a run's evidence.
It deliberately excludes decision outputs and oracle labels so a text probe
cannot learn the answer from its own input. The text version is part of every
artifact and label record that uses it.
"""

from __future__ import annotations

from typing import Any

EVIDENCE_TEXT_VERSION = 1


def _number(value: float | None, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def evidence_text(result: Any, max_chars: int = 8000) -> str:
    recorded = getattr(result, "recorded_text", None)
    if isinstance(recorded, str):
        return recorded[:max_chars]
    lines: list[str] = []
    lines.append(f"run_id={getattr(result, 'run_id', '')}")
    lines.append(f"dataset={getattr(result, 'dataset', '')}")
    lines.append(f"status={getattr(result, 'status', '')}")

    pair = getattr(result, "version_pair", None)
    if pair is not None:
        lines.append(
            f"versions {pair.previous_id}->{pair.current_id} "
            f"shape={pair.shape} overlap={pair.overlap_start}..{pair.overlap_end} "
            f"new_periods={list(pair.new_periods)}"
        )

    contracts = getattr(result, "contracts", None)
    if contracts is not None:
        failed = ",".join(check.name for check in contracts.failed) or "none"
        lines.append(f"contracts status={contracts.status} failed={failed}")

    for event in getattr(result, "events", ()) or ():
        lines.append(
            f"event {event.entity_type}:{event.entity_id} "
            f"class={event.classification} "
            f"weeks_added={list(event.historical_weeks_added)} "
            f"weeks_removed={list(event.historical_weeks_removed)}"
        )

    attribution = getattr(result, "attribution", None)
    if attribution is not None:
        lines.append(
            "revision "
            f"raw={_number(attribution.raw_delta)} "
            f"explained={_number(attribution.explained_delta)} "
            f"unexplained={_number(attribution.unexplained_delta)} "
            f"explained_fraction={attribution.explained_fraction:.4f} "
            f"breadth={attribution.breadth:.4f} "
            f"material={attribution.material}"
        )
        for contributor in attribution.contributors[:10]:
            lines.append(
                f"contributor {contributor.entity_type}:{contributor.entity_id} "
                f"delta={_number(contributor.delta)}"
            )

    counterfactual = getattr(result, "counterfactual", None)
    if counterfactual is not None:
        lines.append(
            f"counterfactual reconstructed={_number(counterfactual.reconstructed_delta)} "
            "score="
            + (
                f"{counterfactual.reconciliation_score:.4f}"
                if counterfactual.reconciliation_score is not None
                else "n/a"
            )
        )

    reconciliation = getattr(result, "reconciliation", None)
    if reconciliation is not None:
        failed = ",".join(check.name for check in reconciliation.failed) or "none"
        lines.append(
            f"reconciliation status={reconciliation.status} failed={failed} "
            f"ratio_flags={len(reconciliation.ratio_flags)}"
        )

    lineage = getattr(result, "lineage", None)
    if lineage is not None:
        lines.append(
            f"lineage first_divergence={lineage.first_divergence} "
            f"status={lineage.status}"
        )
        for divergence in lineage.divergences:
            lines.append(
                f"lineage_stage {divergence['stage']} "
                f"diverged={divergence['diverged']} "
                f"relative={_number(divergence['relative_divergence'])} "
                f"row_delta={divergence['row_count_delta']}"
            )

    temporal = getattr(result, "temporal", None)
    if temporal is not None:
        lines.append(
            f"temporal week={temporal.target_week} anomaly={temporal.anomaly} "
            f"flags={temporal.flags}"
        )
        for item in temporal.series:
            if not item.anomaly and item.series_id != "national":
                continue
            lines.append(
                f"series {item.series_id} actual={_number(item.actual, 2)} "
                f"adjusted={_number(item.adjusted_actual, 2)} "
                f"forecast={_number(item.forecast_median, 2)} "
                f"residual={_number(item.relative_residual, 4)} "
                f"percentile="
                f"{item.calibrated_percentile if item.calibrated_percentile is not None else 'n/a'} "
                f"z={_number(item.standardized_residual, 3)} flags={item.flags}"
            )

    for relationship in getattr(result, "relationships", ()) or ():
        lines.append(
            f"relationship {relationship.source_id}->{relationship.target_id} "
            f"type={relationship.relationship} "
            f"confidence={_number(relationship.confidence, 3)}"
        )

    for reason in getattr(result, "reasons", ()) or ():
        lines.append(f"reason {reason}")

    text = "\n".join(lines)
    if len(text) <= max_chars:
        return text
    trimmed = lines[:]
    while trimmed and len("\n".join(trimmed)) > max_chars:
        trimmed.pop()
    return "\n".join(trimmed)
