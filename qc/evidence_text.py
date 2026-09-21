"""Canonical evidence text for encoder-based providers (version 2).

A deterministic, bounded, human-readable serialization of the evidence
package. It deliberately excludes the final QC status, finding dispositions,
certificate verdicts, effective review decisions and oracle labels so a text
probe cannot learn the answer from its own input. The text version is part of
every artifact and label record that uses it.
"""

from __future__ import annotations

from typing import Any

EVIDENCE_TEXT_VERSION = 3


def _number(value: float | None, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _package_lines(package: Any) -> list[str]:
    lines: list[str] = []
    lines.append(f"assessment_id={getattr(package, 'assessment_id', '')}")
    lines.append(f"run_id={getattr(package, 'run_id', '')}")
    lines.append(f"dataset={getattr(package, 'dataset', '')}")
    lines.append(f"provenance={getattr(package, 'provenance', '')}")
    cutoff = getattr(package, "observation_cutoff", None)
    lines.append(f"observation_cutoff={cutoff or 'n/a'}")
    calendar = getattr(package, "calendar_identity", {}) or {}
    if calendar:
        lines.append(f"calendar {calendar}")
    for check in getattr(package, "contracts", ()):
        if not isinstance(check, dict):
            continue
        lines.append(
            f"contract {check.get('name')} status={check.get('status')} "
            f"detail={check.get('detail')}"
        )
    ledger = getattr(package, "ledger", None) or {}
    if ledger:
        lines.append(
            "ledger "
            f"overlap_delta={_number(ledger.get('overlap_delta'), 2)} "
            f"new_period_movement={_number(ledger.get('new_period_movement'), 2)} "
            f"net_unexplained={_number(ledger.get('net_unexplained'), 2)} "
            f"gross_unexplained={_number(ledger.get('gross_unexplained'), 2)} "
            f"support={ledger.get('support_level', 'unknown')}"
        )
        for contribution in ledger.get("contributions", [])[:40]:
            lines.append(
                f"contribution {contribution.get('category')} "
                f"scope={contribution.get('scope')} "
                f"value={_number(contribution.get('value'), 2)} "
                f"support={contribution.get('support')}"
            )
    metric_ledgers = getattr(package, "metric_ledgers", {}) or {}
    for metric in sorted(metric_ledgers):
        entry = metric_ledgers[metric] or {}
        lines.append(
            f"metric_ledger metric={metric} "
            f"overlap_delta={_number(entry.get('overlap_delta'), 2)} "
            f"new_period_movement={_number(entry.get('new_period_movement'), 2)} "
            f"net_unexplained={_number(entry.get('net_unexplained'), 2)} "
            f"explained_fraction={_number(entry.get('explained_fraction'), 4)}"
        )
    lines.append(
        f"evidence net_unexplained={_number(getattr(package, 'net_unexplained', 0.0), 2)} "
        f"gross_unexplained={_number(getattr(package, 'gross_unexplained', 0.0), 2)} "
        f"explained_fraction={_number(getattr(package, 'explained_fraction', 0.0), 4)} "
        f"materiality={_number(getattr(package, 'materiality_threshold', 0.0), 2)}"
    )
    for prediction in getattr(package, "predictions", ()):
        lines.append(
            f"series {prediction.series_id} metric={prediction.metric} "
            f"level={prediction.level} week={prediction.target_week} "
            f"horizon={prediction.horizon} actual={_number(prediction.actual, 2)} "
            f"forecast={_number(prediction.expected, 2)} "
            f"residual={_number(prediction.residual, 2)} "
            f"relative={_number(prediction.relative_residual, 4)} "
            f"z={_number(prediction.standardized_residual, 3)} "
            f"interval=[{_number(prediction.interval_lower, 2)},"
            f"{_number(prediction.interval_upper, 2)}] "
            f"coverage={_number(prediction.interval_coverage, 4)} "
            f"calibration={prediction.calibration_status} "
            f"n={prediction.calibration_n} model={prediction.selected_model} "
            f"support={prediction.support} history_observed={prediction.history_observed} "
            f"history_missing={prediction.history_missing}"
        )
    for check in getattr(package, "failed_checks", ()):
        if not isinstance(check, dict):
            continue
        lines.append(
            f"failed {check.get('check')} scope={check.get('scope')} "
            f"metric={check.get('metric')} level={check.get('level')} "
            f"period={check.get('period')} outcome={check.get('outcome')} "
            f"impact={_number(check.get('impact'), 2)} "
            f"materiality={_number(check.get('materiality'), 2)}"
        )
    for entry in getattr(package, "missing_required", ()):
        lines.append(f"missing_required {entry}")
    for flag in getattr(package, "contradictions", ()):
        lines.append(f"contradiction {flag}")
    for omitted in getattr(package, "omitted", ()):
        lines.append(f"omitted {omitted}")
    return lines


def evidence_text(result: Any, max_chars: int = 8000) -> str:
    recorded = getattr(result, "recorded_text", None)
    if isinstance(recorded, str):
        return recorded[:max_chars]
    packages = getattr(result, "evidence_packages", None) or []
    package = getattr(result, "evidence_package", None)
    if not packages and package is not None:
        packages = [package]
    if packages:
        lines: list[str] = []
        for entry in packages:
            lines.extend(_package_lines(entry))
    else:
        lines = _legacy_lines(result)
    text = "\n".join(lines)
    if len(text) <= max_chars:
        return text
    trimmed = lines[:]
    while trimmed and len("\n".join(trimmed)) > max_chars:
        trimmed.pop()
    return "\n".join(trimmed)


def _legacy_lines(result: Any) -> list[str]:
    """Legacy artifacts without a package: evidence only, never policy outputs."""
    lines: list[str] = []
    lines.append(f"run_id={getattr(result, 'run_id', '')}")
    lines.append(f"dataset={getattr(result, 'dataset', '')}")
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
    reconciliation = getattr(result, "reconciliation", None)
    if reconciliation is not None:
        failed = ",".join(check.name for check in reconciliation.failed) or "none"
        lines.append(f"reconciliation failed={failed}")
    lineage = getattr(result, "lineage", None)
    if lineage is not None:
        lines.append(
            f"lineage first_divergence={lineage.first_divergence}"
        )
    temporal = getattr(result, "temporal", None)
    if temporal is not None:
        lines.append(f"temporal week={temporal.target_week}")
        for item in temporal.series:
            lines.append(
                f"series {item.series_id} actual={_number(item.actual, 2)} "
                f"forecast={_number(item.forecast_median, 2)} "
                f"residual={_number(item.relative_residual, 4)} "
                f"z={_number(item.standardized_residual, 3)} "
                f"calibration={item.calibration_status}"
            )
    for relationship in getattr(result, "relationships", ()) or ():
        lines.append(
            f"relationship {relationship.source_id}->{relationship.target_id} "
            f"type={relationship.relationship} "
            f"confidence={_number(relationship.confidence, 3)}"
        )
    return lines
