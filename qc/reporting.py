"""Human-readable reports.

Markdown is the human view; the machine JSON written beside it remains the
contract for downstream code and scheduled agents. The report is rendered from
that machine output alone, so a stored ``report.json`` renders the same report
later. It leads with what needs a decision and why, then the evidence, then
what the refresh could not have seen. Report directories are created
exclusively, so an existing run is never overwritten.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .jsonutil import dumps as json_dumps

# Latest-week series listed in full before the rest are counted.
MAX_SERIES_ROWS = 15
MAX_EVENT_ROWS = 15
MAX_WEEK_ROWS = 10

_STATUS_LINES = {
    "PASS": "No check failed.",
    "PASS_WITH_EXPLANATION": "Every failed check is covered by an approval.",
    "INVESTIGATE": "Findings need review. Nothing is cleared automatically.",
    "DATA_CONTRACT_FAILURE": "The data failed its contracts, so nothing else was assessed.",
    "INCOMPLETE": "Required evidence was unavailable, so the refresh cannot be passed.",
}

_HISTORY_REASONS = {
    "missing_entities": "entities missing from the latest week carry a material share of it",
    "unmatched_events": "structural changes are not in the approved registry",
    "events_not_explained": "approved changes do not account for the revision",
    "reconstruction_incomplete": "approved changes could not be reconstructed from the data",
    "unexplained_value_change": "a material revision has no structural explanation",
    "broad_historical_recalculation": "small changes are spread across many rows",
    "events_matched": "every structural change matches an approved entry",
}

_EVENT_LABELS = {
    "LATEST_WEEK_MISSING": "missing from the latest week",
    "NEW_ENTITY_HISTORICAL_BACKFILL": "new, with backfilled history",
    "NEW_ENTITY_RECENT": "new in the latest week",
    "ENTITY_REMOVED": "removed",
    "ENTITY_HISTORY_EXTENDED": "history extended",
    "ENTITY_HISTORY_TRUNCATED": "history truncated",
    "POSSIBLE_RECLASSIFICATION": "moved category",
}

_REVIEW = {"UNEXPLAINED_ANOMALY", "HARD_FAILURE", "UNAVAILABLE_EVIDENCE"}


def _cell(value: Any) -> str:
    """Escape untrusted text for a Markdown table cell."""
    text = str(value)
    return (
        text.replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("\r", " ")
        .replace("\n", " ")
    )


def _line(value: Any) -> str:
    """Flatten untrusted text for a Markdown bullet."""
    return " ".join(str(value).replace("\r", "\n").splitlines())


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        lines.append("| " + " | ".join(_cell(value) for value in row) + " |")
    return "\n".join(lines)


def _number(value: Any, digits: int = 0) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):,.{digits}f}"


def _signed(value: Any, digits: int = 0) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):+,.{digits}f}"


def _percent(value: Any, digits: int = 1, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{100 * float(value):{'+' if signed else ''}.{digits}f}%"


def _p(value: Any) -> str:
    if value is None:
        return "n/a"
    value = float(value)
    return f"{value:.1e}" if value < 0.001 else f"{value:.3f}"


def _plural(count: int, word: str, plural: str | None = None) -> str:
    return f"{count} {word if count == 1 else (plural or word + 's')}"


def _series_name(series_id: str) -> str:
    level, _, entity = str(series_id).partition(":")
    if not entity:
        return str(series_id)
    return f"{level.removesuffix('_id')} {entity}"


def _weeks(weeks: list[Any]) -> str:
    values = sorted(int(week) for week in weeks)
    if not values:
        return "none"
    if values == list(range(values[0], values[-1] + 1)) and len(values) > 2:
        return f"{values[0]}-{values[-1]}"
    return ", ".join(str(week) for week in values)


def _review_findings(machine: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item for item in machine.get("findings", [])
        if item.get("disposition") in _REVIEW
    ]


def _anomalous_series(machine: dict[str, Any]) -> list[dict[str, Any]]:
    latest = machine.get("latest_week") or {}
    series = [item for item in latest.get("series", []) if item.get("anomaly")]

    # Strongest evidence first, then the largest move.
    return sorted(
        series,
        key=lambda item: (
            float(item.get("adjusted_p") if item.get("adjusted_p") is not None else 1.0),
            -abs(float(item.get("relative_residual") or 0.0)),
        ),
    )


def _summary(machine: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    pair = machine.get("version_pair")
    contracts = machine.get("contracts") or {}
    failed_contracts = [c for c in contracts.get("checks", []) if not c.get("passed")]
    if failed_contracts:
        names = ", ".join(_line(c.get("name")) for c in failed_contracts)
        lines.append(f"- **Contracts failed:** {names}.")
    if pair:
        shape = "" if pair.get("shape") == "NORMAL" else f" Version shape: {pair.get('shape')}."
        lines.append(
            f"- **Compared:** weeks {pair.get('overlap_start')}-{pair.get('overlap_end')}; "
            f"new: {_weeks(pair.get('new_periods', []))}.{shape}"
        )

    history = machine.get("historical_revision")
    if history:
        raw = float(history.get("raw_delta") or 0.0)
        total = float(history.get("previous_total") or 0.0)
        if raw == 0.0 and not float(history.get("breadth") or 0.0):
            lines.append("- **History:** unchanged.")
        else:
            share = f" ({_percent(raw / total, 2, signed=True)} of the previous total)" if total else ""
            explained = float(history.get("explained_delta") or 0.0)
            lines.append(
                f"- **History:** revised by {_signed(raw)}{share}; "
                f"{_signed(explained)} explained by structural changes."
            )
        reason = history.get("reason")
        # Missing entities have their own line below.
        if (
            history.get("status") in ("INVESTIGATE", "PASS_WITH_EXPLANATION")
            and reason in _HISTORY_REASONS
            and reason != "missing_entities"
        ):
            verdict = "Review" if history.get("status") == "INVESTIGATE" else "Explained"
            lines.append(f"  - {verdict}: {_HISTORY_REASONS[reason]}.")

    restated = machine.get("week_revisions") or []
    if restated:
        lines.append(
            f"- **Restated weeks:** {_weeks([item['week'] for item in restated])} "
            "changed beyond their tolerance with no structural explanation."
        )

    latest_weeks = _weeks(pair.get("new_periods", [])) if pair else "the latest week"
    for entity_type, entry in sorted((machine.get("missing_entities") or {}).items()):
        share = entry.get("share")
        total = entry.get("expected_period_total") or 0.0
        threshold = entry.get("threshold")
        limit = f" (limit {_percent(threshold / total, 2)})" if total and threshold is not None else ""
        baseline = entry.get("baseline_max")
        usual = (
            f"; clean weeks of this feed lose up to {_percent(baseline, 2)} "
            f"(previous {entry.get('baseline_weeks')} weeks)"
            if baseline is not None
            else ""
        )
        verdict = "material" if entry.get("material") else "within normal churn" if baseline is not None and share is not None and share <= baseline else "below materiality"
        lines.append(
            f"- **Missing:** {_plural(int(entry.get('entities', 0)), entity_type)} absent from week "
            f"{latest_weeks}, expected to carry {_percent(share, 2)} of it{limit}{usual}: {verdict}."
        )

    anomalous = _anomalous_series(machine)
    if anomalous:
        by_metric = Counter(str(item.get("metric") or "value") for item in anomalous)
        detail = ", ".join(f"{metric} {count}" for metric, count in sorted(by_metric.items()))
        lines.append(
            f"- **Latest-week movements:** {_plural(len(anomalous), 'series', 'series')} "
            f"outside their expected range ({detail})."
        )
    unexpected = (machine.get("expectations") or {}).get("unexpected") or []
    for flag in unexpected:
        lines.append(
            f"- **Ratio outlier:** {flag.get('metric')} in week {flag.get('week')} "
            f"is {_number(flag.get('ratio'), 2)}, outside its usual range, with no approved expectation."
        )

    lineage = machine.get("lineage") or {}
    if lineage.get("first_divergence"):
        lines.append(f"- **Lineage:** changes first appear at the `{_line(lineage['first_divergence'])}` stage.")
    reference = machine.get("reference")
    if reference:
        lines.append(f"- **Reference table:** {_line(reference.get('status'))}.")

    approved = [item for item in machine.get("findings", []) if item.get("disposition") == "HUMAN_APPROVED"]
    if approved:
        ids = sorted({approval for item in approved for approval in item.get("approval_ids", [])})
        lines.append(
            f"- **Approvals applied:** {_plural(len(approved), 'finding')} explained by "
            + ", ".join(f"`{_line(approval)}`" for approval in ids) + "."
        )
    causes = _causes(machine)
    if causes:
        lines.append(f"- **Likely cause (rules, advisory):** {causes}.")
    return lines


def _causes(machine: dict[str, Any]) -> str:
    values = (machine.get("decision") or {}).get("values") or {}
    names = (values.get("likely_causes") or {}).get("value") or []
    if not names:
        return ""
    origin = (values.get("likely_origin") or {}).get("value")
    suffix = f", origin {origin}" if origin and origin != "UNKNOWN" else ""
    return ", ".join(str(name) for name in names) + suffix


def _review_rows(machine: dict[str, Any]) -> list[list[Any]]:
    """One row per kind of finding needing a decision, with how it can be cleared."""
    findings = _review_findings(machine)
    rows: list[list[Any]] = []
    history = machine.get("historical_revision") or {}
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in findings:
        groups.setdefault(str(item.get("check", "")), []).append(item)

    for check, items in groups.items():
        unavailable = all(item.get("outcome") == "UNAVAILABLE" for item in items)
        if check == "historical_revision":
            reason = str(history.get("reason"))
            what = f"History: {_HISTORY_REASONS.get(reason, 'revision not explained')}"
            if reason == "missing_entities":
                what = "Missing from the latest week"
                missing = machine.get("missing_entities") or {}
                scope = "; ".join(
                    f"{_plural(int(entry['entities']), kind)}"
                    for kind, entry in sorted(missing.items()) if entry.get("material")
                )
                size = "; ".join(
                    f"{_number(entry['expected_lost'])} ({_percent(entry.get('share'), 2)})"
                    for _, entry in sorted(missing.items()) if entry.get("material")
                )
                clear = "Approve each closure (LATEST_WEEK_MISSING) if expected; otherwise fix the feed"
            elif reason == "unmatched_events":
                what = "Structural changes not in the approved registry"
                scope = _plural(len([e for e in machine.get("events", []) if e.get("classification") != "LATEST_WEEK_MISSING"]), "structural change")
                raw = float(history.get("raw_delta") or 0.0)
                size = _signed(raw) if raw else ""
                clear = "Approve each change in the registry if expected"
            else:
                scope = "whole history"
                size = f"{_signed(history.get('unexplained_delta'))} unexplained"
                clear = "No approval applies; investigate the source"
            rows.append([what, scope, size, clear])
        elif check == "revision_week":
            weeks = _weeks([item.get("period") for item in items if item.get("period") is not None])
            size = _number(sum(abs(float(item.get("impact") or 0.0)) for item in items))
            rows.append([
                "Restated weeks", f"weeks {weeks}", size,
                "Declare a restatement window if late data is expected; otherwise investigate",
            ])
        elif check == "temporal":
            metrics = sorted({str(item.get("metric") or "value") for item in items})
            what = "Latest-week series unavailable" if unavailable else "Latest-week movements"
            clear = (
                "Supply the missing history or measure"
                if unavailable
                else "No approval applies; investigate (an approved closure or move re-tests like-for-like)"
            )
            rows.append([what, f"{_plural(len(items), 'series', 'series')}: {', '.join(metrics)}",
                         "see Latest week" if not unavailable else "", clear])
        elif check.startswith("ratio:"):
            weeks = _weeks([item.get("period") for item in items if item.get("period") is not None])
            ratios = {
                int(flag["week"]): flag.get("ratio")
                for flag in (machine.get("expectations") or {}).get("unexpected", [])
            }
            size = ", ".join(_number(ratios.get(int(item.get("period") or 0)), 2) for item in items)
            rows.append([f"Ratio {check.split(':', 1)[1]}", f"week {weeks}", size,
                         "Approve a ratio expectation if a price change is expected"])
        elif check.startswith("contracts:"):
            name = check.split(":", 1)[1]
            details = {
                c.get("name"): c.get("detail") for c in (machine.get("contracts") or {}).get("checks", [])
            }
            rows.append([f"Contract {name}", _line(details.get(name) or items[0].get("scope")), "",
                         "Fix the data; contracts cannot be approved"])
        elif check == "reference":
            rows.append(["Reference table mismatch", _line(items[0].get("scope")), "",
                         "Reconcile with the reference owner"])
        elif check == "counterfactual_residual":
            item = items[0]
            rows.append(["Reconstruction residual", "whole history",
                         f"{_number(item.get('impact'))} (limit {_number(item.get('materiality'))})",
                         "Investigate the revision"])
        else:
            scope = ", ".join(sorted({_line(item.get("scope")) for item in items}))
            what = f"{check} unavailable" if unavailable else check
            clear = "Supply the evidence" if unavailable else "Investigate"
            rows.append([what, scope, "", clear])
    return rows


def _structural_section(machine: dict[str, Any]) -> list[str]:
    events = machine.get("events") or []
    if not events:
        return []
    lines = ["## Structural changes", ""]
    approvals = _event_approvals(machine)
    rows = []
    for event in events[:MAX_EVENT_ROWS]:
        classification = str(event.get("classification"))
        if classification == "LATEST_WEEK_MISSING":
            value = f"expected {_number(event.get('expected_value'))}"
            weeks = ""
        elif classification == "POSSIBLE_RECLASSIFICATION":
            details = event.get("details") or {}
            count = details.get("product_count", len(details.get("products", [])))
            value = f"{_plural(int(count), 'product')}, {_number(details.get('moved_previous'))} of history"
            weeks = ""
        else:
            added = float(event.get("historical_value_added") or 0.0)
            removed = float(event.get("historical_value_removed") or 0.0)
            value = _signed(added - removed)
            weeks = _weeks(list(event.get("historical_weeks_added", [])) + list(event.get("historical_weeks_removed", [])))
        key = f"{event.get('entity_type')}:{event.get('entity_id')}"
        rows.append([
            f"{event.get('entity_type')} {event.get('entity_id')}",
            _EVENT_LABELS.get(classification, classification),
            weeks,
            value,
            approvals.get(key, "not approved"),
        ])
    headers = ["entity", "change", "weeks", "value", "approval"]
    keep = [index for index in range(len(headers)) if index != 2 or any(row[2] for row in rows)]
    lines.append(_table([headers[i] for i in keep], [[row[i] for i in keep] for row in rows]))
    if len(events) > MAX_EVENT_ROWS:
        lines.append(f"\n{len(events) - MAX_EVENT_ROWS} more in report.json.")
    lines.append("")
    return lines


def _event_approvals(machine: dict[str, Any]) -> dict[str, str]:
    """Entity key -> approval id, for structural changes an approval explained."""
    approvals: dict[str, str] = {}
    for item in (machine.get("like_for_like") or {}).get("excluded", []):
        approvals[str(item.get("entity"))] = str(item.get("approval_id"))
    for item in (machine.get("like_for_like") or {}).get("restated", []):
        approvals[f"commodity:{item.get('move')}"] = str(item.get("approval_id"))
    for item in machine.get("findings", []):
        if item.get("disposition") == "HUMAN_APPROVED" and str(item.get("check", "")).startswith(
            ("historical_event", "absence_event")
        ):
            # Scope is "<type>:<id>:<classification>".
            key = str(item.get("scope")).rsplit(":", 1)[0]
            approvals.setdefault(key, ", ".join(item.get("approval_ids", [])))
    return approvals


def _history_section(machine: dict[str, Any]) -> list[str]:
    history = machine.get("historical_revision")
    restated = machine.get("week_revisions") or []
    if not history or (float(history.get("raw_delta") or 0.0) == 0.0 and not restated):
        return []
    lines = ["## History", ""]
    lines.append(_table(
        ["revised", "explained", "unexplained", "rows changed", "status"],
        [[
            _signed(history.get("raw_delta")),
            _signed(history.get("explained_delta")),
            _signed(history.get("unexplained_delta")),
            _percent(history.get("breadth"), 1),
            history.get("status"),
        ]],
    ))
    if restated:
        largest = sorted(restated, key=lambda item: -abs(float(item.get("unexplained") or 0.0)))
        shown = sorted(largest[:MAX_WEEK_ROWS], key=lambda item: int(item.get("week", 0)))
        lines.append("")
        lines.append(_table(
            ["week", "revised", "unexplained", "of the week", "tolerance"],
            [[
                item.get("week"),
                _signed(item.get("delta")),
                _signed(item.get("unexplained")),
                _percent(item.get("relative"), 2),
                _percent(item.get("tolerance"), 1),
            ] for item in shown],
        ))
        if len(restated) > MAX_WEEK_ROWS:
            lines.append(
                f"\nThe {MAX_WEEK_ROWS} largest of {len(restated)} restated weeks; the rest are in report.json."
            )
    lines.append("")
    return lines


def _latest_section(machine: dict[str, Any]) -> list[str]:
    latest = machine.get("latest_week")
    if not latest:
        return []
    lines = [f"## Latest week {latest.get('target_week')}", ""]
    anomalous = _anomalous_series(machine)
    if anomalous:
        rows = []
        for item in anomalous[:MAX_SERIES_ROWS]:
            share_test = item.get("share_p") is not None and "share_shift" in (item.get("flags") or [])
            rows.append([
                _series_name(item.get("series_id", "")),
                item.get("metric", ""),
                _number(item.get("actual")),
                _number(item.get("forecast_median")),
                _percent(item.get("relative_residual"), 1, signed=True),
                "share of parent" if share_test else "forecast",
                _p(item.get("adjusted_p")),
            ])
        lines.append(_table(
            ["series", "measure", "actual", "expected", "vs expected", "test", "adjusted p"], rows,
        ))
        if len(anomalous) > MAX_SERIES_ROWS:
            lines.append(f"\n{len(anomalous) - MAX_SERIES_ROWS} more in report.json.")
        lines.append("")
        lines.append(
            f"Each series is one test in a single false-discovery family per refresh "
            f"(q = {(machine.get('statistical') or {}).get('fdr_q', 'n/a')}). Category series are "
            "tested on their share of the national total, so a move in one category "
            "shifts the others' shares too."
        )
    else:
        assessed = len(latest.get("series", []))
        lines.append(f"No series moved outside its expected range ({_plural(assessed, 'series', 'series')} assessed).")
    unavailable = latest.get("unavailable") or []
    if unavailable or latest.get("unavailable_series"):
        count = len(unavailable) or len(latest.get("unavailable_series", []))
        lines.append(f"\n{_plural(count, 'series', 'series')} could not be assessed (see report.json).")
    lines.append("")
    return lines


def _sensitivity_section(machine: dict[str, Any]) -> list[str]:
    sensitivity = (machine.get("latest_week") or {}).get("sensitivity") or {}
    if not sensitivity:
        return []
    lines = [
        "## What this refresh could not see",
        "",
        "The smallest single-week change a category could show and still be flagged on its own "
        "(median across the categories at that level). Smaller changes pass unseen.",
        "",
    ]
    rows = []
    for key, entry in sorted(sensitivity.items()):
        level, _, metric = key.partition(":")
        rows.append([
            level.removesuffix("_id"),
            metric,
            entry.get("leaves"),
            _percent(entry.get("median_detectable_change"), 1),
            _series_name(entry.get("least_sensitive", "")),
        ])
    lines.append(_table(["level", "measure", "categories", "median detectable change", "least sensitive"], rows))
    lines.append("")
    return lines


def _passed_section(machine: dict[str, Any]) -> list[str]:
    names = {"temporal": "latest week", "revision_week": "week revisions", "ratio": "ratios"}
    families: dict[str, list[str]] = {}
    for item in machine.get("findings", []):
        family = str(item.get("check", "")).split(":", 1)[0]
        if family.startswith("hierarchy"):
            family = "hierarchy"
        family = names.get(family, family.replace("_", " "))
        # Informational findings are below escalation thresholds, not failures.
        families.setdefault(family, []).append(str(item.get("disposition")))
    parts = []
    for family, outcomes in sorted(families.items()):
        passed = sum(outcome in ("PASS", "INFORMATIONAL") for outcome in outcomes)
        approved = outcomes.count("HUMAN_APPROVED")
        part = f"{family} {passed}/{len(outcomes)}"
        parts.append(part + (f" ({approved} approved)" if approved else ""))
    if not parts:
        return []
    return ["## Checks", "", "Passed, by check: " + "; ".join(parts) + ".", ""]


def _labels_section(machine: dict[str, Any]) -> list[str]:
    decision = machine.get("decision") or {}
    values = decision.get("values") or {}
    if not values:
        return []
    lines = [
        "## Cause labels",
        "",
        "Rule-based and advisory: they never change the status or the review.",
        "",
    ]
    causes = (values.get("likely_causes") or {}).get("value") or []
    if not causes:
        return []
    lines.append(f"- Causes: {', '.join(str(c) for c in causes)}")
    origin = (values.get("likely_origin") or {}).get("value")
    if origin:
        lines.append(f"- Origin: {origin}")
    severity = (values.get("severity") or {}).get("value")
    if severity:
        lines.append(f"- Severity: {severity}")
    evidence = (values.get("likely_cause") or {}).get("evidence") or []
    if evidence:
        shown = ", ".join(f"`{_line(item)}`" for item in evidence[:8])
        more = f" and {len(evidence) - 8} more" if len(evidence) > 8 else ""
        lines.append(f"- Evidence: {shown}{more}")
    lines.append("")
    return lines


def render_machine(machine: dict[str, Any]) -> str:
    """Render the human report from a machine output dictionary."""
    pair = machine.get("version_pair") or {}
    status = str(machine.get("status"))
    title = f"{pair.get('previous_id')} -> {pair.get('current_id')}" if pair else machine.get("run_id")
    lines = [
        f"# Verity QC: {_line(machine.get('dataset'))}, {_line(title)}",
        "",
        f"**{status}**: {_STATUS_LINES.get(status, '')}",
        "",
    ]
    lines.extend(_summary(machine))
    lines.append("")

    rows = _review_rows(machine)
    if rows:
        lines.extend(["## Needs a decision", ""])
        lines.append(_table(["what", "where", "size", "how it can be cleared"], rows))
        lines.append("")

    lines.extend(_structural_section(machine))
    lines.extend(_history_section(machine))
    lines.extend(_latest_section(machine))
    lines.extend(_sensitivity_section(machine))
    lines.extend(_passed_section(machine))
    lines.extend(_labels_section(machine))

    findings = machine.get("findings", [])
    lines.append(
        f"Run `{_line(machine.get('run_id'))}`; {_plural(len(findings), 'finding')}; "
        f"machine schema {machine.get('schema_version', 'n/a')}. Full evidence is in report.json."
    )
    return "\n".join(lines).rstrip() + "\n"


def render_markdown(result: Any) -> str:
    # Round-trip through JSON so the live result renders exactly as its stored
    # report.json would.
    return render_machine(json.loads(json_dumps(result.machine, sort_keys=True, default=str)))


def write_report(result: Any, directory: str | Path) -> dict[str, Path]:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=False)
    markdown_path = path / "report.md"
    machine_path = path / "report.json"
    markdown_path.write_text(render_markdown(result))
    machine_path.write_text(
        json_dumps(result.machine, indent=2, sort_keys=True, default=str)
    )
    return {"markdown": markdown_path, "machine": machine_path}
