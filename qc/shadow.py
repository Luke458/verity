"""Shadow-mode evaluation harness.

Runs the engine over every scenario in a suite, persists one record per
scenario, and (for synthetic suites with a manifest) scores engine outcomes
against the fault oracle. On real data the oracle fields are absent and the
records become the analyst-feedback and calibration store.

The scenario adapter is imported lazily; the engine package remains free of
generator dependencies.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import DatasetConfig
from .events import load_registry
from .labels import LabelStore, oracle_labels_for_result
from .prequential import CalibrationRecord, PrequentialStore
from .run import run_qc

# Families whose divergence is expected to be visible in overlap history.
LINEAGE_COMPARABLE_FAMILIES = (
    "new_store_backfill",
    "expected_event",
    "history_truncation",
    "entity_merge",
    "coding_error",
    "warehouse_transform_error",
    "recalculation",
)


@dataclass
class ShadowRecord:
    scenario_id: str
    run_id: str
    family: str | None
    case_kind: str | None
    engine_status: str
    historical_status: str | None = None
    latest_week_anomaly: bool | None = None
    latest_week_flags: list[str] = field(default_factory=list)
    oracle_status: str | None = None
    oracle_class: str | None = None
    injection_stage: str | None = None
    first_divergence: str | None = None
    matched_expected_events: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    raw_delta: float = 0.0
    explained_fraction: float = 0.0
    reconstruction_score: float | None = None
    reconciliation_status: str | None = "PASS"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _json_default(value):
    import numpy as np

    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def _summarize(records: list[ShadowRecord], suite: dict) -> dict[str, Any]:
    def rate(values: list[bool]) -> float:
        return sum(values) / len(values) if values else 0.0

    faulty = [r for r in records if r.case_kind in ("fault", "expected_event")]
    controls = [r for r in records if r.case_kind == "control"]
    expected_events = [r for r in records if r.case_kind == "expected_event"]
    comparable = [
        r
        for r in records
        if r.family in LINEAGE_COMPARABLE_FAMILIES and r.injection_stage is not None
    ]

    status_counts: dict[str, int] = {}
    for record in records:
        status_counts[record.engine_status] = (
            status_counts.get(record.engine_status, 0) + 1
        )

    return {
        "suite_id": suite.get("suite_id"),
        "scenarios": len(records),
        "engine_status_counts": status_counts,
        "detection_rate": rate([r.engine_status != "PASS" for r in faulty]),
        "false_positive_rate": rate(
            [
                r.historical_status not in (None, "PASS")
                for r in controls
            ]
        ),
        "latest_week_detection_rate": rate(
            [bool(r.latest_week_anomaly) for r in faulty]
        ),
        "latest_week_control_rate": rate(
            [bool(r.latest_week_anomaly) for r in controls]
        ),
        "expected_event_pass_rate": rate(
            [
                r.historical_status == "PASS_WITH_EXPLANATION"
                for r in expected_events
            ]
        ),
        "lineage_first_divergence_accuracy": rate(
            [r.first_divergence == r.injection_stage for r in comparable]
        ),
        "lineage_comparable": len(comparable),
        "mean_reconstruction_score": (
            sum(
                r.reconstruction_score
                for r in faulty
                if r.reconstruction_score is not None
            )
            / len(
                [
                    r
                    for r in faulty
                    if r.reconstruction_score is not None
                ]
            )
            if any(r.reconstruction_score is not None for r in faulty)
            else 0.0
        ),
        "reconstruction_evaluated": sum(
            1 for r in faulty if r.reconstruction_score is not None
        ),
        "reconciliation_failures": sum(
            1
            for r in records
            if r.reconciliation_status not in (None, "PASS")
        ),
        "records": [r.to_dict() for r in records],
    }


def run_shadow(
    suite_dir: str | Path,
    config: DatasetConfig | None = None,
    registry_path: str | Path | None = None,
    out_dir: str | Path | None = None,
    source_factory: Callable[[Path], Any] | None = None,
    labels_out: str | Path | None = None,
    prequential_store: str | Path | None = None,
) -> dict[str, Any]:
    suite_dir = Path(suite_dir)
    config = config or DatasetConfig()
    suite = json.loads((suite_dir / "suite.json").read_text())
    registry = load_registry(registry_path) if registry_path else None
    label_store = LabelStore(labels_out) if labels_out else None
    calibration_store = (
        PrequentialStore(prequential_store) if prequential_store else None
    )

    if source_factory is None:
        from qcgen.sources import ScenarioSource

        source_factory = ScenarioSource

    records: list[ShadowRecord] = []
    for entry in suite["scenarios"]:
        scenario_dir = suite_dir / entry["scenario_id"]
        manifest = json.loads((scenario_dir / "manifest.json").read_text())
        case = manifest["cases"][0] if manifest.get("cases") else {}
        expected_events = (
            registry
            if registry is not None
            else manifest.get("expected_events", [])
        )
        result = run_qc(
            source_factory(scenario_dir),
            manifest["current_version"],
            manifest["previous_version"],
            config,
            expected_events=expected_events,
        )
        if label_store is not None:
            label_record = oracle_labels_for_result(result, manifest)
            label_record.metadata["suite_id"] = suite.get("suite_id")
            label_store.append([label_record])
        if calibration_store is not None and result.temporal is not None:
            calibration_store.add(
                [
                    CalibrationRecord(
                        series_id=item.series_id,
                        target_week=item.target_week,
                        available_on=item.target_week,
                        residual=item.standardized_residual,
                        run_id=result.run_id,
                        scope=str(suite.get("suite_id", "default")),
                    )
                    for item in result.temporal.series
                ]
            )
        records.append(
            ShadowRecord(
                scenario_id=entry["scenario_id"],
                run_id=result.run_id,
                family=manifest.get("family"),
                case_kind=case.get("kind"),
                engine_status=result.status,
                historical_status=(
                    str(result.machine["historical_revision"]["status"])
                    if result.machine.get("historical_revision")
                    else None
                ),
                latest_week_anomaly=(
                    result.temporal.anomaly if result.temporal is not None else None
                ),
                latest_week_flags=(
                    list(result.temporal.flags)
                    if result.temporal is not None
                    else []
                ),
                oracle_status=case.get("expected_status"),
                oracle_class=case.get("expected_class"),
                injection_stage=case.get("injection_stage"),
                first_divergence=(
                    result.lineage.first_divergence
                    if result.lineage is not None
                    else None
                ),
                matched_expected_events=(
                    list(result.attribution.matched_event_ids)
                    if result.attribution is not None
                    else []
                ),
                reasons=list(result.reasons),
                raw_delta=(
                    result.attribution.raw_delta
                    if result.attribution is not None
                    else 0.0
                ),
                explained_fraction=(
                    result.attribution.explained_fraction
                    if result.attribution is not None
                    else 0.0
                ),
                reconstruction_score=(
                    result.counterfactual.reconciliation_score
                    if result.counterfactual is not None
                    else None
                ),
                reconciliation_status=(
                    result.reconciliation.status
                    if result.reconciliation is not None
                    else None
                ),
            )
        )

    summary = _summarize(records, suite)
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "shadow.jsonl").write_text(
            "\n".join(
                json.dumps(record.to_dict(), default=_json_default)
                for record in records
            )
            + "\n"
        )
        (out / "shadow.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, default=_json_default)
        )
    return summary
