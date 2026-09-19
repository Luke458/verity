"""Weekly refresh orchestrator: one entry point after each load.

Runs the full chain for a newly committed version and returns a status that a
scheduler can alert on:

    version resolution -> contracts/engine -> optional reference control
    -> scoped expectations -> engine run -> regression/decision layer
    -> calibration append -> drift check -> durable store -> report + brief

Idempotent by report directory: a version already processed is returned as
``ALREADY_PROCESSED`` without doing work unless ``force`` is set. Nothing here
schedules itself; it is designed to be called from cron, Airflow, or a
Databricks job once the refresh commit lands.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .cohort import code_sha256
from .config import DatasetConfig
from .jsonutil import dumps as json_dumps


@dataclass
class WeeklyResult:
    dataset: str
    current_version: str | None
    previous_version: str | None
    stage: str
    status: str
    run_id: str | None = None
    report_dir: str | None = None
    artifacts: dict[str, str] = field(default_factory=dict)
    decision: dict[str, Any] | None = None
    drift: dict[str, Any] | None = None
    prequential_records: int = 0
    expectations: dict[str, Any] | None = None
    reference: dict[str, Any] | None = None
    recorded: bool = False
    skipped: bool = False
    notes: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    code_sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _decision_summary(result: Any) -> dict[str, Any] | None:
    decisions = getattr(result, "decisions", None)
    if decisions is None:
        return None
    cause = decisions.get("likely_cause")
    origin = decisions.get("likely_origin")
    severity = decisions.get("severity")
    return {
        "provider": decisions.provider,
        "likely_cause": str(cause.value) if cause is not None else None,
        "likely_origin": str(origin.value) if origin is not None else None,
        "severity": str(severity.value) if severity is not None else None,
        "severity_index": severity.index if severity is not None else None,
        "requires_investigation": decisions.requires_investigation,
    }


def _ratio_flags(result: Any) -> list[dict[str, Any]]:
    reconciliation = getattr(result, "reconciliation", None)
    if reconciliation is None:
        return []
    return [
        {
            "metric": "dollar_per_unit",
            "week": flag["week"],
            "ratio": flag["dollar_per_unit"],
            "median": flag["median"],
            "robust_z": flag["robust_z"],
        }
        for flag in reconciliation.ratio_flags
    ]


def _version_key(value: str) -> tuple[int, object]:
    text = str(value)
    return (0, int(text)) if text.isdigit() else (1, text)


def run_weekly(
    uri: str,
    config: DatasetConfig | None = None,
    store_path: str | Path | None = None,
    calibration_path: str | Path | None = None,
    out_root: str | Path = "reports/weekly",
    stage: str = "warehouse",
    current: str | None = None,
    previous: str | None = None,
    expectations_path: str | Path | None = None,
    reference_uri: str | None = None,
    reference_spec_path: str | Path | None = None,
    reference_stage: str | None = None,
    reference_version: str | None = None,
    storage_options: dict[str, str] | None = None,
    min_samples: int = 9,
    decision_provider: Any | None = None,
    force: bool = False,
) -> WeeklyResult:
    from .delta import DeltaSource
    from .drift import monitor_drift
    from .expectations import apply_expectations, load_expectations
    from .prequential import PrequentialStore, records_from_result
    from .reporting import render_markdown
    from .run import run_qc
    from .source import mapped

    started = time.time()
    config = config or DatasetConfig()
    source = mapped(
        DeltaSource(uri=uri, storage_options=storage_options, stage=stage),
        config.column_map_dict(),
    )
    versions = source.list_versions()
    current = current or (versions[-1] if versions else None)
    previous = previous or (versions[-2] if len(versions) > 1 else None)
    if not current or not previous:
        raise ValueError("weekly run needs at least two versions")
    config_hash = hashlib.sha256(
        json.dumps(config.to_dict(), sort_keys=True, default=str).encode()
    ).hexdigest()[:8]
    run_id = f"{config.name}:{stage}:{current}:{config_hash}"
    report_dir = Path(out_root) / config.name / str(current)

    out_root_path = Path(out_root)
    out_root_path.mkdir(parents=True, exist_ok=True)
    lock_path = out_root_path / f".{config.name}.lock"
    lock_handle = lock_path.open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        lock_handle.close()
        return WeeklyResult(
            dataset=config.name,
            current_version=current,
            previous_version=previous,
            stage=stage,
            status="LOCKED",
            run_id=run_id,
            report_dir=str(report_dir),
            skipped=True,
            notes=["another weekly run holds the lock; retry later"],
            elapsed_seconds=time.time() - started,
            code_sha256=code_sha256(),
        )
    try:
        completed = (report_dir / "weekly.json").exists()
        if completed and not force:
            return WeeklyResult(
                dataset=config.name,
                current_version=current,
                previous_version=previous,
                stage=stage,
                status="ALREADY_PROCESSED",
                run_id=run_id,
                report_dir=str(report_dir),
                skipped=True,
                notes=["completed report exists; use force=true to rerun"],
                elapsed_seconds=time.time() - started,
                code_sha256=code_sha256(),
            )

        notes: list[str] = []
        if report_dir.exists() and not completed:
            notes.append(
                "incomplete report directory found (no weekly.json); reprocessing"
            )

        reference_frame = None
        reference_spec = None
        if reference_uri and reference_spec_path:
            from .reference import ReferenceSpec

            reference_source = DeltaSource(
                uri=reference_uri,
                storage_options=storage_options,
                stage=reference_stage or stage,
            )
            reference_spec = ReferenceSpec.from_dict(
                json.loads(Path(reference_spec_path).read_text())
            )
            reference_versions = reference_source.list_versions()
            chosen = reference_version
            if chosen is None:
                if current in reference_versions:
                    chosen = current
                else:
                    eligible = [
                        value
                        for value in reference_versions
                        if _version_key(value) <= _version_key(current)
                    ]
                    chosen = (
                        max(eligible, key=_version_key) if eligible else None
                    )
            elif _version_key(chosen) > _version_key(current):
                notes.append(
                    "reference version is later than current; reference skipped"
                )
                chosen = None
            if chosen is None:
                notes.append("reference table has no usable version; reference skipped")
                reference_spec = None
            else:
                reference_frame = reference_source.read_fact(
                    chosen, reference_stage or stage
                )
        elif reference_uri or reference_spec_path:
            notes.append("reference_uri and reference_spec are both required; skipped")

        result = run_qc(
            source,
            current,
            previous,
            config,
            run_id=run_id,
            decision_provider=decision_provider,
            reference_frame=reference_frame,
            reference_spec=reference_spec,
        )

        expectations_report = None
        if expectations_path:
            expectations = load_expectations(expectations_path)
            target = (
                int(result.temporal.target_week)
                if result.temporal is not None
                else 0
            )
            expectations_report = apply_expectations(
                _ratio_flags(result),
                expectations,
                {"dataset": result.dataset, "as_of": str(target)},
            )

        calibration_added = 0
        if calibration_path:
            records = records_from_result(result, scope=config.name)
            if records:
                PrequentialStore(calibration_path).add(records)
            calibration_added = len(records)

        drift = None
        if calibration_path:
            pool = PrequentialStore(calibration_path).pool(min_samples=min_samples)
            drift_target = (
                int(result.temporal.target_week)
                if result.temporal is not None
                else None
            )
            if drift_target is not None:
                drift = monitor_drift(
                    pool,
                    as_of=drift_target,
                    target_week=drift_target,
                    scope=config.name,
                ).to_dict()
                if drift["status"] == "DRIFT":
                    notes.append(
                        "calibration drift detected; refit before trusting intervals"
                    )

        recorded = False
        if store_path:
            from .store import SqliteStore

            try:
                with SqliteStore(store_path) as store:
                    store.record_result(result)
                recorded = True
            except ValueError as error:
                notes.append(str(error))

        brief = None
        if result.decisions is not None and result.decisions.requires_investigation:
            from .agent import build_investigation_brief
            from .decisions import FeatureEncoder
            from .incidents import symptom_tags_for

            incidents: list = []
            if store_path:
                try:
                    with SqliteStore(store_path) as store:
                        incidents = store.retrieve(
                            FeatureEncoder().encode(result),
                            k=3,
                            tags=symptom_tags_for(result, result.decisions),
                        )
                except Exception as error:  # noqa: BLE001 - memory is optional
                    notes.append(f"incident retrieval failed: {error}")
            brief = build_investigation_brief(result, result.decisions, incidents)

        tmp_dir = report_dir.with_name(
            f"{report_dir.name}.tmp-{os.getpid()}"
        )
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True)
        try:
            markdown_path = tmp_dir / "report.md"
            machine_path = tmp_dir / "report.json"
            markdown_path.write_text(render_markdown(result, brief))
            machine_path.write_text(
                json_dumps(result.machine, indent=2, sort_keys=True, default=str)
            )
            artifacts = {
                "markdown": str(report_dir / "report.md"),
                "machine": str(report_dir / "report.json"),
            }
            if brief is not None:
                brief_path = tmp_dir / "brief.json"
                brief_path.write_text(
                    json_dumps(brief.to_dict(), indent=2, sort_keys=True, default=str)
                )
                artifacts["brief"] = str(report_dir / "brief.json")

            manifest = WeeklyResult(
                dataset=config.name,
                current_version=current,
                previous_version=previous,
                stage=stage,
                status=result.status,
                run_id=run_id,
                report_dir=str(report_dir),
                artifacts=artifacts,
                decision=_decision_summary(result),
                drift=drift,
                prequential_records=calibration_added,
                expectations=expectations_report,
                reference=result.machine.get("reference"),
                recorded=recorded,
                notes=notes,
                elapsed_seconds=time.time() - started,
                code_sha256=code_sha256(),
            )
            (tmp_dir / "weekly.json").write_text(
                json_dumps(manifest.to_dict(), indent=2, sort_keys=True, default=str)
            )

            backup = report_dir.with_name(f"{report_dir.name}.bak")
            if report_dir.exists():
                if backup.exists():
                    shutil.rmtree(backup)
                os.replace(report_dir, backup)
            os.replace(tmp_dir, report_dir)
            if backup.exists():
                shutil.rmtree(backup, ignore_errors=True)
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise
        return manifest
    finally:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_UN)
        finally:
            lock_handle.close()
