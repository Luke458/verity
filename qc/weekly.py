"""SQLite-journaled local weekly shadow assessments and recoverable publication.

Immutable assessment identities include snapshot content, configuration, references,
approvals, engine code and provider identity. Each invocation journals a separate
execution attempt; cached assessments preserve their original QC status.
"""

from __future__ import annotations

import fcntl
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC
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
    assessment_id: str = ""
    attempt_id: str = ""

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


def _checkpoint(name: str) -> None:
    """Fault-injection seam used by crash-recovery regression tests."""


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
    stage_tables: dict[str, str] | None = None,
    dim_tables: dict[str, str] | None = None,
    version_map: dict[str, dict[str, int]] | None = None,
) -> WeeklyResult:
    from datetime import datetime
    from uuid import uuid4

    import pandas as pd

    from .assessment import artifact_checksums, digest, publish, snapshot_manifest
    from .delta import DeltaSource
    from .expectations import load_expectations
    from .prequential import records_from_result
    from .reference import ReferenceSpec
    from .reporting import render_markdown
    from .run import run_qc
    from .source import CachedSource, mapped
    from .store import SqliteStore
    from .versions import select_versions

    started = time.time()
    config = config or DatasetConfig()
    from .source import ParquetManifestSource
    raw_source = ParquetManifestSource(uri) if uri.endswith(".json") else DeltaSource(uri=uri, storage_options=storage_options, stage=stage, stage_tables=stage_tables, dim_tables=dim_tables, version_map=version_map)
    source = CachedSource(mapped(raw_source, config.column_map_dict()), config)
    current, previous = select_versions(source.list_versions(), current, previous)
    source_id = str(Path(uri).resolve()) if "://" not in uri else uri
    identity = snapshot_manifest(source, previous, current, config, source_id)
    committed = identity["snapshots"][-1]["metadata"].get("committed_at_ms")
    assessment_cutoff = datetime.fromtimestamp(committed / 1000, UTC).isoformat() if committed else datetime.now(UTC).isoformat()
    identity["observation_cutoff"] = assessment_cutoff
    from .calendar import calendar_identity
    from .evidence_package import EVIDENCE_POLICY_VERSION
    identity.update(engine=code_sha256(), config=config.to_dict(),
                    reference_uri=reference_uri, reference_version=reference_version,
                    reference_stage=reference_stage, min_samples=min_samples,
                    calendar=calendar_identity(config),
                    evidence_policy=EVIDENCE_POLICY_VERSION,
                    finding_schema=2, machine_schema=3)
    expectations = load_expectations(expectations_path) if expectations_path else []
    identity["expectations"] = [item.to_dict() for item in expectations]
    reference_frame, reference_spec = None, None
    if reference_uri or reference_spec_path:
        if not reference_spec_path:
            raise ValueError("configured reference requires a specification")
        reference_spec = ReferenceSpec.from_dict(json.loads(Path(reference_spec_path).read_text()))
        if reference_uri and reference_version is not None:
            ref_source = DeltaSource(uri=reference_uri, storage_options=storage_options,
                                     stage=reference_stage or stage)
            reference_meta = ref_source.snapshot_metadata(reference_version)
            identity["reference_snapshot"] = reference_meta
            reference_commit = reference_meta.get("committed_at_ms")
            if committed and (reference_commit is None or reference_commit > committed):
                reference_frame = pd.DataFrame()
            else:
                reference_frame = ref_source.read_fact(reference_version, reference_stage or stage)
        else:
            reference_frame = pd.DataFrame()
        from .assessment import frame_digest
        identity["reference"] = {"spec": reference_spec.to_dict(), "content": frame_digest(reference_frame)}
    if decision_provider is not None:
        serializer = getattr(decision_provider, "to_dict", None)
        artifact = serializer() if serializer else getattr(decision_provider, "artifact_identity", None)
        if artifact is None:
            raise ValueError("weekly providers require pinned artifact_identity or to_dict()")
        identity["provider"] = {"artifact": artifact,
                                "endpoint": getattr(decision_provider, "url", None),
                                "timeout": getattr(decision_provider, "timeout", None),
                                "state_limit": getattr(decision_provider, "state_limit", None)}
    else:
        identity["provider"] = "rule"
    assessment_id = digest(identity)
    run_id = f"{config.name}:{assessment_id}"
    attempt_id = str(uuid4())
    observed_at = datetime.now(UTC).isoformat()
    report_dir = Path(out_root) / config.name / assessment_id
    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    lock_handle = (out / f".{config.name}.lock").open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        lock_handle.close()
        return WeeklyResult(config.name, current, previous, stage, "LOCKED", skipped=True)
    store = None
    try:
        store = SqliteStore(store_path or out / "assessments.sqlite")
        connection = store.connection
        connection.execute("INSERT INTO attempts VALUES (?, ?, ?, ?, NULL)",
                           (attempt_id, assessment_id, observed_at, "STARTED"))
        connection.commit()
        _checkpoint("after_attempt")
        row = connection.execute("SELECT * FROM assessments WHERE assessment_id = ?", (assessment_id,)).fetchone()
        if row is not None:
            if json.loads(row["identity"]) != json.loads(json_dumps(identity, default=str)):
                raise ValueError("assessment identity mismatch")
            artifacts = json.loads(row["artifacts"])
            checksums = json.loads(row["checksums"])
            publish(report_dir, artifacts, checksums)
            connection.execute("UPDATE attempts SET state = 'PUBLISHED' WHERE attempt_id = ?", (attempt_id,))
            connection.commit()
            cached = WeeklyResult(**json.loads(row["result"]))
            cached.skipped = True
            cached.notes.append("reused immutable assessment; verified/regenerated journal artifacts")
            if force:
                cached.notes.append("forced recovery attempt; immutable evidence was reused")
            cached.attempt_id = attempt_id
            cached.report_dir = str(report_dir)
            cached.artifacts = {name: str(report_dir / Path(path).name) for name, path in cached.artifacts.items()}
            return cached
        prior_refreshes = store.recent_refreshes(
            config.name, limit=max(0, config.recurrence_window - 1)
        )
        result = run_qc(source, current, previous, config, run_id=run_id,
                        decision_provider=decision_provider, reference_frame=reference_frame,
                        reference_spec=reference_spec, expectations=expectations,
                        observed_at=assessment_cutoff, prior_refreshes=prior_refreshes)
        result.machine["snapshot_manifest"] = identity
        result.machine["observed_at"] = observed_at
        records = records_from_result(result, scope=config.name)
        drift_report = None
        if result.temporal is not None:
            from .drift import monitor_drift
            from .prequential import CalibrationPool, CalibrationRecord
            pool = CalibrationPool(min_samples=min_samples)
            for prior in connection.execute("SELECT payload FROM calibration_revisions WHERE observed_at < ? ORDER BY observed_at", (assessment_cutoff,)):
                pool.add(CalibrationRecord.from_dict(json.loads(prior["payload"])))
            target = result.temporal.target_week
            drift_report = monitor_drift(pool, target, target, scope=config.name).to_dict()
            drift_report["observation_cutoff"] = assessment_cutoff
        artifacts = {"report.md": render_markdown(result),
                     "report.json": json_dumps(result.machine, indent=2, sort_keys=True, default=str)}
        if result.evidence_package is not None:
            package = result.evidence_package.to_dict()
            package["certificates"] = [
                certificate.to_dict() for certificate in result.certificates
            ]
            artifacts["evidence.json"] = json_dumps(package, indent=2, sort_keys=True, default=str)
        if result.decisions is not None and result.decisions.requires_investigation:
            from .agent import build_investigation_brief
            artifacts["brief.json"] = json_dumps(build_investigation_brief(result, result.decisions, []).to_dict(), default=str)
        manifest = WeeklyResult(config.name, current, previous, stage, result.status,
            run_id=run_id, report_dir=str(report_dir),
            artifacts={name: str(report_dir / name) for name in artifacts},
            decision=_decision_summary(result), prequential_records=len(records), drift=drift_report,
            expectations=result.machine.get("expectations"), reference=result.reference,
            recorded=True, elapsed_seconds=time.time() - started, code_sha256=identity["engine"],
            assessment_id=assessment_id, attempt_id=attempt_id)
        if calibration_path:
            manifest.notes.append("calibration revisions are authoritative in SQLite; legacy JSONL is not appended")
        artifacts["weekly.json"] = json_dumps(manifest.to_dict(), indent=2, default=str)
        checksums = artifact_checksums(artifacts)
        _checkpoint("before_commit")
        with store.transaction():
            store.record_result(result, created=observed_at)
            connection.execute("INSERT INTO assessments VALUES (?, ?, ?, ?, ?, ?)",
                (assessment_id, json_dumps(identity, default=str), result.status,
                 json_dumps(artifacts), json_dumps(checksums), json_dumps(manifest.to_dict())))
            for record in records:
                connection.execute("INSERT INTO calibration_revisions VALUES (?, ?, ?, ?, ?)",
                    (assessment_id, record.series_id, record.target_week, observed_at, json_dumps(record.to_dict())))
            connection.execute("UPDATE attempts SET state = 'COMMITTED' WHERE attempt_id = ?", (attempt_id,))
        _checkpoint("after_commit")
        _checkpoint("before_publish")
        publish(report_dir, artifacts, checksums)
        _checkpoint("after_publish")
        connection.execute("UPDATE attempts SET state = 'PUBLISHED' WHERE attempt_id = ?", (attempt_id,))
        connection.commit()
        return manifest
    except Exception as error:
        if store is not None:
            store.connection.rollback()
            store.connection.execute("UPDATE attempts SET error = ? WHERE attempt_id = ?", (str(error), attempt_id))
            store.connection.commit()
        raise
    finally:
        if store is not None:
            store.close()
        fcntl.flock(lock_handle, fcntl.LOCK_UN)
        lock_handle.close()
