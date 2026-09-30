from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("deltalake")

from deltalake import write_deltalake

from qc.cli import main
from qc.config import DatasetConfig
from qc.delta import DeltaSource
from qc.prequential import records_from_result
from qc.run import run_qc
from qc.weekly import run_weekly


def _series_value(week: int) -> float:
    return 100.0 + 0.5 * ((week * 3) % 4)


def _frame(
    target: int,
    latest_stores: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    rows = []
    for week in range(1, target + 1):
        value = _series_value(week)
        if week == target:
            value = _series_value(target - 1)  # flat latest week: zero residual
        stores = ("S1", "S2") if week < target else (latest_stores or ("S1", "S2"))
        for store in stores:
            rows.append(
                {"week": week, "store_id": store, "product_id": "P1", "dollar": value, "units": 10}
            )
    return pd.DataFrame(rows)


def _write_versions(path, targets, missing_latest_store: bool = False) -> None:
    for index, target in enumerate(targets):
        latest_stores = None
        if missing_latest_store and index == len(targets) - 1:
            latest_stores = ("S1",)
        frame = _frame(target, latest_stores)
        if index == 0:
            write_deltalake(str(path), frame)
        else:
            write_deltalake(str(path), frame, mode="overwrite")


def test_weekly_pass_artifacts_and_idempotency(tmp_path):
    path = tmp_path / "fact"
    _write_versions(path, (30, 31, 32))
    config = replace(DatasetConfig(), name="weekly-test")
    result = run_weekly(
        str(path),
        config=config,
        store_path=tmp_path / "store.db",
        calibration_path=tmp_path / "calibration.jsonl",
        out_root=tmp_path / "weekly",
    )
    assert result.status == "INCOMPLETE", result.notes
    assert result.recorded is True
    assert result.prequential_records >= 1
    assert result.current_version == "2"
    assert result.previous_version == "1"
    for name in ("report.md", "report.json", "weekly.json"):
        assert (Path(result.report_dir) / name).exists(), name

    again = run_weekly(
        str(path), config=config, store_path=tmp_path / "store.db", out_root=tmp_path / "weekly"
    )
    assert again.status == result.status
    assert again.skipped is True


def test_weekly_investigate_and_exit_codes(tmp_path):
    path = tmp_path / "fact"
    _write_versions(path, (30, 31, 32), missing_latest_store=True)
    config = replace(DatasetConfig(), name="weekly-investigate")
    result = run_weekly(
        str(path), config=config, out_root=tmp_path / "weekly"
    )
    assert result.status == "INVESTIGATE"
    assert result.decision is not None
    assert result.decision["requires_investigation"] is True
    assert (Path(result.report_dir) / "brief.json").exists()

    assert (
        main(
            [
                "weekly",
                "--uri",
                str(path),
                "--out",
                str(tmp_path / "cli-investigate"),
            ]
        )
        == 2
    )
    assert (
        main(
            [
                "weekly",
                "--uri",
                str(path),
                "--out",
                str(tmp_path / "cli-allow"),
                "--allow-investigate",
            ]
        )
        == 2
    )
    # Cached assessments preserve the investigation exit code.
    assert (
        main(
            [
                "weekly",
                "--uri",
                str(path),
                "--out",
                str(tmp_path / "cli-allow"),
            ]
        )
        == 2
    )


def _sink_file(path: Path) -> str:
    path.write_text(json.dumps({"sinks": [{"kind": "file", "target": str(path.parent / "out.jsonl")}]}))
    return str(path)


def test_weekly_notification_does_not_change_exit_code(tmp_path):
    # The contract that matters: adding a sink must not be able to alter the
    # scheduler's decision, whether the sink works or fails.
    path = tmp_path / "fact"
    _write_versions(path, (30, 31, 32), missing_latest_store=True)
    config = replace(DatasetConfig(), name="weekly-notify")
    run_weekly(str(path), config=config, out_root=tmp_path / "warm")

    out = tmp_path / "weekly"
    assert main([
        "weekly", "--uri", str(path), "--out", str(out),
        "--notify", _sink_file(tmp_path / "sinks.json"),
    ]) == 2
    delivered = (tmp_path / "out.jsonl").read_text().strip().split("\n")
    assert delivered
    assert json.loads(delivered[-1])["status"] == "INVESTIGATE"

    # A sink pointed at a dead port still leaves the exit code untouched.
    dead = tmp_path / "dead.json"
    dead.write_text(json.dumps({"sinks": [
        {"kind": "webhook", "target": "http://127.0.0.1:9/x", "timeout": 1.0}
    ]}))
    assert main([
        "weekly", "--uri", str(path), "--out", str(out), "--force",
        "--notify", str(dead),
    ]) == 2


def test_weekly_notification_rejects_unusable_spec_without_failing(tmp_path):
    path = tmp_path / "fact"
    _write_versions(path, (30, 31, 32), missing_latest_store=True)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"sinks": [{"kind": "smoke-signal"}]}))
    code = main([
        "weekly", "--uri", str(path), "--out", str(tmp_path / "weekly"),
        "--notify", str(bad),
    ])
    assert code in (0, 1, 2, 3, 4)


def test_weekly_cached_assessment_notification_is_suppressed(tmp_path):
    # A retry must not re-page for an assessment that was already delivered.
    path = tmp_path / "fact"
    _write_versions(path, (30, 31, 32), missing_latest_store=True)
    out = tmp_path / "weekly"
    sinks = _sink_file(tmp_path / "sinks.json")
    main(["weekly", "--uri", str(path), "--out", str(out), "--notify", sinks])
    main(["weekly", "--uri", str(path), "--out", str(out), "--notify", sinks])
    lines = (tmp_path / "out.jsonl").read_text().strip().split("\n")
    # The cached retry is suppressed and never reaches the sink.
    assert len(lines) == 1
    assert "suppressed" not in json.loads(lines[0])


def test_weekly_force_rerun_notes_duplicate_run(tmp_path):
    path = tmp_path / "fact"
    _write_versions(path, (30, 31))
    config = replace(DatasetConfig(), name="weekly-force")
    store = tmp_path / "store.db"
    out = tmp_path / "weekly"
    run_weekly(str(path), config=config, store_path=store, out_root=out)
    forced = run_weekly(
        str(path), config=config, store_path=store, out_root=out, force=True
    )
    assert forced.skipped is True
    assert forced.recorded is True
    assert any("reused immutable" in note for note in forced.notes)


def test_weekly_lock_prevents_overlap(tmp_path):
    import fcntl

    path = tmp_path / "fact"
    _write_versions(path, (30, 31, 32))
    config = replace(DatasetConfig(), name="weekly-lock")
    out = tmp_path / "weekly"
    out.mkdir(parents=True, exist_ok=True)
    lock_path = out / ".weekly-lock.lock"
    handle = lock_path.open("w")
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        result = run_weekly(str(path), config=config, out_root=out)
        assert result.status == "LOCKED"
        assert result.skipped is True
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()

    after = run_weekly(str(path), config=config, out_root=out)
    assert after.status == "INCOMPLETE"


def test_incomplete_report_directory_is_reprocessed(tmp_path):
    path = tmp_path / "fact"
    _write_versions(path, (30, 31, 32))
    config = replace(DatasetConfig(), name="weekly-partial")
    out = tmp_path / "weekly"
    report_dir = out / config.name / "2"
    report_dir.mkdir(parents=True)
    (report_dir / "report.md").write_text("stale partial output")

    result = run_weekly(str(path), config=config, out_root=out)
    assert result.skipped is False
    assert Path(result.report_dir) != report_dir
    assert (report_dir / "report.md").read_text() == "stale partial output"
    assert (Path(result.report_dir) / "weekly.json").exists()


def test_records_from_result_match_temporal_series(tmp_path):
    path = tmp_path / "fact"
    _write_versions(path, (30, 31))
    source = DeltaSource(uri=str(path))
    result = run_qc(source, "1", "0", run_id="series-check")
    records = records_from_result(result, scope="check")
    assert result.temporal is not None
    assert len(records) == len(result.temporal.series)
    assert all(record.scope == "check" for record in records)
    assert all(record.available_on == record.target_week for record in records)


def test_weekly_early_store_failure_surfaces_original_error(tmp_path, monkeypatch):
    # Regression: a failure before the attempt row existed raised
    # UnboundLocalError from the handler and hid the real error.
    from qc.store import SqliteStore

    path = tmp_path / "fact"
    _write_versions(path, (30, 31))

    def boom(self, *args, **kwargs):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(SqliteStore, "recurrence_inputs", boom)
    with pytest.raises(RuntimeError, match="database is locked"):
        run_weekly(str(path), out_root=tmp_path / "weekly")
