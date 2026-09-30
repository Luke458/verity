from __future__ import annotations

import json
from dataclasses import replace

import pandas as pd
import pytest

from qc.config import DatasetConfig
from qc.reference import ReferenceSpec, compare_reference
from qc.run import run_qc
from qc.source import MappedSource
from qc.store import SqliteStore
from qc.versions import select_versions


class HandSource:
    def __init__(self):
        self.frames = {}
        for version, weeks in (("v1", 3), ("v2", 4)):
            base = pd.DataFrame(
                [
                    {
                        "week": w,
                        "store_id": s,
                        "product_id": "p",
                        "dollar": 10.0,
                        "units": 1.0,
                    }
                    for w in range(1, weeks + 1)
                    for s in ("a", "b")
                ]
            )
            self.frames[version, "warehouse"] = base
            self.frames[version, "report"] = base.groupby("week", as_index=False)[
                ["dollar", "units"]
            ].sum()

    def list_versions(self):
        return ["v1", "v2"]

    def available_stages(self, version):
        return ["warehouse", "report"]

    def read_fact(self, version, stage):
        return self.frames[version, stage].copy()

    def read_dim(self, version, name):
        return None


CONFIG = DatasetConfig(report_grain=(), temporal_enabled=False)


def test_doubled_report_never_passes():
    source = HandSource()
    source.frames["v2", "report"]["dollar"] *= 2

    result = run_qc(source, "v2", "v1", CONFIG)
    assert result.status == "INVESTIGATE"
    assert result.machine["requires_investigation"] is True
    assert result.decisions.requires_investigation is True
    assert any(
        f["outcome"] == "FAIL" and "mass_balance" in f["check"]
        for f in result.machine["findings"]
    )


@pytest.mark.parametrize(
    "column,value", [("dollar", float("inf")), ("week", 1.5), ("store_id", None)]
)
@pytest.mark.parametrize("version", ["v1", "v2"])
def test_both_snapshots_reject_invalid_evidence(column, value, version):
    source = HandSource()
    frame = source.frames[version, "warehouse"]
    if column == "week":
        frame[column] = frame[column].astype(float)
    frame.loc[0, column] = value
    result = run_qc(source, "v2", "v1", CONFIG)
    assert result.status == "DATA_CONTRACT_FAILURE"


def test_reference_offsets_and_missing_coverage():
    source = HandSource()
    frame = source.read_fact("v2", "report")
    reference = frame.copy()
    reference.loc[0, "dollar"] += 5
    reference.loc[1, "dollar"] -= 5
    spec = ReferenceSpec("ref", CONFIG.name, ("dollar",))
    assert compare_reference(frame, reference, spec, CONFIG)["status"] == "MISMATCH"
    assert (
        compare_reference(frame, reference.iloc[:0], spec, CONFIG)["status"]
        == "INCOMPLETE"
    )
    assert (
        compare_reference(frame, frame, replace(spec, dataset="wrong"), CONFIG)[
            "status"
        ]
        == "INCOMPLETE"
    )
    result = run_qc(
        source, "v2", "v1", CONFIG, reference_frame=frame.iloc[:0], reference_spec=spec
    )
    assert result.status == "INCOMPLETE"


def test_versions_are_relative_and_ordered():
    assert select_versions(["0", "1", "2"], "1") == ("1", "0")
    for current, previous in (("1", "1"), ("0", "1"), ("2", "9")):
        with pytest.raises(ValueError):
            select_versions(["0", "1", "2"], current, previous)


def test_mapping_collision_and_config_validation():
    source = MappedSource(HandSource(), {"dollar": "amount"})
    with pytest.raises(ValueError, match="duplicate"):
        source._apply(pd.DataFrame({"amount": [1], "dollar": [2]}))
    for kwargs in (
        {"name": "../escape"},
        {"temporal_min_history": 0},
        {"null_fraction_limit": float("nan")},
    ):
        with pytest.raises(ValueError):
            DatasetConfig(**kwargs)


def test_weekly_recovers_committed_artifacts_and_changes_identity(
    tmp_path, monkeypatch
):
    from deltalake import write_deltalake

    import qc.assessment as assessment
    from qc.weekly import run_weekly

    source = HandSource()
    path = tmp_path / "delta"
    for version in source.list_versions():
        write_deltalake(path, source.read_fact(version, "warehouse"), mode="overwrite")
    out = tmp_path / "reports"
    original = assessment.publish

    def crash(*args):
        raise RuntimeError("injected publication crash")

    monkeypatch.setattr(assessment, "publish", crash)
    with pytest.raises(RuntimeError, match="injected"):
        run_weekly(str(path), CONFIG, out_root=out)
    monkeypatch.setattr(assessment, "publish", original)
    recovered = run_weekly(str(path), CONFIG, out_root=out)
    assert recovered.skipped
    assert recovered.status == "INCOMPLETE"
    with SqliteStore(out / "assessments.sqlite") as store:
        assert store.connection.execute("SELECT count(*) FROM runs").fetchone()[0] == 1
        assert (
            store.connection.execute("SELECT count(*) FROM attempts").fetchone()[0] == 2
        )
    changed = run_weekly(str(path), replace(CONFIG, materiality_abs=3.0), out_root=out)
    assert changed.assessment_id != recovered.assessment_id
    from pathlib import Path

    report = Path(recovered.report_dir) / "report.json"
    report.write_text("corrupted")
    run_weekly(str(path), CONFIG, out_root=out)
    assert json.loads(report.read_text())["status"] == recovered.status


@pytest.mark.parametrize(
    "boundary",
    [
        "after_attempt",
        "before_commit",
        "after_commit",
        "before_publish",
        "after_publish",
    ],
)
def test_weekly_all_publication_boundaries_recover(tmp_path, monkeypatch, boundary):
    from deltalake import write_deltalake

    import qc.weekly as weekly

    source = HandSource()
    path = tmp_path / "delta"
    for version in source.list_versions():
        write_deltalake(path, source.read_fact(version, "warehouse"), mode="overwrite")
    out = tmp_path / "out"

    def checkpoint(name):
        if name == boundary:
            raise RuntimeError("injected " + name)

    monkeypatch.setattr(weekly, "_checkpoint", checkpoint)
    with pytest.raises(RuntimeError, match="injected"):
        weekly.run_weekly(str(path), CONFIG, out_root=out)
    monkeypatch.setattr(weekly, "_checkpoint", lambda name: None)
    result = weekly.run_weekly(str(path), CONFIG, out_root=out)
    assert result.status == "INCOMPLETE"
    with SqliteStore(out / "assessments.sqlite") as store:
        assert store.connection.execute("SELECT count(*) FROM runs").fetchone()[0] == 1
        assert (
            store.connection.execute("SELECT count(*) FROM assessments").fetchone()[0]
            == 1
        )
        assert (
            store.connection.execute("SELECT count(*) FROM attempts").fetchone()[0] == 2
        )


def test_parquet_manifest_weekly_flow(tmp_path):
    from qc.weekly import run_weekly

    source = HandSource()
    snapshots = []
    for index, version in enumerate(source.list_versions()):
        stages = {}
        for stage in source.available_stages(version):
            filename = f"{version}-{stage}.parquet"
            source.read_fact(version, stage).to_parquet(tmp_path / filename)
            stages[stage] = filename
        snapshots.append(
            {
                "version": version,
                "observed_at": f"2026-01-0{index + 1}T00:00:00Z",
                "stages": stages,
            }
        )
    path = tmp_path / "snapshots.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_id": "hand-authored-fixture",
                "snapshots": snapshots,
            }
        )
    )
    result = run_weekly(str(path), CONFIG, out_root=tmp_path / "out")
    assert result.status == "PASS"
    assert run_weekly(str(path), CONFIG, out_root=tmp_path / "out").skipped


def test_missing_configured_reference_and_temporal_failure_need_review(monkeypatch):
    spec = ReferenceSpec(
        reference_id="control", dataset=CONFIG.name, metrics=("dollar",)
    )
    result = run_qc(HandSource(), "v2", "v1", CONFIG, reference_spec=spec)
    assert result.status == "INCOMPLETE"

    def fail(*args):
        raise TimeoutError("forecast deadline")

    monkeypatch.setattr("qc.run.run_temporal_qc", fail)
    result = run_qc(HandSource(), "v2", "v1", replace(CONFIG, temporal_enabled=True))
    assert result.status == "INCOMPLETE"
    assert result.machine["requires_investigation"]
    assert "forecast deadline" in result.temporal.calibration["unavailable_reason"]


def test_explicit_calendars_and_weekday_consistency():
    from qc.source import CachedSource

    source = HandSource()
    for frame in source.frames.values():
        frame["week"] = pd.Timestamp("2026-01-05") + pd.to_timedelta(
            (frame.week - 1) * 7, unit="D"
        )
    result = run_qc(source, "v2", "v1", replace(CONFIG, calendar="weekly_date"))
    assert result.status == "PASS"
    source.frames["v2", "warehouse"]["week"] += pd.Timedelta(days=1)
    cached = CachedSource(source, replace(CONFIG, calendar="weekly_date"))
    cached.read_fact("v1", "warehouse")
    with pytest.raises(ValueError, match="same weekday"):
        cached.read_fact("v2", "warehouse")

    source = HandSource()
    for frame in source.frames.values():
        frame["week"] = frame.week.map(lambda w: f"FY26-P{w}")
    config = replace(
        CONFIG,
        calendar="mapped",
        period_map=tuple((f"FY26-P{w}", w) for w in range(1, 5)),
    )
    assert run_qc(source, "v2", "v1", config).status == "PASS"
    incomplete = replace(config, period_map=config.period_map[:-1])
    assert run_qc(source, "v2", "v1", incomplete).status == "DATA_CONTRACT_FAILURE"


def test_required_dimensions_and_optional_skips():
    required = run_qc(
        HandSource(), "v2", "v1", replace(CONFIG, required_dimensions=("stores",))
    )
    assert required.status == "INCOMPLETE"
    optional = run_qc(HandSource(), "v2", "v1", CONFIG)
    assert optional.status == "PASS"
    assert any(
        f["outcome"] == "UNAVAILABLE" and not f["required"]
        for f in optional.machine["findings"]
    )
    with pytest.raises(ValueError, match="snapshot measures"):
        replace(CONFIG, primary_metric="stock")


def test_delta_future_dimensions_and_stages_are_unavailable(monkeypatch):
    from qc.delta import DeltaSource

    class Table:
        def __init__(self, uri):
            self.uri = uri

        def history(self):
            return [{"version": 0, "timestamp": 100 if self.uri == "fact" else 200}]

        def metadata(self):
            from types import SimpleNamespace

            return SimpleNamespace(id=self.uri)

        def version(self):
            return 0

        def to_pandas(self):
            raise AssertionError("future data must never be read")

    monkeypatch.setattr(
        "qc.delta._delta_table", lambda uri, options, version: Table(uri)
    )
    source = DeltaSource(
        "fact",
        stage_tables={"warehouse": "fact", "report": "future"},
        dim_tables={"stores": "future"},
        version_map={"0": {"warehouse": 0, "report": 0, "dim:stores": 0}},
    )
    assert source.available_stages("0") == ["warehouse"]
    assert source.read_dim("0", "stores") is None
    assert source.snapshot_metadata("0")["table_id"] == "fact"
    source.version_map["0"]["warehouse"] = 1
    with pytest.raises(ValueError, match="primary stage"):
        source.available_stages("0")


def test_future_ratio_approval_cannot_clear_historical_review(monkeypatch):
    from qc.expectations import RatioExpectation
    from qc.reconciliation import run_reconciliation

    def flagged(*args, **kwargs):
        report = run_reconciliation(*args, **kwargs)
        report.ratio_flags.append({"week": 4, "dollar_per_unit": 10.0})
        return report

    monkeypatch.setattr("qc.run.run_reconciliation", flagged)
    approval = RatioExpectation(
        "approved-ratio",
        CONFIG.name,
        "dollar_per_unit",
        9,
        11,
        week=4,
        approved_by="fixture",
        approved_at="2026-02-01T00:00:00Z",
    )
    old = run_qc(
        HandSource(),
        "v2",
        "v1",
        CONFIG,
        expectations=[approval],
        observed_at="2026-01-01",
    )
    assert old.status == "INVESTIGATE"
    later = run_qc(
        HandSource(),
        "v2",
        "v1",
        CONFIG,
        expectations=[approval],
        observed_at="2026-03-01",
    )
    assert later.status == "PASS_WITH_EXPLANATION"
    assert any(
        item["approval_ids"] == ("approved-ratio",)
        for item in later.machine["approval_coverage"]
    )


def test_future_selected_snapshot_is_rejected_before_read():
    class FutureSource(HandSource):
        def snapshot_metadata(self, version):
            return {"committed_at_ms": 2000000000000}

        def read_fact(self, version, stage):
            raise AssertionError("future facts must not be read")

    with pytest.raises(ValueError, match="unavailable at observation cutoff"):
        run_qc(FutureSource(), "v2", "v1", CONFIG, observed_at="2026-01-01")


def test_publication_rejects_unsafe_artifact_paths(tmp_path):
    from qc.assessment import artifact_checksums, publish

    for name in ("../outside.json", "/tmp/outside.json", "..", ""):
        artifacts = {name: "malformed"}
        with pytest.raises(ValueError, match="unsafe artifact name"):
            publish(tmp_path / "report", artifacts, artifact_checksums(artifacts))
    assert not (tmp_path / "outside.json").exists()


def test_missing_all_required_stages_is_incomplete():
    class EmptyStages(HandSource):
        def available_stages(self, version):
            return []

    result = run_qc(EmptyStages(), "v2", "v1", CONFIG)
    assert result.status == "INCOMPLETE"
    assert result.machine["requires_investigation"]
