from __future__ import annotations

import shutil

import pytest

from qc.labels import LabelStore
from qc.prequential import PrequentialStore
from qc.shadow import run_shadow
from qcgen.config import suite_config
from qcgen.oracle_vault import default_oracle_root
from qcgen.scenarios import generate_suite


def test_shadow_over_generated_suite(tmp_path):
    suite_dir = generate_suite(
        suite_config("tiny"),
        tmp_path,
        5,
        "shadow",
        ("source", "coded", "warehouse", "report"),
    )
    out_dir = tmp_path / "out"
    calibration_path = tmp_path / "prequential.jsonl"
    summary = run_shadow(
        suite_dir,
        out_dir=out_dir,
        prequential_store=calibration_path,
    )

    assert summary["scenarios"] == 5
    assert summary["detection_rate"] == 1.0
    assert summary["false_positive_rate"] == 0.0
    assert summary["latest_week_control_rate"] == 1.0
    assert summary["reconciliation_failures"] == 0
    assert summary["lineage_first_divergence_accuracy"] == 1.0
    assert summary["lineage_comparable"] >= 2
    assert summary["reconstruction_evaluated"] >= 2
    assert summary["engine_status_counts"].get("PASS", 0) == 0

    assert (out_dir / "shadow.json").exists()
    assert (out_dir / "shadow.jsonl").exists()
    assert len((out_dir / "shadow.jsonl").read_text().strip().splitlines()) == 5

    pool = PrequentialStore(calibration_path).pool()
    assert pool.records
    assert all(record.available_on == record.target_week for record in pool.records)


def test_labels_out_writes_real_oracle_labels(tmp_path):
    """--labels-out means oracle labels, so they must carry oracle truth."""
    suite_dir = generate_suite(
        suite_config("tiny"), tmp_path, 5, "labels", ("source", "report")
    )
    labels_path = tmp_path / "labels.jsonl"
    run_shadow(suite_dir, out_dir=tmp_path / "out", labels_out=labels_path)

    records = LabelStore(labels_path).load()
    assert len(records) == 5
    for record in records:
        assert record.source == "oracle"
        assert record.family, "oracle labels must name the fault family"
        assert record.labels["likely_cause"] != "UNKNOWN"
        assert record.text


def test_labels_out_refuses_a_missing_oracle_vault(tmp_path):
    """Placeholders must never be tagged as ground truth.

    Without an oracle payload the engine cannot know the true cause. Writing
    UNKNOWN labels with provenance "oracle" would let `qc train` fit a
    provider to noise and report it as trained on ground truth, so the write
    fails instead.
    """
    suite_dir = generate_suite(
        suite_config("tiny"), tmp_path, 5, "nolabels", ("source", "report")
    )
    shutil.rmtree(default_oracle_root(suite_dir))

    with pytest.raises(ValueError, match="oracle payload"):
        run_shadow(
            suite_dir,
            out_dir=tmp_path / "out",
            labels_out=tmp_path / "labels.jsonl",
        )
    assert not (tmp_path / "labels.jsonl").exists()
