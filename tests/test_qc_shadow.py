from __future__ import annotations

from qc.shadow import run_shadow
from qcgen.config import suite_config
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
    summary = run_shadow(suite_dir, out_dir=out_dir)

    assert summary["scenarios"] == 5
    assert summary["detection_rate"] == 1.0
    # The clean control passes end to end: no false positive at any layer.
    assert summary["false_positive_rate"] == 0.0
    assert summary["latest_week_control_rate"] == 0.0
    assert summary["reconciliation_failures"] == 0
    assert summary["lineage_first_divergence_accuracy"] == 1.0
    assert summary["lineage_comparable"] >= 2
    assert summary["reconstruction_evaluated"] >= 2
    assert summary["engine_status_counts"].get("PASS", 0) >= 1

    assert (out_dir / "shadow.json").exists()
    assert (out_dir / "shadow.jsonl").exists()
    assert len((out_dir / "shadow.jsonl").read_text().strip().splitlines()) == 5
