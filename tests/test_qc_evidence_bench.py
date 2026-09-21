from __future__ import annotations

import json

from qc.cli import main
from qc.evidence_bench import run_evidence_bench
from qcgen.config import suite_config
from qcgen.scenarios import generate_suite


def test_evidence_bench_runs_frozen_thresholds(tmp_path):
    suite_dir = generate_suite(
        suite_config("tiny"),
        tmp_path,
        3,
        "bench",
        ("source", "coded", "warehouse", "report"),
    )
    result = run_evidence_bench(suite_dir, thresholds=(0.001,))

    assert result.plan_sha256
    assert result.thresholds == [0.001]
    assert result.ablation["levels"] == [
        "summary",
        "forecast",
        "hierarchy",
        "complete",
    ]
    # A suite too small to establish the registered gates must stay visibly
    # unselected and unqualified rather than falling back to a candidate.
    assert result.report["selection_status"] == "NO_SAFE_CANDIDATE"
    assert result.selection["selected"] is None
    assert result.gates["status"] == "INSUFFICIENT_EVIDENCE"
    bound = result.gates["gates"]["false_clearance"]["bound"]
    assert bound["status"] == "INSUFFICIENT_EVIDENCE"
    assert result.ablation["mode"] == "provider_rerun"
    assert result.ablation["baseline"]["label"] == "rule_baseline"
    assert result.plan_manifest["oracle_labels"]
    assert result.qualification["provenance"] == "synthetic"
    assert result.qualification["status"] == "UNQUALIFIED"
    assert result.to_dict()["schema_version"] == 3
    assert result.report["qualification_status"] == "UNQUALIFIED"


def test_evidence_bench_cli_writes_report(tmp_path, capsys):
    suite_dir = generate_suite(
        suite_config("tiny"),
        tmp_path,
        2,
        "bench-cli",
        ("source", "coded", "warehouse", "report"),
    )
    out = tmp_path / "bench.json"
    code = main(
        [
            "evidence-bench",
            "--suite",
            str(suite_dir),
            "--thresholds",
            "0.001",
            "--out",
            str(out),
            "--json",
        ]
    )
    assert code in (0, 2)
    payload = json.loads(out.read_text())
    assert payload["suite"] == str(suite_dir)
    assert payload["thresholds"] == [0.001]
    printed = json.loads(capsys.readouterr().out)
    assert printed["plan_sha256"] == payload["plan_sha256"]
