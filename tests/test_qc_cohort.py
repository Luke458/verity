from __future__ import annotations

import json

import pytest

from qc.cohort import CohortPlan, run_cohort


def test_cohort_plan_rejects_overlapping_seeds():
    with pytest.raises(ValueError):
        CohortPlan(dev_seeds=(1,), heldout_seeds=(1,))


def test_small_cohort_end_to_end(tmp_path):
    plan = CohortPlan(
        families=("missing_stores", "coding_error"),
        controls=(),
        scenarios_per_family=1,
        dev_seeds=(1,),
        heldout_seeds=(2,),
        gates={
            "min_detection_rate": 0.9,
            "max_false_positive_rate": 0.1,
            "min_lineage_first_divergence_accuracy": 0.0,
        },
    )
    result = run_cohort(plan, workdir=tmp_path / "work", out_dir=tmp_path / "out")

    assert result.metrics["cases"] == 2  # held-out split only
    assert result.metrics["dev"]["cases"] == 2
    assert result.metrics["fault_cases"] == 2
    assert result.metrics["detection_rate"] == 1.0
    assert result.metrics["false_positive_rate"] == 0.0
    assert result.metrics["contract_failure_rate"] == 0.0
    assert len(result.code_sha256) == 64
    assert result.plan_sha256
    assert result.production_eligible is False
    assert len(result.gate_results) == 3
    assert result.gates_passed is True
    assert any("Synthetic" in limitation for limitation in result.limitations)

    payload = json.loads((tmp_path / "out" / "cohort.json").read_text())
    assert payload["metrics"]["cases"] == 2
    lines = (tmp_path / "out" / "cases.jsonl").read_text().strip().splitlines()
    assert len(lines) == 4
    assert {json.loads(line)["split"] for line in lines} == {"dev", "heldout"}


def test_cohort_gate_failure_is_reported(tmp_path):
    plan = CohortPlan(
        families=("missing_stores",),
        controls=(),
        scenarios_per_family=1,
        dev_seeds=(1,),
        heldout_seeds=(2,),
        gates={"min_detection_rate": 1.01},
    )
    result = run_cohort(plan, workdir=tmp_path / "work")
    assert result.gates_passed is False
    assert result.gate_results[0]["passed"] is False
