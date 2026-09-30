from __future__ import annotations

import hashlib
import json

import pytest

from qc.cohort import (
    DEFAULT_GATES,
    CohortCase,
    CohortPlan,
    _gate_results,
    _metrics,
    run_cohort,
)


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
    assert result.metrics["false_positive_rate"] is None
    assert result.metrics["contract_failure_rate"] is None
    assert len(result.code_sha256) == 64
    assert result.plan_sha256
    # Detection, FPR and lineage gates; FPR is INSUFFICIENT_EVIDENCE without
    # controls, so two faults and no controls cannot qualify.
    assert [check["gate"] for check in result.gate_results] == [
        "min_detection_rate",
        "max_false_positive_rate",
        "min_lineage_first_divergence_accuracy",
    ]
    assert result.gate_results[1]["status"] == "INSUFFICIENT_EVIDENCE"
    assert result.gates_passed is False
    assert "oracle_disagreements" in result.metrics["common_gates"]
    # Cause labels are scored against the oracle for every fault case.
    cause = result.metrics["labels"]["cause"]
    assert cause["n"] == 2
    assert set(cause["by_family"]) == {"missing_stores", "coding_error"}
    assert cause["by_family"]["missing_stores"]["accuracy"] == 1.0
    assert any("Synthetic" in limitation for limitation in result.limitations)

    payload = json.loads((tmp_path / "out" / "cohort.json").read_text())
    assert payload["metrics"]["cases"] == 2
    lines = (tmp_path / "out" / "cases.jsonl").read_text().strip().splitlines()
    assert len(lines) == 4
    assert {json.loads(line)["split"] for line in lines} == {"dev", "heldout"}


def test_gates_must_be_declared_and_complete():
    with pytest.raises(ValueError):
        CohortPlan(gates={})
    with pytest.raises(ValueError):
        CohortPlan(gates={"min_detection_rate": 0.9})
    with pytest.raises(ValueError):
        CohortPlan(gates={**DEFAULT_GATES, "min_detection_rate": 1.5})


def test_gate_failure_is_reported():
    metrics = {
        "detection_rate": 0.5,
        "false_positive_rate": 0.0,
        "mean_reconstruction_score": 0.0,
        "lineage_first_divergence_accuracy": 0.0,
        "lineage_comparable": 0,
    }
    checks = _gate_results(metrics, DEFAULT_GATES)
    assert checks
    assert any(not check["passed"] for check in checks)


def _case(**overrides) -> CohortCase:
    data = {
        "case_id": "c",
        "split": "heldout",
        "seed": 1,
        "family": "missing_stores",
        "is_control": False,
        "engine_status": "INVESTIGATE",
        "historical_status": "PASS",
        "expected_status": "INVESTIGATE",
        "expected_class": "missing_stores",
        "injection_stage": "source",
        "first_divergence": "source",
        "reconstruction_score": 1.0,
        "explained_fraction": 1.0,
        "detected": True,
        "expected_match": True,
        "false_positive": False,
    }
    data.update(overrides)
    return CohortCase(**data)


def test_false_positive_uses_full_engine_status():
    # A control escalated only by the latest-week detector still counts as a
    # false positive; the historical status alone would hide it.
    control = _case(
        is_control=True,
        engine_status="INVESTIGATE",
        historical_status="PASS",
        expected_status="INVESTIGATE",
        detected=True,
        false_positive=True,
    )
    metrics = _metrics([control])
    assert metrics["false_positive_rate"] == 1.0
    assert metrics["control_status_counts"] == {"INVESTIGATE": 1}

    clean = _case(
        is_control=True,
        engine_status="PASS",
        historical_status="PASS",
        expected_status="INVESTIGATE",
        detected=False,
        false_positive=False,
    )
    assert _metrics([clean])["false_positive_rate"] == 0.0


def test_plan_hash_pinning(tmp_path):
    plan = CohortPlan(
        families=("missing_stores",),
        controls=(),
        scenarios_per_family=1,
        dev_seeds=(1,),
        heldout_seeds=(2,),
    )
    plan_path = tmp_path / "plan.json"
    plan.save(plan_path)
    sha = hashlib.sha256(
        json.dumps(plan.to_dict(), sort_keys=True).encode()
    ).hexdigest()
    (tmp_path / "plan.json.sha256").write_text(sha + "\n")

    result = run_cohort(plan, workdir=tmp_path / "w1", plan_path=plan_path)
    assert result.plan_hash_verified is True

    tampered = CohortPlan(
        families=("coding_error",),
        controls=(),
        scenarios_per_family=1,
        dev_seeds=(1,),
        heldout_seeds=(2,),
    )
    tampered.save(plan_path)
    with pytest.raises(ValueError, match="does not match"):
        run_cohort(tampered, workdir=tmp_path / "w2", plan_path=plan_path)


def test_controls_can_be_sized_independently():
    plan = CohortPlan.from_dict(
        {**CohortPlan().to_dict(), "scenarios_per_family": 1, "scenarios_per_control": 20}
    )
    assert plan.scenarios_per_control == 20
    with pytest.raises(ValueError):
        CohortPlan(scenarios_per_control=0)


def test_cli_cohort_exits_nonzero_when_a_gate_fails(tmp_path, capsys):
    # Regression: the command exited 0 whatever the gates said, and crashed
    # formatting an INSUFFICIENT_EVIDENCE gate whose actual value is None.
    from qc.cli import main

    plan = CohortPlan(
        families=("missing_stores",),
        controls=(),
        dev_seeds=(1,),
        heldout_seeds=(2,),
        gates={**DEFAULT_GATES, "min_lineage_first_divergence_accuracy": 0.0},
    )
    path = tmp_path / "plan.json"
    plan.save(path)
    code = main(["cohort", "--plan", str(path), "--workdir", str(tmp_path / "work")])
    assert code == 3
    assert "n/a" in capsys.readouterr().out


def test_label_accuracy_counts_only_cases_with_ground_truth():
    from qc.cohort import _label_accuracy

    cases = [
        _case(expected_cause="CODING", predicted_cause="CODING"),
        _case(expected_cause="CODING", predicted_cause="WAREHOUSE", family="coding_error"),
        _case(expected_cause=None, predicted_cause="UNKNOWN", family="clean"),
    ]
    scores = _label_accuracy(cases, "expected_cause", "predicted_cause")
    assert scores["n"] == 2
    assert scores["accuracy"] == 0.5
    assert scores["by_family"]["coding_error"]["predicted"] == {"WAREHOUSE": 1}
    assert "clean" not in scores["by_family"]
