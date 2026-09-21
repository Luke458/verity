"""Independent evaluation protocol tests, not model accuracy evidence."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from qc.assessment import digest
from qc.decisions import (
    DecisionSet,
    DecisionValue,
    RuleDecisionProvider,
    default_fields,
)
from qc.evidence_text import EVIDENCE_TEXT_VERSION
from qc.policy import MACHINE_SCHEMA_VERSION
from qc.run import run_qc
from qc.store_cohort import evaluate_frozen_cohort, production_eligibility
from tests.test_reliability import CONFIG, HandSource


class FixtureProvider:
    artifact_identity = {"test_fixture": "perfect-fields-v1"}

    def decide(self, result):
        index = int(result.run_id)
        fields = {}
        for spec in default_fields():
            label = spec.values[index % len(spec.values)]
            fields[spec.name] = DecisionValue(
                spec.name,
                spec.kind,
                label == "True" if spec.kind == "boolean" else label,
                {k: float(k == label) for k in spec.values},
                "test_fixture",
                "fixture",
            )
        return DecisionSet(result.run_id, "test_fixture", fields, bool(index % 2))


def manifest():
    cases = []
    for split in ("development", "test"):
        for index in range(100):
            labels = {s.name: s.values[index % len(s.values)] for s in default_fields()}
            labels["root_cause"] = labels.pop("likely_cause")
            labels["requires_investigation"] = bool(index % 2)
            cases.append(
                {
                    "run_id": str(index),
                    "group": f"{split}:{index}",
                    "split": split,
                    "labels": labels,
                    "features": [],
                    "text": "unit fixture",
                    "evidence": {
                        "status": "INVESTIGATE" if index % 2 else "PASS",
                        "decision": {"requires_investigation": bool(index % 2)},
                    },
                }
            )
    value = {"cases": cases, "source": "store", "status": "FROZEN"}
    value["manifest_hash"] = digest(value)
    return value


def test_pinned_development_then_test_and_calibration_diagnostics():
    cohort = manifest()
    provider = FixtureProvider()
    development = evaluate_frozen_cohort(cohort, provider)
    assert development["gates_passed"]
    assert not development["evaluation_complete"]
    for field in development["fields"].values():
        assert field["brier"] == 0
        assert field["log_loss"] == 0
        assert field["expected_calibration_error"] == 0
        assert sum(bin_["n"] for bin_ in field["calibration_bins"]) == 100
    with pytest.raises(ValueError, match="development selection"):
        evaluate_frozen_cohort(cohort, provider, "test")
    forged = {**development, "review_volume": -1}
    with pytest.raises(ValueError, match="development selection"):
        evaluate_frozen_cohort(cohort, provider, "test", forged)
    test = evaluate_frozen_cohort(cohort, provider, "test", development)
    assert test["evaluation_complete"] and test["gates_passed"]
    assert test["paired_incident_accuracy_delta_95ci"] == [0, 0]
    assert not test["production_eligible"]
    assert not production_eligibility(test, {})["production_eligible"]
    synthetic = {**test, "provenance": "synthetic"}
    assert not production_eligibility(synthetic, {})["production_eligible"]


def test_sparse_classes_or_provider_failure_never_pass():
    cohort = manifest()
    cohort["cases"] = cohort["cases"][:2]
    cohort["manifest_hash"] = digest(
        {k: v for k, v in cohort.items() if k != "manifest_hash"}
    )
    report = evaluate_frozen_cohort(cohort, FixtureProvider())
    assert report["status"] == "INSUFFICIENT_EVIDENCE"
    assert not report["gates_passed"]

    class Unavailable(FixtureProvider):
        def decide(self, result):
            raise TimeoutError("unavailable")

    report = evaluate_frozen_cohort(manifest(), Unavailable())
    assert report["abstentions"] == 100
    assert not report["gates_passed"]
    assert report["fields"]["severity"]["brier"] is None


def test_rules_use_identical_frozen_evidence():
    source = HandSource()
    source.frames["v2", "warehouse"].loc[0, "dollar"] += 50
    result = run_qc(source, "v2", "v1", CONFIG)
    frozen = SimpleNamespace(run_id=result.run_id, machine=deepcopy(result.machine))
    provider = RuleDecisionProvider(CONFIG)
    assert provider.decide(frozen).to_dict() == provider.decide(result).to_dict()
    assert provider.to_dict()["engine_hash"]


def test_frozen_training_keeps_test_untouched_and_records_fit_groups():
    from qc.decisions import FeatureEncoder
    from qc.store_cohort import training_records_from_cohort
    from qc.training import split_records_four, train_decision_provider

    result = run_qc(HandSource(), "v2", "v1", CONFIG)
    cases = []
    for index, split in enumerate(("train", "calibration", "development", "test")):
        cases.append(
            {
                "run_id": f"fixture-{index}",
                "group": f"incident-{index}",
                "split": split,
                "observed_at": f"2026-01-0{index + 1}T00:00:00+00:00",
                "features": FeatureEncoder(CONFIG).encode(result).tolist(),
                "feature_version": FeatureEncoder(CONFIG).feature_version,
                "evidence_version": MACHINE_SCHEMA_VERSION,
                "text_version": EVIDENCE_TEXT_VERSION,
                "text": "unit fixture",
                "labels": {
                    "root_cause": "UNKNOWN",
                    "likely_origin": "UNKNOWN",
                    "severity": "LOW",
                    "requires_investigation": False,
                },
            }
        )
    frozen = {"status": "FROZEN", "cases": cases}
    frozen["manifest_hash"] = digest(frozen)
    records = training_records_from_cohort(frozen)
    assert [
        bucket[0].metadata["frozen_split"] for bucket in split_records_four(records)
    ] == ["train", "calibration", "development", "test"]
    provider, metrics = train_decision_provider(records, CONFIG, epochs=1)
    assert metrics["test"]["status"] == "DEFERRED"
    assert metrics["test"]["overall_accuracy"] is None
    assert provider.metadata["cohort_hash"] == frozen["manifest_hash"]
    assert provider.metadata["fit_groups"] == ["incident-0", "incident-1"]
    assert not provider.metadata["production_eligible"]
    records[-1].metadata["incident_group"] = "incident-0"
    with pytest.raises(ValueError, match="overlap"):
        split_records_four(records)


def test_repeated_snapshots_cannot_inflate_confidence_bounds():
    cohort = manifest()
    for case in cohort["cases"]:
        case["group"] = (
            "one-fault" if case["labels"]["requires_investigation"] else "one-control"
        )
    cohort["manifest_hash"] = digest(
        {k: v for k, v in cohort.items() if k != "manifest_hash"}
    )
    report = evaluate_frozen_cohort(cohort, FixtureProvider())
    assert report["fault_groups"] == report["control_groups"] == 1
    assert report["detection_95ci"][0] < 0.9
    assert not report["gates_passed"]
