from __future__ import annotations

import json

import pytest

from qc.config import DatasetConfig
from qc.decisions import TrainedDecisionProvider
from qc.labels import LabelStore, build_oracle_labels
from qc.training import (
    evaluate_decision_provider,
    save_training_run,
    train_decision_provider,
)
from qcgen.config import suite_config
from qcgen.scenarios import generate_suite


@pytest.fixture(scope="module")
def label_records(tmp_path_factory):
    suite_dir = generate_suite(
        suite_config("tiny"),
        tmp_path_factory.mktemp("train-suite"),
        13,
        "train",
        ("source", "coded", "warehouse", "report"),
    )
    return build_oracle_labels(suite_dir, DatasetConfig())


def test_oracle_labels_are_versioned(label_records):
    assert len(label_records) == 13
    for record in label_records:
        assert record.source == "oracle"
        assert record.feature_version == 3
        assert record.text and "status=" in record.text
        assert set(record.labels) == {
            "likely_cause",
            "likely_origin",
            "severity",
            "requires_investigation",
        }


def test_train_evaluate_and_reload(label_records, tmp_path):
    provider, metrics = train_decision_provider(
        label_records, DatasetConfig(), validation_fraction=0.3, seed=1
    )
    assert metrics["train"]["n"] >= 1
    assert metrics["validation"]["n"] >= 1
    assert metrics["test"]["n"] >= 1
    # The model memorizes its training families; group-disjoint accuracy is
    # expected to be poor on 13 synthetic records. This test asserts honest
    # reporting, not semantic accuracy.
    assert metrics["train"]["overall_accuracy"] >= 0.9
    assert set(metrics["test"]["fields"]) == {
        "likely_cause",
        "likely_origin",
        "severity",
        "requires_investigation",
    }
    for item in metrics["test"]["fields"].values():
        assert 0.0 <= item["accuracy"] <= 1.0
        assert item["accuracy_ci_low"] <= item["accuracy"] <= item["accuracy_ci_high"]
        assert item["ece"] is None or 0.0 <= item["ece"] <= 1.0

    directory = tmp_path / "provider"
    save_training_run(provider, metrics, directory)
    loaded = TrainedDecisionProvider.load(directory)
    assert loaded.metadata["label_sources"] == ["oracle"]
    assert loaded.metadata["production_eligible"] is False
    assert "independent pinned evaluation" in loaded.metadata["warning"]

    from qc.run import run_qc
    from qcgen.scenarios import build_scenario
    from qcgen.sources import ScenarioSource

    built = build_scenario(
        suite_config("tiny"),
        0,
        tmp_path,
        "missing_stores",
        ("source", "coded", "warehouse", "report"),
    )
    result = run_qc(
        ScenarioSource(built.directory),
        "V0002",
        "V0001",
        DatasetConfig(),
        decision_provider=loaded,
    )
    assert result.decisions.provider == "trained"
    assert result.decisions.get("likely_cause").probability_kind == "temperature_scaled"
    evaluation = evaluate_decision_provider(loaded, label_records)
    # In-sample smoke check only; the honest held-out number is metrics["test"].
    assert evaluation["overall_accuracy"] >= 0.5


def test_grouped_split_keeps_families_together(label_records):
    from qc.training import _group_key, split_records_grouped

    train, validation, test = split_records_grouped(
        label_records, validation_fraction=0.3, test_fraction=0.2, seed=1
    )
    train_keys = {_group_key(record) for record in train}
    validation_keys = {_group_key(record) for record in validation}
    test_keys = {_group_key(record) for record in test}
    assert not train_keys & validation_keys
    assert not train_keys & test_keys
    assert not validation_keys & test_keys
    assert len(train) + len(validation) + len(test) == len(label_records)


def test_feature_version_mismatch_rejected(label_records, tmp_path):
    provider, metrics = train_decision_provider(label_records, DatasetConfig())
    save_training_run(provider, metrics, tmp_path)
    path = tmp_path / "provider.json"
    payload = json.loads(path.read_text())
    payload["feature_version"] = 999
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        TrainedDecisionProvider.load(tmp_path)


def test_label_store_roundtrip(label_records, tmp_path):
    store = LabelStore(tmp_path / "labels.jsonl")
    store.append(label_records)
    loaded = store.load()
    assert len(loaded) == len(label_records)
    assert loaded[0].labels == label_records[0].labels
