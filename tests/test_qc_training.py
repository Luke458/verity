from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from qc.config import DatasetConfig
from qc.decisions import CAUSE_VALUES, TrainedDecisionProvider
from qc.labels import LabelStore, build_oracle_labels
from qc.training import (
    FALSE_ALARM_COST,
    MISS_COST,
    WRONG_CAUSE_COST,
    _multiclass,
    cost_matrix,
    evaluate_decision_provider,
    label_metrics,
    pair_cost,
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
    # Reporting smoke check only: most families are group-held-out, so this
    # number carries no accuracy claim; the held-out number is metrics["test"].
    assert 0.0 <= evaluation["overall_accuracy"] <= 1.0


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


# ---------------------------------------------------------------------------
# Cost-weighted scoring and the imbalance-aware metric surface (B4).
# ---------------------------------------------------------------------------


def test_cost_matrix_is_pinned():
    """The declared cost asymmetry is a reviewed decision, not a default.

    Flat accuracy treats a missed actionable cause and a false alarm alike. It
    should not. These values are the project's declared stand-in for measured
    business costs; changing them silently would change which provider
    `qc champion` selects, so every one of them is asserted here.
    """
    assert pair_cost("likely_cause", "CODING", "CODING") == 0.0
    # Miss (real fault reported as UNKNOWN) is the expensive error.
    assert pair_cost("likely_cause", "WAREHOUSE", "UNKNOWN") == 2.0
    # False alarm (UNKNOWN raised as a cause) is the cheap error.
    assert pair_cost("likely_cause", "UNKNOWN", "WAREHOUSE") == 0.5
    # Naming the wrong actionable cause sits between the two.
    assert pair_cost("likely_cause", "CODING", "WAREHOUSE") == 1.0

    assert pair_cost("likely_origin", "SOURCE", "UNKNOWN") == 2.0
    assert pair_cost("likely_origin", "UNKNOWN", "SOURCE") == 0.5

    # Severity is ordered and understating is worse than overstating.
    assert pair_cost("severity", "CRITICAL", "LOW") == 2.0
    assert pair_cost("severity", "LOW", "CRITICAL") == 1.0
    assert pair_cost("severity", "LOW", "MEDIUM") == pytest.approx(1 / 3)
    assert pair_cost("severity", "LOW", "LOW") == 0.0

    assert pair_cost("requires_investigation", "True", "False") == 2.0
    assert pair_cost("requires_investigation", "False", "True") == 0.5

    # The ordering is the point; the numbers merely fix its magnitude.
    assert MISS_COST > WRONG_CAUSE_COST > FALSE_ALARM_COST


def test_cost_matrix_is_square_and_zero_on_the_diagonal():
    matrix = cost_matrix("likely_cause", CAUSE_VALUES)
    assert matrix.shape == (len(CAUSE_VALUES), len(CAUSE_VALUES))
    assert not np.diag(matrix).any()
    assert (matrix >= 0).all()
    # UNKNOWN is the quiet class, so missing it is never the worst error.
    unknown = CAUSE_VALUES.index("UNKNOWN")
    assert matrix[unknown].max() <= matrix[:, unknown].max()


def test_multiclass_metrics_are_reported_per_field(label_records):
    provider, _ = train_decision_provider(label_records, DatasetConfig())
    evaluation = evaluate_decision_provider(provider, label_records)
    for field, item in evaluation["fields"].items():
        for key in (
            "macro_f1",
            "mcc",
            "g_mean",
            "n_mapped",
            "per_class",
            "confusion",
            "mean_cost",
            "cost_score",
        ):
            assert key in item, f"{field} is missing metric {key}"
        if item["n_mapped"]:
            assert item["macro_f1"] is not None
            assert -1.0 <= item["mcc"] <= 1.0
            assert 0.0 <= item["g_mean"] <= 1.0
            assert set(item["per_class"]) == set(provider.heads[field].classes)
            confusion = np.asarray(item["confusion"])
            assert confusion.sum() == item["n_mapped"]
        # Bounded by the declared worst case; a linear head on few records is
        # not perfect and this asserts the metric is well-formed, not that the
        # model is good.
        assert 0.0 <= item["mean_cost"] <= MISS_COST
        assert 0.0 <= item["cost_score"] <= 1.0


def test_cost_charges_unmapped_labels_as_misses(label_records):
    """A truth the head cannot name is the most expensive kind of miss.

    Accuracy already refuses a free pass for unmapped labels; cost must not
    hand one back, or a provider could look cheap by simply never having seen
    a class.
    """
    provider, _ = train_decision_provider(label_records, DatasetConfig())
    unseen = list(label_records)
    unseen[0] = replace(
        label_records[0],
        labels={**label_records[0].labels, "likely_cause": "NOT_A_CLASS"},
    )

    evaluation = evaluate_decision_provider(provider, unseen)
    item = evaluation["fields"]["likely_cause"]
    assert item["n_unmapped"] == 1
    # Every other field on that record is untouched and still mapped.
    assert evaluation["fields"]["likely_origin"]["n_unmapped"] == 0

    # The unmapped label is charged the flat MISS_COST whatever was predicted,
    # on its own and on exactly one case of thirteen. It is not routed through
    # pair_cost, whose outcome would depend on which class the head emitted.
    solo = evaluate_decision_provider(provider, [unseen[0]])
    assert solo["fields"]["likely_cause"]["mean_cost"] == pytest.approx(MISS_COST)
    # The one unmapped case alone contributes MISS_COST / n; the rest is the
    # model's ordinary error on the other twelve.
    assert item["mean_cost"] >= MISS_COST / len(unseen)


def test_gmean_collapses_when_a_class_is_never_found():
    classes = ("a", "b", "c")
    found = _multiclass(np.array([0, 1, 2, 0]), np.array([0, 1, 2, 0]), classes)
    assert found["g_mean"] == pytest.approx(1.0)
    assert found["macro_f1"] == pytest.approx(1.0)
    assert found["mcc"] == pytest.approx(1.0)

    # One real class never predicted: G-mean goes to zero even though accuracy
    # is 2/3 and macro-F1 is still flattering. That is the property we want
    # when a rare but severe defect is in the label set.
    missed = _multiclass(np.array([0, 0, 2, 0]), np.array([0, 1, 2, 0]), classes)
    assert missed["g_mean"] == 0.0
    assert missed["per_class"]["b"]["recall"] == 0.0
    assert missed["per_class"]["b"]["support"] == 1


def test_macro_f1_ignores_unsupported_classes():
    # Class "c" has no support and no predictions: it must not drag macro-F1
    # down to something meaningless.
    classes = ("a", "b", "c")
    result = _multiclass(np.array([0, 1, 1, 0]), np.array([0, 1, 1, 0]), classes)
    assert result["macro_f1"] == pytest.approx(1.0)
    assert result["per_class"]["c"]["support"] == 0


def test_mcc_is_zero_for_a_degenerate_predictor():
    classes = ("a", "b", "c")
    result = _multiclass(np.array([0, 0, 0]), np.array([0, 1, 2]), classes)
    assert result["mcc"] == 0.0


def test_label_metrics_scores_hard_predictions():
    """The same surface must score a rule adapter or a remote provider."""
    perfect = label_metrics("likely_cause", ["A", "B", "C"], ["A", "B", "C"])
    assert perfect["macro_f1"] == pytest.approx(1.0)
    assert perfect["mcc"] == pytest.approx(1.0)
    assert perfect["mean_cost"] == pytest.approx(0.0)
    assert perfect["cost_score"] == pytest.approx(1.0)

    # One real class never found: macro-F1 and MCC both fall, and cost records
    # the miss at its declared weight - spread over all three cases, because
    # mean_cost is a mean.
    partial = label_metrics("likely_cause", ["A", "B", "C"], ["A", "B", "A"])
    assert partial["g_mean"] == 0.0
    assert partial["per_class"]["C"]["recall"] == 0.0
    assert partial["mean_cost"] == pytest.approx(WRONG_CAUSE_COST / 3)
    assert partial["mcc"] < perfect["mcc"]


def test_label_metrics_charges_unanswered_as_misses():
    answered = label_metrics("likely_cause", ["A", "A"], ["A", "A"])
    assert answered["mean_cost"] == pytest.approx(0.0)

    with_unanswered = label_metrics(
        "likely_cause", ["A", "A"], ["A", "A"], unanswered=3
    )
    assert with_unanswered["n"] == 5
    assert with_unanswered["n_unanswered"] == 3
    # Three flat misses over five cases, and the shape metrics see only the
    # two answered ones - so n_unanswered must stay visible beside them.
    assert with_unanswered["mean_cost"] == pytest.approx(3 * MISS_COST / 5)
    assert with_unanswered["macro_f1"] == pytest.approx(1.0)

    only_unanswered = label_metrics("likely_cause", [], [], unanswered=3)
    assert only_unanswered["mean_cost"] == pytest.approx(MISS_COST)
    assert only_unanswered["cost_score"] == pytest.approx(0.0)
    assert only_unanswered["macro_f1"] is None


def test_label_metrics_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        label_metrics("likely_cause", ["A"], ["A", "B"])
