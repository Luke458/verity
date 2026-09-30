"""Unit tests for decomposed decision fields (docs/next-phase-plan.md, B1).

The decomposition changes the target formulation only. These tests pin the
parts that are easy to get subtly wrong and expensive to debug later: the
question schema is total and stable, the ordinal reconstruction is a valid
distribution even when the independent binary heads invert adjacent
thresholds, and no new supervision is invented.
"""

from __future__ import annotations

import json

import pytest

from qc.config import DatasetConfig
from qc.decisions import field_index
from qc.decomposition import (
    BINARY_CLASSES,
    DECOMPOSITION_VERSION,
    BinaryQuestion,
    DecomposedProvider,
    decomposition_questions,
    evaluate_decomposed,
    question_targets,
    reconstruct_field,
    train_decomposed_provider,
)
from qc.labels import build_oracle_labels
from qcgen.config import suite_config
from qcgen.scenarios import generate_suite


@pytest.fixture(scope="module")
def label_records(tmp_path_factory):
    suite_dir = generate_suite(
        suite_config("tiny"),
        tmp_path_factory.mktemp("decomposition-suite"),
        13,
        "decomposition",
        ("source", "coded", "warehouse", "report"),
    )
    return build_oracle_labels(suite_dir, DatasetConfig())


def test_question_schema_is_total_and_stable():
    """The same fields must always yield the same question ids and order."""
    questions = decomposition_questions()
    again = decomposition_questions()
    assert [question.question_id for question in questions] == [
        question.question_id for question in again
    ]
    assert len(questions) == len({question.question_id for question in questions})

    specs = field_index()
    covered = {question.field for question in questions}
    assert covered == set(specs)
    # Nominal fields decompose one-vs-rest; the ordinal field decomposes into
    # thresholds above the lowest level; the boolean field is already narrow.
    for spec in specs.values():
        mine = [question for question in questions if question.field == spec.name]
        if spec.kind == "choice":
            assert [question.target for question in mine] == list(spec.values)
            assert all(question.kind == "one_vs_rest" for question in mine)
        elif spec.kind == "score":
            assert [question.target for question in mine] == list(spec.values[1:])
            assert all(question.kind == "threshold" for question in mine)
        else:
            assert len(mine) == 1
            assert mine[0].kind == "boolean"


def test_ordinal_reconstruction_is_a_valid_distribution():
    questions = decomposition_questions()
    spec = field_index()["severity"]
    ordered = [question.question_id for question in questions if question.field == "severity"]

    coherent = dict(zip(ordered, [0.9, 0.6, 0.1], strict=True))
    distribution = reconstruct_field(spec, coherent, questions)
    assert distribution == {
        "LOW": pytest.approx(0.1),
        "MEDIUM": pytest.approx(0.3),
        "HIGH": pytest.approx(0.5),
        "CRITICAL": pytest.approx(0.1),
    }
    assert sum(distribution.values()) == pytest.approx(1.0)


def test_inverted_thresholds_still_reconstruct_validly():
    """Independent binary heads can invert adjacent thresholds.

    The reconstruction must remain a probability distribution rather than
    producing negative level probabilities.
    """
    questions = decomposition_questions()
    spec = field_index()["severity"]
    ordered = [question.question_id for question in questions if question.field == "severity"]
    inverted = dict(zip(ordered, [0.2, 0.7, 0.5], strict=True))

    distribution = reconstruct_field(spec, inverted, questions)
    assert all(value >= 0.0 for value in distribution.values())
    assert sum(distribution.values()) == pytest.approx(1.0)
    # Cumulative-minimum projection collapses the inversion rather than
    # flipping the ordering it cannot justify.
    assert distribution["CRITICAL"] == pytest.approx(0.2)
    assert distribution["LOW"] == pytest.approx(0.8)


def test_nominal_reconstruction_normalizes():
    questions = decomposition_questions()
    spec = field_index()["likely_cause"]
    distribution = reconstruct_field(spec, {"likely_cause=CODING": 0.3}, questions)
    assert sum(distribution.values()) == pytest.approx(1.0)

    # Nothing expressed at all resolves to a uniform spread, not to a guess.
    flat = reconstruct_field(spec, {}, questions)
    assert all(value == pytest.approx(1 / len(spec.values)) for value in flat.values())


def test_targets_project_existing_labels_without_inventing_any():
    """Supervision comes from the four fields we already label. Nothing new."""
    questions = decomposition_questions()
    labels = {
        "likely_cause": "WAREHOUSE",
        "likely_origin": "WAREHOUSE",
        "severity": "HIGH",
        "requires_investigation": "True",
    }
    targets = question_targets(labels, questions)
    assert set(targets) == {question.question_id for question in questions}
    positives = {key for key, value in targets.items() if value == "True"}
    assert positives == {
        "likely_cause=WAREHOUSE",
        "likely_origin=WAREHOUSE",
        "severity>=MEDIUM",
        "severity>=HIGH",
        "requires_investigation",
    }
    assert set(targets.values()) <= {"True", "False"}


def test_threshold_targets_are_monotone_by_construction():
    """A higher severity must imply every lower threshold."""
    questions = decomposition_questions()
    for level in ("LOW", "MEDIUM", "HIGH", "CRITICAL"):
        targets = question_targets(
            {"severity": level, "likely_cause": "CODING", "requires_investigation": "False"},
            questions,
        )
        flags = [
            targets[f"severity>={step}"] for step in ("MEDIUM", "HIGH", "CRITICAL")
        ]
        truth = [flag == "True" for flag in flags]
        assert truth == sorted(truth, reverse=True), f"non-monotone at {level}"


def test_train_decompose_and_reload(label_records, tmp_path):
    provider, metrics = train_decomposed_provider(
        label_records, DatasetConfig(), validation_fraction=0.3, test_fraction=0.2, seed=0
    )
    assert len(provider.heads) == len(provider.questions) == 22
    assert all(head.classes == BINARY_CLASSES for head in provider.heads.values())
    for split in ("train", "validation", "test"):
        assert metrics[split]["n"] >= 0
        assert set(metrics[split]["fields"]) == {
            "likely_cause",
            "likely_origin",
            "severity",
            "requires_investigation",
        }

    # Every field distribution is a probability distribution.
    evaluation = evaluate_decomposed(provider, label_records)
    for field_name, item in evaluation["fields"].items():
        assert 0.0 <= item["accuracy"] <= 1.0
        assert item["macro_f1"] is None or 0.0 <= item["macro_f1"] <= 1.0
        assert item["mean_cost"] is None or 0.0 <= item["mean_cost"]
        assert field_name in provider.heads or field_name

    payload = json.loads(json.dumps(provider.to_dict(), default=str))
    (tmp_path / "provider.json").write_text(json.dumps(payload))
    loaded = DecomposedProvider.load(tmp_path)
    assert [question.question_id for question in loaded.questions] == [
        question.question_id for question in provider.questions
    ]
    assert set(loaded.heads) == set(provider.heads)


def test_decomposed_version_mismatch_rejected(label_records, tmp_path):
    provider, _ = train_decomposed_provider(label_records, DatasetConfig())
    payload = provider.to_dict()
    payload["decomposition_version"] = DECOMPOSITION_VERSION + 1
    (tmp_path / "provider.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="decomposition version"):
        DecomposedProvider.load(tmp_path)


def test_feature_version_mismatch_rejected(label_records, tmp_path):
    provider, _ = train_decomposed_provider(label_records, DatasetConfig())
    payload = provider.to_dict()
    payload["feature_version"] = 999
    (tmp_path / "provider.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="feature version"):
        DecomposedProvider.load(tmp_path)


def test_question_columns_follow_question_order():
    """The reconstruction must not depend on question iteration order."""
    questions = decomposition_questions()
    spec = field_index()["likely_cause"]
    probabilities = {
        question.question_id: (0.9 if question.target == spec.values[0] else 0.05)
        for question in questions
    }
    shuffled = dict(reversed(list(probabilities.items())))
    assert reconstruct_field(spec, probabilities, questions) == reconstruct_field(
        spec, shuffled, questions
    )
    assert questions[0].question_id == f"{spec.name}={spec.values[0]}"


def test_binary_questions_are_well_formed():
    for question in decomposition_questions():
        assert isinstance(question, BinaryQuestion)
        assert question.question_id
        assert question.instructions.endswith("?")
        assert question.positive and question.negative
        assert question.field in field_index()
