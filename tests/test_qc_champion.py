from __future__ import annotations

import json

import numpy as np
import pytest

from qc.champion import (
    _accuracy,
    eval_cases_from_suite,
    run_champion,
    write_champion_report,
)
from qc.cli import main
from qc.config import DatasetConfig
from qc.labels import LabelStore, build_oracle_labels, records_from_store
from qc.run import run_qc
from qc.store import SqliteStore
from qcgen.config import suite_config
from qcgen.scenarios import build_scenario, generate_suite
from qcgen.sources import ScenarioSource


class FakeEmbedder:
    name = "fake"

    def __init__(self) -> None:
        self.dimension = 6

    def metadata(self) -> dict:
        return {
            "kind": "hf_encoder",
            "name": self.name,
            "model_id": "fake-model",
            "revision": "test",
            "max_length": 64,
            "dimension": self.dimension,
        }

    def embed(self, texts):
        vectors = []
        for text in texts:
            lowered = text.lower()
            vector = np.zeros(6)
            for token, index in (
                ("missing", 0),
                ("store", 1),
                ("product", 2),
                ("coding", 3),
                ("contract", 4),
                ("backfill", 5),
            ):
                if token in lowered:
                    vector[index] += 1.0
            vectors.append(vector)
        return np.asarray(vectors)


@pytest.fixture(scope="module")
def cohorts(tmp_path_factory):
    root = tmp_path_factory.mktemp("champion")
    stages = ("source", "coded", "warehouse", "report")
    train_dir = generate_suite(
        suite_config("tiny", seed=11), root, 4, "train", stages
    )
    eval_dir = generate_suite(
        suite_config("tiny", seed=21), root, 2, "eval", stages
    )
    return build_oracle_labels(train_dir), eval_cases_from_suite(eval_dir), eval_dir


def _permissive_gates() -> dict[str, float]:
    return {"min_overall_accuracy": 0.0, "min_cause_accuracy": 0.0}


def test_partial_gates_rejected(cohorts):
    records, items, _ = cohorts
    with pytest.raises(ValueError, match="complete"):
        run_champion(
            records, items, DatasetConfig(), gates={"min_overall_accuracy": 0.0}
        )


def test_paired_test_flags_ties_and_separations():
    from qc.champion import ProviderScore, _paired_p_value

    a = ProviderScore(
        "a",
        True,
        correct_by_case={"c1": 1, "c2": 1},
        total_by_case={"c1": 1, "c2": 1},
    )
    b = ProviderScore(
        "b",
        True,
        correct_by_case={"c1": 0, "c2": 0},
        total_by_case={"c1": 1, "c2": 1},
    )
    assert _paired_p_value(a, a) == 1.0
    assert _paired_p_value(a, b) == pytest.approx(0.5)
    c = ProviderScore(
        "c",
        True,
        correct_by_case={f"c{i}": 1 for i in range(10)},
        total_by_case={f"c{i}": 1 for i in range(10)},
    )
    d = ProviderScore(
        "d",
        True,
        correct_by_case={f"c{i}": 0 for i in range(10)},
        total_by_case={f"c{i}": 1 for i in range(10)},
    )
    assert _paired_p_value(c, d) < 0.01


def test_accuracy_handles_fields_with_no_correct_predictions():
    assert _accuracy(
        {"likely_cause": 2}, {"likely_cause": 2, "severity": 1}
    ) == {"likely_cause": 1.0, "severity": 0.0}


def test_bakeoff_reports_every_provider(cohorts):
    records, items, _ = cohorts
    result = run_champion(
        records, items, DatasetConfig(), gates=_permissive_gates()
    )
    by_name = {score.provider: score for score in result.scores}
    assert set(by_name) == {"rule", "feature_head", "text_probe", "remote"}
    # The paired test may find the two leaders statistically inseparable on
    # this tiny cohort; that is an honest "no champion", not a failure.
    assert result.champion in {None, "rule", "feature_head"}
    if result.champion is None:
        assert result.selection["p_value"] > 0.05
    assert result.production_eligible is False
    assert result.provenance["labels"] == "oracle"
    assert by_name["rule"].available
    assert by_name["rule"].cases == len(items)
    assert by_name["feature_head"].available
    assert not by_name["text_probe"].available
    assert "no text embedder" in by_name["text_probe"].note
    assert not by_name["remote"].available
    json.dumps(result.to_dict(), default=str)


def test_text_probe_substrate_is_scored(cohorts):
    records, items, _ = cohorts
    result = run_champion(
        records,
        items,
        DatasetConfig(),
        text_embedder=FakeEmbedder(),
        gates=_permissive_gates(),
    )
    by_name = {score.provider: score for score in result.scores}
    assert by_name["text_probe"].available
    assert by_name["text_probe"].cases == len(items)
    assert 0.0 <= by_name["text_probe"].overall_accuracy <= 1.0


def test_overlapping_train_and_eval_rejected(cohorts, tmp_path):
    _, items, _ = cohorts
    overlap_dir = tmp_path / "overlap"
    generate_suite(
        suite_config("tiny", seed=21),
        overlap_dir,
        2,
        "eval",
        ("source", "coded", "warehouse", "report"),
    )
    records = build_oracle_labels(overlap_dir / "eval")
    with pytest.raises(ValueError):
        run_champion(records, items, DatasetConfig())


def test_champion_report_written_once(cohorts, tmp_path):
    records, items, _ = cohorts
    result = run_champion(records, items, gates=_permissive_gates())
    paths = write_champion_report(result, tmp_path / "report")
    assert paths["json"].exists() and paths["markdown"].exists()
    payload = json.loads(paths["json"].read_text())
    assert "champion" in payload
    assert "Provider champion selection" in paths["markdown"].read_text()
    with pytest.raises(FileExistsError):
        write_champion_report(result, tmp_path / "report")


def test_store_labels_are_confirmed_only(tmp_path):
    stages = ("source", "coded", "warehouse", "report")
    confirmed_case = build_scenario(
        suite_config("tiny"), 0, tmp_path, "missing_stores", stages
    )
    confirmed_result = run_qc(
        ScenarioSource(confirmed_case.directory), "V0002", "V0001", run_id="case-1"
    )
    draft_case = build_scenario(
        suite_config("tiny"), 1, tmp_path, "coding_error", stages
    )
    draft_result = run_qc(
        ScenarioSource(draft_case.directory), "V0002", "V0001", run_id="case-2"
    )

    database = tmp_path / "store.db"
    with SqliteStore(database) as store:
        store.record_result(confirmed_result)
        store.record_outcome(
            "case-1",
            root_cause="MISSING_STORES",
            confirmed=True,
            likely_origin="SOURCE",
            severity="HIGH",
            requires_investigation=True,
            provenance="analyst",
        )
        store.record_result(draft_result)
        store.record_outcome("case-2", root_cause="CODING", confirmed=False)

    records = records_from_store(database)
    assert len(records) == 1
    record = records[0]
    assert record.source == "analyst"
    assert record.labels["likely_cause"] == "MISSING_STORES"
    assert record.labels["likely_origin"] == "SOURCE"
    assert record.labels["requires_investigation"] == "True"
    assert record.text and "status=" in record.text


def test_outcome_without_provenance_is_not_analyst(tmp_path):
    from qc.labels import records_from_store

    case = build_scenario(
        suite_config("tiny"),
        0,
        tmp_path,
        "missing_stores",
        ("source", "coded", "warehouse", "report"),
    )
    result = run_qc(ScenarioSource(case.directory), "V0002", "V0001", run_id="case-x")
    database = tmp_path / "unknown.db"
    with SqliteStore(database) as store:
        store.record_result(result)
        store.record_outcome("case-x", root_cause="MISSING_STORES", confirmed=True)
    records = records_from_store(database)
    assert records == []  # Review labels are missing; never infer them from cause.
    with SqliteStore(database) as store:
        assert store.latest_outcome("case-x")["provenance"] == "unknown"


def test_cli_champion(cohorts, tmp_path, capsys):
    records, _, eval_dir = cohorts
    labels_path = tmp_path / "labels.jsonl"
    LabelStore(labels_path).append(records)
    out = tmp_path / "cli-report"
    assert (
        main(
            [
                "champion",
                "--train-labels",
                str(labels_path),
                "--suite-dir",
                str(eval_dir),
                "--min-overall-accuracy",
                "0.0",
                "--min-cause-accuracy",
                "0.0",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "champion=" in output
    assert (out / "champion.md").exists()
