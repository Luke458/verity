from __future__ import annotations

import json

import numpy as np
import pytest

from qc.decisions import FEATURE_VERSION
from qc.evidence_text import EVIDENCE_TEXT_VERSION, evidence_text
from qc.labels import LabelRecord
from qc.run import run_qc
from qc.text_provider import (
    ModernBertEmbedder,
    TextDecisionProvider,
    default_text_embedder,
    load_decision_provider,
    train_text_provider,
)
from qcgen.config import suite_config
from qcgen.scenarios import build_scenario
from qcgen.sources import ScenarioSource


class FakeEmbedder:
    """Deterministic token-presence embedder for tests; no model download."""

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
                ("coding", 2),
                ("warehouse", 3),
                ("backfill", 4),
                ("contract", 5),
            ):
                if token in lowered:
                    vector[index] += 1.0
            vectors.append(vector)
        return np.asarray(vectors)


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    built = build_scenario(
        suite_config("tiny"),
        0,
        tmp_path_factory.mktemp("text-provider"),
        "missing_stores",
        ("source", "coded", "warehouse", "report"),
    )
    return run_qc(ScenarioSource(built.directory), "V0002", "V0001")


def _records() -> list[LabelRecord]:
    texts = {
        "MISSING_STORES": "event store missing latest week reason missing store",
        "CODING": "revision unexplained lineage coded reason dollar change",
        "WAREHOUSE": "revision unexplained lineage warehouse reason transform",
        "BACKFILL": "event backfill historical weeks added reason backfill store",
    }
    records: list[LabelRecord] = []
    for index in range(16):
        for cause, text in texts.items():
            records.append(
                LabelRecord(
                    run_id=f"run-{cause}-{index}",
                    source="oracle",
                    family=cause.lower(),
                    labels={
                        "likely_cause": cause,
                        "likely_origin": "CODING" if cause == "CODING" else "SOURCE",
                        "severity": "MEDIUM",
                        "requires_investigation": (
                            "False" if cause == "BACKFILL" else "True"
                        ),
                    },
                    features=[0.0, 0.0, 0.0],
                    feature_version=FEATURE_VERSION,
                    metadata={"text_version": EVIDENCE_TEXT_VERSION},
                    text=text,
                )
            )
    return records


def test_evidence_text_is_bounded_and_leak_free(case):
    text = evidence_text(case, max_chars=1500)
    assert len(text) <= 1500
    assert "status=" in text
    assert "expected_class" not in text
    assert "expected_status" not in text
    assert "MISSING_STORES" not in text
    assert evidence_text(case, max_chars=1500) == text  # deterministic


def test_train_text_provider_and_roundtrip(tmp_path):
    embedder = FakeEmbedder()
    provider, metrics = train_text_provider(
        _records(), embedder, validation_fraction=0.3, seed=3
    )
    assert provider.name == "text_probe"
    assert metrics["train"]["overall_accuracy"] >= 0.9
    assert metrics["validation"]["overall_accuracy"] >= 0.5
    assert set(metrics["validation"]["fields"]) == {
        "likely_cause",
        "likely_origin",
        "severity",
        "requires_investigation",
    }
    assert provider.metadata["text_version"] == EVIDENCE_TEXT_VERSION
    assert provider.metadata["embedder"]["model_id"] == "fake-model"

    directory = tmp_path / "text-probe"
    provider.save(directory)
    loaded = TextDecisionProvider.load(directory, embedder=FakeEmbedder())
    assert isinstance(loaded, TextDecisionProvider)

    dispatched = load_decision_provider(directory, embedder=FakeEmbedder())
    assert isinstance(dispatched, TextDecisionProvider)


def test_text_provider_decides_on_evidence(case):
    provider, _ = train_text_provider(
        _records(), FakeEmbedder(), validation_fraction=0.3, seed=3
    )
    decisions = provider.decide(case)
    assert decisions.provider == "text_probe"
    cause = decisions.get("likely_cause")
    # Entire families are held out: unseen class accuracy is not a protocol guarantee.
    assert cause.value in provider.heads["likely_cause"].classes
    assert cause.probability_kind == "frozen_encoder_probe"
    severity = decisions.get("severity")
    assert severity.kind == "score"
    assert severity.index is not None
    assert decisions.requires_investigation is True


def test_text_version_mismatch_rejected(tmp_path):
    provider, _ = train_text_provider(_records(), FakeEmbedder())
    provider.save(tmp_path)
    path = tmp_path / "provider.json"
    payload = json.loads(path.read_text())
    payload["text_version"] = 999
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        TextDecisionProvider.load(tmp_path, embedder=FakeEmbedder())


def test_default_text_embedder_rejects_unknown_kind():
    with pytest.raises(ValueError):
        default_text_embedder({"kind": "magic"})


def test_modernbert_metadata_defaults():
    metadata = ModernBertEmbedder().metadata()
    assert metadata["kind"] == "hf_encoder"
    assert "ModernBERT" in metadata["model_id"]
    assert metadata["dimension"] is None
