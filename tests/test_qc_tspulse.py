from __future__ import annotations

import pytest

from qc.tspulse import EmbeddingResult, TSPulseResearch, cosine
from qc.tspulse_benchmark import run_tspulse_benchmark
from qcgen.config import suite_config
from qcgen.scenarios import generate_suite


def test_embedding_length_gate_without_model_load():
    adapter = TSPulseResearch()
    result = adapter.embed([float(i) for i in range(100)])
    assert result.status == "INSUFFICIENT_LENGTH"
    assert result.embedding is None
    assert result.input_length == 100
    assert result.production_eligible is False


def test_anomaly_gate_for_weekly_frequency():
    adapter = TSPulseResearch()
    result = adapter.anomaly([float(i) for i in range(100)], frequency="W")
    assert result.status == "RESEARCH_GATE"
    assert result.scores is None
    assert result.production_eligible is False


def test_cosine_validation():
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine([0.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        cosine([1.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        cosine([], [])
    with pytest.raises(ValueError):
        cosine([1.0, float("nan")], [1.0, 0.0])


def _fake_embedder(values):
    """Shape-based stand-in for TSPulse: scale-normalised revision signature.

    This validates the benchmark plumbing (grouping, embedding, nearest
    centroid, control norms), not representation quality.
    """
    series = [float(value) for value in values]
    mass = sum(abs(value) for value in series) or 1.0
    signs = sum(1.0 if value > 1e-9 else -1.0 if value < -1e-9 else 0.0 for value in series)
    nonzero = sum(1 for value in series if abs(value) > 1e-9) / len(series)
    first = next(
        (index for index, value in enumerate(series) if abs(value) > 1e-9), 0
    ) / len(series)
    peak = max(range(len(series)), key=lambda index: abs(series[index])) / len(series)
    return EmbeddingResult(
        status="EXPERIMENTAL",
        embedding=[signs, nonzero, first, peak, sum(series) / mass],
        input_length=len(series),
        transformation="fake_test_embedder",
    )


def test_benchmark_with_injected_embedder(tmp_path):
    # Historical-shape families only: faults confined to the new period have a
    # zero overlap revision and are excluded by design.
    config = suite_config(
        "tiny",
        families=("new_store_backfill", "history_truncation", "commodity_remap"),
        controls=(),
    )
    suite_dir = generate_suite(
        config,
        tmp_path,
        6,
        "tsp",
        ("source", "coded", "warehouse", "report"),
    )
    result = run_tspulse_benchmark(
        suite_dir, embedder=_fake_embedder, allow_resample=True
    )
    assert result.n_scenarios == 6
    assert result.embedded_series == result.n_series
    assert result.embedded_series > 0
    assert result.nearest_centroid_accuracy >= 0.8
    assert result.production_eligible is False
    assert result.transformations == ["fake_test_embedder"]
    assert result.limitations


def test_benchmark_flags_zero_revision_families(tmp_path):
    suite_dir = generate_suite(
        suite_config("tiny"),
        tmp_path,
        4,
        "tsp-mixed",
        ("source", "coded", "warehouse", "report"),
    )
    result = run_tspulse_benchmark(suite_dir, embedder=_fake_embedder)
    # missing_stores and market_movement only touch the new week.
    assert result.zero_revision_series > 0
    assert any(
        "all-zero overlap revision" in limitation
        for limitation in result.limitations
    )


def test_benchmark_gate_requires_scenario_holdout(tmp_path):
    import json

    config = suite_config(
        "tiny",
        families=("new_store_backfill", "history_truncation", "commodity_remap"),
        controls=(),
    )
    suite_dir = generate_suite(
        config, tmp_path, 8, "tsp-gate", ("source", "coded", "warehouse", "report")
    )
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(
        json.dumps(
            {
                "metric": "scenario_holdout_accuracy",
                "min_value": 0.0,
                "require_scenario_holdout": True,
                "note": "test gate",
            }
        )
    )
    result = run_tspulse_benchmark(
        suite_dir, embedder=_fake_embedder, gate_path=gate_path
    )
    assert result.scenario_holdout_accuracy is not None
    assert result.gate["passed"] is True
    assert result.production_eligible is False  # synthetic gates never confer eligibility

    strict = tmp_path / "strict.json"
    strict.write_text(
        json.dumps(
            {
                "metric": "scenario_holdout_accuracy",
                "min_value": 1.01,
            }
        )
    )
    failed = run_tspulse_benchmark(
        suite_dir, embedder=_fake_embedder, gate_path=strict
    )
    assert failed.gate["passed"] is False
    assert failed.production_eligible is False


def test_benchmark_reports_no_embeddings(tmp_path):
    suite_dir = generate_suite(
        suite_config("tiny"),
        tmp_path,
        2,
        "tsp-gated",
        ("source", "coded", "warehouse", "report"),
    )

    def gated(values):
        return EmbeddingResult(status="INSUFFICIENT_LENGTH", input_length=len(values))

    result = run_tspulse_benchmark(suite_dir, embedder=gated)
    assert result.embedded_series == 0
    assert result.nearest_centroid_accuracy == 0.0
    assert any("no embeddings" in limitation for limitation in result.limitations)
