"""TSPulse revision-shape suitability benchmark (Phase 11).

Weekly retail histories are shorter than the 512-point TSPulse context, so
this benchmark answers one question honestly: do TSPulse embeddings of
*revision series* (current minus previous over overlapping weeks) separate
known fault families from each other and from controls? It is research-only
and never production-eligible until the weekly-length question is settled on
real data.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import DatasetConfig
from .tspulse import TSPulseResearch

Embedder = Callable[[Sequence[float]], Any]


def _embedding(case: Any) -> list[float]:
    assert case.embedding is not None
    return case.embedding


@dataclass
class RevisionSeries:
    scenario_id: str
    family: str
    expected_class: str
    series_id: str
    is_control: bool
    values: list[float]
    is_zero_revision: bool = False
    embedding: list[float] | None = None
    status: str = "PENDING"
    transformation: str | None = None


@dataclass
class BenchmarkResult:
    suite_id: str
    n_scenarios: int
    n_series: int
    embedded_series: int
    nearest_centroid_accuracy: float
    control_mean_embedding_norm: float
    fault_mean_embedding_norm: float
    mean_intra_family_similarity: float = 0.0
    mean_inter_family_similarity: float = 0.0
    zero_revision_series: int = 0
    scenario_holdout_accuracy: float | None = None
    gate: dict[str, Any] = field(default_factory=dict)
    production_eligible: bool = False
    transformations: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    cases: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _scenario_holdout_accuracy(cases: Sequence[RevisionSeries]) -> float | None:
    """Nearest-centroid accuracy with scenarios (not series) held out.

    The leave-one-series-out number can see sibling series from the same
    scenario during centroid construction. This splits scenarios into
    centroid and test halves; returns ``None`` when the suite is too small to
    estimate it.
    """
    scenarios = sorted({case.scenario_id for case in cases})
    if len(scenarios) < 4:
        return None
    train_scenarios = {name for index, name in enumerate(scenarios) if index % 2 == 0}
    train = [case for case in cases if case.scenario_id in train_scenarios]
    test = [case for case in cases if case.scenario_id not in train_scenarios]
    if not test or len({case.family for case in train}) < 2:
        return None
    families = sorted({case.family for case in train})
    centroids = {
        family: np.mean(
            [_embedding(case) for case in train if case.family == family], axis=0
        )
        for family in families
    }
    correct = 0
    for case in test:
        vector = np.asarray(_embedding(case), dtype=float)
        distances = {
            family: float(np.linalg.norm(vector - centroid))
            for family, centroid in centroids.items()
        }
        predicted = min(distances, key=lambda family: distances[family])
        correct += int(predicted == case.family)
    return correct / len(test)


def load_tspulse_gate(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict) or "metric" not in payload:
        raise ValueError("tspulse gate must be an object with a 'metric'")
    return payload


def evaluate_tspulse_gate(
    gate: dict[str, Any], metrics: dict[str, float | None]
) -> dict[str, Any]:
    metric = str(gate["metric"])
    if metric not in metrics:
        raise ValueError(f"unknown gate metric {metric!r}")
    actual = metrics[metric]
    threshold = gate.get("min_value")
    passed = (
        actual is not None
        and threshold is not None
        and float(actual) >= float(threshold)
    )
    return {
        "metric": metric,
        "threshold": threshold,
        "actual": actual,
        "require_scenario_holdout": bool(gate.get("require_scenario_holdout", False)),
        "note": str(gate.get("note", "")),
        "passed": passed,
        "detail": (
            "gate not evaluable on this suite"
            if actual is None
            else "passed" if passed else "below threshold"
        ),
    }


def _default_embedder(allow_resample: bool) -> Embedder:
    adapter = TSPulseResearch()

    def embed(values: Sequence[float]):
        return adapter.embed(values, allow_resample=allow_resample)

    return embed


def build_revision_series(
    suite_dir: str | Path,
    config: DatasetConfig | None = None,
    max_scenarios: int | None = None,
    oracle_dir: str | Path | None = None,
) -> list[RevisionSeries]:
    from qcgen.oracle_vault import OracleVault, default_oracle_root

    suite_dir = Path(suite_dir)
    config = config or DatasetConfig()
    suite = json.loads((suite_dir / "suite.json").read_text())
    vault = OracleVault(oracle_dir or default_oracle_root(suite_dir))
    week = config.week_column
    metric = config.primary_metric
    series: list[RevisionSeries] = []

    for entry in suite["scenarios"][: max_scenarios or len(suite["scenarios"])]:
        scenario_id = entry["scenario_id"]
        oracle = vault.read(scenario_id) or {}
        case = oracle["cases"][0] if oracle.get("cases") else {}
        scenario_dir = suite_dir / scenario_id
        manifest = json.loads((scenario_dir / "manifest.json").read_text())
        previous = pd.read_parquet(
            scenario_dir / "versions" / manifest["previous_version"] / "report" / "fact.parquet"
        )
        current = pd.read_parquet(
            scenario_dir / "versions" / manifest["current_version"] / "report" / "fact.parquet"
        )
        is_control = case.get("kind") == "control"
        if (
            metric not in previous.columns
            or metric not in current.columns
            or week not in previous.columns
            or week not in current.columns
        ):
            continue

        previous_map = _weekly_totals_by_series(previous, week, metric, config)
        current_map = _weekly_totals_by_series(current, week, metric, config)
        for series_id in sorted(set(previous_map) & set(current_map)):
            aligned = pd.concat(
                [
                    previous_map[series_id].rename("previous"),
                    current_map[series_id].rename("current"),
                ],
                axis=1,
                join="inner",
            ).fillna(0.0)
            revision = (aligned["current"] - aligned["previous"]).tolist()
            series.append(
                RevisionSeries(
                    scenario_id=scenario_id,
                    family=str(oracle.get("family")),
                    expected_class=str(case.get("expected_class")),
                    series_id=series_id,
                    is_control=is_control,
                    values=[float(value) for value in revision],
                    is_zero_revision=all(
                        abs(float(value)) < 1e-9 for value in revision
                    ),
                )
            )
    return series


def _weekly_totals_by_series(
    frame: pd.DataFrame,
    week: str,
    metric: str,
    config: DatasetConfig,
) -> dict[str, pd.Series]:
    """Weekly totals per configured series: national plus each entity value."""
    result: dict[str, pd.Series] = {}
    result["national"] = frame.groupby(week)[metric].sum().sort_index()
    for column in config.temporal_entity_columns:
        if column not in frame.columns:
            continue
        grouped = frame.groupby([column, week])[metric].sum()
        by_series: dict[str, dict[int, float]] = {}
        for (value, week_id), total in grouped.items():
            by_series.setdefault(f"{column}:{value}", {})[int(week_id)] = float(
                total
            )
        for series_id, values in by_series.items():
            result[series_id] = pd.Series(values).sort_index()
    return result


def run_tspulse_benchmark(
    suite_dir: str | Path,
    config: DatasetConfig | None = None,
    embedder: Embedder | None = None,
    allow_resample: bool = True,
    max_scenarios: int | None = None,
    oracle_dir: str | Path | None = None,
    gate_path: str | Path | None = None,
) -> BenchmarkResult:
    suite_dir = Path(suite_dir)
    suite = json.loads((suite_dir / "suite.json").read_text())
    embedder = embedder or _default_embedder(allow_resample)
    cases = build_revision_series(suite_dir, config, max_scenarios, oracle_dir)

    for case in cases:
        result = embedder(case.values)
        payload = result.to_dict() if hasattr(result, "to_dict") else dict(result)
        case.status = str(payload.get("status"))
        case.embedding = payload.get("embedding")
        case.transformation = payload.get("transformation")

    embedded = [case for case in cases if case.embedding]
    if not embedded:
        statuses = sorted({case.status for case in cases})
        return BenchmarkResult(
            suite_id=suite.get("suite_id", suite_dir.name),
            n_scenarios=len({case.scenario_id for case in cases}),
            n_series=len(cases),
            embedded_series=0,
            nearest_centroid_accuracy=0.0,
            control_mean_embedding_norm=0.0,
            fault_mean_embedding_norm=0.0,
            limitations=[f"no embeddings produced; statuses={statuses}"],
        )

    accuracy = _leave_one_series_out_accuracy(embedded)
    holdout_accuracy = _scenario_holdout_accuracy(embedded)
    intra_similarity, inter_similarity = _similarity_stats(embedded)
    gate: dict[str, Any] = {}
    if gate_path is not None and Path(gate_path).exists():
        gate = evaluate_tspulse_gate(
            load_tspulse_gate(gate_path),
            {
                "nearest_centroid_accuracy": accuracy,
                "scenario_holdout_accuracy": holdout_accuracy,
            },
        )
    zero_revision = sum(1 for case in embedded if case.is_zero_revision)
    control_norms = [
        float(np.linalg.norm(_embedding(case)))
        for case in embedded
        if case.is_control
    ]
    fault_norms = [
        float(np.linalg.norm(_embedding(case)))
        for case in embedded
        if not case.is_control
    ]
    limitations = [
        "Weekly histories are shorter than the 512-point TSPulse context; "
        "embeddings use a linear resample and are not valid for production "
        "similarity claims.",
        "Nearest-centroid accuracy is series-level leave-one-out: with one "
        "scenario per family a scenario-level holdout is not estimable, so "
        "same-scenario series are visible during centroid construction.",
        "The benchmark answers separation, not detection thresholds.",
        "Promotion requires the pre-registered scenario-holdout gate "
        "(config/tspulse-gate.json) to pass; production_eligible stays false "
        "until then.",
    ]
    if zero_revision:
        limitations.append(
            f"{zero_revision} series have an all-zero overlap revision "
            "(faults confined to the new period, e.g. missing stores or market "
            "movement); they carry no revision signal and are excluded from "
            "the accuracy, which is the expected limitation of a "
            "revision-series representation."
        )
    embedded_scenarios = len({case.scenario_id for case in cases})
    declared_scenarios = len(suite["scenarios"])
    if embedded_scenarios < declared_scenarios:
        limitations.append(
            f"{declared_scenarios - embedded_scenarios} scenarios were skipped "
            "because the report table lacked the primary metric (for example "
            "schema-contract failures)."
        )
    return BenchmarkResult(
        suite_id=suite.get("suite_id", suite_dir.name),
        n_scenarios=len({case.scenario_id for case in cases}),
        n_series=len(cases),
        embedded_series=len(embedded),
        nearest_centroid_accuracy=accuracy,
        control_mean_embedding_norm=(
            float(np.mean(control_norms)) if control_norms else 0.0
        ),
        fault_mean_embedding_norm=(
            float(np.mean(fault_norms)) if fault_norms else 0.0
        ),
        mean_intra_family_similarity=intra_similarity,
        mean_inter_family_similarity=inter_similarity,
        zero_revision_series=zero_revision,
        scenario_holdout_accuracy=holdout_accuracy,
        gate=gate,
        production_eligible=bool(gate.get("passed", False)),
        transformations=sorted(
            {case.transformation for case in embedded if case.transformation}
        ),
        limitations=limitations,
        cases=[
            {
                "scenario_id": case.scenario_id,
                "family": case.family,
                "series_id": case.series_id,
                "is_control": case.is_control,
                "status": case.status,
                "transformation": case.transformation,
                "embedding_norm": (
                    float(np.linalg.norm(case.embedding))
                    if case.embedding
                    else None
                ),
            }
            for case in cases
        ],
    )


def _leave_one_series_out_accuracy(cases: list[RevisionSeries]) -> float:
    """Nearest-centroid family classification, holding out each series.

    Series with an all-zero revision (faults confined to the new period) carry
    no signal and are excluded from both centroids and scoring.
    """
    usable = [
        case
        for case in cases
        if case.embedding is not None
        and not case.is_control
        and not case.is_zero_revision
    ]
    by_family: dict[str, list[RevisionSeries]] = {}
    for case in usable:
        by_family.setdefault(case.family, []).append(case)
    correct = 0
    total = 0
    for held_out in usable:
        centroids: dict[str, np.ndarray] = {}
        for family, family_cases in by_family.items():
            vectors = [
                np.asarray(case.embedding)
                for case in family_cases
                if case is not held_out
            ]
            if vectors:
                centroids[family] = np.mean(vectors, axis=0)
        if not centroids:
            continue
        query = np.asarray(held_out.embedding)
        predicted = min(
            centroids,
            key=lambda name: float(np.linalg.norm(query - centroids[name])),
        )
        correct += int(predicted == held_out.family)
        total += 1
    return correct / total if total else 0.0


def _similarity_stats(cases: list[RevisionSeries]) -> tuple[float, float]:
    from .tspulse import cosine

    usable = [
        case
        for case in cases
        if case.embedding is not None
        and not case.is_control
        and not case.is_zero_revision
    ]
    intra: list[float] = []
    inter: list[float] = []
    for index, left in enumerate(usable):
        for right in usable[index + 1 :]:
            similarity = cosine(_embedding(left), _embedding(right))
            if left.family == right.family:
                intra.append(similarity)
            else:
                inter.append(similarity)
    return (
        float(np.mean(intra)) if intra else 0.0,
        float(np.mean(inter)) if inter else 0.0,
    )
