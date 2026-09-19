"""Provider bake-off on a frozen cohort (champion selection).

Trains the learned substrates on a label set, scores every available provider
against held-out cases, applies pre-registered gates, and selects a champion
only if one passes. Providers:

- ``rule``: deterministic evidence rules, always available for runs;
- ``feature_head``: linear head over the versioned numeric feature encoder;
- ``text_probe``: linear probe over frozen encoder embeddings of the evidence
  text (needs a text embedder);
- ``remote``: a Jev-compatible decision provider (needs an endpoint).

Provenance is enforced: training and evaluation cases must not overlap, and a
champion is production-eligible only when every training label came from
confirmed analyst outcomes.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .cohort import code_sha256
from .config import DatasetConfig
from .decisions import RuleDecisionProvider
from .labels import LabelRecord, oracle_labels_for_result
from .text_provider import TextDecisionProvider, TextEmbedder, train_text_provider
from .training import train_decision_provider

DEFAULT_CHAMPION_GATES: dict[str, float] = {
    "min_overall_accuracy": 0.85,
    "min_cause_accuracy": 0.85,
}
PROVIDER_PRIORITY = {"rule": 0, "feature_head": 1, "text_probe": 2, "remote": 3}


@dataclass
class EvalItem:
    case_id: str
    family: str | None
    labels: dict[str, str]
    result: Any | None = None
    record: LabelRecord | None = None


@dataclass
class ProviderScore:
    provider: str
    available: bool
    overall_accuracy: float = 0.0
    cause_accuracy: float = 0.0
    per_field: dict[str, float] = field(default_factory=dict)
    cases: int = 0
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ChampionResult:
    champion: str | None
    production_eligible: bool
    gates: dict[str, float]
    scores: list[ProviderScore]
    provenance: dict[str, Any]
    code_sha256: str
    limitations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["scores"] = [score.to_dict() for score in self.scores]
        return payload


def eval_cases_from_suite(
    suite_dir: str | Path, config: DatasetConfig | None = None
) -> list[EvalItem]:
    from qcgen.sources import ScenarioSource

    from .run import run_qc

    suite_dir = Path(suite_dir)
    config = config or DatasetConfig()
    suite = json.loads((suite_dir / "suite.json").read_text())
    items: list[EvalItem] = []
    for entry in suite["scenarios"]:
        scenario_dir = suite_dir / entry["scenario_id"]
        manifest = json.loads((scenario_dir / "manifest.json").read_text())
        result = run_qc(
            ScenarioSource(scenario_dir),
            manifest["current_version"],
            manifest["previous_version"],
            config,
            expected_events=manifest.get("expected_events", []),
        )
        record = oracle_labels_for_result(result, manifest)
        items.append(
            EvalItem(
                case_id=f"{suite.get('suite_id')}:{manifest.get('scenario_id', entry['scenario_id'])}",
                family=manifest.get("family"),
                labels=record.labels,
                result=result,
            )
        )
    return items


def _train_key(record: LabelRecord) -> str:
    suite_id = record.metadata.get("suite_id")
    scenario_id = record.metadata.get("scenario_id")
    if suite_id and scenario_id:
        return f"{suite_id}:{scenario_id}"
    return str(record.run_id)


def _record_key(item: EvalItem) -> str:
    return item.case_id


def _normalize(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def _accuracy(correct: dict[str, int], total: dict[str, int]) -> dict[str, float]:
    return {
        field: (correct.get(field, 0) / total[field] if total[field] else 0.0)
        for field in total
    }


def _score_result_provider(provider: Any, items: Sequence[EvalItem]) -> ProviderScore:
    relevant = [item for item in items if item.result is not None]
    correct: dict[str, int] = {}
    total: dict[str, int] = {}
    for item in relevant:
        decisions = provider.decide(item.result)
        for field_name, expected in item.labels.items():
            decision = decisions.get(field_name)
            if decision is None:
                continue
            total[field_name] = total.get(field_name, 0) + 1
            if _normalize(decision.value) == expected:
                correct[field_name] = correct.get(field_name, 0) + 1
    per_field = _accuracy(correct, total)
    overall = (
        sum(correct.values()) / sum(total.values()) if sum(total.values()) else 0.0
    )
    return ProviderScore(
        provider=provider.name,
        available=True,
        overall_accuracy=overall,
        cause_accuracy=per_field.get("likely_cause", 0.0),
        per_field=per_field,
        cases=len(relevant),
    )


def _head_features(provider: Any, records: Sequence[LabelRecord]) -> np.ndarray:
    if isinstance(provider, TextDecisionProvider):
        texts = [record.text for record in records]
        if any(text is None for text in texts):
            raise ValueError("text probe evaluation requires evidence text")
        embeddings = provider.embedder.embed([str(text) for text in texts])
        return provider._standardize(embeddings)
    features = np.asarray([record.features for record in records], dtype=float)
    return provider._standardize(features)


def _score_head_provider(provider: Any, items: Sequence[EvalItem]) -> ProviderScore:
    records = [item.record for item in items if item.record is not None]
    if not records:
        raise ValueError("no record-based evaluation items")
    features = _head_features(provider, records)
    correct: dict[str, int] = {}
    total: dict[str, int] = {}
    for field_name, head in provider.heads.items():
        probabilities = head.probabilities(features)
        predictions = [
            head.classes[int(index)] for index in probabilities.argmax(axis=1)
        ]
        for record, predicted in zip(records, predictions):
            expected = record.labels.get(field_name)
            if expected is None:
                continue
            total[field_name] = total.get(field_name, 0) + 1
            if predicted == expected:
                correct[field_name] = correct.get(field_name, 0) + 1
    per_field = _accuracy(correct, total)
    overall = (
        sum(correct.values()) / sum(total.values()) if sum(total.values()) else 0.0
    )
    return ProviderScore(
        provider=provider.name,
        available=True,
        overall_accuracy=overall,
        cause_accuracy=per_field.get("likely_cause", 0.0),
        per_field=per_field,
        cases=len(records),
    )


def _unavailable(provider: str, note: str) -> ProviderScore:
    return ProviderScore(provider=provider, available=False, note=note)


def _provider_failures(score: ProviderScore, gates: dict[str, float]) -> list[str]:
    failed: list[str] = []
    if "min_overall_accuracy" in gates and score.overall_accuracy < gates["min_overall_accuracy"]:
        failed.append("min_overall_accuracy")
    if "min_cause_accuracy" in gates and score.cause_accuracy < gates["min_cause_accuracy"]:
        failed.append("min_cause_accuracy")
    return failed


def run_champion(
    train_records: Sequence[LabelRecord],
    eval_items: Sequence[EvalItem],
    config: DatasetConfig | None = None,
    text_embedder: TextEmbedder | None = None,
    remote_provider: Any | None = None,
    gates: dict[str, float] | None = None,
    validation_fraction: float = 0.3,
    seed: int = 0,
) -> ChampionResult:
    if not train_records:
        raise ValueError("training requires at least one confirmed label record")
    if not eval_items:
        raise ValueError("evaluation requires at least one held-out case")
    config = config or DatasetConfig()
    gates = dict(gates or DEFAULT_CHAMPION_GATES)
    unknown = set(gates) - set(DEFAULT_CHAMPION_GATES)
    if unknown:
        raise ValueError(f"unknown gates: {sorted(unknown)}")

    train_keys = {_train_key(record) for record in train_records}
    eval_keys = {_record_key(item) for item in eval_items}
    overlap = train_keys & eval_keys
    if overlap:
        raise ValueError(
            f"training and evaluation cases overlap: {sorted(overlap)}; "
            "split the cohorts before selecting a champion"
        )

    scores: list[ProviderScore] = []

    rule = RuleDecisionProvider(config)
    scores.append(_score_result_provider(rule, eval_items))

    try:
        feature_provider, _ = train_decision_provider(
            train_records,
            config,
            validation_fraction=validation_fraction,
            seed=seed,
        )
    except ValueError as error:
        scores.append(_unavailable("feature_head", str(error)))
    else:
        if all(item.result is not None for item in eval_items):
            score = _score_result_provider(feature_provider, eval_items)
            score.provider = "feature_head"
            scores.append(score)
        else:
            try:
                score = _score_head_provider(feature_provider, eval_items)
                score.provider = "feature_head"
                scores.append(score)
            except ValueError as error:
                scores.append(_unavailable("feature_head", str(error)))

    if text_embedder is None:
        scores.append(_unavailable("text_probe", "no text embedder configured"))
    else:
        try:
            text_provider, _ = train_text_provider(
                train_records,
                text_embedder,
                config,
                validation_fraction=validation_fraction,
                seed=seed,
            )
        except ValueError as error:
            scores.append(_unavailable("text_probe", str(error)))
        else:
            if all(item.result is not None for item in eval_items):
                score = _score_result_provider(text_provider, eval_items)
                score.provider = "text_probe"
                scores.append(score)
            else:
                try:
                    score = _score_head_provider(text_provider, eval_items)
                    score.provider = "text_probe"
                    scores.append(score)
                except ValueError as error:
                    scores.append(_unavailable("text_probe", str(error)))

    if remote_provider is None:
        scores.append(_unavailable("remote", "no remote provider configured"))
    else:
        try:
            score = _score_result_provider(remote_provider, eval_items)
            score.provider = "remote"
            scores.append(score)
        except Exception as error:  # noqa: BLE001 - fail closed: report, never assume success
            scores.append(_unavailable("remote", f"{type(error).__name__}: {error}"))

    eligible = [
        score
        for score in scores
        if score.available and not _provider_failures(score, gates)
    ]
    champion: str | None = None
    if eligible:
        eligible.sort(
            key=lambda score: (
                -score.overall_accuracy,
                PROVIDER_PRIORITY.get(score.provider, 9),
            )
        )
        champion = eligible[0].provider

    sources = sorted({record.source for record in train_records})
    if sources == ["oracle"]:
        label_provenance = "oracle"
    elif sources == ["analyst"]:
        label_provenance = "analyst"
    else:
        label_provenance = "+".join(sources) if sources else "none"
    provenance = {
        "train_records": len(train_records),
        "train_sources": sources,
        "train_keys": len(train_keys),
        "eval_cases": len(eval_items),
        "eval_keys": sorted(eval_keys),
        "labels": label_provenance,
    }
    limitations = [
        "Synthetic labels validate the harness and the engine, not semantic accuracy.",
        "A learned champion is only meaningful once trained on confirmed analyst outcomes and evaluated on a frozen held-out cohort.",
        "Gates are pre-registered; changing them after seeing results invalidates the comparison.",
    ]
    return ChampionResult(
        champion=champion,
        production_eligible=(label_provenance == "analyst") and champion is not None,
        gates=gates,
        scores=scores,
        provenance=provenance,
        code_sha256=code_sha256(),
        limitations=limitations,
    )


def render_champion_markdown(result: ChampionResult) -> str:
    lines = ["# Provider champion selection", ""]
    lines.append(f"**Champion: {result.champion or 'none'}**")
    lines.append("")
    lines.append(
        f"- production eligible: {result.production_eligible} "
        f"(labels: {result.provenance['labels']})"
    )
    lines.append(
        f"- train records: {result.provenance['train_records']} "
        f"({', '.join(result.provenance['train_sources']) or 'none'}); "
        f"eval cases: {result.provenance['eval_cases']}"
    )
    lines.append(f"- code sha256: {result.code_sha256}")
    lines.append("")
    lines.append("| provider | available | overall | cause | cases | note |")
    lines.append("|---|---|---|---|---|---|")
    for score in result.scores:
        lines.append(
            f"| {score.provider} | {score.available} | "
            f"{score.overall_accuracy:.3f} | {score.cause_accuracy:.3f} | "
            f"{score.cases} | {score.note} |"
        )
    lines.append("")
    lines.append("## Gates")
    for name, threshold in sorted(result.gates.items()):
        lines.append(f"- {name} >= {threshold}")
    lines.append("")
    lines.append("## Limitations")
    for limitation in result.limitations:
        lines.append(f"- {limitation}")
    return "\n".join(lines) + "\n"


def write_champion_report(
    result: ChampionResult, directory: str | Path
) -> dict[str, Path]:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=False)
    json_path = path / "champion.json"
    markdown_path = path / "champion.md"
    json_path.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True, default=str)
    )
    markdown_path.write_text(render_champion_markdown(result))
    return {"json": json_path, "markdown": markdown_path}
