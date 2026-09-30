"""Decomposed decision fields: narrow binary questions and their combiner.

One broad question is hard. A single 11-way ``choice`` over an imbalanced cause
taxonomy measured 62.6% accuracy and ECE 0.154 in the nearest public benchmark,
while the same model asked five *narrow* signal questions with a small combiner
on top measured 95.0% and ECE 0.027 - a +32 point move and a 5.7x calibration
improvement for no extra model capacity (docs/next-phase-plan.md, B1).

This module applies that move to our own fields without inventing labels we do
not have. Supervision stays exactly the four oracle/analyst fields; the *target
formulation* changes:

* a nominal field (``choice``) becomes one-vs-rest binary questions, one per
  declared value - "is the best-supported cause exactly WAREHOUSE?";
* the ordinal field (``score``) becomes threshold questions - "is severity at
  least HIGH?" - the standard ordinal decomposition, reconstructed as an
  ordinal distribution;
* the boolean field is already one narrow question.

Three properties follow that a shared softmax does not give us:

1. every question is calibrated on its own, so one field's confidence is not
   hostage to a shared temperature - relevant because temperature scaling
   cannot correct a wrong-direction bias (docs/next-phase-plan.md, 2.4);
2. narrow binary targets are the documented low-label regime trick, which is
   where we are heading with real data;
3. the same question set drives the System One wire protocol: a Jev-compatible
   endpoint is asked 22 narrow ``noul`` questions instead of 4 broad ones.

Synthetic labels validate plumbing only. Nothing here is production eligible.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dc_field
from pathlib import Path
from typing import Any

import numpy as np

from .decisions import (
    FEATURE_VERSION,
    DecisionSet,
    DecisionValue,
    FeatureEncoder,
    FieldSpec,
    LinearHead,
    default_fields,
    field_index,
    score_index,
)
from .labels import LabelRecord
from .training import fit_head, fit_temperature, split_records_four

# Bump when the question set or the reconstruction rule changes. An artifact
# built against a different version is rejected at load, never silently reused.
DECOMPOSITION_VERSION = 1

ONE_VS_REST = "one_vs_rest"
THRESHOLD = "threshold"
BOOLEAN = "boolean"

BINARY_CLASSES: tuple[str, ...] = ("False", "True")


@dataclass(frozen=True)
class BinaryQuestion:
    """One narrow question whose answer is a single probability."""

    question_id: str
    field: str
    kind: str
    target: str
    instructions: str
    positive: str
    negative: str

    def to_dict(self) -> dict[str, str]:
        return {
            "question_id": self.question_id,
            "field": self.field,
            "kind": self.kind,
            "target": self.target,
            "instructions": self.instructions,
            "positive": self.positive,
            "negative": self.negative,
        }


def decomposition_questions(
    fields: Sequence[FieldSpec] | None = None,
) -> tuple[BinaryQuestion, ...]:
    """Narrow questions covering every decision field.

    Deterministic and total: the same fields always yield the same question ids
    in the same order, so artifacts and wire payloads stay comparable across
    runs.
    """
    questions: list[BinaryQuestion] = []
    for spec in fields or default_fields():
        stem = spec.question or f"What is {spec.name}?"
        if spec.kind == "choice":
            for value in spec.values:
                others = ", ".join(item for item in spec.values if item != value)
                questions.append(
                    BinaryQuestion(
                        question_id=f"{spec.name}={value}",
                        field=spec.name,
                        kind=ONE_VS_REST,
                        target=value,
                        instructions=f"{stem} Narrow question: is the answer exactly {value}?",
                        positive=value,
                        negative=f"one of: {others}",
                    )
                )
        elif spec.kind == "score":
            # "at least level" for every level above the lowest. P(level) then
            # falls out of adjacent differences, so the levels stay mutually
            # exclusive and exhaustive by construction.
            for level in spec.values[1:]:
                questions.append(
                    BinaryQuestion(
                        question_id=f"{spec.name}>={level}",
                        field=spec.name,
                        kind=THRESHOLD,
                        target=level,
                        instructions=f"{stem} Narrow question: is it at least {level}?",
                        positive=f"{level} or above",
                        negative=f"below {level}",
                    )
                )
        else:
            questions.append(
                BinaryQuestion(
                    question_id=spec.name,
                    field=spec.name,
                    kind=BOOLEAN,
                    target="True",
                    instructions=spec.question or f"Is {spec.name} true?",
                    positive=spec.name,
                    negative=f"not {spec.name}",
                )
            )
    return tuple(questions)


def question_targets(
    labels: Mapping[str, str], questions: Sequence[BinaryQuestion]
) -> dict[str, str]:
    """Supervise every narrow question from the labels we already have.

    No new annotation is required and none is invented: each question is
    answered by projecting the existing field label onto that question's
    target.
    """
    specs = field_index()
    targets: dict[str, str] = {}
    for question in questions:
        value = labels.get(question.field)
        if value is None:
            continue
        if question.kind == ONE_VS_REST:
            truth = value == question.target
        elif question.kind == THRESHOLD:
            order = list(specs[question.field].values)
            truth = order.index(value) >= order.index(question.target)
        else:
            truth = value == "True"
        targets[question.question_id] = "True" if truth else "False"
    return targets


def _monotone(exceedance: list[float]) -> list[float]:
    """Enforce non-increasing P(>= level) with a cumulative minimum.

    Independent binary heads can invert two adjacent thresholds. Cumulative
    minimum is the cheap isotonic projection here: it keeps every reconstructed
    level probability non-negative and the distribution summing to one, and it
    never invents confidence the heads did not express.
    """
    for index in range(1, len(exceedance)):
        exceedance[index] = min(exceedance[index], exceedance[index - 1])
    return exceedance


def reconstruct_field(
    spec: FieldSpec,
    probabilities: Mapping[str, float],
    questions: Sequence[BinaryQuestion],
) -> dict[str, float]:
    """Turn narrow question probabilities back into a field distribution."""
    mine = [question for question in questions if question.field == spec.name]
    if spec.kind == "boolean":
        positive = float(np.clip(probabilities.get(spec.name, 0.5), 0.0, 1.0))
        return {"True": positive, "False": 1.0 - positive}
    if spec.kind == "score":
        order = list(spec.values)
        ordered = sorted(mine, key=lambda question: order.index(question.target))
        exceedance = _monotone(
            [
                float(np.clip(probabilities.get(question.question_id, 0.0), 0.0, 1.0))
                for question in ordered
            ]
        )
        distribution = dict.fromkeys(order, 0.0)
        carried = 1.0
        for question, level_probability in zip(ordered, exceedance, strict=True):
            below = order[order.index(question.target) - 1]
            distribution[below] = carried - level_probability
            carried = level_probability
        distribution[order[-1]] = carried
        return distribution
    raw = {
        question.target: float(np.clip(probabilities.get(question.question_id, 0.0), 0.0, 1.0))
        for question in mine
    }
    total = sum(raw.values())
    if total <= 0.0:
        uniform = 1.0 / max(len(spec.values), 1)
        return {value: uniform for value in spec.values}
    return {value: raw.get(value, 0.0) / total for value in spec.values}


@dataclass
class DecomposedProvider:
    """A decision provider built from one binary head per narrow question."""

    encoder: FeatureEncoder
    heads: dict[str, LinearHead]
    questions: tuple[BinaryQuestion, ...]
    feature_mean: np.ndarray
    feature_std: np.ndarray
    metadata: dict[str, Any] = dc_field(default_factory=dict)
    name: str = "decomposed"
    decomposition_version: int = DECOMPOSITION_VERSION

    def _standardize(self, features: np.ndarray) -> np.ndarray:
        safe_std = np.where(self.feature_std > 1e-12, self.feature_std, 1.0)
        return (features - self.feature_mean) / safe_std

    def question_matrix(self, features: np.ndarray) -> np.ndarray:
        """P(true) for every narrow question, shape ``(n_cases, n_questions)``.

        Columns follow ``self.questions`` order, so the layout is stable across
        runs and artifacts regardless of head insertion order.
        """
        columns = []
        for question in self.questions:
            head = self.heads.get(question.question_id)
            if head is None:
                columns.append(np.zeros(len(features)))
                continue
            probabilities = head.probabilities(features)
            columns.append(probabilities[:, list(head.classes).index("True")])
        return np.stack(columns, axis=1) if columns else np.zeros((len(features), 0))

    def question_probabilities(self, features: np.ndarray) -> dict[str, float]:
        """P(true) per question id for a single case."""
        row = self.question_matrix(np.asarray(features)[:1])[0]
        return {
            question.question_id: float(row[index])
            for index, question in enumerate(self.questions)
        }

    def field_distributions(self, features: np.ndarray) -> dict[str, np.ndarray]:
        """Reconstructed per-field distributions, each ``(n_cases, n_values)``."""
        matrix = self.question_matrix(features)
        specs = field_index()
        fields = {question.field for question in self.questions}
        distributions: dict[str, np.ndarray] = {}
        for field_name in specs:
            if field_name not in fields:
                continue
            spec = specs[field_name]
            rows = []
            for row in matrix:
                probabilities = {
                    question.question_id: float(row[index])
                    for index, question in enumerate(self.questions)
                }
                distribution = reconstruct_field(spec, probabilities, self.questions)
                rows.append([distribution.get(value, 0.0) for value in spec.values])
            distributions[field_name] = np.asarray(rows, dtype=float)
        return distributions

    def decide(self, result: Any) -> DecisionSet:
        raw = self.encoder.encode(result)
        if isinstance(getattr(result, "machine", None), dict):
            from .evidence_package import evidence_digest

            result.machine["provider_payload_digest"] = evidence_digest(
                {
                    "provider": self.name,
                    "feature_version": self.encoder.feature_version,
                    "features": [float(value) for value in raw],
                }
            )
        features = self._standardize(raw.reshape(1, -1))
        probabilities = self.question_probabilities(features)
        specs = field_index()
        fields = {question.field for question in self.questions}
        values: dict[str, DecisionValue] = {}
        requires = False
        for field_name in specs:
            if field_name not in fields:
                continue
            spec = specs[field_name]
            distribution = reconstruct_field(spec, probabilities, self.questions)
            best = max(distribution, key=lambda key: distribution[key])
            value: Any = best
            if spec.kind == "boolean":
                value = best == "True"
                requires = bool(value)
            values[field_name] = DecisionValue(
                field=field_name,
                kind=spec.kind,
                value=value,
                probabilities=distribution,
                strategy=self.name,
                probability_kind="decomposed_binary",
                evidence=[
                    question.question_id
                    for question in self.questions
                    if question.field == field_name
                ],
                index=(
                    score_index(distribution, spec.values)
                    if spec.kind == "score"
                    else None
                ),
            )
        return DecisionSet(
            run_id=getattr(result, "run_id", ""),
            provider=self.name,
            values=values,
            requires_investigation=requires,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "decomposition_version": self.decomposition_version,
            "feature_version": self.encoder.feature_version,
            "feature_names": self.encoder.feature_names,
            "feature_mean": self.feature_mean.tolist(),
            "feature_std": self.feature_std.tolist(),
            "questions": [question.to_dict() for question in self.questions],
            "heads": {
                question_id: head.to_dict() for question_id, head in self.heads.items()
            },
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DecomposedProvider:
        version = int(data.get("decomposition_version", 0))
        if version != DECOMPOSITION_VERSION:
            raise ValueError(f"unexpected decomposition version {version}")
        feature_version = int(data.get("feature_version", 0))
        if feature_version != FEATURE_VERSION:
            raise ValueError(f"unexpected feature version {feature_version}")
        encoder = FeatureEncoder()
        if list(data.get("feature_names", [])) != list(encoder.feature_names):
            raise ValueError("feature names do not match the current encoder")
        questions = tuple(
            BinaryQuestion(**{key: str(value) for key, value in item.items()})
            for item in data["questions"]
        )
        return cls(
            encoder=encoder,
            heads={
                question_id: LinearHead.from_dict(head)
                for question_id, head in data["heads"].items()
            },
            questions=questions,
            feature_mean=np.asarray(data["feature_mean"], dtype=float),
            feature_std=np.asarray(data["feature_std"], dtype=float),
            metadata=dict(data.get("metadata", {})),
            name=str(data.get("provider", "decomposed")),
            decomposition_version=version,
        )

    @classmethod
    def load(cls, path: str | Path) -> DecomposedProvider:
        import json

        payload = json.loads((Path(path) / "provider.json").read_text())
        return cls.from_dict(payload)


def evaluate_decomposed(
    provider: DecomposedProvider, records: Sequence[LabelRecord]
) -> dict[str, Any]:
    """Score a decomposed provider on exactly the flat arm's metric surface."""
    if not records:
        return {"n": 0, "fields": {}, "overall_accuracy": 0.0}
    from .training import evaluate_field

    features = provider._standardize(
        np.asarray([record.features for record in records], dtype=float)
    )
    specs = field_index()
    distributions = provider.field_distributions(features)
    per_field: dict[str, Any] = {}
    correct_total = 0
    predictions_total = 0
    for field_name, matrix in distributions.items():
        metrics = evaluate_field(
            field_name, matrix, specs[field_name].values, records
        )
        per_field[field_name] = metrics
        correct_total += int(round(metrics["accuracy"] * metrics["n"]))
        predictions_total += metrics["n"]
    return {
        "n": len(records),
        "fields": per_field,
        "overall_accuracy": (
            correct_total / predictions_total if predictions_total else 0.0
        ),
    }


def train_decomposed_provider(
    records: Sequence[LabelRecord],
    config: Any = None,
    validation_fraction: float = 0.3,
    seed: int = 0,
    l2: float = 1.0,
    epochs: int = 400,
    learning_rate: float = 0.5,
    test_fraction: float = 0.2,
) -> tuple[DecomposedProvider, dict[str, Any]]:
    """Fit one binary head per narrow question.

    Deliberately parallel to ``train_decision_provider``: same validation, same
    ``split_records_four`` split at the same seed, same optimizer budget. Only
    the target formulation differs, so a comparison between the two isolates
    the effect of decomposition (config/decomposition-gate.json).
    """
    from .config import DatasetConfig

    if len(records) < 4:
        raise ValueError("at least four label records are required")
    config = config or DatasetConfig()
    specs = field_index()
    for record in records:
        if record.feature_version != FEATURE_VERSION:
            raise ValueError(f"unexpected feature version {record.feature_version}")
        for field_name, value in record.labels.items():
            if field_name not in specs:
                raise ValueError(f"unknown label field {field_name!r}")
            if value not in specs[field_name].values:
                raise ValueError(f"label {value!r} is not a class of {field_name!r}")

    questions = decomposition_questions()
    train_records, validation_records, _development, test_records = split_records_four(
        records,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    if not train_records or not validation_records or (
        test_fraction > 0 and not test_records
    ):
        raise ValueError(
            "INSUFFICIENT_EVIDENCE: independent train/calibration/test groups required"
        )

    train_features = np.asarray([record.features for record in train_records], dtype=float)
    feature_mean = train_features.mean(axis=0)
    feature_std = train_features.std(axis=0)
    safe_std = np.where(feature_std > 1e-12, feature_std, 1.0)
    train_standardized = (train_features - feature_mean) / safe_std
    validation_standardized = (
        np.asarray([record.features for record in validation_records], dtype=float)
        - feature_mean
    ) / safe_std

    heads: dict[str, LinearHead] = {}
    for question in questions:
        train_targets = np.asarray(
            [
                question_targets(record.labels, (question,))[question.question_id]
                for record in train_records
            ]
        )
        encoded = np.asarray(
            [BINARY_CLASSES.index(label) for label in train_targets], dtype=int
        )
        weights, bias = fit_head(
            train_standardized,
            encoded,
            n_classes=len(BINARY_CLASSES),
            l2=l2,
            epochs=epochs,
            learning_rate=learning_rate,
        )
        validation_targets = np.asarray(
            [
                BINARY_CLASSES.index(
                    question_targets(record.labels, (question,))[question.question_id]
                )
                for record in validation_records
            ],
            dtype=int,
        )
        logits = validation_standardized @ weights + bias
        temperature = fit_temperature(logits, validation_targets)
        heads[question.question_id] = LinearHead(
            field=question.question_id,
            classes=BINARY_CLASSES,
            weights=weights,
            bias=bias,
            temperature=temperature,
        )

    provider = DecomposedProvider(
        encoder=FeatureEncoder(config),
        heads=heads,
        questions=questions,
        feature_mean=feature_mean,
        feature_std=feature_std,
        metadata={
            "decomposition_version": DECOMPOSITION_VERSION,
            "feature_version": FEATURE_VERSION,
            "records": len(records),
            "question_count": len(questions),
            "warning": "Training provenance is not production eligibility; independent pinned evaluation and operational gates remain required.",
        },
    )
    metrics = {
        "train": evaluate_decomposed(provider, train_records),
        "validation": evaluate_decomposed(provider, validation_records),
        "test": evaluate_decomposed(provider, test_records),
    }
    return provider, metrics
