"""Training and evaluation for learned decision providers.

Fits one linear softmax head per field over the versioned feature encoder and
fits a scalar temperature on a held-out split. Metrics are reported per field
(accuracy, multiclass Brier, top-label ECE) and overall. Artifacts record their
training provenance; a provider trained on synthetic oracle labels must not be
presented as validated on real refresh behaviour.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .config import DatasetConfig
from .conformal import wilson_interval
from .decisions import (
    FEATURE_VERSION,
    FeatureEncoder,
    LinearHead,
    TrainedDecisionProvider,
    default_fields,
)
from .labels import LabelRecord


def split_records(
    records: Sequence[LabelRecord],
    validation_fraction: float = 0.3,
    seed: int = 0,
) -> tuple[list[LabelRecord], list[LabelRecord]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    order = np.random.default_rng(seed).permutation(len(records))
    cut = max(1, int(round(len(records) * (1.0 - validation_fraction))))
    train = [records[int(i)] for i in order[:cut]]
    validation = [records[int(i)] for i in order[cut:]]
    if not validation:
        validation = train[-1:]
        train = train[:-1] or train
    return train, validation


def _group_key(record: LabelRecord) -> str:
    """Group sibling scenarios so they cannot straddle a split."""
    suite_id = record.metadata.get("suite_id")
    family = record.family or record.metadata.get("scenario_id")
    return f"{suite_id or '-'}:{family or record.run_id}"


def split_records_grouped(
    records: Sequence[LabelRecord],
    validation_fraction: float = 0.3,
    test_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[list[LabelRecord], list[LabelRecord], list[LabelRecord]]:
    """Group-aware train/calibration/test split.

    Sibling scenarios from one suite and fault family stay together, so the
    calibration temperature and the reported test metrics are not fitted to
    near-duplicates of the training cases.
    """
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    if not 0.0 <= test_fraction < 1.0:
        raise ValueError("test_fraction must be in [0, 1)")
    if validation_fraction + test_fraction >= 1.0:
        raise ValueError("validation_fraction + test_fraction must be < 1")
    groups: dict[str, list[LabelRecord]] = {}
    for record in records:
        groups.setdefault(_group_key(record), []).append(record)
    keys = sorted(groups)
    order = [keys[int(i)] for i in np.random.default_rng(seed).permutation(len(keys))]

    n = len(records)
    target_test = int(round(n * test_fraction))
    target_validation = int(round(n * validation_fraction))
    test: list[LabelRecord] = []
    validation: list[LabelRecord] = []
    train: list[LabelRecord] = []
    for key in order:
        bucket = groups[key]
        if len(test) < target_test and len(train) + len(validation) > 0:
            test.extend(bucket)
        elif len(validation) < target_validation and len(train) > 0:
            validation.extend(bucket)
        else:
            train.extend(bucket)
    if not train:
        train = validation or test
        validation = []
        test = []
    if not test and not validation:
        # Everything landed in train (tiny input): fall back to a row split.
        train, validation = split_records(
            records, validation_fraction=validation_fraction, seed=seed
        )
    return train, validation, test


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=-1, keepdims=True)


def fit_head(
    features: np.ndarray,
    targets: np.ndarray,
    n_classes: int,
    l2: float = 1.0,
    epochs: int = 400,
    learning_rate: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    n_samples, n_features = features.shape
    weights = np.zeros((n_features, n_classes))
    bias = np.zeros(n_classes)
    one_hot = np.zeros((n_samples, n_classes))
    one_hot[np.arange(n_samples), targets] = 1.0
    for _ in range(epochs):
        probabilities = _softmax(features @ weights + bias)
        gradient = features.T @ (probabilities - one_hot) / n_samples
        gradient += (l2 / n_samples) * weights
        bias_gradient = (probabilities - one_hot).mean(axis=0)
        weights -= learning_rate * gradient
        bias -= learning_rate * bias_gradient
    return weights, bias


def fit_temperature(
    logits: np.ndarray,
    targets: np.ndarray,
    grid: Sequence[float] | None = None,
) -> float:
    grid = grid or (0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0)
    best_temperature = 1.0
    best_nll = float("inf")
    for temperature in grid:
        probabilities = _softmax(logits / max(temperature, 1e-6))
        picked = probabilities[np.arange(len(targets)), targets]
        nll = float(-np.log(np.clip(picked, 1e-12, 1.0)).mean())
        if nll < best_nll:
            best_nll = nll
            best_temperature = float(temperature)
    return best_temperature


def _brier(probabilities: np.ndarray, targets: np.ndarray) -> float:
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(len(targets)), targets] = 1.0
    return float(((probabilities - one_hot) ** 2).sum(axis=1).mean())


def _ece(probabilities: np.ndarray, targets: np.ndarray, bins: int = 10) -> float:
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = (predictions == targets).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(targets)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidences > low) & (confidences <= high)
        count = int(mask.sum())
        if count == 0:
            continue
        gap = abs(correct[mask].mean() - confidences[mask].mean())
        ece += gap * count / total
    return float(ece)


def evaluate_decision_provider(
    provider: TrainedDecisionProvider,
    records: Sequence[LabelRecord],
) -> dict[str, Any]:
    if not records:
        return {"n": 0, "fields": {}, "overall_accuracy": 0.0}
    features = provider._standardize(
        np.asarray([record.features for record in records], dtype=float)
    )
    per_field: dict[str, Any] = {}
    correct_total = 0
    predictions_total = 0
    for field_name, head in provider.heads.items():
        # A label outside the head's classes is a miss, not a free pass: it is
        # counted in the denominator and never in the numerator.
        targets = np.asarray(
            [
                head.classes.index(record.labels[field_name])
                if record.labels.get(field_name) in head.classes
                else -1
                for record in records
            ]
        )
        valid = targets >= 0
        probabilities = head.probabilities(features)
        predictions = probabilities.argmax(axis=1)
        correct = (predictions == targets) & valid
        n_total = len(records)
        n_correct = int(correct.sum())
        ci_low, ci_high = wilson_interval(n_correct, n_total)
        valid_targets = targets[valid]
        per_field[field_name] = {
            "n": n_total,
            "n_unmapped": int((~valid).sum()),
            "accuracy": n_correct / n_total if n_total else 0.0,
            "accuracy_ci_low": ci_low,
            "accuracy_ci_high": ci_high,
            "brier": (
                _brier(probabilities[valid], valid_targets)
                if len(valid_targets)
                else None
            ),
            "ece": (
                _ece(probabilities[valid], valid_targets)
                if len(valid_targets)
                else None
            ),
        }
        correct_total += n_correct
        predictions_total += n_total
    return {
        "n": len(records),
        "fields": per_field,
        "overall_accuracy": (
            correct_total / predictions_total if predictions_total else 0.0
        ),
    }


def train_decision_provider(
    records: Sequence[LabelRecord],
    config: DatasetConfig | None = None,
    validation_fraction: float = 0.3,
    seed: int = 0,
    l2: float = 1.0,
    epochs: int = 400,
    learning_rate: float = 0.5,
    test_fraction: float = 0.2,
) -> tuple[TrainedDecisionProvider, dict[str, Any]]:
    if len(records) < 4:
        raise ValueError("at least four label records are required")
    config = config or DatasetConfig()
    fields = {spec.name: spec for spec in default_fields()}
    for record in records:
        if record.feature_version != FEATURE_VERSION:
            raise ValueError(
                f"unexpected feature version {record.feature_version}"
            )
        for field_name, value in record.labels.items():
            if field_name not in fields:
                raise ValueError(f"unknown label field {field_name!r}")
            if value not in fields[field_name].values:
                raise ValueError(
                    f"label {value!r} is not a class of {field_name!r}"
                )

    train_records, validation_records, test_records = split_records_grouped(
        records,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    train_features = np.asarray(
        [record.features for record in train_records], dtype=float
    )
    feature_mean = train_features.mean(axis=0)
    feature_std = train_features.std(axis=0)
    safe_std = np.where(feature_std > 1e-12, feature_std, 1.0)
    train_standardized = (train_features - feature_mean) / safe_std
    validation_standardized = (
        np.asarray([record.features for record in validation_records], dtype=float)
        - feature_mean
    ) / safe_std

    heads: dict[str, LinearHead] = {}
    for field_name, spec in fields.items():
        targets = np.asarray(
            [spec.values.index(record.labels[field_name]) for record in train_records]
        )
        weights, bias = fit_head(
            train_standardized,
            targets,
            n_classes=len(spec.values),
            l2=l2,
            epochs=epochs,
            learning_rate=learning_rate,
        )
        validation_targets = np.asarray(
            [
                spec.values.index(record.labels[field_name])
                for record in validation_records
            ]
        )
        logits = validation_standardized @ weights + bias
        temperature = fit_temperature(logits, validation_targets)
        heads[field_name] = LinearHead(
            field=field_name,
            classes=spec.values,
            weights=weights,
            bias=bias,
            temperature=temperature,
        )

    provider = TrainedDecisionProvider(
        encoder=FeatureEncoder(config),
        heads=heads,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )
    metrics = {
        "train": evaluate_decision_provider(provider, train_records),
        "validation": evaluate_decision_provider(provider, validation_records),
        "test": evaluate_decision_provider(provider, test_records),
        "temperatures": {
            name: head.temperature for name, head in heads.items()
        },
    }
    sources = sorted({record.source for record in records})
    families = sorted({str(record.family) for record in records})
    production_eligible = bool(sources) and all(
        source == "analyst" for source in sources
    )
    provider.metadata = {
        "n_records": len(records),
        "n_train": len(train_records),
        "n_validation": len(validation_records),
        "n_test": len(test_records),
        "label_sources": sources,
        "families": families,
        "feature_version": provider.encoder.feature_version,
        "production_eligible": production_eligible,
        "warning": (
            "Trained on non-analyst labels (oracle/synthetic); plumbing only. "
            "Retrain on real analyst labels before relying on probabilities."
            if not production_eligible
            else "Trained on analyst labels; probability calibration is not "
            "certified."
        ),
    }
    return provider, metrics


def save_training_run(
    provider: TrainedDecisionProvider,
    metrics: dict[str, Any],
    directory: str | Path,
) -> Path:
    path = Path(directory)
    provider.save(path)
    (path / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True)
    )
    return path
