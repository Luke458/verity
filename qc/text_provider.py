"""Frozen-encoder text probe decision provider.

A ModernBERT-class encoder (or any HF encoder) embeds the canonical evidence
text once; per-field linear heads are trained offline on labels. This is a
third substrate beside the evidence-feature head and a remote Jev-compatible
provider: one forward pass per case, CPU-friendly, and strong when labels are
few because the representation is pretrained.

Probabilities are ``frozen_encoder_probe``: a fitted probe, not a calibration
certificate, and synthetic labels validate plumbing only.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .config import DatasetConfig
from .decisions import (
    DecisionSet,
    DecisionValue,
    LinearHead,
    default_fields,
    field_index,
    score_index,
)
from .evidence_text import EVIDENCE_TEXT_VERSION, evidence_text
from .labels import LabelRecord
from .training import _brier, _ece, fit_head, fit_temperature, split_records


class TextEmbedder(Protocol):
    name: str

    def metadata(self) -> dict[str, Any]: ...

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


@dataclass
class ModernBertEmbedder:
    """HF encoder with mean pooling over the attention mask.

    ModernBERT's 8192-token context fits the bounded evidence text without
    truncation. Weights load lazily; nothing is downloaded at import.
    """

    model_id: str = "answerdotai/ModernBERT-base"
    revision: str | None = None
    device: str = "cpu"
    max_length: int = 4096
    name: str = "modernbert"
    _tokenizer: Any = field(default=None, init=False, repr=False)
    _model: Any = field(default=None, init=False, repr=False)
    dimension: int | None = field(default=None, init=False)

    def _load(self):
        if self._model is None:
            from transformers import AutoModel, AutoTokenizer  # type: ignore

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, revision=self.revision
            )
            self._model = AutoModel.from_pretrained(
                self.model_id, revision=self.revision
            )
            self._model.to(self.device)
            self._model.eval()
        return self._tokenizer, self._model

    def metadata(self) -> dict[str, Any]:
        return {
            "kind": "hf_encoder",
            "name": self.name,
            "model_id": self.model_id,
            "revision": self.revision,
            "max_length": self.max_length,
            "dimension": self.dimension,
        }

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        import torch  # type: ignore

        tokenizer, model = self._load()
        with torch.inference_mode():
            batch = tokenizer(
                list(texts),
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            hidden = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            pooled = torch.nn.functional.normalize(pooled.float(), dim=-1)
        array = pooled.cpu().numpy()
        self.dimension = int(array.shape[1])
        return array


def default_text_embedder(metadata: dict[str, Any]) -> TextEmbedder:
    if metadata.get("kind") != "hf_encoder":
        raise ValueError(f"unsupported text embedder metadata: {metadata!r}")
    return ModernBertEmbedder(
        model_id=str(metadata["model_id"]),
        revision=metadata.get("revision"),
        max_length=int(metadata.get("max_length", 4096)),
    )


@dataclass
class TextDecisionProvider:
    embedder: TextEmbedder
    heads: dict[str, LinearHead]
    feature_mean: np.ndarray
    feature_std: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    name: str = "text_probe"

    def _standardize(self, embeddings: np.ndarray) -> np.ndarray:
        safe = np.where(self.feature_std > 1e-12, self.feature_std, 1.0)
        return (embeddings - self.feature_mean) / safe

    def decide(self, result: Any) -> DecisionSet:
        embedding = self.embedder.embed([evidence_text(result)])
        features = self._standardize(embedding)
        specs = field_index()
        values: dict[str, DecisionValue] = {}
        requires = False
        for field_name, head in self.heads.items():
            probabilities = head.probabilities(features)[0]
            best = int(np.argmax(probabilities))
            label = head.classes[best]
            spec = specs.get(field_name)
            kind = spec.kind if spec is not None else "choice"
            value: Any = label
            if kind == "boolean":
                value = label == "True"
                requires = bool(value)
            distribution = {
                class_name: float(probability)
                for class_name, probability in zip(head.classes, probabilities)
            }
            values[field_name] = DecisionValue(
                field=field_name,
                kind=kind,
                value=value,
                probabilities=distribution,
                strategy=self.name,
                probability_kind="frozen_encoder_probe",
                index=(
                    score_index(distribution, head.classes)
                    if kind == "score"
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
            "provider": "text_probe",
            "text_version": EVIDENCE_TEXT_VERSION,
            "embedder": self.embedder.metadata(),
            "feature_mean": self.feature_mean.tolist(),
            "feature_std": self.feature_std.tolist(),
            "heads": {name: head.to_dict() for name, head in self.heads.items()},
            "metadata": self.metadata,
        }

    def save(self, directory: str | Path) -> Path:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        target = path / "provider.json"
        target.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return target

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], embedder: TextEmbedder | None = None
    ) -> TextDecisionProvider:
        if int(data.get("text_version", -1)) != EVIDENCE_TEXT_VERSION:
            raise ValueError(
                f"text version mismatch: artifact {data.get('text_version')} "
                f"vs runtime {EVIDENCE_TEXT_VERSION}"
            )
        heads = {
            name: LinearHead.from_dict(payload)
            for name, payload in data["heads"].items()
        }
        return cls(
            embedder=embedder or default_text_embedder(data.get("embedder", {})),
            heads=heads,
            feature_mean=np.asarray(data["feature_mean"], dtype=float),
            feature_std=np.asarray(data["feature_std"], dtype=float),
            metadata=dict(data.get("metadata", {})),
        )

    @classmethod
    def load(
        cls, directory: str | Path, embedder: TextEmbedder | None = None
    ) -> TextDecisionProvider:
        path = Path(directory) / "provider.json"
        return cls.from_dict(json.loads(path.read_text()), embedder=embedder)


def load_decision_provider(
    directory: str | Path, embedder: TextEmbedder | None = None
):
    """Load a feature-head or text-probe artifact by its declared provider."""
    data = json.loads((Path(directory) / "provider.json").read_text())
    if data.get("provider") == "text_probe":
        return TextDecisionProvider.from_dict(data, embedder=embedder)
    from .decisions import TrainedDecisionProvider

    return TrainedDecisionProvider.from_dict(data)


def evaluate_text_provider(
    provider: TextDecisionProvider, records: Sequence[LabelRecord]
) -> dict[str, Any]:
    usable = [record for record in records if record.text]
    if not usable:
        return {"n": 0, "fields": {}, "overall_accuracy": 0.0}
    features = provider._standardize(
        provider.embedder.embed([str(record.text) for record in usable])
    )
    per_field: dict[str, Any] = {}
    correct_total = 0
    predictions_total = 0
    for field_name, head in provider.heads.items():
        targets = np.asarray(
            [
                head.classes.index(record.labels[field_name])
                if record.labels.get(field_name) in head.classes
                else -1
                for record in usable
            ]
        )
        valid = targets >= 0
        probabilities = head.probabilities(features[valid])
        valid_targets = targets[valid]
        accuracy = float(
            (probabilities.argmax(axis=1) == valid_targets).mean()
        ) if len(valid_targets) else 0.0
        per_field[field_name] = {
            "n": int(len(valid_targets)),
            "accuracy": accuracy,
            "brier": _brier(probabilities, valid_targets) if len(valid_targets) else 0.0,
            "ece": _ece(probabilities, valid_targets) if len(valid_targets) else 0.0,
        }
        correct_total += int((probabilities.argmax(axis=1) == valid_targets).sum())
        predictions_total += int(len(valid_targets))
    return {
        "n": len(usable),
        "fields": per_field,
        "overall_accuracy": (
            correct_total / predictions_total if predictions_total else 0.0
        ),
    }


def train_text_provider(
    records: Sequence[LabelRecord],
    embedder: TextEmbedder,
    config: DatasetConfig | None = None,
    validation_fraction: float = 0.3,
    seed: int = 0,
    l2: float = 1.0,
    epochs: int = 400,
    learning_rate: float = 0.5,
) -> tuple[TextDecisionProvider, dict[str, Any]]:
    usable = [record for record in records if record.text]
    if len(usable) < 4:
        raise ValueError("at least four label records with evidence text are required")
    config = config or DatasetConfig()
    fields = {spec.name: spec for spec in default_fields()}
    for record in usable:
        for field_name, value in record.labels.items():
            if field_name not in fields:
                raise ValueError(f"unknown label field {field_name!r}")
            if value not in fields[field_name].values:
                raise ValueError(f"label {value!r} is not a class of {field_name!r}")

    train_records, validation_records = split_records(
        usable, validation_fraction=validation_fraction, seed=seed
    )
    train_embeddings = embedder.embed(
        [str(record.text) for record in train_records]
    )
    mean = train_embeddings.mean(axis=0)
    std = train_embeddings.std(axis=0)
    safe_std = np.where(std > 1e-12, std, 1.0)
    train_features = (train_embeddings - mean) / safe_std
    validation_features = (
        embedder.embed([str(record.text) for record in validation_records]) - mean
    ) / safe_std

    heads: dict[str, LinearHead] = {}
    for field_name, spec in fields.items():
        targets = np.asarray(
            [spec.values.index(record.labels[field_name]) for record in train_records]
        )
        weights, bias = fit_head(
            train_features,
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
        logits = validation_features @ weights + bias
        temperature = fit_temperature(logits, validation_targets)
        heads[field_name] = LinearHead(
            field=field_name,
            classes=spec.values,
            weights=weights,
            bias=bias,
            temperature=temperature,
        )

    provider = TextDecisionProvider(
        embedder=embedder,
        heads=heads,
        feature_mean=mean,
        feature_std=std,
    )
    metrics = {
        "train": evaluate_text_provider(provider, train_records),
        "validation": evaluate_text_provider(provider, validation_records),
        "temperatures": {
            name: head.temperature for name, head in heads.items()
        },
    }
    sources = sorted({record.source for record in usable})
    production_eligible = bool(sources) and all(
        source == "analyst" for source in sources
    )
    provider.metadata = {
        "n_records": len(usable),
        "n_train": len(train_records),
        "n_validation": len(validation_records),
        "label_sources": sources,
        "families": sorted({str(record.family) for record in usable}),
        "text_version": EVIDENCE_TEXT_VERSION,
        "embedder": embedder.metadata(),
        "production_eligible": production_eligible,
        "warning": (
            "Trained on non-analyst labels (oracle/synthetic); plumbing only. "
            "Retrain on real analyst labels before relying on probabilities."
            if not production_eligible
            else "Trained on analyst labels; calibration is not certified."
        ),
    }
    return provider, metrics
