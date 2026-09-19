"""Typed semantic decisions over the evidence graph (Milestone D).

The runtime returns typed fields - ``likely_cause``, ``likely_origin``,
``severity`` and ``requires_investigation`` - with distributions and explicit
evidence references. Two providers share one interface:

- ``RuleDecisionProvider`` maps deterministic and temporal evidence to fields
  with heuristic probabilities. It needs no training and is the working
  provider until real analyst labels exist.
- ``TrainedDecisionProvider`` is a linear softmax head per field over the
  versioned feature encoder, fitted offline by ``qc.training``.

``probability_kind`` is ``heuristic`` for rules and ``temperature_scaled`` for
trained heads; neither is a calibration certificate.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from dataclasses import field as dc_field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .config import DatasetConfig

CAUSE_VALUES: tuple[str, ...] = (
    "MISSING_STORES",
    "MISSING_PRODUCTS",
    "SOURCE_INGESTION",
    "CODING",
    "WAREHOUSE",
    "MARKET_MOVEMENT",
    "HISTORICAL_CORRECTION",
    "RECLASSIFICATION",
    "ENTITY_MERGE",
    "BACKFILL",
    "SCHEMA_FAILURE",
    "UNKNOWN",
)
ORIGIN_VALUES: tuple[str, ...] = (
    "SOURCE",
    "PREPROCESSING",
    "CODING",
    "WAREHOUSE",
    "REPORT",
    "UNKNOWN",
)
SEVERITY_VALUES: tuple[str, ...] = ("LOW", "MEDIUM", "HIGH", "CRITICAL")
BOOLEAN_VALUES: tuple[str, ...] = ("False", "True")


@dataclass(frozen=True)
class FieldSpec:
    name: str
    kind: str  # "choice" | "boolean" | "score"
    values: tuple[str, ...]
    question: str = ""

    def classes(self) -> tuple[str, ...]:
        return self.values

    @property
    def ordered(self) -> bool:
        """Score values are ordered lowest to highest."""
        return self.kind == "score"


def default_fields() -> tuple[FieldSpec, ...]:
    return (
        FieldSpec(
            "likely_cause",
            "choice",
            CAUSE_VALUES,
            "What is the best-supported cause of the change?",
        ),
        FieldSpec(
            "likely_origin",
            "choice",
            ORIGIN_VALUES,
            "Where in the pipeline did the change first appear?",
        ),
        FieldSpec(
            "severity",
            "score",
            SEVERITY_VALUES,
            "How severe is the change for downstream consumers?",
        ),
        FieldSpec(
            "requires_investigation",
            "boolean",
            BOOLEAN_VALUES,
            "Does this case require analyst or agent investigation?",
        ),
    )


def field_index(fields: Sequence[FieldSpec] | None = None) -> dict[str, FieldSpec]:
    return {spec.name: spec for spec in (fields or default_fields())}


def score_index(probabilities: dict[str, float], values: Sequence[str]) -> float:
    """Probability-weighted level index, lowest level 0 (System One `score`)."""
    total = sum(probabilities.values()) or 1.0
    return float(
        sum(
            index * probabilities.get(level, 0.0)
            for index, level in enumerate(values)
        )
        / total
    )


@dataclass
class DecisionValue:
    field: str
    kind: str
    value: Any
    probabilities: dict[str, float] | None
    strategy: str
    probability_kind: str
    evidence: list[str] = dc_field(default_factory=list)
    index: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DecisionSet:
    run_id: str
    provider: str
    values: dict[str, DecisionValue]
    requires_investigation: bool

    def get(self, name: str) -> DecisionValue | None:
        return self.values.get(name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "provider": self.provider,
            "requires_investigation": self.requires_investigation,
            "values": {
                name: value.to_dict() for name, value in self.values.items()
            },
        }


class DecisionProvider(Protocol):
    name: str

    def decide(self, result: Any) -> DecisionSet: ...


# ---------------------------------------------------------------------------
# Feature encoder
# ---------------------------------------------------------------------------

FEATURE_VERSION = 1


def feature_version() -> int:
    """Current feature-encoding version for artifacts and label stores."""
    return FEATURE_VERSION

EVENT_CLASSES: tuple[str, ...] = (
    "NEW_ENTITY_HISTORICAL_BACKFILL",
    "NEW_ENTITY_RECENT",
    "ENTITY_REMOVED",
    "ENTITY_HISTORY_EXTENDED",
    "ENTITY_HISTORY_TRUNCATED",
    "LATEST_WEEK_MISSING",
    "POSSIBLE_RECLASSIFICATION",
)
DIVERGENCE_STAGES: tuple[str, ...] = ("source", "coded", "warehouse", "report")
TEMPORAL_FLAGS: tuple[str, ...] = (
    "forecast_lower",
    "forecast_upper",
    "robust_z",
    "seasonal_z",
    "ewma",
    "change_point",
)
STATUS_VALUES: tuple[str, ...] = (
    "PASS",
    "PASS_WITH_EXPLANATION",
    "INVESTIGATE",
    "DATA_CONTRACT_FAILURE",
)


class FeatureEncoder:
    """Versioned numeric encoding of a QC run for learned providers."""

    feature_version = FEATURE_VERSION

    def __init__(self, config: DatasetConfig | None = None):
        self.config = config or DatasetConfig()
        self._names = self._build_names()

    def _build_names(self) -> list[str]:
        names = [
            "contract_failed",
            "raw_delta_relative",
            "unexplained_relative",
            "explained_fraction",
            "material",
            "breadth",
            "reconciliation_failures",
            "temporal_anomalies",
            "temporal_min_percentile",
            "temporal_max_relative_residual",
            "cross_metric_flags",
        ]
        names += [f"event:{value}" for value in EVENT_CLASSES]
        names += [f"divergence:{value}" for value in DIVERGENCE_STAGES]
        names += ["divergence:none"]
        names += [f"temporal_flag:{value}" for value in TEMPORAL_FLAGS]
        names += [f"status:{value}" for value in STATUS_VALUES]
        return names

    @property
    def feature_names(self) -> list[str]:
        return list(self._names)

    def encode(self, result: Any) -> np.ndarray:
        values: dict[str, float] = {name: 0.0 for name in self._names}
        contracts = getattr(result, "contracts", None)
        if contracts is not None and contracts.status == "DATA_CONTRACT_FAILURE":
            values["contract_failed"] = 1.0

        attribution = getattr(result, "attribution", None)
        previous_total = 0.0
        if attribution is not None:
            previous_total = float(attribution.previous_total)
            denominator = previous_total if abs(previous_total) > 1e-12 else 1.0
            values["raw_delta_relative"] = float(attribution.raw_delta) / denominator
            values["unexplained_relative"] = (
                float(attribution.unexplained_delta) / denominator
            )
            values["explained_fraction"] = float(attribution.explained_fraction)
            values["material"] = 1.0 if attribution.material else 0.0
            values["breadth"] = float(attribution.breadth)
            if attribution.cross_metric_flags:
                values["cross_metric_flags"] = float(
                    len(attribution.cross_metric_flags)
                )

        reconciliation = getattr(result, "reconciliation", None)
        if reconciliation is not None:
            values["reconciliation_failures"] = float(len(reconciliation.failed))

        event_counts: dict[str, int] = {}
        for event in getattr(result, "events", ()) or ():
            event_counts[event.classification] = (
                event_counts.get(event.classification, 0) + 1
            )
        for name, count in event_counts.items():
            key = f"event:{name}"
            if key in values:
                values[key] = float(count)

        lineage = getattr(result, "lineage", None)
        first = lineage.first_divergence if lineage is not None else None
        if first in DIVERGENCE_STAGES:
            values[f"divergence:{first}"] = 1.0
        else:
            values["divergence:none"] = 1.0

        temporal = getattr(result, "temporal", None)
        if temporal is not None:
            values["temporal_anomalies"] = float(
                sum(1 for item in temporal.series if item.anomaly)
            )
            percentiles = [
                item.calibrated_percentile
                for item in temporal.series
                if item.calibrated_percentile is not None
            ]
            values["temporal_min_percentile"] = (
                float(min(percentiles)) if percentiles else 0.5
            )
            residuals = [
                abs(float(item.relative_residual)) for item in temporal.series
            ]
            values["temporal_max_relative_residual"] = (
                float(max(residuals)) if residuals else 0.0
            )
            for item in temporal.series:
                for flag in item.flags:
                    key = f"temporal_flag:{flag}"
                    if key in values:
                        values[key] += 1.0

        historical_status = None
        machine = getattr(result, "machine", None)
        if isinstance(machine, dict) and machine.get("historical_revision"):
            historical_status = machine["historical_revision"].get("status")
        elif contracts is not None and contracts.status == "DATA_CONTRACT_FAILURE":
            historical_status = "DATA_CONTRACT_FAILURE"
        if historical_status in STATUS_VALUES:
            values[f"status:{historical_status}"] = 1.0

        return np.asarray([values[name] for name in self._names], dtype=float)


# ---------------------------------------------------------------------------
# Rule provider
# ---------------------------------------------------------------------------


def _heuristic_distribution(
    chosen: str, values: Sequence[str], strength: float
) -> dict[str, float]:
    others = [value for value in values if value != chosen]
    if not others:
        return {chosen: 1.0}
    chosen_probability = min(max(float(strength), 0.05), 0.95)
    remainder = (1.0 - chosen_probability) / len(others)
    distribution = {value: remainder for value in others}
    distribution[chosen] = chosen_probability
    return distribution


def _stage_origin(stage: str) -> str:
    return {
        "source": "SOURCE",
        "preprocessing": "PREPROCESSING",
        "coded": "CODING",
        "warehouse": "WAREHOUSE",
        "report": "REPORT",
    }.get(stage, "UNKNOWN")


class RuleDecisionProvider:
    """Evidence-rule decisions with heuristic probabilities.

    The rules mirror the deterministic semantics: registered structural changes
    do not require investigation, unregistered ones do, material unexplained
    changes escalate, and a temporal-only movement is reported as market
    movement rather than a pipeline defect.
    """

    name = "rule"

    def __init__(self, config: DatasetConfig | None = None):
        self.config = config or DatasetConfig()

    def decide(self, result: Any) -> DecisionSet:
        cause, origin, severity, requires, strength, evidence = self._infer(result)
        distributions = {
            "likely_cause": _heuristic_distribution(cause, CAUSE_VALUES, strength),
            "likely_origin": _heuristic_distribution(origin, ORIGIN_VALUES, strength),
            "severity": _heuristic_distribution(severity, SEVERITY_VALUES, strength),
            "requires_investigation": _heuristic_distribution(
                "True" if requires else "False",
                BOOLEAN_VALUES,
                0.8,
            ),
        }
        fields = field_index()
        values: dict[str, DecisionValue] = {}
        for name in ("likely_cause", "likely_origin", "severity", "requires_investigation"):
            spec = fields[name]
            if name == "requires_investigation":
                value: Any = bool(requires)
            elif name == "likely_cause":
                value = cause
            elif name == "likely_origin":
                value = origin
            else:
                value = severity
            distribution = distributions[name]
            values[name] = DecisionValue(
                field=name,
                kind=spec.kind,
                value=value,
                probabilities=distribution,
                strategy=self.name,
                probability_kind="heuristic",
                evidence=list(evidence),
                index=(
                    score_index(distribution, spec.values)
                    if spec.ordered
                    else None
                ),
            )
        return DecisionSet(
            run_id=getattr(result, "run_id", ""),
            provider=self.name,
            values=values,
            requires_investigation=requires,
        )

    def _infer(self, result: Any) -> tuple[str, str, str, bool, float, list[str]]:
        config = self.config
        contracts = getattr(result, "contracts", None)
        if contracts is not None and contracts.status == "DATA_CONTRACT_FAILURE":
            failed = contracts.failed
            stage = failed[0].name.split(":", 1)[0] if failed else "report"
            evidence: list[str] = [
                f"contract:{check.name}" for check in failed
            ]
            return (
                "SCHEMA_FAILURE",
                _stage_origin(stage),
                "HIGH",
                True,
                0.75,
                evidence,
            )

        events = list(getattr(result, "events", ()) or ())
        by_class: dict[str, list[Any]] = {}
        for event in events:
            by_class.setdefault(event.classification, []).append(event)
        attribution = getattr(result, "attribution", None)
        lineage = getattr(result, "lineage", None)
        temporal = getattr(result, "temporal", None)
        first = lineage.first_divergence if lineage is not None else None
        evidence = []

        merge_candidates = [
            relationship
            for relationship in (getattr(result, "relationships", ()) or ())
            if relationship.relationship
            in ("replaced_by", "merged_into", "superseded_by")
        ]
        if merge_candidates:
            evidence = [
                f"relationship:{relationship.source_id}->{relationship.target_id}"
                for relationship in merge_candidates
            ]
            # A merge candidate always needs confirmation.
            return "ENTITY_MERGE", "SOURCE", "MEDIUM", True, 0.6, evidence

        latest_missing = by_class.get("LATEST_WEEK_MISSING", [])
        stores_missing = [e for e in latest_missing if e.entity_type == "store"]
        products_missing = [e for e in latest_missing if e.entity_type == "product"]
        if stores_missing:
            evidence = [f"event:{e.entity_type}:{e.entity_id}" for e in stores_missing]
            return "MISSING_STORES", "SOURCE", "HIGH", True, 0.8, evidence
        if products_missing:
            evidence = [f"event:{e.entity_type}:{e.entity_id}" for e in products_missing]
            return "MISSING_PRODUCTS", "SOURCE", "HIGH", True, 0.8, evidence

        reclass = by_class.get("POSSIBLE_RECLASSIFICATION", [])
        if reclass:
            evidence = [f"event:{e.entity_type}:{e.entity_id}" for e in reclass]
            unmatched = bool(attribution and attribution.unmatched_events)
            return "RECLASSIFICATION", "SOURCE", "MEDIUM", unmatched, 0.6, evidence

        truncated = by_class.get("ENTITY_HISTORY_TRUNCATED", []) + by_class.get(
            "ENTITY_REMOVED", []
        )
        if truncated:
            evidence = [f"event:{e.entity_type}:{e.entity_id}" for e in truncated]
            unmatched = bool(attribution and attribution.unmatched_events)
            return "HISTORICAL_CORRECTION", "SOURCE", "MEDIUM", unmatched, 0.6, evidence

        backfill = by_class.get("NEW_ENTITY_HISTORICAL_BACKFILL", [])
        if backfill:
            evidence = [f"event:{e.entity_type}:{e.entity_id}" for e in backfill]
            unmatched = bool(attribution and attribution.unmatched_events)
            strength = 0.65 if not unmatched else 0.6
            return "BACKFILL", "SOURCE", "MEDIUM", unmatched, strength, evidence

        previous_total = float(attribution.previous_total) if attribution else 0.0
        denominator = previous_total if abs(previous_total) > 1e-12 else 1.0
        unexplained_relative = (
            float(attribution.unexplained_delta) / denominator if attribution else 0.0
        )
        unexplained_material = bool(
            attribution
            and attribution.material
            and attribution.explained_fraction < config.explained_fraction_threshold
        )
        if unexplained_material:
            if first == "coded":
                return "CODING", "CODING", self._severity(unexplained_relative), True, 0.7, ["lineage:coded"]
            if first == "warehouse":
                return "WAREHOUSE", "WAREHOUSE", self._severity(unexplained_relative), True, 0.7, ["lineage:warehouse"]
            if first == "source":
                return "SOURCE_INGESTION", "SOURCE", self._severity(unexplained_relative), True, 0.6, ["lineage:source"]
            return "UNKNOWN", "UNKNOWN", self._severity(unexplained_relative), True, 0.5, ["attribution"]

        if attribution is not None and not attribution.material and attribution.breadth > config.broad_recalculation_breadth:
            return "HISTORICAL_CORRECTION", "SOURCE", "LOW", True, 0.55, ["attribution:breadth"]

        if temporal is not None and temporal.anomaly:
            evidence = [
                f"temporal:{item.series_id}"
                for item in temporal.series
                if item.anomaly
            ]
            return "MARKET_MOVEMENT", "UNKNOWN", "LOW", False, 0.6, evidence

        return "UNKNOWN", "UNKNOWN", "LOW", False, 0.4, []

    @staticmethod
    def _severity(relative: float) -> str:
        magnitude = abs(relative)
        if magnitude >= 0.10:
            return "HIGH"
        if magnitude >= 0.02:
            return "MEDIUM"
        return "LOW"


# ---------------------------------------------------------------------------
# Trained provider
# ---------------------------------------------------------------------------


@dataclass
class LinearHead:
    field: str
    classes: tuple[str, ...]
    weights: np.ndarray
    bias: np.ndarray
    temperature: float = 1.0

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        logits = features @ self.weights + self.bias
        scaled = logits / max(self.temperature, 1e-6)
        scaled = scaled - scaled.max(axis=-1, keepdims=True)
        exponentials = np.exp(scaled)
        return exponentials / exponentials.sum(axis=-1, keepdims=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "classes": list(self.classes),
            "weights": self.weights.tolist(),
            "bias": self.bias.tolist(),
            "temperature": float(self.temperature),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LinearHead:
        return cls(
            field=str(data["field"]),
            classes=tuple(str(c) for c in data["classes"]),
            weights=np.asarray(data["weights"], dtype=float),
            bias=np.asarray(data["bias"], dtype=float),
            temperature=float(data.get("temperature", 1.0)),
        )


@dataclass
class TrainedDecisionProvider:
    encoder: FeatureEncoder
    heads: dict[str, LinearHead]
    feature_mean: np.ndarray
    feature_std: np.ndarray
    metadata: dict[str, Any] = dc_field(default_factory=dict)
    name: str = "trained"

    def _standardize(self, features: np.ndarray) -> np.ndarray:
        safe_std = np.where(self.feature_std > 1e-12, self.feature_std, 1.0)
        return (features - self.feature_mean) / safe_std

    def decide(self, result: Any) -> DecisionSet:
        raw = self.encoder.encode(result)
        features = self._standardize(raw.reshape(1, -1))
        specs = field_index()
        values: dict[str, DecisionValue] = {}
        requires = False
        for field_name, head in self.heads.items():
            probabilities = head.probabilities(features)[0]
            best = int(np.argmax(probabilities))
            label = head.classes[best]
            value: Any = label
            spec = specs.get(field_name)
            kind = spec.kind if spec is not None else "choice"
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
                probability_kind="temperature_scaled",
                evidence=[],
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
        return DecisionSet(
            run_id=getattr(result, "run_id", ""),
            provider=self.name,
            values=values,
            requires_investigation=requires,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "feature_version": self.encoder.feature_version,
            "feature_names": self.encoder.feature_names,
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
    def from_dict(cls, data: dict[str, Any]) -> TrainedDecisionProvider:
        encoder = FeatureEncoder()
        if int(data.get("feature_version", -1)) != encoder.feature_version:
            raise ValueError(
                f"feature version mismatch: artifact {data.get('feature_version')} "
                f"vs runtime {encoder.feature_version}"
            )
        if list(data.get("feature_names", [])) != encoder.feature_names:
            raise ValueError("feature names do not match the runtime encoder")
        heads = {
            name: LinearHead.from_dict(payload)
            for name, payload in data["heads"].items()
        }
        return cls(
            encoder=encoder,
            heads=heads,
            feature_mean=np.asarray(data["feature_mean"], dtype=float),
            feature_std=np.asarray(data["feature_std"], dtype=float),
            metadata=dict(data.get("metadata", {})),
        )

    @classmethod
    def load(cls, directory: str | Path) -> TrainedDecisionProvider:
        path = Path(directory) / "provider.json"
        return cls.from_dict(json.loads(path.read_text()))
