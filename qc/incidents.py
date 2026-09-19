"""Incident memory (architecture section 33).

Confirmed incidents are stored with their feature vector and symptom tags so a
new case can retrieve historically similar incidents and their resolutions.
Retrieval combines cosine similarity over the versioned feature vector with a
symptom-tag overlap bonus. Incidents saved by an agent are drafts
(``confirmed=False``) until an analyst confirms them.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .decisions import DecisionSet, FeatureEncoder


@dataclass
class IncidentRecord:
    incident_id: str
    run_id: str
    root_cause: str
    likely_origin: str
    resolution: str
    analyst_summary: str
    symptom_tags: list[str]
    features: list[float]
    feature_version: int
    affected_entities: list[str] = field(default_factory=list)
    temporal_embedding: list[float] | None = None
    confirmed: bool = False
    created_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IncidentRecord":
        return cls(
            incident_id=str(data["incident_id"]),
            run_id=str(data["run_id"]),
            root_cause=str(data.get("root_cause", "UNKNOWN")),
            likely_origin=str(data.get("likely_origin", "UNKNOWN")),
            resolution=str(data.get("resolution", "")),
            analyst_summary=str(data.get("analyst_summary", "")),
            symptom_tags=[str(tag) for tag in data.get("symptom_tags", [])],
            features=[float(value) for value in data.get("features", [])],
            feature_version=int(data.get("feature_version", 1)),
            affected_entities=[
                str(entity) for entity in data.get("affected_entities", [])
            ],
            temporal_embedding=(
                [float(v) for v in data["temporal_embedding"]]
                if data.get("temporal_embedding") is not None
                else None
            ),
            confirmed=bool(data.get("confirmed", False)),
            created_at=data.get("created_at"),
        )


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


class IncidentStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def add(self, record: IncidentRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as handle:
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")

    def load(self) -> list[IncidentRecord]:
        if not self.path.exists():
            return []
        return [
            IncidentRecord.from_dict(json.loads(line))
            for line in self.path.read_text().splitlines()
            if line.strip()
        ]

    def retrieve(
        self,
        features: Sequence[float],
        k: int = 3,
        tags: Sequence[str] | None = None,
        embedding: Sequence[float] | None = None,
        embedding_weight: float = 0.5,
    ) -> list[tuple[IncidentRecord, float]]:
        query_vector = np.asarray(features, dtype=float)
        query_tags = {str(tag) for tag in (tags or [])}
        scored: list[tuple[IncidentRecord, float]] = []
        for record in self.load():
            vector = np.asarray(record.features, dtype=float)
            if len(vector) != len(query_vector):
                continue
            score = cosine_similarity(query_vector, vector)
            if embedding is not None and record.temporal_embedding:
                embedding_score = (
                    cosine_similarity(
                        np.asarray(embedding, dtype=float),
                        np.asarray(record.temporal_embedding, dtype=float),
                    )
                    + 1.0
                ) / 2.0
                score = (
                    (1.0 - embedding_weight) * score
                    + embedding_weight * embedding_score
                )
            if query_tags:
                record_tags = set(record.symptom_tags)
                union = query_tags | record_tags
                overlap = (
                    len(query_tags & record_tags) / len(union) if union else 0.0
                )
                score += 0.1 * overlap
            scored.append((record, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:k]


def symptom_tags_for(result: Any, decisions: DecisionSet) -> list[str]:
    tags: set[str] = set()
    cause = decisions.get("likely_cause")
    origin = decisions.get("likely_origin")
    if cause is not None:
        tags.add(str(cause.value))
    if origin is not None:
        tags.add(str(origin.value))
    for reason in getattr(result, "reasons", ()) or ():
        tags.add(str(reason).split(":", 1)[0])
    temporal = getattr(result, "temporal", None)
    if temporal is not None:
        for flag in temporal.flags:
            tags.add(f"temporal:{flag}")
    return sorted(tags)


def build_incident_record(
    result: Any,
    decisions: DecisionSet,
    feedback: dict[str, Any] | None = None,
    encoder: FeatureEncoder | None = None,
) -> IncidentRecord:
    feedback = feedback or {}
    encoder = encoder or FeatureEncoder()
    cause = decisions.get("likely_cause")
    origin = decisions.get("likely_origin")
    entities: list[str] = []
    for event in getattr(result, "events", ()) or ():
        entities.append(f"{event.entity_type}:{event.entity_id}")
    return IncidentRecord(
        incident_id=str(feedback.get("incident_id") or f"inc-{result.run_id}"),
        run_id=str(result.run_id),
        root_cause=str(feedback.get("root_cause") or (cause.value if cause else "UNKNOWN")),
        likely_origin=str(
            feedback.get("likely_origin") or (origin.value if origin else "UNKNOWN")
        ),
        resolution=str(feedback.get("resolution", "")),
        analyst_summary=str(feedback.get("analyst_summary", "")),
        symptom_tags=symptom_tags_for(result, decisions),
        features=encoder.encode(result).tolist(),
        feature_version=encoder.feature_version,
        affected_entities=sorted(set(entities)),
        confirmed=bool(feedback.get("confirmed", False)),
        created_at=feedback.get("created_at"),
    )
