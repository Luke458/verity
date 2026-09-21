"""Lightweight client for the optional pinned localhost Laya service."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .decisions import DecisionSet, default_fields
from .systemone import (
    SystemOneDecisionProvider,
    _questions,
    answers_to_decisions,
    build_evidence_state,
)

MODEL = "convaiinnovations/laya"
REVISION = "1c5edc17a7acd8701df6fc341c0d179f1c62c982"


@dataclass
class LayaDecisionProvider(SystemOneDecisionProvider):
    url: str = "http://127.0.0.1:8766/v1/systemone"
    model: str = MODEL
    name: str = "laya"
    state_limit: int = 128 * 1024
    artifact_identity: dict[str, Any] = field(
        default_factory=lambda: {
            "provider": "laya",
            "model": MODEL,
            "revision": REVISION,
            "runtime": "laya==0.3.4",
            "evidence_version": 2,
            "weights_sha256": "891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c",
        }
    )

    def decide(self, result: Any) -> DecisionSet:
        questions = _questions(default_fields())
        questions["likely_cause"]["criteria"] = {
            "MISSING_STORES": "absent stores",
            "MISSING_PRODUCTS": "absent products",
            "SOURCE_INGESTION": "bad delivery",
            "CODING": "bad mapping",
            "WAREHOUSE": "bad transform",
            "MARKET_MOVEMENT": "independent market evidence",
            "HISTORICAL_CORRECTION": "approved history correction",
            "RECLASSIFICATION": "category moved",
            "ENTITY_MERGE": "entities consolidated",
            "BACKFILL": "historical onboarding",
            "SCHEMA_FAILURE": "broken contract",
            "UNKNOWN": "unsupported cause",
        }
        questions["likely_origin"]["criteria"] = {
            name: (
                "no evidenced origin" if name == "UNKNOWN" else f"{name.lower()} stage"
            )
            for name in questions["likely_origin"]["criteria"]
        }
        questions["likely_cause"]["instructions"] = (
            "Which cause has independent QC evidence? Choose UNKNOWN if unexplained."
        )
        questions["likely_origin"]["instructions"] = (
            "First evidenced pipeline failure stage; UNKNOWN without evidence."
        )
        questions["severity"]["criteria"] = [
            "LOW: limited impact",
            "MEDIUM: analyst review",
            "HIGH: material corruption",
            "CRITICAL: widespread unusable data",
        ]
        questions["requires_investigation"]["instructions"] = (
            "Does a failed or incomplete required check need analyst review?"
        )
        response = self._post(
            {
                "model": MODEL,
                "state": build_evidence_state(result, self.state_limit),
                "questions": questions,
            }
        )
        if response.get("abstained"):
            raise ValueError("Laya abstained")
        metadata = response.get("metadata", {})
        if metadata.get("revision") != REVISION or metadata.get("model") != MODEL:
            raise ValueError("Laya model identity mismatch")
        result.machine["provider_response"] = response
        return answers_to_decisions(
            response["answers"], self.fields, result.run_id, provider=self.name
        )
