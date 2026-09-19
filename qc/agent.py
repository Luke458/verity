"""Investigation handoff to a reasoning agent (architecture section 18).

The deterministic and temporal layers produce an evidence graph and typed
decisions; when investigation is required, they are packaged into a structured
``InvestigationBrief`` containing candidate causes, findings, open questions,
recommended first queries, similar historical incidents and a response schema.
An ``InvestigationAgent`` consumes the brief and returns a validated
``InvestigationResult``.

``CommandAgent`` runs any external command (for example an LLM CLI) with the
brief JSON on stdin and parses the JSON response from stdout, so no provider is
built into the engine.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from .decisions import DecisionSet

RESPONSE_SCHEMA: dict[str, Any] = {
    "root_cause": "string",
    "confidence": "float in [0, 1]",
    "evidence_ids": "list of evidence ids cited",
    "recommended_actions": "list of strings",
    "summary": "string",
    "follow_up_questions": "list of strings",
}

CAUSE_QUESTIONS: dict[str, list[str]] = {
    "MISSING_STORES": [
        "Which stores are present in the previous period but absent now, and at which pipeline stage do they first disappear?",
        "Is the deficit confined to the source extract or does it persist through coding and warehouse stages?",
    ],
    "MISSING_PRODUCTS": [
        "Which products are absent for the affected entities, and is the assortment change registered?",
    ],
    "SOURCE_INGESTION": [
        "Which source delivery, file or job run produced this version, and did it complete normally?",
        "Does the divergence appear at the source stage before any transformation?",
    ],
    "CODING": [
        "Which mapping or coding change was applied in this refresh, and who requested it?",
    ],
    "WAREHOUSE": [
        "Which transformation changed at the warehouse stage, and does it match a deployment or config change?",
    ],
    "MARKET_MOVEMENT": [
        "Is the movement isolated to commodities with plausible external drivers (seasonality, promotions, competitor activity)?",
    ],
    "HISTORICAL_CORRECTION": [
        "Is this correction registered, and does its expected history window match what was observed?",
    ],
    "RECLASSIFICATION": [
        "Which product-to-commodity remapping was applied, and was it intentional?",
    ],
    "BACKFILL": [
        "Was this entity onboarding registered with the expected history window?",
    ],
    "SCHEMA_FAILURE": [
        "Which upstream contract or schema change caused the failure, and when was it deployed?",
    ],
    "UNKNOWN": [
        "Which pipeline stages can be ruled out by comparing stage-level totals?",
        "Are there similar historical incidents with a confirmed cause?",
    ],
}

CAUSE_QUERIES: dict[str, list[str]] = {
    "MISSING_STORES": [
        "SELECT store_id, MIN(week), MAX(week) FROM <fact_table> WHERE week = {week} GROUP BY store_id; compare with the prior week and the expected store list.",
        "Compare DISTINCT store_id counts per stage for week {week} against week {previous_week}.",
    ],
    "MISSING_PRODUCTS": [
        "SELECT product_id FROM <fact_table> WHERE week = {week} GROUP BY product_id; diff against week {previous_week}.",
    ],
    "SOURCE_INGESTION": [
        "Compare stage fingerprints (row_count, store_count, dollar) for week {week} across source, coded and warehouse snapshots.",
        "Check the source delivery manifest for week {week} for missing files or partial loads.",
    ],
    "CODING": [
        "Compare source and coded values for affected products for overlapping weeks around {week}.",
    ],
    "WAREHOUSE": [
        "Recompute warehouse totals from coded inputs for the affected commodities and compare with the published values.",
    ],
    "MARKET_MOVEMENT": [
        "Compare week {week} movement across commodities and banners to see whether it is broad-based or concentrated.",
    ],
    "HISTORICAL_CORRECTION": [
        "List the weeks and entities covered by the correction and compare with the expected-event registry window.",
    ],
    "RECLASSIFICATION": [
        "List products whose commodity mapping changed between versions and compute the offsetting commodity deltas.",
    ],
    "BACKFILL": [
        "Check the expected-event registry for {entities} and compare the registered history window.",
    ],
    "SCHEMA_FAILURE": [
        "Diff the delivered schema against the contract for the failing stage.",
    ],
    "UNKNOWN": [
        "Build the per-stage fingerprint table for the overlapping weeks and locate the first divergence.",
    ],
}


CAUSE_TOOL_PLANS: dict[str, list[tuple[str, dict[str, Any]]]] = {
    "MISSING_STORES": [
        ("lifecycle_changes", {"classification": "LATEST_WEEK_MISSING"}),
        ("historical_residuals", {}),
        ("temporal_anomalies", {}),
    ],
    "MISSING_PRODUCTS": [
        ("lifecycle_changes", {"classification": "LATEST_WEEK_MISSING"}),
        ("historical_residuals", {}),
    ],
    "CODING": [
        ("historical_residuals", {}),
        ("first_divergence", {}),
        ("contributors", {}),
    ],
    "WAREHOUSE": [
        ("historical_residuals", {}),
        ("first_divergence", {}),
        ("contributors", {}),
    ],
    "SOURCE_INGESTION": [
        ("first_divergence", {}),
        ("historical_residuals", {}),
        ("contributors", {}),
    ],
    "HISTORICAL_CORRECTION": [
        ("lifecycle_changes", {}),
        ("historical_residuals", {}),
    ],
    "RECLASSIFICATION": [
        ("relationships", {}),
        ("lifecycle_changes", {}),
        ("contributors", {}),
    ],
    "BACKFILL": [
        ("lifecycle_changes", {}),
        ("relationships", {}),
    ],
    "SCHEMA_FAILURE": [("contract_failures", {})],
    "MARKET_MOVEMENT": [("temporal_anomalies", {})],
    "UNKNOWN": [
        ("first_divergence", {}),
        ("historical_residuals", {}),
        ("temporal_anomalies", {}),
        ("contributors", {}),
    ],
}


@dataclass
class InvestigationBrief:
    run_id: str
    status: str
    question: str
    candidate_causes: list[dict[str, Any]] = field(default_factory=list)
    findings: list[dict[str, Any]] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    recommended_queries: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    similar_incidents: list[dict[str, Any]] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    constraints: dict[str, Any] = field(default_factory=dict)
    response_schema: dict[str, Any] = field(
        default_factory=lambda: dict(RESPONSE_SCHEMA)
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class InvestigationResult:
    run_id: str
    agent: str
    root_cause: str
    confidence: float
    evidence_ids: list[str]
    recommended_actions: list[str]
    summary: str
    follow_up_questions: list[str]
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _findings(result: Any) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    contracts = getattr(result, "contracts", None)
    if contracts is not None:
        for check in contracts.failed:
            findings.append(
                {"type": "contract_check", "name": check.name, "detail": check.detail}
            )
    for event in getattr(result, "events", ()) or ():
        findings.append(
            {
                "type": "lifecycle_event",
                "entity": f"{event.entity_type}:{event.entity_id}",
                "classification": event.classification,
                "weeks_removed": list(event.historical_weeks_removed),
                "weeks_added": list(event.historical_weeks_added),
            }
        )
    attribution = getattr(result, "attribution", None)
    if attribution is not None:
        findings.append(
            {
                "type": "attribution",
                "raw_delta": attribution.raw_delta,
                "explained_delta": attribution.explained_delta,
                "unexplained_delta": attribution.unexplained_delta,
                "explained_fraction": attribution.explained_fraction,
                "material": attribution.material,
                "matched_expected_events": list(attribution.matched_event_ids),
            }
        )
    counterfactual = getattr(result, "counterfactual", None)
    if counterfactual is not None:
        findings.append(
            {
                "type": "counterfactual",
                "reconciliation_score": counterfactual.reconciliation_score,
                "reconstructed_delta": counterfactual.reconstructed_delta,
            }
        )
    lineage = getattr(result, "lineage", None)
    if lineage is not None:
        findings.append(
            {
                "type": "lineage",
                "first_divergence": lineage.first_divergence,
                "divergences": lineage.divergences,
            }
        )
    temporal = getattr(result, "temporal", None)
    if temporal is not None:
        for item in temporal.series:
            if not item.anomaly and item.series_id != "national":
                continue
            findings.append(
                {
                    "type": "temporal",
                    "series_id": item.series_id,
                    "week": item.target_week,
                    "actual": item.actual,
                    "adjusted_actual": item.adjusted_actual,
                    "forecast_median": item.forecast_median,
                    "relative_residual": item.relative_residual,
                    "calibrated_percentile": item.calibrated_percentile,
                    "anomaly": item.anomaly,
                    "flags": item.flags,
                }
            )
    return findings


def build_investigation_brief(
    result: Any,
    decisions: DecisionSet,
    incidents: Sequence[tuple[Any, float]] = (),
    previous_week: int | None = None,
) -> InvestigationBrief:
    cause_value = decisions.get("likely_cause")
    cause = str(cause_value.value) if cause_value is not None else "UNKNOWN"
    origin_value = decisions.get("likely_origin")
    origin = str(origin_value.value) if origin_value is not None else "UNKNOWN"
    events = list(getattr(result, "events", ()) or ())
    entities = sorted(
        {f"{event.entity_type}:{event.entity_id}" for event in events}
    )
    if not entities:
        entities = ["national"]

    target_week = None
    temporal = getattr(result, "temporal", None)
    if temporal is not None:
        target_week = temporal.target_week

    candidate_causes: list[dict[str, Any]] = []
    if cause_value is not None and cause_value.probabilities:
        for name, probability in sorted(
            cause_value.probabilities.items(), key=lambda item: item[1], reverse=True
        )[:4]:
            candidate_causes.append(
                {"cause": name, "probability": probability, "probability_kind": cause_value.probability_kind}
            )

    open_questions = list(
        CAUSE_QUESTIONS.get(cause, CAUSE_QUESTIONS["UNKNOWN"])
    )
    attribution = getattr(result, "attribution", None)
    if attribution is not None and attribution.unmatched_events:
        open_questions.append(
            "The following structural changes are not covered by the expected-event registry: "
            + ", ".join(
                f"{event.entity_type}:{event.entity_id}"
                for event in attribution.unmatched_events
            )
            + "."
        )
    if origin == "UNKNOWN":
        open_questions.append(
            "The first-divergence stage is unresolved; compare stage fingerprints before assuming an origin."
        )

    queries = [
        query.format(
            week=target_week,
            previous_week=previous_week or (target_week - 1 if target_week else "?"),
            entities=", ".join(entities),
            run_id=getattr(result, "run_id", "unknown"),
        )
        for query in CAUSE_QUERIES.get(cause, CAUSE_QUERIES["UNKNOWN"])
    ]

    evidence_ids: list[str] = []
    machine = getattr(result, "machine", None)
    if isinstance(machine, dict) and machine.get("evidence_graph"):
        evidence_ids = [
            node["evidence_id"] for node in machine["evidence_graph"]["nodes"]
        ][:50]

    incidents_payload = [
        {
            "incident_id": record.incident_id,
            "root_cause": record.root_cause,
            "likely_origin": record.likely_origin,
            "resolution": record.resolution,
            "analyst_summary": record.analyst_summary,
            "similarity": score,
        }
        for record, score in incidents
    ]

    tool_calls = [
        {"tool": tool, "args": args}
        for tool, args in CAUSE_TOOL_PLANS.get(cause, CAUSE_TOOL_PLANS["UNKNOWN"])
    ]

    return InvestigationBrief(
        run_id=str(getattr(result, "run_id", "")),
        status=str(getattr(result, "status", "")),
        question=(
            f"Determine the root cause of the {cause.lower().replace('_', ' ')} "
            f"indicated for run {getattr(result, 'run_id', '')} and recommend the "
            "first actions."
        ),
        candidate_causes=candidate_causes,
        findings=_findings(result),
        open_questions=open_questions,
        recommended_queries=queries,
        tool_calls=tool_calls,
        similar_incidents=incidents_payload,
        evidence_ids=evidence_ids,
        constraints={
            "read_only": True,
            "cite_evidence_ids": True,
            "do_not_modify_pipelines": True,
            "return_json_only": True,
        },
    )


class InvestigationAgent(Protocol):
    name: str

    def investigate(self, brief: InvestigationBrief) -> InvestigationResult: ...


def parse_investigation_result(
    payload: dict[str, Any], run_id: str, agent: str
) -> InvestigationResult:
    if not isinstance(payload, dict):
        raise ValueError("investigation response must be a JSON object")
    missing = [
        key
        for key in ("root_cause", "confidence", "summary")
        if key not in payload
    ]
    if missing:
        raise ValueError(f"investigation response missing keys: {missing}")
    confidence = float(payload["confidence"])
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be in [0, 1]")
    evidence_ids = [str(value) for value in payload.get("evidence_ids", [])]
    actions = [str(value) for value in payload.get("recommended_actions", [])]
    questions = [str(value) for value in payload.get("follow_up_questions", [])]
    return InvestigationResult(
        run_id=run_id,
        agent=agent,
        root_cause=str(payload["root_cause"]),
        confidence=confidence,
        evidence_ids=evidence_ids,
        recommended_actions=actions,
        summary=str(payload["summary"]),
        follow_up_questions=questions,
        raw=payload,
    )


@dataclass
class NullAgent:
    name: str = "null"

    def investigate(self, brief: InvestigationBrief) -> InvestigationResult:
        return InvestigationResult(
            run_id=brief.run_id,
            agent=self.name,
            root_cause="UNKNOWN",
            confidence=0.0,
            evidence_ids=[],
            recommended_actions=[],
            summary="No investigation agent configured; brief generated only.",
            follow_up_questions=list(brief.open_questions),
            raw={},
        )


@dataclass
class CommandAgent:
    command: str
    timeout: float = 300.0
    name: str = "command"

    def investigate(self, brief: InvestigationBrief) -> InvestigationResult:
        completed = subprocess.run(
            self.command,
            shell=True,
            input=json.dumps(brief.to_dict()),
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"agent command failed ({completed.returncode}): {completed.stderr.strip()}"
            )
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"agent command did not return JSON: {error}"
            ) from error
        return parse_investigation_result(payload, brief.run_id, self.name)
