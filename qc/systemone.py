"""System One / Jev-compatible remote decision provider.

Any server implementing ``POST /v1/systemone`` - TypeSafe's hosted Jev, or the
open ``mmastrac/djev-spark`` container serving structured reads locally - can
act as the semantic decision layer. The engine sends a bounded JSON state and
typed questions and maps typed answers back onto ``DecisionSet`` fields.

This is the concrete reuse of the djev-spark reference: our deterministic
evidence remains the source of truth, and a Jev-compatible model supplies the
semantic fields when it is configured. No provider is embedded; the endpoint
and key are caller-supplied.
"""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .decisions import (
    DecisionProvider,
    DecisionSet,
    DecisionValue,
    FieldSpec,
    default_fields,
    field_index,
    score_index,
)
from .jsonutil import dumps as json_dumps

STATE_LIMIT = 32768


class ProviderAbstention(RuntimeError):
    """Mandatory evidence cannot fit the provider budget; the provider abstains."""


def build_provider_state(result: Any, max_chars: int = STATE_LIMIT) -> str:
    """Provider input routed through the recorded evidence package.

    The package view excludes final status, finding dispositions, certificate
    verdicts, effective decisions and outcome labels; it retains integrity
    results, calendar, forecast uncertainty, the per-measure ledgers and
    provenance. Legacy payloads that carry the final status are never used for
    an operational provider request: an artifact without canonical evidence
    abstains instead.
    """
    packages = list(getattr(result, "evidence_packages", ()) or ())
    package = getattr(result, "evidence_package", None)
    if not packages and package is not None:
        packages = [package]
    if not packages:
        raise ProviderAbstention(
            "canonical evidence package is unavailable; legacy "
            "status-bearing state is not a provider input"
        )
    parts: list[str] = []
    per_package = max(256, max_chars // max(1, len(packages)))
    for index, entry in enumerate(packages):
        # Operational provider requests never silently fall back to a thinner
        # evidence level: a view that cannot carry its mandatory evidence
        # abstains instead of dropping failed checks.
        view = entry.provider_view(
            per_package, level="complete" if index == 0 else "forecast"
        )
        if view.get("abstain"):
            raise ProviderAbstention(
                str(view.get("reason", "evidence exceeds provider budget"))
            )
        parts.append(json_dumps(view, default=str, sort_keys=True))
    if len("\n".join(parts)) > max_chars:
        raise ProviderAbstention("evidence exceeds provider budget")
    return "\n".join(parts)


def build_evidence_state(result: Any, max_chars: int = STATE_LIMIT) -> str:
    """Bounded JSON state for a decision provider.

    Deterministic evidence is compressed to findings, lifecycle events and
    temporal flags; raw rows are never sent.
    """
    machine = getattr(result, "machine", None) or {}
    recorded = machine.get("recorded_evidence_state")
    if isinstance(recorded, str) and len(recorded) <= max_chars:
        return recorded
    state: dict[str, Any] = {
        "findings": [f for f in machine.get("findings", []) if f.get("outcome") != "PASS"],
        "reference": machine.get("reference"),
        "run_id": getattr(result, "run_id", ""),
        "status": getattr(result, "status", ""),
        "version_pair": machine.get("version_pair"),
        "contracts": {
            "status": (machine.get("contracts") or {}).get("status"),
            "failed": [
                check["name"]
                for check in (machine.get("contracts") or {}).get("checks", [])
                if not check.get("passed")
            ],
        },
        "historical_revision": machine.get("historical_revision"),
        "events": [
            {
                "entity": f"{event.entity_type}:{event.entity_id}",
                "classification": event.classification,
                "weeks_removed": list(event.historical_weeks_removed),
                "weeks_added": list(event.historical_weeks_added),
            }
            for event in getattr(result, "events", ()) or ()
        ],
        "lineage": {
            "first_divergence": (
                getattr(result.lineage, "first_divergence", None)
                if getattr(result, "lineage", None)
                else None
            )
        },
        "temporal": {
            "anomaly": bool(
                getattr(getattr(result, "temporal", None), "anomaly", False)
            ),
            "flags": list(
                getattr(getattr(result, "temporal", None), "flags", []) or []
            ),
            "series": [
                {
                    "series_id": item.series_id,
                    "actual": item.actual,
                    "adjusted_actual": item.adjusted_actual,
                    "forecast_median": item.forecast_median,
                    "relative_residual": item.relative_residual,
                    "calibrated_percentile": item.calibrated_percentile,
                    "anomaly": item.anomaly,
                    "flags": item.flags,
                }
                for item in (
                    getattr(getattr(result, "temporal", None), "series", []) or []
                )
                if item.anomaly or item.series_id == "national"
            ],
        },
        "reasons": list(getattr(result, "reasons", []) or []),
    }
    serialized = json_dumps(state, default=str, sort_keys=True)
    if len(serialized) <= max_chars:
        return serialized

    # Reduce structurally, never by slicing the JSON text.
    state["truncated"] = True
    temporal = state.get("temporal")
    if isinstance(temporal, dict):
        temporal.pop("series", None)
    state["reasons"] = list(state.get("reasons", []))[:10]
    serialized = json_dumps(state, default=str, sort_keys=True)
    if len(serialized) <= max_chars:
        return serialized

    events = state.get("events")
    if isinstance(events, list):
        trimmed_events = []
        for event in events[:20]:
            if isinstance(event, dict):
                event = dict(event)
                event["weeks_added"] = list(event.get("weeks_added", []))[:10]
                event["weeks_removed"] = list(event.get("weeks_removed", []))[:10]
            trimmed_events.append(event)
        state["events"] = trimmed_events
    serialized = json_dumps(state, default=str, sort_keys=True)
    if len(serialized) <= max_chars:
        return serialized

    minimal = {
        "historical_revision": state.get("historical_revision"),
        "run_id": state.get("run_id"),
        "status": state.get("status"),
        "truncated": True,
        "events": list(state.get("events", []))[:5],
        "reasons": list(state.get("reasons", []))[:5],
    }
    serialized = json_dumps(minimal, default=str, sort_keys=True)
    if len(serialized) <= max_chars:
        return serialized

    minimal.pop("events", None)
    minimal.pop("reasons", None)
    serialized = json_dumps(minimal, default=str, sort_keys=True)
    if len(serialized) > max_chars:
        minimal.pop("historical_revision", None)
        serialized = json_dumps(minimal, default=str, sort_keys=True)
    if len(serialized) > max_chars:
        raise ValueError(
            f"evidence state cannot fit in {max_chars} characters; "
            "increase state_limit"
        )
    return serialized


def _questions(fields: Sequence[FieldSpec]) -> dict[str, dict[str, Any]]:
    questions: dict[str, dict[str, Any]] = {}
    for spec in fields:
        if spec.kind == "boolean":
            questions[spec.name] = {
                "type": "noul",
                "instructions": spec.question or f"Is {spec.name} true?",
                "criteria": {"true": spec.name, "false": f"not {spec.name}"},
            }
        elif spec.kind == "score":
            questions[spec.name] = {
                "type": "score",
                "instructions": spec.question or f"How would you score {spec.name}?",
                "criteria": list(spec.values),
            }
        else:
            questions[spec.name] = {
                "type": "choice",
                "instructions": spec.question or f"Which {spec.name}?",
                "criteria": {value: None for value in spec.values},
            }
    return questions


def _parse_noul(spec: FieldSpec, answer: dict[str, Any]) -> DecisionValue:
    probability = float(answer["noul"])
    if not 0.0 <= probability <= 1.0:
        raise ValueError("noul must be in [0, 1]")
    return DecisionValue(
        field=spec.name,
        kind="boolean",
        value=probability >= 0.5,
        probabilities={"True": probability, "False": 1.0 - probability},
        strategy="systemone",
        probability_kind="provider_reported",
    )


def _validated_probabilities(
    reported: Any, values: Sequence[str], field: str
) -> dict[str, float]:
    """Parse reported probabilities, rejecting values outside [0, 1]."""
    if reported is None:
        reported = {}
    if not isinstance(reported, dict):
        raise ValueError(f"probabilities for {field!r} must be an object")
    probabilities: dict[str, float] = {}
    for value in values:
        raw = reported.get(value, 0.0)
        try:
            probability = float(raw)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"probability for {field!r}/{value!r} is not a number"
            ) from error
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(
                f"probability for {field!r}/{value!r} must be in [0, 1]"
            )
        probabilities[value] = probability
    return probabilities


def _parse_choice(spec: FieldSpec, answer: dict[str, Any]) -> DecisionValue:
    choice = str(answer.get("choice"))
    if choice not in spec.values:
        raise ValueError(f"choice {choice!r} is not a class of {spec.name!r}")
    probabilities = _validated_probabilities(
        answer.get("probabilities"), spec.values, spec.name
    )
    total = sum(probabilities.values())
    if total > 0.0:
        probabilities = {
            level: value / total for level, value in probabilities.items()
        }
    else:
        probabilities = {
            level: (1.0 if level == choice else 0.0) for level in spec.values
        }
    return DecisionValue(
        field=spec.name,
        kind="choice",
        value=choice,
        probabilities=probabilities,
        strategy="systemone",
        probability_kind="provider_reported",
    )


def _parse_score(spec: FieldSpec, answer: dict[str, Any]) -> DecisionValue:
    """Map a System One score answer onto ordered levels.

    Accepts probabilities keyed by level index (the wire format) or by level
    name (our emitted format). Falls back to a one-hot at the nearest index
    when a provider returns only the weighted score. A plain ``choice`` answer
    for an ordinal field is also accepted, with the weighted index set to the
    class rank, so older or simpler servers remain compatible.
    """
    if "score" not in answer:
        if "choice" in answer:
            choice = str(answer["choice"])
            if choice not in spec.values:
                raise ValueError(
                    f"choice {choice!r} is not a class of {spec.name!r}"
                )
            reported = answer.get("probabilities") or {}
            probabilities = {
                level: float(
                    reported.get(level, reported.get(str(index), 0.0))
                )
                for index, level in enumerate(spec.values)
            }
            for level, probability in probabilities.items():
                if (
                    not math.isfinite(probability)
                    or not 0.0 <= probability <= 1.0
                ):
                    raise ValueError(
                        f"probability for {spec.name!r}/{level!r} must be in [0, 1]"
                    )
            total = sum(probabilities.values())
            if total > 0.0:
                probabilities = {
                    level: value / total for level, value in probabilities.items()
                }
            else:
                probabilities = {
                    level: (1.0 if level == choice else 0.0)
                    for level in spec.values
                }
            return DecisionValue(
                field=spec.name,
                kind="score",
                value=choice,
                probabilities=probabilities,
                strategy="systemone",
                probability_kind="provider_reported",
                index=float(spec.values.index(choice)),
            )
        raise ValueError(
            f"answer for score field {spec.name!r} has neither 'score' nor 'choice'"
        )

    score = float(answer["score"])
    if not -0.5 <= score <= len(spec.values) - 0.5:
        raise ValueError(
            f"score {score} is outside [0, {len(spec.values) - 1}] for {spec.name!r}"
        )
    reported = answer.get("probabilities") or {}
    if not isinstance(reported, dict):
        raise ValueError(f"probabilities for {spec.name!r} must be an object")
    probabilities = {}
    for index, level in enumerate(spec.values):
        if str(index) in reported:
            probabilities[level] = float(reported[str(index)])
        elif index in reported:
            probabilities[level] = float(reported[index])
        elif level in reported:
            probabilities[level] = float(reported[level])
        else:
            probabilities[level] = 0.0
        probability = probabilities[level]
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(
                f"probability for {spec.name!r}/{level!r} must be in [0, 1]"
            )
    total = sum(probabilities.values())
    if total <= 0.0:
        nearest = min(
            range(len(spec.values)), key=lambda i: abs(i - score)
        )
        probabilities = {
            level: (1.0 if index == nearest else 0.0)
            for index, level in enumerate(spec.values)
        }
    else:
        probabilities = {
            level: value / total for level, value in probabilities.items()
        }
    best = max(probabilities, key=lambda level: probabilities[level])
    return DecisionValue(
        field=spec.name,
        kind="score",
        value=best,
        probabilities=probabilities,
        strategy="systemone",
        probability_kind="provider_reported",
        index=score,
    )


def answers_to_decisions(
    answers: dict[str, Any],
    fields: Sequence[FieldSpec],
    run_id: str,
    provider: str = "systemone",
) -> DecisionSet:
    """Map System One answers (noul / choice / score) onto typed decisions."""
    values: dict[str, DecisionValue] = {}
    requires = False
    for spec in fields:
        answer = answers.get(spec.name)
        if not isinstance(answer, dict):
            raise ValueError(f"systemone response missing answer {spec.name!r}")
        if spec.kind == "boolean":
            decision_value = _parse_noul(spec, answer)
            if spec.name == "requires_investigation":
                requires = bool(decision_value.value)
        elif spec.kind == "score":
            decision_value = _parse_score(spec, answer)
        else:
            decision_value = _parse_choice(spec, answer)
        decision_value.strategy = provider
        values[spec.name] = decision_value
    return DecisionSet(
        run_id=run_id,
        provider=provider,
        values=values,
        requires_investigation=requires,
    )


def decision_to_systemone_answers(
    decision_set: DecisionSet,
    fields: Sequence[FieldSpec] | None = None,
) -> dict[str, dict[str, Any]]:
    """Emit our decisions in System One wire format.

    Ordinal fields become ``score`` answers with a level legend and
    index-keyed probabilities; booleans become ``noul``; everything else
    becomes ``choice``. Round-trips through ``answers_to_decisions``.
    """
    specs = field_index(fields)
    answers: dict[str, dict[str, Any]] = {}
    for name, decision_value in decision_set.values.items():
        probabilities = decision_value.probabilities or {}
        confidence = max(probabilities.values(), default=0.0)
        spec = specs.get(name)
        kind = decision_value.kind or (spec.kind if spec else "choice")
        if kind == "boolean":
            probability = float(probabilities.get("True", 0.0))
            answers[name] = {"type": "noul", "noul": probability}
        elif kind == "score" and spec is not None:
            index = decision_value.index
            if index is None:
                index = score_index(probabilities, spec.values)
            answers[name] = {
                "type": "score",
                "score": index,
                "legend": {str(i): level for i, level in enumerate(spec.values)},
                "probabilities": {
                    str(i): float(probabilities.get(level, 0.0))
                    for i, level in enumerate(spec.values)
                },
                "confidence": confidence,
            }
        else:
            answers[name] = {
                "type": "choice",
                "choice": str(decision_value.value),
                "probabilities": {
                    key: float(value) for key, value in probabilities.items()
                },
                "confidence": confidence,
            }
    return answers


@dataclass
class SystemOneDecisionProvider:
    url: str
    model: str = "jev-latest"
    api_key: str | None = None
    timeout: float = 60.0
    fields: tuple[FieldSpec, ...] = field(default_factory=default_fields)
    state_limit: int = STATE_LIMIT
    max_response_bytes: int = 4 * 1024 * 1024
    name: str = "systemone"
    artifact_identity: dict[str, Any] | None = None

    def decide(self, result: Any) -> DecisionSet:
        payload = {
            "model": self.model,
            "state": build_provider_state(result, self.state_limit),
            "questions": _questions(self.fields),
        }
        if isinstance(getattr(result, "machine", None), dict):
            from .evidence_package import evidence_digest

            result.machine["provider_payload_digest"] = evidence_digest(payload)
        response = self._post(payload)
        answers = response.get("answers")
        if not isinstance(answers, dict):
            raise ValueError("systemone response missing 'answers'")
        return answers_to_decisions(
            answers,
            self.fields,
            getattr(result, "run_id", ""),
            provider=self.name,
        )

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            self.url,
            data=json_dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                **(
                    {"Authorization": f"Bearer {self.api_key}"}
                    if self.api_key
                    else {}
                ),
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read(self.max_response_bytes + 1)
        except urllib.error.HTTPError as error:
            detail = error.read(self.max_response_bytes + 1).decode("utf-8", errors="replace")[:400]
            raise RuntimeError(
                f"systemone request failed with HTTP {error.code}: {detail}"
            ) from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"systemone request failed: {error.reason}") from error
        if len(raw) > self.max_response_bytes:
            raise ValueError(
                f"systemone response exceeded {self.max_response_bytes} bytes"
            )
        body = raw.decode("utf-8")
        try:
            return json.loads(body)
        except json.JSONDecodeError as error:
            raise ValueError(f"systemone returned non-JSON: {error}") from error

@dataclass
class FallbackDecisionProvider:
    """Use a local provider, escalate to a remote one when uncertain.

    Escalation happens when the local decision requires investigation and its
    selected cause probability is below ``escalate_below``. Remote errors
    propagate; a failed escalation is never silently treated as success.
    """

    local: DecisionProvider
    remote: DecisionProvider
    escalate_below: float = 0.85
    name: str = "fallback"

    def decide(self, result: Any) -> DecisionSet:
        local_decision = self.local.decide(result)
        cause = local_decision.get("likely_cause")
        confidence = (
            max(cause.probabilities.values())
            if cause is not None and cause.probabilities
            else 1.0
        )
        if local_decision.requires_investigation and confidence < self.escalate_below:
            remote_decision = self.remote.decide(result)
            # Keep local fields the remote provider did not answer so the
            # local evidence trail is not lost on escalation.
            values = dict(remote_decision.values)
            for name, decision in local_decision.values.items():
                values.setdefault(name, decision)
            return DecisionSet(
                run_id=remote_decision.run_id,
                provider=f"{self.local.name}->{remote_decision.provider}",
                values=values,
                requires_investigation=remote_decision.requires_investigation,
            )
        return local_decision
