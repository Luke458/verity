"""Typed cause labels for a completed run.

``RuleDecisionProvider`` maps deterministic and temporal evidence to
``likely_cause``, ``likely_origin`` and ``severity`` with explicit evidence
references, and ``likely_causes`` to every cause the evidence supports (two
simultaneous faults can both be named). ``requires_investigation`` is never a provider opinion: it is the
policy verdict (the final status is not PASS or PASS_WITH_EXPLANATION), so a
label can neither clear nor escalate a run. Probabilities are heuristic
(``probability_kind="heuristic"``), not calibrated.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from dataclasses import field as dc_field
from typing import Any, Protocol

from .config import DatasetConfig
from .lifecycle import missing_entity_impact

CAUSE_VALUES: tuple[str, ...] = (
    "MISSING_STORES",
    "MISSING_PRODUCTS",
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
            "likely_causes",
            "set",
            tuple(value for value in CAUSE_VALUES if value != "UNKNOWN"),
            "Which causes does the evidence support (empty when none)?",
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


@dataclass(frozen=True)
class _Match:
    cause: str
    origin: str
    severity: str
    strength: float
    evidence: list[str]


class RuleDecisionProvider:
    """Evidence-rule decisions with heuristic probabilities.

    Rules are evaluated in priority order. The first match names
    ``likely_cause``, origin and severity; every match contributes to
    ``likely_causes``. Structural rules (merges, missing entities,
    reclassification, truncation, backfill) can all match; the revision rules
    are mutually exclusive, so one unexplained revision is never counted twice.
    ``requires_investigation`` is copied from the policy status computed
    before the provider runs.
    """

    name = "rule"

    def __init__(self, config: DatasetConfig | None = None):
        self.config = config or DatasetConfig()

    def decide(self, result: Any) -> DecisionSet:
        matches = self._matches(result)
        primary = matches[0] if matches else _Match("UNKNOWN", "UNKNOWN", "LOW", 0.4, [])
        cause, origin, severity, strength = primary.cause, primary.origin, primary.severity, primary.strength
        evidence = primary.evidence
        causes = list(dict.fromkeys(match.cause for match in matches if match.cause != "UNKNOWN"))
        requires = str(getattr(result, "status", "")) not in (
            "PASS",
            "PASS_WITH_EXPLANATION",
        )
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
        values["likely_causes"] = DecisionValue(
            field="likely_causes",
            kind="set",
            value=causes,
            probabilities={match.cause: match.strength for match in matches if match.cause in causes},
            strategy=self.name,
            probability_kind="heuristic",
            evidence=[item for match in matches for item in match.evidence],
        )
        return DecisionSet(
            run_id=getattr(result, "run_id", ""),
            provider=self.name,
            values=values,
            requires_investigation=requires,
        )

    def _matches(self, result: Any) -> list[_Match]:
        config = self.config
        contracts = getattr(result, "contracts", None)
        if contracts is not None and contracts.status == "DATA_CONTRACT_FAILURE":
            failed = contracts.failed
            stage = failed[0].name.split(":", 1)[0] if failed else "report"
            # Nothing downstream of a contract failure is trustworthy.
            return [
                _Match("SCHEMA_FAILURE", _stage_origin(stage), "HIGH", 0.75,
                       [f"contract:{check.name}" for check in failed])
            ]

        events = list(getattr(result, "events", ()) or ())
        by_class: dict[str, list[Any]] = {}
        for event in events:
            by_class.setdefault(event.classification, []).append(event)
        attribution = getattr(result, "attribution", None)
        lineage = getattr(result, "lineage", None)
        temporal = getattr(result, "temporal", None)
        first = lineage.first_divergence if lineage is not None else None
        matches: list[_Match] = []

        def event_evidence(items: list[Any]) -> list[str]:
            return [f"event:{e.entity_type}:{e.entity_id}" for e in items]

        merge_candidates = [
            relationship
            for relationship in (getattr(result, "relationships", ()) or ())
            if relationship.relationship
            in ("replaced_by", "merged_into", "superseded_by")
        ]
        if merge_candidates:
            # A merge candidate always needs confirmation.
            matches.append(_Match("ENTITY_MERGE", "SOURCE", "MEDIUM", 0.6, [
                f"relationship:{relationship.source_id}->{relationship.target_id}"
                for relationship in merge_candidates
            ]))
        # A merge retires one id and backfills another, and a reclassification
        # can empty a category ("C01->C07" removes C01); those events are the
        # merge or reclassification itself, not separate truncations,
        # removals or backfills.
        explained_ids = {
            str(entity_id)
            for relationship in merge_candidates
            for entity_id in (relationship.source_id, relationship.target_id)
        } | {
            part
            for event in by_class.get("POSSIBLE_RECLASSIFICATION", [])
            for part in str(event.entity_id).split("->")
        }
        by_class = {
            name: [
                e for e in items
                if name == "POSSIBLE_RECLASSIFICATION" or str(e.entity_id) not in explained_ids
            ]
            for name, items in by_class.items()
        }

        # Only entity types whose absence is material name the cause; an
        # immaterial absence stays in the evidence without driving the label.
        material_types = {
            entity_type
            for entity_type, impact in missing_entity_impact(events, config).items()
            if impact["material"]
        }
        latest_missing = [
            e
            for e in by_class.get("LATEST_WEEK_MISSING", [])
            if e.entity_type in material_types
        ]
        stores_missing = [e for e in latest_missing if e.entity_type == "store"]
        products_missing = [e for e in latest_missing if e.entity_type == "product"]
        if stores_missing:
            matches.append(_Match("MISSING_STORES", "SOURCE", "HIGH", 0.8, event_evidence(stores_missing)))
        # Products sold only in the missing stores vanish with them, so a
        # product absence is named only when no store absence explains it.
        if products_missing and not stores_missing:
            matches.append(_Match("MISSING_PRODUCTS", "SOURCE", "HIGH", 0.8, event_evidence(products_missing)))

        reclass = by_class.get("POSSIBLE_RECLASSIFICATION", [])
        if reclass:
            matches.append(_Match("RECLASSIFICATION", "SOURCE", "MEDIUM", 0.6, event_evidence(reclass)))

        truncated = by_class.get("ENTITY_HISTORY_TRUNCATED", []) + by_class.get(
            "ENTITY_REMOVED", []
        )
        if truncated:
            matches.append(_Match("HISTORICAL_CORRECTION", "SOURCE", "MEDIUM", 0.6, event_evidence(truncated)))

        backfill = by_class.get("NEW_ENTITY_HISTORICAL_BACKFILL", [])
        if backfill:
            unmatched = bool(attribution and attribution.unmatched_events)
            matches.append(_Match("BACKFILL", "SOURCE", "MEDIUM", 0.6 if unmatched else 0.65, event_evidence(backfill)))

        rewrites_history = any(
            match.cause in ("ENTITY_MERGE", "RECLASSIFICATION", "HISTORICAL_CORRECTION", "BACKFILL")
            for match in matches
        )
        revision = self._revision_match(
            result, attribution, first, structural=bool(matches), rewrites_history=rewrites_history
        )
        if revision is not None:
            matches.append(revision)

        if not matches and temporal is not None and temporal.anomaly:
            # A temporal anomaly alone could be the market or an upstream loss
            # the engine cannot localize; it is never given a cause.
            matches.append(_Match("UNKNOWN", "UNKNOWN", "LOW", 0.4, [
                f"temporal:{item.series_id}" for item in temporal.series if item.anomaly
            ]))
        return matches

    def _revision_match(
        self,
        result: Any,
        attribution: Any,
        first: str | None,
        structural: bool,
        rewrites_history: bool,
    ) -> _Match | None:
        """At most one explanation for the unexplained historical revision.

        When a structural event already rewrites history (merge,
        reclassification, truncation, backfill), it is taken to explain any
        source-stage revision; only a downstream stage localized by lineage
        is named alongside it.
        """
        config = self.config
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
            severity = self._severity(unexplained_relative)
            if first == "coded":
                return _Match("CODING", "CODING", severity, 0.7, ["lineage:coded"])
            if first == "warehouse":
                return _Match("WAREHOUSE", "WAREHOUSE", severity, 0.7, ["lineage:warehouse"])
            if rewrites_history:
                return None
            if first == "source":
                # Attribution measures overlap (historical) weeks only, so an
                # unexplained material change that starts at the source is the
                # source restating history.
                return _Match("HISTORICAL_CORRECTION", "SOURCE", severity, 0.6, ["lineage:source"])
            return _Match("UNKNOWN", "UNKNOWN", severity, 0.5, ["attribution"])

        if rewrites_history:
            return None
        if attribution is not None and not attribution.material and attribution.breadth > config.broad_recalculation_breadth:
            return _Match("HISTORICAL_CORRECTION", "SOURCE", "LOW", 0.55, ["attribution:breadth"])

        restated = [
            item for item in (getattr(result, "week_revisions", ()) or ()) if item.material
        ]
        if restated:
            origin = _stage_origin(first) if first else "SOURCE"
            return _Match("HISTORICAL_CORRECTION", origin, "MEDIUM", 0.6,
                          [f"revision_week:{item.week}" for item in restated])

        if not structural and first in ("coded", "warehouse"):
            # Lineage localized a revision too small to be material. It names
            # the stage as a label only; the status is unaffected.
            cause = "CODING" if first == "coded" else "WAREHOUSE"
            return _Match(cause, _stage_origin(first), "LOW", 0.5, [f"lineage:{first}"])
        return None

    @staticmethod
    def _severity(relative: float) -> str:
        magnitude = abs(relative)
        if magnitude >= 0.10:
            return "HIGH"
        if magnitude >= 0.02:
            return "MEDIUM"
        return "LOW"
