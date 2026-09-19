"""Canonical evidence graph (architecture section 31).

Every deterministic and temporal observation becomes a node with a stable id,
a source layer and optional confidence, so later layers (and the semantic
runtime) consume one auditable structure instead of scattered tables.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any

from .attribution import AttributionResult
from .contracts import ContractResult
from .counterfactual import CounterfactualResult
from .lifecycle import LifecycleEvent
from .lineage import LineageResult
from .reconciliation import ReconciliationResult
from .relationships import EntityRelationship
from .temporal import TemporalResult
from .versions import VersionPair


@dataclass
class EvidenceNode:
    evidence_id: str
    type: str
    scope: str
    value: Any = None
    source_layer: str = "deterministic"
    entity: str | None = None
    metric: str | None = None
    confidence: float | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    related: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceGraph:
    nodes: list[EvidenceNode] = field(default_factory=list)
    edges: list[dict[str, str]] = field(default_factory=list)

    def add(self, node: EvidenceNode, related: Iterable[str] = ()) -> str:
        self.nodes.append(node)
        for other in related:
            self.edges.append({"from": other, "to": node.evidence_id})
        return node.evidence_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": list(self.edges),
        }


def build_evidence_graph(
    *,
    run_id: str,
    pair: VersionPair | None = None,
    contracts: ContractResult | None = None,
    events: Iterable[LifecycleEvent] = (),
    attribution: AttributionResult | None = None,
    counterfactual: CounterfactualResult | None = None,
    reconciliation: ReconciliationResult | None = None,
    lineage: LineageResult | None = None,
    temporal: TemporalResult | None = None,
    relationships: Iterable[EntityRelationship] = (),
) -> EvidenceGraph:
    graph = EvidenceGraph()
    root_id = f"{run_id}:version_pair"
    if pair is not None:
        graph.add(
            EvidenceNode(
                root_id,
                "version_pair",
                "dataset",
                value=pair.shape,
                payload=pair.to_dict(),
            )
        )

    if contracts is not None:
        for check in contracts.checks:
            if check.passed:
                continue
            graph.add(
                EvidenceNode(
                    f"{run_id}:contract:{check.name}",
                    "contract_check",
                    "dataset",
                    value=check.detail,
                    payload={"name": check.name},
                ),
                related=[root_id],
            )

    event_ids: list[str] = []
    for event in events:
        node_id = f"{run_id}:event:{event.entity_type}:{event.entity_id}"
        graph.add(
            EvidenceNode(
                node_id,
                "lifecycle_event",
                "entity",
                value=event.classification,
                entity=event.entity_id,
                source_layer="deterministic",
                payload=event.to_dict(),
            ),
            related=[root_id],
        )
        event_ids.append(node_id)

    if attribution is not None:
        graph.add(
            EvidenceNode(
                f"{run_id}:attribution",
                "attribution",
                "dataset",
                value=round(attribution.explained_fraction, 4),
                confidence=attribution.explained_fraction,
                payload=attribution.to_dict(),
            ),
            related=[root_id, *event_ids],
        )

    if counterfactual is not None:
        graph.add(
            EvidenceNode(
                f"{run_id}:counterfactual",
                "counterfactual",
                "dataset",
                value=round(counterfactual.reconciliation_score, 4),
                confidence=counterfactual.reconciliation_score,
                payload=counterfactual.to_dict(),
            ),
            related=[root_id],
        )

    if reconciliation is not None:
        for recon_check in reconciliation.failed:
            graph.add(
                EvidenceNode(
                    f"{run_id}:reconciliation:{recon_check.name}",
                    "reconciliation_check",
                    "dataset",
                    value=recon_check.detail,
                    payload={"name": recon_check.name},
                ),
                related=[root_id],
            )

    if lineage is not None:
        for divergence in lineage.divergences:
            if not divergence.get("diverged"):
                continue
            stage = str(divergence["stage"])
            graph.add(
                EvidenceNode(
                    f"{run_id}:lineage:{stage}",
                    "lineage_divergence",
                    "pipeline",
                    value=stage,
                    payload=divergence,
                ),
                related=[root_id],
            )

    if temporal is not None:
        for item in temporal.series:
            if not item.anomaly and item.series_id != "national":
                continue
            graph.add(
                EvidenceNode(
                    f"{run_id}:temporal:{item.series_id}",
                    "temporal_evidence",
                    "series",
                    value=item.flags or ["normal"],
                    source_layer="temporal",
                    entity=item.series_id,
                    confidence=item.calibrated_percentile,
                    payload=item.to_dict(),
                ),
                related=[root_id],
            )

    for relationship in relationships:
        graph.add(
            EvidenceNode(
                f"{run_id}:relationship:{relationship.source_id}->{relationship.target_id}",
                "entity_relationship",
                "entity",
                value=relationship.relationship,
                entity=relationship.source_id,
                confidence=relationship.confidence,
                payload=relationship.to_dict(),
            ),
            related=[root_id],
        )

    return graph
