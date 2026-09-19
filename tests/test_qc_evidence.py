from __future__ import annotations

import json

import pytest

from qc import run_qc
from qc.evidence import build_evidence_graph
from qcgen.config import suite_config
from qcgen.scenarios import build_scenario
from qcgen.sources import ScenarioSource


@pytest.fixture(scope="module")
def market_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("evidence")
    built = build_scenario(
        suite_config("tiny"),
        0,
        root,
        "market_movement",
        ("source", "coded", "warehouse", "report"),
    )
    return run_qc(ScenarioSource(built.directory), "V0002", "V0001")


def test_evidence_graph_from_run(market_run):
    graph = market_run.evidence
    assert graph is not None
    types = {node.type for node in graph.nodes}
    assert {"version_pair", "attribution", "counterfactual", "temporal_evidence"} <= types

    ids = {node.evidence_id for node in graph.nodes}
    for edge in graph.edges:
        assert edge["from"] in ids
        assert edge["to"] in ids

    temporal_nodes = [node for node in graph.nodes if node.type == "temporal_evidence"]
    assert any(node.entity == "national" for node in temporal_nodes)
    assert any(node.source_layer == "temporal" for node in graph.nodes)

    json.dumps(graph.to_dict())


def test_evidence_graph_minimal():
    graph = build_evidence_graph(run_id="r")
    assert graph.to_dict() == {"nodes": [], "edges": []}
