from __future__ import annotations

import pytest

from qc.cli import main
from qc.evidence_query import ALLOWED_QUERIES, query_evidence
from qc.run import run_qc
from qcgen.config import suite_config
from qcgen.scenarios import build_scenario
from qcgen.sources import ScenarioSource


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    built = build_scenario(
        suite_config("tiny"),
        0,
        tmp_path_factory.mktemp("evidence-query"),
        "missing_stores",
        ("source", "coded", "warehouse", "report"),
    )
    return run_qc(ScenarioSource(built.directory), "V0002", "V0001")


def test_unknown_query_rejected(case):
    with pytest.raises(ValueError):
        query_evidence(case, "DROP TABLE")


def test_lifecycle_query_reports_truncation(case):
    outcome = query_evidence(case, "lifecycle_changes", limit=1)
    assert outcome.total_rows >= 2
    assert len(outcome.rows) == 1
    assert outcome.truncated is True


def test_query_filters(case):
    outcome = query_evidence(
        case, "lifecycle_changes", classification="LATEST_WEEK_MISSING"
    )
    assert outcome.rows
    assert all(row["classification"] == "LATEST_WEEK_MISSING" for row in outcome.rows)


def test_historical_residuals_include_total(case):
    outcome = query_evidence(case, "historical_residuals")
    assert outcome.rows[0]["entity_type"] == "total"
    assert "unexplained_delta" in outcome.rows[0]


def test_all_allowlisted_queries_run(case):
    # Each query must return a real, shape-checked answer, not just echo its
    # own name: a stub returning an empty Outcome would pass a name check.
    expected_keys = {
        "lifecycle_changes": {"entity_type", "entity_id", "classification"},
        "historical_residuals": {"entity_type", "delta"},
        "temporal_anomalies": {"series_id", "week"},
        "first_divergence": {"stage", "row_count_delta"},
        "contract_failures": {"name", "detail"},
        "contributors": {"entity_type", "delta"},
        "relationships": {"source_id", "target_id", "relationship"},
    }
    assert set(expected_keys) == set(ALLOWED_QUERIES)
    for query in ALLOWED_QUERIES:
        outcome = query_evidence(case, query, limit=5)
        assert outcome.query == query
        assert outcome.total_rows >= 0
        if query == "lifecycle_changes":
            assert outcome.rows, "missing_stores must produce lifecycle events"
        for row in outcome.rows:
            assert expected_keys[query] <= set(row), (query, row)


def test_unknown_filters_rejected(case):
    with pytest.raises(ValueError, match="unknown filter keys"):
        query_evidence(case, "lifecycle_changes", not_a_column="x")


def test_limit_is_capped(case):
    from qc.evidence_query import MAX_LIMIT

    outcome = query_evidence(case, "lifecycle_changes", limit=10**9)
    assert len(outcome.rows) <= MAX_LIMIT


def test_cli_evidence(case, capsys, tmp_path):
    scenario_dir = tmp_path / "scenario"
    scenario_dir.mkdir()
    # Reuse the module scenario directory through the CLI by pointing at the
    # original path stored on the result is not possible; rebuild a tiny one.
    built = build_scenario(
        suite_config("tiny"),
        0,
        tmp_path,
        "missing_stores",
        ("source", "coded", "warehouse", "report"),
    )
    assert (
        main(
            [
                "evidence",
                "--scenario-dir",
                str(built.directory),
                "--query",
                "lifecycle_changes",
                "--limit",
                "2",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "query=lifecycle_changes" in output
    assert "LATEST_WEEK_MISSING" in output
