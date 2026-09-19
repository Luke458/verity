from __future__ import annotations

import json

import pytest

from qc.cli import main
from qc.decisions import FeatureEncoder
from qc.relationships import EntityRelationship
from qc.store import SqliteStore, import_outcomes


@pytest.fixture()
def store(tmp_path):
    with SqliteStore(tmp_path / "qc.db") as database:
        yield database


def test_runs_are_immutable(store):
    store.record_run(
        "run-1",
        "demo",
        "PASS",
        {"status": "PASS"},
        features=[1.0, 0.0],
        feature_version=1,
    )
    with pytest.raises(ValueError):
        store.record_run("run-1", "demo", "PASS", {"status": "PASS"})
    rows = store.list_runs()
    assert rows[0]["run_id"] == "run-1"
    assert rows[0]["confirmed"] in (0, None)


def test_confirmed_only_retrieval(store):
    store.record_run(
        "run-1",
        "demo",
        "INVESTIGATE",
        {"status": "INVESTIGATE"},
        features=[1.0, 0.0],
        feature_version=1,
    )
    store.record_run(
        "run-2",
        "demo",
        "INVESTIGATE",
        {"status": "INVESTIGATE"},
        features=[0.9, 0.1],
        feature_version=1,
    )
    with pytest.raises(ValueError):
        store.record_outcome("missing-run", root_cause="X")

    store.record_outcome(
        "run-1", root_cause="MISSING_STORES", confirmed=True, resolution="fixed"
    )
    store.record_outcome("run-2", root_cause="CODING", confirmed=False)

    confirmed = store.confirmed_runs()
    assert [row["run_id"] for row in confirmed] == ["run-1"]
    hits = store.retrieve([1.0, 0.0], k=5)
    assert [hit["run_id"] for hit in hits] == ["run-1"]
    assert hits[0]["root_cause"] == "MISSING_STORES"
    assert hits[0]["resolution"] == "fixed"


def test_latest_outcome_wins(store):
    store.record_run("run-1", "demo", "INVESTIGATE", {}, features=[1.0], feature_version=1)
    store.record_outcome("run-1", root_cause="UNKNOWN", confirmed=False)
    store.record_outcome(
        "run-1", root_cause="CODING", confirmed=True, resolution="mapping fix"
    )
    latest = store.latest_outcome("run-1")
    assert latest is not None
    assert latest["root_cause"] == "CODING"
    assert bool(latest["confirmed"]) is True
    assert [hit["run_id"] for hit in store.retrieve([1.0])] == ["run-1"]


def test_registry_revisions(store):
    first = store.save_registry(
        [{"event_id": "e1", "entity_type": "store", "entity_ids": ["S1"]}]
    )
    second = store.save_registry(
        [
            {
                "event_id": "e1",
                "entity_type": "store",
                "entity_ids": ["S1", "S2"],
            }
        ]
    )
    assert (first, second) == (1, 2)
    latest = store.load_registry()
    assert len(latest) == 1
    assert latest[0]["entity_ids"] == ["S1", "S2"]
    assert store.registry_revisions() == [1, 2]


def test_relationships_are_deduplicated_and_filterable(store):
    relationship = EntityRelationship(
        source_id="A",
        target_id="C",
        entity_type="store",
        relationship="replaced_by",
        confirmed=False,
        confidence=0.95,
    )
    assert store.add_relationship(relationship) is True
    assert store.add_relationship(relationship) is False
    assert store.relationships() == [relationship]
    assert store.relationships(confirmed_only=True) == []
    assert store.for_entity("store", ["A"]) == [relationship]
    assert store.for_entity("store", ["Z"]) == []


def test_record_result_end_to_end(tmp_path):
    from qc.run import run_qc
    from qcgen.config import suite_config
    from qcgen.scenarios import build_scenario
    from qcgen.sources import ScenarioSource

    built = build_scenario(
        suite_config("tiny"),
        0,
        tmp_path,
        "missing_stores",
        ("source", "coded", "warehouse", "report"),
    )
    result = run_qc(ScenarioSource(built.directory), "V0002", "V0001")

    with SqliteStore(tmp_path / "runs.db") as database:
        database.record_result(result)
        database.record_outcome(
            result.run_id, root_cause="MISSING_STORES", confirmed=True
        )
        hits = database.retrieve(FeatureEncoder().encode(result).tolist())

    assert hits
    assert hits[0]["run_id"] == result.run_id
    assert hits[0]["root_cause"] == "MISSING_STORES"


def test_import_outcomes_reports_bad_rows(tmp_path):
    with SqliteStore(tmp_path / "store.db") as store:
        store.record_run(
            "run-1", "demo", "INVESTIGATE", {"status": "INVESTIGATE"},
            features=[1.0], feature_version=1,
        )
        report = import_outcomes(
            store,
            [
                {
                    "run_id": "run-1",
                    "root_cause": "MISSING_STORES",
                    "confirmed": "true",
                    "origin": "SOURCE",
                    "severity": "HIGH",
                    "requires_investigation": "yes",
                    "resolution": "fixed",
                    "symptom_tags": "missing,source",
                },
                {"run_id": "missing-run", "root_cause": "X"},
                {"run_id": "run-1"},
            ],
        )
        assert report["imported"] == 1
        assert len(report["errors"]) == 2
        latest = store.latest_outcome("run-1")
        assert latest is not None
        assert latest["root_cause"] == "MISSING_STORES"
        assert bool(latest["confirmed"]) is True
        assert bool(latest["requires_investigation"]) is True
        assert latest["likely_origin"] == "SOURCE"
        assert json.loads(latest["symptom_tags"]) == ["missing", "source"]


def test_import_dry_run_does_not_write(tmp_path):
    with SqliteStore(tmp_path / "store.db") as store:
        store.record_run("run-1", "demo", "PASS", {"status": "PASS"})
        report = import_outcomes(
            store,
            [{"run_id": "run-1", "root_cause": "UNKNOWN"}],
            dry_run=True,
        )
        assert report["imported"] == 1
        assert store.latest_outcome("run-1") is None


def test_cli_store_import(tmp_path, capsys):
    database = tmp_path / "store.db"
    with SqliteStore(database) as store:
        store.record_run("run-1", "demo", "INVESTIGATE", {"status": "INVESTIGATE"})
    csv_path = tmp_path / "outcomes.csv"
    csv_path.write_text(
        "run_id,root_cause,confirmed,resolution\n"
        "run-1,MISSING_STORES,true,source re-delivery fixed it\n"
    )
    assert (
        main(["store", "import", "--store", str(database), "--csv", str(csv_path)])
        == 0
    )
    output = capsys.readouterr().out
    assert "imported=1" in output
    assert main(["store", "list", "--store", str(database), "--confirmed-only"]) == 0
    assert "MISSING_STORES" in capsys.readouterr().out
