"""Pilot readiness must be measured, not declared."""

from __future__ import annotations

import hashlib
import json

from qc.cohort import CohortPlan
from qc.pilot import pilot_readiness
from qc.run import run_qc
from qc.store import SqliteStore
from qcgen.config import suite_config
from qcgen.scenarios import build_scenario
from qcgen.sources import ScenarioSource


def _pinned_plan(tmp_path):
    plan = CohortPlan(
        families=("missing_stores",),
        controls=(),
        scenarios_per_family=1,
        dev_seeds=(1,),
        heldout_seeds=(2,),
    )
    path = tmp_path / "plan.json"
    plan.save(path)
    sha = hashlib.sha256(
        json.dumps(plan.to_dict(), sort_keys=True).encode()
    ).hexdigest()
    (tmp_path / "plan.json.sha256").write_text(sha + "\n")
    return path


def _record_outcome(store, result, provenance, index):
    store.record_result(result)
    store.record_outcome(
        result.run_id,
        root_cause="MISSING_STORES",
        confirmed=True,
        provenance=provenance,
    )


def test_synthetic_store_is_not_ready(tmp_path):
    case = build_scenario(
        suite_config("tiny"),
        0,
        tmp_path,
        "missing_stores",
        ("source", "coded", "warehouse", "report"),
    )
    result = run_qc(ScenarioSource(case.directory), "V0002", "V0001")
    store_path = tmp_path / "store.db"
    with SqliteStore(store_path) as store:
        _record_outcome(store, result, "synthetic", 0)
    report = pilot_readiness(
        store_path, _pinned_plan(tmp_path), min_analyst_labels=1
    )
    assert report.status == "NOT_READY"
    assert any("analyst outcomes" in blocker for blocker in report.blockers)
    assert any("synthetic outcomes" in blocker for blocker in report.blockers)


def test_analyst_store_with_pinned_plan_is_ready(tmp_path):
    case = build_scenario(
        suite_config("tiny"),
        0,
        tmp_path,
        "missing_stores",
        ("source", "coded", "warehouse", "report"),
    )
    result = run_qc(
        ScenarioSource(case.directory), "V0002", "V0001", run_id="analyst-1"
    )
    store_path = tmp_path / "store.db"
    with SqliteStore(store_path) as store:
        _record_outcome(store, result, "analyst", 0)
    report = pilot_readiness(
        store_path, _pinned_plan(tmp_path), min_analyst_labels=1
    )
    assert report.status == "READY"
    assert report.analyst_labels == 1
    assert report.plan_hash_verified is True
    assert report.blockers == []


def test_unpinned_plan_blocks_readiness(tmp_path):
    case = build_scenario(
        suite_config("tiny"),
        0,
        tmp_path,
        "missing_stores",
        ("source", "coded", "warehouse", "report"),
    )
    result = run_qc(
        ScenarioSource(case.directory), "V0002", "V0001", run_id="analyst-2"
    )
    store_path = tmp_path / "store.db"
    with SqliteStore(store_path) as store:
        _record_outcome(store, result, "analyst", 0)
    plan = CohortPlan(
        families=("missing_stores",),
        controls=(),
        scenarios_per_family=1,
        dev_seeds=(1,),
        heldout_seeds=(2,),
    )
    plan_path = tmp_path / "plan.json"
    plan.save(plan_path)  # no .sha256 beside it
    report = pilot_readiness(store_path, plan_path, min_analyst_labels=1)
    assert report.status == "NOT_READY"
    assert any("not pinned" in blocker for blocker in report.blockers)
