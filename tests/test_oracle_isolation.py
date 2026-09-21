"""Oracle isolation: the engine must not be able to read ground truth.

Three independent guards:

1. Import boundary: engine modules in ``qc/`` may not import ``qcgen``.
   Evaluation harnesses are explicitly allowlisted because they score runs;
   the engine itself never sees the generator.
2. Filesystem boundary: scenario data (Parquet versions plus data-only
   ``manifest.json``) and ``suite.json`` contain no family, fault, case or
   expected-event fields; ground truth lives in a separate vault root.
3. Behavioural boundary: a blind run of a registered-event scenario must not
   report ``PASS_WITH_EXPLANATION``; the same run with an explicit registry
   does. The engine works with the vault deleted.
"""

from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path

from qc.registry import StaticRegistry
from qc.run import run_qc
from qcgen.config import suite_config
from qcgen.scenarios import build_scenario, generate_suite
from qcgen.sources import ScenarioSource

ROOT = Path(__file__).resolve().parents[1]

# Harnesses may read the oracle to score; engine modules may not import qcgen.
HARNESS_ALLOWLIST = {
    "champion.py",
    "cli.py",
    "cohort.py",
    "evidence_bench.py",
    "labels.py",
    "replay.py",
    "shadow.py",
    "synthetic_analyst.py",
    "tspulse_benchmark.py",
}

GROUND_TRUTH_KEYS = {
    "family",
    "fault",
    "cases",
    "expected_events",
    "expected_status",
    "expected_class",
    "injected_effect",
}


def _imports_qcgen(path: Path) -> bool:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.split(".")[0] == "qcgen" for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] == "qcgen":
                return True
    return False


def test_engine_modules_do_not_import_qcgen() -> None:
    violations = []
    for path in sorted((ROOT / "qc").glob("*.py")):
        if path.name in HARNESS_ALLOWLIST:
            continue
        if _imports_qcgen(path):
            violations.append(path.name)
    assert not violations, f"engine modules import the generator: {violations}"


def test_allowlisted_harnesses_exist() -> None:
    names = {path.name for path in (ROOT / "qc").glob("*.py")}
    missing = HARNESS_ALLOWLIST - names
    assert not missing, f"stale harness allowlist entries: {sorted(missing)}"


def test_scenario_data_contains_no_ground_truth(tmp_path) -> None:
    built = build_scenario(
        suite_config("tiny"),
        0,
        tmp_path,
        "expected_event",
        ("source", "coded", "warehouse", "report"),
        oracle_root=tmp_path / "_oracle",
    )
    manifest = json.loads((built.directory / "manifest.json").read_text())
    assert not (GROUND_TRUTH_KEYS & set(manifest))
    assert "expected_class" not in (built.directory / "manifest.json").read_text()
    assert built.oracle["family"] == "expected_event"


def test_suite_index_contains_no_ground_truth(tmp_path) -> None:
    suite_dir = generate_suite(
        suite_config("tiny"),
        tmp_path / "suites",
        3,
        "blind",
        ("source", "coded", "warehouse", "report"),
    )
    suite = json.loads((suite_dir / "suite.json").read_text())
    assert "families" not in suite
    for entry in suite["scenarios"]:
        assert set(entry) == {"scenario_id", "row_counts"}
    # The vault sits outside the suite root; the suite index stays data-only.
    vault_root = tmp_path / "suites" / "oracle" / "blind"
    assert vault_root.exists()
    assert [path.name for path in suite_dir.glob("*.json")] == ["suite.json"]


def test_blind_run_cannot_match_registered_event(tmp_path) -> None:
    built = build_scenario(
        suite_config("tiny"),
        0,
        tmp_path,
        "expected_event",
        ("source", "coded", "warehouse", "report"),
        oracle_root=tmp_path / "_oracle",
    )
    source = ScenarioSource(built.directory)
    blind = run_qc(source, "V0002", "V0001")
    assert blind.attribution is not None
    assert not blind.attribution.matched_event_ids
    assert blind.machine["historical_revision"]["status"] == "INVESTIGATE"

    registered = run_qc(
        source,
        "V0002",
        "V0001",
        registry=StaticRegistry(list(built.oracle["expected_events"])),
    )
    assert registered.attribution is not None
    assert registered.attribution.matched_event_ids
    assert (
        registered.machine["historical_revision"]["status"]
        == "PASS_WITH_EXPLANATION"
    )


def test_engine_runs_with_vault_deleted(tmp_path) -> None:
    built = build_scenario(
        suite_config("tiny"),
        0,
        tmp_path,
        "missing_stores",
        ("source", "coded", "warehouse", "report"),
        oracle_root=tmp_path / "_oracle",
    )
    shutil.rmtree(tmp_path / "_oracle")
    result = run_qc(ScenarioSource(built.directory), "V0002", "V0001")
    assert result.status == "INVESTIGATE"
    assert result.machine["latest_week"] is not None
