from __future__ import annotations

import json
from dataclasses import replace

from qcgen.config import suite_config
from qcgen.scenarios import build_scenario, generate_suite, schedule
from qcgen.verify import verify_suite


def test_schedule_interleaves_controls():
    config = suite_config(
        "tiny",
        families=(
            "missing_stores",
            "new_store_backfill",
            "history_truncation",
            "coding_error",
        ),
        controls=("market_movement",),
    )
    planned = schedule(config, 10)
    assert planned[:5] == [
        "missing_stores",
        "new_store_backfill",
        "history_truncation",
        "market_movement",
        "coding_error",
    ]
    assert set(planned) == {
        "missing_stores",
        "new_store_backfill",
        "history_truncation",
        "coding_error",
        "market_movement",
    }


def test_generate_suite_and_verify(tiny_config, tmp_path):
    stages = ("source", "coded", "warehouse", "report")
    suite_dir = generate_suite(tiny_config, tmp_path, 4, "suite", stages)
    suite = json.loads((suite_dir / "suite.json").read_text())

    assert suite["n_scenarios"] == 4
    assert suite["stages"] == list(stages)
    for entry in suite["scenarios"]:
        scenario_dir = suite_dir / entry["scenario_id"]
        assert (scenario_dir / "manifest.json").exists()
        for stage in stages:
            assert (scenario_dir / "versions" / "V0002" / stage / "fact.parquet").exists()
        assert (scenario_dir / "versions" / "V0001" / "report" / "fact.parquet").exists()

    passed, report = verify_suite(suite_dir)
    assert passed, report["failed"]


def test_scenario_is_deterministic(tiny_config, tmp_path):
    first = build_scenario(tiny_config, 2, tmp_path / "a", "commodity_remap", ("report",))
    second = build_scenario(tiny_config, 2, tmp_path / "b", "commodity_remap", ("report",))
    for version in ("V0001", "V0002"):
        for stage, fingerprint in first.manifest["versions"][version]["fingerprints"].items():
            assert (
                fingerprint
                == second.manifest["versions"][version]["fingerprints"][stage]
            )


def test_different_seed_changes_data(tiny_config, tmp_path):
    other = replace(tiny_config, seed=tiny_config.seed + 1)
    first = build_scenario(tiny_config, 1, tmp_path / "a", "missing_stores", ("report",))
    second = build_scenario(other, 1, tmp_path / "b", "missing_stores", ("report",))
    assert (
        first.manifest["versions"]["V0002"]["fingerprints"]["report"]
        != second.manifest["versions"]["V0002"]["fingerprints"]["report"]
    )


def test_previous_version_is_one_week_shorter(tiny_config, tmp_path):
    result = build_scenario(tiny_config, 0, tmp_path, "missing_stores", ("report",))
    previous = result.manifest["versions"]["V0001"]["row_counts"]["report"]
    current = result.manifest["versions"]["V0002"]["row_counts"]["report"]
    assert previous < current
    assert result.manifest["n_previous_weeks"] == tiny_config.history.n_weeks - 1


def test_verify_detects_tampered_fingerprint(tiny_config, tmp_path):
    suite_dir = generate_suite(tiny_config, tmp_path, 2, "tampered", ("report",))
    manifest_path = suite_dir / "scenario-0000" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["versions"]["V0002"]["fingerprints"]["report"] = "deadbeefdeadbeef"
    manifest_path.write_text(json.dumps(manifest))

    passed, report = verify_suite(suite_dir)
    assert not passed
    assert any("fingerprint" in failure["name"] for failure in report["failed"])
