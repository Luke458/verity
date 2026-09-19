from __future__ import annotations

import json

from qcgen.cli import main


def test_cli_generate_verify_and_list(tmp_path):
    assert (
        main(
            [
                "generate",
                "--suite",
                "cli",
                "--scenarios",
                "2",
                "--profile",
                "tiny",
                "--out",
                str(tmp_path),
            ]
        )
        == 0
    )
    suite_dir = tmp_path / "cli"
    assert (suite_dir / "suite.json").exists()
    assert main(["verify", "--suite-dir", str(suite_dir)]) == 0
    assert main(["list-faults"]) == 0
    assert main(["list-faults", "--json"]) == 0


def test_cli_verify_fails_on_tampering(tmp_path):
    main(
        [
            "generate",
            "--suite",
            "cli",
            "--scenarios",
            "1",
            "--profile",
            "tiny",
            "--out",
            str(tmp_path),
        ]
    )
    suite_dir = tmp_path / "cli"
    manifest_path = suite_dir / "scenario-0000" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["versions"]["V0002"]["fingerprints"]["report"] = "tampered"
    manifest_path.write_text(json.dumps(manifest))

    assert main(["verify", "--suite-dir", str(suite_dir)]) == 1


def test_cli_generate_from_yaml_config(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "dataset: synthetic\n"
        "profile: tiny\n"
        "seed: 11\n"
        "universe:\n"
        "  n_stores: 6\n"
        "history:\n"
        "  n_weeks: 12\n"
        "faults:\n"
        "  include: [missing_stores, market_movement]\n"
        "  controls: [market_movement]\n"
        "stages: report\n"
    )
    assert (
        main(
            [
                "generate",
                "--suite",
                "yaml",
                "--scenarios",
                "2",
                "--config",
                str(config_path),
                "--out",
                str(tmp_path),
            ]
        )
        == 0
    )
    suite = json.loads((tmp_path / "yaml" / "suite.json").read_text())
    assert suite["families"] == ["market_movement", "missing_stores"]
