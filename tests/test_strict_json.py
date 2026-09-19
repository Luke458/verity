"""Machine output must be strict JSON: no NaN/Infinity tokens."""

from __future__ import annotations

import json

import numpy as np

from qc.cli import main
from qc.jsonutil import dumps, sanitize_json
from qcgen.config import suite_config
from qcgen.scenarios import build_scenario


def _reject(token: str):
    raise AssertionError(f"non-standard JSON token: {token}")


def test_dumps_sanitizes_non_finite_values():
    payload = {
        "nan": float("nan"),
        "inf": float("inf"),
        "neg_inf": float("-inf"),
        "nested": [1.0, float("inf")],
        "np_nan": np.float32("nan"),
    }
    text = dumps(payload)
    assert "NaN" not in text
    assert "Infinity" not in text
    parsed = json.loads(text, parse_constant=_reject)
    assert parsed["nan"] is None
    assert parsed["nested"][1] is None
    assert parsed["np_nan"] is None


def test_sanitize_json_is_recursive():
    assert sanitize_json({"a": [float("inf"), {"b": float("nan")}]}) == {
        "a": [None, {"b": None}]
    }


def test_cli_run_json_is_strict(tmp_path, capsys):
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
                "run",
                "--scenario-dir",
                str(built.directory),
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out, parse_constant=_reject)
    assert payload["status"] == "INVESTIGATE"
