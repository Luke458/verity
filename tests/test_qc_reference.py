from __future__ import annotations

import pandas as pd
import pytest

from qc.reference import ReferenceSpec, compare_reference
from qc.registry import StaticRegistry
from qc.run import run_qc
from qcgen.config import suite_config
from qcgen.scenarios import build_scenario
from qcgen.sources import ScenarioSource

STAGES = ("source", "coded", "warehouse", "report")


def test_compare_reference_statuses():
    spec = ReferenceSpec("r1", "dataset", ("dollar",))
    current = pd.DataFrame({"dollar": [10.0, 20.0]})
    assert compare_reference(current, current.copy(), spec)["status"] == "MATCH"

    mismatched = compare_reference(
        current, pd.DataFrame({"dollar": [10.0, 10.0]}), spec
    )
    assert mismatched["status"] == "MISMATCH"
    assert mismatched["checks"][0]["relative_delta"] > 0

    incomplete = compare_reference(
        current, pd.DataFrame({"units": [1.0]}), spec
    )
    assert incomplete["status"] == "INCOMPLETE"


@pytest.fixture(scope="module")
def expected_event_case(tmp_path_factory):
    built = build_scenario(
        suite_config("tiny"),
        2,
        tmp_path_factory.mktemp("reference"),
        "expected_event",
        STAGES,
    )
    return built.directory, list(built.oracle.get("expected_events", []))


def test_reference_match_preserves_status(expected_event_case):
    scenario_dir, expected_events = expected_event_case
    source = ScenarioSource(scenario_dir)
    frame = source.read_fact("V0002", "report")
    spec = ReferenceSpec("r1", "synthetic-retail", ("dollar",))
    result = run_qc(
        source,
        "V0002",
        "V0001",
        registry=StaticRegistry(expected_events),
        reference_frame=frame,
        reference_spec=spec,
    )
    assert result.machine["reference"]["status"] == "MATCH"
    assert result.status == "PASS_WITH_EXPLANATION"


def test_reference_mismatch_forces_investigate(expected_event_case):
    scenario_dir, expected_events = expected_event_case
    source = ScenarioSource(scenario_dir)
    frame = source.read_fact("V0002", "report")
    tampered = frame.copy()
    tampered["dollar"] = tampered["dollar"] * 0.5
    spec = ReferenceSpec("r1", "synthetic-retail", ("dollar",))
    result = run_qc(
        source,
        "V0002",
        "V0001",
        registry=StaticRegistry(expected_events),
        reference_frame=tampered,
        reference_spec=spec,
    )
    assert result.machine["reference"]["status"] == "MISMATCH"
    assert result.status == "INVESTIGATE"
    assert any(
        reason.startswith("reference_mismatch:") for reason in result.reasons
    )


def test_reference_partial_arguments_rejected(expected_event_case):
    scenario_dir, _ = expected_event_case
    source = ScenarioSource(scenario_dir)
    frame = source.read_fact("V0002", "report")
    with pytest.raises(ValueError):
        run_qc(source, "V0002", "V0001", reference_frame=frame)
