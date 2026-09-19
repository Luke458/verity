"""Negative controls: every check must be able to fail.

A check that cannot be made to fail is not a check. Each registered control
feeds deliberately corrupted input to a check and asserts that it reports
failure. When a new check is added, register a control here; when a control
stops failing, the check has become vacuous.

The registry is intentionally non-empty from the start so the meta-test cannot
pass by having nothing to run.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from qc.config import DatasetConfig
from qc.contracts import validate_contracts
from qc.reconciliation import run_reconciliation

NegativeControl = Callable[[], None]

NEGATIVE_CONTROLS: list[tuple[str, NegativeControl]] = []


def negative_control(name: str) -> Callable[[NegativeControl], NegativeControl]:
    def decorate(function: NegativeControl) -> NegativeControl:
        NEGATIVE_CONTROLS.append((name, function))
        return function

    return decorate


def _failed(result, name: str) -> bool:
    return any(check.name == name and not check.passed for check in result.checks)


@negative_control("contracts:required_columns")
def _required_columns_fails() -> None:
    frame = pd.DataFrame({"week": [1], "dollar": [1.0]})
    result = validate_contracts(frame, frame, DatasetConfig())
    assert result.status == "DATA_CONTRACT_FAILURE"
    assert _failed(result, "required_columns")


@negative_control("contracts:metric_dtypes")
def _metric_dtypes_fails() -> None:
    frame = pd.DataFrame({"week": [1], "dollar": ["not-a-number"], "units": [1]})
    result = validate_contracts(frame, frame, DatasetConfig())
    assert result.status == "DATA_CONTRACT_FAILURE"
    assert _failed(result, "metric_dtypes")


@negative_control("contracts:week_progression")
def _week_progression_fails() -> None:
    frame = pd.DataFrame(
        {"week": [1, 3], "dollar": [1.0, 2.0], "units": [1, 2]}
    )
    result = validate_contracts(frame, frame, DatasetConfig())
    assert result.status == "DATA_CONTRACT_FAILURE"
    assert _failed(result, "week_progression")


@negative_control("contracts:null_fraction")
def _null_fraction_fails() -> None:
    frame = pd.DataFrame(
        {"week": [1, 2], "dollar": [float("nan"), float("nan")], "units": [1, 2]}
    )
    result = validate_contracts(frame, frame, DatasetConfig())
    assert result.status == "DATA_CONTRACT_FAILURE"
    assert _failed(result, "null_fraction")


@negative_control("contracts:duplicate_fraction")
def _duplicate_fraction_fails() -> None:
    frame = pd.DataFrame(
        {
            "week": [1, 1, 2],
            "store_id": ["s1", "s1", "s2"],
            "dollar": [1.0, 1.0, 2.0],
            "units": [1, 1, 2],
        }
    )
    result = validate_contracts(frame, frame, DatasetConfig())
    assert result.status == "DATA_CONTRACT_FAILURE"
    assert _failed(result, "duplicate_fraction")


@negative_control("reconciliation:mass_balance")
def _mass_balance_fails() -> None:
    base = pd.DataFrame(
        {
            "week": [1, 2],
            "dollar": [60.0, 40.0],
            "units": [1, 1],
            "store_id": ["s1", "s2"],
        }
    )
    report = base.copy()
    report["dollar"] = [30.0, 20.0]
    result = run_reconciliation(base, report, DatasetConfig())
    assert result.status == "RECONCILIATION_FAILURE"
    assert _failed(result, "mass_balance:dollar")


def test_negative_controls_are_registered() -> None:
    assert len(NEGATIVE_CONTROLS) >= 6, (
        "the negative-control registry shrank; every check must keep a control"
    )


def test_every_negative_control_fails() -> None:
    for name, control in NEGATIVE_CONTROLS:
        try:
            control()
        except AssertionError as error:  # pragma: no cover - diagnostic path
            raise AssertionError(f"negative control {name!r} did not fail") from error
