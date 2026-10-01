"""Harder synthetic benchmark: realism knobs, explicit fault sizes, two-fault
refreshes and the sweep harness."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from qc.sweep import SweepCase, render_table, summarize
from qcgen.config import HistoryConfig, suite_config
from qcgen.dgp import generate_history
from qcgen.scenarios import MAGNITUDE_MEANING, build_scenario
from qcgen.universe import generate_universe

STAGES = ("source", "coded", "warehouse", "report")


def _history(**knobs):
    config = suite_config("small", seed=3)
    rng = np.random.default_rng(42)
    universe = generate_universe(config.universe, rng, horizon_weeks=config.history.n_weeks)
    history = generate_history(universe, replace(config.history, **knobs), rng)
    return universe, history


def _share_swing(universe, history) -> float:
    commodity = universe.products.set_index("product_id")["commodity_id"]
    totals = history.assign(c=history["product_id"].map(commodity)).groupby(["week", "c"])["dollar"].sum().unstack()
    share = totals.div(totals.sum(axis=1), axis=0)
    return float((share.max() / share.min()).median())


def test_realism_knobs_validate_and_default_off():
    defaults = HistoryConfig()
    assert defaults.commodity_season_amplitude == 0.0
    assert defaults.intermittency == 0.0
    with pytest.raises(ValueError):
        HistoryConfig(intermittency=1.5)
    with pytest.raises(ValueError):
        HistoryConfig(market_shock_sigma=-0.1)


def test_commodity_seasonality_moves_shares_and_intermittency_drops_rows():
    universe, plain = _history()
    _, seasonal = _history(commodity_season_amplitude=0.35)
    _, sparse = _history(intermittency=0.5)
    assert _share_swing(universe, seasonal) > _share_swing(universe, plain) + 0.1
    assert len(sparse) < 0.95 * len(plain)


@pytest.mark.parametrize("family", sorted(MAGNITUDE_MEANING))
def test_magnitude_sets_the_fault_size(tmp_path, family):
    small = build_scenario(suite_config("tiny", seed=5), 0, tmp_path / "s", family, STAGES, magnitude=0.1)
    large = build_scenario(suite_config("tiny", seed=5), 0, tmp_path / "l", family, STAGES, magnitude=0.5)
    assert small.oracle["magnitude"] == 0.1
    assert large.oracle["relative_effect"] >= small.oracle["relative_effect"]


def test_magnitude_is_rejected_for_families_without_a_size(tmp_path):
    with pytest.raises(ValueError, match="no magnitude"):
        build_scenario(suite_config("tiny"), 0, tmp_path, "schema_failure", STAGES, magnitude=0.1)


def test_two_faults_each_get_an_oracle_case(tmp_path):
    built = build_scenario(
        suite_config("tiny", seed=5), 0, tmp_path, "missing_stores", STAGES,
        also=("market_movement",),
    )
    assert built.oracle["families"] == ["missing_stores", "market_movement"]
    assert [case["family"] for case in built.oracle["cases"]] == ["missing_stores", "market_movement"]
    with pytest.raises(ValueError, match="same family twice"):
        build_scenario(suite_config("tiny"), 0, tmp_path / "x", "clean", STAGES, also=("clean",))


def _case(family, magnitude, detected, also=(), predicted="MISSING_STORES", causes=("MISSING_STORES",)):
    return SweepCase(
        family=family, magnitude=magnitude, seed=1, index=0,
        status="INVESTIGATE" if detected else "PASS", detected=detected,
        relative_effect=magnitude, expected_cause=causes[0] if causes else None,
        predicted_cause=predicted, also=also, expected_causes=causes,
    )


def test_summary_builds_curves_pairs_and_false_alarms():
    cases = [
        _case("missing_stores", 0.01, False),
        _case("missing_stores", 0.1, True),
        _case("missing_stores", None, True, also=("market_movement",),
              predicted="UNKNOWN", causes=("MARKET_MOVEMENT", "MISSING_STORES")),
        _case("clean", None, False, causes=()),
        _case("clean", None, True, causes=()),
    ]
    summary = summarize(cases)
    points = summary["curves"]["missing_stores"]
    assert [point["detection"]["hits"] for point in points] == [0, 1]
    pair = summary["pairs"]["missing_stores+market_movement"]
    assert pair["detection"]["hits"] == 1 and pair["cause_any_of"]["hits"] == 0
    assert summary["clean_false_alarms"]["hits"] == 1
    assert summary["clean_false_alarms"]["n"] == 2
    table = render_table({"magnitudes": [0.01, 0.1], "summary": summary})
    assert "| missing_stores | 0/1 | 1/1 |" in table
