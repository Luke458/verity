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


class _StageSource:
    def __init__(self, frames):
        self.frames = frames

    def read_fact(self, version, stage):
        return self.frames[version, stage].copy()


def _stage_frames(restate_source: float, coded_factor: float):
    import pandas as pd

    rows = [
        {"week": week, "store_id": "S1", "product_id": product, "dollar": 100.0, "units": 1}
        for week in range(1, 11)
        for product in ("P1", "P2")
    ]
    previous = pd.DataFrame(rows)
    current = pd.DataFrame(rows + [
        {"week": 11, "store_id": "S1", "product_id": product, "dollar": 100.0, "units": 1}
        for product in ("P1", "P2")
    ])
    # Late arrival: the current source restates the last overlap week upward.
    current.loc[current["week"] == 10, "dollar"] *= 1.0 + restate_source
    coded = current.copy()
    coded.loc[coded["product_id"] == "P1", "dollar"] *= coded_factor
    return {
        ("v1", "source"): previous, ("v2", "source"): current,
        ("v1", "coded"): previous, ("v2", "coded"): coded,
    }


def test_lineage_blames_the_stage_that_added_the_change():
    # Regression: "first divergent stage" blamed the source for every
    # downstream fault once the source legitimately restated recent weeks.
    from qc.config import DatasetConfig
    from qc.lineage import analyze_lineage
    from qc.versions import build_version_pair

    config = DatasetConfig(
        entity_key_columns=("store_id", "product_id"), pipeline_order=("source", "coded")
    )
    frames = _stage_frames(restate_source=0.03, coded_factor=0.8)
    pair = build_version_pair(frames["v1", "source"], frames["v2", "source"], "v1", "v2", "source", "source", config)
    result = analyze_lineage(_StageSource(frames), "v1", "v2", ["source", "coded"], config, pair)
    assert result.first_divergence == "coded"
    increments = {item["stage"]: item["increment"] for item in result.divergences}
    assert increments["coded"] > increments["source"] > 0

    # The restatement alone still points at the source; a declared window
    # excludes it entirely.
    alone = _stage_frames(restate_source=0.03, coded_factor=1.0)
    assert analyze_lineage(_StageSource(alone), "v1", "v2", ["source", "coded"], config, pair).first_divergence == "source"
    windowed = replace(config, restatement_weeks=1)
    assert analyze_lineage(_StageSource(alone), "v1", "v2", ["source", "coded"], windowed, pair).first_divergence is None


def test_seasonal_share_baseline_detects_drops_a_trailing_mean_cannot():
    from qc.temporal import share_shift_test

    rng = np.random.default_rng(7)
    weeks = list(range(1, 104))
    parent = [1000.0] * len(weeks)
    # A leaf whose share follows its own strong annual cycle.
    share = [0.3 * (1 + 0.35 * np.sin(2 * np.pi * w / 52)) * (1 + 0.01 * rng.standard_normal()) for w in weeks]
    child = [p * s for p, s in zip(parent, share, strict=True)]
    target = 104
    expected = 0.3 * (1 + 0.35 * np.sin(2 * np.pi * target / 52))
    unchanged = share_shift_test(weeks, child, weeks, parent, expected * 1000, 1000.0, 26, target_week=target)
    dropped = share_shift_test(weeks, child, weeks, parent, 0.85 * expected * 1000, 1000.0, 26, target_week=target)
    assert unchanged.method == "seasonal" and unchanged.p > 0.01
    assert dropped.method == "seasonal" and dropped.p < 1e-4 and dropped.impact < 0
    # Without a year of history only the trailing baseline is available.
    short = share_shift_test(weeks[-30:], child[-30:], weeks[-30:], parent[-30:], expected * 1000, 1000.0, 26, target_week=target)
    assert short.method == "trailing"
