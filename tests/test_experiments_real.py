from __future__ import annotations

import pandas as pd
import pytest

from experiments.real.m5 import CONFIG, FrameSource, cases, weekly_frame


def _raw(tmp_path):
    # 16 days from a Saturday: two full Walmart weeks and a 2-day partial week.
    days = [f"d_{i}" for i in range(1, 17)]
    weeks = [11101] * 7 + [11102] * 7 + [11103] * 2
    pd.DataFrame({"d": days, "wm_yr_wk": weeks}).to_csv(tmp_path / "calendar.csv", index=False)
    sales = pd.DataFrame(
        {
            "id": ["A_CA_1", "B_CA_1"],
            "item_id": ["A", "B"],
            "dept_id": ["FOODS_1", "HOBBIES_1"],
            "cat_id": ["FOODS", "HOBBIES"],
            "store_id": ["CA_1", "CA_1"],
            "state_id": ["CA", "CA"],
            # A sells 1 a day; B sells only in the first week, then nothing.
            **{day: [1, 2 if index < 7 else 0] for index, day in enumerate(days)},
        }
    )
    sales.to_csv(tmp_path / "sales_train_evaluation.csv", index=False)
    pd.DataFrame(
        {
            "store_id": ["CA_1"] * 4,
            "item_id": ["A", "A", "B", "B"],
            "wm_yr_wk": [11101, 11102, 11101, 11102],
            "sell_price": [2.0, 2.5, 10.0, 10.0],
        }
    ).to_csv(tmp_path / "sell_prices.csv", index=False)
    return tmp_path


def test_weekly_frame_keeps_full_weeks_and_rows_that_sold(tmp_path):
    frame = weekly_frame(_raw(tmp_path))
    assert sorted(frame["week"].unique()) == [1, 2]  # the partial week is dropped
    a = frame.loc[frame["product_id"] == "A"].set_index("week")
    assert a["units"].tolist() == [7, 7]
    assert a["dollar"].tolist() == pytest.approx([14.0, 17.5])  # priced per week
    b = frame.loc[frame["product_id"] == "B"]
    assert b["week"].tolist() == [1]  # no row for a week without sales
    assert b["commodity_id"].tolist() == ["HOBBIES_1"]


def test_frame_source_serves_one_stage():
    frame = pd.DataFrame({"week": [1]})
    source = FrameSource({"W0002": frame, "W0001": frame}, pd.DataFrame())
    assert source.list_versions() == ["W0001", "W0002"]
    assert source.available_stages("W0001") == [CONFIG.contract_stage] == [CONFIG.analysis_stage]
    assert source.read_dim("W0001", "stores") is None


def test_test_split_gives_each_fault_twenty_cut_points():
    test = cases("test")
    assert sum(case["family"] == "clean" for case in test) == 100
    sizes: dict[tuple[str, str], list[int]] = {}
    for case in test:
        if case["family"] != "clean":
            sizes.setdefault((case["family"], str(case["params"])), []).append(case["week"])
    assert all(len(weeks) == 20 for weeks in sizes.values())
    assert all(178 <= case["week"] <= 277 for case in test)


def test_splits_are_disjoint_and_confirm_gives_each_fault_thirteen_cut_points():
    from experiments.real.m5 import SPLITS

    weeks = [set(split) for split in SPLITS.values()]
    assert sum(map(len, weeks)) == len(set().union(*weeks))
    confirm = cases("confirm")
    assert sum(case["family"] == "clean" for case in confirm) == 52
    counts: dict[tuple[str, str], int] = {}
    for case in confirm:
        if case["family"] != "clean":
            key = (case["family"], str(case["params"]))
            counts[key] = counts.get(key, 0) + 1
    assert set(counts.values()) == {13}


def _favorita_raw(tmp_path):
    # Tuesday 2013-01-01 (dropped), then two Wednesday-Tuesday weeks.
    dates = pd.date_range("2013-01-01", "2013-01-15").strftime("%Y-%m-%d")
    rows = [{"id": i, "date": d, "store_nbr": 1, "item_nbr": 10, "unit_sales": 0.5, "onpromotion": False}
            for i, d in enumerate(dates)]
    rows += [{"id": 100, "date": "2013-01-03", "store_nbr": 1, "item_nbr": 20, "unit_sales": 2.0, "onpromotion": False},
             {"id": 101, "date": "2013-01-04", "store_nbr": 1, "item_nbr": 20, "unit_sales": -2.0, "onpromotion": False},
             {"id": 102, "date": "2013-01-10", "store_nbr": 1, "item_nbr": 20, "unit_sales": -1.0, "onpromotion": False}]
    pd.DataFrame(rows).to_parquet(tmp_path / "train.parquet")
    pd.DataFrame({"item_nbr": [10, 20], "family": ["PRODUCE", "DAIRY"], "class": [1, 2], "perishable": [1, 0]}) \
        .to_parquet(tmp_path / "items.parquet")
    pd.DataFrame({"store_nbr": [1], "city": ["Quito"], "state": ["Pichincha"], "type": ["D"], "cluster": [13]}) \
        .to_parquet(tmp_path / "stores.parquet")
    return tmp_path


def test_favorita_weeks_run_wednesday_to_tuesday_and_keep_nonzero_net_sales(tmp_path):
    from experiments.real.favorita import weekly_frame as favorita_weekly

    frame = favorita_weekly(_favorita_raw(tmp_path))
    produce = frame.loc[frame["product_id"] == "I10"].set_index("week")["units"]
    assert produce.to_dict() == {1: 3.5, 2: 3.5}  # the lone Tuesday is dropped
    dairy = frame.loc[frame["product_id"] == "I20"]
    assert dairy[["week", "units"]].values.tolist() == [[2, -1.0]]  # week 1 nets to zero: no row
    assert set(frame["state_id"]) == {"Pichincha"} and set(frame["commodity_id"]) == {"PRODUCE", "DAIRY"}


def test_favorita_faults_keep_fractional_units():
    from qcgen.faults import INJECTORS, FaultContext

    current = pd.DataFrame({
        "week": [1, 2, 2], "store_id": ["S1"] * 3, "state_id": ["P"] * 3,
        "commodity_id": ["PRODUCE"] * 3, "product_id": ["I1", "I1", "I2"], "units": [1.25, 0.75, 3.5],
    })
    stores = pd.DataFrame({"store_id": ["S1"], "state_id": ["P"], "banner_id": ["P"]})
    context = FaultContext(scenario_id="t", family="market_movement", stage="report", current_week=2,
                           rng=__import__("numpy").random.default_rng(0))
    # The adapter's carry-through: units travel as dollars, which the injector scales without rounding.
    state, _ = INJECTORS["market_movement"](
        {"fact": current.assign(dollar=current["units"]), "stores": stores}, context,
        {"factor": 0.6, "commodity_id": "PRODUCE"},
    )
    assert state["fact"]["dollar"].tolist() == pytest.approx([1.25, 0.45, 2.1])


def test_market_movement_moves_only_a_category_that_trades_in_the_target_week():
    import numpy as np

    from qcgen.faults import INJECTORS, FaultContext

    # BABY CARE left the table before week 2; picking it would inject nothing.
    fact = pd.DataFrame({
        "week": [1, 1, 2], "store_id": ["S1"] * 3, "commodity_id": ["BABY CARE", "PRODUCE", "PRODUCE"],
        "product_id": ["I1", "I2", "I2"], "units": [1, 1, 1], "dollar": [1.0, 1.0, 1.0],
    })
    stores = pd.DataFrame({"store_id": ["S1"], "state_id": ["P"], "banner_id": ["P"]})
    for seed in range(20):
        context = FaultContext(scenario_id="t", family="market_movement", stage="report", current_week=2,
                               rng=np.random.default_rng(seed))
        _, truth = INJECTORS["market_movement"]({"fact": fact, "stores": stores}, context, {"factor": 0.5})
        assert truth.affected == {"commodities": ["PRODUCE"]}


def test_favorita_splits_are_disjoint():
    from experiments.real.favorita import SPLITS as favorita_splits
    from experiments.real.favorita import cases as favorita_cases

    weeks = [set(split) for split in favorita_splits.values()]
    assert sum(map(len, weeks)) == len(set().union(*weeks))
    assert max(favorita_splits["test"]) == 241
    assert sum(case["family"] == "clean" for case in favorita_cases("test")) == 100


def test_favorita_real_event_alarm_needs_a_real_disappearance_of_an_event_type():
    from experiments.real.favorita import real_absences, real_event_alarm

    weekly = pd.DataFrame({
        "week": [1, 1, 1, 2, 2], "store_id": ["S1", "S2", "S2", "S1", "S1"],
        "state_id": ["A", "B", "B", "A", "A"], "commodity_id": ["X", "X", "Y", "X", "X"],
        "product_id": ["I1", "I1", "I2", "I1", "I1"], "units": [1.0] * 5,
    })
    gone = real_absences(weekly, 2)
    assert gone == {"store": ["S2"], "state": ["B"], "commodity": ["Y"]}
    alarm = {"status": "INVESTIGATE", "review_checks": ["historical_revision"], "history_reason": "missing_entities",
             "missing_entities": {"store": {"material": True}, "product": {"material": False}}, "real_absences": gone}
    assert real_event_alarm(alarm)
    assert not real_event_alarm({**alarm, "real_absences": {}})  # nothing really vanished
    assert not real_event_alarm({**alarm, "missing_entities": {"product": {"material": True}}})  # churn
    assert not real_event_alarm({**alarm, "review_checks": ["historical_revision", "temporal"]})


def test_favorita_coverage_approvals_end_when_the_entity_trades_again():
    from experiments.real.favorita import COVERAGE_CHANGE_WEEK, coverage_change_registry

    weeks = range(COVERAGE_CHANGE_WEEK - 12, COVERAGE_CHANGE_WEEK + 6)
    rows = [("S1", "Pr", "KEEP", "I1", w) for w in weeks]
    # Family BACK leaves at the change and returns three weeks later; GONE never returns.
    rows += [("S1", "Pr", "BACK", "I2", w) for w in weeks if not COVERAGE_CHANGE_WEEK <= w < COVERAGE_CHANGE_WEEK + 3]
    rows += [("S1", "Pr", "GONE", "I3", w) for w in weeks if w < COVERAGE_CHANGE_WEEK]
    weekly = pd.DataFrame(rows, columns=["store_id", "state_id", "commodity_id", "product_id", "week"]).assign(
        units=100.0)
    ends = {(e["entity_type"], tuple(e["entity_ids"])): e["effective_to_week"] for e in coverage_change_registry(weekly)}
    assert ends[("commodity", ("BACK",))] == COVERAGE_CHANGE_WEEK + 2
    assert ends[("commodity", ("GONE",))] == COVERAGE_CHANGE_WEEK + 5


def test_share_rules_sees_the_engines_choice_and_both_baselines():
    import numpy as np

    from experiments.real.share_rules import _share_test
    from qc.temporal import share_shift_test

    rng = np.random.default_rng(4)
    weeks = list(range(1, 121))
    parent = [1000.0 + 50 * np.sin(week / 8) for week in weeks]
    child = [value * (0.2 + 0.01 * rng.standard_normal()) for value in parent]
    args = (weeks, child, weeks, parent, 150.0, 1000.0, 26)
    for exact in (False, True):
        chosen, candidates = _share_test(exact)(*args, target_week=121, season=52)
        assert set(candidates) == {"trailing", "seasonal"}
        assert chosen in candidates.values()
    assert _share_test(False)(*args, target_week=121, season=52)[0] == share_shift_test(*args, target_week=121, season=52)
