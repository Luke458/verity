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
