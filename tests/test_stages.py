from __future__ import annotations

import numpy as np
import pandas as pd

from qcgen.dgp import generate_history, week_end_dates
from qcgen.stages import STAGE_ORDER, build_all_stages, build_source
from qcgen.universe import generate_universe


def _tiny_pipeline(tiny_config):
    universe = generate_universe(tiny_config.universe, np.random.default_rng(11), 30)
    truth = generate_history(universe, tiny_config.history, np.random.default_rng(12))
    dates = week_end_dates(tiny_config.history.start_week, tiny_config.history.n_weeks)
    calendar = pd.DataFrame(
        {
            "week": np.arange(1, tiny_config.history.n_weeks + 1, dtype=np.int32),
            "week_end": dates,
        }
    )
    return build_all_stages(build_source(truth, universe, calendar))


def test_all_stages_computed(tiny_config):
    states = _tiny_pipeline(tiny_config)
    assert set(states) == set(STAGE_ORDER)


def test_coded_enriches_without_changing_measures(tiny_config):
    states = _tiny_pipeline(tiny_config)
    source = states["source"]["fact"]
    coded = states["coded"]["fact"]
    assert {"commodity_id", "banner_id", "state_id"} <= set(coded.columns)
    assert not coded[["commodity_id", "banner_id", "state_id"]].isna().any().any()
    assert len(coded) == len(source)
    assert np.isclose(coded["dollar"].sum(), source["dollar"].sum())


def test_warehouse_derived_measure(tiny_config):
    states = _tiny_pipeline(tiny_config)
    warehouse = states["warehouse"]["fact"]
    assert "dollar_per_unit" in warehouse.columns
    positive = warehouse.loc[warehouse["units"] > 0]
    expected = positive["dollar"] / positive["units"]
    assert np.allclose(positive["dollar_per_unit"], expected, rtol=1e-3, atol=1e-4)


def test_report_aggregates_preserve_totals(tiny_config):
    states = _tiny_pipeline(tiny_config)
    warehouse = states["warehouse"]["fact"]
    report = states["report"]["fact"]

    assert not report["week_end"].isna().any()
    expected_rows = warehouse.groupby(["week", "banner_id", "state_id"]).ngroups
    assert len(report) == expected_rows
    assert np.isclose(report["dollar"].sum(), warehouse["dollar"].sum())
    assert np.isclose(report["units"].sum(), warehouse["units"].sum())
    assert report["store_count"].max() <= tiny_config.universe.n_stores
    assert (report["store_count"] > 0).all()
