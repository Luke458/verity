from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from qc.config import DatasetConfig
from qc.hierarchy import (
    HierarchyCache,
    build_aggregate,
    build_hierarchy,
    default_levels,
    hierarchy_checks,
    national_share,
)

CONFIG = DatasetConfig(
    hierarchy_intersections=(("commodity_id", "banner_id"), ("commodity_id", "state_id")),
)


def _frame() -> pd.DataFrame:
    rows = []
    for week in (1, 2):
        for store, banner, state, commodity, product, value in (
            ("s1", "b1", "nsw", "c1", "p1", 10.0),
            ("s2", "b1", "nsw", "c1", "p2", 20.0),
            ("s3", "b2", "vic", "c2", "p3", 30.0),
        ):
            rows.append(
                {
                    "week": week,
                    "store_id": store,
                    "banner_id": banner,
                    "state_id": state,
                    "commodity_id": commodity,
                    "product_id": product,
                    "dollar": value,
                    "units": value / 10.0,
                    "stock": value,
                }
            )
    return pd.DataFrame(rows)


def test_default_levels_cover_declared_views():
    names = [level.name for level in default_levels(CONFIG)]
    assert names[0] == "national"
    for column in ("store_id", "product_id", "commodity_id", "banner_id", "state_id"):
        assert column in names
    assert "commodity_id_x_banner_id" in names
    assert "commodity_id_x_state_id" in names
    intersection = next(
        level for level in default_levels(CONFIG) if level.name == "commodity_id_x_banner_id"
    )
    assert intersection.parent == "commodity_id"
    assert intersection.additive_to_total is False


def test_build_hierarchy_reconciles_every_declared_view():
    frame = _frame()
    built = build_hierarchy(frame, CONFIG)
    assert built["national"]["dollar"].sum() == pytest.approx(120.0)
    assert built["banner_id"]["dollar"].sum() == pytest.approx(120.0)
    assert built["state_id"]["dollar"].sum() == pytest.approx(120.0)
    assert built["commodity_id"]["dollar"].sum() == pytest.approx(120.0)
    assert built["store_id"]["dollar"].sum() == pytest.approx(120.0)
    intersection = built["commodity_id_x_banner_id"]
    assert intersection["dollar"].sum() == pytest.approx(120.0)
    # Counts are distinct entities per group, not summed identifiers.
    national = built["national"].set_index("week")
    assert national.loc[1, "store_count"] == 3
    assert national.loc[1, "product_count"] == 3

    checks = hierarchy_checks(frame, CONFIG)
    assert {check.status for check in checks} == {"PASS"}
    assert any(
        check.name == "hierarchy_total:commodity_id_x_banner_id->commodity_id:dollar"
        for check in checks
    )


def test_hierarchy_checks_flag_missing_keys():
    frame = _frame()
    frame.loc[0, "banner_id"] = None
    checks = hierarchy_checks(frame, CONFIG)
    failures = [check for check in checks if check.status == "FAIL"]
    assert failures
    assert any(check.name == "hierarchy_keys:banner_id" for check in failures)


def test_hierarchy_cache_reuses_aggregates_by_frame_identity():
    cache = HierarchyCache(CONFIG)
    frame = _frame()
    first = cache.get(frame)
    assert cache.get(frame) is first
    second = cache.get(frame.copy())
    assert cache.get(frame.copy()) is not second
    assert first["national"]["dollar"].sum() == second["national"]["dollar"].sum()


def test_build_aggregate_rejects_snapshot_over_time():
    with pytest.raises(ValueError, match="snapshot"):
        build_aggregate(_frame(), ["store_id"], CONFIG)


def test_national_share_is_historical_arithmetic_support():
    frame = _frame()
    share = national_share(frame, "banner_id", "b1", CONFIG, through_week=2)
    assert share == pytest.approx(0.5)
    assert national_share(frame, "banner_id", "missing", CONFIG, through_week=2) is None
    assert national_share(frame, "commodity_id", "c1", CONFIG, through_week=1) == pytest.approx(
        30.0 / 60.0
    )


def test_intersections_require_unique_keys():
    with pytest.raises(ValueError, match="unique keys"):
        replace(CONFIG, hierarchy_intersections=(("commodity_id",),))
    with pytest.raises(ValueError, match="must be unique"):
        replace(
            CONFIG,
            hierarchy_intersections=(("commodity_id", "banner_id"), ("commodity_id", "banner_id")),
        )
