"""Pipeline stage transforms.

Source -> coded -> warehouse -> report, mirroring the spec's stage snapshots.
All transforms are value-preserving unless a fault is injected, so a fault's
first-divergence stage is exactly its injection stage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .universe import Universe

STAGE_ORDER: tuple[str, ...] = ("source", "coded", "warehouse", "report")

State = dict[str, pd.DataFrame]


def build_source(truth: pd.DataFrame, universe: Universe, weeks: pd.DataFrame) -> State:
    """Raw extract grain: one row per store x product x week, no codes."""
    fact = truth[
        ["week", "store_id", "product_id", "units", "dollar", "scripts", "stock"]
    ].copy()
    return {
        "fact": fact,
        "calendar": weeks[["week", "week_end"]].copy(),
        **universe.dimensions(),
    }


def build_coded(state: State) -> State:
    """Coding enrichment: attach commodity and banner/state codes."""
    fact = state["fact"].merge(
        state["products"][["product_id", "commodity_id"]],
        on="product_id",
        how="left",
    )
    fact = fact.merge(
        state["stores"][["store_id", "banner_id", "state_id"]],
        on="store_id",
        how="left",
    )
    return {**state, "fact": fact}


def build_warehouse(state: State) -> State:
    """Warehouse table: coded grain plus derived measures."""
    fact = state["fact"].copy()
    fact["dollar_per_unit"] = (
        fact["dollar"] / fact["units"].replace(0, np.nan)
    ).astype(np.float32)
    columns = [
        "week",
        "store_id",
        "banner_id",
        "state_id",
        "product_id",
        "commodity_id",
        "units",
        "dollar",
        "scripts",
        "stock",
        "dollar_per_unit",
    ]
    return {**state, "fact": fact[columns]}


def build_report(state: State) -> State:
    """Report grain: week x banner x state aggregates."""
    calendar = state["calendar"]
    fact = state["fact"]
    grouped = (
        fact.groupby(["week", "banner_id", "state_id"], dropna=False, observed=True)
        .agg(
            dollar=("dollar", "sum"),
            units=("units", "sum"),
            scripts=("scripts", "sum"),
            stock=("stock", "sum"),
            store_count=("store_id", "nunique"),
            product_count=("product_id", "nunique"),
        )
        .reset_index()
    )
    grouped = grouped.merge(calendar, on="week", how="left")
    columns = [
        "week",
        "week_end",
        "banner_id",
        "state_id",
        "dollar",
        "units",
        "scripts",
        "stock",
        "store_count",
        "product_count",
    ]
    grouped = grouped[columns].sort_values(
        ["week", "banner_id", "state_id"], kind="stable"
    )
    return {**state, "fact": grouped.reset_index(drop=True)}


ADVANCE = {
    "coded": build_coded,
    "warehouse": build_warehouse,
    "report": build_report,
}


def build_all_stages(source_state: State) -> dict[str, State]:
    states: dict[str, State] = {"source": source_state}
    for stage in STAGE_ORDER[1:]:
        states[stage] = ADVANCE[stage](states[STAGE_ORDER[STAGE_ORDER.index(stage) - 1]])
    return states
