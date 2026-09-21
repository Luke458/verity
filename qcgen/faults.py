"""Fault injectors.

Each injector mutates a pipeline state at one stage and returns a
``GroundTruthCase``. Injection at an earlier stage propagates through every
later transform, so the recorded injection stage is the first-divergence stage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .oracle import GroundTruthCase
from .spec import case_expectations
from .stages import State


@dataclass
class FaultContext:
    scenario_id: str
    family: str
    stage: str
    current_week: int
    rng: np.random.Generator


def _sample_ids(rng: np.random.Generator, values, size: int) -> list[str]:
    unique = sorted({str(v) for v in values})
    if size >= len(unique):
        return unique
    idx = rng.choice(len(unique), size=size, replace=False)
    return [unique[int(i)] for i in np.sort(idx)]


def _pick(rng: np.random.Generator, values) -> str:
    unique = sorted({str(v) for v in values})
    if not unique:
        raise ValueError("cannot pick from an empty population")
    return unique[int(rng.integers(len(unique)))]


def _case(ctx: FaultContext, **kwargs) -> GroundTruthCase:
    expected = bool(kwargs.pop("expected", False))
    meta = case_expectations(ctx.family, expected=expected)
    kwargs.setdefault("kind", meta["kind"])
    kwargs.setdefault("expected_status", meta["expected_status"])
    kwargs.setdefault("expected_class", meta["expected_class"])
    kwargs.setdefault("expected_origin", meta["expected_origin"])
    return GroundTruthCase(
        case_id=f"{ctx.scenario_id}-{ctx.family}", family=ctx.family, **kwargs
    )


def _entity_scope(dim: pd.DataFrame, store_ids: list[str]) -> dict[str, list[str]]:
    rows = dim.loc[dim["store_id"].isin(store_ids)]
    return {
        "stores": list(store_ids),
        "banners": sorted(str(v) for v in rows["banner_id"].unique()),
        "states": sorted(str(v) for v in rows["state_id"].unique()),
    }


def inject_missing_stores(state: State, ctx: FaultContext, params: dict) -> tuple[State, GroundTruthCase]:
    fact = state["fact"]
    week = int(params.get("week", ctx.current_week))
    n = int(params.get("n_stores", 1))
    present = sorted(str(v) for v in fact.loc[fact["week"] == week, "store_id"].unique())
    if not present:
        raise ValueError("missing_stores: no stores present in the target week")
    stores = _sample_ids(ctx.rng, present, n)
    mask = (fact["week"] == week) & fact["store_id"].isin(stores)
    new_fact = fact.loc[~mask].copy()
    case = _case(
        ctx,
        injection_stage=ctx.stage,
        affected=_entity_scope(state["stores"], stores),
        weeks=[week],
        details={"n_stores": len(stores), "week": week},
    )
    return {**state, "fact": new_fact}, case


def inject_new_store_backfill(
    state: State, ctx: FaultContext, params: dict
) -> tuple[State, GroundTruthCase]:
    fact = state["fact"]
    dim = state["stores"]
    n = int(params.get("n_stores", 1))
    expected = bool(params.get("expected", ctx.family == "expected_event"))
    end = int(params.get("week", ctx.current_week))
    backfill_weeks = int(params.get("backfill_weeks", 12))
    start = max(1, end - backfill_weeks + 1)

    window = fact.loc[fact["week"].between(start, end)]
    sources = _sample_ids(ctx.rng, window["store_id"].unique(), n)
    existing = {str(v) for v in dim["store_id"]}
    new_ids: list[str] = []
    candidate = 900
    while len(new_ids) < n:
        store_id = f"S{candidate:03d}"
        candidate += 1
        if store_id not in existing and store_id not in new_ids:
            new_ids.append(store_id)

    pieces: list[pd.DataFrame] = []
    new_rows: list[dict] = []
    for new_id, source_id in zip(new_ids, sources):
        history = fact.loc[
            (fact["store_id"] == source_id) & fact["week"].between(start, end)
        ].copy()
        history["store_id"] = new_id
        factor = float(ctx.rng.uniform(0.6, 1.4))
        history["units"] = np.rint(
            history["units"].to_numpy(dtype=np.float64) * factor
        ).astype(np.int32)
        for column in ("dollar", "scripts", "stock"):
            history[column] = (
                history[column].to_numpy(dtype=np.float64) * factor
            ).astype(np.float32)
        pieces.append(history)
        source_row = dim.loc[dim["store_id"] == source_id].iloc[0].to_dict()
        source_row.update(
            {
                "store_id": new_id,
                "store_name": f"New Store {new_id}",
                "open_week": start,
            }
        )
        new_rows.append(source_row)

    new_fact = pd.concat([fact, *pieces], ignore_index=True)
    new_dim = pd.concat([dim, pd.DataFrame(new_rows)], ignore_index=True)
    registry = None
    if expected:
        registry = {
            "event_id": f"{ctx.scenario_id}-event",
            "schema_version": 2,
            "dataset": "synthetic-retail",
            "confirmed": True,
            "approved_by": "synthetic-fixture",
            "approved_at": "2000-01-01T00:00:00+00:00",
            "provenance": "synthetic",
            "event_type": "new_store_historical_backfill",
            "entity_type": "store",
            "entity_ids": list(new_ids),
            "effective_week": end,
            "expected_history_start": start,
            "expected_history_end": end,
            "description": "New store onboarded with historical backfill.",
        }
    case = _case(
        ctx,
        expected=expected,
        injection_stage=ctx.stage,
        affected=_entity_scope(new_dim, new_ids),
        weeks=list(range(start, end + 1)),
        details={"backfill_start": start, "backfill_end": end, "expected": expected},
        event_registry=registry,
    )
    return {**state, "fact": new_fact, "stores": new_dim}, case


def inject_history_truncation(
    state: State, ctx: FaultContext, params: dict
) -> tuple[State, GroundTruthCase]:
    fact = state["fact"]
    n = int(params.get("n_stores", 1))
    n_weeks = int(params.get("n_weeks", 4))
    stores = _sample_ids(ctx.rng, fact["store_id"].unique(), n)
    start = max(1, ctx.current_week - n_weeks)
    weeks = list(range(start, ctx.current_week))
    mask = fact["store_id"].isin(stores) & fact["week"].isin(weeks)
    new_fact = fact.loc[~mask].copy()
    case = _case(
        ctx,
        injection_stage=ctx.stage,
        affected=_entity_scope(state["stores"], stores),
        weeks=weeks,
        details={"n_weeks": n_weeks},
    )
    return {**state, "fact": new_fact}, case


def inject_commodity_remap(
    state: State, ctx: FaultContext, params: dict
) -> tuple[State, GroundTruthCase]:
    products = state["products"]
    n = int(params.get("n_products", 1))
    from_commodity = params.get("from_commodity")
    if from_commodity is None:
        from_commodity = _pick(ctx.rng, products["commodity_id"])
    from_commodity = str(from_commodity)
    candidates = sorted(
        str(v)
        for v in products.loc[products["commodity_id"] == from_commodity, "product_id"].unique()
    )
    if not candidates:
        raise ValueError(f"commodity_remap: no products in commodity {from_commodity}")
    selected = _sample_ids(ctx.rng, candidates, n)
    others = [
        str(v) for v in products["commodity_id"].unique() if str(v) != from_commodity
    ]
    to_commodity = str(params.get("to_commodity") or _pick(ctx.rng, others))

    new_products = products.copy()
    new_products.loc[new_products["product_id"].isin(selected), "commodity_id"] = to_commodity
    new_state = {**state, "products": new_products}
    if "commodity_id" in state["fact"].columns:
        new_fact = state["fact"].copy()
        new_fact.loc[new_fact["product_id"].isin(selected), "commodity_id"] = to_commodity
        new_state = {**new_state, "fact": new_fact}

    case = _case(
        ctx,
        injection_stage=ctx.stage,
        affected={"products": selected, "commodities": [from_commodity, to_commodity]},
        weeks=[],
        details={
            "from_commodity": from_commodity,
            "to_commodity": to_commodity,
            "split_by": "commodity_id",
            "split_values": [from_commodity, to_commodity],
        },
    )
    return new_state, case


def inject_coding_error(
    state: State, ctx: FaultContext, params: dict
) -> tuple[State, GroundTruthCase]:
    fact = state["fact"]
    selected = _sample_ids(ctx.rng, fact["product_id"].unique(), int(params.get("n_products", 1)))
    factor = float(params.get("factor", 0.7))
    mask = fact["product_id"].isin(selected)
    new_fact = fact.copy()
    new_fact.loc[mask, "dollar"] = (
        new_fact.loc[mask, "dollar"].to_numpy(dtype=np.float64) * factor
    ).astype(np.float32)
    case = _case(
        ctx,
        injection_stage=ctx.stage,
        affected={"products": selected},
        weeks=[],
        details={"factor": factor},
    )
    return {**state, "fact": new_fact}, case


def inject_warehouse_transform_error(
    state: State, ctx: FaultContext, params: dict
) -> tuple[State, GroundTruthCase]:
    fact = state["fact"]
    if "commodity_id" not in fact.columns:
        raise ValueError("warehouse_transform_error requires a coded stage")
    commodity = str(params.get("commodity_id") or _pick(ctx.rng, fact["commodity_id"].dropna()))
    factor = float(params.get("factor", 0.5))
    mask = fact["commodity_id"] == commodity
    new_fact = fact.copy()
    new_fact.loc[mask, "dollar"] = (
        new_fact.loc[mask, "dollar"].to_numpy(dtype=np.float64) * factor
    ).astype(np.float32)
    case = _case(
        ctx,
        injection_stage=ctx.stage,
        affected={"commodities": [commodity]},
        weeks=[],
        details={"factor": factor},
    )
    return {**state, "fact": new_fact}, case


def inject_recalculation(
    state: State, ctx: FaultContext, params: dict
) -> tuple[State, GroundTruthCase]:
    fact = state["fact"]
    sigma = float(params.get("sigma", 0.004))
    mask = fact["week"] < ctx.current_week
    count = int(mask.sum())
    new_fact = fact.copy()
    if count:
        noise = np.exp(ctx.rng.normal(0.0, sigma, size=count))
        new_fact.loc[mask, "dollar"] = (
            new_fact.loc[mask, "dollar"].to_numpy(dtype=np.float64) * noise
        ).astype(np.float32)
    weeks = sorted(int(v) for v in fact.loc[mask, "week"].unique())
    case = _case(
        ctx,
        injection_stage=ctx.stage,
        affected={},
        weeks=weeks,
        details={"sigma": sigma, "changed_rows": count},
    )
    return {**state, "fact": new_fact}, case


def inject_schema_failure(
    state: State, ctx: FaultContext, params: dict
) -> tuple[State, GroundTruthCase]:
    column = str(params.get("column", "dollar"))
    fact = state["fact"]
    if column not in fact.columns:
        raise ValueError(f"schema_failure: column {column!r} not present")
    new_fact = fact.drop(columns=[column])
    case = _case(
        ctx,
        injection_stage=ctx.stage,
        affected={},
        weeks=[],
        details={"column": column},
    )
    return {**state, "fact": new_fact}, case


def inject_null_duplicate_storm(
    state: State, ctx: FaultContext, params: dict
) -> tuple[State, GroundTruthCase]:
    fact = state["fact"]
    null_fraction = float(params.get("null_fraction", 0.05))
    duplicate_fraction = float(params.get("duplicate_fraction", 0.02))
    new_fact = fact.copy()
    n = len(new_fact)
    duplicated = 0
    if n:
        null_mask = ctx.rng.random(n) < null_fraction
        new_fact.loc[null_mask, "dollar"] = np.nan
        duplicated = int(n * duplicate_fraction)
        if duplicated > 0:
            idx = np.sort(ctx.rng.choice(n, size=duplicated, replace=False))
            new_fact = pd.concat([new_fact, new_fact.iloc[idx]], ignore_index=True)
    case = _case(
        ctx,
        injection_stage=ctx.stage,
        affected={},
        weeks=[],
        details={
            "null_fraction": null_fraction,
            "duplicate_fraction": duplicate_fraction,
            "duplicated_rows": duplicated,
        },
    )
    return {**state, "fact": new_fact}, case


def inject_market_movement(
    state: State, ctx: FaultContext, params: dict
) -> tuple[State, GroundTruthCase]:
    fact = state["fact"]
    if "commodity_id" not in fact.columns:
        raise ValueError("market_movement requires a coded stage")
    commodity = str(params.get("commodity_id") or _pick(ctx.rng, fact["commodity_id"].dropna()))
    factor = float(params.get("factor", 0.8))
    mask = (fact["commodity_id"] == commodity) & (fact["week"] == ctx.current_week)
    new_fact = fact.copy()
    new_fact.loc[mask, "units"] = np.rint(
        new_fact.loc[mask, "units"].to_numpy(dtype=np.float64) * factor
    ).astype(np.int32)
    new_fact.loc[mask, "dollar"] = (
        new_fact.loc[mask, "dollar"].to_numpy(dtype=np.float64) * factor
    ).astype(np.float32)
    case = _case(
        ctx,
        injection_stage=ctx.stage,
        affected={"commodities": [commodity]},
        weeks=[ctx.current_week],
        details={"factor": factor},
    )
    return {**state, "fact": new_fact}, case


def inject_missing_products(
    state: State, ctx: FaultContext, params: dict
) -> tuple[State, GroundTruthCase]:
    fact = state["fact"]
    week = int(params.get("week", ctx.current_week))
    n = int(params.get("n_products", 1))
    present = sorted(
        str(value) for value in fact.loc[fact["week"] == week, "product_id"].unique()
    )
    if not present:
        raise ValueError("missing_products: no products present in the target week")
    products = _sample_ids(ctx.rng, present, n)
    mask = (fact["week"] == week) & fact["product_id"].isin(products)
    new_fact = fact.loc[~mask].copy()
    case = _case(
        ctx,
        injection_stage=ctx.stage,
        affected={"products": products},
        weeks=[week],
        details={"n_products": len(products), "week": week},
    )
    return {**state, "fact": new_fact}, case


def inject_entity_merge(
    state: State, ctx: FaultContext, params: dict
) -> tuple[State, GroundTruthCase]:
    """One store disappears while a new store absorbs its history."""
    fact = state["fact"]
    dim = state["stores"]
    source_id = _pick(ctx.rng, fact["store_id"].unique())
    existing = {str(value) for value in dim["store_id"]}
    candidate = 900
    while f"S{candidate:03d}" in existing:
        candidate += 1
    target_id = f"S{candidate:03d}"

    history = fact.loc[fact["store_id"] == source_id].copy()
    factor = float(ctx.rng.uniform(0.9, 1.1))
    history["units"] = np.rint(
        history["units"].to_numpy(dtype=np.float64) * factor
    ).astype(np.int32)
    for column in ("dollar", "scripts", "stock"):
        history[column] = (
            history[column].to_numpy(dtype=np.float64) * factor
        ).astype(np.float32)
    history["store_id"] = target_id
    remaining = fact.loc[fact["store_id"] != source_id]
    new_fact = pd.concat([remaining, history], ignore_index=True)

    source_row = dim.loc[dim["store_id"] == source_id].iloc[0].to_dict()
    source_row.update(
        {
            "store_id": target_id,
            "store_name": f"Merged Store {target_id}",
            "open_week": int(source_row.get("open_week", 1)),
        }
    )
    new_dim = pd.concat([dim, pd.DataFrame([source_row])], ignore_index=True)
    new_dim = new_dim.loc[new_dim["store_id"] != source_id]

    case = _case(
        ctx,
        injection_stage=ctx.stage,
        affected=_entity_scope(new_dim, [source_id, target_id]),
        weeks=[],
        details={
            "source_store": source_id,
            "target_store": target_id,
            "factor": factor,
        },
    )
    return {**state, "fact": new_fact, "stores": new_dim}, case


INJECTORS = {
    "missing_stores": inject_missing_stores,
    "missing_products": inject_missing_products,
    "entity_merge": inject_entity_merge,
    "new_store_backfill": inject_new_store_backfill,
    "expected_event": inject_new_store_backfill,
    "history_truncation": inject_history_truncation,
    "commodity_remap": inject_commodity_remap,
    "coding_error": inject_coding_error,
    "warehouse_transform_error": inject_warehouse_transform_error,
    "recalculation": inject_recalculation,
    "schema_failure": inject_schema_failure,
    "null_duplicate_storm": inject_null_duplicate_storm,
    "market_movement": inject_market_movement,
}
