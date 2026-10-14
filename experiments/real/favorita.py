"""The engine on a second real retailer: Corporación Favorita (Ecuador).

    python -m experiments.real.favorita build --raw data/real/favorita --out data/real/favorita/weekly.parquet
    python -m experiments.real.favorita run --weekly data/real/favorita/weekly.parquet --split dev --jobs 2 \\
        --out reports/real/favorita-dev.json

Favorita (https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting;
mirrored at https://huggingface.co/datasets/hsnalmasri/favorita) holds daily
unit sales of 4,100 items in 54 stores across 16 Ecuadorian provinces,
2013-01-01 to 2017-08-15. It has no prices, so the engine runs on units alone
and the price check does not apply. Perishables are sold by weight, so units
are fractional, and returns make some daily values negative.

``build`` aggregates days to Wednesday-Tuesday weeks (the last day,
2017-08-15, is a Tuesday; the lone Tuesday 2013-01-01 is dropped), keeps store
x item x week rows with nonzero net sales (the source has no rows for items
that did not sell), and joins each item's family (the department) and each
store's province. Refreshes, faults and scoring are M5's (``m5.py``); the
engine runs with its defaults and ``CONFIG`` only maps columns.
Registered in docs/real-data-favorita.md before the held-out run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from experiments.real.m5 import (
    FAULTS,
    STAGE,
    FrameSource,
    _sha256,
    _summary,
    _wilson,
    score,
)
from qc.config import DatasetConfig

FIRST_DAY = pd.Timestamp("2013-01-02")  # a Wednesday
CONFIG = replace(
    DatasetConfig(),
    name="favorita-ecuador",
    entity_columns=("store_id", "product_id", "commodity_id", "state_id"),
    metric_columns=("units",),
    primary_metric="units",
    unit_value_metric="",
    required_columns=("week", "units"),
    analysis_stage=STAGE,
    contract_stage=STAGE,
    stage_preference=(STAGE,),
    pipeline_order=(STAGE,),
    report_grain=("state_id",),
    stage_keys=((STAGE, ("store_id", "product_id")),),
    hierarchy_columns=(("state_id",),),
    explanation_entity_types=("store", "state", "product", "commodity"),
    snapshot_metrics=(),
    temporal_entity_columns=("state_id", "commodity_id"),
)

# The one structural change before the test window: in the week to 2015-01-13
# (week 106) the item universe shrinks. Eight families and 605 items drop out
# of the feed for 12 to 20 weeks (all families are back by week 126, all but
# two items by week 141). The engine flags it; an operator approves what the
# engine listed as missing, as the workflow intends, so later refreshes
# compare like-for-like, and ends each approval when the entity trades again.
# Nothing later is approved: real store events in the test weeks are scored
# by ``real_event_alarm``.
COVERAGE_CHANGE_WEEK = 106


def coverage_change_registry(weekly: pd.DataFrame) -> list[dict[str, Any]]:
    """Approvals of the absences the engine flags at ``COVERAGE_CHANGE_WEEK``.

    The simulated operator approves each absence when it is flagged and ends
    the approval when the entity trades again, so a later disappearance needs
    an approval of its own. An entity that never returns stays approved to the
    last week of the table.
    """
    from qc.lifecycle import LATEST_MISSING, classify_entity_changes
    from qc.versions import build_version_pair

    week = COVERAGE_CHANGE_WEEK
    previous = weekly.loc[weekly["week"] < week]
    current = weekly.loc[weekly["week"] <= week]
    pair = build_version_pair(previous, current, "before", "after", STAGE, STAGE, CONFIG)
    missing: dict[str, list[str]] = {}
    for event in classify_entity_changes(previous, current, pair, CONFIG):
        if LATEST_MISSING in (event.classifications or (event.classification,)):
            missing.setdefault(event.entity_type, []).append(event.entity_id)
    last = int(weekly["week"].max())
    later = weekly.loc[weekly["week"] > week]
    ends: dict[tuple[str, int], list[str]] = {}
    for entity_type, ids in missing.items():
        column = f"{entity_type}_id"
        returns = later.loc[later[column].isin(ids)].groupby(column)["week"].min()
        for entity_id in ids:
            end = int(returns[entity_id]) - 1 if entity_id in returns.index else last
            ends.setdefault((entity_type, end), []).append(entity_id)
    return [
        {
            "event_id": f"favorita-2015-01-coverage-{entity_type}-to-{end}",
            "dataset": CONFIG.name,
            "confirmed": True,
            "approved_by": "operator (simulated)",
            "approved_at": "2015-01-14T00:00:00+00:00",
            "classification": LATEST_MISSING,
            "entity_type": entity_type,
            "entity_ids": sorted(ids),
            "effective_from_week": week,
            "effective_to_week": end,
        }
        for (entity_type, end), ids in sorted(ends.items())
    ]

# Week k is the refresh that appends week k; the first assessable refresh
# appends week 106 (104 completed weeks of annual baseline). ``reserve`` is
# held back, unrun, for a later registered change.
SPLITS = {"dev": range(106, 126), "reserve": range(126, 142), "test": range(142, 242)}
FAULT_STRIDE = {"dev": 4, "reserve": 4, "test": 5}


def weekly_frame(raw: Path) -> pd.DataFrame:
    """Store x item x week net unit sales; nonzero rows, full weeks only."""
    parts = []
    train = pq.ParquetFile(raw / "train.parquet")
    for group in range(train.num_row_groups):
        batch = train.read_row_group(group, columns=["date", "store_nbr", "item_nbr", "unit_sales"]).to_pandas()
        days = (pd.to_datetime(batch["date"].astype("category")).astype("datetime64[ns]") - FIRST_DAY).dt.days
        batch = batch.loc[days.to_numpy() >= 0].assign(week=(days[days >= 0] // 7 + 1).astype(np.int32))
        parts.append(batch.groupby(["week", "store_nbr", "item_nbr"], as_index=False)["unit_sales"].sum())
    frame = pd.concat(parts).groupby(["week", "store_nbr", "item_nbr"], as_index=False)["unit_sales"].sum()
    frame = frame.loc[frame["unit_sales"] != 0.0]
    items = pd.read_parquet(raw / "items.parquet", columns=["item_nbr", "family"])
    stores = pd.read_parquet(raw / "stores.parquet", columns=["store_nbr", "state"])
    frame = frame.merge(items, on="item_nbr", how="left", validate="many_to_one")
    frame = frame.merge(stores, on="store_nbr", how="left", validate="many_to_one")
    if frame[["family", "state"]].isna().any().any():
        raise ValueError("sold rows without an item family or store province")
    out = pd.DataFrame({
        "week": frame["week"].to_numpy(dtype=np.int32),
        "store_id": "S" + frame["store_nbr"].astype(str),
        "state_id": frame["state"].astype(str),
        "commodity_id": frame["family"].astype(str),
        "product_id": "I" + frame["item_nbr"].astype(str),
        "units": frame["unit_sales"].to_numpy(dtype=np.float64),
    })
    return out.sort_values(["week", "store_id", "product_id"], ignore_index=True)


def build(raw: Path, out: Path) -> dict[str, Any]:
    frame = weekly_frame(raw)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out, index=False)
    provenance = {
        "source": "https://huggingface.co/datasets/hsnalmasri/favorita",
        "inputs": {name: _sha256(raw / name) for name in ("train.parquet", "items.parquet", "stores.parquet")},
        "rows": len(frame),
        "weeks": int(frame["week"].max()),
        "first_day": str(FIRST_DAY.date()),
        "stores": int(frame["store_id"].nunique()),
        "products": int(frame["product_id"].nunique()),
        "families": int(frame["commodity_id"].nunique()),
        "provinces": int(frame["state_id"].nunique()),
    }
    out.with_suffix(".json").write_text(json.dumps(provenance, indent=2))
    return provenance


_WEEKLY: pd.DataFrame | None = None
_REGISTRY: list[dict[str, Any]] | None = None


def _weekly(path: str) -> pd.DataFrame:
    global _WEEKLY
    if _WEEKLY is None:
        _WEEKLY = pd.read_parquet(path)
    return _WEEKLY


def _registry(weekly: pd.DataFrame) -> list[dict[str, Any]]:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = coverage_change_registry(weekly)
    return _REGISTRY


def cases(split: str) -> list[dict[str, Any]]:
    weeks = list(SPLITS[split])
    out: list[dict[str, Any]] = [
        {"family": "clean", "params": {}, "week": week, "seed": week} for week in weeks
    ]
    stride = FAULT_STRIDE[split]
    for index, (family, params) in enumerate(FAULTS):
        for week in weeks[index % stride::stride]:
            out.append({"family": family, "params": dict(params), "week": week, "seed": week * 100 + index})
    return out


# Entity types whose real disappearance from the feed is an event a reviewer
# must see (a store or province going dark, a family leaving the feed), never
# churn. Product absences are churn.
EVENT_TYPES = {"store": "store_id", "state": "state_id", "commodity": "commodity_id"}


def real_absences(weekly: pd.DataFrame, week: int) -> dict[str, list[str]]:
    """Stores, provinces and families that sold in ``week - 1`` and not in ``week``."""
    before = weekly.loc[weekly["week"] == week - 1]
    now = weekly.loc[weekly["week"] == week]
    out = {}
    for kind, column in EVENT_TYPES.items():
        gone = sorted(set(before[column]) - set(now[column]))
        if gone:
            out[kind] = gone
    return out


def real_event_alarm(record: dict[str, Any]) -> bool:
    """A clean-refresh alarm that flags a real disappearance (registered rule).

    The alarm's only finding is a material absence, every materially absent
    entity type is a store, province or family type, and entities of each such
    type really did vanish from the feed that week.
    """
    if record["status"] == "PASS" or record["review_checks"] != ["historical_revision"]:
        return False
    if record.get("history_reason") != "missing_entities":
        return False
    material = [kind for kind, entry in record["missing_entities"].items() if entry.get("material")]
    real = record.get("real_absences") or {}
    return bool(material) and all(kind in EVENT_TYPES and kind in real for kind in material)


def score_favorita(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary = score(records)
    clean = [record for record in records if record["family"] == "clean"]
    events = sum(real_event_alarm(record) for record in clean)
    alarms = sum(record["status"] != "PASS" and not real_event_alarm(record) for record in clean)
    summary["clean"].update({
        "real_event_alarms": events,
        "false_alarms": alarms,
        "ci": _wilson(alarms, len(clean)),
        "weeks_with_real_absences": sum(bool(record.get("real_absences")) for record in clean),
    })
    return summary


def run_case(weekly_path: str, case: dict[str, Any]) -> dict[str, Any]:
    from qc.run import run_qc
    from qcgen.faults import INJECTORS, FaultContext

    weekly = _weekly(weekly_path)
    week = int(case["week"])
    previous = weekly.loc[weekly["week"] < week]
    current = weekly.loc[weekly["week"] <= week]
    truth: dict[str, Any] = {}
    if case["family"] != "clean":
        stores = current[["store_id", "state_id"]].drop_duplicates()
        stores = stores.assign(banner_id=stores["state_id"])
        context = FaultContext(
            scenario_id=f"favorita-{week}", family=case["family"], stage=STAGE,
            current_week=week, rng=np.random.default_rng(int(case["seed"])),
        )
        # The injectors round units to integers and recalculate dollars only;
        # Favorita's units are fractional and its only measure. Inject on a
        # copy of units carried as dollars, then take it back as units.
        fact = current.assign(dollar=current["units"])
        state, ground = INJECTORS[case["family"]]({"fact": fact, "stores": stores}, context, case["params"])
        current = state["fact"].assign(units=state["fact"]["dollar"].astype(np.float64)).drop(columns=["dollar"])
        truth = dict(ground.affected)
    products = weekly[["product_id", "commodity_id"]].drop_duplicates("product_id")
    source = FrameSource({f"W{week - 1:04d}": previous, f"W{week:04d}": current}, products)
    result = run_qc(
        source, f"W{week:04d}", f"W{week - 1:04d}", CONFIG, registry=_registry(weekly),
        observed_at="2030-01-01T00:00:00+00:00",
    )
    return {**_summary(result, case, truth), "real_absences": real_absences(weekly, week)}


def _case_name(case: dict[str, Any]) -> str:
    return f"{case['week']:04d}-{case['family']}-{case['seed']}.json"


def run(weekly_path: str, split: str, jobs: int, out: Path) -> dict[str, Any]:
    """Run every case of ``split``; each record is kept in ``<out>.work`` so a
    crashed worker (this table is large) costs one case, and a rerun resumes."""
    from qc.cohort import code_sha256

    engine = code_sha256()[:12]
    # Cached records are valid only for this engine, adapter and fault injectors.
    import qcgen.faults

    adapter = hashlib.sha256(b"".join(
        path.read_bytes() for path in (
            Path(__file__), Path(__file__).with_name("m5.py"), Path(qcgen.faults.__file__))
    )).hexdigest()[:12]
    work = out.with_suffix(".work") / f"{engine}-{adapter}"
    work.mkdir(parents=True, exist_ok=True)
    todo = [case for case in cases(split) if not (work / _case_name(case)).exists()]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=jobs, mp_context=context) as pool:
        futures = {pool.submit(run_case, weekly_path, case): case for case in todo}
        for future in as_completed(futures):
            (work / _case_name(futures[future])).write_text(json.dumps(future.result(), default=str))
    records = [json.loads((work / _case_name(case)).read_text()) for case in cases(split)]

    report = {
        "dataset": "favorita",
        "split": split,
        "weeks": [SPLITS[split].start, SPLITS[split].stop - 1],
        "engine": engine,
        "summary": score_favorita(records),
        "records": records,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=str))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--raw", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("run")
    r.add_argument("--weekly", required=True)
    r.add_argument("--split", choices=sorted(SPLITS), required=True)
    r.add_argument("--jobs", type=int, default=4)
    r.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "build":
        print(json.dumps(build(args.raw, args.out), indent=1))
    else:
        print(json.dumps(run(args.weekly, args.split, args.jobs, args.out)["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
