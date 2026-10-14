"""The engine on real sales: Walmart's M5 competition data.

    python -m experiments.real.m5 build --raw data/real/m5 --out data/real/m5/weekly.parquet
    python -m experiments.real.m5 run --weekly data/real/m5/weekly.parquet --split dev --jobs 4 \\
        --out reports/real/m5-dev.json
    python -m experiments.real.m5 release --raw data/real/m5 --out reports/real/m5-release.json

M5 (https://www.kaggle.com/competitions/m5-forecasting-accuracy; mirrored at
https://huggingface.co/datasets/kashif/M5) holds daily unit sales of 3,049
products in 10 stores across 3 US states, 2011-01-29 to 2016-05-22, with weekly
shelf prices. ``build`` aggregates it to Walmart weeks (Saturday-Friday; the
trailing partial week is dropped) at store x product grain, keeps only rows that
sold (a sales feed has no row for a product that did not sell), and prices
units at the week's shelf price to give dollars.

A refresh is the table through week ``k - 1`` followed by the table through
week ``k``. M5's two public releases agree on every overlapping cell
(``release``), so these refreshes are clean by construction: history never
changes and every alarm is a false one. Faulted refreshes inject the
generator's own single-stage faults into the current version. The engine runs
with its defaults; ``CONFIG`` only maps M5's columns and its single stage.
Registered in docs/real-data.md before the held-out run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qc.config import DatasetConfig

STAGE = "report"
CONFIG = replace(
    DatasetConfig(),
    name="m5-walmart",
    entity_columns=("store_id", "product_id", "commodity_id", "state_id"),
    metric_columns=("dollar", "units"),
    analysis_stage=STAGE,
    contract_stage=STAGE,
    stage_preference=(STAGE,),
    pipeline_order=(STAGE,),
    report_grain=("state_id",),
    # One stage at store x product grain: that is the key the contracts check.
    stage_keys=((STAGE, ("store_id", "product_id")),),
    hierarchy_columns=(("state_id",),),
    explanation_entity_types=("store", "state", "product", "commodity"),
    snapshot_metrics=(),
    temporal_entity_columns=("state_id", "commodity_id"),
)

# Week index k is the refresh that appends week k. Annual baselines need 104
# completed weeks, so the earliest assessed refresh appends week 106.
# ``confirm`` (weeks 126-177) was held back from the test and registered for
# the price-reference fix (docs/real-data.md).
SPLITS = {"dev": range(106, 126), "confirm": range(126, 178), "test": range(178, 278)}

# (family, injector parameters, cut points per split): faults are injected at
# every ``stride``-th cut point of the split.
FAULTS: tuple[tuple[str, dict[str, Any]], ...] = (
    ("market_movement", {"factor": 0.95}),
    ("market_movement", {"factor": 0.9}),
    ("market_movement", {"factor": 0.8}),
    ("market_movement", {"factor": 0.6}),
    ("missing_stores", {"n_stores": 1}),
    ("missing_products", {"n_products": 10}),
    ("missing_products", {"n_products": 100}),
    ("week_restatement", {"factor": 0.9}),
    ("recalculation", {"sigma": 0.004}),
)
FAULT_STRIDE = {"dev": 4, "confirm": 4, "test": 5}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def weekly_frame(raw: Path, sales_file: str = "sales_train_evaluation.csv") -> pd.DataFrame:
    """Store x product x week sales; only rows that sold, full weeks only."""
    calendar = pd.read_csv(raw / "calendar.csv", usecols=["d", "wm_yr_wk"])
    sales = pd.read_csv(raw / sales_file)
    days = [column for column in sales.columns if column.startswith("d_")]
    week_of_day = calendar.set_index("d").loc[days, "wm_yr_wk"].to_numpy()
    walmart_weeks, day_counts = np.unique(week_of_day, return_counts=True)
    full = walmart_weeks[day_counts == 7]
    index = {int(week): position + 1 for position, week in enumerate(full)}

    daily = sales[days].to_numpy(dtype=np.int32)
    keep = np.isin(week_of_day, full)
    week_positions = np.searchsorted(full, week_of_day[keep])
    weekly = np.zeros((len(sales), len(full)), dtype=np.int32)
    np.add.at(weekly.T, week_positions, daily[:, keep].T)

    rows, columns = np.nonzero(weekly)
    frame = pd.DataFrame({
        "week": columns.astype(np.int32) + 1,
        "store_id": sales["store_id"].to_numpy()[rows],
        "state_id": sales["state_id"].to_numpy()[rows],
        "commodity_id": sales["dept_id"].to_numpy()[rows],
        "product_id": sales["item_id"].to_numpy()[rows],
        "units": weekly[rows, columns],
        "wm_yr_wk": full[columns],
    })
    prices = pd.read_csv(raw / "sell_prices.csv")
    frame = frame.merge(
        prices.rename(columns={"item_id": "product_id"}),
        on=["store_id", "product_id", "wm_yr_wk"],
        how="left",
        validate="many_to_one",
    )
    if frame["sell_price"].isna().any():
        raise ValueError(f"{int(frame['sell_price'].isna().sum())} sold rows have no shelf price")
    frame["dollar"] = (frame["units"] * frame["sell_price"]).astype(np.float32)
    frame = frame.drop(columns=["sell_price"])
    frame.attrs["weeks"] = index
    for column in ("store_id", "state_id", "commodity_id", "product_id"):
        frame[column] = frame[column].astype(str)
    return frame.sort_values(["week", "store_id", "product_id"], ignore_index=True)


def build(raw: Path, out: Path) -> dict[str, Any]:
    frame = weekly_frame(raw)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out, index=False)
    provenance = {
        "source": "https://huggingface.co/datasets/kashif/M5",
        "inputs": {
            name: _sha256(raw / name)
            for name in ("calendar.csv", "sales_train_evaluation.csv", "sell_prices.csv")
        },
        "rows": len(frame),
        "weeks": int(frame["week"].max()),
        "first_walmart_week": int(frame["wm_yr_wk"].min()),
        "last_walmart_week": int(frame["wm_yr_wk"].max()),
        "stores": int(frame["store_id"].nunique()),
        "products": int(frame["product_id"].nunique()),
        "departments": sorted(frame["commodity_id"].unique().tolist()),
    }
    out.with_suffix(".json").write_text(json.dumps(provenance, indent=2))
    return provenance


class FrameSource:
    """A VersionSource over in-memory versions, one stage each."""

    def __init__(self, versions: dict[str, pd.DataFrame], products: pd.DataFrame) -> None:
        self.versions = versions
        self.products = products

    def list_versions(self) -> list[str]:
        return sorted(self.versions)

    def available_stages(self, version: str) -> list[str]:
        return [STAGE]

    def resolve_stage(self, version: str, preferred: Any) -> str:
        return STAGE

    def read_fact(self, version: str, stage: str) -> pd.DataFrame:
        return self.versions[version]

    def read_dim(self, version: str, name: str) -> pd.DataFrame | None:
        return self.products if name == "products" else None


_WEEKLY: pd.DataFrame | None = None


def _weekly(path: str) -> pd.DataFrame:
    global _WEEKLY
    if _WEEKLY is None:
        _WEEKLY = pd.read_parquet(path).drop(columns=["wm_yr_wk"])
    return _WEEKLY


def cases(split: str) -> list[dict[str, Any]]:
    weeks = list(SPLITS[split])
    out: list[dict[str, Any]] = [
        {"family": "clean", "params": {}, "week": week, "seed": week} for week in weeks
    ]
    stride = FAULT_STRIDE[split]
    for index, (family, params) in enumerate(FAULTS):
        # Each fault size gets its own offset so sizes do not share cut points.
        for week in weeks[index % stride::stride]:
            out.append({"family": family, "params": dict(params), "week": week, "seed": week * 100 + index})
    return out


def _summary(result: Any, case: dict[str, Any], truth: dict[str, Any]) -> dict[str, Any]:
    machine = result.machine
    latest = machine.get("latest_week") or {}
    anomalous = [
        {
            "series": item["series_id"],
            "metric": item["metric"],
            "adjusted_p": item.get("adjusted_p"),
            "relative_residual": item.get("relative_residual"),
            "flags": item.get("flags", []),
        }
        for item in latest.get("series", [])
        if item.get("anomaly")
    ]
    review = sorted({
        item["check"] for item in machine.get("findings", [])
        if item.get("disposition") in ("UNEXPLAINED_ANOMALY", "HARD_FAILURE", "UNAVAILABLE_EVIDENCE")
    })
    history = machine.get("historical_revision") or {}
    return {
        **case,
        "status": result.status,
        "history_reason": history.get("reason"),
        "review_checks": review,
        "missing_entities": machine.get("missing_entities") or {},
        "anomalies": anomalous,
        "week_revisions": [item["week"] for item in machine.get("week_revisions", [])],
        "truth": truth,
        "detected": _detected(case["family"], result.status, review, history, anomalous, truth, machine),
    }


def _detected(
    family: str,
    status: str,
    review: list[str],
    history: dict[str, Any],
    anomalous: list[dict[str, Any]],
    truth: dict[str, Any],
    machine: dict[str, Any],
) -> bool:
    """The fault is flagged by the check that should see it (registered rule)."""
    if status == "PASS":
        return False
    if family == "market_movement":
        target = f"commodity_id:{truth['commodities'][0]}"
        return any(item["series"] == target for item in anomalous)
    if family == "missing_stores":
        return bool((machine.get("missing_entities") or {}).get("store", {}).get("material"))
    if family == "missing_products":
        return bool((machine.get("missing_entities") or {}).get("product", {}).get("material"))
    if family in ("week_restatement", "recalculation"):
        return "historical_revision" in review or "revision_week" in review
    return status != "PASS"


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
            scenario_id=f"m5-{week}", family=case["family"], stage=STAGE,
            current_week=week, rng=np.random.default_rng(int(case["seed"])),
        )
        state, ground = INJECTORS[case["family"]]({"fact": current.copy(), "stores": stores}, context, case["params"])
        current = state["fact"]
        truth = dict(ground.affected)
    products = weekly[["product_id", "commodity_id"]].drop_duplicates("product_id")
    source = FrameSource({f"W{week - 1:04d}": previous, f"W{week:04d}": current}, products)
    result = run_qc(source, f"W{week:04d}", f"W{week - 1:04d}", CONFIG, observed_at="2030-01-01T00:00:00+00:00")
    return _summary(result, case, truth)


def _wilson(hits: int, n: int, z: float = 1.959964) -> list[float]:
    if n == 0:
        return [0.0, 1.0]
    p = hits / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(float(centre - half), 4), round(float(centre + half), 4)]


def score(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        key = record["family"]
        if record["family"] == "market_movement":
            key += f"@{1 - record['params']['factor']:.2f}"
        elif record["family"] == "missing_products":
            key += f"@{record['params']['n_products']}"
        groups.setdefault(key, []).append(record)
    summary: dict[str, Any] = {}
    for key, items in sorted(groups.items()):
        if key == "clean":
            alarms = sum(item["status"] != "PASS" for item in items)
            by_check: dict[str, int] = {}
            for item in items:
                for check in item["review_checks"]:
                    by_check[check] = by_check.get(check, 0) + 1
            summary[key] = {
                "n": len(items), "false_alarms": alarms, "ci": _wilson(alarms, len(items)),
                "by_check": dict(sorted(by_check.items())),
            }
        else:
            hits = sum(item["detected"] for item in items)
            flagged = sum(item["status"] != "PASS" for item in items)
            summary[key] = {"n": len(items), "detected": hits, "flagged_any": flagged, "ci": _wilson(hits, len(items))}
    return summary


def run(weekly_path: str, split: str, jobs: int, out: Path) -> dict[str, Any]:
    todo = cases(split)
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=jobs, mp_context=context) as pool:
        records = list(pool.map(run_case, [weekly_path] * len(todo), todo))
    from qc.cohort import code_sha256

    report = {
        "split": split,
        "weeks": [SPLITS[split].start, SPLITS[split].stop - 1],
        "engine": code_sha256()[:12],
        "summary": score(records),
        "records": records,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=str))
    return report


def release(raw: Path, out: Path) -> dict[str, Any]:
    """Run the engine on M5's own two releases (validation -> evaluation)."""
    from qc.run import run_qc

    previous = weekly_frame(raw, "sales_train_validation.csv").drop(columns=["wm_yr_wk"])
    current = weekly_frame(raw, "sales_train_evaluation.csv").drop(columns=["wm_yr_wk"])
    products = current[["product_id", "commodity_id"]].drop_duplicates("product_id")
    source = FrameSource({"V1-validation": previous, "V2-evaluation": current}, products)
    result = run_qc(source, "V2-evaluation", "V1-validation", CONFIG, observed_at="2030-01-01T00:00:00+00:00")
    case = {"family": "release", "params": {}, "week": int(current["week"].max()), "seed": 0}
    record = _summary(result, case, {})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1, default=str))
    from qc.reporting import render_markdown

    out.with_suffix(".md").write_text(render_markdown(result))
    return record


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
    rel = sub.add_parser("release")
    rel.add_argument("--raw", type=Path, required=True)
    rel.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "build":
        print(json.dumps(build(args.raw, args.out), indent=1))
    elif args.command == "run":
        report = run(args.weekly, args.split, args.jobs, args.out)
        print(json.dumps(report["summary"], indent=1))
    else:
        record = release(args.raw, args.out)
        print(json.dumps({k: record[k] for k in ("status", "review_checks", "history_reason")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
