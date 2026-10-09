"""Refresh sequences through ``qc weekly``: recurrence and slow movements.

    python -m experiments.sequences.run --seeds 6301-6310 --jobs 4 --out reports/sequences/test.json

Every other measurement in this repo scores one version pair. Here one seeded
world is published as 11 consecutive weekly snapshots (each a full refresh,
the last ``late_arrival_weeks`` under-counted on ``realistic``), and
``qc weekly`` assesses the 10 consecutive pairs with a journal, as deployed,
once with recurrence on and once off. Kinds of sequence:

- ``clean``: no change;
- ``drift``: one category's sales fall a further ``DRIFT`` per week from the
  first assessed week on (a slow decline);
- ``step``: one category's sales drop by ``STEP`` from the first assessed week
  on and stay there (a sustained small shift).

Both movements are genuine and should eventually be flagged. Pre-registered in
docs/evaluation.md ("Recurrence and slow movements").
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROFILES: dict[str, dict[str, Any]] = {"small": {}, "realistic": {"restatement_weeks": 2}}
KINDS = ("clean", "drift", "step")
REFRESHES = 10
DRIFT = 0.03
STEP = 0.10


def _move(truth: pd.DataFrame, products: set[str], first_week: int, kind: str) -> pd.DataFrame:
    if kind == "clean":
        return truth
    weeks = truth["week"].to_numpy()
    age = np.maximum(weeks - first_week + 1, 0)
    if kind == "drift":
        factor = (1.0 - DRIFT) ** age
    else:
        factor = np.where(age > 0, 1.0 - STEP, 1.0)
    factor = np.where(truth["product_id"].isin(products).to_numpy(), factor, 1.0)
    truth = truth.copy()
    truth["dollar"] = (truth["dollar"].to_numpy(dtype=np.float64) * factor).astype(np.float32)
    units = np.rint(truth["units"].to_numpy(dtype=np.float64) * factor)
    scripts = truth["scripts"].to_numpy(dtype=np.float64)
    truth["scripts"] = np.where(scripts > 0, np.minimum(scripts, units), 0.0).astype(np.float32)
    truth["units"] = units.astype(np.int32)
    return truth


def build_sequence(profile: str, seed: int, kind: str, root: Path) -> dict[str, Any]:
    """Write 11 snapshots and a Parquet manifest; return what was injected."""
    from qcgen.config import suite_config
    from qcgen.dgp import generate_history, week_end_dates
    from qcgen.scenarios import _late_arrival
    from qcgen.stages import ADVANCE, build_source
    from qcgen.universe import generate_universe

    config = suite_config(profile, seed=seed)
    rng = np.random.default_rng(seed)
    total = config.history.n_weeks
    universe = generate_universe(config.universe, rng, horizon_weeks=total)
    truth = generate_history(universe, config.history, rng)
    commodity = str(rng.choice(sorted(universe.products["commodity_id"].unique())))
    products = set(universe.products.loc[universe.products["commodity_id"] == commodity, "product_id"].astype(str))
    first = total - REFRESHES + 1  # latest week of the first assessed refresh
    truth = _move(truth, products, first, kind)
    calendar = pd.DataFrame({"week": np.arange(1, total + 1, dtype=np.int32),
                             "week_end": week_end_dates(config.history.start_week, total)})
    root.mkdir(parents=True, exist_ok=True)
    snapshots = []
    for end in range(first - 1, total + 1):
        snapshot = _late_arrival(truth.loc[truth["week"] <= end].copy(), config, end + 1, rng)
        state = build_source(snapshot, universe, calendar.loc[calendar["week"] <= end])
        version = f"w{end}"
        stages = {}
        for stage in ("coded", "warehouse", "report"):
            state = ADVANCE[stage](state)
            if stage in ("warehouse", "report"):
                state["fact"].to_parquet(root / f"{version}-{stage}.parquet", index=False)
                stages[stage] = f"{version}-{stage}.parquet"
        dims = {}
        for name in universe.dimensions():
            state[name].to_parquet(root / f"{version}-{name}.parquet", index=False)
            dims[name] = f"{version}-{name}.parquet"
        snapshots.append({"version": version, "observed_at": "", "stages": stages, "dimensions": dims})
    for index, item in enumerate(snapshots):  # one week apart, in order
        item["observed_at"] = (pd.Timestamp("2026-01-03", tz="UTC") + pd.Timedelta(weeks=index)).isoformat()
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "source_id": f"{profile}-{seed}-{kind}",
                                    "snapshots": snapshots}))
    return {"manifest": str(manifest), "commodity": commodity, "versions": [s["version"] for s in snapshots],
            "first_week": first}


def run_job(job: tuple[str, int, str, str]) -> list[dict[str, Any]]:
    from qc.config import DatasetConfig
    from qc.weekly import run_weekly
    from qcgen.config import dataset_calendar

    profile, seed, kind, workdir = job
    root = Path(workdir) / f"{profile}-{seed}-{kind}"
    shutil.rmtree(root, ignore_errors=True)
    try:
        built = build_sequence(profile, seed, kind, root / "data")
        rows = []
        for recurrence in (True, False):
            config = replace(DatasetConfig(), name=f"seq-{profile}", recurrence_enabled=recurrence,
                             **dataset_calendar(profile), **PROFILES[profile])
            store = root / f"journal-{recurrence}.db"
            versions = built["versions"]
            for step, (previous, current) in enumerate(zip(versions, versions[1:], strict=False)):
                result = run_weekly(built["manifest"], config=config, store_path=store,
                                    out_root=root / f"reports-{recurrence}", stage="warehouse",
                                    current=current, previous=previous)
                report = json.loads((Path(result.report_dir) / "report.json").read_text())
                unexplained = [f for f in report["findings"] if f["disposition"] == "UNEXPLAINED_ANOMALY"]
                leaf = f"commodity_id:{built['commodity']}"
                rows.append({
                    "profile": profile, "seed": seed, "kind": kind, "recurrence": recurrence, "step": step,
                    "status": result.status,
                    "recurrence_findings": sum(f["check"] == "recurrence" for f in report["findings"]),
                    "flagged_leaf": any(f["check"] == "temporal" and f["scope"] == leaf for f in unexplained),
                    "unexplained": sorted({f"{f['check']}|{f['scope']}|{f['metric']}" for f in unexplained}),
                })
        return rows
    finally:
        shutil.rmtree(root, ignore_errors=True)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from qc.conformal import wilson_interval

    out: dict[str, Any] = {}
    on = {(r["profile"], r["seed"], r["kind"], r["step"]): r for r in rows if r["recurrence"]}
    off = {(r["profile"], r["seed"], r["kind"], r["step"]): r for r in rows if not r["recurrence"]}
    keys = sorted(set(on) & set(off))
    out["status_identical"] = {"same": sum(on[k]["status"] == off[k]["status"] for k in keys), "n": len(keys)}
    with_recurrence = [r for r in on.values() if r["recurrence_findings"]]
    out["recurrence_findings"] = {
        "refreshes": len(with_recurrence),
        "already_investigate_without_it": sum(off[(r["profile"], r["seed"], r["kind"], r["step"])]["status"]
                                              == "INVESTIGATE" for r in with_recurrence),
    }
    for profile in PROFILES:
        mine = [r for r in on.values() if r["profile"] == profile]
        clean = [r for r in mine if r["kind"] == "clean"]
        flagged = sum(r["status"] != "PASS" for r in clean)
        low, high = wilson_interval(flagged, len(clean)) if clean else (0.0, 1.0)
        entry: dict[str, Any] = {"clean_refreshes_flagged": {"hits": flagged, "n": len(clean), "ci": [low, high]}}
        for kind in ("drift", "step"):
            delays = []
            for seed in sorted({r["seed"] for r in mine if r["kind"] == kind}):
                steps = sorted((r for r in mine if r["kind"] == kind and r["seed"] == seed), key=lambda r: r["step"])
                first = next((r["step"] for r in steps if r["flagged_leaf"]), None)
                delays.append(first)
            entry[kind] = {"sequences": len(delays), "ever_flagged": sum(d is not None for d in delays),
                           "first_flagged_step": delays,
                           "cumulative_change_at_step": [round(1 - (1 - DRIFT) ** (s + 1), 3) if kind == "drift"
                                                         else STEP for s in range(REFRESHES)]}
        out[profile] = entry
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", default="6301-6302", help="inclusive range")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--workdir", default="reports/sequences-work")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    first, last = (int(x) for x in args.seeds.split("-"))
    jobs = [(profile, seed, kind, args.workdir) for profile in PROFILES for seed in range(first, last + 1) for kind in KINDS]
    if args.jobs > 1:
        with ProcessPoolExecutor(args.jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
            rows = [row for chunk in pool.map(run_job, jobs) for row in chunk]
    else:
        rows = [row for job in jobs for row in run_job(job)]
    report = {"seeds": args.seeds, "summary": summarize(rows), "rows": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report["summary"], indent=1))


if __name__ == "__main__":
    main()
