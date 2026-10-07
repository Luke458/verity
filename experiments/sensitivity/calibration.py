"""Does the reported detection limit predict what the engine actually flags?

    python -m experiments.sensitivity.calibration --jobs 4 --out reports/sensitivity.json

For each injected single-category movement of size m, the affected category's
dollar and units leaves predict their own chance of being flagged from their
spread, degrees of freedom and the refresh's test family (the same quantities
behind ``detectable_change``). The prediction is compared with whether the
leaf was flagged.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

from qc.conformal import wilson_interval

MAGNITUDES = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4)
PROFILES: dict[str, dict[str, Any]] = {"small": {}, "realistic": {"restatement_weeks": 2}}
BINS = ((0.0, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 1.01))


def _t_cdf(x: float, df: int) -> float:
    from qc.temporal import student_t_two_sided_p

    tail = student_t_two_sided_p(abs(x), df) / 2.0
    return 1.0 - tail if x >= 0 else tail


def predicted_probability(leaf: Any, magnitude: float, tests: int, q: float) -> float:
    """P(flagged) for a drop of ``magnitude`` if it were the refresh's only anomaly."""
    from qc.temporal import student_t_quantile

    df = leaf.share_n - 1
    floor = leaf.materiality / abs(leaf.forecast_median) if leaf.forecast_median else 0.0
    if magnitude < floor:
        return 0.0
    from qc.temporal import share_drop

    drop = share_drop(magnitude, leaf.share_level if leaf.share_level is not None else 0.0)
    shift = (-math.log(1.0 - drop) if leaf.share_method == "seasonal" else drop) / leaf.share_spread
    critical = student_t_quantile(1.0 - q / (2.0 * tests), df)
    return 1.0 - _t_cdf(critical - shift, df)


def run_case(job: tuple[str, int, float, str]) -> list[dict[str, Any]]:
    from qc.config import DatasetConfig
    from qc.run import run_qc
    from qcgen.config import dataset_calendar, suite_config
    from qcgen.scenarios import build_scenario
    from qcgen.sources import ScenarioSource

    profile, seed, magnitude, workdir = job
    root = Path(workdir) / f"{profile}-{seed}-{magnitude}"
    try:
        built = build_scenario(
            suite_config(profile, seed=seed), 0, root, "market_movement",
            ("source", "coded", "warehouse", "report"), oracle_root=root / "_oracle", magnitude=magnitude,
        )
        config = replace(DatasetConfig(), **dataset_calendar(profile), **PROFILES[profile])
        result = run_qc(ScenarioSource(built.directory), built.manifest["current_version"], built.manifest["previous_version"], config)
        commodity = built.oracle["cases"][0]["affected"]["commodities"][0]
        series = list(result.temporal.series) if result.temporal is not None else []
        tests = len(series) if config.temporal_fdr_enabled else 1
        out = []
        for leaf in series:
            if leaf.series_id != f"commodity_id:{commodity}" or leaf.metric not in ("dollar", "units"):
                continue
            if leaf.share_p is None or leaf.share_spread is None:
                continue
            out.append({
                "profile": profile, "seed": seed, "magnitude": magnitude, "metric": leaf.metric,
                "predicted": predicted_probability(leaf, magnitude, tests, config.temporal_fdr_q),
                "detectable_change": leaf.detectable_change,
                "flagged": bool(leaf.anomaly),
            })
        return out
    finally:
        shutil.rmtree(root, ignore_errors=True)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    observed = sum(r["flagged"] for r in rows) / n
    predicted = sum(r["predicted"] for r in rows) / n
    brier = sum((r["predicted"] - r["flagged"]) ** 2 for r in rows) / n
    base = sum((observed - r["flagged"]) ** 2 for r in rows) / n
    bins = []
    for low, high in BINS:
        cell = [r for r in rows if low <= r["predicted"] < high]
        if not cell:
            continue
        hits = sum(r["flagged"] for r in cell)
        ci = wilson_interval(hits, len(cell))
        mean_pred = sum(r["predicted"] for r in cell) / len(cell)
        bins.append({"bin": [low, min(high, 1.0)], "n": len(cell), "mean_predicted": mean_pred,
                     "observed": hits / len(cell), "ci": list(ci), "prediction_inside_ci": ci[0] <= mean_pred <= ci[1]})
    by_magnitude = {}
    for m in MAGNITUDES:
        cell = [r for r in rows if r["magnitude"] == m]
        if cell:
            by_magnitude[str(m)] = {"n": len(cell), "predicted": sum(r["predicted"] for r in cell) / len(cell),
                                    "observed": sum(r["flagged"] for r in cell) / len(cell)}
    return {"series": n, "mean_predicted": predicted, "observed_rate": observed, "brier": brier,
            "brier_constant": base, "bins": bins, "by_magnitude": by_magnitude}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--workdir", default="reports/sensitivity-work")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", default="5001-5010", help="inclusive range, e.g. 7001-7010")
    args = parser.parse_args()
    first, last = (int(x) for x in args.seeds.split("-"))
    jobs = [(p, s, m, args.workdir) for p in PROFILES for s in range(first, last + 1) for m in MAGNITUDES]
    if args.jobs > 1:
        with ProcessPoolExecutor(args.jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
            rows = [row for case in pool.map(run_case, jobs) for row in case]
    else:
        rows = [row for job in jobs for row in run_case(job)]
    report: dict[str, Any] = {"all": summarize(rows), **{p: summarize([r for r in rows if r["profile"] == p]) for p in PROFILES}, "rows": rows}
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    for name in ("all", *PROFILES):
        s = report[name]
        print(f"{name}: n={s['series']} predicted {s['mean_predicted']:.3f} observed {s['observed_rate']:.3f} "
              f"brier {s['brier']:.3f} (constant {s['brier_constant']:.3f})")
        for b in s["bins"]:
            print(f"   bin {b['bin']}: n={b['n']} predicted {b['mean_predicted']:.2f} observed {b['observed']:.2f} "
                  f"CI [{b['ci'][0]:.2f}, {b['ci'][1]:.2f}] inside={b['prediction_inside_ci']}")
        print("   by magnitude:", {m: (round(v['predicted'], 2), round(v['observed'], 2)) for m, v in s["by_magnitude"].items()})


if __name__ == "__main__":
    main()
