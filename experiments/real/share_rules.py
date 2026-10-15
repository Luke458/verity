"""Latest-week share-test rules on the spent real-data weeks (no engine change).

    python -m experiments.real.share_rules tests --dataset favorita --weekly data/real/favorita/weekly.parquet \\
        --out reports/real/share-favorita.parquet
    python -m experiments.real.share_rules score --dataset favorita --tests reports/real/share-favorita.parquet

``tests`` runs the engine's ``share_shift_test`` for every province and family
leaf and every week from 54 on, from that week's raw history (training weeks
before it, the leaf and national totals as the actuals): once clean, and once
with the family's week cut by 40% (as the market-movement injector does). It
keeps the chosen test and both of its baselines. No statistic is computed for a
week of Favorita's unrun reserve split (126-141).

``score`` approximates the refresh-level decision on the spent weeks: one BH
family per refresh over every leaf of every measure plus one national test per
measure (p = 1), and a share impact of at least 2% of the leaf's recent mean
level. On Favorita test weeks 142-241 this flags the engine's 25 share-flagged
clean refreshes and one more (week 215). Detection is the mean, over every
family and week, of a 40% cut of that family being flagged with the other
leaves clean. Rules:

- ``current``: the engine's (the baseline with the smaller spread);
- ``trailing``, ``seasonal``: one baseline only; ``larger``: the larger p of
  the two; ``bonferroni``: twice the smaller p;
- ``calibrated``: the chosen t divided by the root mean square of the leaf's
  own chosen t at the ``--origins`` weeks before (in null-sd units, never
  below 1), so a leaf whose shares habitually move more than the test allows
  gets a wider interval.

``--exact-year-ago`` compares with the single week a year earlier instead of
the three-week mean centred on it. Results: docs/real-data-favorita.md.
"""

from __future__ import annotations

import argparse
import builtins
import inspect
import math
from typing import Any

import numpy as np
import pandas as pd

import qc.temporal as temporal
from qc.conformal import benjamini_hochberg

DATASETS: dict[str, dict[str, Any]] = {
    "m5": {"metrics": ("dollar", "units"), "weeks": range(54, 278), "spent": range(106, 278)},
    "favorita": {
        "metrics": ("units",),
        "weeks": [week for week in range(54, 242) if not 126 <= week <= 141],
        "spent": [*range(106, 126), *range(142, 242)],
    },
}
LEAVES = ("state_id", "commodity_id")
DROP = 0.4
WINDOW = 26  # the engine's: 2 x temporal_window
QS = (0.01, 0.003, 0.001, 0.0003)


def _share_test(exact_year_ago: bool):
    """The engine's share test, returning the chosen test and every candidate by method."""
    function = temporal.share_shift_test
    if exact_year_ago:
        source = inspect.getsource(temporal.share_shift_test).replace(
            "for other in (week - season - 1, week - season, week - season + 1)", "for other in (week - season,)"
        )
        namespace = dict(temporal.__dict__)
        exec(source, namespace)
        function = namespace["share_shift_test"]

    def run(*args: Any, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        candidates: dict[str, Any] = {}

        def spy(items: Any, *more: Any, key: Any = None) -> Any:
            # The test ends with min(candidates, key=spread): record them all.
            if not more and key is not None and items and hasattr(items[0][1], "method"):
                candidates.update({test.method: test for _, test in items})
            return builtins.min(items, *more, key=key) if key else builtins.min(items, *more)

        function.__globals__["min"] = spy
        try:
            chosen = function(*args, **kwargs)
        finally:
            del function.__globals__["min"]
        return chosen, candidates

    return run


def leaf_tests(weekly: pd.DataFrame, dataset: str, exact_year_ago: bool = False) -> pd.DataFrame:
    spec = DATASETS[dataset]
    run = _share_test(exact_year_ago)
    rows = []
    for metric in spec["metrics"]:
        national = weekly.groupby("week")[metric].sum()
        for column in LEAVES:
            grouped = weekly.groupby([column, "week"])[metric].sum()
            for leaf in grouped.index.get_level_values(0).unique():
                series = grouped[leaf]
                for week in spec["weeks"]:
                    if week not in series.index or week not in national.index:
                        continue
                    history = series[series.index < week]
                    parent = national[national.index < week]
                    if len(history) < 26:
                        continue
                    child, total = float(series[week]), float(national[week])
                    for drop in (0.0, DROP) if column == "commodity_id" else (0.0,):
                        chosen, candidates = run(
                            history.index, history.values, parent.index, parent.values,
                            child * (1 - drop), total - drop * child, WINDOW, target_week=week, season=52,
                        )
                        if chosen is None:
                            continue
                        row = {
                            "metric": metric, "leaf": f"{column}:{leaf}", "week": week, "drop": drop,
                            "level": float(history.iloc[-13:].mean()),
                            "t": chosen.t, "p": chosen.p, "n": chosen.n, "impact": chosen.impact,
                        }
                        for method in ("trailing", "seasonal"):
                            test = candidates.get(method)
                            row[f"{method}_p"] = test.p if test else np.nan
                        rows.append(row)
    return pd.DataFrame(rows)


def add_rules(tests: pd.DataFrame, origins: int = 52, min_origins: int = 20) -> pd.DataFrame:
    tests = tests.copy()
    both = tests["trailing_p"].notna() & tests["seasonal_p"].notna()
    tests["current"] = tests["p"]
    tests["trailing"] = tests["trailing_p"].fillna(tests["p"])
    tests["seasonal"] = tests["seasonal_p"].fillna(tests["p"])
    tests["larger"] = np.fmax(tests["trailing_p"], tests["seasonal_p"]).where(both, tests["p"])
    tests["bonferroni"] = np.minimum(1.0, 2 * np.fmin(tests["trailing_p"], tests["seasonal_p"])).where(both, tests["p"])
    clean = tests.loc[tests["drop"] == 0.0].set_index(["metric", "leaf"])[["week", "t"]].sort_index()
    rms = []
    for metric, leaf, week in zip(tests["metric"], tests["leaf"], tests["week"], strict=True):
        past = clean.loc[(metric, leaf)]
        past = past.loc[(past["week"] >= week - origins) & (past["week"] < week), "t"].to_numpy()
        rms.append(math.sqrt(float(np.mean(past**2))) if len(past) >= min_origins else np.nan)
    df = (tests["n"] - 1).clip(lower=3)
    scale = (pd.Series(rms, index=tests.index) / np.sqrt(df / (df - 2))).clip(lower=1.0).fillna(1.0)
    tests["calibrated"] = [
        temporal.student_t_two_sided_p(t / factor, int(freedom))
        for t, factor, freedom in zip(tests["t"], scale, df, strict=True)
    ]
    return tests


def _flags(p: np.ndarray, extra: int, q: float) -> np.ndarray:
    _, significant = benjamini_hochberg([*p, *([1.0] * extra)], q)
    return np.asarray(significant[: len(p)], dtype=bool)


def score(tests: pd.DataFrame, rule: str, weeks: Any, q: float) -> dict[str, Any]:
    clean = tests.loc[tests["drop"] == 0.0]
    cut = tests.loc[tests["drop"] == DROP]
    extra = clean["metric"].nunique()
    flagged, hits, cases = [], 0, 0
    for week in weeks:
        now = clean.loc[clean["week"] == week]
        if now.empty:
            continue
        p = now[rule].fillna(1.0).to_numpy()
        material = (now["impact"].abs() >= 0.02 * now["level"]).to_numpy()
        if (_flags(p, extra, q) & material).any():
            flagged.append(int(week))
        for _, row in cut.loc[cut["week"] == week].iterrows():
            mask = ((now["leaf"] == row["leaf"]) & (now["metric"] == row["metric"])).to_numpy()
            if not mask.any():
                continue
            moved = p.copy()
            moved[mask] = 1.0 if pd.isna(row[rule]) else row[rule]
            hits += bool(_flags(moved, extra, q)[mask].any() and abs(row["impact"]) >= 0.02 * row["level"])
            cases += 1
    refreshes = clean.loc[clean["week"].isin(list(weeks)), "week"].nunique()
    return {"rule": rule, "q": q, "flagged": len(flagged), "refreshes": refreshes,
            "detection_40": round(hits / max(cases, 1), 3), "flagged_weeks": flagged}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("tests")
    t.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    t.add_argument("--weekly", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--exact-year-ago", action="store_true")
    s = sub.add_parser("score")
    s.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    s.add_argument("--tests", required=True)
    s.add_argument("--origins", type=int, default=52)
    args = parser.parse_args(argv)
    if args.command == "tests":
        columns = ["week", *LEAVES, *DATASETS[args.dataset]["metrics"]]
        frame = leaf_tests(pd.read_parquet(args.weekly, columns=columns), args.dataset, args.exact_year_ago)
        frame.to_parquet(args.out)
        print(f"{len(frame)} leaf tests -> {args.out}")
        return 0
    tests = add_rules(pd.read_parquet(args.tests), origins=args.origins, min_origins=min(20, args.origins // 2))
    spent = DATASETS[args.dataset]["spent"]
    clean = tests.loc[(tests["drop"] == 0.0) & tests["week"].isin(list(spent))]
    print(f"clean leaf tests {len(clean)}; chosen t sd {clean['t'].std():.2f}")
    for rule in ("current", "trailing", "seasonal", "larger", "bonferroni", "calibrated"):
        below = int((clean[rule] < 0.01).sum())
        curve = [score(tests, rule, spent, q) for q in QS]
        points = " | ".join(f"q {c['q']}: {c['flagged']}/{c['refreshes']} {c['detection_40']:.2f}" for c in curve)
        print(f"{rule:10s} leaf p<0.01 {below:4d} (expected {0.01 * len(clean):.0f})  {points}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
