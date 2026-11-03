"""Latest-week share-test rules on the spent real-data weeks (no engine change).

    python -m experiments.real.share_rules tests --dataset favorita --weekly data/real/favorita/weekly.parquet \\
        --out reports/real/share-favorita.parquet
    python -m experiments.real.share_rules score --dataset favorita --tests reports/real/share-favorita.parquet

``tests`` runs the engine's ``share_shift_test`` for every province and family
leaf and every week from 54 on, from that week's raw history (training weeks
before it, the leaf and national totals as the actuals): once clean, and once
with the family's week cut by 40% (``--drop``; as the market-movement injector does). It
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
    # A qcgen feed no pretrained model can have seen (``build-synthetic``), M5's length.
    "synthetic": {"metrics": ("dollar", "units"), "weeks": range(54, 279), "spent": range(106, 279)},
}
SYNTHETIC_WEEKS = 278
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


def leaf_tests(weekly: pd.DataFrame, dataset: str, exact_year_ago: bool = False, drop: float = DROP) -> pd.DataFrame:
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
                    for cut in (0.0, drop) if column == "commodity_id" else (0.0,):
                        chosen, candidates = run(
                            history.index, history.values, parent.index, parent.values,
                            child * (1 - cut), total - cut * child, WINDOW, target_week=week, season=52,
                        )
                        if chosen is None:
                            continue
                        row = {
                            "metric": metric, "leaf": f"{column}:{leaf}", "week": week, "drop": cut,
                            "level": float(history.iloc[-13:].mean()),
                            "t": chosen.t, "p": chosen.p, "n": chosen.n, "impact": chosen.impact,
                        }
                        for method in ("trailing", "seasonal"):
                            test = candidates.get(method)
                            row[f"{method}_p"] = test.p if test else np.nan
                            # Relative spread and impact, for the min-spread hybrids in ``add_rules``.
                            row[f"{method}_spread"] = float(test.spread) if test else np.nan
                            row[f"{method}_effect"] = test.impact if test else np.nan
                        rows.append(row)
    return pd.DataFrame(rows)


def synthetic_weekly(seed: int) -> pd.DataFrame:
    """A store x product x week feed from qcgen: the ``realistic`` history
    assumptions (category seasonality, market and category shocks,
    intermittency) over ``SYNTHETIC_WEEKS`` weeks on the ``full`` universe
    (5 states, 15 commodities)."""
    from dataclasses import replace

    from qcgen.config import profile_history, profile_universe
    from qcgen.dgp import generate_history
    from qcgen.universe import generate_universe

    rng = np.random.default_rng(seed)
    universe = generate_universe(profile_universe("full"), rng, horizon_weeks=SYNTHETIC_WEEKS)
    history = generate_history(universe, replace(profile_history("realistic"), n_weeks=SYNTHETIC_WEEKS), rng)
    return (
        history.merge(universe.stores[["store_id", "state_id"]], on="store_id")
        .merge(universe.products[["product_id", "commodity_id"]], on="product_id")
        [["week", *LEAVES, "dollar", "units"]]
        .astype({"dollar": "float64", "units": "float64"})
    )


def timestral_tests(weekly: pd.DataFrame, dataset: str, tests: pd.DataFrame, model: Any) -> pd.DataFrame:
    """Add the pretrained share baseline's p, z and impact to every row of ``tests``.

    Each (metric, leaf type, week) is one batched forecast from the same
    history the engine rule saw (weeks before ``week``); the clean and the cut
    observation of a leaf are scored against the same quantiles, with the cut
    taken out of the parent as in ``leaf_tests``. A row the model cannot score
    stays NaN (p = 1 in ``score``).
    """
    from qc import share_model

    spec = DATASETS[dataset]
    rows = {key: i for i, key in enumerate(zip(tests["metric"], tests["leaf"], tests["week"], tests["drop"], strict=True))}
    p, z, impact, spread = (np.full(len(tests), np.nan) for _ in range(4))
    weeks = sorted(int(week) for week in tests["week"].unique())
    drops = sorted(float(drop) for drop in tests["drop"].unique())
    for metric in spec["metrics"]:
        national = weekly.groupby("week")[metric].sum()
        for column in LEAVES:
            grouped = weekly.groupby([column, "week"])[metric].sum()
            series = {leaf: grouped[leaf] for leaf in grouped.index.get_level_values(0).unique()}
            for week in weeks:
                names = [
                    leaf for leaf in series
                    if any((metric, f"{column}:{leaf}", week, drop) in rows for drop in drops)
                ]
                if not names:
                    continue
                parent = national[national.index < week]
                histories = [series[leaf][series[leaf].index < week] for leaf in names]
                built = share_model.share_rows(
                    [(history.index, history.to_numpy()) for history in histories], parent.index, parent.to_numpy()
                )
                if built is None:
                    continue
                context, _, last_week = built
                quantiles, levels = model.forecast(context, max(1, week - last_week))
                total = float(national[week])
                for position, leaf in enumerate(names):
                    child = float(series[leaf][week])
                    for drop in drops:
                        row = rows.get((metric, f"{column}:{leaf}", week, drop))
                        if row is None:
                            continue
                        parent_now = total - drop * child
                        scored = share_model.quantile_p(child * (1 - drop) / parent_now, quantiles[position], levels)
                        if scored is None:
                            continue
                        z[row], p[row], median, spread[row] = scored
                        impact[row] = (child * (1 - drop) / parent_now - median) * parent_now
    return tests.assign(timestral=p, timestral_z=z, timestral_impact=impact, timestral_spread=spread)


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
    if "timestral_spread" in tests and "seasonal_spread" in tests:
        # Timestral against the seasonal baseline, and against both engine
        # baselines: whichever candidate has the smallest relative spread.
        for rule, others in (("hybrid", ("seasonal",)), ("hybrid3", ("trailing", "seasonal"))):
            best = tests["timestral_spread"].where(tests["timestral"].notna(), np.inf)
            tests[rule], tests[f"{rule}_impact"] = tests["timestral"], tests["timestral_impact"]
            for method in others:
                tighter = tests[f"{method}_spread"].fillna(np.inf) < best
                tests.loc[tighter, rule] = tests.loc[tighter, f"{method}_p"]
                tests.loc[tighter, f"{rule}_impact"] = tests.loc[tighter, f"{method}_effect"]
                best = best.where(~tighter, tests[f"{method}_spread"])
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
    cut = tests.loc[tests["drop"] > 0.0]
    extra = clean["metric"].nunique()
    # A rule with its own baseline is judged material on its own impact.
    impact = f"{rule}_impact" if f"{rule}_impact" in tests else "impact"
    flagged, hits, cases = [], 0, 0
    for week in weeks:
        now = clean.loc[clean["week"] == week]
        if now.empty:
            continue
        p = now[rule].fillna(1.0).to_numpy()
        material = (now[impact].abs() >= 0.02 * now["level"]).to_numpy()
        if (_flags(p, extra, q) & material).any():
            flagged.append(int(week))
        for _, row in cut.loc[cut["week"] == week].iterrows():
            mask = ((now["leaf"] == row["leaf"]) & (now["metric"] == row["metric"])).to_numpy()
            if not mask.any():
                continue
            moved = p.copy()
            moved[mask] = 1.0 if pd.isna(row[rule]) else row[rule]
            hits += bool(_flags(moved, extra, q)[mask].any() and abs(row[impact]) >= 0.02 * row["level"])
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
    t.add_argument("--drop", type=float, default=DROP, help="the family cut (default 0.4)")
    s = sub.add_parser("score")
    s.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    s.add_argument("--tests", required=True)
    s.add_argument("--origins", type=int, default=52)
    s.add_argument("--qs", type=float, nargs="+", default=list(QS))
    g = sub.add_parser("build-synthetic")
    g.add_argument("--seed", type=int, required=True)
    g.add_argument("--out", required=True)
    m = sub.add_parser("timestral")
    m.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    m.add_argument("--weekly", required=True)
    m.add_argument("--tests", required=True, help="a ``tests`` table; the output adds the timestral rule")
    m.add_argument("--out", required=True)
    m.add_argument("--model", default="mistralai/Timestral-28M")
    args = parser.parse_args(argv)
    if args.command == "build-synthetic":
        frame = synthetic_weekly(args.seed)
        frame.to_parquet(args.out)
        print(f"{len(frame)} rows, weeks {frame['week'].min()}-{frame['week'].max()} -> {args.out}")
        return 0
    if args.command == "timestral":
        from experiments.real.m5 import SHARE_MODEL_REVISIONS
        from qc.share_model import load_share_model

        columns = ["week", *LEAVES, *DATASETS[args.dataset]["metrics"]]
        model = load_share_model(args.model, SHARE_MODEL_REVISIONS[args.model])
        frame = timestral_tests(pd.read_parquet(args.weekly, columns=columns), args.dataset,
                                pd.read_parquet(args.tests), model)
        frame.to_parquet(args.out)
        print(f"{int(frame['timestral'].notna().sum())}/{len(frame)} rows scored -> {args.out}")
        return 0
    if args.command == "tests":
        columns = ["week", *LEAVES, *DATASETS[args.dataset]["metrics"]]
        frame = leaf_tests(
            pd.read_parquet(args.weekly, columns=columns), args.dataset, args.exact_year_ago, args.drop
        )
        frame.to_parquet(args.out)
        print(f"{len(frame)} leaf tests -> {args.out}")
        return 0
    tests = add_rules(pd.read_parquet(args.tests), origins=args.origins, min_origins=min(20, args.origins // 2))
    spent = DATASETS[args.dataset]["spent"]
    clean = tests.loc[(tests["drop"] == 0.0) & tests["week"].isin(list(spent))]
    print(f"clean leaf tests {len(clean)}; chosen t sd {clean['t'].std():.2f}")
    rules = ["current", "trailing", "seasonal", "larger", "bonferroni", "calibrated"]
    rules += [rule for rule in ("timestral", "hybrid", "hybrid3") if rule in tests]
    for rule in rules:
        below = int((clean[rule] < 0.01).sum())
        curve = [score(tests, rule, spent, q) for q in args.qs]
        points = " | ".join(f"q {c['q']}: {c['flagged']}/{c['refreshes']} {c['detection_40']:.2f}" for c in curve)
        print(f"{rule:10s} leaf p<0.01 {below:4d} (expected {0.01 * len(clean):.0f})  {points}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
