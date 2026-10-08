"""Do approved closures and category moves clear exactly their own findings?

    python -m experiments.approvals.run --seeds 6201-6220 --jobs 4 --out reports/approvals.json

For each closure (``missing_stores``) and category move (``commodity_remap``),
alone and paired with a second fault, the refresh is assessed blind and again
with the registry entry ``qc notices`` would draft for the change, approved.
Each second fault is also run alone on the same seed. The pair injects the
second fault first (``build_scenario`` draws specs primary first), so alone
and paired share the world and the second fault's parameters. They are not fully paired: a fault's
target (a market movement's category) is drawn at injection, after the
change. A missed market movement is therefore judged against the engine's
own reported detection limit for that category in the run that missed it.
A clean refresh per seed gives the world's background flags. Pre-registered
in docs/evaluation.md ("Approving closures and category moves").
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

from qc.conformal import wilson_interval

PROFILES: dict[str, dict[str, Any]] = {"small": {}, "realistic": {"restatement_weeks": 2}}
CHANGES = {"missing_stores": ("LATEST_WEEK_MISSING", "store"), "commodity_remap": ("POSSIBLE_RECLASSIFICATION", "commodity")}
SECOND = ("coding_error", "warehouse_transform_error", "market_movement", "week_restatement")
STAGES = ("source", "coded", "warehouse", "report")
PASSING = ("PASS", "PASS_WITH_EXPLANATION")


def _unexplained(result: Any) -> list[str]:
    return sorted({f"{f['check']}|{f['scope']}|{f['metric']}" for f in result.machine["findings"]
                   if f["disposition"] == "UNEXPLAINED_ANOMALY"})


def _market_leaf(built: Any, result: Any) -> dict[str, Any] | None:
    """The market movement's injected drop and its category's evidence in ``result``."""
    case = next((c for c in built.oracle["cases"] if c.get("family") == "market_movement"), None)
    if case is None or result.temporal is None:
        return None
    series_id = f"commodity_id:{case['affected']['commodities'][0]}"
    leaf = next((s for s in result.temporal.series if s.series_id == series_id and s.metric == "dollar"), None)
    if leaf is None:
        return None
    return {"drop": 1.0 - float(case["details"]["factor"]), "series": series_id, "flagged": bool(leaf.anomaly),
            "detectable_change": leaf.detectable_change, "detectable_change_80": leaf.detectable_change_80}


def _assess(profile: str, seed: int, family: str, also: tuple[str, ...], workdir: str) -> dict[str, Any]:
    """``family`` is injected first; the change to approve is ``family`` or the one in ``also``."""
    from qc.config import DatasetConfig
    from qc.notices import candidates, registry_draft
    from qc.run import run_qc
    from qcgen.config import dataset_calendar, suite_config
    from qcgen.scenarios import build_scenario
    from qcgen.sources import ScenarioSource

    tag = "+".join((family, *also))
    root = Path(workdir) / f"{profile}-{seed}-{tag}"
    try:
        built = build_scenario(suite_config(profile, seed=seed), 0, root, family, STAGES,
                               oracle_root=root / "_oracle", also=also)
        config = replace(DatasetConfig(), **dataset_calendar(profile), **PROFILES[profile])
        source = ScenarioSource(built.directory)
        current, previous = built.manifest["current_version"], built.manifest["previous_version"]
        blind = run_qc(source, current, previous, config)
        row: dict[str, Any] = {"profile": profile, "seed": seed, "family": family, "also": list(also),
                               "blind": blind.status, "blind_unexplained": _unexplained(blind),
                               "market": _market_leaf(built, blind)}
        change = next((name for name in (family, *also) if name in CHANGES), None)
        if change is None:
            return row
        row["change"] = change
        classification, entity_type = CHANGES[change]
        latest = blind.version_pair.current_max_week if blind.version_pair is not None else None
        drafts = [registry_draft(c, config.name, latest) for c in candidates(blind)
                  if c.classification == classification and c.entity_type == entity_type]
        registry = [{**d, "confirmed": True, "approved_by": "experiment", "approved_at": "2000-01-01T00:00:00+00:00"}
                    for d in drafts if d is not None]
        approved = run_qc(source, current, previous, config, registry=registry)
        row.update(drafted=len(registry), approved=approved.status, unexplained=_unexplained(approved),
                   like_for_like=approved.machine.get("like_for_like"), market=_market_leaf(built, approved))
        return row
    finally:
        shutil.rmtree(root, ignore_errors=True)


def run_job(job: tuple[str, int, str, tuple[str, ...], str]) -> dict[str, Any]:
    return _assess(*job)


def _rate(hits: int, n: int) -> dict[str, Any]:
    low, high = wilson_interval(hits, n) if n else (0.0, 1.0)
    return {"hits": hits, "n": n, "rate": hits / n if n else None, "ci": [low, high]}


def _market_misses(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Missed market movements, and how many were at least the reported 80%-power limit."""
    leaves = [r["market"] for r in rows if r.get("market") and r["market"]["detectable_change_80"] is not None]
    missed = [m for m in leaves if not m["flagged"]]
    return {"n": len(leaves), "missed": len(missed),
            "missed_at_or_above_limit_80": sum(m["drop"] >= m["detectable_change_80"] for m in missed),
            "missed_at_or_above_limit_50": sum(m["drop"] >= m["detectable_change"] for m in missed)}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for profile in PROFILES:
        mine = [r for r in rows if r["profile"] == profile]
        clean = {r["seed"]: set(r["blind_unexplained"]) for r in mine if r["family"] == "clean"}
        entry: dict[str, Any] = {"clean_flagged": _rate(sum(bool(v) for v in clean.values()), len(clean))}
        for change in CHANGES:
            alone = [r for r in mine if r["family"] == change and not r["also"]]
            residual = [(r["seed"], set(r["unexplained"])) for r in alone]
            entry[change] = {
                "blind_flagged": _rate(sum(r["blind"] not in PASSING for r in alone), len(alone)),
                "approved_explained": _rate(sum(r["approved"] == "PASS_WITH_EXPLANATION" for r in alone), len(alone)),
                "approved_flagged_only_clean_findings": sum(
                    bool(found) and found <= clean.get(seed, set()) for seed, found in residual),
                "residual_findings": sorted({u for _, found in residual for u in found}),
                "residual_change_findings": sum(
                    any(u.startswith(("historical_revision", "absence_event", "historical_event")) for u in found)
                    for _, found in residual),
            }
            for second in SECOND:
                base = {r["seed"]: r for r in mine if r["family"] == second and not r["also"]}
                paired = {r["seed"]: r for r in mine if r["family"] == second and r["also"] == [change]}
                seeds = sorted(set(base) & set(paired))
                cell: dict[str, Any] = {
                    "second_alone_flagged": _rate(sum(base[s]["blind"] not in PASSING for s in seeds), len(seeds)),
                    "approved_pair_flagged": _rate(sum(paired[s]["approved"] not in PASSING for s in seeds), len(seeds)),
                }
                if second == "market_movement":
                    cell["alone_misses"] = _market_misses([base[s] for s in seeds])
                    cell["approved_pair_misses"] = _market_misses([paired[s] for s in seeds])
                entry[f"{second}+approved {change}"] = cell
        out[profile] = entry
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", default="6101-6105", help="inclusive range")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--workdir", default="reports/approvals-work")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    first, last = (int(x) for x in args.seeds.split("-"))
    jobs: list[tuple[str, int, str, tuple[str, ...], str]] = []
    for profile in PROFILES:
        for seed in range(first, last + 1):
            jobs += [(profile, seed, change, (), args.workdir) for change in (*CHANGES, "clean")]
            jobs += [(profile, seed, second, also, args.workdir)
                     for second in SECOND for also in ((), *((change,) for change in CHANGES))]
    if args.jobs > 1:
        with ProcessPoolExecutor(args.jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
            rows = list(pool.map(run_job, jobs))
    else:
        rows = [run_job(job) for job in jobs]
    report = {"seeds": args.seeds, "summary": summarize(rows), "rows": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    for profile, entry in report["summary"].items():
        print(profile)
        for key, cell in entry.items():
            if "hits" in cell:
                cell = {"rate": cell}
            print("  ", key, {k: (f"{v['hits']}/{v['n']}" if isinstance(v, dict) and "hits" in v else v)
                              for k, v in cell.items()})


if __name__ == "__main__":
    main()
