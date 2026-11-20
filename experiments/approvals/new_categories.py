"""Does an approved move into a new category explain it, as a move into an existing one does?

    python -m experiments.approvals.new_categories --seeds 6601-6620 --jobs 4 --out reports/new-categories.json

For each profile and seed, a category move (``commodity_remap``) is built
three ways: into an existing category (the generator's own draw, the case
``experiments.approvals.run`` measured), into a category the previous version
never had (``C99``), and with every product of its category moved there (a
*rename*: the source category is emptied). Each is assessed blind and again
with the one ``POSSIBLE_RECLASSIFICATION`` entry ``qc notices`` drafts for it,
approved.

The move into a new category is also paired with a second fault (coding
error, warehouse transform error, market movement, one-week restatement),
approved, and each second fault is run alone on the same seed, as in
``experiments.approvals.run``. A missed market movement is judged against the
engine's reported 80%-power limit in the run that missed it. A clean refresh
per seed gives the world's background flags. Pre-registered in
docs/evaluation.md ("Approving moves into a new category").
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

from experiments.approvals.run import (
    PASSING,
    PROFILES,
    SECOND,
    STAGES,
    _market_leaf,
    _market_misses,
    _rate,
    _unexplained,
)

NEW = "C99"
# Arm name -> commodity_remap overrides (None: the generator's own draw).
ARMS: dict[str, dict[str, Any] | None] = {
    "existing": None,
    "new": {"to_commodity": NEW},
    "rename": {"to_commodity": NEW, "n_products": 100_000},
}
CHANGE_FINDINGS = ("historical_revision", "historical_event", "revision_week")


def _assess(profile: str, seed: int, family: str, also: tuple[str, ...], arm: str | None, workdir: str) -> dict[str, Any]:
    from qc.config import DatasetConfig
    from qc.lifecycle import RECLASSIFIED
    from qc.notices import candidates, registry_draft
    from qc.run import run_qc
    from qcgen.config import dataset_calendar, suite_config
    from qcgen.scenarios import build_scenario
    from qcgen.sources import ScenarioSource

    tag = "+".join((family, *also, *([arm] if arm else [])))
    root = Path(workdir) / f"{profile}-{seed}-{tag}"
    overrides = ARMS[arm] if arm else None
    params = {"commodity_remap": overrides} if overrides else None
    try:
        built = build_scenario(suite_config(profile, seed=seed), 0, root, family, STAGES,
                               oracle_root=root / "_oracle", also=also, params=params)
        config = replace(DatasetConfig(), **dataset_calendar(profile), **PROFILES[profile])
        source = ScenarioSource(built.directory)
        current, previous = built.manifest["current_version"], built.manifest["previous_version"]
        blind = run_qc(source, current, previous, config)
        row: dict[str, Any] = {"profile": profile, "seed": seed, "family": family, "also": list(also), "arm": arm,
                               "blind": blind.status, "blind_unexplained": _unexplained(blind),
                               "market": _market_leaf(built, blind)}
        case = next((c for c in built.oracle["cases"] if c.get("family") == "commodity_remap"), None)
        if case is None:
            return row
        move = f"{case['details']['from_commodity']}->{case['details']['to_commodity']}"
        latest = blind.version_pair.current_max_week if blind.version_pair is not None else None
        drafts = [registry_draft(c, config.name, latest) for c in candidates(blind)
                  if c.classification == RECLASSIFIED and c.entity_type == "commodity"]
        registry = [{**d, "confirmed": True, "approved_by": "experiment", "approved_at": "2000-01-01T00:00:00+00:00"}
                    for d in drafts if d is not None]
        approved = run_qc(source, current, previous, config, registry=registry)
        row.update(
            move=move, moved_products=len(case["affected"]["products"]),
            blind_events=sorted({f"{e.classification}:{e.entity_type}:{e.entity_id}" for e in blind.events}),
            drafted=[d["entity_ids"] for d in registry],
            approved=approved.status, unexplained=_unexplained(approved),
            consequences=dict(approved.attribution.move_consequences) if approved.attribution else {},
            like_for_like=approved.machine.get("like_for_like"), market=_market_leaf(built, approved),
        )
        return row
    finally:
        shutil.rmtree(root, ignore_errors=True)


def run_job(job: tuple[str, int, str, tuple[str, ...], str | None, str]) -> dict[str, Any]:
    return _assess(*job)


def _beyond_clean(row: dict[str, Any], clean: dict[int, set[str]]) -> list[str]:
    return sorted(set(row["unexplained"]) - clean.get(row["seed"], set()))


def _own_findings(found: list[str]) -> list[str]:
    """Change findings, and any finding on the new category."""
    return [item for item in found if item.startswith(CHANGE_FINDINGS) or f"commodity_id:{NEW}|" in item]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for profile in PROFILES:
        mine = [r for r in rows if r["profile"] == profile]
        clean = {r["seed"]: set(r["blind_unexplained"]) for r in mine if r["family"] == "clean"}
        entry: dict[str, Any] = {"clean_flagged": _rate(sum(bool(v) for v in clean.values()), len(clean))}
        for arm in ARMS:
            alone = [r for r in mine if r["family"] == "commodity_remap" and not r["also"] and r["arm"] == arm]
            entry[arm] = {
                "blind_flagged": _rate(sum(r["blind"] not in PASSING for r in alone), len(alone)),
                "blind_revision_weeks": sorted(sum(u.startswith("revision_week") for u in r["blind_unexplained"])
                                               for r in alone),
                "drafted_one_move": sum(r["drafted"] == [[r["move"]]] for r in alone),
                "new_category_credited": sum(f"commodity:{NEW}" in r["consequences"] for r in alone),
                "emptied_category_credited": sum(
                    f"commodity:{r['move'].split('->')[0]}" in r["consequences"] for r in alone),
                "approved_explained": _rate(sum(r["approved"] == "PASS_WITH_EXPLANATION" for r in alone), len(alone)),
                "approved_findings_beyond_clean": sum(bool(_beyond_clean(r, clean)) for r in alone),
                "approved_own_findings_beyond_clean": sum(bool(_own_findings(_beyond_clean(r, clean))) for r in alone),
                "approved_flagged_only_clean_findings": sum(
                    bool(r["unexplained"]) and not _beyond_clean(r, clean) for r in alone),
                "residual_findings": sorted({u for r in alone for u in r["unexplained"]}),
            }
        for second in SECOND:
            base = {r["seed"]: r for r in mine if r["family"] == second and not r["also"]}
            paired = {r["seed"]: r for r in mine if r["family"] == second and r["also"] == ["commodity_remap"]}
            seeds = sorted(set(base) & set(paired))
            cell: dict[str, Any] = {
                "second_alone_flagged": _rate(sum(base[s]["blind"] not in PASSING for s in seeds), len(seeds)),
                "approved_pair_flagged": _rate(sum(paired[s]["approved"] not in PASSING for s in seeds), len(seeds)),
                "approved_pair_credited": sum(f"commodity:{NEW}" in paired[s].get("consequences", {}) for s in seeds),
            }
            if second == "market_movement":
                cell["alone_misses"] = _market_misses([base[s] for s in seeds])
                cell["approved_pair_misses"] = _market_misses([paired[s] for s in seeds])
            entry[f"{second}+approved new-category move"] = cell
        out[profile] = entry
    return out


def plan(first: int, last: int, workdir: str) -> list[tuple[str, int, str, tuple[str, ...], str | None, str]]:
    jobs: list[tuple[str, int, str, tuple[str, ...], str | None, str]] = []
    for profile in PROFILES:
        for seed in range(first, last + 1):
            jobs += [(profile, seed, "commodity_remap", (), arm, workdir) for arm in ARMS]
            jobs.append((profile, seed, "clean", (), None, workdir))
            for second in SECOND:
                jobs.append((profile, seed, second, (), None, workdir))
                jobs.append((profile, seed, second, ("commodity_remap",), "new", workdir))
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", default="6401-6405", help="inclusive range")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--workdir", default="reports/new-categories-work")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    first, last = (int(x) for x in args.seeds.split("-"))
    jobs = plan(first, last, args.workdir)
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
