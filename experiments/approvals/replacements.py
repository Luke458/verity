"""Does an approved store replacement explain exactly a value-conserving merge?

    python -m experiments.approvals.replacements --seeds 6501-6520 --jobs 4 --out reports/replacements.json

For each profile and seed, a store merge (``entity_merge``: one store's history
moves to a new id) is built three ways: history carried across unchanged
(factor 1.0), rescaled by 5% (factor 0.95), and with the generator's own draw
(0.9-1.1). Each is assessed blind and again with the one ``ENTITY_REPLACED``
entry ``qc notices`` drafts for it, approved. A conserving merge should then
be explained; a rescaled one should stay flagged whenever the value it changed
is material (above ``materiality_ratio`` of the overlap total).

The conserving merge is also paired with a second fault (coding error,
warehouse transform error, market movement, one-week restatement), approved,
and each second fault is run alone on the same seed, as in
``experiments.approvals.run``. The merge is injected at the source stage, so a
market movement's category (drawn at injection, warehouse stage) can differ
between the alone and paired runs; a missed movement is judged against the
engine's reported 80%-power limit in the run that missed it. A clean refresh
per seed gives the world's background flags. Pre-registered in
docs/evaluation.md ("Approving store replacements").
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

# Arm name -> forced merge factor (None: the generator's own draw).
ARMS: dict[str, float | None] = {"conserved": 1.0, "rescaled": 0.95, "drawn": None}
CHANGE_FINDINGS = ("historical_revision", "historical_event", "revision_week")


def _assess(profile: str, seed: int, family: str, also: tuple[str, ...], arm: str | None, workdir: str) -> dict[str, Any]:
    from qc.config import DatasetConfig
    from qc.notices import REPLACED, candidates, registry_draft, replacement_pairs
    from qc.run import run_qc
    from qcgen.config import dataset_calendar, suite_config
    from qcgen.scenarios import build_scenario
    from qcgen.sources import ScenarioSource

    tag = "+".join((family, *also, *([arm] if arm else [])))
    root = Path(workdir) / f"{profile}-{seed}-{tag}"
    factor = ARMS[arm] if arm else None
    params = {"entity_merge": {"factor": factor}} if factor is not None else None
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
        merge = next((c for c in built.oracle["cases"] if c.get("family") == "entity_merge"), None)
        if merge is None:
            return row
        pair = f"{merge['details']['source_store']}->{merge['details']['target_store']}"
        removed = next((e for e in blind.events if e.classification == "ENTITY_REMOVED"
                        and e.entity_id == merge["details"]["source_store"]), None)
        history = removed.historical_value_removed if removed is not None else None
        total = blind.attribution.previous_total if blind.attribution is not None else None
        latest = blind.version_pair.current_max_week if blind.version_pair is not None else None
        drafts = [registry_draft(p, config.name, latest)
                  for c in candidates(blind) if c.classification == REPLACED for p in replacement_pairs(blind, c)]
        registry = [{**d, "confirmed": True, "approved_by": "experiment", "approved_at": "2000-01-01T00:00:00+00:00"}
                    for d in drafts if d is not None]
        approved = run_qc(source, current, previous, config, registry=registry)
        factor_used = float(merge["details"]["factor"])
        row.update(
            pair=pair, factor=factor_used, drafted=[d["entity_ids"] for d in registry],
            # The value the rescale changed, as a share of the overlap total.
            rescale_share=(abs(1.0 - factor_used) * history / total) if history and total else None,
            approved=approved.status, unexplained=_unexplained(approved),
            historical_reason=(approved.machine.get("historical_revision") or {}).get("reason"),
            matched=dict(approved.attribution.matched_replacements) if approved.attribution else {},
            conservation=dict(approved.attribution.conservation) if approved.attribution else {},
            like_for_like=approved.machine.get("like_for_like"), market=_market_leaf(built, approved),
        )
        return row
    finally:
        shutil.rmtree(root, ignore_errors=True)


def run_job(job: tuple[str, int, str, tuple[str, ...], str | None, str]) -> dict[str, Any]:
    return _assess(*job)


def _change_findings(found: list[str]) -> bool:
    return any(item.startswith(CHANGE_FINDINGS) for item in found)


def summarize(rows: list[dict[str, Any]], materiality: float = 0.001) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for profile in PROFILES:
        mine = [r for r in rows if r["profile"] == profile]
        clean = {r["seed"]: set(r["blind_unexplained"]) for r in mine if r["family"] == "clean"}
        entry: dict[str, Any] = {"clean_flagged": _rate(sum(bool(v) for v in clean.values()), len(clean))}
        for arm in ARMS:
            alone = [r for r in mine if r["family"] == "entity_merge" and not r["also"] and r["arm"] == arm]
            material = [r for r in alone if r["rescale_share"] is not None and r["rescale_share"] > materiality]
            entry[arm] = {
                "blind_flagged": _rate(sum(r["blind"] not in PASSING for r in alone), len(alone)),
                "drafted_one_pair": sum(r["drafted"] == [[r["pair"]]] for r in alone),
                "pair_covered": sum(f"store:{r['pair']}" in r["matched"] for r in alone),
                "approved_explained": _rate(sum(r["approved"] == "PASS_WITH_EXPLANATION" for r in alone), len(alone)),
                "approved_change_findings": sum(_change_findings(r["unexplained"]) for r in alone),
                "approved_change_findings_beyond_clean": sum(
                    _change_findings(sorted(set(r["unexplained"]) - clean.get(r["seed"], set()))) for r in alone),
                "approved_flagged_only_clean_findings": sum(
                    bool(r["unexplained"]) and set(r["unexplained"]) <= clean.get(r["seed"], set()) for r in alone),
                "material_rescale": len(material),
                "material_rescale_change_flagged": sum(_change_findings(r["unexplained"]) for r in material),
                "immaterial_rescale_change_flagged": sum(
                    _change_findings(r["unexplained"]) for r in alone if r not in material),
                "residual_findings": sorted({u for r in alone for u in r["unexplained"]}),
            }
        for second in SECOND:
            base = {r["seed"]: r for r in mine if r["family"] == second and not r["also"]}
            paired = {r["seed"]: r for r in mine if r["family"] == second and r["also"] == ["entity_merge"]}
            seeds = sorted(set(base) & set(paired))
            cell: dict[str, Any] = {
                "second_alone_flagged": _rate(sum(base[s]["blind"] not in PASSING for s in seeds), len(seeds)),
                "approved_pair_flagged": _rate(sum(paired[s]["approved"] not in PASSING for s in seeds), len(seeds)),
                "approved_pair_covered": sum(f"store:{paired[s]['pair']}" in paired[s]["matched"] for s in seeds),
            }
            if second == "market_movement":
                cell["alone_misses"] = _market_misses([base[s] for s in seeds])
                cell["approved_pair_misses"] = _market_misses([paired[s] for s in seeds])
            entry[f"{second}+approved merge"] = cell
        out[profile] = entry
    return out


def plan(first: int, last: int, workdir: str) -> list[tuple[str, int, str, tuple[str, ...], str | None, str]]:
    jobs: list[tuple[str, int, str, tuple[str, ...], str | None, str]] = []
    for profile in PROFILES:
        for seed in range(first, last + 1):
            jobs += [(profile, seed, "entity_merge", (), arm, workdir) for arm in ARMS]
            jobs.append((profile, seed, "clean", (), None, workdir))
            for second in SECOND:
                jobs.append((profile, seed, second, (), None, workdir))
                jobs.append((profile, seed, second, ("entity_merge",), "conserved", workdir))
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", default="6401-6405", help="inclusive range")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--workdir", default="reports/replacements-work")
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
