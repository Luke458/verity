"""Fault-size sweep: detection curves and the clean false-alarm rate.

A single detection rate over default-sized faults says little: every detector
looks perfect on large enough faults. The sweep injects each family at a range
of explicit magnitudes (``qcgen.scenarios.MAGNITUDE_MEANING``) on the same
seeded worlds, so each family gets a curve from "invisible" to "obvious", and
measures false alarms on clean refreshes of the same profile. Everything is
scored on the engine's final status, as in ``qc cohort``.

Reuse of a seed across magnitudes is deliberate (a paired design): the
underlying world is identical and only the fault size changes.
"""

from __future__ import annotations

import json
import multiprocessing
import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from .cohort import ORACLE_CAUSE, code_sha256
from .config import DatasetConfig
from .conformal import wilson_interval
from .jsonutil import dumps as json_dumps

SWEEP_SCHEMA = 1
DEFAULT_FAMILIES: tuple[str, ...] = (
    "missing_stores",
    "missing_products",
    "history_truncation",
    "commodity_remap",
    "coding_error",
    "warehouse_transform_error",
    "recalculation",
    "market_movement",
)
DEFAULT_MAGNITUDES: tuple[float, ...] = (0.01, 0.02, 0.05, 0.1, 0.2, 0.4)
# Two faults in one refresh, at default sizes: one structural or value fault
# combined with a different kind, so a single cause label must pick one.
DEFAULT_PAIRS: tuple[tuple[str, str], ...] = (
    ("missing_stores", "market_movement"),
    ("coding_error", "market_movement"),
    ("missing_products", "recalculation"),
    ("commodity_remap", "warehouse_transform_error"),
    ("history_truncation", "coding_error"),
    ("missing_stores", "coding_error"),
)
STAGES = ("source", "coded", "warehouse", "report")
PASSING = ("PASS", "PASS_WITH_EXPLANATION")


@dataclass(frozen=True)
class SweepCase:
    family: str
    magnitude: float | None
    seed: int
    index: int
    status: str
    detected: bool
    relative_effect: float | None
    expected_cause: str | None
    predicted_cause: str | None
    also: tuple[str, ...] = ()
    expected_causes: tuple[str, ...] = ()


@dataclass(frozen=True)
class SweepJob:
    profile: str
    family: str
    magnitude: float | None
    seed: int
    index: int
    workdir: str
    overrides: dict[str, Any]
    also: tuple[str, ...] = ()


def _run_case(job: SweepJob) -> SweepCase:
    profile, family, magnitude, seed, index = (
        job.profile, job.family, job.magnitude, job.seed, job.index
    )
    from qcgen.config import dataset_calendar, suite_config
    from qcgen.scenarios import build_scenario
    from qcgen.sources import ScenarioSource

    from .run import run_qc

    tag = "+".join((family, *job.also)) + ("" if magnitude is None else f"-{magnitude:g}")
    root = Path(job.workdir) / profile / f"{seed}-{tag}-{index}"
    built = build_scenario(
        suite_config(profile, seed=seed),
        index,
        root,
        family,
        STAGES,
        oracle_root=root / "_oracle",
        magnitude=magnitude,
        also=job.also,
    )
    config = replace(DatasetConfig(), **dataset_calendar(profile), **job.overrides)
    result = run_qc(
        ScenarioSource(built.directory),
        built.manifest["current_version"],
        built.manifest["previous_version"],
        config,
    )
    case = built.oracle["cases"][0]
    causes = tuple(
        sorted(
            {
                ORACLE_CAUSE[str(item.get("expected_class"))]
                for item in built.oracle["cases"]
                if str(item.get("expected_class")) in ORACLE_CAUSE
            }
        )
    )
    cause = result.decisions.get("likely_cause") if result.decisions else None
    return SweepCase(
        family=family,
        magnitude=magnitude,
        seed=seed,
        index=index,
        status=result.status,
        detected=result.status not in PASSING,
        relative_effect=built.oracle.get("relative_effect"),
        expected_cause=ORACLE_CAUSE.get(str(case.get("expected_class"))),
        predicted_cause=str(cause.value) if cause is not None else None,
        also=tuple(job.also),
        expected_causes=causes,
    )


def _rate(hits: int, n: int) -> dict[str, Any]:
    low, high = wilson_interval(hits, n)
    return {
        "n": n,
        "hits": hits,
        "rate": hits / n if n else None,
        "ci_low": low if n else None,
        "ci_high": high if n else None,
    }


def summarize(cases: Sequence[SweepCase]) -> dict[str, Any]:
    faults = [case for case in cases if case.magnitude is not None and not case.also]
    controls = [case for case in cases if case.family == "clean"]
    pairs = [case for case in cases if case.also]
    curves: dict[str, list[dict[str, Any]]] = {}
    for family in sorted({case.family for case in faults}):
        points = []
        sizes = {case.magnitude for case in faults if case.family == family and case.magnitude is not None}
        for magnitude in sorted(sizes):
            cell = [c for c in faults if c.family == family and c.magnitude == magnitude]
            effects = [c.relative_effect for c in cell if c.relative_effect is not None]
            labelled = [c for c in cell if c.detected and c.expected_cause is not None]
            points.append(
                {
                    "magnitude": magnitude,
                    "detection": _rate(sum(c.detected for c in cell), len(cell)),
                    "median_relative_effect": (
                        sorted(effects)[len(effects) // 2] if effects else None
                    ),
                    # Cause accuracy among detected cases of this size.
                    "cause": _rate(
                        sum(c.predicted_cause == c.expected_cause for c in labelled),
                        len(labelled),
                    ),
                }
            )
        curves[family] = points
    false_alarms = sum(case.detected for case in controls)
    pair_summary: dict[str, dict[str, Any]] = {}
    for key in sorted({"+".join((case.family, *case.also)) for case in pairs}):
        cell = [case for case in pairs if "+".join((case.family, *case.also)) == key]
        detected = [case for case in cell if case.detected]
        predicted: dict[str, int] = {}
        for case in detected:
            predicted[str(case.predicted_cause)] = predicted.get(str(case.predicted_cause), 0) + 1
        pair_summary[key] = {
            "detection": _rate(len(detected), len(cell)),
            # A single label is right if it names either true cause.
            "cause_any_of": _rate(
                sum(case.predicted_cause in case.expected_causes for case in detected),
                len(detected),
            ),
            "expected_causes": sorted({c for case in cell for c in case.expected_causes}),
            "predicted": predicted,
        }
    return {
        "curves": curves,
        "pairs": pair_summary,
        "clean_false_alarms": _rate(false_alarms, len(controls)),
        "clean_status_counts": {
            status: sum(1 for c in controls if c.status == status)
            for status in sorted({c.status for c in controls})
        },
    }


def run_sweep(
    profile: str,
    seeds: Sequence[int],
    families: Sequence[str] = DEFAULT_FAMILIES,
    magnitudes: Sequence[float] = DEFAULT_MAGNITUDES,
    controls_per_seed: int = 4,
    workdir: str | Path = "reports/sweep/work",
    config_overrides: dict[str, Any] | None = None,
    jobs: int = 1,
    pairs: Sequence[tuple[str, str]] = (),
) -> dict[str, Any]:
    """Run family x magnitude x seed, fault pairs and clean controls."""
    overrides = dict(config_overrides or {})
    work = str(Path(workdir))
    plan: list[SweepJob] = []
    for seed in seeds:
        for family in families:
            for magnitude in magnitudes:
                plan.append(SweepJob(profile, family, float(magnitude), int(seed), 0, work, overrides))
        for primary, secondary in pairs:
            plan.append(SweepJob(profile, primary, None, int(seed), 0, work, overrides, (secondary,)))
        for index in range(controls_per_seed):
            plan.append(SweepJob(profile, "clean", None, int(seed), index, work, overrides))
    if jobs > 1:
        # One BLAS/OpenMP thread per worker: N processes each spawning a full
        # thread pool oversubscribe the CPU and run slower than one process.
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(name, "1")
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=jobs, mp_context=context) as pool:
            cases = list(pool.map(_run_case, plan, chunksize=4))
    else:
        cases = [_run_case(job) for job in plan]
    return {
        "schema_version": SWEEP_SCHEMA,
        "profile": profile,
        "seeds": list(seeds),
        "families": list(families),
        "magnitudes": [float(value) for value in magnitudes],
        "controls_per_seed": controls_per_seed,
        "pairs": ["+".join(pair) for pair in pairs],
        "config_overrides": overrides,
        "code_sha256": code_sha256(),
        "summary": summarize(cases),
        "cases": [asdict(case) for case in cases],
    }


def write_sweep(report: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json_dumps(report, indent=2, sort_keys=True))


def render_table(report: dict[str, Any]) -> str:
    """Markdown table: detection rate per family and magnitude."""
    summary = report["summary"]
    magnitudes = report["magnitudes"]
    lines = [
        "| family | " + " | ".join(f"{m:g}" for m in magnitudes) + " |",
        "|---|" + "---|" * len(magnitudes),
    ]
    for family, points in summary["curves"].items():
        cells = {point["magnitude"]: point["detection"] for point in points}
        row = []
        for magnitude in magnitudes:
            cell = cells.get(magnitude)
            row.append(f"{cell['hits']}/{cell['n']}" if cell else "")
        lines.append(f"| {family} | " + " | ".join(row) + " |")
    if summary.get("pairs"):
        lines.append("")
        lines.append("| fault pair | detected | cause names either fault |")
        lines.append("|---|---|---|")
        for key, entry in summary["pairs"].items():
            detection, cause = entry["detection"], entry["cause_any_of"]
            lines.append(
                f"| {key} | {detection['hits']}/{detection['n']} | {cause['hits']}/{cause['n']} |"
            )
    alarms = summary["clean_false_alarms"]
    lines.append("")
    lines.append(
        f"Clean false alarms: {alarms['hits']}/{alarms['n']}"
        + (
            f" (Wilson 95% upper {alarms['ci_high']:.3f})"
            if alarms["n"]
            else ""
        )
    )
    return "\n".join(lines)


def load_sweep(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())
