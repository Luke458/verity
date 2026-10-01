"""Labelled dataset of engine outputs for cause labelling.

Each row is one generated refresh run through the full engine: its feature
vector (``features.extract``), the rule labeller's cause, and the oracle's set
of true causes (empty for a clean refresh, two for a fault pair). Splits use
disjoint seed ranges, and the realistic profile always declares its 2-week
late-arrival window, as an operator of that feed would.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .features import FEATURE_VERSION, extract

# Families whose ground-truth cause is defined without a registry.
SINGLE_FAMILIES = (
    "missing_stores",
    "missing_products",
    "entity_merge",
    "new_store_backfill",
    "history_truncation",
    "commodity_remap",
    "coding_error",
    "warehouse_transform_error",
    "recalculation",
    "week_restatement",
    "market_movement",
    "schema_failure",
    "null_duplicate_storm",
)
# Contract failures stop the analysis, so they never appear inside a pair.
PAIR_FAMILIES = tuple(
    family for family in SINGLE_FAMILIES if family not in ("schema_failure", "null_duplicate_storm")
)
SPLITS = {"train": range(8001, 8041), "dev": range(8101, 8111), "test": range(8201, 8221)}
PAIRS_PER_SEED = 4
CLEAN_PER_SEED = 2
PROFILE_CONFIG: dict[str, dict[str, Any]] = {"small": {}, "realistic": {"restatement_weeks": 2}}


@dataclass(frozen=True)
class Job:
    split: str
    seed: int
    index: int
    profile: str
    family: str
    also: tuple[str, ...]
    magnitude: float | None
    workdir: str


def _sizable(family: str) -> bool:
    from qcgen.scenarios import MAGNITUDE_MEANING

    return family in MAGNITUDE_MEANING


def plan(split: str, workdir: str) -> list[Job]:
    jobs: list[Job] = []
    for seed in SPLITS[split]:
        rng = np.random.default_rng(seed)
        profile = "small" if seed % 2 else "realistic"
        index = 0

        def size(family: str, rng: np.random.Generator = rng) -> float | None:
            # Log-uniform 1%-40% so small, hard faults are well represented.
            return float(np.exp(rng.uniform(np.log(0.01), np.log(0.4)))) if _sizable(family) else None

        for family in SINGLE_FAMILIES:
            jobs.append(Job(split, seed, index, profile, family, (), size(family), workdir))
            index += 1
        for _ in range(PAIRS_PER_SEED):
            first, second = rng.choice(len(PAIR_FAMILIES), size=2, replace=False)
            primary = PAIR_FAMILIES[int(first)]
            jobs.append(
                Job(split, seed, index, profile, primary, (PAIR_FAMILIES[int(second)],), size(primary), workdir)
            )
            index += 1
        for _ in range(CLEAN_PER_SEED):
            jobs.append(Job(split, seed, index, profile, "clean", (), None, workdir))
            index += 1
    return jobs


def build_row(job: Job) -> dict[str, Any]:
    from qc.cohort import ORACLE_CAUSE, code_sha256
    from qc.config import DatasetConfig
    from qc.run import run_qc
    from qcgen.config import dataset_calendar, suite_config
    from qcgen.scenarios import build_scenario
    from qcgen.sources import ScenarioSource

    root = Path(job.workdir) / f"{job.split}-{job.seed}-{job.index}"
    try:
        built = build_scenario(
            suite_config(job.profile, seed=job.seed),
            job.index,
            root,
            job.family,
            ("source", "coded", "warehouse", "report"),
            oracle_root=root / "_oracle",
            magnitude=job.magnitude,
            also=job.also,
        )
        config = replace(
            DatasetConfig(), **dataset_calendar(job.profile), **PROFILE_CONFIG[job.profile]
        )
        result = run_qc(
            ScenarioSource(built.directory),
            built.manifest["current_version"],
            built.manifest["previous_version"],
            config,
        )
        causes = sorted(
            {
                ORACLE_CAUSE[str(case.get("expected_class"))]
                for case in built.oracle["cases"]
                if str(case.get("expected_class")) in ORACLE_CAUSE
            }
        )
        rule = result.decisions.get("likely_causes") if result.decisions else None
        return {
            "split": job.split,
            "seed": job.seed,
            "index": job.index,
            "profile": job.profile,
            "families": [job.family, *job.also],
            "magnitude": job.magnitude,
            "kind": "clean" if job.family == "clean" else ("pair" if job.also else "single"),
            "status": result.status,
            "causes": causes,
            "rule_causes": sorted(rule.value) if rule is not None else [],
            "features": extract(result, config),
            "feature_version": FEATURE_VERSION,
            "code_sha256": code_sha256(),
        }
    finally:
        shutil.rmtree(root, ignore_errors=True)


def build(split: str, out: str | Path, workdir: str | Path, jobs: int = 1) -> Path:
    """Generate one split as JSONL (one row per refresh)."""
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    planned = plan(split, str(work))
    if jobs > 1:
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(name, "1")
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=jobs, mp_context=context) as pool:
            rows = list(pool.map(build_row, planned, chunksize=2))
    else:
        rows = [build_row(job) for job in planned]
    target.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    return target


def load(path: str | Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    versions = {row["feature_version"] for row in rows}
    if versions != {FEATURE_VERSION}:
        raise ValueError(f"dataset feature version {versions} != {FEATURE_VERSION}")
    return rows
