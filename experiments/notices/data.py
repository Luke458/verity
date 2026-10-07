"""Notice-matching dataset: generated refreshes, their observed changes and notices.

Each row is one refresh run through the full engine: its candidates (observed
lifecycle changes, ``notices.candidates``) and its notices with their oracle
answers. Splits reuse the labeller's disjoint seed ranges. No model is trained
here; dev is for iteration and threshold tuning, test is scored once.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .notices import CHANGE_TYPES, candidates, notices_for, world_from_dims

SPLITS = {"dev": range(8101, 8111), "test": range(8201, 8221)}
# Faults with a notice template, two without one, and clean refreshes.
FAMILIES = (*CHANGE_TYPES, "coding_error", "market_movement", "clean", "clean")
DATA_VERSION = 1


@dataclass(frozen=True)
class Job:
    split: str
    seed: int
    index: int
    profile: str
    family: str
    workdir: str


def plan(split: str, workdir: str) -> list[Job]:
    return [
        Job(split, seed, index, "small" if seed % 2 else "realistic", family, workdir)
        for seed in SPLITS[split]
        for index, family in enumerate(FAMILIES)
    ]


def build_row(job: Job) -> dict[str, Any]:
    import pandas as pd

    from experiments.labeller.data import PROFILE_CONFIG
    from qc.cohort import code_sha256
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
        )
        config = replace(DatasetConfig(), **dataset_calendar(job.profile), **PROFILE_CONFIG[job.profile])
        current = built.manifest["current_version"]
        result = run_qc(ScenarioSource(built.directory), current, built.manifest["previous_version"], config)
        dims_dir = Path(built.directory) / "versions" / current / "dims"
        dims = {name: pd.read_parquet(dims_dir / f"{name}.parquet") for name in ("stores", "products", "commodities")}
        latest = result.version_pair.current_max_week if result.version_pair is not None else 0
        world = world_from_dims(dims, config.name, latest)
        cands = candidates(result, world)
        rng = np.random.default_rng([job.seed, job.index, 7])
        notes = notices_for(built.oracle["cases"][0], world, cands, rng)
        return {
            "split": job.split,
            "seed": job.seed,
            "index": job.index,
            "profile": job.profile,
            "family": job.family,
            "status": result.status,
            "dataset": config.name,
            "latest_week": latest,
            "commodities": world.commodities,
            "candidates": [asdict(c) for c in cands],
            "notices": [n.to_dict() for n in notes],
            "data_version": DATA_VERSION,
            "code_sha256": code_sha256(),
        }
    finally:
        shutil.rmtree(root, ignore_errors=True)


def build(split: str, out: str | Path, workdir: str | Path, jobs: int = 1) -> Path:
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    planned = plan(split, str(work))
    if jobs > 1:
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(name, "1")
        with ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
            rows = list(pool.map(build_row, planned, chunksize=2))
    else:
        rows = [build_row(job) for job in planned]
    target.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    return target


def load(path: str | Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if {row["data_version"] for row in rows} != {DATA_VERSION}:
        raise ValueError(f"{path}: data version differs from {DATA_VERSION}")
    return rows
