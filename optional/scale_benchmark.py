"""Reproducible local Parquet CPU scale measurement; no distributed-scale claim.

Generates 320-week version pairs at a requested row count with numpy, runs the
core engine (optionally with temporal processing) and records runtime, peak RSS
and snapshot-read counts. A run that cannot reach the requested rows or that
fails is reported as an explicit limitation rather than a scale qualification.
"""

import argparse
import json
import resource
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from qc.config import DatasetConfig
from qc.run import run_qc
from qc.source import CachedSource

PROFILES = {
    "one-million": {"rows": 1_000_000, "weeks": 320},
    "five-million": {"rows": 5_000_000, "weeks": 320},
}


class ParquetSource:
    def __init__(self, root):
        self.root = root

    def list_versions(self):
        return ["v1", "v2"]

    def available_stages(self, version):
        return ["warehouse", "report"]

    def read_fact(self, version, stage):
        return pd.read_parquet(self.root / f"{version}-{stage}.parquet")

    def read_dim(self, version, name):
        return None


def build_frame(weeks: int, rows: int) -> pd.DataFrame:
    rows = max(int(rows), weeks)
    weeks_array = np.arange(1, weeks + 1, dtype=np.int64)
    stores_needed = int(np.ceil(rows / weeks))
    store_index = np.repeat(np.arange(stores_needed, dtype=np.int64), weeks)[:rows]
    week = np.tile(weeks_array, stores_needed)[:rows]
    base = 100.0 + 5.0 * np.sin(2.0 * np.pi * week / 52.0) + 0.01 * week
    dollar = base + (store_index % 7) * 0.5
    return pd.DataFrame(
        {
            "week": week,
            "store_id": np.char.add("s", store_index.astype(str)),
            "product_id": "p",
            "dollar": dollar,
            "units": dollar / 10.0,
        }
    )


def run_benchmark(
    *,
    rows: int,
    weeks: int,
    temporal: bool = True,
    profile: str = "",
) -> dict:
    started = time.perf_counter()
    result_status = ""
    source = None
    error = ""
    result = None
    config = DatasetConfig(report_grain=(), temporal_enabled=temporal)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        current_rows = previous_rows = 0
        for version, version_weeks in (("v1", weeks - 1), ("v2", weeks)):
            frame = build_frame(version_weeks, rows)
            if version == "v2":
                current_rows = len(frame)
            else:
                previous_rows = len(frame)
            frame.to_parquet(root / f"{version}-warehouse.parquet")
            frame.groupby("week", as_index=False)[["dollar", "units"]].sum().to_parquet(
                root / f"{version}-report.parquet"
            )
        source = CachedSource(ParquetSource(root), config)
        try:
            result = run_qc(source, "v2", "v1", config)
            result_status = result.status
        except Exception as exc:  # noqa: BLE001 - recorded as a limitation
            error = f"{type(exc).__name__}: {exc}"
    runtime = time.perf_counter() - started
    requested = int(rows)
    complete = not error and current_rows >= requested
    temporal_result = getattr(result, "temporal", None)
    machine = getattr(result, "machine", {}) or {}
    metric_status = dict(getattr(temporal_result, "metric_status", {}) or {})
    findings = machine.get("findings", [])
    payload = {
        "profile": profile or f"{requested}-rows",
        "requested_rows": requested,
        "previous_rows": previous_rows,
        "current_rows": current_rows,
        "weeks": weeks,
        "runtime_seconds": runtime,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "snapshot_reads": int(getattr(source, "read_count", 0)),
        "status": result_status,
        "temporal_enabled": temporal,
        "provider": "rule",
        "assessed_metrics": sorted(
            metric for metric, state in metric_status.items() if state == "ASSESSED"
        ),
        "absent_metrics": sorted(
            metric for metric, state in metric_status.items() if state == "ABSENT"
        ),
        "required_metrics": list(config.required_temporal_metrics()),
        "snapshot_metrics": list(config.snapshot_metrics),
        "periods_assessed": (
            len(getattr(temporal_result, "targets", ()) or ())
            if temporal_result is not None
            else 0
        ),
        "hierarchy_checks": len(getattr(result, "hierarchy", ()) or ()),
        "finding_outcomes": {
            outcome: sum(1 for item in findings if item.get("outcome") == outcome)
            for outcome in ("PASS", "FAIL", "UNAVAILABLE", "CONTRACT_FAILURE")
        },
        "complete": complete,
    }
    if error:
        payload["error"] = error
        payload["limitations"] = [
            "run failed; not a scale measurement",
            error,
        ]
    elif not complete:
        payload["limitations"] = [
            "requested row count not reached; not a scale qualification"
        ]
    else:
        payload["limitations"] = [
            "single-machine Parquet CPU measurement; no distributed-scale claim",
        ]
    return payload


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=sorted(PROFILES), default=None)
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--weeks", type=int, default=320)
    parser.add_argument("--stores", type=int, default=None)
    parser.add_argument("--no-temporal", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    profile = args.profile or ""
    rows = args.rows
    weeks = args.weeks
    if profile:
        rows = PROFILES[profile]["rows"]
        weeks = PROFILES[profile]["weeks"]
    if rows is None:
        if args.stores is not None:
            rows = args.stores * weeks
        else:
            rows = PROFILES["one-million"]["rows"]
    payload = run_benchmark(
        rows=rows,
        weeks=weeks,
        temporal=not args.no_temporal,
        profile=profile,
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text + "\n")
    print(text)
    return payload


if __name__ == "__main__":
    main()
