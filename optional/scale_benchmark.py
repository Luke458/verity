"""Repeatable local Parquet CPU measurement; no distributed-scale claim."""

import argparse
import json
import resource
import tempfile
import time
from pathlib import Path

import pandas as pd

from qc.config import DatasetConfig
from qc.run import run_qc
from qc.source import CachedSource


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stores", type=int, default=500)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for version, weeks in (("v1", 51), ("v2", 52)):
            frame = pd.DataFrame(
                [
                    {
                        "week": w,
                        "store_id": f"s{s}",
                        "product_id": "p",
                        "dollar": 10.0,
                        "units": 1.0,
                    }
                    for w in range(1, weeks + 1)
                    for s in range(args.stores)
                ]
            )
            frame.to_parquet(root / f"{version}-warehouse.parquet")
            frame.groupby("week", as_index=False)[["dollar", "units"]].sum().to_parquet(
                root / f"{version}-report.parquet"
            )
        config = DatasetConfig(report_grain=(), temporal_enabled=False)
        source = CachedSource(ParquetSource(root), config)
        started = time.perf_counter()
        result = run_qc(source, "v2", "v1", config)
        print(
            json.dumps(
                {
                    "current_rows": 52 * args.stores,
                    "previous_rows": 51 * args.stores,
                    "runtime_seconds": time.perf_counter() - started,
                    "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    "fact_reads": source.read_count,
                    "status": result.status,
                    "temporal_enabled": False,
                    "provider": "rule",
                }
            )
        )


if __name__ == "__main__":
    main()
