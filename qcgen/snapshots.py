"""Snapshot storage.

Local development writes Parquet under a scenario directory; the Delta-backed
implementation is a later port (see docs/synthetic-data.md). Fingerprints are
computed from DataFrame values, not file bytes, so they are stable across
formats.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .oracle import json_default
from .stages import State


def frame_fingerprint(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps([str(c) for c in frame.columns]).encode())
    digest.update(json.dumps([str(t) for t in frame.dtypes]).encode())
    digest.update(str(len(frame)).encode())
    if len(frame):
        hashed = pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype=np.uint64)
        digest.update(hashed.tobytes())
    return digest.hexdigest()[:16]


class SnapshotStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def scenario_dir(self, scenario_id: str) -> Path:
        return self.root / scenario_id

    def version_dir(self, scenario_dir: str | Path, version: str) -> Path:
        return Path(scenario_dir) / "versions" / version

    def write_version(
        self,
        scenario_dir: str | Path,
        version: str,
        states: dict[str, State],
        stages: tuple[str, ...],
        dims: dict[str, pd.DataFrame],
    ) -> dict:
        version_dir = self.version_dir(scenario_dir, version)
        manifest: dict = {
            "version": version,
            "stages": list(stages),
            "fingerprints": {},
            "row_counts": {},
        }
        for stage in stages:
            stage_dir = version_dir / stage
            stage_dir.mkdir(parents=True, exist_ok=True)
            fact = states[stage]["fact"]
            fact.to_parquet(stage_dir / "fact.parquet", index=False)
            manifest["fingerprints"][stage] = frame_fingerprint(fact)
            manifest["row_counts"][stage] = int(len(fact))

        dims_dir = version_dir / "dims"
        dims_dir.mkdir(parents=True, exist_ok=True)
        for name, frame in dims.items():
            frame.to_parquet(dims_dir / f"{name}.parquet", index=False)

        (version_dir / "version_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=json_default)
        )
        return manifest

    def read_fact(
        self, scenario_dir: str | Path, version: str, stage: str
    ) -> pd.DataFrame:
        path = self.version_dir(scenario_dir, version) / stage / "fact.parquet"
        return pd.read_parquet(path)

    def available_stages(self, scenario_dir: str | Path, version: str) -> list[str]:
        manifest_path = self.version_dir(scenario_dir, version) / "version_manifest.json"
        if not manifest_path.exists():
            return []
        manifest = json.loads(manifest_path.read_text())
        return list(manifest.get("stages", []))

    def read_dims(self, scenario_dir: str | Path, version: str) -> dict[str, pd.DataFrame]:
        dims_dir = self.version_dir(scenario_dir, version) / "dims"
        return {
            path.stem: pd.read_parquet(path)
            for path in sorted(dims_dir.glob("*.parquet"))
        }
