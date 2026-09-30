"""Content identity and recoverable publication for local shadow assessments."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

from .jsonutil import dumps


def digest(value) -> str:
    return hashlib.sha256(
        dumps(value, sort_keys=True, default=str).encode()
    ).hexdigest()


def frame_digest(frame) -> str | None:
    if frame is None:
        return None
    import pandas as pd

    h = hashlib.sha256()
    h.update(
        dumps([(str(c), str(t)) for c, t in zip(frame.columns, frame.dtypes, strict=False)]).encode()
    )
    h.update(pd.util.hash_pandas_object(frame, index=False).values.tobytes())
    return h.hexdigest()


def snapshot_manifest(source, previous, current, config, identity: str) -> dict:
    snapshots = []
    for version in (previous, current):
        facts = {
            stage: frame_digest(source.read_fact(version, stage))
            for stage in source.available_stages(version)
        }
        dims = {
            name: frame_digest(source.read_dim(version, name))
            for name in sorted(set(("products", "stores", *config.required_dimensions)))
        }
        snapshots.append(
            {
                "version": version,
                "facts": facts,
                "dimensions": dims,
                "metadata": source.snapshot_metadata(version)
                if hasattr(source, "snapshot_metadata")
                else {},
            }
        )
    return {
        "schema_version": 1,
        "source": identity,
        "snapshots": snapshots,
        "config_hash": digest(config.to_dict()),
        "engine_version": "0.22.0",
    }


def artifact_checksums(artifacts: dict[str, str]) -> dict[str, str]:
    return {
        name: hashlib.sha256(text.encode()).hexdigest()
        for name, text in artifacts.items()
    }


def publish(
    directory: Path, artifacts: dict[str, str], checksums: dict[str, str]
) -> None:
    """Regenerate from journal, then rename; never claim a DB/filesystem transaction."""
    if any(not name or name in (".", "..") or Path(name).name != name for name in artifacts):
        raise ValueError("unsafe artifact name")
    if artifact_checksums(artifacts) != checksums:
        raise ValueError("journal artifact checksum mismatch")
    if directory.exists() and all(
        (directory / name).is_file()
        and hashlib.sha256((directory / name).read_bytes()).hexdigest() == checksum
        for name, checksum in checksums.items()
    ):
        return
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".publish-", dir=directory.parent))
    for name, content in artifacts.items():
        if Path(name).name != name:
            raise ValueError("unsafe artifact name")
        with (temporary / name).open("w") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    if directory.exists():
        # Preserve damaged output for inspection; don't overwrite recovery evidence.
        os.replace(
            directory,
            directory.with_name(directory.name + ".damaged-" + temporary.name),
        )
    os.replace(temporary, directory)
    fd = os.open(directory.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
