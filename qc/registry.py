"""Expected-event registry stores.

The registry is an operator-maintained input, not part of the data under test.
``run_qc`` accepts a ``RegistryStore`` so a blind evaluation can pass nothing
(or a registry loaded from a production path) while a plumbing test can pass a
``StaticRegistry`` built from known events.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .events import load_registry


@runtime_checkable
class RegistryStore(Protocol):
    def events(self) -> list[dict[str, Any]]: ...


@dataclass
class StaticRegistry:
    entries: list[dict[str, Any]] = field(default_factory=list)

    def events(self) -> list[dict[str, Any]]:
        return list(self.entries)


class FileRegistry:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def events(self) -> list[dict[str, Any]]:
        return load_registry(self.path)


class NullRegistry:
    def events(self) -> list[dict[str, Any]]:
        return []


def as_registry(
    registry: RegistryStore | list[dict[str, Any]] | None,
) -> RegistryStore:
    """Normalize a registry argument; ``None`` becomes an empty registry."""
    if registry is None:
        return NullRegistry()
    if isinstance(registry, list):
        return StaticRegistry(registry)
    return registry
