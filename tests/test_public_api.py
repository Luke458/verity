"""The package's declared public surface must actually resolve.

`qc/__init__.py` re-exports ~185 names through `__all__`. A name listed there
but never imported is a silent breakage: `qc.some_name` raises AttributeError
while every linter stays green, because none of them cross-check `__all__`
against the module namespace. This does.
"""

from __future__ import annotations

import qc


def test_every_declared_export_resolves() -> None:
    missing = sorted(name for name in qc.__all__ if not hasattr(qc, name))
    assert not missing, f"qc.__all__ declares names that do not exist: {missing}"


def test_public_surface_has_no_duplicates() -> None:
    assert len(qc.__all__) == len(set(qc.__all__))
