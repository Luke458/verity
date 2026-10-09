"""Every README capability must appear in the claims matrix.

The claims matrix exists so no capability is described as validated without
recorded evidence. This test keeps the README and docs/claims.md in sync.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
CLAIMS = ROOT / "docs" / "claims.md"


def _readme_capabilities() -> list[str]:
    text = README.read_text()
    section = text.split("## Capabilities", 1)[1].split("\n## ", 1)[0]
    rows = re.findall(r"^\|\s*([^|]+?)\s*\|", section, flags=re.MULTILINE)
    return [
        row
        for row in rows
        if row != "Capability" and set(row) - {"-", ":"}
    ]


def test_readme_capabilities_are_in_claims_matrix() -> None:
    claims = CLAIMS.read_text()
    capabilities = _readme_capabilities()
    assert capabilities, "README lost its capabilities table"
    missing = [name for name in capabilities if f"| {name} " not in claims]
    assert not missing, f"capabilities missing from docs/claims.md: {missing}"


def test_claims_matrix_has_statuses() -> None:
    claims = CLAIMS.read_text()
    for status in ("validated-synthetic", "implemented-unmeasured", "research"):
        assert status in claims, f"claims matrix lost the {status} status"
