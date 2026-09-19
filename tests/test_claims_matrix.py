"""Every README milestone must appear in the claims matrix.

The claims matrix exists so no milestone is described as validated without
recorded evidence. This test keeps the README and docs/claims.md in sync.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
CLAIMS = ROOT / "docs" / "claims.md"


def _readme_milestones() -> list[str]:
    text = README.read_text()
    section = text.split("## Milestones", 1)[1]
    rows = re.findall(r"^\|\s*([^|]+?)\s*\|", section, flags=re.MULTILINE)
    return [
        row
        for row in rows
        if row != "Milestone" and set(row) - {"-", ":"}
    ]


def test_readme_milestones_are_in_claims_matrix() -> None:
    claims = CLAIMS.read_text()
    missing = [
        milestone
        for milestone in _readme_milestones()
        if f"| {milestone} " not in claims
    ]
    assert not missing, f"milestones missing from docs/claims.md: {missing}"


def test_claims_matrix_has_statuses() -> None:
    claims = CLAIMS.read_text()
    for status in ("validated-real", "validated-synthetic", "plumbing-only", "research"):
        assert status in claims, f"claims matrix lost the {status} status"
