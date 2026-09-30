"""Documentation tables must agree with the code they document.

`qcgen/spec.py` declares `FAMILY_SPECS` as the single source of truth for fault
-family expectations. `docs/synthetic-data.md` renders the same table for human
readers, and had already drifted from it on three families. This test diffs the
two so the drift is a build failure rather than a quiet inconsistency: change
`FAMILY_SPECS`, then update the table (or fail here).
"""

from __future__ import annotations

from pathlib import Path

from qcgen.spec import FAMILY_SPECS

ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_DATA = ROOT / "docs" / "synthetic-data.md"

# Family | Stage | Kind | Expected class | Expected origin | Expected status
_EXPECTED_COLUMNS = (
    "Family",
    "Stage",
    "Kind",
    "Expected class",
    "Expected origin",
    "Expected status",
)
NONE_TOKEN = "(none)"


def _family_rows() -> dict[str, dict[str, str]]:
    text = SYNTHETIC_DATA.read_text()
    section = text.split("## Fault families", 1)[1]
    rows: dict[str, dict[str, str]] = {}
    started = False
    for line in section.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            if started:
                break
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if not started:
            # Header row, then the |---| delimiter.
            if tuple(cells[: len(_EXPECTED_COLUMNS)]) != _EXPECTED_COLUMNS:
                raise AssertionError(
                    f"fault-family table header changed: {cells[: len(_EXPECTED_COLUMNS)]}"
                )
            started = True
            continue
        if set("".join(cells)) <= {"-", ":", " "}:
            continue
        family = cells[0]
        rows[family] = dict(
            zip(_EXPECTED_COLUMNS[1:], cells[1 : len(_EXPECTED_COLUMNS)], strict=True)
        )
    return rows


def test_fault_family_table_matches_spec() -> None:
    documented = _family_rows()
    expected = {family: spec for family, spec in FAMILY_SPECS.items()}

    assert set(documented) == set(expected), (
        "docs/synthetic-data.md families differ from FAMILY_SPECS: "
        f"missing={sorted(set(expected) - set(documented))} "
        f"extra={sorted(set(documented) - set(expected))}"
    )

    for family, spec in expected.items():
        row = documented[family]
        want = {
            "Stage": spec.stage,
            "Kind": spec.kind,
            "Expected class": spec.expected_class,
            "Expected origin": spec.expected_origin or NONE_TOKEN,
            "Expected status": spec.expected_status,
        }
        for column, value in want.items():
            assert row[column] == value, (
                f"{family}.{column}: docs/synthetic-data.md says "
                f"{row[column]!r}, FAMILY_SPECS says {value!r}"
            )


def test_documented_requires_investigation_is_derived() -> None:
    """The doc claims requires_investigation follows the expected status."""
    text = SYNTHETIC_DATA.read_text()
    assert "requires_investigation" in text
    for family, spec in FAMILY_SPECS.items():
        derived = spec.requires_investigation
        assert derived == (
            spec.expected_status in {"INVESTIGATE", "DATA_CONTRACT_FAILURE"}
        ), f"{family}: requires_investigation disagrees with expected_status"
