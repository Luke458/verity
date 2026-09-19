from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qcgen.config import suite_config  # noqa: E402


@pytest.fixture()
def tiny_config():
    return suite_config("tiny")


@pytest.fixture()
def all_stages():
    return ("source", "coded", "warehouse", "report")
