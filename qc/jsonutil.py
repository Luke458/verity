"""Strict JSON helpers.

Machine output must be valid JSON: ``NaN`` and ``Infinity`` are not. Every
serializer used for reports, CLI output or the store goes through
``sanitize_json`` and ``json.dumps(..., allow_nan=False)`` so a non-finite
float becomes ``null`` instead of an invalid token.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np


def sanitize_json(value: Any) -> Any:
    """Recursively replace non-finite floats with ``None``."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json(item) for item in value]
    return value


def json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def dumps(value: Any, **kwargs: Any) -> str:
    """Strict JSON: non-finite floats sanitized, NaN/Infinity disallowed."""
    kwargs.setdefault("allow_nan", False)
    kwargs.setdefault("default", json_default)
    return json.dumps(sanitize_json(value), **kwargs)
