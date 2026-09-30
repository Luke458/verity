"""Notification sinks for weekly assessment outcomes.

A scheduler that only reports through an exit code still needs to reach a human
when a refresh stops being clean. This module renders a bounded, deterministic
payload from a :class:`~qc.weekly.WeeklyResult` and dispatches it to one or
more sinks.

Design rules, all of them load-bearing:

* **Fail-soft.** Delivery is auxiliary to the assessment. A sink that errors,
  times out or refuses the payload records a note and never changes the QC
  status, the exit code or the published report. A notification outage must not
  be able to turn a passing week into a failed one, or an investigation into a
  silent one.
* **Bounded.** The payload is capped, not truncated arbitrarily: a report is
  built by dropping optional sections in a fixed order until it fits, and the
  dropped sections are named in ``omitted`` so a reader can tell the difference
  between "not applicable" and "did not fit".
* **Deterministic.** Keys are sorted and the payload carries a content digest,
  so a redelivery of the same assessment is byte-identical and an operator can
  prove which assessment a notification referred to.
* **Credentials never in argv.** A sink reads its token from the environment
  variable named in the sink spec, matching how ``qc weekly`` already handles
  storage options.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .jsonutil import dumps as json_dumps

NOTIFY_SCHEMA = 1

# Statuses worth waking someone for. A clean or explained week is silent by
# default: the orchestrator already records it and a noisy sink trains people to
# ignore the sink.
DEFAULT_NOTIFY_STATUSES: tuple[str, ...] = (
    "INVESTIGATE",
    "DATA_CONTRACT_FAILURE",
    "CONTRACT_FAILURE",
    "INCOMPLETE",
)

DEFAULT_MAX_PAYLOAD_BYTES = 16_384
DEFAULT_TIMEOUT_SECONDS = 10.0

# Ordered by how much a human needs them; ``_shrink`` drops from the end of the
# tuple first, so ``run_id`` goes before ``decision`` and ``expectations`` last.
_OPTIONAL_SECTIONS: tuple[str, ...] = (
    "expectations",
    "reference",
    "decision",
    "artifacts",
    "report_dir",
    "run_id",
)


@dataclass(frozen=True)
class NotificationResult:
    """Outcome of one dispatch attempt to one sink."""

    sink: str
    delivered: bool
    bytes_sent: int
    detail: str = ""
    payload_digest: str = ""
    suppressed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "sink": self.sink,
            "delivered": self.delivered,
            "bytes_sent": self.bytes_sent,
            "detail": self.detail,
            "payload_digest": self.payload_digest,
            "suppressed": self.suppressed,
        }


_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def _validate_webhook_target(target: str) -> None:
    """HTTPS only, except plain HTTP to a loopback listener (local relays, tests)."""
    from urllib.parse import urlsplit

    parts = urlsplit(target)
    if parts.scheme == "https" and parts.hostname:
        return
    if parts.scheme == "http" and parts.hostname in _LOOPBACK_HOSTS:
        return
    raise ValueError(
        f"webhook target must be https:// (or http:// to loopback), got {target!r}"
    )


class _RefuseRedirect(urllib.request.HTTPRedirectHandler):
    """Never follow a redirect: it would forward the bearer token elsewhere."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        raise urllib.error.HTTPError(
            req.full_url, code, f"redirect to {newurl} refused", headers, fp
        )


_OPENER = urllib.request.build_opener(_RefuseRedirect)


@dataclass(frozen=True)
class NotificationSpec:
    """One configured destination."""

    kind: str
    target: str
    timeout: float = DEFAULT_TIMEOUT_SECONDS
    token_env: str = ""
    headers: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, spec: Mapping[str, Any]) -> NotificationSpec:
        kind = str(spec.get("kind", "")).strip().lower()
        if kind not in ("webhook", "file", "stdout"):
            raise ValueError(
                f"unknown notification sink kind {kind!r}; "
                "expected 'webhook', 'file' or 'stdout'"
            )
        target = str(spec.get("target", ""))
        if kind in ("webhook", "file") and not target:
            raise ValueError(f"notification sink {kind!r} requires a target")
        if kind == "webhook":
            _validate_webhook_target(target)
        timeout = float(spec.get("timeout", DEFAULT_TIMEOUT_SECONDS))
        if timeout <= 0:
            raise ValueError("notification timeout must be positive")
        headers = spec.get("headers") or {}
        if not isinstance(headers, Mapping):
            raise ValueError("notification headers must be an object")
        return cls(
            kind=kind,
            target=target,
            timeout=timeout,
            token_env=str(spec.get("token_env", "")),
            headers={str(key): str(value) for key, value in headers.items()},
        )


def load_notification_specs(path: str | Path) -> list[NotificationSpec]:
    """Read a JSON document of ``{"sinks": [...]}`` or a bare list of sinks."""
    import json

    payload = json.loads(Path(path).read_text())
    if isinstance(payload, Mapping):
        payload = payload.get("sinks", [])
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        raise ValueError("notification spec must be a list of sinks")
    return [NotificationSpec.from_dict(item) for item in payload]


def build_payload(
    result: Any,
    statuses: Sequence[str] = DEFAULT_NOTIFY_STATUSES,
    max_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
) -> dict[str, Any]:
    """Render the notification body for an assessment result.

    The returned mapping always carries ``notify_schema``, ``status`` and
    ``payload_digest`` once serialised. ``omitted`` names any optional section
    dropped to respect ``max_bytes``; it is empty when nothing was dropped.
    """
    status = str(getattr(result, "status", "") or "UNKNOWN")
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")

    payload: dict[str, Any] = {
        "notify_schema": NOTIFY_SCHEMA,
        "dataset": getattr(result, "dataset", None),
        "status": status,
        "previous_version": getattr(result, "previous_version", None),
        "current_version": getattr(result, "current_version", None),
        "stage": getattr(result, "stage", None),
        "assessment_id": getattr(result, "assessment_id", ""),
        "attempt_id": getattr(result, "attempt_id", ""),
        "elapsed_seconds": getattr(result, "elapsed_seconds", 0.0),
        "notify_on": list(statuses),
        "omitted": [],
    }
    if status not in set(statuses):
        payload["suppressed"] = "status not in notify_on"
    if getattr(result, "skipped", False):
        payload["suppressed"] = "assessment was cached; no new notification"

    if getattr(result, "notes", None):
        payload["notes"] = [str(note) for note in result.notes]

    for name in _OPTIONAL_SECTIONS:
        if not hasattr(result, name):
            continue
        value = getattr(result, name)
        if value in (None, {}, [], ""):
            continue
        payload[name] = value

    encoded = json_dumps(payload, sort_keys=True)
    if len(encoded.encode("utf-8")) > max_bytes:
        payload = _shrink(payload, max_bytes)
    encoded = json_dumps(payload, sort_keys=True)
    if len(encoded.encode("utf-8")) > max_bytes:
        # Even the required fields alone do not fit. Refuse rather than emit a
        # payload that violates the caller's bound.
        raise ValueError(
            "notification payload exceeds max_bytes even after dropping "
            "optional sections"
        )
    payload["payload_digest"] = _digest(payload)
    return payload


def _digest(payload: Mapping[str, Any]) -> str:
    import hashlib

    body = {key: value for key, value in payload.items() if key != "payload_digest"}
    return hashlib.sha256(
        json_dumps(body, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]


def _shrink(payload: dict[str, Any], max_bytes: int) -> dict[str, Any]:
    """Drop optional sections in a fixed order until the body fits.

    Drops accumulate: several sections may have to go before the body fits, and
    nothing is dropped once it does.
    """
    payload = dict(payload)
    omitted: list[str] = []
    for name in reversed(_OPTIONAL_SECTIONS):
        if len(json_dumps(payload, sort_keys=True).encode("utf-8")) <= max_bytes:
            break
        if name not in payload:
            continue
        payload.pop(name)
        omitted.append(name)
        payload["omitted"] = sorted(omitted)
    return payload


def notify_webhook(spec: NotificationSpec, payload: Mapping[str, Any]) -> NotificationResult:
    """POST the payload as JSON. Token comes from the environment, never argv."""
    _validate_webhook_target(spec.target)
    body = json_dumps(payload, sort_keys=True)
    data = body.encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        **dict(spec.headers),
    }
    if spec.token_env:
        token = os.environ.get(spec.token_env, "")
        if not token:
            return NotificationResult(
                sink="webhook",
                delivered=False,
                bytes_sent=0,
                detail=f"environment variable {spec.token_env!r} is unset or empty",
                payload_digest=str(payload.get("payload_digest", "")),
            )
        headers.setdefault("Authorization", f"Bearer {token}")
    request = urllib.request.Request(
        spec.target, data=data, headers=headers, method="POST"
    )
    try:
        with _OPENER.open(request, timeout=spec.timeout) as response:
            response.read(4096)
    except urllib.error.HTTPError as error:
        return NotificationResult(
            sink="webhook",
            delivered=False,
            bytes_sent=len(data),
            detail=f"HTTP {error.code}",
            payload_digest=str(payload.get("payload_digest", "")),
        )
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        return NotificationResult(
            sink="webhook",
            delivered=False,
            bytes_sent=len(data),
            detail=f"{type(error).__name__}: {error}",
            payload_digest=str(payload.get("payload_digest", "")),
        )
    return NotificationResult(
        sink="webhook",
        delivered=True,
        bytes_sent=len(data),
        payload_digest=str(payload.get("payload_digest", "")),
    )


def notify_file(spec: NotificationSpec, payload: Mapping[str, Any]) -> NotificationResult:
    """Append the payload as one JSON line. Parent directories are created."""
    body = json_dumps(payload, sort_keys=True)
    data = body.encode("utf-8")
    path = Path(spec.target)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(body + "\n")
    except OSError as error:
        return NotificationResult(
            sink="file",
            delivered=False,
            bytes_sent=len(data),
            detail=f"{type(error).__name__}: {error}",
            payload_digest=str(payload.get("payload_digest", "")),
        )
    return NotificationResult(
        sink="file",
        delivered=True,
        bytes_sent=len(data),
        payload_digest=str(payload.get("payload_digest", "")),
    )


def notify_stdout(spec: NotificationSpec, payload: Mapping[str, Any]) -> NotificationResult:
    """Print the payload. Useful for smoke tests and for piping to a pager."""
    import sys

    body = json_dumps(payload, sort_keys=True)
    print(body)
    del sys
    return NotificationResult(
        sink="stdout",
        delivered=True,
        bytes_sent=len(body.encode("utf-8")),
        payload_digest=str(payload.get("payload_digest", "")),
    )


_DISPATCH = {
    "webhook": notify_webhook,
    "file": notify_file,
    "stdout": notify_stdout,
}


def dispatch(
    spec: NotificationSpec,
    payload: Mapping[str, Any],
) -> NotificationResult:
    """Deliver to one sink, converting any failure into a recorded outcome."""
    handler = _DISPATCH[spec.kind]
    try:
        return handler(spec, payload)
    except Exception as error:  # noqa: BLE001 - a sink must never break a run
        return NotificationResult(
            sink=spec.kind,
            delivered=False,
            bytes_sent=0,
            detail=f"{type(error).__name__}: {error}",
            payload_digest=str(payload.get("payload_digest", "")),
        )


def dispatch_all(
    specs: Sequence[NotificationSpec],
    result: Any,
    statuses: Sequence[str] = DEFAULT_NOTIFY_STATUSES,
    max_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
) -> list[NotificationResult]:
    """Render once and fan out to every sink. Never raises."""
    if not specs:
        return []
    try:
        payload = build_payload(result, statuses=statuses, max_bytes=max_bytes)
    except Exception as error:  # noqa: BLE001 - fail-soft at the boundary
        return [
            NotificationResult(
                sink=spec.kind,
                delivered=False,
                bytes_sent=0,
                detail=f"payload error: {type(error).__name__}: {error}",
            )
            for spec in specs
        ]
    if "suppressed" in payload:
        # A clean status or a cached retry must not reach any sink.
        return [
            NotificationResult(
                sink=spec.kind,
                delivered=False,
                bytes_sent=0,
                detail=f"suppressed: {payload['suppressed']}",
                payload_digest=str(payload.get("payload_digest", "")),
                suppressed=True,
            )
            for spec in specs
        ]
    return [dispatch(spec, payload) for spec in specs]
