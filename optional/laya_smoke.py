"""Integration evidence only: pinned CPU HTTP protocol smoke."""

import json
import time
import urllib.request
from pathlib import Path
from urllib.error import HTTPError

started = time.monotonic()
payload = {
    "model": "convaiinnovations/laya",
    "state": json.dumps(
        {
            "status": "INVESTIGATE",
            "findings": [{"check": "mass_balance", "outcome": "FAIL"}],
        }
    ),
    "questions": {
        "cause": {
            "type": "choice",
            "instructions": "Which cause is supported?",
            "criteria": {
                "REPORT": "Report totals doubled",
                "UNKNOWN": "Cause unsupported",
            },
        },
        "severity": {
            "type": "score",
            "instructions": "How severe is this failure?",
            "criteria": ["Limited", "Material", "Widespread"],
        },
        "review": {
            "type": "noul",
            "instructions": "Does the failed check require analyst review?",
        },
    },
}
request = urllib.request.Request(
    "http://127.0.0.1:8766/v1/systemone",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=65) as response:
    result = json.loads(response.read(262145))
assert set(result["answers"]) == set(payload["questions"])
assert result["metadata"]["device"] == "cpu"
assert result["metadata"]["revision"] == "1c5edc17a7acd8701df6fc341c0d179f1c62c982"
result.update(
    elapsed_seconds=time.monotonic() - started,
    validation="integration only",
    production_eligible=False,
)
Path("reports/laya-smoke.json").write_text(
    json.dumps(result, indent=2, allow_nan=False)
)
print(
    json.dumps(
        {
            "device": "cpu",
            "elapsed_seconds": result["elapsed_seconds"],
            "answers": result["answers"],
        }
    )
)

# Invalid requests are bounded, explicit abstentions and leave the service alive.

for malformed in (
    b"[]",
    b"{",
    json.dumps({"model": payload["model"], "state": "{}", "questions": {}}).encode(),
    b" " * 262145,
):
    request = urllib.request.Request(
        "http://127.0.0.1:8766/v1/systemone",
        data=malformed,
        headers={"Content-Type": "application/json"},
    )
    try:
        urllib.request.urlopen(request, timeout=65)
        raise AssertionError("malformed request was accepted")
    except HTTPError as error:
        assert error.code in (422, 503)
        body = error.read(262145)
        assert len(body) < 262145
        assert json.loads(body)["abstained"]
with urllib.request.urlopen("http://127.0.0.1:8766/health", timeout=10) as response:
    assert response.status == 200
result["malformed_request_checks"] = 4
Path("reports/laya-smoke.json").write_text(
    json.dumps(result, indent=2, allow_nan=False)
)
