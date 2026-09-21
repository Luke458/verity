"""Two hand-authored integration cases, not an accuracy benchmark."""

import json
import time
from pathlib import Path

from qc.laya import LayaDecisionProvider
from qc.run import run_qc
from tests.test_reliability import CONFIG, HandSource

rows = []
for doubled in (False, True):
    source = HandSource()
    if doubled:
        source.frames["v2", "report"]["dollar"] *= 2
    incumbent = run_qc(source, "v2", "v1", CONFIG)
    started = time.perf_counter()
    challenger = run_qc(
        source, "v2", "v1", CONFIG, decision_provider=LayaDecisionProvider()
    )
    rows.append(
        {
            "case": "doubled_report" if doubled else "unchanged_overlap",
            "rule_status": incumbent.status,
            "challenger_status": challenger.status,
            "elapsed_seconds": time.perf_counter() - started,
            "availability": challenger.machine["provider_availability"],
            "recommendation": challenger.machine["provider_recommendation"],
            "effective_review": challenger.machine["requires_investigation"],
            "response": challenger.machine.get("provider_response"),
        }
    )
assert rows[1]["challenger_status"] == "INVESTIGATE"
assert rows[1]["effective_review"]
Path("reports/laya-comparison.json").write_text(
    json.dumps(
        {"validation": "integration only", "production_eligible": False, "cases": rows},
        indent=2,
        allow_nan=False,
    )
)
print(
    json.dumps(
        [
            {
                key: row[key]
                for key in (
                    "case",
                    "availability",
                    "challenger_status",
                    "elapsed_seconds",
                )
            }
            for row in rows
        ]
    )
)
