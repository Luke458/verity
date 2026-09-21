"""Exercise onboarding, durable shadow, outcome import, and honest sparse evaluation."""

import csv
import json
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
from deltalake import write_deltalake

from qc.store import SqliteStore


def main():
    stages = []
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        lake = root / "fact"
        for weeks in (3, 4):
            frame = pd.DataFrame(
                [
                    {
                        "week": week,
                        "store_id": "s",
                        "product_id": "p",
                        "dollar": 10.0,
                        "units": 1.0,
                    }
                    for week in range(1, weeks + 1)
                ]
            )
            write_deltalake(str(lake), pa.Table.from_pandas(frame), mode="overwrite")

        def call(name, arguments, allowed=(0,)):
            process = subprocess.run(
                [sys.executable, "-m", "qc.cli", *arguments],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if process.returncode not in allowed:
                raise RuntimeError(
                    f"{name}: {process.returncode}: {process.stderr}: {process.stdout}"
                )
            payload = json.loads(process.stdout)
            stages.append(
                {"step": name, "exit_code": process.returncode, "result": payload}
            )
            return payload

        config = root / "retail.yaml"
        database = root / "qc.sqlite"
        call(
            "onboard",
            ["onboard", "--uri", str(lake), "--out", str(config), "--json"],
            (0, 2),
        )
        call(
            "weekly",
            [
                "weekly",
                "--uri",
                str(lake),
                "--config",
                str(config),
                "--store",
                str(database),
                "--out",
                str(root / "reports"),
                "--json",
            ],
            (2, 3, 4),
        )
        with SqliteStore(database) as store:
            run_id = store.connection.execute("SELECT run_id FROM runs").fetchone()[0]
        outcomes = root / "outcomes.csv"
        with outcomes.open("w") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "run_id",
                    "root_cause",
                    "confirmed",
                    "analyst",
                    "provenance",
                    "requires_investigation",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "run_id": run_id,
                    "root_cause": "UNKNOWN",
                    "confirmed": "true",
                    "analyst": "fixture",
                    "provenance": "synthetic",
                    "requires_investigation": "true",
                }
            )
        arguments = [
            "store",
            "import",
            "--store",
            str(database),
            "--csv",
            str(outcomes),
            "--json",
        ]
        call("import_dry_run", [*arguments, "--dry-run"])
        call("import", arguments)
        cohort = call(
            "cohort",
            [
                "cohort",
                "--source",
                "store",
                "--store",
                str(database),
                "--cutoff",
                datetime.now(UTC).isoformat(),
                "--out",
                str(root / "cohort"),
                "--json",
            ],
            (4,),
        )
        assert cohort["status"] == "INSUFFICIENT_EVIDENCE"
        assert not cohort["cases"], (
            "Synthetic feedback must not become real analyst evidence"
        )
    report = {
        "validation": "synthetic integration only",
        "production_eligible": False,
        "steps": stages,
    }
    target = Path("reports/onboarding-feedback.json")
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(
        json.dumps(
            {
                "artifact": str(target),
                "steps": [s["step"] for s in stages],
                "status": cohort["status"],
            }
        )
    )


if __name__ == "__main__":
    main()
