# Real-data analyst shadow runbook

There is no real-data effectiveness evidence yet. The supported path is local
CPU with Parquet/delta-rs and read-only source access. Rules stay the default.

1. Onboard a table with at least two committed versions: `qc onboard --uri
   <table> --out config/datasets/pilot.yaml`. Review declared grain, metrics,
   calendar and field mappings. Supply explicit stage/dimension version maps.
2. Persist each assessment before importing labels: `qc weekly --uri <table>
   --config config/datasets/pilot.yaml --store data/pilot.db --out reports/pilot`.
   Read the final status and every required unavailable finding. INCOMPLETE
   requires review; it is not a pass. See [weekly recovery](weekly-run.md).
3. Import outcomes with `qc store import --store data/pilot.db --csv outcomes.csv
   --dry-run`, then repeat without `--dry-run`. Use the stored run ID, explicit
   `provenance=analyst`, analyst identity, review label, incident group and label
   observation timestamp. Corrections append revisions; they never overwrite.
4. `qc pilot-check` checks label prerequisites against a pinned plan. Its
   `LABEL_PREREQUISITES_MET` status does not certify engineering readiness,
   completed evaluation or production eligibility. These are separate fields.
5. Freeze the real store cohort at an explicit UTC cutoff and follow the
   [evaluation workflow](evaluation.md). Train, calibrate, select on development,
   then confirm one frozen challenger on untouched test incidents. Insufficient
   groups, classes, controls or confidence bounds cannot pass. Keep repeated
   snapshots and related incidents together; never substitute row splits.
6. Central eligibility requires a pinned real analyst test artifact, passed
   confidence-bound and paired comparison gates, and completed operational
   checks. Trainers, pilot readiness and synthetic benchmark wins cannot grant
   eligibility. Do not change a validation claim to real until it links to
   actual supporting data and results.

The reproducible [local walkthrough](reliability-limitations.md) uses synthetic
feedback and must remain insufficient for real qualification. Approval and label
authority, threshold governance and operational sign-off remain organizational
decisions. Spark/Databricks deployment, automatic publication blocking and GPU
serving are excluded. See the explicit [limitations register](reliability-limitations.md).
