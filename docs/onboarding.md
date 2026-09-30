# Onboarding a table

Pointing the engine at a new table should be a checklist, not a debugging
session. `qc onboard` profiles a Delta table, proposes a configuration and
reports blockers.

## Profile and propose

```sh
qc onboard --uri abfss://container@account.dfs.core.windows.net/gold/fact \
  --storage-options '{"account_name": "...", "account_key": "..."}' \
  --out config/datasets/retail.yaml

# With a version pair: contract and shape assessment too
qc onboard --uri ./lake/fact --previous 41 --current 42 --out config/datasets/retail.yaml
```

What it reads (never writes): schema, a bounded sample (default 50k rows;
`--count` for an exact row count), and, when versions are given, both versions.

Proposed config fields: `week_column`, `metric_columns`, `primary_metric`,
`entity_columns`, `entity_key_columns`, `report_grain`, `count_entity_map`,
`required_columns`. The YAML loads directly with `load_dataset_config`.

Findings:

| Severity | Code | Meaning |
|---|---|---|
| BLOCKER | `MISSING_WEEK_COLUMN` / `WEEK_NOT_INTEGER` | no integer week column; derive one from the date |
| BLOCKER | `WEEK_NOT_CONTIGUOUS` | gaps in sampled weeks |
| BLOCKER | `MISSING_METRIC_COLUMNS` | no numeric measures |
| BLOCKER | `DUPLICATE_KEYS` | week + entity keys duplicated |
| BLOCKER | `CONTRACT_FAILED` | engine contract failure on the version pair |
| BLOCKER | `VERSION_READ_FAILED` / `PAIR_BUILD_FAILED` | version or pair could not be read |
| WARNING | `MISSING_ENTITY_COLUMNS`, `NULL_METRICS`, `WEEK_RANGE_UNUSUAL`, `PAIR_*` | review before trusting |

Inference is heuristic: always review the proposal, and prefer explicit
overrides for anything unusual. Datetime-only tables are rejected by design —
the engine compares integer week identifiers.

## Production field mapping

Canonical names are what the engine consumes: `week`, `store_id`,
`product_id`, `banner_id`, `state_id`, `commodity_id`, `dollar`, `units`,
`scripts`, `stock`. When production names differ, pass aliases as
`production=canonical` and onboarding emits a `column_map` (canonical ->
production) that the source layer applies on read:

```sh
qc onboard --uri ./lake/fact \
  --alias wk=week --alias sty=store_id --alias pfc=product_id --alias dol=dollar \
  --out config/datasets/retail.yaml

qc delta-run --uri ./lake/fact --config config/datasets/retail.yaml
qc weekly --uri ./lake/fact --config config/datasets/retail.yaml --store data/qc.db
```

Mapping rules:

- direction is canonical -> production in the YAML (`product_id: pfc`), so the
  engine and all its config keys stay canonical;
- the mapping applies to facts and dimension tables, at every stage;
- a mapped production column that is missing from a read frame is recorded as
  a source warning, and the contract check fails loudly if a required canonical
  column never appears - a wrong column is never scored;
- mappings are validated: canonical and production names must each be unique.

## Runbook

1. `qc onboard --uri ... --out config/datasets/retail.yaml`, fix blockers and
   review the proposal.
2. `qc delta-info --uri ...` to confirm the version history.
3. `qc delta-run --uri ... --previous N --current N+1 --config ... --json` for
   a first assessment.
4. Schedule `qc weekly` (see [weekly-run.md](weekly-run.md)).
5. Before trusting a threshold, evaluate the configuration on synthetic data
   with `qc cohort --plan config/cohort.json --config ...`.

Limitations: onboarding reads a sample, so rare null/duplicate patterns can
hide beyond 50k rows; raise `--sample-rows` or use `--count`. Version pairs are
assumed to share one stage table unless `--stage` is set.
