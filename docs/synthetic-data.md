# Synthetic data and fault oracle

`qcgen` builds a simulated retail transactional dataset and emits version pairs
with known injected faults. It exists so the deterministic QC engine can be
developed and tested against exact ground truth before any real data access.

## What it generates

- Entity universe: banners, states, stores, commodities, products, and a
  store-level assortment.
- Weekly truth at store x product x week with trend, seasonal shape, moving
  holidays (Christmas, Easter, EOFY), promotions and lognormal noise.
- A pipeline: `source -> coded -> warehouse -> report`, where report is
  aggregated to week x banner x state. Transforms are value-preserving, so a
  fault's first-divergence stage is exactly its injection stage.
- Version pairs: `V0001` (previous, clean, one week shorter) and `V0002`
  (current, faulty). Overlapping weeks are byte-identical except for injected
  faults, so revision differences are the fault plus the appended week.

## Profiles

| Profile | Banners | States | Stores | Products | Commodities | Weeks |
|---|---|---|---|---|---|---|
| tiny | 2 | 2 | 4 | 24 | 3 | 30 |
| small | 4 | 3 | 24 | 80 | 8 | 104 |
| full | 8 | 5 | 60 | 300 | 15 | 320 |

Any field can be overridden in `config/datasets/*.yaml`.

## Fault families

| Family | Stage | Expected class | Expected status |
|---|---|---|---|
| missing_stores | source | missing_stores | INVESTIGATE |
| missing_products | source | missing_products | INVESTIGATE |
| entity_merge | source | entity_merge | INVESTIGATE |
| new_store_backfill | source | backfill | INVESTIGATE |
| expected_event | source | backfill | PASS_WITH_EXPLANATION |
| history_truncation | source | truncation | INVESTIGATE |
| commodity_remap | source | reclassification | INVESTIGATE |
| coding_error | coded | coding | INVESTIGATE |
| warehouse_transform_error | warehouse | warehouse | INVESTIGATE |
| recalculation | source | historical_correction | INVESTIGATE |
| schema_failure | report | schema_failure | DATA_CONTRACT_FAILURE |
| null_duplicate_storm | warehouse | null_duplicate_storm | INVESTIGATE |
| market_movement | warehouse | market_movement | PASS (negative control) |

A suite cycles through the families and interleaves controls after every third
fault, so a run with `--scenarios 5` on the default config covers a mixed plan
and a larger run covers every family.

The product-level and merge families close coverage gaps rather than add
volume: `missing_products` exercises the `MISSING_PRODUCTS` semantic class, and
`entity_merge` removes one store while a new store absorbs its history, which
produces a `replaced_by` relationship candidate and the `ENTITY_MERGE` cause.
Both existed in the decision layer before but had no oracle scenarios.

Analyst behaviour is simulated separately (drafts, mistakes, corrections,
latency, investigation disagreement) with explicit `synthetic` provenance, so
the feedback loop can run end to end without being mistaken for real labels;
see [docs/synthetic-analyst.md](synthetic-analyst.md).

## Manifest

Each `scenario-XXXX/manifest.json` records:

- `fault`: family, injection stage and parameters;
- `cases[]`: one ground-truth case with `kind`, `expected_status`,
  `expected_class`, `expected_origin`, affected entities and weeks, the exact
  per-stage `effects`, and `injected_effect`;
- `expected_events[]`: registry entries for expected-event scenarios;
- `versions`: per-stage fingerprints and row counts for both versions.

Effects are measured against a clean pipeline built from the same truth, so
they are exact, not estimates. For `commodity_remap`, `details.per_value`
records the per-commodity redistribution and checks conservation.

## Verification

```sh
qcgen verify --suite-dir data/suites/demo
```

Verifies that every written Parquet file matches its manifest fingerprint and
that each fault satisfies its family invariant (sign, conservation,
units/dollar consistency, dropped column, duplicated units). It does not run
the QC engine; it is the generator's self-test.

## Using synthetic labels correctly

Use synthetic ground truth to assert the deterministic engine's status,
attribution, residual and first-divergence outputs exactly. Do **not** report
semantic-model accuracy on synthetic labels as evidence of quality: the
decision model is only as realistic as the simulator, and no simulator encodes
the ambiguity of real analyst judgement. Semantic metrics require real
confirmed outcomes.

## Delta port

`SnapshotStore` writes Parquet today so A0 runs anywhere. When Databricks is
available, add a Delta-backed store with the same interface (versioned tables
in Unity Catalog) and keep the Parquet backend for tests. The manifest and
fingerprint format are storage-independent.
