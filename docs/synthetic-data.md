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
| realistic | 4 | 3 | 24 | 80 | 8 | 104 |

Any field can be overridden in `config/datasets/*.yaml`.

`tiny`, `small` and `full` share one seasonal curve across every commodity and
never restate history between versions, which makes commodity shares nearly
constant and revisions exact. `realistic` turns on the realism knobs in
`HistoryConfig` (all off by default, and drawing nothing from the RNG when off,
so other profiles are byte-identical):

| Knob | realistic | Effect |
|---|---|---|
| `commodity_season_amplitude` | 0.35 | each commodity gets its own annual cycle; shares swing ~1.5x over a year |
| `market_shock_sigma` | 0.04 | market-wide weekly demand shock |
| `commodity_shock_sigma` | 0.03 | category-level weekly noise in shares |
| `intermittency` | 0.3 | a third of products trade in a store-week with probability 0.25-0.75; no row otherwise |
| `late_arrival_weeks`, `late_arrival_fraction` | 2, 0.03 | the previous snapshot misses ~3% of its last week and ~1.5% of the week before; every clean refresh restates them |

The values are plausible, uncalibrated assumptions about pharmacy retail. They
were fixed before any detector ran on the profile, except late arrival, which
was added after the first sweep showed that clean refreshes never changing
history flattered the revision checks.

## Fault size and fault pairs

`build_scenario(..., magnitude=m)` sets a family's size explicitly; the meaning
of `m` per family is in `qcgen.scenarios.MAGNITUDE_MEANING` (for example the
fraction of stores absent, or the fractional dollar cut). The oracle records the
magnitude and the realised relative effect (largest relative change in any
week's dollar total at any stage). `build_scenario(..., also=(family,))`
injects further families at their own stages, each with its own oracle case.
`qc sweep` drives both; see [evaluation.md](evaluation.md).

## Fault families

This table is generated from `qcgen/spec.py` (`FAMILY_SPECS`), which is the
single source of truth. `tests/test_docs_sync.py` diffs the two, so the table
cannot drift again: change `FAMILY_SPECS`, then update this table to match.

| Family | Stage | Kind | Expected class | Expected origin | Expected status | Notes |
|---|---|---|---|---|---|---|
| missing_stores | source | fault | missing_stores | source | INVESTIGATE | |
| missing_products | source | fault | missing_products | source | INVESTIGATE | |
| entity_merge | source | fault | entity_merge | source | INVESTIGATE | |
| new_store_backfill | source | fault | backfill | source | INVESTIGATE | |
| expected_event | source | expected_event | backfill | source | PASS_WITH_EXPLANATION | Registered event explains the historical revision; blind runs (no registry) report INVESTIGATE. |
| history_truncation | source | fault | truncation | source | INVESTIGATE | |
| commodity_remap | source | fault | reclassification | source | INVESTIGATE | |
| coding_error | coded | fault | coding | coded | INVESTIGATE | |
| warehouse_transform_error | warehouse | fault | warehouse | warehouse | INVESTIGATE | |
| recalculation | source | fault | historical_correction | source | INVESTIGATE | |
| schema_failure | report | fault | schema_failure | report | DATA_CONTRACT_FAILURE | |
| null_duplicate_storm | warehouse | fault | null_duplicate_storm | warehouse | DATA_CONTRACT_FAILURE | Contract failure — the report table is not fit for revision QC — not an ordinary INVESTIGATE. |
| market_movement | warehouse | movement | market_movement | (none) | INVESTIGATE | Genuine latest-week business movement, not a data fault; it must surface as a latest-week anomaly and is scored as a detection target, never as a control. |
| clean | warehouse | control | clean | (none) | PASS | Negative control: no injection. The only family expected to PASS; the false-positive rate is measured on it. |

`requires_investigation` is derived, not declared: it is true exactly when the
expected status is `INVESTIGATE` or `DATA_CONTRACT_FAILURE`
(`FamilySpec.requires_investigation`). A suite cycles through the families and
interleaves controls after every third fault, so a run with `--scenarios 5` on
the default config covers a mixed plan and a larger run covers every family.

The product-level and merge families close coverage gaps rather than add
volume: `missing_products` exercises the `MISSING_PRODUCTS` cause label, and
`entity_merge` removes one store while a new store absorbs its history, which
produces a `replaced_by` relationship candidate and the `ENTITY_MERGE` cause.

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

## Using the oracle correctly

Use the oracle to assert the engine's status, attribution, residual and
first-divergence outputs and to measure detection and false-positive rates
(see [evaluation.md](evaluation.md)). The generator shares the engine
authors' assumptions, so these rates validate the engine against its own model
of retail faults, not against real refreshes.
