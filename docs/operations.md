# Operations: durable store and drift monitoring

The operational layer is a single SQLite database plus drift checks over the
across-load calibration pool. It is deliberately small: history and
confirmation semantics, no service.

## Confirmed-only store

```sh
qc store add-run --store data/qc.db --scenario-dir data/suites/demo/scenario-0000 \
  --root-cause MISSING_STORES --confirmed --resolution "supplier extract fixed"
qc store add-outcome --store data/qc.db --run-id V0001->V0002 \
  --root-cause MISSING_STORES --confirmed
qc store list --store data/qc.db [--confirmed-only]
qc store incidents --store data/qc.db --scenario-dir data/suites/demo/scenario-0000 --k 3
```

Tables:

| Table | Semantics |
|---|---|
| `runs` | immutable machine payload per run, plus the versioned feature vector used for retrieval |
| `outcomes` | append-only analyst confirmations; the latest outcome per run wins |
| `registry` | expected-event entries revisioned by save; `load_registry` returns the latest revision per event |
| `assessments` | immutable complete-input identity, results and publication journal |
| `attempts` | separate execution attempts, including retries |
| `calibration_revisions` | append-only evidence with observation cutoffs |
| `relationships` | entity relationships, deduplicated per `(source, target, entity type, relationship)` |

Rules:

- a run is recorded once; re-recording raises rather than overwriting history;
- retrieval (`qc store incidents`) only ever uses runs whose **latest** outcome
  is confirmed, so model guesses and drafts can never become memory;
- outcomes are never mutated; a correction is a new row and becomes the latest;
- the JSONL stores (`IncidentStore`, `RelationshipStore`, `PrequentialStore`)
  remain for lightweight use; the SQLite store is the durable replacement.

## Drift monitoring

```sh
qc drift --store data/calibration.jsonl --as-of 105 --target-week 105
```

`monitor_drift` splits the usable prequential residuals (in target order) into
a baseline and a recent window and compares:

- **residual scale**: median absolute residual ratio. Above
  `ratio_threshold` (default 1.5) flags `residual_scale_drift`;
- **coverage**: the fraction of recent residuals inside the baseline conformal
  interval. Below `1 - alpha - coverage_margin` flags `coverage_drift`;
- **insufficient data**: fewer than two usable windows reports `INSUFFICIENT`
  and never a verdict.

`STABLE` means the windows are consistent, not that calibration is certified.
Drift checks detect distribution shift in residuals; they do not detect
concept drift in the business meaning of a field, and they should be run before
trusting calibration after a pipeline or model change.

## Suggested weekly loop

```text
refresh completes
  -> qc weekly --store data/qc.db     (assessment journal and report publication)
  -> scheduled agent reads reports    (investigation, polls)
  -> analyst confirms outcome         (qc store add-outcome)
  -> qc drift                         (calibration health)
  -> retrain / recalibrate when gates or drift demand it
```

Threshold governance (who may change `materiality_ratio`, temporal thresholds
or promotion gates, and when) remains a process decision, not code.

Schema 4 migrations back up existing stores and preserve labels/provenance.
Historical retrieval selects the latest revision available at its cutoff.
See [weekly recovery and exit codes](weekly-run.md).
