# Synthetic analyst outcomes

Real analyst feedback is the one thing the project cannot manufacture. Until it
exists, this simulator exercises the *process* around it: confirmation and
drafts, latency, mistakes, unknowns, corrections, and disagreement with the
engine about whether investigation was needed.

It is a stand-in for the workflow, not evidence about models. Outcomes carry
provenance `synthetic`, so `records_from_store` and `qc champion` treat them as
unvalidated: `production_eligible` stays False even when every other gate
passes.

## Profiles

| Profile | Cause acc. | Origin acc. | Severity acc. | Confirm | Unknown | Correction | FP / FN investigation |
|---|---|---|---|---|---|---|---|
| `perfect` | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0 / 0 |
| `careful` | 0.92 | 0.95 | 0.90 | 1.00 | 0.02 | 0.05 | 0.02 / 0.02 |
| `typical` | 0.80 | 0.85 | 0.75 | 0.85 | 0.10 | 0.10 | 0.05 / 0.08 |
| `sloppy` | 0.55 | 0.70 | 0.55 | 0.60 | 0.20 | 0.15 | 0.10 / 0.15 |

`perfect` is for pipeline tests; `typical` and `sloppy` produce the messiness a
real feedback loop must survive. A custom `AnalystProfile` can be passed
programmatically for deterministic scenarios (for example
`correction_rate=1.0` to test that corrections replace first passes).

## What is simulated

- **Drafts**: unconfirmed outcomes are invisible to training forever.
- **Mistakes**: wrong cause / origin / severity, sampled uniformly among the
  other classes, plus an `UNKNOWN` rate.
- **Corrections**: a later confirmed outcome with the oracle labels; the
  store's latest-wins rule means the correction becomes the label.
- **Latency**: `created` dates spread over `max_delay_weeks`, so the store
  looks like weekly batches rather than a single dump.
- **Investigation disagreement**: false positives and false negatives on
  `requires_investigation`, which stress the escalation gate labels.
- **Notes**: templated resolutions per cause, which also feed incident-memory
  text for future extraction work.

## Usage

```sh
qc simulate-analyst --suite-dir data/suites/demo --store data/analyst.db \
  --profile typical --seed 7

qc store list --store data/analyst.db
qc champion --store data/analyst.db --suite-dir data/suites/champion-eval \
  --text-embedder answerdotai/ModernBERT-base --out reports/artifacts/champion-v1
```

`qc champion` reports `labels: synthetic` in its provenance block and
`production_eligible: False`. To promote a champion, the same rows must come
from real analysts: `provenance` is stored per outcome, and only
`provenance == "analyst"` unlocks production eligibility.

## Limits

- The simulator uses the fault oracle as ground truth, so it inherits the
  generator's view of what is a cause. Real analysts disagree with each other
  and with the oracle in ways no profile captures.
- Random mistakes are not realistic confusion structure; a future version
  could take a confusion matrix per team or per cause family.
- Latency is synthetic dates, not queue dynamics; prequential calibration
  still needs real `available_on` timing.
