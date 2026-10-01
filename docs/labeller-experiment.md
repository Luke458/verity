# Experiment: a learned cause labeller

Can a small model name a refresh's cause(s) better than the engine's rules,
using only what the engine already outputs? This is a research sandbox
(`experiments/labeller/`). Its labels never reach `apply_policy`, so they
cannot change a status.

## Setup

- **Rows.** Each row is one generated refresh run through the full engine
  (engine `6da01c2966b2`). Seeds alternate between the `small` and `realistic`
  profiles; `realistic` declares its 2-week late-arrival window. Per seed: all
  13 single-fault families (sizable ones log-uniform at 1%-40%), 4 random
  two-fault pairs and 2 clean refreshes.
- **Splits** use disjoint seed ranges: train 8001-8040 (760 rows), dev
  8101-8110 (190), test 8201-8220 (380). Trees are fitted on train, their
  thresholds tuned on dev, and they are scored once on test.
- **Truth** is the oracle's set of causes: empty for clean, two for a pair.
- **Features** (`features.py`, 64): status, lifecycle events, missing-entity
  shares, attribution, reconstruction, reconciliation, lineage origin and
  per-stage increments, per-week revisions, temporal anomaly counts and the
  strongest share-test statistic, and failing checks by family. None read
  the oracle.
- **Models.** `rules` is the engine's own label: at most one cause, with
  UNKNOWN treated as no cause. `trees` is one gradient-boosted classifier per
  cause, so it can name several, with each threshold set for F1 on dev.

```
pip install -e ".[experiments]"
python -m experiments.labeller.run build --split train --jobs 6 --data reports/labeller
python -m experiments.labeller.run build --split dev --jobs 6 --data reports/labeller
python -m experiments.labeller.run build --split test --jobs 6 --data reports/labeller
python -m experiments.labeller.run evaluate --data reports/labeller --out reports/labeller/report.json
```

## Results (test split, Wilson 95% intervals)

| Exact cause set | rules | trees |
|---|---|---|
| All 380 refreshes | 274 (0.72, 0.67-0.76) | 350 (0.92, 0.89-0.94) |
| 260 single faults | 231 (0.89, 0.84-0.92) | 242 (0.93, 0.89-0.96) |
| 40 clean | 36 (0.90) | 40 (1.00) |
| 80 pairs, both causes named | 7 (0.09) | 70 (0.88) |

Paired exact McNemar tests (same test refreshes):

- **All refreshes:** 79 cases only the trees got right and 3 only the rules
  got right, p = 4e-20.
- **Single faults:** 14 vs 3, p = 0.013.

Full report: [`results/labeller.json`](results/labeller.json).

## Reading it

- **The headline is mostly structural.** The rules emit one label, so a pair
  can be named in full only when both faults share a cause. Most of the
  all-refresh gap is that design choice, not learning. The like-for-like
  comparison is single faults: +4 points, a real but small gain.
- **The single-fault gain comes from two families:**
  - `market_movement`: the rules leave it UNKNOWN by design (0/20); the trees
    name it 6/20 (F1 0.51).
  - `coding_error` at 1-2%: lineage finds the right stage, but the change is
    below materiality, so the rules say nothing (16/20); the trees name all
    20.

  Elsewhere the two are level, and on small `missing_products` the rules are
  slightly better (19/20 vs 17/20).
- **The trees do not transfer across profiles.**

  | Single faults | rules | trees |
  |---|---|---|
  | Fitted on `small`, tested on `realistic` | 114/130 (0.88) | 97/130 (0.75) |
  | Fitted on `realistic`, tested on `small` | 117/130 (0.90) | 121/130 (0.93) |

  The trees learn the scale of the generator's noise; the rules carry no
  training distribution to drift from. A labeller trained on one feed would
  need re-fitting, or proof that it transfers, before it could be trusted on
  another, and this sandbox has only two synthetic feeds to test that on.
- **The rules' four clean "errors"** are refreshes where the engine itself
  raised INVESTIGATE (historical revisions under late arrival). Those are status
  false alarms, not labelling mistakes. Pooled over all splits, clean
  `realistic` refreshes alarm 5/70 (7%), in line with the sweep's 3/60.

## Verdict

A per-cause tree model names simultaneous faults, which the rule labeller
cannot do at all, and is modestly better on single faults. It is also the more
brittle of the two: trained on one profile and tested on the other, it falls
13 points below the rules. That is the case for keeping it a sandbox. The
useful lessons for the rules are concrete: emit a set rather than one label,
and name a lineage-localized stage even below materiality, as a label rather
than a status change.

## Caveats

- The truth and the features both come from the generator written alongside
  the engine. These are synthetic accuracies, not real-world ones.
- One test split, one model family, no hyperparameter search. The dev split
  tunes thresholds only.
