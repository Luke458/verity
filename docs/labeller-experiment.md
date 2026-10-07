# Experiment: a learned cause labeller

Can a small model name a refresh's cause(s) better than the engine's rules,
using only what the engine already outputs? This is a research sandbox
(`experiments/labeller/`). Its labels never reach `apply_policy`, so they
cannot change a status.

## Setup

- **Rows.** Each row is one generated refresh run through the full engine
  (engine `7d122859c6bd`). Seeds alternate between the `small` and `realistic`
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
- **Models.** `rules` is the engine's `likely_causes`: every cause its rules
  support (see [engine.md](engine.md)). `trees` is one gradient-boosted
  classifier per cause, with each threshold set for F1 on dev.

```
pip install -e ".[experiments]"
python -m experiments.labeller.run build --split train --jobs 4 --data reports/labeller
python -m experiments.labeller.run build --split dev --jobs 4 --data reports/labeller
python -m experiments.labeller.run build --split test --jobs 4 --data reports/labeller
python -m experiments.labeller.run evaluate --data reports/labeller --out reports/labeller/report.json
```

## Results (test split, Wilson 95% intervals)

| Exact cause set | rules | trees |
|---|---|---|
| All 380 refreshes | 322 (0.85, 0.81-0.88) | 350 (0.92, 0.89-0.94) |
| 260 single faults | 236 (0.91, 0.87-0.94) | 242 (0.93, 0.89-0.96) |
| 40 clean | 36 (0.90) | 39 (0.97) |
| 80 pairs, both causes named | 51 (0.64, 0.53-0.73) | 70 (0.88, 0.78-0.93) |

Paired exact McNemar tests (same test refreshes):

- **All refreshes:** 32 cases only the trees got right and 4 only the rules
  got right, p = 2e-6.
- **Single faults:** 9 vs 3, p = 0.15.

| Single faults, fitted on one profile and tested on the other | rules | trees |
|---|---|---|
| Fitted on `small`, tested on `realistic` | 116/130 (0.89) | 97/130 (0.75) |
| Fitted on `realistic`, tested on `small` | 120/130 (0.92) | 112/130 (0.86) |

Full report: [`results/labeller.json`](results/labeller.json).

## How the rules got here

The first comparison was against the rules as they were: one label per
refresh, the first matching rule. On the same test split those scored 231/260
on single faults and named both causes of a pair 7/80 times; the trees beat
them on single faults (14 vs 3, p = 0.013). The trees' gains showed where the
rules fell short, and two changes followed:

- **Every supported cause, not just the first.** Most of the work was
  stopping one fault from being counted twice. A store outage also removes the
  products sold only there; a merge retires one id and backfills another; a
  remap can empty a category; and a structural change also shows up as an
  unexplained source-stage revision. Each was a false second cause until the
  rules credited it to the fault that caused it.
- **Name a lineage-localized stage below materiality.** A 1-2% coding error
  that lineage pins to the coding stage was labelled nothing; it is now
  CODING at LOW severity, as a label only (the status is unchanged).

## Reading it

- **On single faults the rules have caught up.** The remaining gap is not
  significant. Every family but `market_movement` scores 19-20/20. The rules
  leave market movement unlabelled by design: a category moving with nothing
  else wrong could be the market or an upstream loss the engine cannot
  localize. The trees name it 6/20.
- **Pairs are still the trees' advantage.** Of the 29 pairs the rules do not
  fully name, 16 include a market movement. 10 are a historical correction
  next to a merge, backfill or truncation, credited to that structural event
  on purpose, to avoid naming a single fault twice.
- **The trees do not transfer across profiles; the rules do.** Fitted on
  `small` and tested on `realistic`, the trees fall to 0.75 against the rules'
  0.89; the rules have no training distribution to drift from.
- **The four clean "errors"** are clean refreshes on which the engine itself
  raised INVESTIGATE (historical revisions under late arrival). Those are
  status false alarms, not labelling mistakes. Pooled over all splits, clean
  `realistic` refreshes alarm 5/70 (7%), in line with the sweep's 3/60.

## Verdict

The rules, with the two changes the trees pointed to, now match them on single
faults and transfer across profiles. The trees still name more simultaneous
faults, mostly ones involving a market movement the rules deliberately leave
alone. The learned model's real contribution was diagnostic: it showed where
the rules were weak. It stays in the sandbox.

## Caveats

- The truth and the features both come from the generator written alongside
  the engine. These are synthetic accuracies, not real-world ones.
- The rule changes were developed while looking at this dataset's train and
  dev splits, and the test split had been scored before. The test split is no
  longer an untouched held-out set for the rules. The sweeps on seeds
  7001-7010 are.
- One test split, one model family, no hyperparameter search.
- The datasets and sweeps were run twice with 4 workers and matched exactly.
  Runs with 6 workers on the development machine occasionally crashed or
  silently changed a result; that was traced to the machine, not the code.

## Decision model: Decision-2.0-Lux-9B (registered before results)

[Decision-2.0-Lux-9B](https://huggingface.co/vllm-sr/Decision-2.0-Lux-9B)
(Apache-2.0) is a Jev-style decision model: a state plus typed questions in,
a probability per option out. It sees the same 64 features as the trees,
rendered as readable JSON (`service.render_state`), and is asked one yes/no
question per cause. It is never trained on the generator.

It does not fit unmodified on the 16 GB development GPU, so the backbone is
quantized to EXL3 and run by a ROCm ExLlamaV3 fork, while Lux's own prompt
encoding, decision head and answer normalization are kept (`lux.py`).
Calibration rows for the quantizer come from train-split prompts only.

**Fidelity gate**, against the unmodified runtime (CPU, fp32) on 30 random dev
refreshes (300 answers):

- 8.0 bpw passes with median |ΔP(yes)| ≤ 0.01, max ≤ 0.05 and ≥ 99% of yes/no
  decisions unchanged.
- 4.0 bpw is used only with ≥ 97% of decisions unchanged and median
  |ΔP(yes)| ≤ 0.03; otherwise the 8.0 bpw model is used.

**Predictions**, on the 380 test refreshes:

1. Zero-shot (P ≥ 0.5) Lux names fewer single faults exactly right than the
   rules (236/260).
2. It over-names MARKET_MOVEMENT: more false positives for it than any
   other model.
3. With per-cause thresholds tuned on dev it improves, but still trails the
   trees on single faults.
4. It has no training distribution to drift from, so its dev-tuned
   cross-profile score drops less than the trees' (0.93 to 0.75).

### Results

**Fidelity.** The 4.0 bpw backbone (5.9 GB) passed its gate: 297/300 yes/no
decisions unchanged against the unmodified runtime, median |ΔP| 0.0055,
95th percentile 0.020, max 0.032. All three flips were within 0.03 of 0.5.
It also met the stricter 8.0 bpw thresholds, so no 8-bit model was built.
On the RX 9070 XT, 300 questions took about 40 s, model load included.

| Test split (Wilson 95%) | rules | trees | Lux, P ≥ 0.5 | Lux, dev-tuned |
|---|---|---|---|---|
| Single faults exactly right | 236/260 (0.91) | 242/260 (0.93) | 5/260 (0.02) | 98/260 (0.38, 0.32-0.44) |
| Clean refreshes left clean | 36/40 | 39/40 | 1/40 | 9/40 |
| Pairs, both causes named | 51/80 (0.64) | 70/80 (0.88) | 72/80 (0.90) | 56/80 (0.70) |
| Micro F1 | 0.924 | 0.961 | 0.472 | 0.697 |

Lux names 3.2 causes per refresh at P ≥ 0.5 against a true average of 1.1,
which is also why it "names both" causes of most pairs.

| Per-cause ranking (test AUC) | trees | Lux, zero-shot |
|---|---|---|
| ENTITY_MERGE, SCHEMA_FAILURE | 1.00 | 1.00 |
| MISSING_STORES, RECLASSIFICATION, BACKFILL, MISSING_PRODUCTS | 0.99-1.00 | 0.98-0.99 |
| WAREHOUSE, CODING | 1.00 | 0.96-0.97 |
| MARKET_MOVEMENT | 0.89 | 0.65 |
| HISTORICAL_CORRECTION | 1.00 | 0.60 |

**Predictions, scored.**

1. Fewer single faults right than the rules: **yes**, by far (5 and 98
   against 236).
2. Over-names MARKET_MOVEMENT: **yes**, with 196 false "yes" answers at 0.5
   against 0 for the rules and 3 for the trees. Its worst cause was in fact
   RECLASSIFICATION (199).
3. Tuning helps but trails the trees: **yes** (98 against 242).
4. Smaller cross-profile drop than the trees: **not meaningfully testable**.
   Dev-tuned Lux scores 0.22 (fitted on `small`, tested on `realistic`) and
   0.45 (the reverse), around its in-distribution 0.38; at that level the
   comparison says little.

**Reading it.** Lux reads the evidence: for eight of ten causes its
zero-shot ranking is nearly as good as trees trained on this generator. It
fails where the label depends on this domain's conventions, which the state
does not spell out. Every `realistic` refresh restates its last two weeks
through declared late arrival, which here is normal; asked whether "the
source restated history", Lux reasonably says yes. MARKET_MOVEMENT is the
call the rules deliberately refuse to make. An exact cause set needs all ten
answers right, and those two causes, plus a poor sense of base rates, spoil
most refreshes even with tuned thresholds.

**Verdict.** Not a usable labeller here, zero-shot or dev-tuned; the rules
stay the baseline. Next steps worth testing (new experiments, registered
first): stating the domain's conventions in the state (the declared
late-arrival window and what is expected), or fine-tuning on the train
split. Results: [`results/labeller.json`](results/labeller.json) (all
models) and [`results/lux.json`](results/lux.json) (fidelity, per-cause AUC
and provenance).

## Follow-up: the labelling policy as text (registered before the test run)

Decision models are built to apply a written policy to a state. Lux was given
the evidence and bare questions, but not the definitions an annotator would
get, or the feed's declared late-arrival window. Two prompt versions were
tried on the **dev split only** (`service.py`):

- **v2** adds a written labelling policy (one definition per cause) and the
  declared window to the state, and asks per cause whether it applies under
  the policy.
- **v3** also states absent evidence explicitly ("none"), and reports whether
  the *unexplained* part of a revision is material and where it first
  appears, with latest-week movement only when the engine flagged it.

| Dev split (190) | Single faults exact | Clean left clean | Pairs, both named | Micro F1 |
|---|---|---|---|---|
| Rules | 119/130 | 20/20 | 26/40 | 0.934 |
| Lux v1 | 3/130 | 0/20 | 36/40 | 0.468 |
| Lux v2 | 44/130 | 0/20 | 34/40 | 0.629 |
| Lux v3 | 36/130 | 19/20 | 28/40 | 0.591 |

The policy lifted HISTORICAL_CORRECTION's dev AUC from 0.61 to 0.86, and
MARKET_MOVEMENT's from 0.63 to 0.76. The remaining errors are conditional
rules the model does not apply: in v3 it still answers "coding error" when the
evidence says the change first appears at the source, and "market movement"
when missing stores explain the anomaly.

**Frozen and registered:** v2, the better dev micro F1, is scored once on the
test split. Prediction: zero-shot micro F1 within 0.05 of dev (0.58-0.68),
and fewer than half the rules' exact single faults (fewer than 118 of 260).

### Result

| Test split (380) | rules | trees | Lux v1 | Lux v1, tuned | Lux v2 (policy) | Lux v2, tuned |
|---|---|---|---|---|---|---|
| Single faults exact | 236/260 | 242/260 | 5/260 | 98/260 | 88/260 | 168/260 |
| Clean left clean | 36/40 | 39/40 | 1/40 | 9/40 | 0/40 | 14/40 |
| Pairs, both named | 51/80 | 70/80 | 72/80 | 56/80 | 68/80 | 32/80 |
| Micro F1 | 0.924 | 0.961 | 0.472 | 0.697 | 0.634 | 0.746 |

Both predictions held: micro F1 0.634 (registered range 0.58-0.68) and 88
exact single faults (registered: fewer than 118).

**Conclusion.** Writing the policy down roughly doubled what Lux gets right,
but the rules still win by a wide margin. The rules *are* the policy, applied
exactly; the model recognizes direct evidence well and applies conditional
definitions unreliably. Turning the engine's own numbers into causes is the
wrong job for a decision model. Its strength is reading text, which the
engine cannot do. That is the next experiment: matching free-text change
notices to findings.
