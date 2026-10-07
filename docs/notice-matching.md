# Experiment: matching free-text notices to findings

**Status: done; test results at the end. Shipped as `qc notices`**
([weekly-run.md](weekly-run.md#drafting-explanations-from-notices)); the sandbox
is `experiments/notices/`.
Research sandbox; nothing here changes how the engine decides a status.

## The gap

The engine can clear a structural change only through a registered, approved
expected event (classification, entity, week range; `attribution._match_expected`).
In practice the information arrives as text, in the days before a refresh:

> Chain 4 store S012 closed for refit from week 118, reopening in about six weeks.
> Products in Cough & Cold moving to Respiratory from wk 120 (range review).
> Store S900 (new, Burwood) loaded with 12 weeks of history.

Someone has to read the notices, work out which flagged change each one
describes, and write the registry entry. That is a reading task. The cause
labeller experiment ([labeller-experiment.md](labeller-experiment.md)) showed
that a decision model recognizes direct evidence well but applies conditional
rules poorly. Matching a notice to one of a few observed changes needs the
first skill, not the second.

## The task

For one assessed refresh and one notice, the candidates are the refresh's
observed lifecycle changes, grouped by classification (for example "latest
week missing: stores S012, S044" or "new entity with history: store S900,
weeks 109-120"), plus **none of these**. A decision model answers one
`choice` question per notice: which candidate does the notice describe? The
candidates are exactly the options Decision-2.0-Lux scores jointly.

A match becomes a **draft** registry entry (classification, entity IDs, week
range, notice text as provenance), never an approved one. A person approves
it, and only then does the deterministic engine use it on a rerun. The
status path is unchanged, and so is the rule that nothing clears itself.

## Ground truth and data

Generated alongside each scenario (`qcgen`), so the oracle knows the answer:

- **True notices** for the injected change: store closures (missing stores),
  new stores with history (backfill), history removed (truncation), category
  moves (reclassification), store merges (entity merge). Several templates
  per change type, with varied ID formats ("S012", "store 12"), week phrasing
  and irrelevant detail.
- **Distractors**, the dangerous case:
  - the right change type for an entity not in this refresh;
  - the right entity with the wrong change ("S012 refit" against a backfill
    of S012);
  - the right entity and change outside the observed weeks;
  - notices about another dataset, and unrelated operational chatter.
- **Clean refreshes** with only distractors.

A later, harder set rewrites the templated notices with a local LLM
(Qwen3.8-27B on the development GPU) so wording is not template-regular.

## Measurement

Per notice: the chosen option against the oracle's (a candidate or none).
Reported separately:

- **False explanation rate:** a distractor matched to a real change. This is
  the error that could get a real fault approved away, so it is the headline.
- **Recall:** true notices matched to the right change.
- By distractor kind, and templated versus LLM-rewritten.

Baseline: a deterministic matcher that extracts entity IDs, weeks and
change-type keywords with patterns and matches them to candidates. On
templated notices it should be hard to beat. The question is whether it
degrades on rewritten text while the model does not.

Splits by seed as in the labeller (train for any threshold tuning, dev for
iteration, test scored once), and predictions registered before the test
run.

## Dev results (iteration)

Dev split: 100 refreshes (seeds 8101-8110), 366 notices on refreshes with at
least one observed change. Methods:

- **patterns**: written before any result and never changed.
- **lux**: one `choice` question, answering the most probable option.
- **lux + checks**: the model is asked only direct questions, and code combines
  them. After the choice, four yes/no checks on the chosen change: about this
  dataset? already happened by the latest week? same kind of change? weeks
  cover the observed weeks? The match stands only if every check clears 0.5
  (or one shared threshold tuned on dev).

The rewritten set is the same notices reworded by Qwen3.8-27B
(`rewrite.py`). A rewrite must keep every number and any other dataset's
name, or the template text is kept (40 of 510 on dev). A hand audit of 40 random
rewrites found 39 that keep their answer, and one borderline case.

| Dev | patterns | lux | lux + checks | lux + checks, tuned |
|---|---|---|---|---|
| Templated: accuracy | 0.98 | 0.70 | 0.80 | 0.87 |
| Templated: recall | 56/60 | 60/60 | 60/60 | 21/60 |
| Templated: false explanations | 1% | 30% | 20% | 3% |
| Rewritten: accuracy | 0.87 | 0.70 | 0.90 | 0.90 |
| Rewritten: recall | 34/60 | 60/60 | 56/60 | 45/60 |
| Rewritten: false explanations | 6% | 30% | 9% | 6% |

- **Templated notices:** the patterns know the template vocabulary by
  construction and win easily.
- **Rewritten notices:** the patterns lose 22 of their 56 matches.
- **A single choice question** finds every true notice but falsely explains
  30%: it recognizes the entity and the change and ignores the condition
  (another dataset, a future date). That is the cause labeller's weakness
  again.
- **Asking the conditions as separate yes/no questions** fixes the weeks
  (46% to 2% false explanations on rewritten notices). Other datasets remain
  the weak spot for both methods, especially when the notice names the other
  data source only in passing.

## Registered predictions (before the test run)

Frozen: patterns, lux + checks at 0.5, and lux + checks with the dev-tuned
threshold. On the test split (200 refreshes, seeds 8201-8220):

1. **Templated:** patterns beat lux + checks (0.5) on accuracy and on false
   explanations.
2. **Rewritten:** lux + checks (0.5) recall ≥ 0.85, and patterns recall ≤ 0.70.
3. **Rewritten:** every method's false explanation rate is between 3% and 12%.
4. **Rewritten:** other-dataset notices are the largest source of lux + checks'
   false explanations.

## Test results

Test split: 200 refreshes (seeds 8201-8220), 720 notices on refreshes with at
least one observed change. 940 of 1,020 notices were rewritten; the other 80
kept their template text.

| Test | patterns | lux + checks | lux + checks, tuned | lux, one question |
|---|---|---|---|---|
| Templated: accuracy | **0.97** | 0.77 | 0.86 | 0.68 |
| Templated: recall | 114/120 | 116/120 | 42/120 | 120/120 |
| Templated: false explanations | **2%** | 22% | 4% | 32% |
| Rewritten: accuracy | 0.83 | **0.87** | **0.87** | 0.70 |
| Rewritten: recall | 62/120 (0.52) | **107/120 (0.89)** | 89/120 (0.74) | 120/120 |
| Rewritten: false explanations | 9% | 11% | 9% | 30% |
| Rewritten: false explanations, other-dataset notices | 48/125 | 60/125 | 49/125 | 120/125 |

**Registered predictions** (methods = the three frozen ones):

1. **Templated, patterns win: held.** Accuracy 0.97 against 0.77; false
   explanations 2% against 22%.
2. **Rewritten recall: held.** Lux + checks 0.89 (needed ≥ 0.85); patterns 0.52
   (needed ≤ 0.70).
3. **Rewritten false explanations in 3-12%: held**, at 9%, 11% and 9%. The
   single-question model, never a frozen method, was 30%.
4. **Other datasets dominate Lux + checks' false explanations: held**, 60 of
   82.

**Two analyses not registered in advance** (rewritten test set):

- **Suggestion precision.** Of the matches each method proposes, 107/189
  (0.57) are right for Lux + checks and 62/127 (0.49) for patterns.
- **Notices already scoped to the dataset** (other-dataset notices removed):
  false explanations fall to 22/595 (3.7%) for Lux + checks and 17/595 (2.9%)
  for patterns.

## Conclusion

On realistically worded notices, a decision model asked direct questions
finds 89% of the notices that explain a flagged change, against 52% for
hand-written patterns, at a similar false explanation rate. The patterns win
only on the templates they were written for. This is the first role in this
repo where the model beats the deterministic alternative, and it is the one
the deterministic engine cannot fill: reading text.

It is a **suggestion aid, not an approver.** About four in ten of its
suggestions are wrong, mostly notices about another dataset that mention the
same stores. Its matches belong in draft registry entries that a person
approves, which is the only way the engine accepts an explanation anyway.
Notices should be scoped to their dataset upstream (by channel or metadata),
where both methods' false explanations fall to about 3%.

What the result does not show: the notices, rewrites and distractors are
synthetic. Real notices are messier and refer to things this generator does
not model (banners, regions, promotions, partial weeks).

Results: [`results/notices-test.json`](results/notices-test.json) and
[`results/notices-test-rewritten.json`](results/notices-test-rewritten.json).
