# SSC CGL 2026 — Insights & Reasoning Notes

**Companion to `HANDOVER.md`.** That document records *what was built and what the
numbers were*. This one records *what was learned* — the judgment, the failed
hypotheses, and the reasoning that would otherwise have to be re-derived.

Read both and you can continue the work from the middle rather than the start.

---

## 1. The single most important realisation

**Prediction fails at the topic level and succeeds one layer below it.**

I spent most of the project forecasting *how many* questions per topic. Every
model tied with "average the last 3–5 years". That looked like the project's
answer. It was not — it was a symptom of asking at the wrong resolution.

Here is the actual hierarchy of signal:

| Layer | Predictability | Why |
|---|---|---|
| Section counts (25/25/25/25) | **Perfect but useless** | Fixed by the exam pattern. Zero information. |
| Topic counts within a section | **Weak** | Constrained near an average; little variance left to predict. Ties with naive averaging. |
| **Method mix within a topic** | **Real signal** | Reflects setter habits. Pooled 2021–24 beat copy-2024 on **16/22 topics**. |
| Specific questions | **None** | Not predictable by any method. |

The insight generalises: when an outer layer is *institutionally constrained*,
there is nothing to forecast there — the constraint already tells you the answer.
Push inward until you reach a layer where humans make discretionary choices.
Question-setters choose *which kind* of percentage question to write; they do not
choose how many questions the section has.

**If you resume this project, work at method level. Do not re-run the topic-count
modelling — it is a solved dead end.**

---

## 2. Entering the setter's mind — what the data reveals

The corpus is evidence about a process, not just a bag of questions. What it says
about how SSC actually builds a paper:

**They work from templates, not from scratch.** `"Select the most appropriate
antonym of"` appears 120 times *verbatim*; `"synonym"` 110. A regex over question
stems tags better than a 7B language model precisely because the wording is
institutionally standardised. Setters fill slots in frames.

**Each topic has a strongly preferred sub-method.** Geometry is 59%
circle/chord/tangent. Coding-decoding is 68% one transform family. Spotting-errors
is 82% subject–verb agreement. These are not uniform draws across a syllabus —
they are habits, and habits persist. Stability scores of 86–89% on these say the
preference is stable across years, not drifting.

**Multiple shifts per day are variants of one blueprint.** 2025 has 45 papers
across ~15 days × 3 shifts. Papers within a year resemble each other far more than
papers across years. This is why *data volume per year* mattered so much: more
shifts sharpen the estimate of that year's blueprint (MAE 1.05 → 0.37 going from
1 paper/year to 12).

**General awareness is recency-driven, not random.** Of GA questions citing an
explicit year, **53% reference the 0–3 years before the exam, peaking at the
previous year (21%)**. Setters draw current affairs from a rolling recent window.
For a 2026 paper: 2025 events are the richest source, then 2024, then 2023.

**Some anchors never age.** 2011 is the most-cited historical year in GA (12
references, vs 6 for 1857) because it is the Census year — a permanent reference
point for demographic questions. Look for these fixed anchors rather than assuming
all history is equally weighted.

**Anniversaries are NOT a driver.** I expected round-number anniversaries (75th of
independence, 150th of a figure's birth) to spike. Only 3% of historical
references fall on a 25-year boundary — indistinguishable from chance. *Tested and
rejected; do not spend time on it.*

---

## 3. Data engineering beat modelling, every single time

Ranked by actual impact on results:

| Intervention | Effect |
|---|---|
| More shifts per year (1 → 12, on controlled data) | MAE **1.05 → 0.37** |
| OCR of 14 scanned papers | baseline MAE **−11%**, interval coverage 61–79% → 81–86% |
| Fixing the section-scaling bug | MAE **−33%**, and the model ranking *reversed* |
| Rules instead of a local LLM for tagging | precision **35% → 80%** |
| Every modelling improvement combined | **~0%** (all ties) |

**Not one modelling change mattered.** Dirichlet-multinomial with recency
weighting, hyperparameter sweeps, shrinkage variants — all inside noise of a
five-year average.

The lesson for whoever continues: when a result plateaus, the next win is almost
never a better model. It is more data, cleaner data, or a bug in the pipeline you
have not found yet.

---

## 4. When deterministic beats probabilistic

The tagger comparison is the sharpest lesson in the project:

| | Local 7B LLM | Regex rules |
|---|---|---|
| Precision | 35% | **~80%** |
| Runtime | 4.2 hours | **~1 second** |
| Deterministic | no | **yes** |
| Cost | free | free |

**Why the LLM lost:** the failure was systematic, not noisy. `series_completion`
became a dumping ground absorbing 872 questions (11% of the corpus), while
`coding_decoding` was **never assigned once** in 7,858 questions despite being an
SSC staple. A model with no domain grounding collapses distinct categories into
whichever it has seen most.

**Why rules won:** the domain is templated. `"In a certain code language"` is
*always* coding-decoding. There is no ambiguity for a model to add value on.

**The general rule:** if the input is institutionally standardised, prefer
deterministic matching. Reach for a model where genuine judgment is required —
which for this corpus is general awareness, where the signal is in named entities
rather than phrasing, and where rules only reach 9.9 of 25 questions per paper.

**Corollary — low coverage beats wrong labels.** A wrong label *biases* every
downstream count; a missing label only *shrinks the sample*. The rule tagger
declines 32% of questions on purpose. That is a feature.

---

## 5. Verification discipline — the bugs that looked like successes

Every serious error in this project produced *plausible output*. None of them
crashed. This is the part most worth internalising.

**The first parse returned 10,473 questions and was entirely garbage.** Every
"question" was the tail of the previous answer glued to the next question,
misaligned by one — because SSC marks questions `Q.1` but options `1. 2. 3. 4.`,
and the splitter matched both. A count in the right ballpark proved nothing.

**Section totals near 25 hid offsetting errors.** Reasoning looked correct while
simultaneously *losing* figure-based questions to image extraction and *gaining*
misfiled English questions. Aggregate agreement is weak evidence: it cannot
distinguish "correct" from "wrong in two directions that cancel".

**The section-scaling bug inverted the model ranking.** Predictions were forced to
the nominal 25/section while actuals under partial coverage summed to 61. Every
error metric measured that 39-question gap. Fixing it cut MAE by a third and
changed which model won. **Always confirm predictions and actuals are on the same
scale before believing any error metric.**

**Backtesting and forecasting need opposite scaling.** Backtest against *observed*
(tagged) totals or you measure coverage gaps; forecast against *nominal* 25/section
because the real paper has 25 regardless of what your tagger reached. Conflating
these is easy and silent.

**I tuned on my own test set.** Rule precision measured 100% on the sample used to
fix the rules, and 78–80% on fresh held-out samples. Both numbers were "real"; only
the second was meaningful. **Draw a new sample after every tuning round.**

**A verdict threshold without a sample-size check over-claims.** The harness
announced "Real signal" off two test years. Effect size alone is not evidence.

**Practical protocol:** after every pipeline stage, check the output against an
*independent* expectation — not the stage's own success signal. Hand-check twenty
items. It caught a 35%-accuracy tagger that reported 96% mean confidence.

---

## 6. What a fresh model should do first

Given `HANDOVER.md` + this file + the repo, in order:

1. **Do not re-derive the topic-count forecast.** It is done and it ties with
   naive averaging. `out/forecast_2026.json` has the numbers.
2. **Start at method level** (`src/methods.py`, `src/archetypes.py`,
   `src/method_report.py`). That is where the remaining signal is.
3. **Improve method coverage.** Only 41% of tagged questions match a method
   signature. Every point gained here is worth more than any model change.
4. **Extend the arithmetic solvers.** `archetypes.py` recovers number-series rules
   *computationally* — no model, no guessing. The same is possible for letter
   series, number analogies, and matrix-style questions. This is the highest-value
   unexploited direction.
5. **Get more YEARS.** 2019–2020 papers would take the walk-forward from 3 test
   years to 5. This is the binding statistical constraint and no amount of
   modelling substitutes for it.
6. **Tag general awareness properly** (~$1.60 via Claude Batch API). It is the one
   section rules structurally cannot reach.

---

## 7. Hypotheses worth testing that I did not

Recorded so they are not lost:

- **Shift-position effects.** Does Shift 1 differ systematically from Shift 3?
  Papers within a day may be calibrated for difficulty parity — testable with the
  difficulty scorer in `src/difficulty.py`.
- **Question recycling across years.** Are near-duplicate questions reused with
  numbers changed? A fuzzy-match pass over stems would answer this, and if the
  recycling rate is meaningful it is directly exploitable.
- **Numeric habits.** Do setters prefer particular numbers (percentages that
  divide cleanly, ratios like 3:4:5)? Mining the constants might narrow expected
  answer ranges.
- **Option-position bias — TESTED, INCONCLUSIVE.** Memory-based papers show
  (a) 31% / (b) 32% / (c) 26% / (d) 12%, which looks like a strong bias against
  (d). It cannot be trusted: the same regex reads *official* papers as 92% (c),
  which is impossible, so the answer parser is broken for that format — and the
  memory-based figures may reflect whoever compiled the reconstruction rather than
  SSC. **Fix the official-paper answer parser first**; that also unlocks 3,000+
  official questions currently excluded from the question pool. Do not treat the
  (d) deficit as an exam strategy until it is reproduced on official papers.
- **Difficulty drift.** Is the exam getting harder year over year? The structural
  difficulty scorer could be regressed against year.
- **Cross-section correlation.** When quant is heavier on geometry, is reasoning
  heavier on figures? Papers may be balanced as a whole.

---

## 8. Honest limits of everything produced here

- **~80% tagger precision** caps every downstream number. Measured on ~50
  hand-checked questions, partly tuned on. The true figure could be low-to-mid 70s.
- **41% method coverage** means over half the corpus has no method label.
- **3 usable test years.** Every skill score carries wide error bars.
- **"Actual" is measured by my own tagger**, so validation bounds *process* error,
  not *total* error. A topic the tagger cannot see is invisible to both sides of
  the comparison and contributes zero measured error while being genuinely wrong.
- **Figure-based questions are permanently lost** — direction sense, matrix and
  ranking have no extractable text at all. Generated papers cannot include them,
  which is why every paper carries a "not covered" page.
- **42 of 98 papers are memory-based reconstructions**, concentrated in 2025 —
  the most recent and most heavily weighted year is also the least authoritative.

---

## 9. The one-line version

Specific questions are unpredictable; topic counts are predictable but trivially
so; **method mixes are predictable and useful** — and the fastest route to a better
2026 prediction is more years of data and better method coverage, not a better model.
