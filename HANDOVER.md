# SSC CGL Tier-1 2026 — Prediction Project Handover

**Objective:** forecast the topic composition of the SSC CGL Tier-1 2026 paper from past-year papers, and generate realistic practice papers from that forecast.

**Status:** working end-to-end. 98 papers, 9,102 questions parsed and tagged; walk-forward backtest running; 2026 forecast produced; 18 mock papers generated.

**Read this first if you are restarting from zero.** It contains the findings, the dead ends, and the bugs — most of the value here is knowing what *not* to repeat.

---

## 0. The one-paragraph version

Specific 2026 questions cannot be predicted; topic *weights* can. Five years of papers (2021–2025) give enough data to estimate topic proportions, but only 2–3 usable test years, so no model reliably beats "average the last 3–5 years". A **rule-based regex tagger beats a local 7B LLM by more than 2× on labelling accuracy** (~80% vs 35%) because SSC questions are heavily templated. The binding constraint on forecast quality is **number of years of data**, not model sophistication or tagger quality.

---

## 1. Data acquisition

**Source:** `careerpower.in/ssc-cgl-previous-year-question-paper.html` — 226 direct PDF links, no login.

**Filtering 226 → 98 papers (Tier-1, English):**

| Excluded | Count |
|---|---|
| Hindi versions | 84 |
| Tier-2 papers | 33 |
| Subject-split (Maths-only, English-only) | 6 |
| Answer keys | 3 |
| No exam year in filename | 2 |

> **Trap:** the URL's upload date is *not* the exam year. Files named `15-Maths-English.pdf` uploaded in 2021 are actually **CGL 2019 Tier-2** papers. Requiring an explicit year *in the filename* is the fix; falling back to the upload path silently mislabels old Tier-2 papers as recent Tier-1 ones.

**Final corpus:**

| Year | Papers | Questions |
|---|---|---|
| 2021 | 3 | 188 |
| 2022 | 9 | 779 |
| 2023 | 17 | 1,551 |
| 2024 | 24 | 2,152 |
| 2025 | 45 | 4,432 |
| **Total** | **98** | **9,102** |

**Source quality:** 56 official, 42 "memory-based" reconstructions (nearly all 2025). Both are recorded per paper via `source_type`, so their effect can be measured rather than assumed.

**OCR:** 14 papers (all 2023) were image-only scans. `ocrmypdf --force-ocr` recovered **all 14**, 80–96 questions each, in 7 minutes. This tripled 2023 from 3 papers to 17 and cut baseline MAE by 11%.

---

## 2. PDF parsing — four bugs that each destroy the data silently

The first parse "succeeded" with 10,473 questions and was entirely garbage. Every extracted question was the tail of the previous answer glued to the next question, misaligned by one.

1. **Splitting on bare numbers splits on answer options.** SSC marks questions `Q.1` but options `1. 2. 3. 4.` — matching both inflates counts ~60%.
2. **Two marker dialects.** 2022–2024 use `Q.1`; 2025 uses `Q1.`. Matching only one falls back to bare-number splitting.
3. **No trailing space is guaranteed** — `Q.2How many squares…` appears in 2024 papers. Requiring `\s` after the number drops the marker entirely.
4. **Response-sheet detection must key on content, not markers.** SSC's official papers embed candidate responses *alongside* full question text. Rejecting on "Chosen Option" throws away good papers; the correct test is **median chunk length** (< 40 chars ⇒ questions are images).

**Sanity check that actually works:** section totals should land near 25 per paper. But **aggregate totals matching expectations is weak evidence** — errors cancel. Reasoning looked correct at ~25 while simultaneously *losing* figure-based questions to image extraction and *gaining* misfiled English questions.

**Known permanent loss:** figure-based questions (paper folding, mirror images, counting shapes) have no extractable text. This is a *systematic* loss that under-counts exactly those topics.

---

## 3. Tagging — the most important finding

### Local LLMs failed

| Model | Speed | Valid labels | Full-corpus time | **Accuracy** |
|---|---|---|---|---|
| deepseek-r1:8b | 0.02 q/s | 58% | 125 h | not measured |
| qwen2.5:7b | 0.52 q/s | 100% | 4.2 h | **35%** |

**35% accuracy, hand-verified on 20 reasoning questions.** The failure was systematic, not random:

- `series_completion` became a dumping ground — **872 questions (11% of the corpus)**, absorbing coding-decoding, odd-one-out, word-formation and blood-relations questions
- `coding_decoding` was **never assigned once** across 7,858 questions, despite being an SSC staple

**Two prompt lessons worth keeping:**

- **Ask for a topic *number*, not a name.** Listing topics grouped under section headings made both models answer with the *heading* (`general_awareness`) instead of a topic. Validation rejected those as invalid — the models were classifying correctly and the harness was discarding it. Switching to a flat numbered list and an integer reply took valid output from 42% → **100%**.
- **Model confidence is useless for routing.** Mean 0.96, median 1.00, minimum 0.80 — not one question below 0.7. A small model is confident whether or not it is right.

### Rules won

| | Local LLM | Regex rules |
|---|---|---|
| Precision | 35% | **~78–80%** |
| Coverage | 100% (guessed) | 67.8% (declines rest) |
| Runtime | 4.2 h | **~1 s** |
| Cost | free | free |
| Deterministic | no | **yes** |

**Why:** SSC questions are heavily templated. `"Select the most appropriate antonym of"` appears 120 times verbatim, `"synonym"` 110, `"In a certain code language"` is *always* coding-decoding. A regex gets those right every time.

**Design principle: low coverage beats wrong labels.** A wrong label *biases* topic counts; a missing label only *shrinks the sample*. Questions matching no rule are left untagged deliberately.

**General awareness needs a different approach.** GA has no stem templates — *"Who among the following…"* says nothing about topic. The signal is in **named entities**: dynasties, rivers, articles, schemes. An entity gazetteer took GA from 6.2 → 9.9 questions per paper.

### Rule-writing traps

- **Curly quotes.** PDFs contain `‘+’` and `–`, not `'+'` and `-`. Normalise punctuation before matching.
- **Word-boundary stripping for de-spaced text.** 10 papers extract without spaces (`selectthemissingnumber`). Stripping `\b` to match them turns `\bratio\b` into bare `ratio`, which matches inside **ope*ratio*n**. Only de-space alternatives ≥12 characters.
- **Over-broad keywords.** `"prime minister"` claimed ordinary GK for polity; bare `"festival"` claimed an *International Paragliding Festival* for art & culture; `\bboat` claimed the *Nehru Trophy Boat Race* for boats-and-streams.
- **Order matters absolutely.** Disambiguators must precede general rules — `"how is X related to"` must beat `coding_decoding` on coded blood-relations questions.

---

## 4. Forecasting — no model beat naive averaging

**Walk-forward protocol:** train on years < T, predict T, score, advance. Never let post-T information reach the model.

**Results across every data variant tried** (synthetic, LLM-labelled, rule-labelled, OCR-expanded):

| Model | MAE | Skill vs baseline |
|---|---|---|
| dirichlet_a0.5_hl2 | 0.514 | +4.6% |
| mean_last_5 | 0.515 | +4.5% |
| **mean_last_3 (baseline)** | 0.539 | — |
| dirichlet_a2_hlinf | 0.544 | −0.9% |
| last_year | 0.587 | −8.8% |

**Nothing beat the baseline by more than noise.** The one solid finding: `last_year` is clearly *worse* — a single paper is too noisy to extrapolate from, and averaging several years genuinely helps.

### Statistical limits that dominate everything

- **Only 5 years of data ⇒ 2–3 test years.** Any skill score at n=2 is nearly meaningless. This is the binding constraint.
- **Data volume per year dominates model choice.** On synthetic data, going from 1 paper/year to 12 shifts/year moved MAE 1.05 → 0.37 and top-10 hit rate 0.50 → 0.71. Collecting more shifts beats any modelling improvement.
- **Interval coverage is 81–86% against a 90% target** — slightly overconfident even after fixes.

### The section-scaling bug (subtle, invalidates results)

Predictions were forced to the nominal 25 questions/section while actuals — under 64% tagger coverage — summed to 61. Every error metric measured that 39-question gap instead of forecast skill. Fixing it **cut MAE by a third and reversed the model ranking.**

**Backtesting and forecasting need opposite scaling:**

- **Backtest** → scale to *observed* (tagged) totals, or you measure coverage gaps
- **Forecast** → scale to the *nominal* 25/section, because the real 2026 paper will have 25 per section regardless of what your tagger reached

---

## 5. The 2026 forecast

Model `mean_last_5`, trained 2021–2025. Expected questions per topic, per 100-question paper.

### Quantitative Aptitude (25)
Percentage 3.4 · Geometry 3.3 · Profit & Loss 3.3 · Mensuration 3.0 · Time-Speed-Distance 1.7 · SI & CI 1.6 · Ratio 1.4 · Algebra 1.2 · Trigonometry 1.2 · Average 1.1 · Time & Work 0.8 · Number System 0.8 · Simplification 0.6 · Pipes & Cisterns 0.5 · Data Interpretation 0.5 · Mixture 0.4 · Boats 0.2 · Height & Distance 0.1

### Reasoning (25)
Series Completion 5.8 · Coding-Decoding 3.6 · Odd-one-out 2.9 · Syllogism 2.2 · Blood Relations 1.9 · Math Operations 1.9 · Analogy 1.7 · Statement-Conclusion 1.2 · Mirror/Water Image 1.0 · Word Formation 0.8 · Cube & Dice 0.6 · Embedded Figures 0.5 · Paper Folding 0.4 · Venn 0.3 · **Direction Sense 0.0** · **Matrix 0.0** · **Ranking 0.0**

### English (25)
Cloze 3.8 · Fill in the Blanks 2.6 · One-Word Substitution 2.5 · Antonyms 2.2 · Para Jumbles 2.0 · Sentence Improvement 1.9 · Active/Passive 1.9 · Synonyms 1.9 · Spelling 1.8 · Idioms 1.8 · Spotting Errors 1.3 · Direct/Indirect 0.9 · **Reading Comprehension 0.5**

### General Awareness (25)
Polity 5.6 · Sports 3.5 · Art & Culture 3.1 · Geography (India) 1.9 · Modern History 1.7 · Economics 1.6 · Biology 1.5 · Medieval History 1.3 · Ancient History 1.2 · Physics 0.9 · Chemistry 0.8 · Static GK 0.7 · Schemes 0.7 · **Current Affairs 0.3** · **World Geography 0.1**

> **Bold/zero entries are extraction gaps, not predictions.** Direction Sense, Matrix and Ranking read 0.0 because no rule matches them and figure-based questions lose their text — SSC still asks all three. Do not skip them.

**Confidence by section:** Quant, Reasoning and English rest on ~68% coverage at ~80% precision — usable for allocating study time. GA reaches only 9.9 of 25 questions per paper; treat its *ordering* as indicative and its *magnitudes* as soft.

---

## 5b. Method-level prediction — the layer that actually works

Topic counts answer "how many series questions?". **Method** answers "which kind,
solved how?" — the level a candidate prepares at. This is the most predictive
finding in the project.

### It validates forward

Pooling 2021–2024 method mixes predicts the 2025 mix better than copying 2024
alone, on **16 of 22 topics** (mean TVD 0.30 vs 0.34; uniform 0.34):

| Topic | Predicted | Copy 2024 | Uniform |
|---|---|---|---|
| geometry | **0.10** | 0.11 | 0.38 |
| ratio_and_proportion | **0.11** | 0.11 | 0.69 |
| time_speed_distance | **0.14** | 0.30 | 0.40 |
| coding_decoding | **0.17** | 0.21 | 0.18 |
| profit_and_loss | **0.28** | 0.38 | 0.44 |

Geometry at 0.10 means its method mix — chord/tangent vs triangle-centres vs
polygon — is ~90% reproducible year to year.

### Number series: rules recovered arithmetically

The generating rule is computable from the digits, so this needs no model at all:

| Family | Share | Example → rule |
|---|---|---|
| quadratic (2nd difference constant) | 30% | `382, 322, 272, 232, 202` |
| linear recurrence ×p+q | 15% | `2, 4, 10, 28, ?, 244` → **×3 − 2** |
| geometric | 9% | `111, 222, 444, ?, 1776` → ×2 |
| cubic | 5% | `24, 35, 51, 73, 102` |
| prime differences | 3% | `1225, 1184, 1147…` → −41, −37, −31, −29 |
| cyclic differences | 2% | `82, 79, 71, ?, 60, 57, 49` → −3, −8 repeating |

Solver notes that cost real debugging time: keep the `?` as a gap (collapsing it
breaks every recurrence); fit polynomials through known (index, value) points so
the gap position stops mattering; derive cyclic/prime/alternating rules from the
leading known run and generate forward.

### Highest-stability methods — prepare these first

Stability = 1 − variation of the method's yearly share.

| Topic | Method | Share | Stability |
|---|---|---|---|
| coding-decoding | length-change transform | 74% | 89% |
| polity | institution roles | 38% | 89% |
| geometry | circle / chord / tangent | 58% | 88% |
| profit & loss | marked price + discount | 43% | 88% |
| spotting errors | subject–verb agreement | 82% | 88% |
| art & culture | dance forms | 60% | 86% |

### General awareness is NOT random

Of 206 GA questions citing an explicit year:

| Lag from exam year | Share |
|---|---|
| same year | 10% |
| **1 year before** | **21%** |
| 2 years | 14% |
| 3 years | 8% |
| historical (>8y) | 44% |

**53% of year-citing GA questions draw on the 0–3 years before the exam, peaking
at the previous year.** For CGL 2026: 2025 events are the richest source, then
2024 and 2023.

**Anniversary theory: tested and rejected.** Only 3% of historical references fall
on a 25-year boundary — indistinguishable from chance. Recorded so nobody spends
time on it again.

**2011 is the most-cited historical year** (12 references vs 6 for 1857) — the
Census, a permanent anchor for demographic questions.

### Coverage limit

41% of tagged questions match a method signature. The rest are unclassified —
declined rather than guessed, same principle as the topic rules. Series and
coding-decoding are solved arithmetically; everything else is lexical patterns
and therefore capped by how templated the wording is.

---

## 6. Practice paper generation

18 mock papers generated (100 questions each, 25 per section, answer key included), as PDFs.

- Questions are **real SSC questions with SSC's published answer keys** — never generated. A generated question can carry a subtly wrong key, and a practice paper that teaches the wrong answer is worse than none.
- Topic mix follows the 2026 forecast; largest-remainder rounding keeps each section at exactly 25.
- **Difficulty bands are within-topic terciles** based on structural proxies: multi-step wording, distinct quantities, stem length, operators, negation (*"which is NOT"*), and option spread. There is **no ground-truth difficulty data** — SSC publishes none. Bands are relative, not calibrated.
- Because banding is per topic, an easy paper and a hard paper have **identical topic composition**.
- The pool is consumed as papers are built, so papers never share a question. Pool of 2,837 eligible questions supports ~28 non-overlapping papers.

---

## 7. Pipeline and commands

```
src/fetch.py         download PYQ PDFs (rate-limited, resumable)
src/ocr.py           OCR scanned PDFs (backs up originals, verifies before replacing)
src/parse.py         PDF → structured questions
src/rules.py         regex tagger — 68 rules, free, ~80% precision   ← PRIMARY
src/tag.py           local LLM tagger (Ollama) — 35% accurate, not recommended
src/tag_api.py       Claude API tagger — batch + caching + structured outputs
src/cascade.py       route a subset to the paid API (gold / section policies)
src/eval_tagger.py   measure tagger accuracy against a hand-labelled gold set
src/models.py        baselines + Dirichlet-multinomial
src/metrics.py       MAE, RMSE, top-k hit, interval coverage, skill score
src/backtest.py      walk-forward evaluation + verdict
src/predict.py       forward forecast
src/difficulty.py    within-topic difficulty banding
src/mock_paper.py    assemble one mock paper
src/make_papers.py   generate multiple PDFs by difficulty band
src/synth.py         synthetic papers for validating the harness
src/archetypes.py    number-series + coding-decoding rule solver (arithmetic)
src/methods.py       method signatures for every topic
src/method_report.py method inventory with year-over-year stability
src/playbook.py      method playbook: real example + how-to-solve per method
src/validate_papers.py  backtest the PAPER GENERATOR, not just the forecast
```

**Full run from scratch:**

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
./.venv/bin/python src/fetch.py --list data/tier1_clean.txt
./.venv/bin/python src/ocr.py              # needs: brew install ocrmypdf
./.venv/bin/python src/parse.py
./.venv/bin/python src/rules.py
./.venv/bin/python src/backtest.py --papers data/tagged/rules.json
./.venv/bin/python src/predict.py --year 2026 --model mean_last_5
./.venv/bin/python src/method_report.py --md out/METHODS.md
./.venv/bin/python src/playbook.py
./.venv/bin/python src/validate_papers.py
./.venv/bin/python src/make_papers.py --per-band 4
```

---

## 8. What to do next, in order of value

0. **Work at method level, not topic level.** Section 5b is the most predictive
   result here; topic counts tie with naive averaging, method mixes do not.
1. **Find 2019–2020 papers.** More *years* is the binding constraint — two extra years roughly doubles the test set and is worth more than every other item here combined.
2. **Collect more shifts per year.** Data volume per year dominates model choice (MAE 1.05 → 0.37 in controlled tests).
3. **Hand-label 200 questions** across all four sections as a proper gold set. Current accuracy estimates rest on ~50 hand-checked questions and were partly tuned on.
4. **Tag GA via the Claude API** (~$1.60 with Batch + caching). GA is the one section rules cannot reach; it is the only place paid tagging clearly earns its cost.
5. **Do not invest further in tagger tuning or model sophistication.** Both have hit diminishing returns against the data constraint.

---

## 9. Methodological warnings

Recorded because each cost real time here:

- **Tuning on your test set invalidates it.** Rule precision measured 100% on the sample used to fix the rules, and ~78–80% on fresh held-out samples. Always draw a new sample after tuning.
- **Aggregate agreement is weak evidence.** Section totals near the expected 25 hid offsetting errors in both directions.
- **Agreement between two taggers is not accuracy.** Both can be confidently wrong on the same ambiguous boundary.
- **A verdict threshold without a sample-size check over-claims.** The harness announced "Real signal" off two test years; effect size alone is not enough.
- **Verify the scale before trusting any error metric.** The section-scaling bug made every model look wrong in a way that had nothing to do with forecasting.
- **Test the obvious hypothesis and record the negative.** Anniversaries looked
  like an obvious GA driver and are not (3%, ≈ chance). Writing that down stops
  the next person re-deriving it.
- **A successful-looking parse can be entirely garbage.** The first run produced 10,473 plausible questions that were all misaligned by one.

---

## 10. Honest summary

For SSC CGL 2026, **averaging the topic distribution of the last 3–5 years is the method**. Nothing more sophisticated earned its complexity — a finding that survived synthetic data, LLM labels, rule labels, OCR expansion and GA tagging.

**But that applies to topic COUNTS only.** At method level the picture is better:
pooled 2021–2024 method mixes beat copying the previous year on 16 of 22 topics,
and GA current-affairs questions concentrate measurably in the 1–3 years before
the exam. Prediction works one layer below where I first looked for it.

That is a real answer, not a failure. The forecast in §5 is a well-computed five-year average with known limitations, and the practice papers in §6 are its most directly useful output: authentic SSC questions, in the proportions the exam is most likely to use.
