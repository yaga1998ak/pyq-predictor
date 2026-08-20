# PYQ Predictor — SSC CGL 2026

Backtested topic-distribution forecasting from past year question papers.

**What it predicts:** how many questions each topic is likely to carry, with
honest uncertainty ranges.
**What it does not predict:** specific questions. Nothing here can do that, and
any tool claiming otherwise is selling something.

---

## The one rule

**No number in any output comes from a language model.**

The LLM's only job is turning question text into a topic label. Counting,
weighting, trending and forecasting are deterministic Python you can re-run and
audit. Ask an LLM to "analyse frequency patterns" and it produces confident,
plausible, fabricated numbers — and backtesting won't catch it, because the
output *looks* like analysis.

| Job | Tool |
|---|---|
| PDF → structured questions | `parse.py` (pypdf) |
| Question → topic label | `tag.py` (**LLM**, constrained to taxonomy) |
| Counting, trends, forecasting | `models.py` / `backtest.py` (numpy) |
| Interpreting results | you, plus an LLM if useful |

---

## Pipeline

```bash
# 0. one-time
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt

# 1. drop PYQ PDFs in data/raw/, named <exam>_<year>[_<shift>].pdf
#    e.g. ssc_cgl_2023_shift2.pdf
./.venv/bin/python src/parse.py

# 2. tag topics with a local model
./.venv/bin/python src/tag.py --model deepseek-r1:8b

# 3. MEASURE THE TAGGER before trusting it
./.venv/bin/python src/eval_tagger.py --sample 200   # label gold_topic by hand
./.venv/bin/python src/eval_tagger.py

# 4. walk-forward backtest — which model actually has signal?
./.venv/bin/python src/backtest.py

# 5. forecast, using whichever model WON step 4
./.venv/bin/python src/predict.py --year 2026 --model <winner>
```

Validate the harness without any real data:

```bash
./.venv/bin/python src/synth.py --drift 0.06 --shifts-per-year 12 \
    --out data/tagged/synth.json
./.venv/bin/python src/backtest.py --papers data/tagged/synth.json
```

---

## What the backtest already told us

Run on synthetic data with known ground truth, before any real PYQs:

**1. Data volume dominates everything.** Going from 1 paper/year to 12 shifts/year:

| | MAE | top-10 hit rate |
|---|---|---|
| 1 paper/year | 1.05 | 0.50 |
| 12 shifts/year | **0.37** | **0.71** |

A single 100-question paper spreads 25 questions across ~18 topics — per-topic
counts of 1–2, where multinomial noise swamps any real trend. **Collect every
shift you can find.** This matters more than model choice, by a wide margin.

**2. The naive baseline is hard to beat.** "Average the last 3 years" won on
drifting data. Sophisticated models lost. That is a real result, not a bug — and
it's why `backtest.py` prints an explicit verdict telling you to ship the
baseline when nothing beats it.

**3. Recency weighting helps only under drift.** On stationary data the no-decay
variant won; under drift the aggressive-recency variant overtook it. The harness
detects the difference, which is the evidence that it measures something real.

---

## Why not quantum probability models

Quantum probability is genuine mathematics — used in quantum cognition research
for order and interference effects in *human judgment*. But there's no mechanism
connecting it to this data-generating process, which is mundane: a committee
picks questions from a fixed syllabus under rough weighting conventions.

That process is a multinomial with drifting proportions. A Dirichlet-multinomial
models it directly and gives calibrated intervals. Quantum formalism would add
free parameters without a mechanism — and extra parameters make backtests
*easier* to pass while making predictions *worse*, which is the failure mode
hardest to detect.

If a quantum-probability model ever beats `mean_last_3` in `backtest.py`, that's
evidence worth having. Until then it's complexity without signal.

---

## Honest limits

- **Tagger accuracy caps everything.** At 70% tagger accuracy, 30% of every count
  is wrong. Run `eval_tagger.py` first. Errors cluster (mensuration ↔ geometry),
  distorting exactly the comparisons you care about.
- **An 8B model is weak on Indian-exam-specific content.** Hand-label 200
  questions and measure. If accuracy is poor, collapse the taxonomy to coarser
  topics rather than pushing on.
- **Pattern changes break everything.** SSC revised the CGL pattern in 2022–23.
  History from before a pattern change may be actively misleading; consider
  cutting it.
- **Predicted counts of 1–2 are barely distinguishable from noise.** Trust the
  ranking of heavy topics; don't over-read the tail.

---

## Layout

```
src/parse.py         PDF → questions          data/raw/ → data/parsed/
src/tag.py           questions → topics       (Ollama)
src/eval_tagger.py   tagger accuracy vs hand-labelled gold set
src/models.py        baselines + Dirichlet-multinomial
src/metrics.py       MAE, RMSE, top-k hit, interval coverage, skill score
src/backtest.py      walk-forward evaluation + verdict
src/predict.py       forward forecast with 90% intervals
src/synth.py         synthetic papers for validating the harness
taxonomy/ssc_cgl.yaml
```

Adapt to UPSC / IB ACIO by adding a taxonomy YAML — nothing else is exam-specific.
