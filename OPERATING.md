# Operating the system — no maintainer required

The system runs itself. This file is for the days when you want to know what it
is doing, or when something looks wrong.

## What arrives, and when

**06:00 IST daily** — a PDF at `yaga1998ak@icloud.com`: 100 questions
(25 per section, official Tier-I order) plus a one-page General Awareness brief.

Three independent layers guarantee it:

1. **n8n** builds and sends it (`SSC-CGL-2026-DAILY`, 12 nodes)
2. **A 7-day buffer** of pre-validated papers covers a build failure
3. **A launchd watchdog** (07:30 / 12:00 / 18:00) delivers from the buffer if
   n8n itself is dead

n8n could be uninstalled and papers would still arrive for a week.
If a build fails you get an **email saying so**, with the reason and the
remaining buffer depth. You will not have to notice silence.

**Sunday 03:00 IST** — `SSC-CGL-2026-WEEKLY-REFRESH` re-reads the corpus,
re-runs the backtests, re-audits YAGA's beliefs, reclaims disk, and rebuilds
the buffer.

## Reading the paper honestly

The confidence page is not decoration. Measured on blind backtests:

| Layer | Measured |
|---|---|
| GA knowledge zones | **81.9%** covered-or-adjacent (624 held-out questions) |
| Topic ranking | 33.1% top-10 overlap |
| **Specific questions** | **0.00%** realised, 0.20% ceiling |

**Study the zones and the topic weights. Treat the 100 questions as practice,
not prophecy.** SSC writes fresh questions every shift — that is measured, not
assumed, and YAGA is built so no daily reading can override it.

Quant will usually show as over its 15-minute sectional budget. That is real:
the cheapest question set honouring the forecast costs ~915 s against a 900 s
limit. Treat quant as the section to triage.

## If something looks wrong

```bash
cd ~/Claude/pyq-predictor

tail -40 out/daily/run_$(date +%F).log     # what happened this morning
./.venv/bin/python src/yaga.py             # what the brain believes
./.venv/bin/python src/prebuild.py --status # buffer depth
./.venv/bin/python src/brain.py            # pools, runway, decisions
./.venv/bin/python src/model_evolution.py --status
```

Restart the scheduler if n8n is unresponsive:

```bash
pm2 restart n8n --update-env
```

## Feeding it

- Drop newspaper or magazine PDFs in `data/newspapers/` (or just leave them in
  `~/Downloads`). Both are swept every morning.
- New PYQ papers go in `data/raw/<year>/`. The Sunday refresh picks them up,
  guards out any Tier-II strays, re-tags and re-forecasts automatically.

## Extending it without code

Everything meant to evolve is **data, not code**:

| To add | Edit | Note |
|---|---|---|
| A hypothesis | `out/yaga_proposals.json` | `refuted_if` is mandatory |
| A model | `src/model_roster.py --add NAME PROVIDER MODEL --role probe` | probes are benchmarked, not trusted |
| An exam | `src/exam_profile.py` | one profile object |

A proposal that cannot say what would falsify it is rejected unexamined. A
model is never trusted for being new — it earns its role by measuring better.

## What YAGA is

A general, self-auditing model of exam question-setting teams. It knows nothing
about PDFs or file formats; it consumes `Observation` objects from
`src/evidence.py` and reasons about the people who write the exam. Swap the
evidence adapter and the same brain runs on another exam.

It can **lose** beliefs. Every CONFIRMED claim re-answers power, control and
constraint each pass or it is demoted, and `YAGA/Belief History.md` in Obsidian
is append-only — a belief held and then lost stays visible.

Currently confirmed: contractual topics, round-number habit, day-level
blueprint, anchoring, availability, satisficing, effort economy, serial
position, policy pressure, civic core, and that past topic rank predicts future
(ρ = 0.718).

Currently rejected: frame reuse across shifts. That rejection is why the system
forecasts topic weight and refuses to predict questions.

## Verified findings

`~/Documents/Obsidian/Project Yaga/`
- `YAGA/` — brain state, belief history, verified patterns, parameters
- `SSC CGL/Research/` — the four standing findings, each with what would
  overturn it
- `SSC CGL/Daily Predictions/` — a journal per paper

## Known limits

- **GA and English pools: ~58 days** of non-repeating papers. Generators cannot
  grow them; they need papers with published answer keys.
- **Figure-based reasoning** (mirror images, cube-and-dice, paper folding) is
  permanently unreachable — no extractable text exists in the sources.
- **Model-derived GA answers are refused** until a benchmark passes the 75%
  accuracy floor. The daily paper does not depend on them.
