"""2026 English blueprint, built from OFFICIAL papers only (2022-2024).

2025 is excluded from the English evidence base entirely, not merely
down-weighted. The reason is measured, not stylistic:

  TVD(2023 vs 2024, both official)          = 0.094
  TVD(2024 official vs 2025 reconstruction) = 0.356      -- ~4x larger

and the 2025 files contain families the real exam does not use at that rate:
reading_comprehension at 3.73/shift against ~0.34 in official papers, and
direct_indirect_speech at 2.11 against 0.00 in official 2023-2024. Their own
filenames say "T-I Similar Paper" -- coaching practice papers, not memory-based
reconstructions of real shifts. Pooling them would import invented composition
into the forecast.

Model selection (src/english_backtest.py, test year 2024, official only):
  * FAMILY level    -> pooled_all. Dirichlet smoothing and recency were both
    SIGNIFICANTLY WORSE (p=0.000, p=0.003). Families are dense and stable, so
    there are no zero cells for shrinkage to repair.
  * GRAMMAR MICRO-RULE level -> last_year. Significantly better than pooling
    (-0.0682 MAE, 95% CI [-0.1285, -0.0024], p=0.044), with four more
    recency-flavoured models also significant. Which rule SSC emphasises drifts
    year to year, so recent data beats pooled data.

That is the mirror image of the Reasoning result, where shrinkage helped and
recency hurt -- and it is explainable: sparsity favours smoothing, drift favours
recency.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from reasoning_backtest import load_shifts
from reasoning_forecast import predictive_interval

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
NOMINAL = 25

# Kept deliberately small. pooled_all won at family level and dirichlet_a0.5 was
# significantly worse, so alpha exists only to make the posterior proper enough
# to draw intervals from; it must not shift the point estimate materially.
ALPHA_FAMILY = 0.1


def blueprint(level: str, alpha: float, years=None):
    shifts, meta = load_shifts(OUT / "english_tagged.json", level)
    shifts = {k: v for k, v in shifts.items() if not meta[k]["is_reconstruction"]}
    if years:
        shifts = {k: v for k, v in shifts.items() if k[0] in years}
    vocab = sorted({t for c in shifts.values() for t in c})
    iv = predictive_interval(shifts, vocab, alpha=alpha, total=NOMINAL)
    return iv, len(shifts), vocab


def show(iv, n_shifts, level, note):
    print("\n" + "=" * 92)
    print(f"2026 ENGLISH BLUEPRINT — {level}   ({n_shifts} official shifts)")
    print(note)
    print("=" * 92)
    print(f"{'family' if level=='family' else level:<28}"
          f"{'expected':>9}{'90% range':>12}{'P(appears)':>12}")
    print("-" * 92)
    tot = 0.0
    for t in sorted(iv, key=lambda x: -iv[x]["expected"]):
        d = iv[t]
        tot += d["expected"]
        rng = f"{d['p5']}-{d['p95']}"
        print(f"{t:<28}{d['expected']:>9.2f}{rng:>12}{d['p_appears']:>11.0%}")
    print("-" * 92)
    print(f"{'TOTAL':<28}{tot:>9.2f}")


def main() -> None:
    fam, n, _ = blueprint("family", ALPHA_FAMILY)
    show(fam, n, "family",
         "Model: pooled_all (official 2022-2024). Scaled to the nominal 25.")

    sub, n2, _ = blueprint("subtype", ALPHA_FAMILY)
    show(sub, n2, "subtype", "Model: pooled_all (official 2022-2024).")

    # Grammar rules use last_year, i.e. 2024 alone -- the validated winner.
    mic, n3, _ = blueprint("micro", ALPHA_FAMILY, years={2024})
    show(mic, n3, "micro",
         "Model: last_year = 2024 only (validated: -0.0682 MAE vs pooling, p=0.044).\n"
         "Shares are of GRAMMAR-BEARING questions, not of all 25.")

    json.dump({"family": fam, "subtype": sub, "grammar_micro": mic,
               "model_family": "pooled_all", "model_micro": "last_year",
               "excluded": "2021 (quality), 2025 (instrument: TVD 0.356)"},
              open(OUT / "english_forecast_2026.json", "w"), indent=2)
    print(f"\nWrote {OUT/'english_forecast_2026.json'}")


if __name__ == "__main__":
    main()
