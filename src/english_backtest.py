"""Walk-forward backtest + model selection for the English section.

Drives the harness validated in the Reasoning run against english_tagged.json.
Reusing it deliberately: it already carries the fixes that took real debugging --
predictions scaled to each test shift's OBSERVED total (not the nominal 25), and
a randomization control that shuffles year labels within TRAINING only so the
test set composition cannot move.

2021 is absent from the corpus by instruction, so the walk-forward starts at
2024: train 2022-2023 -> predict 2024. That leaves ONE instrument-constant test
year, which is a genuine weakness and is reported as such rather than papered
over with the 2025 column.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from reasoning_backtest import (MODELS, load_shifts, randomization_control,
                                report, walk_forward)
from reasoning_stability import (leave_one_year_out, failure_map,
                                 paired_bootstrap, per_shift_mae)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
TAGGED = OUT / "english_tagged.json"


def run(level: str, test_years: list[int], official_only: bool, title: str):
    shifts, meta = load_shifts(TAGGED, level)
    if official_only:
        shifts = {k: v for k, v in shifts.items()
                  if not meta[k]["is_reconstruction"]}
    vocab = sorted({t for c in shifts.values() for t in c})
    print("\n" + "#" * 100)
    print(f"# {title}")
    print(f"# level={level}  shifts={len(shifts)}  vocab={len(vocab)}  test={test_years}")
    print("#" * 100)
    rows = walk_forward(shifts, meta, vocab, test_years)
    report(rows, test_years, title=f"{title} — MAE per {level}")
    return shifts, vocab, rows


def stability(shifts, vocab, test_years, label):
    scores = {}
    for name, fn in MODELS.items():
        m = per_shift_mae(shifts, vocab, test_years, fn)
        if m:
            scores[name] = m
    if not scores:
        print("INSUFFICIENT DATA")
        return None
    ranked = sorted(scores, key=lambda n: sum(scores[n].values()) / len(scores[n]))
    best, base = ranked[0], "mean_last3"
    print(f"\nPAIRED BOOTSTRAP vs {base} — {label}   (best={best})")
    print(f"{'comparison':<44}{'diff':>9}{'95% CI':>22}{'p':>8}{'n':>5}")
    print("-" * 100)
    for name in ranked[:6]:
        if name == base or base not in scores:
            continue
        obs, lo, hi, p, n = paired_bootstrap(scores[name], scores[base])
        if lo == hi == 0.0:
            v = "  identical to baseline"
        elif lo < 0 < hi:
            v = ""
        elif hi < 0:
            v = "  SIGNIFICANTLY BETTER"
        else:
            v = "  SIGNIFICANTLY WORSE"
        print(f"{name+' - '+base:<44}{obs:>+9.4f}"
              f"{f'[{lo:+.4f}, {hi:+.4f}]':>22}{p:>8.3f}{n:>5}{v}")

    print(f"\nLEAVE-ONE-TRAINING-YEAR-OUT — top 3")
    for drop, ranking in leave_one_year_out(shifts, vocab, test_years).items():
        print("  drop " + f"{drop:<6} " +
              "   ".join(f"{n}={m:.3f}" for m, n in ranking[:3]))

    fmap, n_sh = failure_map(shifts, vocab, test_years, MODELS[best])
    print(f"\nFAILURE MAP for '{best}' (+ = overpredicted), n={n_sh} shifts")
    for t, v in sorted(fmap.items(), key=lambda x: -abs(x[1]))[:10]:
        print(f"  {t:<32}{v:>+7.2f}  {'#'*int(min(abs(v)*20,36))}")
    return best


def main() -> None:
    shifts_all, meta = load_shifts(TAGGED, "family")
    print("Shifts by year:", dict(sorted(Counter(y for y, _, _ in shifts_all).items())))
    off = {k for k in shifts_all if not meta[k]["is_reconstruction"]}
    print(f"Official-source shifts: {len(off)} of {len(shifts_all)}")

    # PRIMARY: instrument constant (official only). Only 2024 is testable.
    s, v, _ = run("family", [2024], True,
                  "PRIMARY — official only, train 2022-23 -> test 2024")
    best_fam = stability(s, v, [2024], "family / official")

    s2, v2, _ = run("subtype", [2024], True,
                    "PRIMARY — official only, subtype level")
    best_sub = stability(s2, v2, [2024], "subtype / official")

    # Grammar micro-archetype level (the §8 rule-prediction target).
    s3, v3, _ = run("micro", [2024], True,
                    "PRIMARY — official only, grammar micro-archetype level")
    stability(s3, v3, [2024], "micro / official")

    # SECONDARY: includes 2025 reconstructions.
    print("\n" + "#" * 100)
    print("# SECONDARY — all sources incl. 2025 (93% reconstruction, terser text)")
    print("# 2025 is NOT instrument-comparable; see the Reasoning run where the")
    print("# same mismatch made a rejected model appear to win.")
    print("#" * 100)
    sa, va, _ = run("family", [2024, 2025], False, "All sources")
    stability(sa, va, [2024, 2025], "family / all sources")

    # Negative control on the primary configuration.
    print("\n" + "#" * 100)
    print("# RANDOMIZATION CONTROL — year labels shuffled within training only")
    print("#" * 100)
    rr = randomization_control(s, v, [2024])
    report(rr, [2024], title="Shuffled-year control")
    real = {n: sum(x["mae"] for x in _r.values()) / len(_r)
            for n, _r in walk_forward(s, meta, v, [2024]).items() if _r}
    rand = {n: sum(x["mae"] for x in _r.values()) / len(_r)
            for n, _r in rr.items() if _r}
    print("\nReal vs shuffled (a model that does not move was never using time):")
    for n in sorted(real, key=real.get):
        if n in rand:
            print(f"  {n:<20} real {real[n]:.3f}  shuffled {rand[n]:.3f}  "
                  f"degradation {100*(rand[n]-real[n])/real[n]:+.1f}%")

    json.dump({"best_family": best_fam, "best_subtype": best_sub},
              open(OUT / "english_model_selection.json", "w"), indent=2)


if __name__ == "__main__":
    main()
