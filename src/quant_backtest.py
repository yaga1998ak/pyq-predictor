"""Walk-forward backtest, model selection and 2026 blueprint for Quant.

Drives the harness validated in the Reasoning and English runs against
quant_tagged.json, so it inherits the two fixes that cost real debugging there:
predictions scaled to each test shift's OBSERVED total (never the nominal 25),
and a randomization control that shuffles year labels within TRAINING only.

Evidence base: OFFICIAL 2022-2024 (2021 dropped at 38% blank stems; 2025
excluded as coaching-generated -- 0.356 family-TVD in the English run and ZERO
DI sets here, against ~1 per shift in official papers).

That leaves ONE instrument-constant test year (2024, train 2022-2023). Stated
plainly rather than padded with a 2025 column that measures the wrong thing.

Two label populations are reported separately and never silently merged:
  stem      -- topic read from extracted question text
  recovered -- topic inferred from OPTION STRUCTURE on questions whose text was
               lost to an embedded image. Coarse, and the correction it applies
               to the blueprint is shown as a delta so its size is visible.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from reasoning_backtest import (MODELS, load_shifts, randomization_control,
                                report, walk_forward)
from reasoning_forecast import predictive_interval
from reasoning_stability import (failure_map, leave_one_year_out,
                                 paired_bootstrap, per_shift_mae)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
TAGGED = OUT / "quant_tagged.json"
NOMINAL = 25
ALPHA = 0.1


def load(level: str, official_only=True, include_recovered=False):
    """Group into shift Counters, controlling which label population is used."""
    recs = json.load(open(TAGGED))
    keep = []
    for r in recs:
        if official_only and r["is_reconstruction"]:
            continue
        if not r.get(level):
            continue
        if r.get("label_source") == "recovered" and not include_recovered:
            continue
        keep.append(r)
    tmp = OUT / f"_quant_tmp_{level}_{int(include_recovered)}.json"
    tmp.write_text(json.dumps(keep))
    shifts, meta = load_shifts(tmp, level)
    tmp.unlink()
    return shifts, meta


def stability(shifts, vocab, test_years, label):
    scores = {n: m for n, fn in MODELS.items()
              if (m := per_shift_mae(shifts, vocab, test_years, fn))}
    if not scores:
        print("  INSUFFICIENT DATA")
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
    print("\nLEAVE-ONE-TRAINING-YEAR-OUT — top 3")
    for drop, ranking in leave_one_year_out(shifts, vocab, test_years).items():
        print(f"  drop {drop:<6} " +
              "   ".join(f"{n}={m:.3f}" for m, n in ranking[:3]))
    fmap, n_sh = failure_map(shifts, vocab, test_years, MODELS[best])
    print(f"\nFAILURE MAP for '{best}' (+ = overpredicted), n={n_sh} shifts")
    for t, v in sorted(fmap.items(), key=lambda x: -abs(x[1]))[:10]:
        print(f"  {t:<30}{v:>+7.2f}  {'#'*int(min(abs(v)*20, 36))}")
    return best


def run(level, test_years, include_recovered, title):
    shifts, meta = load(level, True, include_recovered)
    vocab = sorted({t for c in shifts.values() for t in c})
    print("\n" + "#" * 100)
    print(f"# {title}")
    print(f"# level={level}  shifts={len(shifts)}  vocab={len(vocab)}  "
          f"recovered_labels={'in' if include_recovered else 'out'}")
    print("#" * 100)
    rows = walk_forward(shifts, meta, vocab, test_years)
    report(rows, test_years, title=f"{title} — MAE per {level}")
    return shifts, vocab, meta


def blueprint(level, include_recovered):
    shifts, meta = load(level, True, include_recovered)
    vocab = sorted({t for c in shifts.values() for t in c})
    return predictive_interval(shifts, vocab, alpha=ALPHA, total=NOMINAL), len(shifts)


def main() -> None:
    s, v, meta = run("topic", [2024], False,
                     "PRIMARY — official 2022-24, stem labels only")
    best = stability(s, v, [2024], "topic / stem-only")

    s2, v2, _ = run("subtopic", [2024], False,
                    "PRIMARY — subtopic (microtopic) level")
    stability(s2, v2, [2024], "subtopic / stem-only")

    print("\n" + "#" * 100)
    print("# RANDOMIZATION CONTROL — year labels shuffled within training only")
    print("#" * 100)
    rr = randomization_control(s, v, [2024])
    report(rr, [2024], title="Shuffled-year control")
    real = {n: sum(x["mae"] for x in r.values()) / len(r)
            for n, r in walk_forward(s, meta, v, [2024]).items() if r}
    rand = {n: sum(x["mae"] for x in r.values()) / len(r)
            for n, r in rr.items() if r}
    print("\nReal vs shuffled (a model that does not move never used time):")
    for n in sorted(real, key=real.get):
        if n in rand:
            print(f"  {n:<20} real {real[n]:.3f}  shuffled {rand[n]:.3f}  "
                  f"degradation {100*(rand[n]-real[n])/real[n]:+.1f}%")

    # ---- blueprint, with and without the image-loss correction --------------
    bp_stem, n1 = blueprint("topic", False)
    bp_corr, n2 = blueprint("topic", True)
    print("\n" + "=" * 100)
    print(f"2026 QUANT BLUEPRINT — official 2022-2024 ({n1} shifts)")
    print("Scaled to the nominal 25. 'corrected' folds in option-recovered")
    print("labels for questions whose text was lost to embedded images.")
    print("=" * 100)
    print(f"{'topic':<28}{'stem-only':>11}{'corrected':>11}{'delta':>8}"
          f"{'90% range':>12}{'P(app)':>9}")
    print("-" * 100)
    tot_s = tot_c = 0.0
    for t in sorted(bp_corr, key=lambda x: -bp_corr[x]["expected"]):
        a = bp_stem.get(t, {}).get("expected", 0.0)
        d = bp_corr[t]
        tot_s += a
        tot_c += d["expected"]
        rng = f"{d['p5']}-{d['p95']}"
        print(f"{t:<28}{a:>11.2f}{d['expected']:>11.2f}"
              f"{d['expected']-a:>+8.2f}{rng:>12}{d['p_appears']:>8.0%}")
    print("-" * 100)
    print(f"{'TOTAL':<28}{tot_s:>11.2f}{tot_c:>11.2f}")

    bp_sub, _ = blueprint("subtopic", True)
    print("\n2026 MICROTOPIC BLUEPRINT (corrected) — top 18")
    print(f"{'microtopic':<32}{'expected':>10}{'90% range':>12}{'P(app)':>9}")
    print("-" * 100)
    for t in sorted(bp_sub, key=lambda x: -bp_sub[x]["expected"])[:18]:
        d = bp_sub[t]
        print(f"{t:<32}{d['expected']:>10.2f}"
              f"{f'{d[chr(112)+chr(53)]}-{d[chr(112)+chr(57)+chr(53)]}':>12}"
              f"{d['p_appears']:>8.0%}")

    json.dump({"topic_stem_only": bp_stem, "topic_corrected": bp_corr,
               "subtopic_corrected": bp_sub, "best_model": best,
               "n_shifts": n1,
               "excluded": "2021 (38% blank), 2025 (coaching-generated, 0 DI)"},
              open(OUT / "quant_forecast_2026.json", "w"), indent=2)
    print(f"\nWrote {OUT/'quant_forecast_2026.json'}")


if __name__ == "__main__":
    main()
