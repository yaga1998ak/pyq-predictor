"""Is the winning margin real? Paired bootstrap + leave-one-year-out + failure map.

The bakeoff separates the best model from the naive baseline by well under 1%
across 2 test years. That is exactly the situation where a scorecard over-claims
(HANDOVER §9: "a verdict threshold without a sample-size check over-claims"), so
this module asks the only question that matters: does the margin survive
resampling?

Paired design: both models are scored on the SAME shifts, so the per-shift
difference removes shift difficulty entirely and the bootstrap only has to
resample the differences.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from reasoning_backtest import (MODELS, load_shifts, score, to_prop,
                                topk_recall)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"


def per_shift_mae(shifts, vocab, test_years, model_fn):
    """MAE for every test shift, keyed so two models can be paired exactly."""
    out = {}
    for ty in test_years:
        train = {k: v for k, v in shifts.items() if k[0] < ty}
        test = {k: v for k, v in shifts.items() if k[0] == ty}
        if not train or not test:
            continue
        pred = model_fn(train, vocab)
        for k, obs in test.items():
            out[k] = score(pred, obs, vocab)[0]
    return out


def paired_bootstrap(a: dict, b: dict, n=20000, seed=11):
    """95% CI on mean(a) - mean(b) over shared shifts."""
    keys = sorted(set(a) & set(b), key=str)
    diffs = [a[k] - b[k] for k in keys]
    n_obs = len(diffs)
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        s = sum(diffs[rng.randrange(n_obs)] for _ in range(n_obs))
        means.append(s / n_obs)
    means.sort()
    obs = sum(diffs) / n_obs
    lo = means[int(0.025 * n)]
    hi = means[int(0.975 * n)]
    # two-sided bootstrap p-value for "no difference"
    p = 2 * min(sum(1 for m in means if m >= 0) / n,
                sum(1 for m in means if m <= 0) / n)
    return obs, lo, hi, min(p, 1.0), n_obs


def leave_one_year_out(shifts, vocab, test_years):
    """Re-rank models with one training year removed at a time.

    A model that only wins when a particular year is present has not earned the
    top slot -- it has fitted that year.
    """
    train_years = sorted({y for y, _, _ in shifts if y < max(test_years)})
    results = {}
    for drop in [None] + train_years:
        sub = {k: v for k, v in shifts.items() if k[0] != drop}
        ranking = []
        for name, fn in MODELS.items():
            maes = per_shift_mae(sub, vocab, test_years, fn)
            if maes:
                ranking.append((sum(maes.values()) / len(maes), name))
        ranking.sort()
        results["none" if drop is None else str(drop)] = ranking
    return results


def failure_map(shifts, vocab, test_years, model_fn):
    """Signed per-topic error: where does the winning model actually fail?"""
    over = defaultdict(float)
    n_shifts = 0
    for ty in test_years:
        train = {k: v for k, v in shifts.items() if k[0] < ty}
        test = {k: v for k, v in shifts.items() if k[0] == ty}
        if not train or not test:
            continue
        pred = model_fn(train, vocab)
        for k, obs in test.items():
            total = sum(obs.values())
            n_shifts += 1
            for i, t in enumerate(vocab):
                over[t] += pred[i] * total - obs.get(t, 0)
    return {t: v / n_shifts for t, v in over.items()}, n_shifts


def run(level: str, test_years: list[int], official_only: bool):
    shifts, meta = load_shifts(OUT / "reasoning_tagged.json", level)
    if official_only:
        shifts = {k: v for k, v in shifts.items() if not meta[k]["is_reconstruction"]}
    vocab = sorted({t for c in shifts.values() for t in c})

    tag = f"level={level}  official_only={official_only}  test={test_years}"
    print("\n" + "=" * 100)
    print(f"STABILITY — {tag}")
    print(f"vocabulary={len(vocab)}  shifts={len(shifts)}")
    print("=" * 100)

    scores = {}
    for name, fn in MODELS.items():
        m = per_shift_mae(shifts, vocab, test_years, fn)
        if m:
            scores[name] = m
    ranked = sorted(scores, key=lambda n: sum(scores[n].values()) / len(scores[n]))
    best, base = ranked[0], "mean_last3"

    print(f"best={best}   baseline={base}\n")
    print(f"{'comparison':<44}{'diff':>9}{'95% CI':>22}{'p':>8}{'n':>5}")
    print("-" * 100)
    for name in ranked[:6]:
        if name == base:
            continue
        obs, lo, hi, p, n = paired_bootstrap(scores[name], scores[base])
        # A degenerate CI of [0, 0] means the two models produced IDENTICAL
        # predictions -- the opposite of a significant difference. Testing only
        # `lo < 0 < hi` mislabelled that case as SIGNIFICANT.
        if lo == hi == 0.0:
            verdict = "  identical to baseline"
        elif lo < 0 < hi:
            verdict = ""
        elif hi < 0:
            verdict = "  SIGNIFICANTLY BETTER"
        else:
            verdict = "  SIGNIFICANTLY WORSE"
        print(f"{name + ' - ' + base:<44}{obs:>+9.4f}"
              f"{f'[{lo:+.4f}, {hi:+.4f}]':>22}{p:>8.3f}{n:>5}{verdict}")
    print("\n(negative diff = better than baseline; CI spanning 0 = indistinguishable)")

    print("\nLEAVE-ONE-TRAINING-YEAR-OUT — top 3 by MAE")
    print("-" * 100)
    lo_out = leave_one_year_out(shifts, vocab, test_years)
    for drop, ranking in lo_out.items():
        top = "   ".join(f"{n}={m:.3f}" for m, n in ranking[:3])
        print(f"  drop {drop:<6} {top}")

    fmap, n_sh = failure_map(shifts, vocab, test_years, MODELS[best])
    print(f"\nFAILURE MAP for '{best}' — signed mean error per shift "
          f"(+ = overpredicted), n={n_sh}")
    print("-" * 100)
    for t, v in sorted(fmap.items(), key=lambda x: -abs(x[1]))[:12]:
        bar = "#" * int(min(abs(v) * 20, 40))
        print(f"  {t:<30}{v:>+7.2f}  {bar}")
    return ranked, scores


def main() -> None:
    run("topic", [2023, 2024], official_only=True)
    run("subtopic", [2023, 2024], official_only=True)
    run("topic", [2023, 2024, 2025], official_only=False)


if __name__ == "__main__":
    main()
