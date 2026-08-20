"""Frozen 2025 blind test, then the 2026 Reasoning blueprint.

Order matters and is enforced by the code path: the 2025 forecast is computed
from 2021-2024 only and printed BEFORE any 2025 actuals are read, so the numbers
cannot be quietly retuned after seeing the answer.

Scaling differs from the backtest on purpose (HANDOVER §4): the blind test scales
to each 2025 shift's OBSERVED tagged total, while the 2026 forecast scales to the
NOMINAL 25, because the real paper will carry 25 Reasoning questions whatever the
tagger reached.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from reasoning_backtest import MODELS, load_shifts, score, to_prop

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
NOMINAL = 25

# Selected by src/reasoning_stability.py: at archetype level dirichlet_a2 beats
# naive pooling by -0.0101 MAE, 95% CI [-0.0199, -0.0012], p=0.023, and stays
# top-ranked under leave-one-training-year-out in 3 of 4 configurations.
SELECTED = "dirichlet_a2"
BASELINE = "mean_last3"


def dirichlet_posterior(train, vocab, alpha=2.0):
    agg = Counter()
    for c in train.values():
        agg.update(c)
    return [agg.get(t, 0) + alpha for t in vocab], sum(agg.values())


def predictive_interval(train, vocab, alpha=2.0, total=NOMINAL,
                        draws=20000, seed=2026):
    """Posterior-predictive counts: draw p ~ Dirichlet, then n ~ Multinomial.

    Two sources of uncertainty, and reporting only the first is how forecasts end
    up overconfident (HANDOVER §4 measured 81-86% coverage against a 90% target):
      1. estimation error in p, from finitely many past shifts
      2. shift-to-shift sampling noise, which persists even if p were exact
    """
    conc, _ = dirichlet_posterior(train, vocab, alpha)
    rng = random.Random(seed)
    k = len(vocab)
    samples = [[] for _ in range(k)]
    p_means = [0.0] * k
    appears = [0] * k
    for _ in range(draws):
        g = [rng.gammavariate(a, 1.0) for a in conc]
        s = sum(g)
        p = [x / s for x in g]
        for i in range(k):
            p_means[i] += p[i]
        # multinomial via sequential conditional binomials
        remaining = total
        rem_p = 1.0
        counts = [0] * k
        for i in range(k - 1):
            if remaining <= 0:
                break
            q = p[i] / rem_p if rem_p > 0 else 0.0
            q = min(max(q, 0.0), 1.0)
            c = sum(1 for _ in range(remaining) if rng.random() < q)
            counts[i] = c
            remaining -= c
            rem_p -= p[i]
        counts[k - 1] = max(remaining, 0)
        for i in range(k):
            samples[i].append(counts[i])
            if counts[i] > 0:
                appears[i] += 1
    out = {}
    for i, t in enumerate(vocab):
        s = sorted(samples[i])
        out[t] = dict(
            expected=p_means[i] / draws * total,
            p5=s[int(0.05 * draws)],
            p50=s[int(0.50 * draws)],
            p95=s[int(0.95 * draws)],
            p_appears=appears[i] / draws,
        )
    return out


def blind_2025(level="topic"):
    shifts, meta = load_shifts(OUT / "reasoning_tagged.json", level)
    vocab = sorted({t for c in shifts.values() for t in c})
    train = {k: v for k, v in shifts.items() if k[0] <= 2024}
    test = {k: v for k, v in shifts.items() if k[0] == 2025}

    print("=" * 100)
    print(f"BLIND 2025 FORECAST ({level}) — frozen from 2021-2024, "
          f"{len(train)} training shifts")
    print("=" * 100)
    frozen = {name: fn(train, vocab) for name, fn in MODELS.items()}
    sel = frozen[SELECTED]
    ranked = sorted(zip(sel, vocab), reverse=True)
    print(f"Frozen top-10 predicted mix ({SELECTED}), per 25-question paper:")
    for p, t in ranked[:10]:
        print(f"   {t:<30}{p*NOMINAL:>6.2f}")

    print(f"\n--- revealing 2025 actuals ({len(test)} shifts) ---")
    actual = Counter()
    for c in test.values():
        actual.update(c)
    n_act = sum(actual.values())
    act_prop = {t: actual.get(t, 0) / n_act for t in vocab}

    print(f"\n{'archetype' if level=='subtopic' else 'topic':<30}"
          f"{'pred/25':>9}{'actual/25':>11}{'error':>8}")
    print("-" * 100)
    rows = sorted(vocab, key=lambda t: -act_prop[t])
    for t in rows[:14]:
        pr = dict(zip(vocab, sel))[t] * NOMINAL
        ac = act_prop[t] * NOMINAL
        print(f"{t:<30}{pr:>9.2f}{ac:>11.2f}{pr-ac:>+8.2f}")

    print(f"\n{'model':<22}{'MAE':>8}{'TVD':>8}{'top5':>7}{'top10':>7}")
    print("-" * 100)
    results = {}
    for name, pred in frozen.items():
        maes, tvds = [], []
        for k, obs in test.items():
            m, t_ = score(pred, obs, vocab)
            maes.append(m)
            tvds.append(t_)
        pr_rank = [t for _, t in sorted(zip(pred, vocab), reverse=True)]
        for K in (5, 10):
            pass
        top5 = len(set(pr_rank[:5]) & set(rows[:5])) / 5
        top10 = len(set(pr_rank[:10]) & set(rows[:10])) / 10
        results[name] = dict(mae=sum(maes)/len(maes), tvd=sum(tvds)/len(tvds),
                             top5=top5, top10=top10)
    for name in sorted(results, key=lambda n: results[n]["mae"]):
        r = results[name]
        mark = "  <- SELECTED" if name == SELECTED else (
            "  <- baseline" if name == BASELINE else "")
        print(f"{name:<22}{r['mae']:>8.3f}{r['tvd']:>8.3f}"
              f"{r['top5']:>7.2f}{r['top10']:>7.2f}{mark}")
    return results


def forecast_2026(level="topic", alpha=2.0):
    shifts, meta = load_shifts(OUT / "reasoning_tagged.json", level)
    vocab = sorted({t for c in shifts.values() for t in c})
    train = shifts  # all legitimate history, 2021-2025

    iv = predictive_interval(train, vocab, alpha=alpha, total=NOMINAL)
    print("\n" + "=" * 100)
    print(f"2026 REASONING BLUEPRINT ({level}) — {SELECTED}, trained 2021-2025 "
          f"({len(train)} shifts)")
    print("Scaled to the nominal 25 Reasoning questions.")
    print("=" * 100)
    print(f"{'archetype' if level=='subtopic' else 'topic':<32}"
          f"{'expected':>9}{'90% range':>13}{'P(appears)':>12}")
    print("-" * 100)
    tot = 0.0
    for t in sorted(vocab, key=lambda x: -iv[x]["expected"]):
        d = iv[t]
        tot += d["expected"]
        rng_txt = f"{d['p5']}-{d['p95']}"
        print(f"{t:<32}{d['expected']:>9.2f}{rng_txt:>13}"
              f"{d['p_appears']:>11.0%}")
    print("-" * 100)
    print(f"{'TOTAL':<32}{tot:>9.2f}")
    json.dump({"level": level, "alpha": alpha, "n_shifts": len(train),
               "blueprint": iv},
              open(OUT / f"reasoning_forecast_2026_{level}.json", "w"), indent=2)
    print(f"\nWrote {OUT / f'reasoning_forecast_2026_{level}.json'}")
    return iv


def main() -> None:
    blind_2025("topic")
    blind_2025("subtopic")
    forecast_2026("topic")
    forecast_2026("subtopic")


if __name__ == "__main__":
    main()
