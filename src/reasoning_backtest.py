"""Walk-forward backtest of Reasoning topic-mix forecasts, at shift level.

Protocol: train on years < T, predict T, score against every shift in T, advance.
No post-T information reaches a model.

Two scaling rules, and mixing them up invalidates everything (HANDOVER §4):
  * BACKTEST  -> scale predicted proportions to each test shift's OBSERVED
                 tagged total. Scaling to the nominal 25 would make the metric
                 measure tagger coverage rather than forecast skill.
  * FORECAST  -> scale to nominal 25, because the real 2026 paper has 25
                 Reasoning questions regardless of what the tagger reached.

A shift is one observation. Treating a year as one paper throws away the
within-year variance that dominates at these sample sizes.
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"

NOMINAL = 25


# --------------------------------------------------------------------------- io
def load_shifts(path: Path, level: str = "topic") -> dict:
    """Group tagged Reasoning questions into {(year, date, shift): Counter}."""
    recs = json.load(open(path))
    shifts: dict[tuple, Counter] = defaultdict(Counter)
    meta: dict[tuple, dict] = {}
    for r in recs:
        key = (r["year"], r["exam_date"], r["shift"])
        meta.setdefault(key, {"year": r["year"],
                              "is_reconstruction": r["is_reconstruction"]})
        lab = r.get(level)
        if lab:
            shifts[key][lab] += 1
    return {k: v for k, v in shifts.items() if sum(v.values()) > 0}, meta


def to_prop(c: Counter, vocab: list[str]) -> list[float]:
    n = sum(c.values())
    return [c.get(t, 0) / n for t in vocab] if n else [0.0] * len(vocab)


# ----------------------------------------------------------------------- models
def m_uniform(train, vocab, **kw):
    return [1 / len(vocab)] * len(vocab)


def _pooled(train, vocab, years=None):
    """Pool QUESTIONS across the selected years, then normalise.

    Pooling questions rather than averaging per-year proportions deliberately
    lets a year with more shifts carry more weight -- those years are better
    measured, and 2021 has only 3 shifts.
    """
    agg = Counter()
    for (y, _, _), c in train.items():
        if years is None or y in years:
            agg.update(c)
    return to_prop(agg, vocab)


def m_pooled_all(train, vocab, **kw):
    return _pooled(train, vocab)


def m_last_year(train, vocab, **kw):
    ys = sorted({y for y, _, _ in train})
    return _pooled(train, vocab, years={ys[-1]})


def m_mean_last2(train, vocab, **kw):
    ys = sorted({y for y, _, _ in train})
    return _pooled(train, vocab, years=set(ys[-2:]))


def m_mean_last3(train, vocab, **kw):
    ys = sorted({y for y, _, _ in train})
    return _pooled(train, vocab, years=set(ys[-3:]))


def _recency(train, vocab, half_life):
    """Exponentially weight each shift by the age of its year."""
    ys = sorted({y for y, _, _ in train})
    newest = ys[-1]
    agg = defaultdict(float)
    for (y, _, _), c in train.items():
        w = 0.5 ** ((newest - y) / half_life)
        for t, n in c.items():
            agg[t] += w * n
    tot = sum(agg.values())
    return [agg.get(t, 0) / tot if tot else 0 for t in vocab]


def m_recency_hl1(train, vocab, **kw):
    return _recency(train, vocab, 1.0)


def m_recency_hl2(train, vocab, **kw):
    return _recency(train, vocab, 2.0)


def _dirichlet(train, vocab, alpha):
    """Dirichlet-multinomial posterior mean, prior = uniform * alpha.

    Shrinks a topic seen zero times toward a small non-zero probability, which
    is the right behaviour here: an archetype absent from 50 shifts is rare, not
    impossible, and SSC demonstrably revives dormant archetypes.
    """
    agg = Counter()
    for c in train.values():
        agg.update(c)
    n = sum(agg.values())
    k = len(vocab)
    return [(agg.get(t, 0) + alpha) / (n + alpha * k) for t in vocab]


def m_dirichlet_a05(train, vocab, **kw):
    return _dirichlet(train, vocab, 0.5)


def m_dirichlet_a2(train, vocab, **kw):
    return _dirichlet(train, vocab, 2.0)


def m_dirichlet_recency(train, vocab, **kw):
    """Recency weighting plus Dirichlet smoothing -- the two ideas combined."""
    base = _recency(train, vocab, 2.0)
    n_eff = sum(sum(c.values()) for c in train.values())
    a = 0.5
    k = len(vocab)
    return [(p * n_eff + a) / (n_eff + a * k) for p in base]


def m_markov(train, vocab, **kw):
    """Predict from the most recent shift only, smoothed toward the pooled mix.

    A first-order transition model on 25-question shifts has almost no data per
    cell, so this is the honest weak form: last observed shift, heavily shrunk.
    Included to test whether recent-shift information adds anything at all.
    """
    latest = max(train, key=lambda k: (k[0], str(k[1]), k[2] or 0))
    last = to_prop(train[latest], vocab)
    pooled = _pooled(train, vocab)
    return [0.25 * a + 0.75 * b for a, b in zip(last, pooled)]


def m_hellinger(train, vocab, **kw):
    """Quantum-inspired: average state vectors in Hilbert space, not simplex.

    Encode each shift's topic mix as an amplitude vector psi = sqrt(p), average
    the amplitudes, then square back. This is the Bhattacharyya/Hellinger
    centroid rather than the arithmetic mean, and it is a genuine quantum-
    inspired construction (state-vector superposition) rather than the word
    "quantum" attached to a classical average.

    It is a real hypothesis, not decoration: sqrt compresses large proportions
    and lifts small ones, so if SSC's dormant archetypes matter more than raw
    frequency suggests, this should beat the arithmetic mean. If it does not, it
    gets rejected -- which is the whole point of testing it.
    """
    props = [to_prop(c, vocab) for c in train.values()]
    if not props:
        return m_uniform(train, vocab)
    amp = [0.0] * len(vocab)
    for p in props:
        for i, v in enumerate(p):
            amp[i] += math.sqrt(v)
    amp = [a / len(props) for a in amp]
    sq = [a * a for a in amp]
    tot = sum(sq)
    return [v / tot if tot else 0 for v in sq]


MODELS = {
    "uniform": m_uniform,
    "last_year": m_last_year,
    "mean_last2": m_mean_last2,
    "mean_last3": m_mean_last3,
    "pooled_all": m_pooled_all,
    "recency_hl1": m_recency_hl1,
    "recency_hl2": m_recency_hl2,
    "dirichlet_a0.5": m_dirichlet_a05,
    "dirichlet_a2": m_dirichlet_a2,
    "dirichlet_recency": m_dirichlet_recency,
    "markov_shrunk": m_markov,
    "quantum_hellinger": m_hellinger,
}


# ---------------------------------------------------------------------- metrics
def score(pred_prop, obs: Counter, vocab):
    """MAE and TVD against one shift, predictions scaled to OBSERVED total."""
    total = sum(obs.values())
    pred_counts = [p * total for p in pred_prop]
    obs_counts = [obs.get(t, 0) for t in vocab]
    mae = sum(abs(a - b) for a, b in zip(pred_counts, obs_counts)) / len(vocab)
    op = [c / total for c in obs_counts]
    tvd = sum(abs(a - b) for a, b in zip(pred_prop, op)) / 2
    return mae, tvd


def topk_recall(pred_prop, obs: Counter, vocab, k=5):
    ranked = [t for _, t in sorted(zip(pred_prop, vocab), reverse=True)][:k]
    actual = {t for t, _ in obs.most_common(k)}
    return len(set(ranked) & actual) / max(len(actual), 1)


# --------------------------------------------------------------------- backtest
def walk_forward(shifts, meta, vocab, test_years, label=""):
    rows = {}
    for name, fn in MODELS.items():
        per_year = {}
        for ty in test_years:
            train = {k: v for k, v in shifts.items() if k[0] < ty}
            test = {k: v for k, v in shifts.items() if k[0] == ty}
            if not train or not test:
                continue
            pred = fn(train, vocab)
            maes, tvds, recs = [], [], []
            for k, obs in test.items():
                mae, tvd = score(pred, obs, vocab)
                maes.append(mae)
                tvds.append(tvd)
                recs.append(topk_recall(pred, obs, vocab))
            per_year[ty] = dict(
                mae=sum(maes) / len(maes),
                tvd=sum(tvds) / len(tvds),
                top5=sum(recs) / len(recs),
                n_shifts=len(test),
            )
        rows[name] = per_year
    return rows


def report(rows, test_years, baseline="mean_last3", title=""):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    hdr = f"{'model':<20}"
    for ty in test_years:
        hdr += f"{ty:>12}"
    hdr += f"{'mean MAE':>11}{'mean TVD':>10}{'top5':>7}{'skill':>8}"
    print(hdr)
    print("-" * 100)

    base_mae = None
    if baseline in rows and rows[baseline]:
        base_mae = sum(v["mae"] for v in rows[baseline].values()) / len(rows[baseline])

    ordered = sorted(
        rows,
        key=lambda n: (sum(v["mae"] for v in rows[n].values()) / len(rows[n]))
        if rows[n] else 9e9)

    for name in ordered:
        pv = rows[name]
        if not pv:
            print(f"{name:<20}{'INSUFFICIENT DATA':>40}")
            continue
        line = f"{name:<20}"
        for ty in test_years:
            line += f"{pv[ty]['mae']:>12.3f}" if ty in pv else f"{'-':>12}"
        mm = sum(v["mae"] for v in pv.values()) / len(pv)
        mt = sum(v["tvd"] for v in pv.values()) / len(pv)
        m5 = sum(v["top5"] for v in pv.values()) / len(pv)
        line += f"{mm:>11.3f}{mt:>10.3f}{m5:>7.2f}"
        if base_mae:
            line += f"{100 * (base_mae - mm) / base_mae:>+7.1f}%"
        marker = "  <- baseline" if name == baseline else ""
        print(line + marker)
    return ordered


def randomization_control(shifts, vocab, test_years, seed=7):
    """Negative control: destroy temporal order in TRAINING only.

    A first attempt permuted year labels across the whole corpus and produced
    *better* scores than the real data (MAE 0.465 vs 0.612) -- which looked like
    the models having no signal, but was an artefact. Shuffling moved shifts
    between train and test, and 2021-22 shifts carry far fewer tagged questions
    per shift, so their smaller absolute counts mechanically shrink an absolute
    error metric. The control was measuring test-set composition, not time.

    Fixed form: hold each real test year fixed, and shuffle year labels only
    among the training shifts. The marginal training distribution is untouched,
    so `pooled_all` must score IDENTICALLY -- that invariance is the built-in
    proof the control is wired up correctly. Only recency-sensitive models should
    move, and if they do not, their apparent skill was never temporal.
    """
    rng = random.Random(seed)
    rows = {}
    for name, fn in MODELS.items():
        per_year = {}
        for ty in test_years:
            train = {k: v for k, v in shifts.items() if k[0] < ty}
            test = {k: v for k, v in shifts.items() if k[0] == ty}
            if not train or not test:
                continue
            keys = list(train)
            years = [k[0] for k in keys]
            rng.shuffle(years)
            scrambled = {(y,) + k[1:]: train[k] for y, k in zip(years, keys)}
            pred = fn(scrambled, vocab)
            maes, tvds, recs = [], [], []
            for k, obs in test.items():
                mae, tvd = score(pred, obs, vocab)
                maes.append(mae)
                tvds.append(tvd)
                recs.append(topk_recall(pred, obs, vocab))
            per_year[ty] = dict(mae=sum(maes) / len(maes),
                                tvd=sum(tvds) / len(tvds),
                                top5=sum(recs) / len(recs),
                                n_shifts=len(test))
        rows[name] = per_year
    return rows


def main() -> None:
    shifts, meta = load_shifts(OUT / "reasoning_tagged.json", "topic")
    vocab = sorted({t for c in shifts.values() for t in c})

    counts = Counter(y for y, _, _ in shifts)
    print("Shifts with >=1 tagged Reasoning question, by year:", dict(sorted(counts.items())))
    print(f"Topic vocabulary: {len(vocab)}")

    official = {k: v for k, v in shifts.items() if not meta[k]["is_reconstruction"]}
    print(f"Official-source shifts: {len(official)}  (of {len(shifts)})")

    # PRIMARY: official -> official, so the measurement instrument is constant.
    print("\n" + "#" * 100)
    print("# PRIMARY BACKTEST — official papers only (instrument held constant)")
    print("#" * 100)
    rows_off = walk_forward(official, meta, vocab, [2023, 2024])
    ordered = report(rows_off, [2023, 2024],
                     title="Official-only walk-forward (MAE per topic, scaled to observed totals)")

    # SECONDARY: includes the 2025 reconstruction year.
    print("\n" + "#" * 100)
    print("# SECONDARY BACKTEST — all sources, 2025 included")
    print("# NOTE: 2025 is 93% reconstruction with terser stems; the instrument")
    print("#       change alone contributes ~7% TVD, so 2025 columns are not")
    print("#       directly comparable to 2023/2024.")
    print("#" * 100)
    rows_all = walk_forward(shifts, meta, vocab, [2023, 2024, 2025])
    report(rows_all, [2023, 2024, 2025],
           title="All-source walk-forward (MAE per topic)")

    # Negative control.
    print("\n" + "#" * 100)
    print("# RANDOMIZATION CONTROL — year labels permuted")
    print("#" * 100)
    rows_rand = randomization_control(official, vocab, [2023, 2024])
    report(rows_rand, [2023, 2024], title="Shuffled-year control (skill should collapse)")

    real = {n: sum(v["mae"] for v in rows_off[n].values()) / len(rows_off[n])
            for n in rows_off if rows_off[n]}
    rand = {n: sum(v["mae"] for v in rows_rand[n].values()) / len(rows_rand[n])
            for n in rows_rand if rows_rand[n]}
    print("\nReal vs shuffled mean MAE (a model that barely moves is fitting the marginal):")
    for n in sorted(real, key=real.get):
        d = rand.get(n)
        if d:
            print(f"  {n:<20} real {real[n]:.3f}   shuffled {d:.3f}   "
                  f"degradation {100*(d-real[n])/real[n]:+.1f}%")

    json.dump({"official": rows_off, "all": rows_all, "shuffled": rows_rand,
               "vocab": vocab},
              open(OUT / "reasoning_backtest.json", "w"), indent=2)
    print(f"\nWrote {OUT/'reasoning_backtest.json'}")


if __name__ == "__main__":
    main()
