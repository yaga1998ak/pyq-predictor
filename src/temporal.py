"""What five years of papers reveal that any single year cannot.

A snapshot answers "what does the exam look like". A time series answers
questions that matter more for a 2026 forecast:

  T1 DRIFT       Is the exam moving in a consistent direction, or just
                 wobbling? Directional drift means averaging the last five
                 years is the WRONG estimator - you would be averaging over a
                 trend and landing behind it.
  T2 ACCELERATION Is year-to-year change growing? Accelerating divergence is
                 the observable signature of a regime change - and the only
                 evidence that would support the "anti-prediction" hypothesis.
                 Flat or falling divergence rejects it.
  T3 TRAJECTORY  Which topics are rising, which are dying? A topic with a
                 consistent slope is predictable in a way its mean is not.
  T4 VOLATILITY  Which topics are stable enough to trust? A high-mean topic
                 that swings wildly is worth less than a lower-mean stable one.

Every number is deterministic numpy over the tagged corpus.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TAGGED = ROOT / "data" / "tagged" / "rules.json"
OUT = ROOT / "out"
# Extended to the full corpus after OCR recovered 2018-19 (they were
# image-only: 100 markers, no extractable prose). Coverage is now
# 84-95% across every year except 2021, which the splitter still
# over-fragments.
YEARS = [2018, 2019, 2021, 2022, 2023, 2024, 2025]


def load() -> dict[int, Counter]:
    papers = json.loads(TAGGED.read_text())
    per_year: dict[int, Counter] = defaultdict(Counter)
    n_papers: Counter = Counter()
    for p in papers:
        y = p.get("year")
        if y not in YEARS:
            continue
        c = Counter(q["topic"] for q in p.get("questions", []) if q.get("topic"))
        if sum(c.values()) >= 20:
            per_year[y].update(c)
            n_papers[y] += 1
    return per_year, n_papers


def props(c: Counter) -> dict[str, float]:
    t = sum(c.values()) or 1
    return {k: v / t for k, v in c.items()}


def tvd(a: dict, b: dict) -> float:
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0) - b.get(k, 0)) for k in keys)


def main() -> None:
    per_year, n_papers = load()
    years = [y for y in YEARS if y in per_year]
    P = {y: props(per_year[y]) for y in years}

    print("=" * 70)
    print("TEMPORAL ANALYSIS — what five years show that one cannot")
    print("=" * 70)
    for y in years:
        print(f"  {y}: {n_papers[y]:>2} papers, {sum(per_year[y].values()):>5} tagged Qs")

    # ---- T1 / T2 : drift and acceleration --------------------------------
    print("\n" + "-" * 70)
    print("T1/T2  DRIFT AND ACCELERATION")
    print("-" * 70)
    consec = [(years[i], years[i+1], tvd(P[years[i]], P[years[i+1]]))
              for i in range(len(years)-1)]
    for a, b, d in consec:
        print(f"  TVD {a} -> {b}: {d:.3f}")
    ds = np.array([d for _, _, d in consec])

    # distance from the first year: growing => directional drift
    from_first = [(y, tvd(P[years[0]], P[y])) for y in years[1:]]
    print()
    for y, d in from_first:
        print(f"  TVD {years[0]} -> {y}: {d:.3f}")
    fd = np.array([d for _, d in from_first])

    monotone = bool(np.all(np.diff(fd) > 0))
    drift_verdict = ("DIRECTIONAL DRIFT - distance from 2021 grows every year; "
                     "averaging all years lands behind the trend"
                     if monotone else
                     "NON-DIRECTIONAL - the exam wobbles around a stable centre; "
                     "averaging is the right estimator and recency weighting is not")
    print(f"\n  => {drift_verdict}")

    slope = float(np.polyfit(range(len(ds)), ds, 1)[0]) if len(ds) > 2 else 0.0
    accel_verdict = ("ACCELERATING - consecutive divergence is growing; this is "
                     "the signature a regime change would leave"
                     if slope > 0.01 else
                     "NOT ACCELERATING - consecutive divergence is flat or falling. "
                     "The anti-prediction / arms-race hypothesis predicts the "
                     "opposite, so this corpus does not support it")
    print(f"  consecutive-TVD slope: {slope:+.4f}")
    print(f"  => {accel_verdict}")

    # ---- T3 / T4 : per-topic trajectory and volatility --------------------
    print("\n" + "-" * 70)
    print("T3/T4  PER-TOPIC TRAJECTORY (per 100-question paper)")
    print("-" * 70)
    topics = sorted({t for y in years for t in P[y]})
    rows = []
    x = np.arange(len(years))
    for t in topics:
        ys = np.array([100 * P[y].get(t, 0.0) for y in years])
        if ys.mean() < 0.4:
            continue
        m, _ = np.polyfit(x, ys, 1)
        rows.append((t, ys.mean(), float(m), float(ys.std())))

    rising = sorted([r for r in rows if r[2] > 0.10], key=lambda r: -r[2])[:6]
    dying = sorted([r for r in rows if r[2] < -0.10], key=lambda r: r[2])[:6]
    stable = sorted([r for r in rows if abs(r[2]) <= 0.10],
                    key=lambda r: r[3])[:6]

    def show(title, rs):
        print(f"\n  {title}")
        for t, mean, sl, sd in rs:
            print(f"    {t:<32} mean {mean:5.2f}  slope {sl:+.2f}/yr  sd {sd:4.2f}")

    show("RISING — weight these above their historical mean:", rising)
    show("DECLINING — weight below their mean:", dying)
    show("MOST STABLE — the invariants worth trusting:", stable)

    res = {"years": years, "consecutive_tvd": [d for _, _, d in consec],
           "tvd_from_first": [d for _, d in from_first],
           "monotone_drift": monotone, "accel_slope": slope,
           "drift_verdict": drift_verdict, "accel_verdict": accel_verdict,
           "rising": [{"topic": t, "mean": m, "slope": s} for t, m, s, _ in rising],
           "declining": [{"topic": t, "mean": m, "slope": s} for t, m, s, _ in dying],
           "stable": [{"topic": t, "mean": m, "sd": d} for t, m, _, d in stable]}
    (OUT / "temporal.json").write_text(json.dumps(res, indent=2))
    print(f"\nwritten -> out/temporal.json")


if __name__ == "__main__":
    main()
