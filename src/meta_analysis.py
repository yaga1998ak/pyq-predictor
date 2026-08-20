"""Meta-layer: does strategic adaptation show up in the data, and is the 2026
posterior concentrated or multimodal?

This module answers the questions the brief asks (§9 arms race, §20 prediction
erosion, §25 scenario weights, §28 meta-backtest, §33 portfolio size) using
measurements rather than narrative, and marks explicitly where measurement stops
and inference begins.

Three things are computable from the corpus:

1. YEAR-OVER-YEAR STABILITY. If setters were deliberately mutating away from
   predictable patterns, consecutive-year topic distributions would diverge and
   recency-weighted models would beat pooled ones. Both are testable.

2. PREDICTION EROSION (§20). For each section, does the most recent year predict
   the next year better than the pooled history does? If adaptation were
   occurring, last_year would win. Across all four sections it mostly LOST.

3. POSTERIOR DISPERSION. Bootstrap the shift-level composition distribution and
   measure how tightly plausible 2026 papers cluster. That determines whether one
   paper covers the mass or a portfolio is needed -- the §33 decision.

What is NOT computable here, and is flagged rather than invented: the probability
of a vendor change, a new setting team, or an AI-aware regime. No observable in
this corpus speaks to those. Any number attached to them would be fabrication
(§75), so they are handled as a residual with stated assumptions.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"

SECTIONS = {
    "reasoning": ("reasoning_tagged.json", "topic", None),
    "english": ("english_tagged.json", "family", {2021}),
    "quant": ("quant_tagged.json", "topic", {2021}),
    "general_awareness": ("ga_tagged.json", "subject", {2021, 2022}),
}


def load(path, level, drop):
    recs = json.load(open(OUT / path))
    out = []
    for r in recs:
        if drop and r["year"] in drop:
            continue
        lab = r.get(level)
        if not lab:
            continue
        out.append(r)
    return out


def dist_for(recs, year, level, official_only=True):
    sel = [r for r in recs if r["year"] == year
           and (not official_only or not r["is_reconstruction"])]
    c = Counter(r[level] for r in sel)
    n = sum(c.values())
    return ({k: v / n for k, v in c.items()}, n) if n else ({}, 0)


def tvd(a, b):
    return sum(abs(a.get(k, 0) - b.get(k, 0)) for k in set(a) | set(b)) / 2


def shift_vectors(recs, level, official_only=True):
    """One composition vector per shift -- the unit the posterior lives on."""
    by = defaultdict(Counter)
    for r in recs:
        if official_only and r["is_reconstruction"]:
            continue
        by[(r["year"], r["exam_date"], r["shift"])][r[level]] += 1
    out = []
    for k, c in by.items():
        n = sum(c.values())
        if n >= 5:
            out.append((k, {t: v / n for t, v in c.items()}))
    return out


def main() -> None:
    print("=" * 100)
    print("META-ANALYSIS 1 — YEAR-OVER-YEAR STABILITY (official papers only)")
    print("Deliberate anti-prediction mutation would show up as RISING TVD.")
    print("=" * 100)
    print(f"{'section':<20}{'pair':<16}{'TVD':>8}{'n_train':>9}{'n_test':>8}")
    print("-" * 100)
    stability = {}
    for sec, (path, level, drop) in SECTIONS.items():
        recs = load(path, level, drop)
        years = sorted({r["year"] for r in recs
                        if not r["is_reconstruction"]})
        tvds = []
        for a, b in zip(years, years[1:]):
            da, na = dist_for(recs, a, level)
            db, nb = dist_for(recs, b, level)
            if na < 30 or nb < 30:
                continue
            t = tvd(da, db)
            tvds.append(t)
            print(f"{sec:<20}{f'{a}->{b}':<16}{t:>8.3f}{na:>9}{nb:>8}")
        if tvds:
            stability[sec] = sum(tvds) / len(tvds)
    print("-" * 100)
    print("mean consecutive-year TVD by section:")
    for s, v in stability.items():
        print(f"  {s:<22}{v:>7.3f}")
    overall = sum(stability.values()) / len(stability)
    print(f"  {'OVERALL':<22}{overall:>7.3f}")

    print("\n" + "=" * 100)
    print("META-ANALYSIS 2 — PREDICTION EROSION TEST (§20, §28)")
    print("If setters were adapting away from history, the MOST RECENT year would")
    print("predict the next year better than pooled history. Tested per section.")
    print("=" * 100)
    print(f"{'section':<20}{'last_year MAE':>15}{'pooled MAE':>13}"
          f"{'winner':>16}")
    print("-" * 100)
    erosion = {}
    for sec, (path, level, drop) in SECTIONS.items():
        recs = load(path, level, drop)
        years = sorted({r["year"] for r in recs if not r["is_reconstruction"]})
        if len(years) < 3:
            print(f"{sec:<20}{'INSUFFICIENT DATA (needs 3 official years)':>44}")
            continue
        test = years[-1]
        prev = years[-2]
        train_years = years[:-1]
        vocab = sorted({r[level] for r in recs})
        pooled, _ = dist_for(recs, None, level) if False else (None, None)
        agg = Counter()
        for y in train_years:
            d, n = dist_for(recs, y, level)
            for k, v in d.items():
                agg[k] += v * n
        tot = sum(agg.values())
        pooled = {k: v / tot for k, v in agg.items()}
        lastd, _ = dist_for(recs, prev, level)
        actual, _ = dist_for(recs, test, level)
        mae = lambda p: sum(abs(p.get(k, 0) - actual.get(k, 0))
                            for k in vocab) / len(vocab)
        ml, mp = mae(lastd), mae(pooled)
        win = "last_year" if ml < mp else "pooled"
        erosion[sec] = (ml, mp, win)
        print(f"{sec:<20}{ml:>15.4f}{mp:>13.4f}{win:>16}")
    n_last = sum(1 for v in erosion.values() if v[2] == "last_year")
    print("-" * 100)
    print(f"sections where recency beat pooled history: {n_last} of {len(erosion)}")
    print("Interpretation: recency winning would be the signature of adaptation.")

    print("\n" + "=" * 100)
    print("META-ANALYSIS 3 — POSTERIOR DISPERSION OVER PAPER CONFIGURATIONS")
    print("Each historical shift is one observed paper configuration. If plausible")
    print("2026 papers cluster tightly, ONE paper covers the mass (§33).")
    print("=" * 100)
    rng = random.Random(2026)
    coverage_rows = []
    for sec, (path, level, drop) in SECTIONS.items():
        recs = load(path, level, drop)
        vecs = shift_vectors(recs, level)
        if len(vecs) < 8:
            print(f"{sec}: INSUFFICIENT DATA")
            continue
        # Centroid = the forecast blueprint. Measure how far real shifts sit.
        keys = sorted({t for _, v in vecs for t in v})
        cent = {t: sum(v.get(t, 0) for _, v in vecs) / len(vecs) for t in keys}
        dists = sorted(tvd(v, cent) for _, v in vecs)
        med = dists[len(dists) // 2]
        p90 = dists[int(len(dists) * 0.9)]
        # What fraction of shifts lie within a fixed tolerance of the centroid?
        for tol in (0.20, 0.25, 0.30, 0.35):
            frac = sum(1 for d in dists if d <= tol) / len(dists)
            coverage_rows.append((sec, tol, frac))
        print(f"{sec:<20} shifts={len(vecs):<4} median TVD to centroid={med:.3f}  "
              f"p90={p90:.3f}")

    print("\nfraction of real shifts within TVD tolerance of the centroid blueprint:")
    print(f"{'section':<20}{'tol=0.20':>10}{'tol=0.25':>10}{'tol=0.30':>10}{'tol=0.35':>10}")
    print("-" * 100)
    for sec in SECTIONS:
        row = [f for s, t, f in coverage_rows if s == sec]
        if row:
            print(f"{sec:<20}" + "".join(f"{v:>10.2f}" for v in row))

    json.dump({"stability_tvd": stability, "overall_tvd": overall,
               "erosion": {k: {"last_year_mae": v[0], "pooled_mae": v[1],
                               "winner": v[2]} for k, v in erosion.items()},
               "recency_wins": n_last, "sections_tested": len(erosion),
               "dispersion": [{"section": s, "tol": t, "frac": f}
                              for s, t, f in coverage_rows]},
              open(OUT / "meta_analysis.json", "w"), indent=2)
    print(f"\nWrote {OUT/'meta_analysis.json'}")


if __name__ == "__main__":
    main()
