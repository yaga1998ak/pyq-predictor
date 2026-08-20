"""Backtest the PAPER GENERATOR, not just the forecast.

The forecast backtest asks "are the predicted topic weights close to the actual
weights?". This asks the question that matters for a mock paper: **if I had built
a paper for year T using only data available before T, how closely would its
composition have matched the real year T papers?**

Protocol, for each held-out year T:
    1. fit the forecast on years < T
    2. allocate 100 questions to topics from that forecast (largest remainder)
    3. compare that allocation to the ACTUAL topic distribution of year T,
       rescaled to 100 questions

Three comparators, because a number with nothing to beat means nothing:
    forecast_paper   what the generator actually builds
    last_year_paper  copy year T-1's distribution -- the obvious cheap strategy
    uniform_paper    equal weight to every topic in a section -- the floor

Headline metric is TOTAL VARIATION DISTANCE, in questions: half the sum of
absolute differences. It reads directly as "how many of the 100 questions are in
the wrong topic bucket" -- the thing a candidate actually cares about.

IMPORTANT: the "actual" distribution is measured by the rule tagger (~80%
precision, 68% coverage), so this validates the generator against the pipeline's
own labels, not ground truth. It bounds process error, not total error.

    python src/validate_papers.py
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from backtest import counts_vector, papers_to_history
from mock_paper import allocate
from models import MeanLastK
from schema import Taxonomy, load_papers, REPO


def actual_per100(counts: Counter, tax: Taxonomy) -> dict[str, float]:
    """Year T's real distribution, rescaled to a 100-question paper.

    Rescaled per SECTION to 25, matching how a real paper is built -- otherwise
    the tagger's uneven coverage (GA is under-tagged) would be read as the exam
    genuinely asking fewer GA questions.
    """
    out: dict[str, float] = {}
    for section in tax.sections:
        topics = tax.sections[section]["topics"]
        tot = sum(counts.get(t, 0) for t in topics)
        size = tax.section_size(section)
        for t in topics:
            out[t] = (counts.get(t, 0) / tot * size) if tot else 0.0
    return out


def tvd(a: dict[str, float], b: dict[str, float], topics: list[str]) -> float:
    return 0.5 * sum(abs(a.get(t, 0) - b.get(t, 0)) for t in topics)


def topk_overlap(a: dict, b: dict, topics: list[str], k: int = 10) -> float:
    ta = {t for t in sorted(topics, key=lambda x: -a.get(x, 0))[:k]}
    tb = {t for t in sorted(topics, key=lambda x: -b.get(x, 0))[:k]}
    return len(ta & tb) / k


def blend(forecast: dict, tax: Taxonomy, lam: float) -> dict:
    """Shrink the forecast toward a uniform-within-section prior.

    lam=1.0 trusts the forecast completely; lam=0.0 ignores history. The middle
    exists because the empirical forecast has a specific, known defect: topics the
    tagger cannot detect (direction_sense, matrix, ranking_and_order) read 0.0,
    and a generated paper then contains NONE of them -- which no real SSC paper
    ever does. Shrinking toward uniform restores a floor for every topic on the
    syllabus without discarding the signal in the topics we can measure.
    """
    out = {}
    for section in tax.sections:
        ts = tax.sections[section]["topics"]
        size = tax.section_size(section)
        tot = sum(forecast.get(t, 0.0) for t in ts) or 1.0
        for t in ts:
            emp = forecast.get(t, 0.0) / tot * size
            out[t] = lam * emp + (1 - lam) * (size / len(ts))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", default=str(REPO / "data/tagged/rules.json"))
    ap.add_argument("--taxonomy", default="ssc_cgl")
    ap.add_argument("--min-train", type=int, default=2)
    args = ap.parse_args()

    tax = Taxonomy.load(args.taxonomy)
    papers = load_papers(Path(args.papers))
    history, shifts = papers_to_history(papers)
    topics = tax.topics

    print(f"\nBacktesting the paper generator on {len(history)} years of data.")
    print("Metric: total variation distance = questions (of 100) in the wrong topic.\n")

    rows = []
    for split in range(args.min_train, len(history)):
        train = history[:split]
        test_year, test_counts = history[split]
        actual = actual_per100(test_counts, tax)

        # 1. what the generator would have built
        m = MeanLastK(tax, k=5)
        m.nominal_sections = True
        m.fit(train)
        pred = m.predict(tax.questions_per_paper)
        forecast = {t: float(pred[i]) for i, t in enumerate(topics)}
        gen = {t: float(v) for t, v in allocate(forecast, tax).items()}

        # 2. copy last year
        ly = actual_per100(train[-1][1], tax)
        ly_paper = {t: float(v) for t, v in allocate(ly, tax).items()}

        # 3. uniform within section
        uni = {}
        for section in tax.sections:
            ts = tax.sections[section]["topics"]
            for t in ts:
                uni[t] = tax.section_size(section) / len(ts)

        rows.append({
            "year": test_year,
            "papers": shifts[test_year],
            "gen": tvd(gen, actual, topics),
            "last": tvd(ly_paper, actual, topics),
            "uni": tvd(uni, actual, topics),
            "gen_top10": topk_overlap(gen, actual, topics),
            "last_top10": topk_overlap(ly_paper, actual, topics),
        })

    hdr = f"{'year':<7}{'papers':>7}{'generated':>12}{'copy T-1':>11}{'uniform':>10}{'top10':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['year']:<7}{r['papers']:>7}{r['gen']:>12.1f}{r['last']:>11.1f}"
              f"{r['uni']:>10.1f}{r['gen_top10']:>8.0%}")

    # Sweep the shrinkage weight on the same held-out years.
    print("\nShrinkage sweep (blending forecast with a uniform syllabus prior):")
    print(f"  {'lambda':<9}{'mean TVD':>10}{'top10':>8}   ")
    best = None
    for lam in [0.0, 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        tv, tk = [], []
        for split in range(args.min_train, len(history)):
            train = history[:split]
            ty, tc = history[split]
            actual = actual_per100(tc, tax)
            m2 = MeanLastK(tax, k=5); m2.nominal_sections = True; m2.fit(train)
            pr = m2.predict(tax.questions_per_paper)
            fc = {t: float(pr[i]) for i, t in enumerate(topics)}
            bl = blend(fc, tax, lam)
            pap = {t: float(v) for t, v in allocate(bl, tax).items()}
            tv.append(tvd(pap, actual, topics))
            tk.append(topk_overlap(pap, actual, topics))
        mtv, mtk = float(np.mean(tv)), float(np.mean(tk))
        mark = ""
        if best is None or mtv < best[1]:
            best = (lam, mtv, mtk); mark = ""
        print(f"  {lam:<9.2f}{mtv:>10.1f}{mtk:>8.0%}")
    print(f"\n  best lambda = {best[0]:.2f}  (TVD {best[1]:.1f}, top10 {best[2]:.0%})")

    g = np.mean([r["gen"] for r in rows])
    l = np.mean([r["last"] for r in rows])
    u = np.mean([r["uni"] for r in rows])
    t10 = np.mean([r["gen_top10"] for r in rows])
    print("-" * len(hdr))
    print(f"{'mean':<7}{'':>7}{g:>12.1f}{l:>11.1f}{u:>10.1f}{t10:>8.0%}")

    print(f"\nA generated paper puts ~{g:.0f} of 100 questions in the wrong topic bucket.")
    print(f"Copying last year: ~{l:.0f}.  Ignoring history entirely: ~{u:.0f}.")
    print(f"So the forecast is worth ~{u-g:.0f} questions vs no history, "
          f"~{l-g:.0f} vs copying last year.")
    print(f"Its top-10 topics match the real top-10 {t10:.0%} of the time.\n")

    if g < l and g < u:
        print("VERDICT: the generator beats both comparators. A paper built this way is")
        print("         a closer match to the real exam than either naive alternative.")
    elif g < u:
        print("VERDICT: beats ignoring history, but not copying last year. Prefer the")
        print("         simpler strategy unless the gap is within noise.")
    else:
        print("VERDICT: no better than ignoring history. Do not trust the composition.")

    print("\nCaveat: 'actual' is measured by the rule tagger (~80% precision, 68%")
    print("coverage), so this bounds PROCESS error, not total error. A topic the")
    print("tagger cannot see is invisible to both sides of this comparison.")


if __name__ == "__main__":
    main()
