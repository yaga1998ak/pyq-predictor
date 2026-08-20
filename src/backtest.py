"""Walk-forward backtest harness.

The protocol that keeps this honest:

  for each test year T:
      fit on papers strictly before T
      predict T
      score against actual T

No information from year >= T ever reaches the model that predicts T. That
includes your own knowledge of what appeared -- which is why model selection
must be driven by this script's output rather than by eyeballing recent papers.

Run:  python src/backtest.py --papers data/tagged/papers.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

import metrics
from models import all_models, MeanLastK
from schema import Taxonomy, load_papers, REPO


def papers_to_history(papers) -> tuple[list[tuple[int, Counter]], dict[int, int]]:
    """Collapse papers (possibly several shifts per year) into per-year counts.

    Multiple shifts are averaged, not summed, so a year with 6 shifts does not
    outweigh a year with 1 when both describe the same 100-question pattern.

    Also returns shifts-per-year, which the interval calculation needs: an
    average over 12 shifts is far less noisy than a single paper, and intervals
    built for one paper will over-cover badly against it.
    """
    by_year: dict[int, list[Counter]] = {}
    for p in papers:
        by_year.setdefault(p.year, []).append(p.topic_counts)
    history = []
    shifts = {}
    for year, counters in sorted(by_year.items()):
        merged = Counter()
        for c in counters:
            for topic, n in c.items():
                merged[topic] += n / len(counters)
        history.append((year, merged))
        shifts[year] = len(counters)
    return history, shifts


def counts_vector(counts: Counter, topics: list[str]) -> np.ndarray:
    return np.array([counts.get(t, 0.0) for t in topics], dtype=float)


def run_backtest(
    papers,
    taxonomy: Taxonomy,
    min_train_years: int = 3,
    top_k: int = 10,
) -> dict:
    history, shifts_per_year = papers_to_history(papers)
    years = [y for y, _ in history]

    if len(years) <= min_train_years:
        raise SystemExit(
            f"Need more than {min_train_years} years of data to backtest; got {len(years)}. "
            "Tag more papers first."
        )

    # Duplicate names would collide in the results dict and average unrelated
    # models together, producing a plausible-looking number for a model that
    # was never run. Fail loudly instead.
    names = [m.name for m in all_models(taxonomy)]
    if len(names) != len(set(names)):
        dupes = {n for n in names if names.count(n) > 1}
        raise SystemExit(f"duplicate model names in all_models(): {sorted(dupes)}")

    results: dict[str, list[dict]] = {}
    per_year: list[dict] = []

    for split in range(min_train_years, len(history)):
        train = history[:split]
        test_year, test_counts = history[split]
        actual = counts_vector(test_counts, taxonomy.topics)
        n_q = int(round(actual.sum())) or taxonomy.questions_per_paper

        n_shifts = shifts_per_year[test_year]
        row = {
            "year": test_year,
            "n_questions": n_q,
            "n_shifts": n_shifts,
            "models": {},
        }
        for model in all_models(taxonomy):
            model.fit(train)
            pred = model.predict(n_q)
            interval = model.predict_interval(n_q, n_papers=n_shifts)
            scores = metrics.evaluate(pred, actual, interval, k=top_k)
            results.setdefault(model.name, []).append(scores)
            row["models"][model.name] = scores
        per_year.append(row)

    # Aggregate, then express every model as skill against the mean-last-3 baseline.
    baseline_name = MeanLastK(taxonomy, k=3).name
    summary = {}
    for name, rows in results.items():
        agg = {k: float(np.mean([r[k] for r in rows if k in r])) for k in rows[0]}
        summary[name] = agg
    baseline_mae = summary[baseline_name]["mae"]
    for name, agg in summary.items():
        agg["skill_vs_baseline"] = metrics.skill_score(agg["mae"], baseline_mae)

    return {
        "test_years": [r["year"] for r in per_year],
        "baseline": baseline_name,
        "summary": summary,
        "per_year": per_year,
    }


def print_report(report: dict) -> None:
    summary = report["summary"]
    baseline = report["baseline"]
    order = sorted(summary.items(), key=lambda kv: kv[1]["mae"])

    print(f"\nWalk-forward backtest over test years: {report['test_years']}")
    print(f"Baseline for skill score: {baseline}\n")
    header = f"{'model':<32}{'MAE':>8}{'RMSE':>8}{'top10':>8}{'cover':>8}{'skill':>9}"
    print(header)
    print("-" * len(header))
    for name, agg in order:
        cover = agg.get("coverage_90")
        cover_s = f"{cover:.2f}" if cover is not None else "  -- "
        star = "  <- baseline" if name == baseline else ""
        print(
            f"{name:<32}{agg['mae']:>8.3f}{agg['rmse']:>8.3f}"
            f"{agg.get('top_10_hit', 0):>8.2f}{cover_s:>8}"
            f"{agg['skill_vs_baseline']:>+9.1%}{star}"
        )

    best, best_agg = order[0]
    print()
    if best == baseline:
        print(
            "VERDICT: no model beat the naive baseline. Ship the baseline and do not\n"
            "         trust the fancier ones -- on this data they add complexity, not signal."
        )
    elif best_agg["skill_vs_baseline"] < 0.05:
        print(
            f"VERDICT: {best} leads but by <5% -- that is inside the noise for this\n"
            f"         sample size. Treat it as a tie with the baseline."
        )
    else:
        print(
            f"VERDICT: {best} beats the baseline by "
            f"{best_agg['skill_vs_baseline']:.1%} MAE. Real signal."
        )
    cov = best_agg.get("coverage_90")
    if cov is not None and abs(cov - 0.90) > 0.10:
        direction = "over" if cov < 0.90 else "under"
        print(
            f"         WARNING: 90% intervals covered {cov:.0%} -- the model is "
            f"{direction}confident. Do not quote those ranges."
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", default=str(REPO / "data/tagged/papers.json"))
    ap.add_argument("--taxonomy", default="ssc_cgl")
    ap.add_argument("--min-train-years", type=int, default=3)
    ap.add_argument("--out", default=str(REPO / "out/backtest.json"))
    args = ap.parse_args()

    tax = Taxonomy.load(args.taxonomy)
    papers = load_papers(Path(args.papers))
    report = run_backtest(papers, tax, args.min_train_years)
    print_report(report)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nfull report -> {args.out}")


if __name__ == "__main__":
    main()
