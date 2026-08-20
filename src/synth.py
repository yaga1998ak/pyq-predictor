"""Synthetic paper generator -- for validating the harness before real PDFs exist.

This exists so you can answer "does my backtest actually detect signal?" on data
whose ground truth you control. Two regimes matter:

  drift=0.0   topic weights are stationary. Recency-weighted models should show
              NO advantage here; if one does, the harness is leaking information.
  drift>0.0   weights trend over time. Recency weighting should now help, and
              MeanLastK should lag behind.

If results do not move in those directions when you change --drift, something
in the pipeline is wrong -- fix that before trusting any real-data result.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from schema import Paper, Question, Taxonomy, save_papers, REPO


def generate(
    taxonomy: Taxonomy,
    start_year: int = 2014,
    end_year: int = 2025,
    drift: float = 0.04,
    noise: float = 0.15,
    shifts_per_year: int = 1,
    seed: int = 42,
) -> list[Paper]:
    rng = np.random.default_rng(seed)
    papers: list[Paper] = []

    # Base weights per section: some topics are structurally heavier than others.
    base: dict[str, np.ndarray] = {}
    trend: dict[str, np.ndarray] = {}
    for section, body in taxonomy.sections.items():
        n = len(body["topics"])
        w = rng.gamma(shape=2.0, scale=1.0, size=n)
        base[section] = w / w.sum()
        trend[section] = rng.normal(0, drift, size=n)

    for year in range(start_year, end_year + 1):
        t = year - start_year
        for shift in range(shifts_per_year):
            questions: list[Question] = []
            qno = 0
            for section, body in taxonomy.sections.items():
                topics = body["topics"]
                n_q = body["questions"]
                # weights drift linearly in log space, plus per-paper noise
                logits = np.log(base[section] + 1e-9) + trend[section] * t
                logits = logits + rng.normal(0, noise, size=len(topics))
                p = np.exp(logits)
                p = p / p.sum()
                draws = rng.multinomial(n_q, p)
                for topic, count in zip(topics, draws):
                    for _ in range(int(count)):
                        qno += 1
                        questions.append(
                            Question(
                                qid=f"{year}-S{shift+1}-Q{qno:03d}",
                                year=year,
                                exam=taxonomy.exam,
                                text=f"[synthetic {topic} question]",
                                section=section,
                                topic=topic,
                                tagger_confidence=1.0,
                            )
                        )
            papers.append(
                Paper(
                    year=year,
                    exam=taxonomy.exam,
                    shift=f"S{shift+1}" if shifts_per_year > 1 else None,
                    questions=questions,
                )
            )
    return papers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taxonomy", default="ssc_cgl")
    ap.add_argument("--start-year", type=int, default=2014)
    ap.add_argument("--end-year", type=int, default=2025)
    ap.add_argument("--drift", type=float, default=0.04,
                    help="0 = stationary topic weights; higher = stronger trend")
    ap.add_argument("--noise", type=float, default=0.15)
    ap.add_argument("--shifts-per-year", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(REPO / "data/tagged/synthetic.json"))
    args = ap.parse_args()

    tax = Taxonomy.load(args.taxonomy)
    papers = generate(
        tax,
        start_year=args.start_year,
        end_year=args.end_year,
        drift=args.drift,
        noise=args.noise,
        shifts_per_year=args.shifts_per_year,
        seed=args.seed,
    )
    save_papers(papers, Path(args.out))
    total = sum(len(p.questions) for p in papers)
    print(f"generated {len(papers)} papers ({total} questions) -> {args.out}")
    print(f"years {args.start_year}-{args.end_year}, drift={args.drift}, noise={args.noise}")


if __name__ == "__main__":
    main()
