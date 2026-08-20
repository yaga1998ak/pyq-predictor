"""Benchmark tagging models against each other on identical questions.

Measures the three things that decide which model to run for hours:

  throughput      q/s -- decides whether a full run takes 2 hours or 55
  valid rate      how often the reply is a real taxonomy label, not an invention
  agreement       where two models agree, both are probably right; where they
                  disagree, at least one is wrong -- and that set is exactly
                  what your hand-labelled gold questions should come from

Agreement is NOT accuracy. Two models can agree and both be wrong, especially
on ambiguous taxonomy boundaries. Use eval_tagger.py against hand labels for
the real number; this only tells you whether a cheaper model is as good as an
expensive one.

Run:  python src/bench_tagger.py --models deepseek-r1:8b qwen2.5:7b -n 30
"""

from __future__ import annotations

import argparse
import random
import time
from collections import Counter
from pathlib import Path

from schema import Taxonomy, load_papers, REPO
from tag import tag_question


def bench(model: str, questions: list[str], tax: Taxonomy) -> dict:
    labels, t0 = [], time.time()
    for text in questions:
        topic, _ = tag_question(text, tax, model)
        labels.append(topic)
    elapsed = time.time() - t0
    valid = sum(1 for t in labels if t)
    return {
        "model": model,
        "labels": labels,
        "seconds": elapsed,
        "q_per_s": len(questions) / elapsed if elapsed else 0.0,
        "valid_rate": valid / len(questions),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", default=str(REPO / "data/parsed/papers.json"))
    ap.add_argument("--models", nargs="+", default=["deepseek-r1:8b", "qwen2.5:7b"])
    ap.add_argument("-n", type=int, default=30)
    ap.add_argument("--taxonomy", default="ssc_cgl")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    tax = Taxonomy.load(args.taxonomy)
    papers = load_papers(Path(args.papers))
    pool = [q.text for p in papers for q in p.questions if len(q.text) > 60]
    random.Random(args.seed).shuffle(pool)
    sample = pool[: args.n]
    print(f"benchmarking {len(args.models)} models on {len(sample)} identical questions\n")

    results = []
    for model in args.models:
        print(f"  running {model} ...", flush=True)
        results.append(bench(model, sample, tax))

    print(f"\n{'model':<20}{'q/s':>8}{'total':>10}{'valid':>8}{'est. 7861q':>14}")
    print("-" * 60)
    for r in results:
        hours = (7861 / r["q_per_s"] / 3600) if r["q_per_s"] else float("inf")
        print(
            f"{r['model']:<20}{r['q_per_s']:>8.2f}{r['seconds']:>9.0f}s"
            f"{r['valid_rate']:>8.0%}{hours:>13.1f}h"
        )

    if len(results) == 2:
        a, b = results
        both = [(x, y) for x, y in zip(a["labels"], b["labels"]) if x and y]
        agree = sum(1 for x, y in both if x == y)
        print(f"\nagreement (both produced a label): {agree}/{len(both)} = "
              f"{agree/len(both):.0%}" if both else "\nno comparable labels")
        disagreements = [(x, y) for x, y in both if x != y]
        if disagreements:
            print("\ndisagreements -- hand-label THESE first:")
            for (x, y), n in Counter(disagreements).most_common(8):
                print(f"  {n:>2}x  {a['model']}={x:<28} {b['model']}={y}")

    fast = max(results, key=lambda r: r["q_per_s"])
    print(f"\nfastest: {fast['model']} ({fast['q_per_s']:.2f} q/s)")
    print("Speed only settles it if valid_rate and agreement hold up. Confirm with\n"
          "eval_tagger.py against hand labels before committing to a long run.")


if __name__ == "__main__":
    main()
