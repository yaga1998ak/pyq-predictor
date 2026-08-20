"""Where does the signal actually live? Measure every layer on ONE protocol.

Companion to question_level_backtest.py. Same blind split (train 2021-2024,
test 2025), so the numbers are directly comparable and the daily PDF can quote
a measured confidence per layer instead of an assumed one.

Layers, outermost to innermost:

  L1 SECTION COUNT   25/25/25/25 - fixed by the exam pattern
  L2 TOPIC COUNT     how many questions per topic
  L3 TOPIC RANKING   which topics are heaviest (ordering, not counts)
  L4 SPECIFIC Q      measured in question_level_backtest.py

Every number is deterministic numpy. No language model touches these.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TAGGED = ROOT / "data" / "tagged" / "rules.json"

TRAIN_YEARS = [2021, 2022, 2023, 2024]
TEST_YEAR = 2025


def load():
    papers = json.loads(TAGGED.read_text())
    by_year = defaultdict(list)
    for p in papers:
        y = p.get("year")
        if y is None:
            continue
        counts = Counter()
        for q in p.get("questions", []):
            t = q.get("topic")
            if t:
                counts[t] += 1
        if sum(counts.values()) >= 20:  # ignore near-empty parses
            by_year[y].append(counts)
    return by_year


def to_proportions(counts: Counter) -> dict:
    tot = sum(counts.values())
    return {k: v / tot for k, v in counts.items()} if tot else {}


def mean_profile(list_of_counts) -> dict:
    """Average topic proportions across papers."""
    acc = defaultdict(list)
    for c in list_of_counts:
        pr = to_proportions(c)
        for k, v in pr.items():
            acc[k].append(v)
    n = len(list_of_counts)
    return {k: sum(v) / n for k, v in acc.items()}


def main():
    by_year = load()
    train_papers = [c for y in TRAIN_YEARS for c in by_year.get(y, [])]
    test_papers = by_year.get(TEST_YEAR, [])

    print("=" * 68)
    print("LAYER BACKTEST — where does predictive signal live?")
    print("=" * 68)
    print(f"Train: {len(train_papers)} papers ({TRAIN_YEARS[0]}-{TRAIN_YEARS[-1]})")
    print(f"Test:  {len(test_papers)} papers ({TEST_YEAR})")
    print()

    # forecast = mean topic proportions over training years
    forecast = mean_profile(train_papers)
    # naive baseline = uniform over topics seen in training
    topics = sorted(forecast)
    uniform = {t: 1 / len(topics) for t in topics}

    # ---- L2: topic count MAE, scaled to a 100-question paper --------------
    def mae_against(pred: dict, actual_counts: Counter) -> float:
        tot = sum(actual_counts.values())
        if not tot:
            return float("nan")
        errs = []
        for t in topics:
            a = 100 * actual_counts.get(t, 0) / tot
            p = 100 * pred.get(t, 0.0)
            errs.append(abs(a - p))
        return float(np.mean(errs))

    mae_f = np.mean([mae_against(forecast, c) for c in test_papers])
    mae_u = np.mean([mae_against(uniform, c) for c in test_papers])
    skill = 1 - mae_f / mae_u if mae_u else 0.0

    print("-" * 68)
    print("L2  TOPIC COUNTS  (questions per topic, per 100-question paper)")
    print("-" * 68)
    print(f"  forecast (mean 2021-24) MAE : {mae_f:.3f}")
    print(f"  uniform baseline        MAE : {mae_u:.3f}")
    print(f"  skill vs uniform            : {100*skill:.1f}%")

    # ---- L3: ranking quality ---------------------------------------------
    print()
    print("-" * 68)
    print("L3  TOPIC RANKING  (are the heavy topics the right ones?)")
    print("-" * 68)
    pred_rank = [t for t, _ in sorted(forecast.items(), key=lambda kv: -kv[1])]
    hit_at = {}
    for k in (5, 10, 15):
        pk = set(pred_rank[:k])
        rates = []
        for c in test_papers:
            actual = [t for t, _ in sorted(c.items(), key=lambda kv: -kv[1])][:k]
            if actual:
                rates.append(len(pk & set(actual)) / len(actual))
        hit_at[k] = float(np.mean(rates)) if rates else 0.0
        print(f"  top-{k:<2d} overlap with actual : {100*hit_at[k]:5.1f}%")

    # ---- L1 for completeness ---------------------------------------------
    print()
    print("-" * 68)
    print("L1  SECTION COUNTS")
    print("-" * 68)
    print("  25/25/25/25 fixed by exam pattern — 100% accurate, 0 information")

    # ---- summary ---------------------------------------------------------
    ql_path = ROOT / "out" / "question_level_backtest.json"
    ql = json.loads(ql_path.read_text()) if ql_path.exists() else {}

    print()
    print("=" * 68)
    print("SIGNAL HIERARCHY — measured, one protocol")
    print("=" * 68)
    print(f"  L1 section counts   : fixed by pattern      (no information)")
    print(f"  L2 topic counts     : {100*skill:5.1f}% skill vs uniform")
    print(f"  L3 topic ranking    : {100*hit_at[10]:5.1f}% top-10 overlap")
    if ql:
        c = ql["ceiling"]["near_dup_080"] * 100
        r = ql["realised"]["implied_sa_rate"] * 100
        print(f"  L4 specific questions: {c:5.2f}% ceiling / {r:.2f}% realised")
    print()
    print("  => Build the paper at L2/L3. L4 is empty.")

    out = {
        "protocol": {"train": TRAIN_YEARS, "test": TEST_YEAR,
                     "train_papers": len(train_papers),
                     "test_papers": len(test_papers)},
        "L2_topic_counts": {"mae": float(mae_f), "baseline_mae": float(mae_u),
                            "skill": float(skill)},
        "L3_topic_ranking": {f"top_{k}": v for k, v in hit_at.items()},
        "L4_specific_questions": ql.get("realised", {}),
    }
    dest = ROOT / "out" / "layer_backtest.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"  written -> {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
