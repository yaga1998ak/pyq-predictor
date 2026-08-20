"""Build the 2026 paper from METHOD predictions, not just topic weights.

The existing generator matches the topic mix: 3 percentage questions, 3 geometry.
This goes a layer deeper and matches the METHOD mix inside each topic — of those
3 percentage questions, how many are successive-change vs income-expenditure —
because that is the layer validation showed to be predictable (pooled 2021–2024
beat copy-2024 on 16 of 22 topics, where topic counts only tied with naive
averaging).

Two mechanisms drive selection:

  method mix     forecast per topic by recency-weighted pooling across
                 2021–2025, then allocated by largest remainder
  GA recency     general-awareness questions citing a year cluster at lag 1–3
                 from the exam (53% of year-citing GA), peaking at the previous
                 year — so GA selection is biased toward recent source papers

Everything else — real questions only, published answer keys, no repeats, 25 per
section — carries over from mock_paper.py.

    python src/method_paper.py --out out/papers_method
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from difficulty import band_within_topic
from make_papers import render_pdf
from mock_paper import allocate, harvest
from schema import Taxonomy, REPO

# Newer years weigh more: the 2025 pattern is a better guide to 2026 than 2021's.
# Half-life of 2 years, matching the recency structure that performed best in the
# forecast backtests.
YEAR_HALFLIFE = 2.0


def forecast_methods(pool: list[dict], latest_year: int) -> dict[str, dict[str, float]]:
    """Per topic, the predicted 2026 share of each method.

    Recency-weighted pooling across all available years -- the configuration
    validated in validate_papers.py, where pooling beat copying the most recent
    year on 16 of 22 topics.
    """
    acc: dict[str, Counter] = defaultdict(Counter)
    for q in pool:
        if not q.get("method"):
            continue
        w = 0.5 ** ((latest_year - q["year"]) / YEAR_HALFLIFE)
        acc[q["topic"]][q["method"]] += w

    out: dict[str, dict[str, float]] = {}
    for topic, c in acc.items():
        tot = sum(c.values())
        if tot > 0:
            out[topic] = {m: v / tot for m, v in c.items()}
    return out


def split_by_method(n: int, shares: dict[str, float]) -> dict[str, int]:
    """Largest remainder, so the per-method counts sum exactly to n."""
    if not shares or n <= 0:
        return {}
    raw = {m: s * n for m, s in shares.items()}
    base = {m: int(v) for m, v in raw.items()}
    short = n - sum(base.values())
    for m in sorted(raw, key=lambda k: raw[k] - base[k], reverse=True)[:short]:
        base[m] += 1
    return {m: k for m, k in base.items() if k > 0}


def select(pool, topic_want, method_fc, tax, used, rng, latest_year):
    """Pick questions matching topic AND method, falling back gracefully."""
    by_tm = defaultdict(list)
    by_t = defaultdict(list)
    for q in pool:
        by_t[q["topic"]].append(q)
        if q.get("method"):
            by_tm[(q["topic"], q["method"])].append(q)

    def order(qs, topic):
        qs = qs[:]
        rng.shuffle(qs)
        # GA questions age badly: a 2021 current-affairs item is stale for a 2026
        # sitting, and the lag analysis says recent years dominate. Elsewhere
        # recency is a mild preference for the current exam pattern.
        if tax.topic_to_section[topic] == "general_awareness":
            qs.sort(key=lambda q: -q["year"])
        else:
            qs.sort(key=lambda q: -(q["year"] + rng.random()))
        return qs

    picked, gaps = [], Counter()
    for topic, n in topic_want.items():
        if n <= 0:
            continue
        want_m = split_by_method(n, method_fc.get(topic, {}))
        got = 0
        for method, k in want_m.items():
            avail = [q for q in by_tm.get((topic, method), []) if id(q) not in used]
            for q in order(avail, topic)[:k]:
                used.add(id(q))
                picked.append(q)
                got += 1
        # top up from the same topic if some method had no stock
        if got < n:
            avail = [q for q in by_t.get(topic, []) if id(q) not in used]
            for q in order(avail, topic)[: n - got]:
                used.add(id(q))
                picked.append(q)
                got += 1
        if got < n:
            gaps[topic] += n - got
            # last resort: same section, so the paper still totals 25
            sec = tax.topic_to_section[topic]
            alt = [q for q in pool
                   if tax.topic_to_section[q["topic"]] == sec and id(q) not in used]
            for q in order(alt, topic)[: n - got]:
                used.add(id(q))
                picked.append(q)
    return picked, gaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forecast", default=str(REPO / "out/forecast_2026.json"))
    ap.add_argument("--raw", default=str(REPO / "data/raw"))
    ap.add_argument("--outdir", default=str(REPO / "out/papers_method"))
    ap.add_argument("--bands", nargs="+", default=["easy", "medium", "hard"])
    ap.add_argument("--per-band", type=int, default=1)
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    tax = Taxonomy.load("ssc_cgl")
    topic_fc = {r["topic"]: r["expected"]
                for r in json.load(open(args.forecast))["forecast"]}

    print("harvesting ...")
    pool = harvest(Path(args.raw))
    band_within_topic(pool)
    latest = max(q["year"] for q in pool)
    method_fc = forecast_methods(pool, latest)
    print(f"  {len(pool)} questions · method forecasts for {len(method_fc)} topics")

    topic_want = allocate(topic_fc, tax)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    used: set = set()
    no = 0

    for band in args.bands:
        band_pool = [q for q in pool if q["difficulty"] == band] or pool
        for _ in range(args.per_band):
            no += 1
            picked, gaps = select(band_pool, topic_want, method_fc, tax,
                                  used, rng, latest)
            if len(picked) < 60:
                print(f"  paper {no}: only {len(picked)} questions, skipped")
                no -= 1
                continue
            path = outdir / f"SSC_CGL_2026_MethodPaper_{no:02d}_{band}.pdf"
            render_pdf(picked, tax, path, no, f"{band} · method-matched", args.year)
            with_m = sum(1 for q in picked if q.get("method"))
            print(f"  paper {no} [{band:<6}] {len(picked)} questions · "
                  f"{with_m} method-matched · {len({q['source'] for q in picked})} shifts")

    # Report what the method forecast expects, so the paper can be checked against it.
    print("\n2026 predicted method mix (top methods per major topic):")
    for topic in ["series_completion", "percentage", "profit_and_loss",
                  "geometry", "coding_decoding", "polity_constitution"]:
        if topic not in method_fc:
            continue
        top = sorted(method_fc[topic].items(), key=lambda kv: -kv[1])[:3]
        s = ", ".join(f"{m.replace('_',' ')} {v:.0%}" for m, v in top)
        print(f"  {topic:<22} {s}")
    print(f"\n-> {outdir}")


if __name__ == "__main__":
    main()
