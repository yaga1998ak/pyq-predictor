"""Full method inventory across every question type, with year-over-year stability.

Answers the question the topic forecast cannot: for each topic, WHICH solution
methods does SSC actually use, how often, and is that mix stable enough to
prepare against?

Stability is the point. A method that holds a steady share across 2021-2025 is
predictable; one that swings wildly is not, and a candidate should know which is
which. Stability here = 1 - (stdev of yearly shares / mean share), computed only
on years with enough data to mean anything.

    python src/method_report.py                 # console summary
    python src/method_report.py --md out/METHODS.md
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

from archetypes import classify_coding, classify_sequence, extract_sequence
from methods import classify_method
from rules import classify
from schema import Taxonomy, REPO

MIN_YEAR_N = 8   # below this a yearly share is noise, not a signal


def mine(papers_json: Path):
    d = json.load(open(papers_json))
    recs = []
    for p in d:
        for q in p["questions"]:
            text = " ".join(q["text"].split())
            topic, _ = classify(text)
            if not topic:
                continue
            method = None
            if topic == "series_completion":
                seq = extract_sequence(text)
                if len(seq) >= 4:
                    fam, rule = classify_sequence(seq)
                    if fam not in ("too_short",):
                        method = fam
            elif topic == "coding_decoding":
                fam, _ = classify_coding(text)
                if fam != "unknown":
                    method = fam
            if method is None:
                method = classify_method(topic, text)
            recs.append({"year": p["year"], "topic": topic, "method": method})
    return recs


def summarise(recs, tax: Taxonomy):
    by_topic = defaultdict(list)
    for r in recs:
        by_topic[r["topic"]].append(r)

    out = []
    for topic, rows in by_topic.items():
        methods = Counter(r["method"] for r in rows if r["method"])
        unclassified = sum(1 for r in rows if not r["method"])
        if not methods:
            continue
        per_year = defaultdict(Counter)
        for r in rows:
            if r["method"]:
                per_year[r["year"]][r["method"]] += 1

        entries = []
        for m, n in methods.most_common():
            shares = [
                per_year[y][m] / sum(per_year[y].values())
                for y in per_year
                if sum(per_year[y].values()) >= MIN_YEAR_N
            ]
            if len(shares) >= 3 and st.mean(shares) > 0:
                cv = st.pstdev(shares) / st.mean(shares)
                stability = max(0.0, 1 - cv)
            else:
                stability = None
            entries.append({
                "method": m,
                "n": n,
                "share": n / sum(methods.values()),
                "stability": stability,
            })
        out.append({
            "topic": topic,
            "section": tax.topic_to_section[topic],
            "total": len(rows),
            "classified": sum(methods.values()),
            "unclassified": unclassified,
            "methods": entries,
        })
    out.sort(key=lambda r: (r["section"], -r["total"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", default=str(REPO / "data/parsed/papers.json"))
    ap.add_argument("--md", default="")
    ap.add_argument("--min-n", type=int, default=12)
    args = ap.parse_args()

    tax = Taxonomy.load("ssc_cgl")
    recs = mine(Path(args.papers))
    rows = summarise(recs, tax)

    tot = len(recs)
    cls = sum(1 for r in recs if r["method"])
    print(f"\n{tot} tagged questions · {cls} method-classified ({cls/tot:.0%})\n")

    L = ["# SSC CGL — Solution Method Inventory", "",
         f"Mined from {tot:,} tagged past-paper questions (2021–2025). "
         f"{cls:,} ({cls/tot:.0%}) matched a method signature.", "",
         "**Stability** = 1 − (variation of the method's yearly share). "
         "High means the method appears at a steady rate every year and is worth "
         "preparing specifically; low means it fluctuates and should be treated as "
         "opportunistic. Blank means too few years had enough data to judge.", ""]

    cur_section = None
    for r in rows:
        if r["total"] < args.min_n:
            continue
        if r["section"] != cur_section:
            cur_section = r["section"]
            title = cur_section.replace("_", " ").title()
            print(f"\n{'='*66}\n{title}\n{'='*66}")
            L += ["", f"## {title}", ""]
        print(f"\n{r['topic']}  ({r['total']} questions, "
              f"{r['unclassified']} unclassified)")
        L += [f"### {r['topic'].replace('_',' ').title()}", "",
              f"*{r['total']} questions · {r['unclassified']} unclassified*", "",
              "| Method | Questions | Share | Stability |", "|---|---|---|---|"]
        for e in r["methods"]:
            stab = f"{e['stability']:.0%}" if e["stability"] is not None else "–"
            print(f"    {e['method']:<26}{e['n']:>5}  {e['share']:>5.0%}  stab {stab}")
            L.append(f"| {e['method'].replace('_',' ')} | {e['n']} | "
                     f"{e['share']:.0%} | {stab} |")
        L.append("")

    if args.md:
        Path(args.md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.md).write_text("\n".join(L))
        print(f"\n-> {args.md}")


if __name__ == "__main__":
    main()
