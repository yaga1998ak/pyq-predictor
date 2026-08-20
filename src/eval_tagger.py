"""Measure tagger accuracy against a hand-labelled gold set.

Do this before trusting any backtest. If the tagger is 70% accurate, 30% of every
count downstream is wrong, and no amount of statistical care downstream repairs
it. Worse, tagging errors are rarely uniform -- a model that confuses
'mensuration' with 'geometry' distorts exactly the comparison you care about.

Build the gold set by hand:
  1. python src/eval_tagger.py --sample 200   (writes a blank CSV to label)
  2. Fill the 'gold_topic' column yourself. No shortcuts -- this is the ruler.
  3. python src/eval_tagger.py --gold data/gold_set.csv

Read confusion pairs carefully: a systematic pair is usually a taxonomy problem
(two topics not cleanly separable), not a model problem. Merge them.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

from schema import Taxonomy, load_papers, REPO


def write_sample(papers, n: int, out: Path, seed: int = 0) -> None:
    questions = [q for p in papers for q in p.questions]
    random.Random(seed).shuffle(questions)
    sample = questions[:n]
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["qid", "year", "text", "predicted_topic", "gold_topic"])
        for q in sample:
            w.writerow([q.qid, q.year, q.text[:400], q.topic or "", ""])
    print(f"wrote {len(sample)} rows -> {out}")
    print("Fill the 'gold_topic' column by hand, then re-run with --gold")


def evaluate(gold_path: Path, taxonomy: Taxonomy) -> None:
    rows = []
    with open(gold_path) as fh:
        for row in csv.DictReader(fh):
            if row.get("gold_topic", "").strip():
                rows.append(row)

    if not rows:
        raise SystemExit(f"no labelled rows in {gold_path} -- fill 'gold_topic' first")

    total = len(rows)
    correct = 0
    section_correct = 0
    confusion: dict[tuple[str, str], int] = defaultdict(int)
    unlabelled = 0

    for row in rows:
        pred = row["predicted_topic"].strip()
        gold = row["gold_topic"].strip()
        if not taxonomy.validate(gold):
            raise SystemExit(f"gold label '{gold}' ({row['qid']}) is not in the taxonomy")
        if not pred:
            unlabelled += 1
            continue
        if pred == gold:
            correct += 1
        else:
            confusion[(gold, pred)] += 1
        if taxonomy.topic_to_section.get(pred) == taxonomy.topic_to_section.get(gold):
            section_correct += 1

    acc = correct / total
    sec_acc = section_correct / total
    print(f"\ngold set: {total} labelled questions")
    print(f"topic accuracy   : {acc:.1%}")
    print(f"section accuracy : {sec_acc:.1%}")
    if unlabelled:
        print(f"untagged by model: {unlabelled}")

    if confusion:
        print("\ntop confusions (gold -> predicted):")
        for (gold, pred), n in Counter(confusion).most_common(10):
            print(f"  {n:>3}  {gold}  ->  {pred}")

    print()
    if acc >= 0.85:
        print("VERDICT: good enough. Downstream counts carry ~"
              f"{(1-acc)*100:.0f}% label noise -- acceptable.")
    elif acc >= 0.70:
        print("VERDICT: marginal. Fix the top confusion pairs (usually by merging\n"
              "         overlapping taxonomy topics) before trusting fine-grained counts.\n"
              "         Section-level analysis is still sound.")
    else:
        print("VERDICT: too weak to build on. Either the taxonomy is ambiguous or the\n"
              "         model is too small. Try a larger model for tagging, or collapse\n"
              "         the taxonomy to coarser topics.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", default=str(REPO / "data/tagged/papers.json"))
    ap.add_argument("--gold", default=str(REPO / "data/gold_set.csv"))
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--taxonomy", default="ssc_cgl")
    args = ap.parse_args()

    tax = Taxonomy.load(args.taxonomy)
    if args.sample:
        write_sample(load_papers(Path(args.papers)), args.sample, Path(args.gold))
    else:
        evaluate(Path(args.gold), tax)


if __name__ == "__main__":
    main()
