"""Route a subset of questions to the paid API; leave the rest on the free local model.

The economics of this pipeline are lopsided: local inference costs nothing but
hours, the API costs cents but is instant. So spend the API where it changes a
decision, and let the local model grind through everything else.

Two policies, in the order you should use them:

  --policy gold        Random N questions -> Opus 5. This is a MEASUREMENT, not a
                       relabel: it tells you how accurate the free tagger is. Until
                       you have this number, every downstream result is unfalsifiable.
                       ~$0.17 for 200 questions.

  --policy low-conf    Re-tag only questions the local model was unsure about.
                       Confidence is a weak signal, but it is strictly better than
                       random, and it concentrates the spend on the tail where
                       small models actually fail. Cost scales with how big that
                       tail turns out to be, typically a dollar or two.

Deliberately NOT offered: a "re-tag everything with the API" policy. That is just
tag_api.py without --limit, and if you are going to do it, do it there with the
Batch API discount rather than through this router.

    python src/cascade.py --policy gold --n 200 --dry-run
    python src/cascade.py --policy gold --n 200
    python src/cascade.py --policy low-conf --threshold 0.7 --batch
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from schema import Taxonomy, load_papers, save_papers, REPO
from tag_api import estimate_cost, tag_batch, tag_sync, DEFAULT_MODEL


def select_gold(papers, n: int, seed: int):
    """Random sample, stratified across years so one year cannot dominate."""
    by_year: dict[int, list] = {}
    for p in papers:
        for q in p.questions:
            if q.topic:
                by_year.setdefault(q.year, []).append(q)
    if not by_year:
        raise SystemExit("no tagged questions -- run src/tag.py first")

    rng = random.Random(seed)
    per_year = max(1, n // len(by_year))
    picked = []
    for year, qs in sorted(by_year.items()):
        rng.shuffle(qs)
        picked.extend(qs[:per_year])
    rng.shuffle(picked)
    return picked[:n]


def select_low_confidence(papers, threshold: float):
    """WARNING: on qwen2.5:7b this selects nothing.

    Measured on the full 7,858-question run: mean confidence 0.96, median 1.00,
    minimum 0.80 -- not one question fell below 0.7. The model reports high
    confidence whether or not it is right, so its confidence carries no routing
    information. Kept for better-calibrated taggers; use --policy section instead.
    """
    return [
        q
        for p in papers
        for q in p.questions
        if q.topic and (q.tagger_confidence is None or q.tagger_confidence < threshold)
    ]


def select_sections(papers, taxonomy: Taxonomy, sections: list[str]):
    """Route whole sections to the API.

    This is the policy the data actually supports. Distribution analysis of the
    local run showed the tagger assigns SECTIONS well (within ~2 questions of the
    expected 25 per paper) but collapses within some of them -- reasoning put 50%
    of its questions in series_completion and never once used coding_decoding, a
    staple SSC topic. That is a systematic failure confined to identifiable
    sections, so pay to redo those and keep the free labels everywhere else.
    """
    valid = set(taxonomy.sections)
    unknown = set(sections) - valid
    if unknown:
        raise SystemExit(f"unknown section(s): {sorted(unknown)}. Valid: {sorted(valid)}")
    return [
        q
        for p in papers
        for q in p.questions
        if q.topic and taxonomy.topic_to_section.get(q.topic) in sections
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", default=str(REPO / "data/tagged/papers.json"))
    ap.add_argument("--policy", choices=["gold", "low-conf", "section"], required=True)
    ap.add_argument("--n", type=int, default=200, help="gold policy: sample size")
    ap.add_argument("--threshold", type=float, default=0.7, help="low-conf policy")
    ap.add_argument("--sections", nargs="+",
                    default=["general_intelligence_reasoning", "english_comprehension"],
                    help="section policy: which sections to re-tag via the API")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--taxonomy", default="ssc_cgl")
    ap.add_argument("--batch", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="select and cost, no API calls")
    ap.add_argument("--out", default=str(REPO / "out/cascade.json"))
    args = ap.parse_args()

    tax = Taxonomy.load(args.taxonomy)
    papers = load_papers(Path(args.papers))

    if args.policy == "gold":
        selected = select_gold(papers, args.n, args.seed)
    elif args.policy == "section":
        selected = select_sections(papers, tax, args.sections)
    else:
        selected = select_low_confidence(papers, args.threshold)

    total_tagged = sum(1 for p in papers for q in p.questions if q.topic)
    if not selected:
        raise SystemExit("nothing selected by this policy")

    est = estimate_cost(len(selected), tax, args.model, args.batch)
    share = len(selected) / total_tagged if total_tagged else 0
    print(f"\npolicy      : {args.policy}")
    print(f"selected    : {len(selected)} of {total_tagged} tagged ({share:.0%})")
    print(f"model       : {args.model}{' (batch)' if args.batch else ''}")
    print(f"est. cost   : ${est['cost_usd']}")

    if args.policy == "gold":
        print("\nThis measures the free tagger; it does not improve it. The output is an\n"
              "accuracy number you can act on -- keep the local labels, or pay to redo them.")

    if args.dry_run:
        print("\n--dry-run: nothing sent.")
        return

    import anthropic

    client = anthropic.Anthropic()

    # Snapshot the local labels BEFORE the API overwrites them -- the comparison
    # is the entire point of the gold policy, and it is unrecoverable afterwards.
    before = {q.qid: (q.topic, q.tagger_confidence) for q in selected}
    for q in selected:
        q.topic = None

    runner = tag_batch if args.batch else tag_sync
    ok, failed = runner(client, selected, tax, args.model, Path(args.out), papers)
    print(f"\nAPI tagged {ok}, failed {failed}")

    agree = disagree = 0
    confusions: Counter = Counter()
    for q in selected:
        old, _ = before[q.qid]
        if not q.topic:
            continue
        if q.topic == old:
            agree += 1
        else:
            disagree += 1
            confusions[(old, q.topic)] += 1

    compared = agree + disagree
    if compared:
        print(f"\nlocal vs {args.model}: {agree}/{compared} agree = {agree/compared:.0%}")
        if confusions:
            print("\ntop disagreements (local -> API):")
            for (a, b), n in confusions.most_common(10):
                print(f"  {n:>3}  {a}  ->  {b}")

        rate = agree / compared
        print()
        if rate >= 0.85:
            print("VERDICT: the free tagger is good enough. Keep the local labels for the\n"
                  "         whole corpus and spend nothing further on tagging.")
        elif rate >= 0.70:
            print("VERDICT: marginal. Run --policy low-conf to repair the uncertain tail,\n"
                  "         or merge the topics that dominate the confusion list above.")
        else:
            print("VERDICT: the free tagger is not reliable enough. Re-tag the corpus with\n"
                  f"         {args.model} via tag_api.py --batch (~$6.57 for the full set).")
        print("\nAgreement is not accuracy -- both taggers can be wrong together. Hand-label\n"
              "a slice of these and run eval_tagger.py for a real number.")

    save_papers(papers, Path(args.papers))
    with open(args.out, "w") as fh:
        json.dump(
            {
                "policy": args.policy,
                "model": args.model,
                "selected": len(selected),
                "agree": agree,
                "disagree": disagree,
                "confusions": [
                    {"local": a, "api": b, "count": n} for (a, b), n in confusions.most_common()
                ],
            },
            fh,
            indent=2,
        )
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
