"""Method playbook: for every recurring method, a real example and how it is solved.

The method report says *what fraction* of questions use each method. This says
*what one looks like* — a real past-paper question plus the technique — so the
answer to "what will 2026 ask?" is a worked pattern rather than a topic name.

Examples are pulled from the corpus, never written: a fabricated example can
misrepresent the technique, and every question here was really asked with SSC's
own published answer.

    python src/playbook.py --md out/METHOD_PLAYBOOK.md
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

# How each method is actually solved. Written once, by hand, because a technique
# explanation is domain knowledge -- it cannot be mined from question text.
HOWTO: dict[str, str] = {
    # --- number series (rule is computed per question, this is the general move)
    "arithmetic": "Take first differences. Constant ⇒ add that constant.",
    "geometric": "Take ratios of consecutive terms. Constant ⇒ multiply by it.",
    "linear_recurrence": "Test a(n) = p·a(n−1) + q. Solve p and q from the first three "
                         "terms, then verify against the rest.",
    "quadratic": "First differences are not constant but SECOND differences are — the "
                 "differences themselves form an AP.",
    "cubic": "Differences twice more; the third difference is constant.",
    "cyclic_difference": "Differences repeat in a short cycle (e.g. +9, −5, +9, −5). "
                         "Check period 2 and 3 before anything fancier.",
    "prime_difference": "Differences are consecutive primes. If diffs look irregular "
                        "but are all prime, this is it.",
    "alternating_ops": "Two operations applied in turn, e.g. ×2 then +3.",
    "power_offset": "Compare against n², n³ — the terms are a power plus a constant.",
    "fibonacci_like": "Each term is the sum of the previous two.",
    "alternating": "Two independent series interleaved; read odd and even positions "
                   "separately.",
    "wrong_term_quadratic": "One term breaks an otherwise clean rule — find the rule "
                            "from the majority, then the outlier.",
    "wrong_term_geometric": "As above, with a constant ratio.",
    "wrong_term_arithmetic": "As above, with a constant difference.",
    "wrong_term_power_offset": "As above, against n² or n³.",
    # --- coding-decoding
    "uniform_shift": "Every letter moves the same number of places. Compare letter "
                     "positions in the worked example.",
    "reversal": "The word is written backwards, sometimes with a shift on top.",
    "opposite_letter": "Each letter maps to its A↔Z complement (A→Z, B→Y).",
    "mixed_shift": "Different shift per position — write the per-letter shifts out and "
                   "look for a repeating pattern.",
    "alternating_shift": "Two shifts alternating across positions.",
    "length_change": "Code length differs from the word: usually positional values or "
                     "letter-count encoding rather than a shift.",
    # --- quant
    "successive_change": "Never add the percentages. Use net = a + b + ab/100 (signed), "
                         "or multiply factors: ×(1+a/100)×(1+b/100).",
    "percent_of_percent": "Multiply the fractions straight through; no addition.",
    "population_growth": "Compound growth — apply the factor once per period.",
    "income_expenditure": "Set up income − expenditure = savings and work in ratios.",
    "marks_passing": "Passing mark = obtained + shortfall; convert to a percentage of total.",
    "net_change": "Same as successive change; report the single equivalent percentage.",
    "successive_discount": "Multiply the remaining fractions: ×(1−d1)×(1−d2). Never add "
                           "discounts.",
    "marked_price_discount": "SP = MP × (1 − discount). Then compare SP with CP for profit.",
    "cp_sp_direct": "Profit% = (SP − CP)/CP × 100. Always on cost price.",
    "profit_percent": "Fix CP = 100 to turn percentages into plain numbers.",
    "two_articles": "Same SP, equal +x% and −x% ⇒ always a net LOSS of x²/100 percent.",
    "dishonest_dealer": "Gain% = error/(true value − error) × 100.",
    "si_ci_difference": "For 2 years, CI − SI = P(r/100)². Memorise it.",
    "compound_interest": "A = P(1 + r/100)^n. Adjust n and r for non-annual compounding.",
    "simple_interest": "SI = PRT/100.",
    "installments": "Discount each instalment back to present value and equate.",
    "rate_or_time_finding": "Rearrange the SI/CI formula for the unknown.",
    "together_time": "Work in rates: 1/A + 1/B = 1/together. Use LCM of days as total work.",
    "efficiency_ratio": "Efficiency is inverse to time; convert the ratio and share the work.",
    "men_days": "M₁D₁/W₁ = M₂D₂/W₂.",
    "alternate_days": "Compute a 2-day block of work, scale to the total, handle the "
                      "remainder separately.",
    "leaves_joins": "Track work done before and after the change in workforce.",
    "wages_share": "Wages split in the ratio of work done, not time spent.",
    "train_crossing": "Crossing a pole ⇒ train length only. Crossing a platform ⇒ length "
                      "of train + platform.",
    "relative_speed": "Opposite directions add speeds; same direction subtracts.",
    "average_speed": "Equal distances ⇒ harmonic mean 2ab/(a+b), NOT the plain average.",
    "late_early": "Distance is fixed; set the two time expressions equal.",
    "replacement_change": "New value = old ± n × (change in average).",
    "weighted_average": "Weight each group by its size before combining.",
    "partnership_profit": "Share profit in the ratio of capital × time.",
    "age_ratio": "Write present ages from the ratio with a common multiplier, then apply "
                 "the time shift.",
    "divide_amount": "Convert the ratio to parts and scale to the total.",
    "mixture_ratio": "Track each component separately, then re-form the ratio.",
    "repeated_replacement": "After n replacements, remaining = initial × (1 − x/V)^n.",
    "alligation": "Cross-difference the two prices around the mean to get the ratio.",
    "remainder": "Use modular arithmetic; for powers, find the cycle length.",
    "hcf_lcm": "HCF × LCM = product of the two numbers.",
    "unit_digit": "Only the last digit matters; powers cycle with period 4.",
    "divisibility": "Apply the divisibility rule for the specific divisor.",
    "factors_primes": "Prime-factorise; number of factors = product of (exponent + 1).",
    "x_plus_reciprocal": "From x + 1/x = k, use x² + 1/x² = k² − 2 and x³ + 1/x³ = k³ − 3k.",
    "identity_expansion": "a³ ± b³, (a ± b)², a² − b² — recognise before expanding.",
    "circle_chord_tangent": "Tangent ⊥ radius at the point of contact; equal tangents from "
                            "an external point; intersecting-chord products are equal.",
    "triangle_centres": "Centroid divides each median 2:1. Know which centre the question "
                        "is about.",
    "similar_triangles": "Corresponding sides are proportional; areas scale as the square "
                         "of the ratio.",
    "pythagoras": "a² + b² = c²; look for the standard triples (3-4-5, 5-12-13).",
    "polygon_properties": "Interior angle sum = (n − 2) × 180°.",
    "angle_chasing": "Angles on a line, in a triangle, and the cyclic-quadrilateral rule.",
    "solid_volume": "Match the solid to its formula; watch for hollow shapes.",
    "melting_recasting": "Volume is conserved — set volume before = volume after.",
    "surface_area": "Distinguish curved from total surface area.",
    "area_2d": "Standard area formulas; for a triangle use ½ × base × height or Heron.",
    "perimeter_circumference": "2πr for a circle; add all sides otherwise.",
    "identity_simplify": "Use sin²+cos²=1 and its two derived identities.",
    "value_at_angle": "Standard-angle table (0, 30, 45, 60, 90).",
    # --- reasoning
    "number_analogy": "Find the operation linking the pair (square, cube, ×n, ±k), then "
                      "apply it to the third.",
    "letter_cluster": "Convert letters to positions (A=1…Z=26) and look at the gaps.",
    "word_analogy": "Name the relationship in words first, then test each option against it.",
    "number_odd": "Test each option for the shared property — divisibility, being a "
                  "square, digit sum.",
    "letter_odd": "Convert to positions and compare the internal gaps.",
    "direct_relation": "Draw the family tree; mark gender explicitly.",
    "coded_relation": "Decode each symbol into a relation, then build the tree.",
    "two_statement": "Venn diagrams; a conclusion follows only if true in EVERY valid diagram.",
    "three_statement": "Same, with more overlap cases to check.",
    "symbol_interchange": "Swap the symbols as instructed FIRST, then apply BODMAS.",
    "symbol_substitution": "Replace each symbol with its meaning, then evaluate.",
    "balance_equation": "Test each option by substitution rather than solving forwards.",
    "dictionary_order": "Compare letter by letter, exactly as a dictionary orders words.",
    "word_from_letters": "Check each option's letters against the source word's supply.",
    "letters_unchanged": "Write the word and its alphabetical sort; compare positions.",
    # --- english
    "subject_verb": "Match the verb to the true subject; ignore words between them.",
    "preposition": "Usually fixed collocation — learn the verb+preposition pairs.",
    "tense": "Check sequence of tenses across clauses.",
    "phrase_replacement": "Read all options into the sentence; pick the one that is both "
                          "grammatical and idiomatic.",
    "vocabulary_fit": "Fit by meaning AND collocation, not meaning alone.",
    # --- GA
    "article_specific": "Learn the frequently-asked articles rather than all 395.",
    "institution_role": "Who appoints, who removes, what the tenure is.",
    "amendment": "Know the landmark amendments and what each changed.",
    "event_year": "Anchor dates to a timeline rather than memorising in isolation.",
    "river_system": "Origin, tributaries, states traversed, and the sea it joins.",
    "location_state": "Tie each site to its state.",
    "dance_form": "Dance form ↔ state pairs.",
    "indicator": "Which body publishes which index or figure.",
    "venue_year": "Host city and year for major tournaments.",
    "event_winner": "Recent winners, weighted to the last 1–3 years.",
}


def mine_examples(papers_json: Path, tax: Taxonomy):
    d = json.load(open(papers_json))
    per = defaultdict(lambda: {"n": 0, "years": Counter(), "examples": []})
    for p in d:
        for q in p["questions"]:
            text = " ".join(q["text"].split())
            topic, _ = classify(text)
            if not topic:
                continue
            method = rule = None
            if topic == "series_completion":
                seq = extract_sequence(text)
                if len(seq) >= 4:
                    fam, rl = classify_sequence(seq)
                    if fam not in ("too_short", "unknown"):
                        method, rule = fam, rl
            elif topic == "coding_decoding":
                fam, rl = classify_coding(text)
                if fam != "unknown":
                    method, rule = fam, rl
            if method is None:
                method = classify_method(topic, text)
            if not method:
                continue
            e = per[(topic, method)]
            e["n"] += 1
            e["years"][p["year"]] += 1
            # Prefer well-spaced text: 10 of 98 papers extract without word
            # spacing, and "Whichofthefollowingnumbers..." is unreadable as a
            # printed example even though it tags correctly.
            if 55 < len(text) < 330 and text.count(" ") / len(text) > 0.12:
                e["examples"].append({"text": text, "rule": rule, "year": p["year"]})
                e["examples"].sort(key=lambda x: -x["year"])   # recent first
                del e["examples"][2:]
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", default=str(REPO / "data/parsed/papers.json"))
    ap.add_argument("--md", default=str(REPO / "out/METHOD_PLAYBOOK.md"))
    ap.add_argument("--min-n", type=int, default=6)
    args = ap.parse_args()

    tax = Taxonomy.load("ssc_cgl")
    per = mine_examples(Path(args.papers), tax)

    by_section = defaultdict(list)
    for (topic, method), e in per.items():
        if e["n"] < args.min_n:
            continue
        by_section[tax.topic_to_section[topic]].append((topic, method, e))

    L = ["# SSC CGL — Method Playbook", "",
         "For every recurring solution method: how often it appears, a **real past-paper "
         "question**, and how it is solved. Examples are taken verbatim from 2021–2025 "
         "papers — none are invented.", "",
         "Use it as a checklist: if you can solve the example, that method is covered.", ""]

    total = 0
    for section in tax.sections:
        rows = sorted(by_section.get(section, []), key=lambda r: -r[2]["n"])
        if not rows:
            continue
        L += [f"## {section.replace('_',' ').title()}", ""]
        for topic, method, e in rows:
            total += 1
            yrs = ", ".join(str(y) for y in sorted(e["years"]))
            L += [f"### {topic.replace('_',' ').title()} › {method.replace('_',' ')}", "",
                  f"**{e['n']} questions** · seen in {yrs}", ""]
            how = HOWTO.get(method)
            if how:
                L += [f"**How to solve:** {how}", ""]
            for ex in e["examples"][:1]:
                L.append(f"> {ex['text']}")
                if ex["rule"]:
                    L.append(f">")
                    L.append(f"> **Rule here:** `{ex['rule']}`")
                L.append("")
    L += ["---", "",
          f"*{total} methods documented. Generated from 9,102 parsed questions; "
          "methods with fewer than "
          f"{args.min_n} occurrences are omitted as too rare to prepare for.*"]

    Path(args.md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.md).write_text("\n".join(L))
    print(f"{total} methods documented -> {args.md}")


if __name__ == "__main__":
    main()
