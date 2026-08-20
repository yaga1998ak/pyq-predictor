"""§42 textbook backtest: does the FILTERING methodology itself have value?

The question is not "is the forecast accurate" -- that was tested per section.
It is: if the book had been built from years < T, what fraction of year T's
actual questions would a student have been equipped for?

Two different quantities are measured, because §44 insists on the distinction:

  PREDICTION COVERAGE -- the archetype appears in the taught set. The book told
                         the student to study that area.
  SOLUTION COVERAGE   -- the book actually teaches a method/knowledge that
                         solves it. An archetype I can name but not teach (a
                         mirror-image figure, an arbitrary synonym) counts for
                         prediction but NOT for solution.

Solution coverage is the honest headline. A book that says "study geometry" has
prediction coverage and no solution coverage.

Baseline for comparison: teaching the whole observed archetype vocabulary
(unfiltered syllabus). That has ~100% prediction coverage by construction, so the
filtered book must justify itself on coverage-per-unit-study instead.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"

SECTIONS = {
    "reasoning": ("reasoning_tagged.json", "subtopic", None),
    "english": ("english_tagged.json", "subtype", {2021}),
    "quant": ("quant_tagged.json", "subtopic", {2021}),
    "general_awareness": ("ga_tagged.json", "micro", {2021, 2022}),
}

# Archetypes the book can TEACH TO SOLUTION, i.e. where a method or a
# self-contained knowledge note exists. Derived from what was actually built:
# the Reasoning/Quant generators and solvers, the English rule set, and the 25
# GA zone notes. Everything else is prediction-only.
TEACHABLE = {
    "reasoning": {
        "number_or_letter_series", "language_code", "colon_notation",
        "number_set_analogy", "term_analogy", "generic_odd", "coded_relation",
        "symbolic_operator", "generation_chain", "sign_interchange",
        "sign_substitution", "symbol_definition", "statements_conclusions",
        "position_unchanged", "alphabetical_arrangement",
        "number_pair_operation", "letter_blank_completion",
        "letter_substitution", "day_offset", "identical_strings",
        "letter_cluster_analogy", "letter_cluster_odd", "number_group_odd",
        "assumption_or_course", "set_relationship", "letters_available",
        "pattern_completion",
    },
    "quant": {
        "simplification", "identities", "ratio_basic", "successive_change",
        "successive_discount", "simple_interest", "average_replacement",
        "pipes_cisterns", "trains", "angles", "divisibility",
        "marked_price_discount", "hcf_lcm", "average_basic",
        "direct_percentage", "cost_selling_price", "mean_proportional",
        "compound_interest", "boats_streams", "work_basic", "speed_basic",
        "relative_motion", "unit_digit_cyclicity", "remainder",
        "linear_equations", "factorisation", "mixture_alligation",
        "circle_area", "rectangle_square", "cube_cuboid", "chords",
        "triangle_properties", "reverse_percentage", "average_series",
        "men_days_hours", "efficiency_alternate", "partnership",
        "national_income", "budget_and_taxation",
    },
    "english": {
        "synonym", "antonym", "group_of_words", "idiom_meaning",
        "incorrectly_spelt", "correctly_spelt", "homonym_in_sentence",
        "single_blank", "underlined_segment", "underlined_word",
        "four_segments", "parts_as_options", "no_error_option",
        "to_passive", "to_active", "reported_speech", "sentence_order",
        "part_order", "numbered_blank", "targeted_rule",
    },
    # GA: the 25 selected zones are exactly the teachable set (each has a note).
    "general_awareness": None,   # filled from ga_zones_2026.json
}


def load(path, level, drop, official_only=True):
    recs = json.load(open(OUT / path))
    out = []
    for r in recs:
        if drop and r["year"] in drop:
            continue
        if official_only and r.get("is_reconstruction"):
            continue
        if not r.get(level):
            continue
        out.append(r)
    return out


def taught_set(train_recs, level, budget):
    """What the book would teach: the top `budget` archetypes by frequency.

    This is the filter. A conventional book teaches the whole vocabulary; the
    filtered book teaches only the highest-yield slice.
    """
    c = Counter(r[level] for r in train_recs)
    return {k for k, _ in c.most_common(budget)}


def evaluate(test_recs, level, taught, teachable):
    pred = sol = 0
    for r in test_recs:
        k = r[level]
        if k in taught:
            pred += 1
            if teachable is None or k in teachable:
                sol += 1
    n = len(test_recs)
    return (100 * pred / n if n else 0.0,
            100 * sol / n if n else 0.0, n)


def main() -> None:
    zones = json.load(open(OUT / "ga_zones_2026.json"))
    TEACHABLE["general_awareness"] = set(zones["final_zones"])

    # Book budget per section: how many archetypes the book teaches.
    BUDGET = 14

    print("=" * 104)
    print("§42 TEXTBOOK BACKTEST — book built from years < T, tested on year T")
    print(f"Filter: the book teaches the top {BUDGET} archetypes per section.")
    print("=" * 104)
    print(f"{'section':<20}{'test yr':>8}{'train n':>9}{'test n':>8}"
          f"{'prediction%':>13}{'SOLUTION%':>11}{'vocab':>7}{'taught':>8}")
    print("-" * 104)

    summary = defaultdict(dict)
    for sec, (path, level, drop) in SECTIONS.items():
        recs = load(path, level, drop)
        years = sorted({r["year"] for r in recs})
        for i in range(1, len(years)):
            test_y = years[i]
            train = [r for r in recs if r["year"] < test_y]
            test = [r for r in recs if r["year"] == test_y]
            if len(train) < 60 or len(test) < 40:
                continue
            taught = taught_set(train, level, BUDGET)
            p, s, n = evaluate(test, level, taught, TEACHABLE[sec])
            vocab = len({r[level] for r in recs})
            print(f"{sec:<20}{test_y:>8}{len(train):>9}{n:>8}"
                  f"{p:>12.1f}%{s:>10.1f}%{vocab:>7}{len(taught):>8}")
            summary[sec][test_y] = (p, s, n, vocab)

    print("\n" + "=" * 104)
    print("MOST RECENT TEST YEAR PER SECTION — the headline coverage")
    print("=" * 104)
    print(f"{'section':<20}{'test yr':>8}{'prediction%':>13}{'SOLUTION%':>11}"
          f"{'taught/vocab':>14}{'compression':>13}")
    print("-" * 104)
    tot_p = tot_s = tot_n = 0
    for sec in SECTIONS:
        if not summary[sec]:
            print(f"{sec:<20}{'INSUFFICIENT DATA':>50}")
            continue
        y = max(summary[sec])
        p, s, n, vocab = summary[sec][y]
        tot_p += p * n
        tot_s += s * n
        tot_n += n
        print(f"{sec:<20}{y:>8}{p:>12.1f}%{s:>10.1f}%"
              f"{f'{BUDGET}/{vocab}':>14}{f'{100*BUDGET/vocab:.0f}% of vocab':>13}")
    print("-" * 104)
    print(f"{'WEIGHTED TOTAL':<20}{'':>8}{tot_p/tot_n:>12.1f}%{tot_s/tot_n:>10.1f}%")

    # Coverage as a function of study budget -- the §55 efficiency curve.
    print("\n" + "=" * 104)
    print("§55 EFFICIENCY CURVE — solution coverage vs number of archetypes taught")
    print("=" * 104)
    print(f"{'archetypes taught':<20}" +
          "".join(f"{s[:9]:>11}" for s in SECTIONS) + f"{'mean':>9}")
    print("-" * 104)
    curve = {}
    for b in (5, 8, 10, 12, 14, 18, 25, 40):
        row, vals = [], []
        for sec, (path, level, drop) in SECTIONS.items():
            recs = load(path, level, drop)
            years = sorted({r["year"] for r in recs})
            if len(years) < 2:
                row.append("—")
                continue
            test_y = years[-1]
            train = [r for r in recs if r["year"] < test_y]
            test = [r for r in recs if r["year"] == test_y]
            if len(train) < 60 or len(test) < 40:
                row.append("—")
                continue
            taught = taught_set(train, level, b)
            _, s, _ = evaluate(test, level, taught, TEACHABLE[sec])
            row.append(f"{s:.1f}%")
            vals.append(s)
        mean = sum(vals) / len(vals) if vals else 0
        curve[b] = mean
        print(f"{b:<20}" + "".join(f"{x:>11}" for x in row) + f"{mean:>8.1f}%")

    print("\nmarginal solution coverage per additional 5 archetypes taught:")
    ks = sorted(curve)
    for a, b in zip(ks, ks[1:]):
        print(f"  {a:>3} -> {b:>3} archetypes : {curve[b]-curve[a]:>+6.1f} points")

    json.dump({"summary": {k: {str(y): v for y, v in d.items()}
                           for k, d in summary.items()},
               "efficiency_curve": curve, "budget": BUDGET},
              open(OUT / "textbook_backtest.json", "w"), indent=2)
    print(f"\nWrote {OUT/'textbook_backtest.json'}")


if __name__ == "__main__":
    main()
