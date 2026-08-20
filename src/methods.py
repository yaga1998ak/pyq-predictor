"""Mine the SOLUTION METHOD of every question type, across all four sections.

Topic tells you *what* is being asked ("percentage"). Method tells you *how it is
solved* ("successive percentage change") — which is the level a candidate
actually prepares at. Two percentage questions can need entirely different
techniques; two questions in different topics can need the same one.

Method signatures are regex over question text, one layer below the topic rules.
Number-series and coding-decoding are handled separately in archetypes.py, where
the rule is recoverable arithmetically rather than lexically.

Like the topic rules, this DECLINES rather than guesses: a question matching no
method signature is reported as unclassified, because an invented method label is
worse than an honest gap.
"""

from __future__ import annotations

import re

# topic -> [(method, regex)] — first match wins, so specific precedes general.
METHODS: dict[str, list[tuple[str, str]]] = {
    # ---------------- Quantitative ----------------
    "percentage": [
        ("successive_change", r"increased by.{0,40}(then|and).{0,25}(decreas|reduc)|"
                              r"decreased by.{0,40}(then|and).{0,25}increas|successive"),
        ("percent_of_percent", r"\d+ ?% of \d+ ?% of|% of.{0,20}% of"),
        ("population_growth", r"population|\bcensus\b"),
        ("marks_passing", r"passing marks|\bexam\b.{0,40}marks|failed by|secured \d+ ?%"),
        ("income_expenditure", r"income|expenditure|salary|savings|budget"),
        ("net_change", r"net (change|effect|increase|decrease)"),
        ("simple_percentage", r"what (is|will be) \d+ ?%|find \d+ ?% of"),
    ],
    "profit_and_loss": [
        ("successive_discount", r"successive discount|discounts? of \d+ ?%.{0,30}(and|,) ?\d+ ?%"),
        ("marked_price_discount", r"marked (price|at)|list price|\bdiscount\b"),
        ("two_articles", r"two (articles|items)|each.{0,25}(sold|bought)"),
        ("dishonest_dealer", r"dishonest|false weight|cheats|claims to sell at cost"),
        ("cp_sp_direct", r"cost price|selling price|\bC\.?P\b|\bS\.?P\b"),
        ("profit_percent", r"profit (of|percent|%)|loss (of|percent|%)|gain"),
    ],
    "simple_and_compound_interest": [
        ("si_ci_difference", r"difference between.{0,40}(compound|simple)|"
                             r"(compound|simple).{0,40}difference"),
        ("compound_interest", r"compound interest|compounded (annually|half|quarterly)"),
        ("installments", r"instal?lment|equal annual"),
        ("rate_or_time_finding", r"find the rate|at what rate|in what time|in how many years"),
        ("simple_interest", r"simple interest"),
    ],
    "time_and_work": [
        ("alternate_days", r"alternate|on alternate days|working on alternate"),
        ("efficiency_ratio", r"efficien|\btwice as|thrice as|as efficient"),
        ("leaves_joins", r"leaves|left after|joined after|after \d+ days"),
        ("wages_share", r"wage|paid|share of the (money|amount)"),
        ("men_days", r"\bmen\b.{0,30}\bdays\b|\bwomen\b.{0,30}\bdays\b|\bboys\b.{0,25}\bdays\b"),
        ("together_time", r"working together|together (they|complete|finish)"),
    ],
    "time_speed_distance": [
        ("train_crossing", r"\btrain\b.{0,60}(cross|pass|overtake|platform|pole|tunnel|bridge)"),
        ("relative_speed", r"opposite direction|same direction|towards each other|relative"),
        ("average_speed", r"average speed"),
        ("late_early", r"late by|early by|reaches? .{0,20}(late|early)|misses the"),
        ("simple_speed", r"speed of|km/h|distance covered"),
    ],
    "pipes_and_cisterns": [
        ("leak", r"\bleak\b|emptied|outlet"),
        ("alternate_open", r"alternate|opened for \d+"),
        ("fill_together", r"together|both pipes|all (three|the) pipes"),
        ("single_pipe", r"pipe can fill|fills the (tank|cistern)"),
    ],
    "average": [
        ("replacement_change", r"replaced by|new (person|man|member) (joins|comes)|"
                               r"average (increases|decreases) by"),
        ("weighted_average", r"average of (all|the combined)|combined average|two groups"),
        ("innings_score", r"innings|batting average|matches"),
        ("simple_average", r"average of"),
    ],
    "ratio_and_proportion": [
        ("partnership_profit", r"partner|invest.{0,40}(profit|share)|capital"),
        ("age_ratio", r"ages? of|years (ago|hence)|present age"),
        ("divide_amount", r"divided (among|between|in the ratio)|share.{0,25}ratio"),
        ("mixture_ratio", r"mixture|milk and water|\balloy\b"),
        ("simple_ratio", r"\bratio\b|proportion"),
    ],
    "mixture_and_alligation": [
        ("repeated_replacement", r"replaced by (water|milk)|process is repeated|drawn off"),
        ("two_mixtures", r"two (vessels|mixtures|containers)|mixed in the ratio"),
        ("alligation", r"alligation|\balloy\b|milk and water"),
    ],
    "number_system": [
        ("remainder", r"\bremainder\b|when divided by"),
        ("hcf_lcm", r"\bH\.?C\.?F\b|\bL\.?C\.?M\b|greatest common|least common"),
        ("unit_digit", r"unit(s)? digit|last digit|ten'?s digit"),
        ("divisibility", r"divisible by|is a multiple of"),
        ("factors_primes", r"prime (number|factor)|number of factors"),
    ],
    "algebra": [
        ("x_plus_reciprocal", r"x ?\+ ?1/x|a ?\+ ?1/a|x ?- ?1/x"),
        ("identity_expansion", r"a\^?[23] ?[+\-] ?b\^?[23]|\(a ?[+\-] ?b\)\^?[23]|identity"),
        ("simultaneous_equations", r"two equations|solve for x and y|system of"),
        ("polynomial_factor", r"factoris|\bpolynomial|\bfactor of\b"),
        ("value_of_expression", r"find the value of|value of x"),
    ],
    "geometry": [
        ("circle_chord_tangent", r"\bchord\b|\btangent\b|\bsecant\b|circle.{0,40}radius"),
        ("triangle_centres", r"centroid|incentre|circumcentre|orthocentre|median|bisector"),
        ("similar_triangles", r"similar triangles?|\bproportional\b.{0,30}sides"),
        ("angle_chasing", r"\bangle\b.{0,40}(equal|find|measure)|degrees?"),
        ("polygon_properties", r"polygon|quadrilateral|parallelogram|rhombus|trapezium"),
        ("pythagoras", r"right[- ]angled|hypotenuse|perpendicular"),
    ],
    "mensuration": [
        ("melting_recasting", r"melted|recast|moulded|converted into"),
        ("solid_volume", r"cuboid|cylinder|\bcone\b|\bsphere\b|hemisphere|\bprism\b"),
        ("surface_area", r"surface area|curved surface|total surface"),
        ("area_2d", r"\barea of\b.{0,40}(triangle|square|rectangle|circle|trapezium)"),
        ("perimeter_circumference", r"perimeter|circumference"),
    ],
    "trigonometry": [
        ("identity_simplify", r"simplify|prove|identity|sin\^?2|cos\^?2"),
        ("value_at_angle", r"(sin|cos|tan|cot|sec|cosec) ?\d+ ?°|at 30|at 45|at 60"),
        ("max_min_value", r"maximum value|minimum value"),
    ],
    "data_interpretation": [
        ("pie_chart", r"pie chart"),
        ("bar_graph", r"bar (graph|chart)"),
        ("table_data", r"\btable\b"),
    ],
    # ---------------- Reasoning ----------------
    "analogy": [
        ("number_analogy", r"\d+ ?: ?\d+|numbers are related"),
        ("letter_cluster", r"[A-Z]{2,} ?: ?[A-Z]{2,}|letter[- ]cluster"),
        ("word_analogy", r"word.{0,30}related|\brelated to\b"),
    ],
    "classification_odd_one_out": [
        ("number_odd", r"\d+ ?, ?\d+|numbers"),
        ("letter_odd", r"[A-Z]{2,} ?, ?[A-Z]{2,}|letter[- ]cluster"),
        ("word_odd", r"\bword\b|which one of the following"),
    ],
    "blood_relations": [
        ("coded_relation", r"means|denotes|represents|[+\-×÷] ?[A-Z]"),
        ("direct_relation", r"how is|related to|pointing to (a )?(photo|picture)"),
    ],
    "syllogism": [
        ("three_statement", r"3\.|III\.|three statements"),
        ("two_statement", r"2\.|II\.|conclusions?"),
    ],
    "mathematical_operations": [
        ("symbol_interchange", r"interchang"),
        ("symbol_substitution", r"means|denotes|stands for|if ['\"‘]?[+\-×÷]"),
        ("balance_equation", r"balance|correct the equation"),
    ],
    "word_formation": [
        ("dictionary_order", r"dictionary|alphabetical order"),
        ("letters_unchanged", r"remain unchanged|position of"),
        ("word_from_letters", r"cannot be formed|can be formed|using the letters"),
    ],
    # ---------------- English ----------------
    "spotting_errors": [
        ("subject_verb", r"\bis\b|\bare\b|\bwas\b|\bwere\b|\bhas\b|\bhave\b"),
        ("preposition", r"\bin\b|\bon\b|\bat\b|\bto\b|\bfor\b|\bwith\b"),
        ("tense", r"\bhad\b|\bwill\b|\bbeen\b|\bing\b"),
    ],
    "fill_in_the_blanks": [
        ("vocabulary_fit", r"most appropriate (word|option)"),
        ("grammar_fit", r"grammatically|correct form"),
    ],
    "sentence_improvement": [
        ("phrase_replacement", r"underlined|bracketed|substitute|replace"),
    ],
    # ---------------- General Awareness ----------------
    "polity_constitution": [
        ("article_specific", r"article \d+"),
        ("schedule_specific", r"\bschedule\b"),
        ("amendment", r"amendment"),
        ("institution_role", r"parliament|lok sabha|rajya sabha|supreme court|election commission"),
    ],
    "history_modern": [
        ("event_year", r"\b1[6-9]\d\d\b"),
        ("person_role", r"who (was|among|led|founded)"),
        ("movement", r"movement|struggle|satyagraha|revolt"),
    ],
    "geography_indian": [
        ("location_state", r"located in|situated in|which state"),
        ("river_system", r"\briver\b|tributary|origin"),
        ("protected_area", r"national park|sanctuary|reserve|biosphere"),
    ],
    "economics": [
        ("institution", r"\bRBI\b|reserve bank|niti aayog|world bank|\bIMF\b|\bWTO\b"),
        ("indicator", r"\bGDP\b|inflation|repo|fiscal|per capita"),
        ("scheme_policy", r"budget|policy|tax|\bGST\b"),
    ],
    "sports": [
        ("event_winner", r"who won|winner of|champion"),
        ("venue_year", r"held in|hosted|venue"),
        ("rules_measures", r"height of|length of|number of players|dimensions"),
    ],
    "art_and_culture": [
        ("dance_form", r"dance"),
        ("festival_state", r"festival|bihu|onam|pongal"),
        ("instrument_artist", r"instrument|santoor|sitar|tabla|veena"),
    ],
}

COMPILED = {
    topic: [(name, re.compile(p, re.IGNORECASE)) for name, p in pats]
    for topic, pats in METHODS.items()
}


def classify_method(topic: str, text: str) -> str | None:
    """Return the method signature for a question, or None if no rule matches."""
    for name, rx in COMPILED.get(topic, []):
        if rx.search(text):
            return name
    return None
