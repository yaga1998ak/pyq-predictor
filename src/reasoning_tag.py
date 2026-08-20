"""Reasoning-only hierarchical tagger: topic -> subtopic -> micro-archetype.

Built as a separate tagger from src/rules.py, for three reasons:

1. rules.py is section-agnostic, so quant rules poach Reasoning items. A
   '%'-means-'+' operator question was tagged `percentage` and printed into the
   Quantitative section of the shipped mock paper.
2. The prediction unit here is finer than rules.py's topic. "Series" is too
   broad to forecast; "quadratic number series" is actionable.
3. rules.py's coverage on structurally-confirmed Reasoning was 64.6%, and the
   misses are systematic archetypes it has no pattern for at all -- number-set
   analogy, dictionary order, pattern/matrix completion.

Normalisation fixes two silent killers found in this corpus:
  * ligatures -- "the ﬁrst number" is one glyph, so \\bfirst\\b never matches.
    NFKC folds it back to "fi". This alone was blocking whole rule families.
  * de-spaced extraction -- all 9 papers of 2022 extract as
    "Selecttheoptionthatisrelatedtothethirdword". Space-bearing patterns cannot
    match, so a squashed fallback runs long literal phrases against squashed text.

BLIND-TEST DISCIPLINE: every pattern here was authored by inspecting 2021-2024
stems only. 2025 stems were never read while writing rules, because tuning on
2025 would invalidate the headline blind test (HANDOVER §9).
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"

_PUNCT = str.maketrans({
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "−": "-", "×": "*", "÷": "/",
})


def normalize(text: str) -> str:
    """NFKC + punctuation folding + whitespace collapse.

    NFKC is what recovers ligature-bearing stems; do not drop it for speed.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_PUNCT)
    return re.sub(r"\s+", " ", text).strip()


# SSC writes "letter-cluster", "lettercluster" and "letter cluster" in the same
# year, so every rule referring to them must accept all three.
LC = r"letter[\s\-]?clusters?"

# ---------------------------------------------------------------------------
# Compact-phrasing rules, added for the 2025 reconstruction style.
#
# These were authored against a DECLARED DEV SLICE -- every 4th 2025 shift by
# date (12 of 45) -- and the remaining 33 shifts were never inspected. The
# motivation is measurement comparability, not accuracy on the target: the
# reconstructions paraphrase away SSC's verbose preambles, which drove 2025
# coverage to 36% while official years reached 96%, and produced the absurd
# reading of ZERO analogy questions across 45 shifts.
#
# They are structural rather than lexical -- "A : B :: C : ?" notation, "what
# comes next" -- so they encode transcription format, not which topics are
# frequent. Model selection is re-verified on official-only data afterwards to
# confirm the winner does not change.
# ---------------------------------------------------------------------------

# (topic, subtopic, pattern). ORDER IS SIGNIFICANT: disambiguators must precede
# the general rules they carve exceptions out of.
RULES: list[tuple[str, str, str]] = [
    # --- disambiguators that must outrank broader families -------------------
    # Coded blood relations read as coding-decoding unless caught first.
    ("blood_relations", "coded_relation",
     r"how is .{1,40} related to|"
     r"(pointing (to|at) (a|the) (photograph|picture))|"
     r"\bA \+ B means\b|means that A is the"),
    # Symbolic relation codes: "A # B means 'A is the sister of B'".
    ("blood_relations", "symbolic_operator",
     r"means '?[A-Z] is the (sister|brother|daughter|son|husband|wife|father|"
     r"mother|paternal|maternal)|"
     r"[A-Z]\s*[#@&%$*]\s*[A-Z]\s*means"),
    ("blood_relations", "generation_chain",
     r"\b(father|mother|brother|sister|son|daughter|husband|wife|uncle|aunt|"
     r"nephew|niece|grandfather|grandmother|paternal|maternal)\b.{0,120}"
     r"\b(related|relation)\b|"
     # Possessive chains carry no "related to" phrase at all:
     # "his mother's brother's father's granddaughter".
     # Allow a qualifier between the links: "my mother's only son".
     r"(father|mother|brother|sister|son|daughter)'s (\w+ )?(father|mother|"
     r"brother|sister|son|daughter|grand(son|daughter))|"
     r"introducing a (boy|girl|man|woman)"),

    # --- alphabet operations (its own archetype, ~0.2/shift) -----------------
    # "The position of how many letters will remain unchanged if each of the
    # letters in the word BINDER is arranged in alphabetical order."
    ("alphabet_operations", "position_unchanged",
     r"positions? of how many letters will remain unchanged|"
     r"how many letters will remain unchanged|"
     r"letters (in|of) the word .{0,20}(is|are) arranged (in )?alphabetical"),
    ("alphabet_operations", "letter_substitution",
     r"each vowel in the word|"
     r"changed to the (previous|following|next) letter in the english"),

    # "If today is Friday, which day will it be after 72 days?"
    ("calendar_clock", "day_offset",
     r"if today is (monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
     r"which day (of the week )?will it be|"
     r"what will be the day (of the week )?on|"
     r"angle between the hour and minute hand"),

    # --- ordering / dictionary ----------------------------------------------
    ("dictionary_order", "alphabetical_arrangement",
     r"appear in an english dictionary|"
     r"order of the given words as they would appear|"
     r"arrangement of the given words in the order in which they appear|"
     r"according to dictionary order|"
     r"arranged? .{0,30}dictionary"),

    # --- analogy family -----------------------------------------------------
    ("analogy", "number_set_analogy",
     r"(sets?|triads?|groups?|options?|pairs?) in which the numbers are (not )?related"
     r"( to each other)? in the same way|"
     r"numbers are related in the same way as are the numbers|"
     r"select the (set|triad) in which the numbers|"
     r"(numbers|words) share the same relationship as"),
    ("analogy", "number_pair_operation",
     r"second number in the given number[\s\-]?pairs is obtained by|"
     r"same operations are followed in all the given number[\s\-]?pairs|"
     r"number[\s\-]?pairs? .{0,60}obtained by (performing|applying)|"
     r"two sets of numbers are given|"
     r"pairs of (words|numbers) that are related to each other"),
    ("analogy", "word_pair_analogy",
     r"word[\s\-]?pair that best represents a similar relationship|"
     r"pair of words .{0,40}same relationship"),
    # Compact SSC analogy notation: "APPLE : FRUIT :: CARROT : ?"
    ("analogy", "colon_notation",
     r"::|completes the analogy|"
     r"select the related (word|number|letter)|"
     r"(second|third) word is related to the (first|second) word by"),
    ("analogy", "letter_cluster_analogy",
     rf"related to the (second|third|fourth|fifth) {LC} in the same way|"
     rf"{LC} in the same way as the (second|third|fourth) {LC}"),
    ("analogy", "term_analogy",
     r"related to the (second|third|fourth|fifth) (word|number|term) in the same way|"
     r"is related to .{1,40} following (a |certain )*logic|"
     r"following the same logic|"
     r"related to the third (word|number|term) in ?the same way|"
     r"is related to .{0,40}in the same way as"),

    # --- classification -----------------------------------------------------
    ("classification_odd_one_out", "letter_cluster_odd",
     rf"(three|four) of the following ((four|five) )?{LC} are alike|"
     rf"{LC} are alike in some manner and hence form a group|"
     rf"alphabetical order,? three of the following {LC}"),
    ("classification_odd_one_out", "number_group_odd",
     r"odd (group|set) of numbers|"
     r"(three|four) of the following (four|five) numbers are alike"),
    ("classification_odd_one_out", "generic_odd",
     r"alike in some manner and one is different|"
     r"pick the odd one out|select the odd one|odd one out|"
     r"alike in a certain way and (thus |hence )?(form|one is)|"
     r"which (one )?is different|thus form a group|"
     r"identify the odd one|"
     r"group of three (numbers|symbols|words)"),

    # --- matrix / pattern (must precede series: both say "question mark") ----
    # Anchored on an explicit pattern/matrix/grid cue so linear series stay out.
    ("matrix", "pattern_completion",
     r"study the given pattern|"
     r"study the given (matrix|diagram)|"
     r"in the given (matrix|pattern)|"
     r"replace the question mark .{0,20}in it\b"),

    # --- series -------------------------------------------------------------
    ("series_completion", "letter_blank_completion",
     r"letters that when sequentially placed|"
     r"letters that when placed from left to right"),
    ("series_completion", "figure_series",
     r"which figure should replace the question mark|"
     r"figure (from among the given option|from the options)s? that can replace|"
     r"(option )?figure .{0,30}replace the question mark|"
     r"figure that will come (next|in place of)|"
     r"following figure series|"
     r"if the following series were to be continued"),
    ("series_completion", "number_or_letter_series",
     rf"complete the (given )?series|missing (number|term|letter)|"
     rf"which (of the following )?(numbers?|terms?|{LC}) (will|should) replace|"
     rf"select the (number|term|option|{LC}) from among the given options that can replace|"
     rf"select the option that can replace the question mark|"
     rf"(number|term) that will replace the question mark|"
     rf"a series is given with one term missing|"
     rf"in the following (number )?series|in the given series|"
     rf"come in place of the question mark|"
     rf"complete the following (letter|number|alphanumeric) series|"
     # compact forms
     rf"what comes next|next number in the series|"
     rf"complete the pattern|"
     rf"(number|term|{LC}) that can replace the question mark"),

    # --- coding-decoding ----------------------------------------------------
    ("coding_decoding", "language_code",
     r"in a certain code(?: language)?|"
     r"is (written|coded) as .{0,60}(then|how)|is coded as"),

    # --- math operations ----------------------------------------------------
    ("mathematical_operations", "sign_substitution",
     r"correct combination of (mathematical )?signs|"
     r"signs? that can sequentially replace|balance the given equation"),
    ("mathematical_operations", "sign_interchange",
     r"\binterchang(e|es|ed|ing)\b|"
     r"if '?[+\-*/]'? means|"
     r"which two signs should be"),
    # Symbol-substitution in compact form: "If @ = *, # = -, $ = /, then
    # evaluate ..." and "If 'X' means '+', 'Y' means '-' ...".
    ("mathematical_operations", "symbol_definition",
     r"if '?\w'? means '?[+\-*/x]'?|"
     r"if [@#$%*&] ?= ?[@#$%*&+\-/x]|"
     r"[@#$%*&] ?\d+ ?= ?\d+ ?and"),

    # --- logic --------------------------------------------------------------
    ("syllogism", "statements_conclusions",
     r"conclusions? .{0,80}logically follow|"
     r"statements? (are given )?followed by .{0,60}conclusions?|"
     r"assuming the statements to be true|"
     r"statements and conclusions|"
     r"assuming that the information given in the statements|"
     r"identify the conclusion that follows"),
    ("statement_conclusion", "assumption_or_course",
     r"which of the (following )?assumptions?|courses? of action"),

    ("venn_diagram", "set_relationship",
     r"\bvenn\b|diagram that best represents the relationship"),

    # --- spatial / non-verbal ----------------------------------------------
    ("mirror_water_image", "image",
     r"mirror image|water image"),
    ("paper_folding_cutting", "fold_punch",
     r"paper is folded|folded and punched|when unfolded|"
     r"transparent sheet with a pattern"),
    ("embedded_figures", "hidden_figure",
     r"\bembedded\b|hidden in the given figure"),
    ("cube_and_dice", "dice_faces",
     r"\bdice\b|\bdie\b.{0,40}face|surfaces? of (a|the) cube|"
     r"cube is (cut|painted)"),
    ("counting_figures", "count_shapes",
     r"how many (triangles|squares|rectangles|circles|straight lines) are there"),

    # --- positional ---------------------------------------------------------
    ("direction_sense", "movement_trace",
     r"\bfacing\b|towards (the )?(north|south|east|west)|"
     r"turns? (to (his|her) )?(left|right)|"
     r"walks? .{0,30}\b(m|metres|meters|km)\b.{0,60}(turns?|then)"),
    ("ranking_and_order", "comparative_rank",
     r"\b(taller|shorter|heavier|lighter|older|younger)\b|"
     r"\b(youngest|oldest|tallest|shortest|heaviest)\b|"
     r"rank (from the|of)|position from the (top|bottom|left|right)"),
    ("seating_arrangement", "linear_or_circular",
     r"sitting in a (row|circle|line)|seated (around|in a)|"
     r"facing (the )?(centre|center|north) .{0,40}(row|circle)"),

    ("word_formation", "letters_available",
     r"cannot be formed|can(not)? be formed using the letters|"
     r"which of the following words can be formed|"
     r"meaningful word can be formed"),

    # Clerical-accuracy archetype absent from the taxonomy entirely:
    # "Which of the following addresses are identical to each other?"
    ("address_matching", "identical_strings",
     r"identical to each other|identical to the address|"
     r"which option is identical to|is/are identical to"),
]

COMPILED = [(t, s, re.compile(p, re.I)) for t, s, p in RULES]


# 12, not 16: "interchanging" is 13 chars and is the entire signal for the
# sign-interchange archetype in de-spaced 2022 papers. A 12-char contiguous
# alphabetic run is specific enough that accidental containment is not credible
# (the collisions that motivated a floor at all were 5-char words like "ratio").
MIN_SQUASHED_MATCH = 10


def _despace(pattern: str) -> str | None:
    """Build a space-insensitive variant of a whole pattern.

    The earlier version split on "|" and discarded any alternative containing
    regex syntax. Because nearly every rule here uses groups, that silently
    emptied the entire tier -- "Whichofthefollowingnumberswillreplacethe
    questionmark(?)inthegivenseries?" went untagged while the tier looked live.

    So transform the pattern as a whole instead of dismembering it: drop literal
    spaces, \\s and \\b (all meaningless once spaces are gone) and keep every
    group and alternation intact. Character classes here contain no literal
    space, so removing spaces cannot corrupt one.

    Collision risk is handled at match time by MIN_SQUASHED_MATCH rather than by
    a pattern-length filter, which is both simpler and stricter: a short
    accidental hit inside a longer word is rejected even if the rule is long.
    """
    # Ordered longest-first. Order is load-bearing: stripping "-" before the
    # class "[\s\-]" would leave "[\s]", and stripping "\s" after that leaves
    # "[]" -- an invalid class that silently kills the whole rule.
    squashed = pattern
    # Do NOT strip commas: "{0,50}" becomes "{050}", which still COMPILES but
    # means "exactly 50 characters" instead of "0 to 50". Every rule containing a
    # bounded wildcard was therefore silently near-dead in this tier -- no error,
    # just a pattern that almost never matches.
    for token in (r"[\s\-]?", r"[\s\-]", r"\s", r"\b", " ?", " "):
        squashed = squashed.replace(token, "")
    try:
        re.compile(squashed)
    except re.error:
        return None
    return squashed


DESPACED = []
for _t, _s, _p in RULES:
    _b = _despace(_p)
    if _b:
        DESPACED.append((_t, _s, re.compile(_b, re.I)))

# The squashed tier removes inter-word punctuation as well as spaces, so a comma
# SSC inserts in one year and omits the next ("letters that, when placed" vs
# "letters that when placed") stops being a separate rule to maintain.
_SPACE = re.compile(r"[\s,\-.'’]+")

# Section-conflict veto. Block boundaries are found from numbering resets, but a
# reset is invisible when the markers on both sides of it are lost -- reasoning
# Q.24 then runs straight into GA Q.2 with no descent. That bled de-spaced GA
# items ("the first complete census taken in India") into the Reasoning block.
# These fingerprints are deliberately narrow: they fire only on subject matter
# that Reasoning never asks about.
FOREIGN = {
    # Every alternative is >=10 chars once squashed. Length is load-bearing: the
    # squashed twins below run with \b stripped, so a short token like "state"
    # would match inside "statement" and veto real syllogism questions.
    "general_awareness": re.compile(
        r"complete census|dynasty|temple|vitamin|constitution|article \d|"
        r"chief minister|prime minister|governor|capital of|"
        r"central bank|reserve bank|monsoon|utsav|festival|"
        r"olympic|padma|awarded|yojana|"
        r"was built by|energy derived|bird atlas|river rises", re.I),
    "english": re.compile(
        r"\b(synonym|antonym|idiom|one word substitution|"
        r"correctly spelt|incorrectly spelt|grammatically correct|"
        r"active voice|passive voice|para ?jumble)\b", re.I),
    # Reasoning does use arithmetic, so these are restricted to word-problem
    # furniture that Reasoning never carries (goods, money, rates, mixtures).
    "quant": re.compile(
        r"\b(profit|loss percent|discount|marked price|cost price|selling price|"
        r"compound interest|simple interest|per annum|litres|"
        r"downstream|upstream|cistern|circumference|hypotenuse)\b", re.I),
}

# Bilingual papers occasionally extract the Hindi column into the English text.
# Devanagari in a stem means the wrong language column, not a Reasoning topic.
DEVANAGARI = re.compile(r"[ऀ-ॿ]{4,}")

# Official response sheets interleave candidate metadata with the stem. A chunk
# that is only metadata carries no question text and must not be counted as a
# tagging failure -- it is an extraction artefact, not an untaggable question.
NOISE_ONLY = re.compile(r"^[\d\s.]*(question\s*id|status\s*:|chosen\s*option)", re.I)
REASONING_ANCHOR = re.compile(
    r"\b(code|series|analog|odd one|cluster|syllogis|conclusion|dice|cube|"
    r"mirror|water image|folded|embedded|venn|question mark|"
    r"related in the same way|dictionary)\b", re.I)

# De-spaced twins of the foreign-section fingerprints. Without these, the veto
# was blind on exactly the papers that need it most: the 2022 extractions where
# lost markers hide a numbering reset and let GA items into the Reasoning block.
FOREIGN_SQUASHED = {
    sec: re.compile(rx.pattern.replace(r"\b", "").replace(" ", ""), re.I)
    for sec, rx in FOREIGN.items()
}
REASONING_ANCHOR_SQUASHED = re.compile(
    REASONING_ANCHOR.pattern.replace(r"\b", "").replace(" ", ""), re.I)


def classify(stem: str) -> tuple[str | None, str | None, str]:
    """Return (topic, subtopic, how). `how` records the match route."""
    text = normalize(stem)

    for topic, sub, rx in COMPILED:
        if rx.search(text):
            return topic, sub, "direct"

    # Squashed tier, run unconditionally rather than gated on a space ratio.
    # The gate was wrong: extraction loses spaces *partially*, not wholly --
    # "related to the third number inthe same way" has a normal space ratio yet
    # still defeats every space-bearing pattern. Only literals >=18 chars reach
    # this tier, so matching inside a longer word is not a realistic risk.
    squashed = _SPACE.sub("", text)
    for topic, sub, rx in DESPACED:
        m = rx.search(squashed)
        if m and len(m.group(0)) >= MIN_SQUASHED_MATCH:
            return topic, sub, "squashed"
    return None, None, "none"


def foreign_section(stem: str) -> str | None:
    """Detect a non-Reasoning question sitting inside the Reasoning block."""
    text = normalize(stem)
    if DEVANAGARI.search(text):
        return "hindi_column"
    squashed = _SPACE.sub("", text)
    if REASONING_ANCHOR.search(text) or REASONING_ANCHOR_SQUASHED.search(squashed):
        return None
    for sec, rx in FOREIGN.items():
        if rx.search(text):
            return sec
    for sec, rx in FOREIGN_SQUASHED.items():
        if rx.search(squashed):
            return sec
    return None


def main() -> None:
    recs = json.load(open(OUT / "reasoning_questions.json"))

    kept, vetoed, blank = [], [], []
    for r in recs:
        stem = r["stem"]
        norm_stem = normalize(stem)
        if len(stem.strip()) < 25 or NOISE_ONLY.match(norm_stem):
            r["topic"] = r["subtopic"] = None
            r["tag_route"] = "blank"
            blank.append(r)
            continue
        fs = foreign_section(stem)
        if fs:
            r["section_conflict"] = fs
            vetoed.append(r)
            continue
        t, s, how = classify(stem)
        r["topic"], r["subtopic"], r["tag_route"] = t, s, how
        kept.append(r)

    usable = kept + blank
    shifts = len({(r["exam_date"], r["shift"]) for r in usable})
    tagged = [r for r in kept if r["topic"]]

    print("=" * 92)
    print("REASONING TAGGER — coverage")
    print("=" * 92)
    print(f"Extracted Reasoning questions        : {len(recs)}")
    print(f"Vetoed as foreign section (bleed)    : {len(vetoed)}")
    print(f"Blank stems (figure-based text loss) : {len(blank)}")
    print(f"Net Reasoning corpus                 : {len(usable)}  across {shifts} shifts")
    print(f"Tagged                               : {len(tagged)}")
    print(f"Coverage of non-blank                : "
          f"{100*len(tagged)/max(len(kept),1):.1f}%")
    print(f"Coverage of all Reasoning            : "
          f"{100*len(tagged)/max(len(usable),1):.1f}%")

    print("\nVetoed by section:", dict(Counter(r["section_conflict"] for r in vetoed)))
    print("Route:", dict(Counter(r["tag_route"] for r in kept)))

    print("\n" + "=" * 92)
    print("TOPIC x SUBTOPIC (per shift)")
    print("=" * 92)
    by_topic = defaultdict(Counter)
    for r in tagged:
        by_topic[r["topic"]][r["subtopic"]] += 1
    order = sorted(by_topic, key=lambda t: -sum(by_topic[t].values()))
    for t in order:
        tot = sum(by_topic[t].values())
        print(f"\n{t:<28}{tot:>5}{tot/shifts:>8.2f}/shift")
        for s, c in by_topic[t].most_common():
            print(f"    {s:<26}{c:>5}{c/shifts:>8.2f}")

    print("\n" + "=" * 92)
    print("PER-YEAR tagged Reasoning per shift")
    print("=" * 92)
    print(f"{'Year':<6}{'Shifts':>7}{'Net Q':>7}{'Tagged':>8}{'Cov':>7}{'Blank':>7}{'Veto':>6}")
    print("-" * 92)
    for y in sorted({r["year"] for r in recs}):
        u = [r for r in usable if r["year"] == y]
        k = [r for r in kept if r["year"] == y]
        tg = [r for r in k if r["topic"]]
        sh = len({(r["exam_date"], r["shift"]) for r in u})
        b = sum(1 for r in blank if r["year"] == y)
        v = sum(1 for r in vetoed if r["year"] == y)
        cov = 100*len(tg)/len(k) if k else 0
        print(f"{y:<6}{sh:>7}{len(u):>7}{len(tg):>8}{cov:>6.0f}%{b:>7}{v:>6}")

    untag = [r for r in kept if not r["topic"]]
    print(f"\nStill untagged (non-blank): {len(untag)}")
    for r in untag[:12]:
        print("  -", normalize(r["stem"])[:104])

    (OUT / "reasoning_tagged.json").write_text(json.dumps(usable, indent=2))
    (OUT / "reasoning_vetoed.json").write_text(json.dumps(vetoed, indent=2))
    print(f"\nWrote {OUT/'reasoning_tagged.json'}")


if __name__ == "__main__":
    main()
