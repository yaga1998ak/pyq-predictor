"""English hierarchical tagger: family -> subtype -> micro-archetype.

Level 1/2 come from SSC's question templates, which are highly stereotyped
("Select the option that can be used as a one-word substitute for ...").

Level 3 for grammar families is derived differently, and this is the useful part:
in Sentence Improvement and Error Spotting the four options are VARIANTS OF ONE
SEGMENT, so diffing them reveals the rule actually under test. If the options
differ only in is/are, the item tests subject-verb agreement; if in a/an/the, it
tests articles. That recovers the grammar-rule distribution (§8) from data rather
than from a hand-written guess about what SSC "usually" asks.

Normalisation and the squashed matching tier are inherited from the Reasoning
tagger, where they were needed for ligatures ("the ﬁrst") and for the de-spaced
2022 papers -- which matters even more here: 158 of 2022's 160 English questions
extract without inter-word spaces.

2021 is excluded from the corpus entirely, per instruction: 53% recall and 65%
blank stems make it noise rather than evidence.

BLIND DISCIPLINE: families were authored from 2022-2024 stems only. Compact
reconstruction phrasing is handled by rules authored on a declared dev slice of
2025 (every 4th shift), with the held-out remainder used for evaluation.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from reasoning_tag import normalize

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"

DROP_YEARS = {2021}

# ---------------------------------------------------------------- families
# (family, subtype, pattern). Order matters: cloze must beat generic
# fill-in-the-blank, and "substitute the underlined segment" must beat the
# broader vocabulary rules that also mention "underlined".
RULES: list[tuple[str, str, str]] = [
    ("cloze_test", "numbered_blank",
     r"fill in blank (number|no\.?)|answer for blank (number|no\.?)|"
     r"blank (number|no\.?) ?\d"),

    ("one_word_substitution", "group_of_words",
     r"one[\s\-]?word substitut(e|ion)|"
     r"single word (for|that) (the )?(given )?group"),

    ("active_passive_voice", "to_passive",
     r"expresses the given sentence in passive voice|"
     r"passive form of the given sentence|change.{0,20}into passive|"
     r"from active to passive|"
     r"convert the sentence.{0,60}(to|into) (its )?passive"),
    ("active_passive_voice", "to_active",
     r"expresses the given sentence in active voice|"
     r"active form of the given sentence|change.{0,20}into active|"
     r"from passive to active|"
     r"convert the sentence.{0,60}(from its )?passive voice structure|"
     r"passive to active voice"),

    ("direct_indirect_speech", "reported_speech",
     r"indirect speech|direct speech|reported speech|"
     r"indirect form of the given sentence|direct form of the given sentence|"
     r"indirect narration|narration of the given sentence"),

    ("para_jumbles", "sentence_order",
     r"sentences of a paragraph are given below in jumbled order|"
     r"jumbled order.{0,40}arrange|"
     r"rearrange the following sentences|"
     r"coherent paragraph"),
    # "Parts of the following sentence have been given as options. Select the
    # option that CONTAINS AN ERROR." is error-spotting, not a jumble. The
    # para_jumble rule claimed it first and mislabelled the family, so the
    # error-spotting cue is checked before any ordering cue.
    ("spotting_errors", "parts_as_options",
     r"(select|find) the (part|option|segment) that contains (an|the) error|"
     r"contains an error"),
    ("para_jumbles", "part_order",
     r"parts of (a|the) (following )?sentence are given below in jumbled|"
     r"parts of the following sentence have been given as options|"
     r"rearrange the parts of the sentence|"
     r"order of the segments"),

    ("spotting_errors", "four_segments",
     r"split into four segments|divided into (four )?(segments|parts)|"
     r"identify the segment that contains (the|an) (grammatical )?error|"
     r"underlined and given as options|"
     r"option that contains (an|the) error|"
     r"find the part of the sentence that contains an error|"
     r"one of them may contain an error"),
    ("spotting_errors", "no_error_option",
     r"if there is no error|sentence (is|has) no error"),

    ("spelling_correction", "incorrectly_spelt",
     r"incorrectly spelt|misspelt|contains a spelling error|"
     r"spelling error"),
    ("spelling_correction", "correctly_spelt",
     r"correctly spelt|no spelling errors|"
     r"spot the correct spelling|correct spelling of"),

    # Family absent from the original taxonomy, discovered in the 2025 dev slice:
    # "Select the sentence containing the homonym of the highlighted word."
    ("homonyms", "homonym_in_sentence",
     r"\bhomonym|\bhomophone"),

    ("idioms_and_phrases", "idiom_meaning",
     r"\bidiom\b|appropriate meaning of the|"
     r"meaning of the (given|underlined) idiom|"
     r"meaning of the (underlined |given )?(phrase|expression)"),

    # 2025 reconstructions say "highlighted" where official papers say
    # "underlined"; both must be accepted or the family reads 0.00/shift in 2025.
    ("sentence_improvement", "underlined_segment",
     r"substitute the underlined (segment|part|phrase)|"
     r"improve the underlined (part|segment)|"
     r"option that will improve|"
     r"replace the highlighted (part|segment|phrase)|"
     r"most suitable option to replace"),
    ("sentence_improvement", "underlined_word",
     r"substitute the underlined word|"
     r"word segment for the underlined"),

    # Each family needs a LONG alternative as well as the bare word: the squashed
    # tier requires a >=10-char match, and "synonym" squashes to 7 chars, so
    # de-spaced 2022 papers matched nothing on the bare word alone.
    ("antonyms", "antonym",
     r"\bantonym\b|appropriate antonym|"
     r"antonym of the (given|underlined|bracketed) word"),
    ("synonyms", "synonym",
     r"\bsynonym\b|appropriate synonym|"
     r"synonym of the (given|underlined|bracketed) word|"
     r"synonym to substitute the underlined"),

    ("reading_comprehension", "passage_question",
     r"according to the (passage|author)|"
     r"the author (of the passage )?(implies|suggests|states|means)|"
     r"main idea of the passage|"
     r"(most )?(suitable|appropriate) title for the passage|"
     r"the passage (is|deals|suggests|implies)"),

    ("grammar_usage", "targeted_rule",
     r"grammatically correct sentence|"
     r"correctly uses the (indefinite|definite) article|"
     r"correct (use|usage) of"),

    ("fill_in_the_blanks", "single_blank",
     r"fill in the blank|to fill in the blank|"
     r"complete the (given )?sentence"),
]

COMPILED = [(f, s, re.compile(p, re.I)) for f, s, p in RULES]


def _despace(pattern: str) -> str | None:
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
for _f, _s, _p in RULES:
    _b = _despace(_p)
    if _b:
        DESPACED.append((_f, _s, re.compile(_b, re.I)))

_SQUASH = re.compile(r"[\s,\-.'’]+")
MIN_SQUASHED_MATCH = 10


# ------------------------------------------------- grammar micro-archetypes
# Closed-class markers. If the options differ ONLY in one of these sets, that
# set names the rule under test.
RULE_MARKERS = {
    "subject_verb_agreement": {"is", "are", "was", "were", "has", "have", "had",
                               "does", "do", "did"},
    "article": {"a", "an", "the"},
    "preposition": {"in", "on", "at", "to", "for", "with", "by", "from", "of",
                    "into", "onto", "upon", "about", "over", "under", "between",
                    "among", "since", "until", "against", "towards", "through"},
    "pronoun": {"he", "she", "it", "they", "them", "him", "her", "his", "their",
                "its", "who", "whom", "whose", "which", "that", "himself",
                "herself", "themselves", "myself"},
    "tense_form": {"been", "being", "will", "would", "shall", "should", "can",
                   "could", "may", "might", "must"},
    "degree_comparison": {"more", "most", "less", "least", "better", "best",
                          "worse", "worst", "than", "as", "so"},
    "quantifier": {"much", "many", "few", "little", "fewer", "less", "some",
                   "any", "each", "every", "either", "neither", "both", "all"},
    "conjunction": {"and", "but", "or", "nor", "yet", "because", "although",
                    "though", "unless", "while", "whereas", "since", "if"},
}

WORD_RX = re.compile(r"[A-Za-z']+")


def grammar_rule(options) -> str | None:
    """Infer the rule under test by diffing the option variants.

    Only the words that DIFFER across options are informative -- the shared
    words are the carrier sentence. If the differing set falls inside one
    closed-class marker set, that is the rule. Verb-form differences that are not
    closed-class (e.g. "go"/"went"/"gone") fall back to tense_form via suffixes.
    """
    if len(options) < 4:
        return None
    toks = [set(w.lower() for w in WORD_RX.findall(t)) for _, t in options]
    shared = set.intersection(*toks)
    diff = set.union(*toks) - shared
    if not diff:
        return None

    scores = {}
    for rule, marks in RULE_MARKERS.items():
        hit = diff & marks
        if hit:
            scores[rule] = len(hit) / len(diff)
    if scores:
        best = max(scores, key=scores.get)
        # Require the marker class to explain a real share of the variation,
        # otherwise a single stray "the" would label every item as `article`.
        if scores[best] >= 0.5:
            return best

    # Inflectional variation with no closed-class marker: -ing / -ed / -s.
    suff = sum(1 for w in diff if re.search(r"(ing|ed|es|s)$", w))
    if suff >= max(2, len(diff) // 2):
        return "verb_form_inflection"
    if scores:
        return max(scores, key=scores.get)
    return "lexical_choice"


# Section-boundary bleed veto. English is the LAST numbering block, so when a
# numbering reset is invisible (markers lost on both sides of it) the preceding
# section's tail runs into it. Observed leaks: "who is serving as the Chief
# Minister of Chhattisgarh", "The Electricity (Amendment) Bill, 2022", "The
# railway system connecting St-Petersburg to Vladivostok is ___".
FOREIGN = {
    "general_awareness": re.compile(
        r"chief minister|prime minister|lok sabha|rajya sabha|amendment\s*\)?\s*bill|"
        r"\bbill,? 20\d{2}|constitution|dynasty|\briver\b|census|"
        r"capital of|olympic|padma|yojana|scheme|governor|"
        r"who among the following is|railway system connecting|"
        r"folk dance|muslim league|fundamental right|five year plan|"
        r"programme launched|guru of|\bvalley\b|\btemple\b|"
        r"which of the following states is", re.I),
    "quant": re.compile(
        r"profit|cost price|selling price|compound interest|simple interest|"
        r"\bratio of\b|per annum|litres|cistern|circumference|hypotenuse|"
        r"find the value of", re.I),
    "reasoning": re.compile(
        r"in a certain code|odd one out|letter[\s\-]?cluster|"
        r"replace the question mark|related in the same way|"
        r"mirror image|dice\b", re.I),
}
ENGLISH_ANCHOR = re.compile(
    r"synonym|antonym|idiom|spelt|spelling|blank|passage|paragraph|"
    r"underlined|segment|voice|speech|narration|jumbled|substitute|"
    r"one[\s\-]?word|grammatical|sentence", re.I)

FOREIGN_SQUASHED = {k: re.compile(v.pattern.replace(r"\b", "").replace(" ", ""), re.I)
                    for k, v in FOREIGN.items()}
ENGLISH_ANCHOR_SQUASHED = re.compile(
    ENGLISH_ANCHOR.pattern.replace(r"\b", "").replace(" ", ""), re.I)


def foreign_section(stem: str) -> str | None:
    text = normalize(stem)
    squashed = _SQUASH.sub("", text)
    if ENGLISH_ANCHOR.search(text) or ENGLISH_ANCHOR_SQUASHED.search(squashed):
        return None
    for sec, rx in FOREIGN.items():
        if rx.search(text):
            return sec
    for sec, rx in FOREIGN_SQUASHED.items():
        if rx.search(squashed):
            return sec
    return None


VOCAB_FAMILIES = {"synonyms", "antonyms", "one_word_substitution",
                  "idioms_and_phrases", "spelling_correction"}
GRAMMAR_FAMILIES = {"spotting_errors", "sentence_improvement",
                    "active_passive_voice", "direct_indirect_speech",
                    "grammar_usage", "fill_in_the_blanks"}


def classify(stem: str) -> tuple[str | None, str | None, str]:
    text = normalize(stem)
    for fam, sub, rx in COMPILED:
        if rx.search(text):
            return fam, sub, "direct"
    squashed = _SQUASH.sub("", text)
    for fam, sub, rx in DESPACED:
        m = rx.search(squashed)
        if m and len(m.group(0)) >= MIN_SQUASHED_MATCH:
            return fam, sub, "squashed"

    # Structural fallback: terse reconstructions often drop the instruction
    # entirely and print only a sentence with a gap ("The railway system
    # connecting St-Petersburg to Vladivostok is ____."). A gap plus no other
    # family match is a single-blank completion item.
    if re.search(r"_{3,}|\.{4,}", text):
        return "fill_in_the_blanks", "single_blank", "structural_gap"
    return None, None, "none"


def main() -> None:
    recs = json.load(open(OUT / "english_questions.json"))
    recs = [r for r in recs if r["year"] not in DROP_YEARS]
    passage_blanks = {p["passage_id"]: p["n_blanks"]
                      for p in json.load(open(OUT / "english_passages.json"))}

    kept, blank, vetoed = [], [], []
    for r in recs:
        stem = normalize(r["stem"])
        if len(stem) < 20:
            r["family"] = r["subtype"] = r["micro"] = None
            r["tag_route"] = "blank"
            blank.append(r)
            continue
        fs = foreign_section(r["stem"])
        if fs:
            r["section_conflict"] = fs
            vetoed.append(r)
            continue
        fam, sub, how = classify(r["stem"])
        r["family"], r["subtype"], r["tag_route"] = fam, sub, how
        # Passage-linked fallback. The discriminator is the PASSAGE, not the
        # question: a cloze passage carries numbered blanks, an RC passage does
        # not. Defaulting all passage-linked questions to RC (the first version)
        # labelled every 2025 cloze item as reading_comprehension, which drove
        # cloze to 0.00/shift in 2025 against 2.33-4.44 in official years, and RC
        # to 4.62 against ~0.05. Neither happened in the exam; both were this rule.
        if r.get("passage_id") and fam is None:
            nb = passage_blanks.get(r["passage_id"], 0)
            if nb >= 3:
                r["family"], r["subtype"] = "cloze_test", "numbered_blank"
            else:
                r["family"], r["subtype"] = "reading_comprehension", "passage_question"
            r["tag_route"] = "passage_link"
            fam = r["family"]
        r["micro"] = (grammar_rule(r["options"])
                      if fam in GRAMMAR_FAMILIES else None)
        kept.append(r)

    usable = kept + blank
    shifts = len({(r["exam_date"], r["shift"]) for r in usable})
    tagged = [r for r in kept if r["family"]]

    print("=" * 96)
    print("ENGLISH TAGGER — coverage (2021 excluded)")
    print("=" * 96)
    print(f"questions           : {len(usable)}  across {shifts} shifts")
    print(f"blank stems         : {len(blank)}")
    print(f"vetoed (bleed)      : {len(vetoed)}  {dict(Counter(r['section_conflict'] for r in vetoed))}")
    print(f"tagged              : {len(tagged)}")
    print(f"coverage (non-blank): {100*len(tagged)/max(len(kept),1):.1f}%")
    print(f"route: {dict(Counter(r['tag_route'] for r in kept))}")

    print("\n" + "=" * 96)
    print("FAMILY x SUBTYPE (per shift)")
    print("=" * 96)
    by = defaultdict(Counter)
    for r in tagged:
        by[r["family"]][r["subtype"]] += 1
    for f in sorted(by, key=lambda k: -sum(by[k].values())):
        tot = sum(by[f].values())
        print(f"\n{f:<28}{tot:>5}{tot/shifts:>8.2f}/shift")
        for s, c in by[f].most_common():
            print(f"    {s:<26}{c:>5}{c/shifts:>8.2f}")

    print("\n" + "=" * 96)
    print("GRAMMAR MICRO-ARCHETYPES (rule under test, from option diffs)")
    print("=" * 96)
    gm = Counter(r["micro"] for r in tagged if r["micro"])
    for k, v in gm.most_common():
        print(f"  {k:<30}{v:>5}{v/shifts:>8.2f}/shift")

    print("\n" + "=" * 96)
    print("PER-YEAR")
    print("=" * 96)
    print(f"{'Year':<6}{'Shifts':>7}{'Qs':>6}{'Tagged':>8}{'Cov':>7}{'Blank':>7}")
    print("-" * 96)
    for y in sorted({r["year"] for r in usable}):
        u = [r for r in usable if r["year"] == y]
        k = [r for r in kept if r["year"] == y]
        t = [r for r in k if r["family"]]
        sh = len({(r["exam_date"], r["shift"]) for r in u})
        b = sum(1 for r in blank if r["year"] == y)
        print(f"{y:<6}{sh:>7}{len(u):>6}{len(t):>8}"
              f"{100*len(t)/max(len(k),1):>6.0f}%{b:>7}")

    untag = [r for r in kept if not r["family"]]
    print(f"\nUntagged (non-blank): {len(untag)}")
    for r in untag[:12]:
        print("  -", normalize(r["stem"])[:100])

    (OUT / "english_tagged.json").write_text(json.dumps(usable, indent=2))
    print(f"\nWrote {OUT/'english_tagged.json'}")


if __name__ == "__main__":
    main()
