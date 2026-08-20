"""Deterministic rule-based tagger — free, instant, auditable.

Why this beats the local LLM on this corpus: SSC CGL questions are heavily
templated. "Select the most appropriate antonym of..." appears 120 times
verbatim; "In a certain code language" is always coding_decoding. A regex over
the question stem gets those right every single time, while a 7B model scored
35% because it collapsed distinct templates into whichever topic it had seen most.

Three properties an LLM tagger cannot offer:

  deterministic   the same question always gets the same label, so year-over-year
                  comparisons are not polluted by sampling noise
  auditable       every label traces to a named rule you can read and argue with
  free            no tokens, no hours -- the full corpus tags in about a second

The design accepts LOW COVERAGE as the price of high precision. A question that
matches no rule is left untagged rather than guessed at, because a wrong label is
worse than a missing one: missing questions shrink the sample, wrong ones bias it.
General awareness is deliberately sparse here -- it is knowledge-based with no
reliable surface templates, so most of it falls through by design.

Rules are ordered: the FIRST match wins, so specific patterns must precede
general ones. "Select the correct mirror image" must be tested before any generic
"select the correct" rule, or the general rule swallows it.

    python src/rules.py --papers data/parsed/papers.json --report
    python src/rules.py --papers data/parsed/papers.json --out data/tagged/rules.json
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

from schema import Taxonomy, load_papers, save_papers, REPO

# (topic, regex). Order is significant — first match wins.
RULES: list[tuple[str, str]] = [
    # ---------- Disambiguators: must precede the general rules they'd lose to ----------
    # "How is M related to Q?" is blood relations even when wrapped in a coding
    # frame ("R + T means R is the sister of T"). Without this first, the
    # coding_decoding rule claims it on "certain code language".
    ("blood_relations", r"how is .{1,40} related to|how are .{1,40} related"),
    # Syllogism vs statement_conclusion both say "Conclusions:". The separator is
    # quantified premises ("All pens are pencils"); a real-world statement is not
    # a syllogism.
    ("syllogism", r"\b(all|some|no)\s+\w+\s+(are|is)\s+\w+"),
    # Require BOTH a statement and a conclusion/assumption. Keying on "statement:"
    # alone false-positives on GK questions phrased "Complete the statement: the
    # Khelo India scheme selected ___ schools", which is general awareness.
    ("statement_conclusion", r"statements?\s*:.{0,500}(conclusion|assumption)|"
                             r"course of action|assumptions? (is|are) implicit"),
    # "Find the wrong term in the series" is a series question in SSC's own
    # framing, not odd-one-out — so it must beat the odd-one-out rule.
    ("series_completion", r"(incorrect|wrong|odd) (term|number) in the series|"
                          r"breaks the pattern|does not fit the series"),
    # A passage question is reading comprehension even when it asks about a word's
    # meaning, which would otherwise be claimed by the synonyms rule.
    # A passage with deleted words is a CLOZE, not comprehension -- and the
    # "read the passage" phrasing appears in both, so cloze must be tested first.
    ("cloze_test", r"\bcloze\b|passage with blanks|words have been deleted|"
                   r"blank number ?\(?\d"),
    ("reading_comprehension", r"read the (given )?passage|comprehension passage|"
                              r"the passage is mainly about|according to the passage|"
                              r"in the passage"),

    # ---------- English: unambiguous single-keyword templates ----------
    ("antonyms", r"\bantonym"),
    ("synonyms", r"\bsynonym|\bsimilar in meaning|closest in meaning"),
    ("idioms_and_phrases", r"\bidiom|\bphrasal verb|\bproverb"),
    ("one_word_substitution", r"one[- ]word subst|single word (for|substitution)|substitute for the given group"),
    ("spelling_correction", r"spelt correctly|spelled correctly|misspel|incorrectly spelt|"
                            r"correctly spelt|spelling error"),
    ("active_passive_voice", r"active to passive|passive to active|\bactive voice|\bpassive voice"),
    ("direct_indirect_speech", r"direct speech|indirect speech|reported speech|\bnarration\b"),
    ("para_jumbles", r"rearrange the following sentences|sentences? of a paragraph|"
                     r"correct order to (form|make)|jumbled"),
    ("spotting_errors", r"part of the sentence that (contains|has) (an )?error|"
                        r"segment.*(contains|has).*error|identify the error|no error"),
    ("sentence_improvement", r"improve the (underlined|bracketed)|substitute the underlined|"
                             r"replace the underlined|most suitable option to replace"),
    ("fill_in_the_blanks", r"fill in the blank|to fill in the blank|fill the blank|"
                           r"option to fill in|most appropriate option to fill"),
    ("spelling_correction", r"correctly spelt word|incorrectly spelt word|"
                            r"select the (correctly|incorrectly) spel"),

    # ---------- Reasoning: template phrases, most specific first ----------
    ("coding_decoding", r"certain code language|coded as|is written as|code for the word"),
    ("mirror_water_image", r"mirror image|water image"),
    ("paper_folding_cutting", r"paper.{0,30}folded|folded and (punched|cut)"),
    ("cube_and_dice", r"\bdice\b|\bdie is thrown|surfaces of a cube|cube is (cut|painted)"),
    ("embedded_figures", r"embedded|hidden figure|part of the given figure"),
    ("venn_diagram", r"venn diagram|best represents the relationship"),
    ("syllogism", r"conclusions?\s*(:|I\.)|which of the conclusions? logically"),
    ("statement_conclusion", r"statement.{0,20}conclusion|course of action|assumptions? (is|are) implicit"),
    ("blood_relations", r"how is .{1,40} related to|(father|mother|brother|sister|son|daughter|"
                        r"husband|wife|nephew|niece|uncle|aunt) of\b"),
    ("direction_sense", r"facing (north|south|east|west)|turns? (to the )?(left|right)|"
                        r"walks? \d+ ?(m|km)"),
    ("ranking_and_order", r"rank (of|from)|position from the (left|right|top|bottom)|"
                          r"stands? (\w+ )?from the"),
    ("word_formation", r"english (alphabetical|dictionary) order|dictionary order|"
                       r"alphabetical order|cannot be formed using the letters|"
                       r"letters? (remain|will remain) unchanged|"
                       r"extra letter|not found in the (main|given) word|"
                       r"using the letters of the word"),
    ("mathematical_operations", r"interchang\w+.{0,40}(sign|symbol)|"
                                r"if ['\"]?\+['\"]? (and|is)|balance the given equation"),
    ("matrix", r"\bmatrix\b|two matrices|classes? of a matrix"),
    ("classification_odd_one_out", r"odd one out|does not belong|not follow the same pattern|"
                                   r"breaks the pattern|which one is different|"
                                   r"three are alike|address that is the same"),
    ("analogy", r"(is|are) related .{0,50} in the same way|"
                r"related to the (first|second) (word|number|set)|"
                r"select the (option|set) that is related to|bears the same relation"),
    ("series_completion", r"missing (number|term|letter)|complete the series|what comes next|"
                          r"next in the series|continues the series|\bseries\b.{0,40}\?|"
                          r"should come in the place of|question mark.{0,50}series|"
                          r"figure.{0,40}series|following series"),

    # ---------- Quant: domain vocabulary ----------
    # Require quantitative context: "\bboat" alone claimed a general-awareness
    # question about the Nehru Trophy Boat Race.
    ("boats_and_streams", r"downstream|upstream|still water|\bboat\b.{0,60}(speed|current|stream|km|hour|rowing)"),
    ("pipes_and_cisterns", r"\bpipes?\b|\bcistern|\btank\b.{0,30}fill|fill the tank"),
    ("simple_and_compound_interest", r"simple interest|compound interest|\bper annum\b|\bprincipal\b"),
    ("profit_and_loss", r"profit|\bloss\b|cost price|selling price|marked price|\bdiscount\b"),
    ("mixture_and_alligation", r"mixture|alligation|\balloy\b|milk and water"),
    ("time_and_work", r"(complete|finish|do) (the|a) (same )?(work|task)|days? to complete|"
                      r"working together|collaborat\w+.{0,30}\bdays\b|\bmen\b.{0,25}\bdays\b"),
    ("time_speed_distance", r"\btrain\b|\bspeed\b|km/h|kmph|\bovertakes?\b|"
                            r"distance.{0,25}(travel|cover)"),
    ("height_and_distance", r"angle of (elevation|depression)"),
    ("trigonometry", r"\bsin\b|\bcos\b|\btan\b|\bcot\b|\bcosec\b|\bsec\b|trigonometric"),
    ("mensuration", r"\barea of\b|\bvolume\b|perimeter|circumference|surface area|"
                    r"cuboid|cylinder|\bcone\b|\bsphere\b|hemisphere|\btrapezium|\brhombus|\bparallelogram|shaped like a"),
    ("geometry", r"triangle|\bcircle\b|\bangle\b|parallel|\bchord\b|\bradius\b|"
                 r"quadrilateral|bisector|centroid|incentre|circumcentre"),
    ("data_interpretation", r"pie chart|bar (graph|chart)|the (table|graph) (shows|given)|"
                            r"study the (table|graph|data)"),
    ("number_system", r"\bH\.?C\.?F\b|\bL\.?C\.?M\b|divisible by|\bremainder\b|"
                      r"prime number|greatest number|smallest number"),
    ("percentage", r"percentage|\bper ?cent\b|\d+ ?%"),
    ("ratio_and_proportion", r"\bratio\b|proportion|\bdivided? (in|into) the ratio"),
    ("average", r"\baverage\b|\bmean\b of"),
    ("algebra", r"\bpolynomial|\bequation\b|value of x|if x ?=|a\^?2 ?\+ ?b\^?2|\bfactoris"),
    ("simplification", r"simplify|value of the following|\bBODMAS\b"),

    # ---------- General awareness: entity gazetteer ----------
    # GA has no question-stem templates -- "Who among the following..." says
    # nothing about the topic. The signal is in the NAMED ENTITIES instead, so
    # these rules match people, places, dynasties and institutions rather than
    # phrasing. Ordered oldest-to-newest within history so "Mughal" cannot be
    # claimed by a modern-era rule.
    ("history_ancient", r"indus valley|harappa|mohenjo|\bmauryan?\b|chandragupta|\bashoka\b|"
                        r"\bgupta (empire|dynasty|period)\b|\bbuddha\b|buddhism|jainism|"
                        r"mahavira|\bvedic\b|rig ?veda|upanishad|kushan|satavahana|"
                        r"\bchola\b|pallava|\bnalanda\b|takshashila"),
    ("history_medieval", r"delhi sultanate|slave dynasty|khil[jz]i|tughlaq|lodi|\bmughal|"
                         r"\bbabur\b|\bakbar\b|aurangzeb|shah jahan|humayun|jahangir|"
                         r"panipat|vijayanagara|\bshivaji\b|maratha|rana sanga|"
                         r"bhakti|\bsufi\b|razia|sher shah|\bkhalji\b"),
    ("history_modern", r"freedom (struggle|movement)|indian national congress|"
                       r"quit india|\b1857\b|gandhi|nehru|east india company|"
                       r"\bbritish (rule|raj|government)|jallianwala|dandi|"
                       r"simon commission|round table|subhas chandra|\bpartition of india|"
                       r"\bplassey\b|\bbuxar\b|swadeshi|\bviceroy\b"),
    ("polity_constitution", r"constitution|article \d+|fundamental (right|duty)|"
                            r"parliament|lok sabha|rajya sabha|amendment|\bschedule of\b|"
                            r"supreme court|high court|"
                            r"election commission|panchayat|directive principle"),
    ("economics", r"\bgdp\b|reserve bank|\brbi\b|inflation|repo rate|fiscal deficit|"
                  r"union budget|niti aayog|\bcensus\b|per capita income|"
                  r"\bwto\b|\bimf\b|world bank|monetary policy|\bgst\b|"
                  r"stock exchange|\bsensex\b|\bnifty\b"),
    ("geography_indian", r"\briver\b|\bmountain|\bplateau|national park|wildlife sanctuary|"
                         r"\bmonsoon\b|\bstate of india|\bganga\b|yamuna|godavari|"
                         r"brahmaputra|narmada|\bkrishna river|western ghats|eastern ghats|"
                         r"thermal power|\bdam\b|\bcanal\b|\bsoil\b|\btribal\b|"
                         r"located in (which|the) (state|indian)"),
    ("geography_world", r"\bcontinent\b|\bsahara\b|\bamazon\b|\bstrait\b|"
                        r"world'?s (largest|longest|highest|deepest)|"
                        r"\bequator\b|tropic of|\blatitude|\blongitude|time zone"),
    ("biology", r"\bvitamin\b|\bblood\b|\bcell\b|\benzyme|\bdisease|\bhormone|photosynthes|"
                r"\bbacteria|\bvirus\b|\bgene\b|chromosome|\bprotein\b|\bkidney|\bliver\b|"
                r"respiration|digestion"),
    ("chemistry", r"chemical (formula|symbol)|atomic (number|mass)|\bcompound\b|"
                  r"periodic table|\bacid\b|\balkali|\bmolecule|\bisotope|"
                  r"chemical (reaction|element)|\bph value|\balloy of\b"),
    ("physics", r"\bnewton|\bvelocity|\bgravit|\bmagnet|\brefract|\bohm\b|"
                r"unit of (force|power|energy|current)|\bcryogenic|\bsatellite\b|"
                r"\bwavelength|\bfrequency of|\blaser\b|semiconductor|\bthermodynam"),
    ("sports", r"\bolympic|\bworld cup|\btrophy\b|\btournament|\bmedal\b|"
               r"\bcricket|\bhockey\b|\bbadminton|\bkabaddi|\bchess\b|"
               r"commonwealth games|asian games|\bkhelo india"),
    ("schemes_and_policies", r"\byojana\b|\babhiyan\b|\bmission\b.{0,20}(launched|government)|"
                             r"government scheme|\bswachh\b|\bayushman\b|\bmgnrega\b|"
                             r"pradhan mantri|\bujjwala\b|\bjan dhan"),
    ("art_and_culture", r"classical dance|folk dance|\braga\b|"
                        r"(harvest|religious|cultural|dance|music|literary) festival|"
                        r"\bmusical instrument|\btemple\b|\bdance form|\bpainting\b|"
                        r"\bbihu\b|\bonam\b|\bpongal\b|\bdiwali\b|\bholi\b|\bbaisakhi|\bhornbill|"
                        r"\bsantoor|\bsitar\b|\btabla\b|\bveena\b|\bpadma (shri|bhushan|vibhushan)|"
                        r"\bunesco\b|world heritage|\bhandicraft"),
    ("static_gk", r"headquarters of|\bcapital of\b|currency of|"
                  r"first (person|indian|woman) to|\bfounded in\b|"
                  r"\bawarded\b|\bindex\b.{0,30}(published|released)|"
                  r"book (written|authored) by|\bnicknamed\b"),
    ("current_affairs", r"\brecently\b|\bin 202[3-6]\b|was (launched|appointed|"
                        r"inaugurated|signed) in|\bsummit\b|\bhosted the\b"),
]

COMPILED = [(topic, re.compile(pat, re.IGNORECASE)) for topic, pat in RULES]

# Ten of the 84 papers extract without inter-word spacing
# ("Inthefollowingquestion,selectthemissingnumber"). Every space-bearing pattern
# above silently fails on them -- ~12% of the corpus lost to a PDF quirk. The fix
# is to match a space-stripped question against space-stripped patterns. Only
# literal-space patterns can be de-spaced safely, so regexes containing
# whitespace classes (\s, \b) are skipped rather than mangled.
def _despace(pattern: str) -> str | None:
    r"""Build a de-spaced variant, or None if it cannot be made safely.

    Two hazards, both learned the hard way:

    1. \b is meaningless once spaces are gone, so it must be stripped -- but
       stripping it turns `\bratio\b` into bare `ratio`, which matches INSIDE
       "operation" (ope-ratio-n). That mislabelled an odd-one-out question as
       ratio_and_proportion.
    2. Short alternatives are the whole problem: long template phrases like
       "selectthemostappropriatesynonym" cannot collide by accident, 5-letter
       words routinely do.

    So drop any alternative under 12 characters and keep the rest. A rule with no
    surviving alternatives returns None and simply does not participate.

    Splitting on "|" also cuts through alternations nested in parentheses
    ("(speed|current|km)"), leaving unbalanced groups -- so each surviving
    alternative is compiled individually and dropped if it does not parse.
    """
    kept: list[str] = []
    for alt in pattern.split("|"):
        # Whitespace metacharacters must go too, not just literal spaces.
        # Patterns written with \s+ were skipped entirely, which left 17
        # topics -- percentage, geometry, syllogism, antonyms,
        # time_speed_distance among them -- with NO despaced variant, so any
        # spacing-degraded question in those topics was untaggable. 2022 tagged
        # at 44% against 65-75% everywhere else for exactly this reason.
        squashed = (alt.replace("\\b", "")
                       .replace("\\s+", "").replace("\\s*", "").replace("\\s", "")
                       .replace(" ", ""))
        # The 12-char floor exists to drop broken alternation fragments like
        # "no)\w+(are", which would match almost anything. But it also dropped
        # every legitimate short keyword -- triangle, circle, percentage,
        # average, antonym -- leaving 16 topics with no despaced variant at all.
        #
        # Separate the two cases: a PURE ALPHABETIC word of >=6 characters is a
        # real keyword and safe to match inside concatenated text; anything
        # carrying regex metacharacters still needs the full 12.
        #
        # >=6 not >=5 deliberately: "train" and "speed" would fire inside
        # "constraint" and "speedily", and INSIGHTS.md §4 is explicit that a
        # wrong label biases every downstream count while a missing one only
        # shrinks the sample.
        if not (squashed.isalpha() and len(squashed) >= 6) and len(squashed) < 12:
            continue
        try:
            re.compile(squashed)
        except re.error:
            continue  # fragment of a nested group — not usable on its own
        kept.append(squashed)
    return "|".join(kept) if kept else None


_DESPACED = []
for _topic, _pat in RULES:
    if " " in _pat or "\\s" in _pat:
        _built = _despace(_pat)
        if _built:
            _DESPACED.append((_topic, re.compile(_built, re.IGNORECASE)))
_SPACE_RE = re.compile(r"\s+")

# PDFs carry typographic punctuation: ‘+’ and – rather than '+' and -.
# A pattern written with straight quotes silently never matches those questions.
_PUNCT = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
                        "\u2013": "-", "\u2014": "-", "\u2212": "-"})


def classify(text: str) -> tuple[str | None, str | None]:
    """Return (topic, matched_pattern). First matching rule wins."""
    text = text.translate(_PUNCT)
    for topic, rx in COMPILED:
        if rx.search(text):
            return topic, rx.pattern[:48]

    # Fallback for spacing-degraded extractions only. Gated on the space ratio so
    # normal text never takes this path -- de-spaced matching is looser and would
    # create false positives on well-formed questions.
    # 20, not 40: "Q.3SelecttheINCORRECTLYspeltword." is 33 chars and was
    # excluded by the old gate, so despaced short questions never got a chance.
    if len(text) > 20 and text.count(" ") / len(text) < 0.10:
        squashed = _SPACE_RE.sub("", text)
        for topic, rx in _DESPACED:
            if rx.search(squashed):
                return topic, rx.pattern[:48] + " [despaced]"
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", default=str(REPO / "data/parsed/papers.json"))
    ap.add_argument("--out", default=str(REPO / "data/tagged/rules.json"))
    ap.add_argument("--taxonomy", default="ssc_cgl")
    ap.add_argument("--report", action="store_true", help="coverage report, no write")
    args = ap.parse_args()

    tax = Taxonomy.load(args.taxonomy)

    # A rule naming a topic outside the taxonomy would silently create a new
    # category and corrupt every year-over-year comparison. Fail at startup.
    bad = sorted({t for t, _ in RULES if not tax.validate(t)})
    if bad:
        raise SystemExit(f"rules reference topics not in the taxonomy: {bad}")

    papers = load_papers(Path(args.papers))
    matched = unmatched = 0
    by_topic: Counter = Counter()
    by_section: Counter = Counter()

    for p in papers:
        for q in p.questions:
            topic, _ = classify(q.text)
            if topic:
                q.topic = topic
                q.tagger_confidence = 1.0  # deterministic: it matched or it didn't
                matched += 1
                by_topic[topic] += 1
                by_section[tax.topic_to_section[topic]] += 1
            else:
                q.topic = None
                unmatched += 1

    total = matched + unmatched
    print(f"\n{total} questions, {len(RULES)} rules")
    print(f"matched   : {matched} ({matched/total:.1%})")
    print(f"unmatched : {unmatched} ({unmatched/total:.1%})  <- left untagged on purpose\n")

    print(f"{'section':<34}{'matched':>9}{'per paper':>11}")
    print("-" * 54)
    for s in tax.sections:
        n = by_section.get(s, 0)
        print(f"{s:<34}{n:>9}{n/len(papers):>11.1f}")

    print("\ntop 12 topics:")
    for t, n in by_topic.most_common(12):
        print(f"  {n:>5}  {t}")

    never = [t for t in tax.topics if by_topic.get(t, 0) == 0]
    if never:
        print(f"\nnever matched ({len(never)}): {', '.join(never[:12])}"
              f"{' ...' if len(never) > 12 else ''}")
        print("  -> these need either a new rule or an LLM pass")

    if not args.report:
        save_papers(papers, Path(args.out))
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
