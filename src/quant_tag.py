"""Quant hierarchical tagger: topic -> subtopic -> microtopic, plus DI linking.

Three things this does that the earlier section taggers did not need:

1. OPTION-SIGNATURE RECOVERY. 12% of Quant stems are lost to embedded images
   ("Simplify:  Ans 1. 9  2. 23  3. 27  4. 15" is the entire extracted text),
   and the loss is NOT random -- it takes the symbolic microtopics
   (simplification, surds, identities, trigonometric expressions) while sparing
   prose word problems. Those questions still carry their OPTIONS, whose
   structure implies a topic (currency -> money maths, ° -> geometry/trig, bare
   integers -> simplification/algebra). Recovered labels are tagged
   `label_source="recovered"` and reported separately, never silently mixed with
   labels read from a stem.

2. DI PROXIMITY LINKING. A Quant DI table is printed ONCE, with its first
   question; the following questions merely refer back to it. That is the mirror
   of the English cloze passage, which is reprinted with every question and had
   to be de-duplicated. Linking only the marked question gave 40 DI sets and 40
   linked questions -- an impossible 1:1 -- so back-references are resolved to
   the most recent table in the same shift.

3. FOREIGN-SECTION VETO for Reasoning bleed. Quant sits in the 3rd numbering
   block, between Reasoning-adjacent numbering and English, and number-series /
   odd-one-out items leak in when a numbering reset is invisible.

2021 is excluded (38% blank stems, 3 shifts). 2025 is retained in the file but
flagged: it is coaching-generated, contains zero DI sets, and is excluded from
the evidence base downstream.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from reasoning_tag import NOISE_ONLY, normalize

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
DROP_YEARS = {2021}

# ------------------------------------------------------------------- taxonomy
# (topic, subtopic, pattern). Order matters: DI must precede every arithmetic
# rule, because a DI question about averages is DI, not averages.
RULES: list[tuple[str, str, str]] = [
    ("data_interpretation", "table",
     r"study the given table|the table (given below|below) shows|"
     r"the table (shows|depicts|represents)|study the following table"),
    ("data_interpretation", "bar_graph",
     r"bar[\s\-]?graph|the bar (chart|diagram)"),
    ("data_interpretation", "pie_chart",
     r"pie[\s\-]?chart"),
    ("data_interpretation", "line_graph",
     r"line[\s\-]?graph"),

    # --- interest (must precede percentage: SI/CI stems are percentage-shaped)
    ("simple_compound_interest", "compound_interest",
     r"compound interest|compounded (annually|half[\s\-]?yearly|quarterly)"),
    ("simple_compound_interest", "simple_interest",
     r"simple interest|\bper annum\b.{0,40}interest|interest.{0,30}per annum"),

    # --- profit/loss/discount (precede percentage for the same reason)
    ("profit_loss_discount", "successive_discount",
     r"two successive discounts?|successive discounts? of|"
     r"discount schemes?|single discount equivalent"),
    ("profit_loss_discount", "false_weight",
     r"dishonest (merchant|shopkeeper|dealer)|false weight|uses a weight of"),
    ("profit_loss_discount", "marked_price_discount",
     r"marked price|list price|after (allowing|giving) a discount"),
    ("profit_loss_discount", "cost_selling_price",
     r"cost price|selling price|\bsold (for|at)\b|"
     r"at a (profit|loss) of|profit percent|loss percent|\bgains?\b.{0,20}%"),

    # --- percentage
    ("percentage", "successive_change",
     r"(increased|decreased|reduced) by \d+(\.\d+)?%.{0,60}"
     r"(increased|decreased|reduced) by|"
     r"first (increased|decreased).{0,40}then"),
    ("percentage", "reverse_percentage",
     r"what was the (original|initial)|the number whose \d+%"),
    ("percentage", "direct_percentage",
     r"\d+(\.\d+)?\s*%\s*(of|more|less|increase|decrease)|"
     r"what percent(age)? (of|is)|expressed as a percentage"),

    # --- ratio / proportion / mixture / partnership
    ("ratio_proportion", "mean_proportional",
     r"mean proportional|mean proportion|third proportional|fourth proportional"),
    ("ratio_proportion", "partnership",
     r"invests?.{0,50}partnership|profit.{0,30}shared in the ratio|"
     r"\bpartners?\b.{0,40}(invest|capital)"),
    ("ratio_proportion", "mixture_alligation",
     r"mixture|alligation|\bmixed with\b|milk and water|\balloy\b"),
    ("ratio_proportion", "ratio_basic",
     r"the ratio of|in the ratio|ratio between|divided in the ratio"),

    # --- averages
    ("average", "average_series",
     r"average of the (squares|cubes)|average of the first \d+"),
    ("average", "average_replacement",
     r"average.{0,60}(increases|decreases|rises|falls) by|"
     r"when (one|a) (person|number).{0,30}(replaced|excluded|included)"),
    ("average", "average_basic",
     r"\baverage\b|\bmean\b(?! proportional)"),

    # --- time & work
    ("time_and_work", "men_days_hours",
     r"\d+ (persons?|men|women|workers?).{0,60}(hours? a day|days?)|"
     r"men and \d+ women can do"),
    ("time_and_work", "pipes_cisterns",
     r"\bpipes?\b|\bcistern\b|\btanks?\b.{0,30}(fill|empty)|inlet|outlet"),
    ("time_and_work", "efficiency_alternate",
     r"alternate days|work(s|ing)? on alternate|"
     r"efficiency of|\bmore efficient\b|leaves? the (work|job)"),
    ("time_and_work", "work_basic",
     r"can (do|complete|finish) (a|the) (piece of )?work|"
     r"working together.{0,30}(complete|finish)"),

    # --- speed / distance
    ("time_speed_distance", "trains",
     r"\btrains?\b.{0,60}(speed|cross|platform|pole|tunnel|each other)"),
    ("time_speed_distance", "boats_streams",
     r"\bboat\b|\bstream\b|upstream|downstream|still water"),
    ("time_speed_distance", "races",
     r"\brace\b.{0,50}beats?|gives? .{0,20}a start of"),
    ("time_speed_distance", "average_speed",
     r"average speed|two[\s\-]?thirds? of a certain distance"),
    ("time_speed_distance", "relative_motion",
     r"\bthief\b.{0,60}policeman|overtakes?|in the same direction|"
     r"opposite directions"),
    ("time_speed_distance", "speed_basic",
     r"\bkm/?h\b|km per hour|\bm/s\b|runs? \d+ ?km|"
     r"speed of \d+|distance of \d+ ?(km|m)\b"),

    # --- number system
    ("number_system", "divisibility",
     r"divisible by|is a factor of \d+|divisibility"),
    ("number_system", "hcf_lcm",
     r"\bhcf\b|\blcm\b|highest common factor|least common multiple"),
    ("number_system", "unit_digit_cyclicity",
     r"unit(s)? digit|last digit|digit at the (unit|ten)"),
    ("number_system", "digit_problem",
     r"\d+[\s\-]?digit number|the digits? of (a|the) number|"
     r"\bpq\b is divisible"),
    ("number_system", "remainder",
     r"\bremainder\b|when divided by \d+.{0,30}leaves"),
    ("number_system", "surds_indices",
     r"\bsurds?\b|\bindices\b|square root of|cube root of|"
     r"\b\d+\^|raised to the power"),
    ("number_system", "simplification",
     r"\bsimplif(y|ication)\b|the value of.{0,40}\bis\b|is equal to"),

    # --- algebra
    ("algebra", "identities",
     r"\ba\s*[+\-]\s*b\b.{0,40}\bb\s*[+\-]\s*c\b|"
     r"a\^?[23]\s*[+\-]\s*b\^?[23]|x\s*\+\s*1/x|x\^?2\s*\+\s*y\^?2"),
    ("algebra", "factorisation",
     r"\bfactor(is|iz)ation\b|\bis a factor of\b|\bfactoris"),
    ("algebra", "linear_equations",
     r"value of (k|p|q|m|n|x|y)\b|solve for|\bequations?\b.{0,30}value"),
    ("algebra", "algebra_general",
     r"\bpolynomial\b|\bquadratic\b|\bexpression\b"),

    # --- geometry
    ("geometry", "circles_tangent",
     r"circles? of radi(i|us).{0,40}touch|touch each other (externally|internally)|"
     r"\btangent\b|direct common tangent"),
    ("geometry", "chords",
     r"\bchord\b"),
    ("geometry", "triangle_properties",
     r"right[\s\-]?angled triangle|\btriangle\b.{0,50}(area|sides?|angle)|"
     r"\bincentre\b|\bcircumcentre\b|\bcentroid\b|\borthocentre\b"),
    ("geometry", "similarity",
     r"\bsimilar\b.{0,30}triangles?|\bcongruent\b"),
    ("geometry", "quadrilateral_polygon",
     r"\bparallelogram\b|\brhombus\b|\btrapezium\b|\bpolygon\b|"
     r"cyclic quadrilateral|regular (hexagon|pentagon|octagon)"),
    ("geometry", "angles",
     r"\bangle\b.{0,40}(centre|circumference|degrees?)|°"),

    # --- mensuration
    ("mensuration", "sphere_hemisphere",
     r"\bsphere\b|\bhemisphere\b"),
    ("mensuration", "cylinder_cone",
     r"\bcylinder\b|\bcone\b|\bfrustum\b"),
    ("mensuration", "cube_cuboid",
     r"\bcube\b|\bcuboid\b|edge of a cube"),
    ("mensuration", "circle_area",
     r"circumference|area of (a|the) circle|\bsector\b"),
    ("mensuration", "rectangle_square",
     r"length and breadth|\brectangle\b|\bsquare\b.{0,30}(area|perimeter|side)"),
    ("mensuration", "mensuration_general",
     r"total surface area|curved surface area|\bvolume\b|"
     r"\barea\b.{0,20}(cm|m)\^?2"),

    # --- trigonometry
    ("trigonometry", "heights_distances",
     r"angle of (elevation|depression)|height of the (tower|pole|building)"),
    ("trigonometry", "identities_values",
     r"trigonometric|θ|\b(sin|cos|tan|cot|sec|cosec)\s*\d|"
     r"\b(sin|cos|tan|cot|sec|cosec)\s*[θΘ]|"
     r"\b(sin|cos|tan|cot|sec|cosec)\b\s*\("),

    # --- statistics
    ("statistics", "central_tendency",
     r"\bmedian\b|\bmode\b|standard deviation|\bvariance\b"),
]

COMPILED = [(t, s, re.compile(p, re.I)) for t, s, p in RULES]


def _despace(pattern: str) -> str | None:
    s = pattern
    # " ?" must be consumed as a UNIT before plain spaces: stripping the space
    # alone turns "\d+ ?km" into "\d+?km", silently converting a greedy
    # quantifier plus optional space into a LAZY quantifier and changing what the
    # rule means.
    # Do NOT strip commas: "{0,50}" becomes "{050}", which still COMPILES but
    # means "exactly 50 characters" instead of "0 to 50". Every rule containing a
    # bounded wildcard was therefore silently near-dead in this tier -- no error,
    # just a pattern that almost never matches.
    for tok in (r"[\s\-]?", r"[\s\-]", r"\s", r"\b", " ?", " "):
        s = s.replace(tok, "")
    try:
        re.compile(s)
    except re.error:
        return None
    return s


DESPACED = []
for _t, _s, _p in RULES:
    _b = _despace(_p)
    if _b:
        DESPACED.append((_t, _s, re.compile(_b, re.I)))

_SQUASH = re.compile(r"[\s,\-.'’]+")
MIN_SQUASHED = 10

# Some technical tokens are shorter than the global floor but cannot collide
# inside ordinary words once spaces are gone -- verified against squashed English
# prose: hcf/lcm/cot/cosec are safe, while sin/cos/tan/sec DO collide
# ("using", "cost", "important", "second"), so those are matched only through a
# distinctive context (a theta, a digit, or the word "trigonometric") instead.
SHORT_SAFE = re.compile(r"^(hcf|lcm|cot|cosec)$", re.I)


def _squashed_ok(match_text: str) -> bool:
    return len(match_text) >= MIN_SQUASHED or bool(SHORT_SAFE.match(match_text))

# ------------------------------------------------------- foreign-section veto
FOREIGN = {
    "reasoning": re.compile(
        r"in a certain code|odd one out|letter[\s\-]?cluster|"
        r"related in the same way|replace the question mark|"
        r"mirror image|water image|paper folded|\bsyllogis|"
        r"select the set in which the numbers|"
        r"\binterchang(e|es|ed|ing)\b.{0,40}(equation|signs|numbers)", re.I),
    "english": re.compile(
        r"\bsynonym\b|\bantonym\b|\bidiom\b|correctly spelt|"
        r"one[\s\-]?word substitut|underlined segment", re.I),
    "general_awareness": re.compile(
        r"chief minister|lok sabha|dynasty|constitution|"
        r"capital of|\byojana\b|padma|olympic", re.I),
}
QUANT_ANCHOR = re.compile(
    r"\d|percent|ratio|average|interest|profit|loss|discount|speed|"
    r"area|volume|angle|triangle|circle|work|days?|hours?|"
    r"\bhcf\b|\blcm\b|table|graph|chart", re.I)

FOREIGN_SQ = {k: re.compile(v.pattern.replace(r"\b", "").replace(" ", ""), re.I)
              for k, v in FOREIGN.items()}

# Back-references that resolve to a DI table printed with an earlier question.
DI_BACKREF = re.compile(
    r"the (given |above |following )?(table|bar[\s\-]?graph|pie[\s\-]?chart|"
    r"line[\s\-]?graph|chart|graph|data)\b", re.I)


def foreign_section(stem: str) -> str | None:
    text = normalize(stem)
    sq = _SQUASH.sub("", text)
    for sec, rx in FOREIGN.items():
        if rx.search(text) or FOREIGN_SQ[sec].search(sq):
            # A Reasoning-style stem that also carries heavy Quant vocabulary is
            # ambiguous; only veto when the Quant anchor is weak.
            if not QUANT_ANCHOR.search(text):
                return sec
            if sec == "reasoning":
                return sec
    return None


def classify(stem: str) -> tuple[str | None, str | None, str]:
    text = normalize(stem)
    for t, s, rx in COMPILED:
        if rx.search(text):
            return t, s, "direct"
    sq = _SQUASH.sub("", text)
    for t, s, rx in DESPACED:
        m = rx.search(sq)
        if m and _squashed_ok(m.group(0)):
            return t, s, "squashed"
    return None, None, "none"


# ------------------------------------------- option-signature recovery (blanks)
CUR = re.compile(r"₹|\bRs\.?\b", re.I)
PCT = re.compile(r"%")
DEG = re.compile(r"°|\bdegrees?\b", re.I)
LEN_UNIT = re.compile(r"\b(cm|mm|m|km)\b|cm\^?2|m\^?2|sq\.?\s?(cm|m)", re.I)
TIME_UNIT = re.compile(r"\b(days?|hours?|minutes?|min|sec|seconds?)\b", re.I)
RATIO = re.compile(r"\d+\s*:\s*\d+")
BARE_NUM = re.compile(r"^-?\d+(?:\.\d+)?$")


def recover_from_options(options) -> tuple[str | None, str | None]:
    """Coarse topic from option structure alone, for image-lost stems."""
    if not options or len(options) != 4:
        return None, None
    texts = [t.strip() for _, t in options]
    joined = " ".join(texts)
    if RATIO.search(joined):
        return "ratio_proportion", "ratio_basic"
    if DEG.search(joined):
        return "geometry", "angles"
    if CUR.search(joined):
        return "profit_loss_discount", "cost_selling_price"
    if PCT.search(joined):
        return "percentage", "direct_percentage"
    if LEN_UNIT.search(joined):
        return "mensuration", "mensuration_general"
    if TIME_UNIT.search(joined):
        return "time_and_work", "work_basic"
    if all(BARE_NUM.match(t) for t in texts):
        return "number_system", "simplification"
    return None, None


def main() -> None:
    recs = [r for r in json.load(open(OUT / "quant_questions.json"))
            if r["year"] not in DROP_YEARS]

    # ---- DI proximity linking (per shift, in printed order) ----------------
    by_shift = defaultdict(list)
    for r in recs:
        by_shift[(r["exam_date"], r["shift"])].append(r)
    linked_extra = 0
    for _, rows in by_shift.items():
        rows.sort(key=lambda r: r["printed_number"])
        current = None
        for r in rows:
            if r.get("di_id"):
                current = r["di_id"]
                continue
            if current and DI_BACKREF.search(normalize(r["stem"])):
                r["di_id"] = current
                linked_extra += 1

    kept, blank, vetoed, recovered = [], [], [], []
    for r in recs:
        stem = normalize(r["stem"])
        if len(stem) < 25 or NOISE_ONLY.match(stem):
            t, s = recover_from_options(r["options"])
            r["topic"], r["subtopic"] = t, s
            r["tag_route"] = "option_recovery" if t else "blank"
            r["label_source"] = "recovered" if t else None
            (recovered if t else blank).append(r)
            continue
        fs = foreign_section(r["stem"])
        if fs:
            r["section_conflict"] = fs
            vetoed.append(r)
            continue
        t, s, how = classify(r["stem"])
        r["topic"], r["subtopic"], r["tag_route"] = t, s, how
        r["label_source"] = "stem" if t else None
        # DI overrides arithmetic: a DI-linked question is a DI question.
        if r.get("di_id") and t != "data_interpretation":
            r["topic"] = "data_interpretation"
            r["subtopic"] = r["subtopic"] or "table"
            r["label_source"] = "stem"
        kept.append(r)

    usable = kept + blank + recovered
    shifts = len({(r["exam_date"], r["shift"]) for r in usable})
    from_stem = [r for r in kept if r["topic"]]

    print("=" * 96)
    print("QUANT TAGGER — coverage (2021 excluded)")
    print("=" * 96)
    print(f"questions                : {len(usable)}  across {shifts} shifts")
    print(f"vetoed (section bleed)   : {len(vetoed)}  "
          f"{dict(Counter(r['section_conflict'] for r in vetoed))}")
    print(f"tagged from stem         : {len(from_stem)}")
    print(f"recovered from options   : {len(recovered)}")
    print(f"unlabelled (image loss)  : {len(blank)}")
    print(f"coverage, stem-only      : {100*len(from_stem)/max(len(kept),1):.1f}%")
    print(f"coverage, incl. recovery : "
          f"{100*(len(from_stem)+len(recovered))/max(len(usable),1):.1f}%")
    print(f"DI questions linked by back-reference: +{linked_extra}")
    print(f"DI-labelled questions total          : "
          f"{sum(1 for r in usable if r.get('di_id'))}")
    print(f"route: {dict(Counter(r['tag_route'] for r in kept))}")

    print("\n" + "=" * 96)
    print("TOPIC x SUBTOPIC (per shift, stem-labelled only)")
    print("=" * 96)
    by = defaultdict(Counter)
    for r in from_stem:
        by[r["topic"]][r["subtopic"]] += 1
    for t in sorted(by, key=lambda k: -sum(by[k].values())):
        tot = sum(by[t].values())
        print(f"\n{t:<28}{tot:>5}{tot/shifts:>8.2f}/shift")
        for s, c in by[t].most_common():
            print(f"    {s:<26}{c:>5}{c/shifts:>8.2f}")

    print("\n" + "=" * 96)
    print("PER-YEAR")
    print("=" * 96)
    print(f"{'Year':<6}{'Shifts':>7}{'Q':>6}{'stem':>7}{'recov':>7}"
          f"{'lost':>6}{'cov':>7}")
    print("-" * 96)
    for y in sorted({r["year"] for r in usable}):
        u = [r for r in usable if r["year"] == y]
        k = [r for r in kept if r["year"] == y]
        st = [r for r in k if r["topic"]]
        rc = [r for r in recovered if r["year"] == y]
        bl = [r for r in blank if r["year"] == y]
        sh = len({(r["exam_date"], r["shift"]) for r in u})
        print(f"{y:<6}{sh:>7}{len(u):>6}{len(st):>7}{len(rc):>7}{len(bl):>6}"
              f"{100*(len(st)+len(rc))/len(u):>6.0f}%")

    untag = [r for r in kept if not r["topic"]]
    print(f"\nStill untagged (non-blank stems): {len(untag)}")
    for r in untag[:10]:
        print("  -", normalize(r["stem"])[:100])

    (OUT / "quant_tagged.json").write_text(json.dumps(usable, indent=2))
    print(f"\nWrote {OUT/'quant_tagged.json'}")


if __name__ == "__main__":
    main()
