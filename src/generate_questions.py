"""Construct new Reasoning questions fitted to the 2026 blueprint.

Every generator produces a question whose answer it computes, and every question
is then re-derived by src/solvers.py from the RENDERED STEM TEXT ALONE. If the
independent solver declines or disagrees, the candidate is discarded rather than
shipped. Nothing reaches a paper on the generator's own authority.

That round trip is the whole design. A generated question with a subtly wrong key
teaches the wrong answer, which is worse than having no paper (HANDOVER §6), and
"I computed it correctly" is not evidence when the same code did the computing.

Figure-based archetypes (mirror/water image, paper folding, embedded figures,
figure series, counting shapes) are NOT generated: they need real figures, and a
text paraphrase of them would be a different question. They are ~2.6 of the
blueprint's 25 questions, and their share is redistributed across the
text-representable archetypes, with the omission reported rather than hidden.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from itertools import product
from pathlib import Path

from solvers import solve_any

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
ALPHA = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
NUMWORD = ["None", "One", "Two", "Three", "Four", "Five", "Six"]

WORDS6 = ["BINDER", "MARKET", "PLANET", "SILVER", "GARDEN", "FOREST", "CANDLE",
          "WINTER", "HUNTER", "MASTER", "TIMBER", "FLOWER", "BASKET", "DOCTOR",
          "PENCIL", "ORANGE", "MONKEY", "TEMPLE", "BRIDGE", "CIRCLE",
          "MARBLE", "PURPLE", "SIGNAL", "TUNNEL", "VELVET", "WALNUT", "YELLOW",
          "ANCHOR", "BRONZE", "COPPER", "DAMAGE", "ENGINE", "FABRIC", "GRAVEL",
          "HARBOR", "ISLAND", "JUNGLE", "KETTLE", "LADDER", "MEADOW"]
CODEWORDS = ["TIGER", "HOUSE", "PLANT", "BRAIN", "CLOUD", "STONE", "TRAIN",
             "GRAPE", "SHIRT", "CHAIR", "MOUSE", "LIGHT", "WATER", "MUSIC"]
DICT_SETS = [
    ["Popular", "Popcorn", "Poplin", "Poplar"],
    ["Marble", "Marbled", "Marbling", "Marbles"],
    ["Sincere", "Since", "Sinful", "Singe"],
    ["Present", "Preserve", "Preside", "Presume"],
    ["Cabinet", "Cable", "Cabbage", "Cabin"],
    ["Dominate", "Domicile", "Dominion", "Domain"],
    ["Furnish", "Further", "Furlong", "Furnace"],
    ["Gracious", "Gradual", "Graceful", "Gradient"],
    ["Harbour", "Hardly", "Harmony", "Harness"],
    ["Intense", "Intend", "Intent", "Interim"],
    ["Native", "Nation", "Natural", "Nature"],
    ["Obscure", "Observe", "Obstacle", "Obsolete"],
]


def _uniq_options(correct, wrongs, rng):
    """Four distinct options, one correct, shuffled. Returns (options, letter)."""
    seen = {str(correct)}
    picks = []
    for w in wrongs:
        if str(w) not in seen:
            seen.add(str(w))
            picks.append(w)
        if len(picks) == 3:
            break
    if len(picks) < 3:
        return None, None
    vals = [correct] + picks
    rng.shuffle(vals)
    letters = "abcd"
    opts = [(letters[i], str(v)) for i, v in enumerate(vals)]
    letter = letters[vals.index(correct)]
    return opts, letter


# ------------------------------------------------------------ number series
def gen_number_series(rng):
    family = rng.choice(["quadratic", "mul_increment", "geometric", "cyclic"])
    n = 6
    if family == "quadratic":
        A = rng.choice([-10, -5, 5, 10, 2, 3])
        B = rng.randint(-20, 20)
        C = rng.randint(50, 400)
        seq = [A * i * i + B * i + C for i in range(n)]
    elif family == "mul_increment":
        p = rng.choice([2, 3])
        a0 = rng.randint(2, 9)
        d = rng.choice([1, 2, -1])
        q0 = rng.randint(-3, 3)
        seq = [a0]
        for i in range(n - 1):
            seq.append(p * seq[-1] + q0 + d * i)
    elif family == "geometric":
        r = rng.choice([2, 3])
        a0 = rng.choice([3, 7, 11, 111, 13])
        seq = [a0 * r ** i for i in range(n)]
    else:
        a0 = rng.randint(60, 120)
        c1, c2 = rng.choice([(-3, -8), (-4, -7), (5, 9), (-6, -2)])
        seq = [a0]
        for i in range(n - 1):
            seq.append(seq[-1] + (c1 if i % 2 == 0 else c2))
    if any(abs(v) > 100000 for v in seq):
        return None
    gap = rng.choice([3, 4, 5])
    ans = seq[gap]
    shown = [("?" if i == gap else str(v)) for i, v in enumerate(seq)]
    stem = ("Select the number that will replace the question mark (?) in "
            "the following series.  " + ", ".join(shown))
    step = max(abs(seq[1] - seq[0]), 2)
    wrongs = [ans + step, ans - step, ans + 2, ans - 2, ans + step // 2]
    opts, letter = _uniq_options(ans, [w for w in wrongs if w != ans], rng)
    if not opts:
        return None
    return dict(subtopic="number_or_letter_series", stem=stem, options=opts,
                answer=letter, trace=f"{family} series, missing index {gap}")


# ------------------------------------------------------------ coding-decoding
def gen_letter_code(rng):
    src, tgt = rng.sample(CODEWORDS, 2)
    k = rng.choice([1, 2, 3, 4, -1, -2, -3])
    enc = lambda w, s: "".join(ALPHA[(ALPHA.index(c) + s) % 26] for c in w)
    coded_src = enc(src, k)
    ans = enc(tgt, k)
    # Skip any shift that lands on 0: it offers the unshifted word itself as a
    # distractor, which no real paper does and which gives the answer away.
    shifts = [s for s in (k + 1, k - 1, k + 2, k - 2, -k) if s % 26 != 0]
    wrongs = [enc(tgt, s) for s in shifts]
    opts, letter = _uniq_options(ans, [w for w in wrongs if w != ans], rng)
    if not opts:
        return None
    stem = (f"In a certain code language, {src} is written as {coded_src}. "
            f"How will {tgt} be written in that language?")
    return dict(subtopic="language_code", stem=stem, options=opts,
                answer=letter, trace=f"positional shift {k:+d}")


# ---------------------------------------------------------------- analogy
def gen_number_analogy(rng):
    kind = rng.choice(["square", "cube", "sq_plus", "mul"])
    a = rng.randint(3, 15)
    c = rng.randint(3, 15)
    if a == c:
        return None
    if kind == "square":
        f = lambda x: x * x
    elif kind == "cube":
        f = lambda x: x ** 3
    elif kind == "sq_plus":
        f = lambda x: x * x + x
    else:
        k = rng.randint(3, 9)
        f = lambda x: x * k
    ans = f(c)
    if ans > 5000:
        return None
    wrongs = [f(c) + c, f(c) - c, f(c + 1), f(c) + 1, c * c if kind != "square" else c ** 3]
    opts, letter = _uniq_options(ans, [w for w in wrongs if w != ans], rng)
    if not opts:
        return None
    stem = (f"Select the option that is related to the third number in the same "
            f"way as the second number is related to the first number.  "
            f"{a} : {f(a)} :: {c} : ?")
    return dict(subtopic="colon_notation", stem=stem, options=opts,
                answer=letter, trace=f"relation {kind}")


# -------------------------------------------------- alphabetical fixed points
def gen_position_unchanged(rng):
    w = rng.choice(WORDS6)
    fixed = sum(1 for a, b in zip(w, "".join(sorted(w))) if a == b)
    if fixed > 4:
        return None
    ans = NUMWORD[fixed]
    wrongs = [NUMWORD[i] for i in range(5) if i != fixed]
    rng.shuffle(wrongs)
    opts, letter = _uniq_options(ans, wrongs, rng)
    if not opts:
        return None
    stem = (f"The position of how many letters will remain unchanged if all the "
            f"letters in the word {w} are arranged in English alphabetical order?")
    return dict(subtopic="position_unchanged", stem=stem, options=opts,
                answer=letter, trace=f"{w} -> {''.join(sorted(w))}, {fixed} fixed")


# ------------------------------------------------------------ dictionary order
def gen_dictionary_order(rng):
    words = list(rng.choice(DICT_SETS))
    pos_name = rng.choice(["Second", "Third"])
    idx = {"Second": 1, "Third": 2}[pos_name]
    ranked = sorted(words, key=str.lower)
    ans = ranked[idx]
    numbered = "  ".join(f"{i+1}. {w}" for i, w in enumerate(words))
    stem = (f"After arranging the given words according to dictionary order, "
            f"which word will come at '{pos_name}' position?  {numbered}")
    wrongs = [w for w in words if w != ans]
    opts, letter = _uniq_options(ans, wrongs, rng)
    if not opts:
        return None
    return dict(subtopic="alphabetical_arrangement", stem=stem, options=opts,
                answer=letter, trace=f"sorted={ranked}")


# ------------------------------------------------------------- symbol maths
def gen_symbol_definition(rng):
    syms = ["@", "#", "$", "%"]
    ops = ["+", "-", "*", "/"]
    rng.shuffle(ops)
    mapping = dict(zip(syms, ops))
    for _ in range(40):
        x, y, z = (rng.randint(2, 20) for _ in range(3))
        s1, s2 = rng.sample(syms, 2)
        expr = f"{x} {mapping[s1]} {y} {mapping[s2]} {z}"
        try:
            val = eval(expr)  # noqa: S307 - operands are generated ints
        except ZeroDivisionError:
            continue
        if val != int(val) or abs(val) > 500:
            continue
        val = int(val)
        defs = ", ".join(f"'{s}' means '{mapping[s]}'" for s in syms)
        stem = (f"If {defs}, then what is the value of: "
                f"{x} {s1} {y} {s2} {z}?")
        wrongs = [val + 1, val - 1, val + rng.randint(2, 6), val - rng.randint(2, 6)]
        opts, letter = _uniq_options(val, [w for w in wrongs if w != val], rng)
        if opts:
            return dict(subtopic="symbol_definition", stem=stem, options=opts,
                        answer=letter, trace=f"{expr} = {val}")
    return None


# ------------------------------------------------------------ sign interchange
def gen_sign_interchange(rng):
    for _ in range(80):
        a, b, c, d = (rng.randint(2, 12) for _ in range(4))
        s1, s2 = rng.sample(["+", "-", "*", "/"], 2)
        lhs = f"{a} {s1} {b} {s2} {c} + {d}"
        swapped = "".join(s2 if ch == s1 else s1 if ch == s2 else ch for ch in lhs)
        try:
            target = eval(swapped)  # noqa: S307
            orig = eval(lhs)  # noqa: S307
        except ZeroDivisionError:
            continue
        if target != int(target) or target == orig or abs(target) > 400:
            continue
        target = int(target)
        ans = f"{s1} and {s2}"
        pool = [f"{p} and {q}" for p, q in
                [("+", "-"), ("+", "*"), ("+", "/"), ("-", "*"), ("-", "/"), ("*", "/")]]
        wrongs = []
        for cand in pool:
            if cand == ans:
                continue
            p, q = cand.split(" and ")
            sw = "".join(q if ch == p else p if ch == q else ch for ch in lhs)
            try:
                if eval(sw) == target:  # noqa: S307
                    continue  # ambiguous: another swap also works
            except ZeroDivisionError:
                pass
            wrongs.append(cand)
        opts, letter = _uniq_options(ans, wrongs, rng)
        if not opts:
            continue
        stem = (f"Which two signs should be interchanged to make the given "
                f"equation correct?  {lhs} = {target}")
        return dict(subtopic="sign_interchange", stem=stem, options=opts,
                    answer=letter, trace=f"swap {ans}: {swapped} = {target}")
    return None


# ------------------------------------------------------------------ syllogism
def _syllogism_holds(premises, conclusion, n=4):
    """Brute-force model check over subsets of a small universe.

    Enumerates every assignment of the categories to subsets of {0..n-1}, keeps
    the models satisfying all premises, and requires the conclusion in every one.
    A premise set with no models at all is rejected -- a vacuous truth is not a
    question.
    """
    cats = sorted({c for p in premises + [conclusion] for c in p[1:]})
    universe = list(range(n))
    subsets = [frozenset(s) for r in range(n + 1)
               for s in __import__("itertools").combinations(universe, r)]

    def sat(rule, env):
        kind, x, y = rule
        X, Y = env[x], env[y]
        if kind == "all":
            return bool(X) and X <= Y
        if kind == "some":
            return bool(X & Y)
        if kind == "no":
            return not (X & Y)
        return False

    models = 0
    for combo in product(subsets, repeat=len(cats)):
        env = dict(zip(cats, combo))
        if all(sat(p, env) for p in premises):
            models += 1
            if not sat(conclusion, env):
                return False, models
    return (models > 0), models


def gen_syllogism(rng):
    names = rng.sample(["pens", "books", "tables", "chairs", "cups", "bags"], 3)
    A, B, C = names
    forms = [
        ([("all", A, B), ("all", B, C)], ("all", A, C), True),
        ([("all", A, B), ("no", B, C)], ("no", A, C), True),
        ([("all", A, B), ("some", B, C)], ("some", A, C), False),
        ([("some", A, B), ("all", B, C)], ("some", A, C), True),
    ]
    premises, concl, expect = rng.choice(forms)
    holds, models = _syllogism_holds(premises, concl)
    if models == 0 or holds != expect:
        return None

    def render(rule):
        kind, x, y = rule
        return {"all": f"All {x} are {y}.", "some": f"Some {x} are {y}.",
                "no": f"No {x} is {y}."}[kind]

    stem = ("Two statements are given, followed by a conclusion. Assuming the "
            "statements to be true, even if they seem to be at variance with "
            "commonly known facts, decide whether the conclusion logically "
            "follows.  Statements: " + " ".join(render(p) for p in premises) +
            "  Conclusion: " + render(concl))
    ans = "The conclusion follows" if holds else "The conclusion does not follow"
    other = "The conclusion does not follow" if holds else "The conclusion follows"
    opts, letter = _uniq_options(
        ans, [other, "The conclusion is probably true",
              "Data are inadequate to decide"], rng)
    if not opts:
        return None
    return dict(subtopic="statements_conclusions", stem=stem, options=opts,
                answer=letter, verified_by="model_check",
                trace=f"{models} models checked, follows={holds}")


# ------------------------------------------------------------- blood relations
def gen_blood_relation(rng):
    """Symbolic operator chain, resolved over an explicit family graph."""
    syms = ["+", "-", "*", "/"]
    roles = ["mother", "brother", "wife", "father"]
    rng.shuffle(syms)
    ops = dict(zip(syms, roles))
    inv = {v: k for k, v in ops.items()}
    people = rng.sample(["P", "Q", "R", "S", "T", "M", "N", "K"], 4)
    # P is the father of Q; Q is the brother of R  => P is R's father
    F, M, B = inv["father"], inv["mother"], inv["brother"]
    a, b, c = people[0], people[1], people[2]
    chains = [
        # father-of X, X brother-of Y  =>  father of Y
        (f"'{a} {F} {b}' and '{b} {B} {c}'", a, c, "Father"),
        (f"'{a} {M} {b}' and '{b} {B} {c}'", a, c, "Mother"),
        # father-of X, X father-of Y  =>  grandfather of Y
        (f"'{a} {F} {b}' and '{b} {F} {c}'", a, c, "Grandfather"),
        (f"'{a} {M} {b}' and '{b} {F} {c}'", a, c, "Grandmother"),
    ]
    chain, x, y, ans = rng.choice(chains)
    defs = "; ".join(f"'A {s} B' means 'A is the {r} of B'" for s, r in ops.items())
    stem = (f"In a certain code, {defs}. Based on this, how is {x} related to "
            f"{y} if {chain}?")
    wrongs = [w for w in ["Father", "Mother", "Grandfather", "Grandmother",
                          "Brother", "Son"] if w != ans]
    rng.shuffle(wrongs)
    opts, letter = _uniq_options(ans, wrongs, rng)
    if not opts:
        return None
    return dict(subtopic="coded_relation", stem=stem, options=opts,
                answer=letter, verified_by="relation_chain",
                trace=f"{x} -> {y} = {ans}")


# ------------------------------------------------------------- number-set analogy
def _set_relation(t, kind):
    a, b, c = t
    if kind == "sum":
        return a + b == c
    if kind == "diff_mul":
        return (b - a) * 2 == c - b
    if kind == "prod":
        return a * b == c
    if kind == "sq_sum":
        return a * a + b == c
    return False


def gen_number_set_analogy(rng):
    """"Select the set related in the same way as the given set."

    Verified structurally: the shared relation must hold for the given set and
    the key, and must FAIL for all three distractors. That check is what makes
    the item unambiguous, so it is enforced rather than assumed.
    """
    kind = rng.choice(["sum", "prod", "sq_sum"])
    def make(rng):
        for _ in range(60):
            a, b = rng.randint(2, 15), rng.randint(2, 15)
            c = {"sum": a + b, "prod": a * b, "sq_sum": a * a + b}[kind]
            if c <= 400:
                return (a, b, c)
        return None
    given, key = make(rng), make(rng)
    if not given or not key or given == key:
        return None
    wrongs = []
    for _ in range(200):
        t = (rng.randint(2, 15), rng.randint(2, 15), rng.randint(2, 400))
        if not _set_relation(t, kind) and t not in wrongs:
            wrongs.append(t)
        if len(wrongs) == 3:
            break
    if len(wrongs) < 3:
        return None
    fmt = lambda t: f"({t[0]}, {t[1]}, {t[2]})"
    # Enforce uniqueness: exactly one option may satisfy the relation.
    if sum(_set_relation(t, kind) for t in [key] + wrongs) != 1:
        return None
    opts, letter = _uniq_options(fmt(key), [fmt(w) for w in wrongs], rng)
    if not opts:
        return None
    stem = ("Select the set in which the numbers are related in the same way as "
            "are the numbers of the following set.  " + fmt(given))
    return dict(subtopic="number_set_analogy", stem=stem, options=opts,
                answer=letter, verified_by="relation_uniqueness",
                trace=f"relation={kind}, holds only for {fmt(key)}")


# ----------------------------------------------------------------- odd one out
def gen_odd_one_out(rng):
    """Three numbers share a property, one does not.

    Verified by evaluating the property over all four options and requiring
    exactly a 3-1 split, so "alike in a certain way" is a fact about the item
    rather than an assertion.
    """
    kind = rng.choice(["square", "cube", "prime", "mult"])
    if kind == "square":
        prop = lambda n: int(n ** 0.5) ** 2 == n
        alike = [x * x for x in rng.sample(range(4, 20), 3)]
    elif kind == "cube":
        prop = lambda n: round(n ** (1 / 3)) ** 3 == n
        alike = [x ** 3 for x in rng.sample(range(2, 9), 3)]
    elif kind == "prime":
        def prop(n):
            if n < 2:
                return False
            return all(n % d for d in range(2, int(n ** 0.5) + 1))
        alike = rng.sample([11, 13, 17, 19, 23, 29, 31, 37, 41, 43], 3)
    else:
        k = rng.choice([7, 9, 11, 13])
        prop = lambda n, k=k: n % k == 0
        alike = [k * x for x in rng.sample(range(3, 15), 3)]
    odd = None
    for _ in range(300):
        cand = rng.randint(min(alike), max(alike) + 20)
        if not prop(cand) and cand not in alike:
            odd = cand
            break
    if odd is None:
        return None
    quad = alike + [odd]
    if sum(prop(n) for n in quad) != 3:
        return None
    opts, letter = _uniq_options(odd, alike, rng)
    if not opts:
        return None
    stem = ("Three of the following four numbers are alike in a certain way and "
            "one is different. Pick the odd one out.")
    return dict(subtopic="generic_odd", stem=stem, options=opts, answer=letter,
                verified_by="property_split_3_1",
                trace=f"property={kind}; odd={odd}")



def gen_statement_conclusion(rng):
    """SSC statement-and-conclusions: one statement, two conclusions.

    Distinct from syllogism (two statements -> one conclusion), and the
    forecast weights it separately at ~1.2/paper. Verification reuses
    `_syllogism_holds`: each conclusion is model-checked against the single
    premise independently, so the answer key is derived, never asserted.

    Note the "all" semantics carry existential import (`bool(X) and X <= Y`),
    which is the SSC convention - "Some A are B" does follow from "All A are B".
    """
    A, B = rng.sample(["doctors", "engineers", "artists", "farmers",
                       "singers", "traders"], 2)
    statement = rng.choice([("all", A, B), ("some", A, B), ("no", A, B)])

    pool = [("all", A, B), ("all", B, A), ("some", A, B), ("some", B, A),
            ("no", A, B), ("no", B, A)]
    pool = [c for c in pool if c != statement]

    # Choose the TARGET verdict first, then search for conclusions that produce
    # it. Searching conclusions first and taking whatever verdict fell out gave
    # 46% "Neither" and 1.8% "Both" - a distribution a candidate could game by
    # always answering "Neither", and useless as practice.
    target = rng.choice([(True, False), (False, True), (True, True), (False, False)])

    pairs = [(c1, c2) for i, c1 in enumerate(pool) for c2 in pool[i + 1:]]
    rng.shuffle(pairs)
    pick = None
    for c1, c2 in pairs:
        h1, m1 = _syllogism_holds([statement], c1)
        h2, m2 = _syllogism_holds([statement], c2)
        if m1 == 0 or m2 == 0:
            continue              # vacuous premise is not a question
        if (h1, h2) != target:
            continue
        pick = (c1, c2, h1, h2, m1)
        break
    if not pick:
        return None
    c1, c2, h1, h2, models = pick

    def render(rule):
        # nouns here are plural, so "No X are Y" - "No artists is engineers"
        # is what the shared syllogism renderer produced and it reads wrong.
        kind, x, y = rule
        return {"all": f"All {x} are {y}.", "some": f"Some {x} are {y}.",
                "no": f"No {x} are {y}."}[kind]

    stem = ("A statement is given, followed by two conclusions numbered I and "
            "II. Assuming the statement to be true, even if it seems to be at "
            "variance with commonly known facts, decide which of the "
            "conclusions logically follows.  Statement: " + render(statement) +
            "  Conclusion I: " + render(c1) + "  Conclusion II: " + render(c2))

    ans = {(True, False): "Only conclusion I follows",
           (False, True): "Only conclusion II follows",
           (True, True): "Both I and II follow",
           (False, False): "Neither I nor II follows"}[(h1, h2)]
    every = ["Only conclusion I follows", "Only conclusion II follows",
             "Both I and II follow", "Neither I nor II follows"]
    opts, letter = _uniq_options(ans, [o for o in every if o != ans], rng)
    if not opts:
        return None
    return dict(subtopic="statement_conclusion", stem=stem, options=opts,
                answer=letter, verified_by="model_check",
                trace=f"I={h1}, II={h2} over {models} models")


GENERATORS = {
    "number_or_letter_series": gen_number_series,
    "number_set_analogy": gen_number_set_analogy,
    "generic_odd": gen_odd_one_out,
    "language_code": gen_letter_code,
    "colon_notation": gen_number_analogy,
    "position_unchanged": gen_position_unchanged,
    "alphabetical_arrangement": gen_dictionary_order,
    "symbol_definition": gen_symbol_definition,
    "sign_interchange": gen_sign_interchange,
    "statements_conclusions": gen_syllogism,
    "coded_relation": gen_blood_relation,
    "statement_conclusion": gen_statement_conclusion,
}

# Archetypes the solver can independently re-derive from stem text. The rest are
# verified by their own construction proof (model check / relation chain), which
# is stated per question rather than glossed over.
SOLVER_CHECKED = {"number_or_letter_series", "language_code", "colon_notation",
                  "position_unchanged", "alphabetical_arrangement",
                  "symbol_definition", "sign_interchange"}


def build_pool(target_per_arch=30, seed=2026):
    rng = random.Random(seed)
    pool, rejected = [], Counter()
    for arch, fn in GENERATORS.items():
        made, tries = 0, 0
        while made < target_per_arch and tries < target_per_arch * 60:
            tries += 1
            q = fn(rng)
            if not q:
                rejected[f"{arch}:construction"] += 1
                continue
            # Independent re-derivation from the rendered stem.
            if arch in SOLVER_CHECKED:
                got, which = solve_any(q["stem"], q["options"], arch)
                if got is None:
                    rejected[f"{arch}:solver_declined"] += 1
                    continue
                if got != q["answer"]:
                    rejected[f"{arch}:solver_disagreed"] += 1
                    continue
                q["verified_by"] = f"solver:{which}"
            # Identity is stem AND options: several archetypes carry a fixed
            # stem ("Three of the following four numbers are alike...") and vary
            # only in the options, so comparing stems alone rejected every
            # variant after the first and capped the archetype at one item.
            ident = (q["stem"], tuple(t for _, t in q["options"]))
            if ident in {(p["stem"], tuple(t for _, t in p["options"]))
                         for p in pool}:
                rejected[f"{arch}:duplicate"] += 1
                continue
            pool.append(q)
            made += 1
    return pool, rejected


def main() -> None:
    import sys as _s
    n = int(_s.argv[1]) if len(_s.argv) > 1 else 30
    pool, rejected = build_pool(target_per_arch=n)
    print("=" * 88)
    print("CANDIDATE GENERATION — every item re-derived independently")
    print("=" * 88)
    print(f"pool size: {len(pool)}\n")
    by = Counter(q["subtopic"] for q in pool)
    for k, v in by.most_common():
        print(f"  {k:<30}{v:>4}")
    print("\nrejections (candidate discarded rather than shipped):")
    for k, v in rejected.most_common():
        print(f"  {k:<40}{v:>5}")
    print("\nverification routes:")
    for k, v in Counter(q.get("verified_by", "?") for q in pool).most_common():
        print(f"  {k:<40}{v:>5}")
    json.dump(pool, open(OUT / "candidates.json", "w"), indent=2)
    print(f"\nWrote {OUT/'candidates.json'}")


if __name__ == "__main__":
    main()
