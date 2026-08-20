"""Exact solvers for computable Reasoning archetypes.

Purpose: derive the correct answer from the question's own content, so that no
answer key has to be trusted. This is load-bearing for this project, because:

  * official 2021-2024 papers contain NO key at all (verified: zero occurrences
    of "Correct Option" / "Answer Key" across a 2024 shift), and
  * the only keys in the corpus come from Tier-3 coaching reconstructions.

Every solver returns None rather than guessing. A wrong key in a practice paper
teaches the wrong answer, which is worse than omitting the question, so silence
is always the correct failure mode here.

Used twice: to verify claimed keys on real PYQs, and to verify the generator's
own answers on newly constructed questions. The second use is only meaningful
because the solver re-derives the answer from the rendered stem text, never from
the generator's internal state.
"""

from __future__ import annotations

import itertools
import re
import unicodedata
from fractions import Fraction

WORD_NUM = {
    "none": 0, "zero": 0, "nil": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12,
}
ALPHA = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.translate(str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"',
                                   "–": "-", "—": "-", "−": "-",
                                   "×": "*", "÷": "/"}))
    return re.sub(r"\s+", " ", s).strip()


# ------------------------------------------------------------------ option match
def _as_number(text: str):
    t = norm(text).lower().strip(" .")
    if t in WORD_NUM:
        return Fraction(WORD_NUM[t])
    m = re.fullmatch(r"-?\d+(?:\.\d+)?", t.replace(",", ""))
    if m:
        return Fraction(t.replace(",", ""))
    return None


def _disambiguate_by_options(candidates, options) -> str | None:
    """Pick the one candidate value that appears among the options.

    "7 : 49 :: 9 : ?" is ambiguous in isolation -- x7, +42 and squaring all fit
    the first pair. A well-posed multiple-choice item resolves this: only the
    intended relation's result is offered. So keep candidates that hit an option
    and accept only if exactly one survives. Two survivors means the item really
    is ambiguous, and None is the right answer.
    """
    hits = {}
    for v in candidates:
        letter = match_option(v, options)
        if letter:
            hits[letter] = v
    return next(iter(hits)) if len(hits) == 1 else None


def match_option(value, options) -> str | None:
    """Find the single option equal to `value`. Ambiguity returns None."""
    hits = []
    if isinstance(value, (int, float, Fraction)):
        target = Fraction(value).limit_denominator(10**6)
        for letter, text in options:
            n = _as_number(text)
            if n is not None and n == target:
                hits.append(letter)
    else:
        want = re.sub(r"[^a-z0-9]", "", norm(str(value)).lower())
        if not want:
            return None
        for letter, text in options:
            got = re.sub(r"[^a-z0-9]", "", norm(text).lower())
            if got == want:
                hits.append(letter)
    return hits[0] if len(hits) == 1 else None


# -------------------------------------------------- alphabetical fixed points
WORD_RX = re.compile(r"\b([A-Z]{4,})\b")


def solve_position_unchanged(stem: str, options) -> str | None:
    """"How many letters remain unchanged if WORD is alphabetically ordered?"

    Worked example: BINDER -> B,D,E,I,N,R vs B,I,N,D,E,R gives fixed points at
    positions 1 and 6, so the answer is Two.
    """
    s = norm(stem)
    if not re.search(r"remain unchanged|arranged (in )?(english )?alphabetical",
                     s, re.I):
        return None
    words = WORD_RX.findall(s)
    words = [w for w in words if w not in {"NOT", "ONLY", "NONE", "ENGLISH"}]
    if len(words) != 1:
        return None
    w = words[0]
    fixed = sum(1 for a, b in zip(w, "".join(sorted(w))) if a == b)
    return match_option(fixed, options)


# ------------------------------------------------------------ dictionary order
def solve_dictionary_order(stem: str, options) -> str | None:
    """Which word appears at position N in dictionary order."""
    s = norm(stem)
    if not re.search(r"dictionary", s, re.I):
        return None
    m = re.search(r"(?:at|in)\s*'?(first|second|third|fourth|fifth|last)'?\s*position",
                  s, re.I)
    if not m:
        return None
    items = re.findall(r"\b\d\.\s*([A-Za-z]+)", s)
    if len(items) < 4:
        return None
    order = {"first": 0, "second": 1, "third": 2, "fourth": 3, "fifth": 4,
             "last": len(items) - 1}
    idx = order[m.group(1).lower()]
    ranked = sorted(items, key=lambda w: w.lower())
    if idx >= len(ranked):
        return None
    return match_option(ranked[idx], options)


# --------------------------------------------------------------- number series
def _seq_from(stem: str) -> list[int | None]:
    """Parse a comma-separated run, keeping the gap as None."""
    s = norm(stem)
    runs = re.findall(r"(?:(?:-?\d+|\?|_+)\s*,\s*){2,}(?:-?\d+|\?|_+)", s)
    if not runs:
        return []
    toks = [t.strip() for t in max(runs, key=len).split(",")]
    out: list[int | None] = []
    for t in toks:
        if t in ("?", "") or set(t) == {"_"}:
            out.append(None)
        elif re.fullmatch(r"-?\d+", t):
            out.append(int(t))
        else:
            return []
    return out


def _fit_poly_predict(known: list[tuple[int, int]], gap: int, deg: int):
    """Exact Lagrange fit on deg+1 points, verified against the REMAINING points.

    Identifiability guard: deg+1 points always determine a degree-deg polynomial
    exactly, so a fit with len(known) == deg+1 is vacuous -- it cannot fail and
    therefore proves nothing. Requiring deg+2 leaves at least one point the fit
    has to survive.

    This was a live bug: 3, 6, 13, 28, ? has four known terms, so a cubic fitted
    them perfectly and confidently predicted 55, while the intended rule
    (x2+1, x2+2, x2+3) gives 59. The guard makes the solver decline instead.
    """
    if len(known) < deg + 2:
        return None
    pts = known[:deg + 1]

    def poly(x):
        total = Fraction(0)
        for i, (xi, yi) in enumerate(pts):
            term = Fraction(yi)
            for j, (xj, _) in enumerate(pts):
                if i != j:
                    term *= Fraction(x - xj, xi - xj)
            total += term
        return total

    for x, y in known:
        if poly(x) != y:
            return None
    v = poly(gap)
    return int(v) if v.denominator == 1 else None


def _fit_recurrence(seq: list[int | None], gap_i: int):
    """x_{n+1} = p*x_n + q, solved from the first two known consecutive gaps."""
    pairs = [(i, seq[i], seq[i + 1]) for i in range(len(seq) - 1)
             if seq[i] is not None and seq[i + 1] is not None]
    if len(pairs) < 2:
        return None
    (_, a1, b1), (_, a2, b2) = pairs[0], pairs[1]
    if a1 == a2:
        return None
    p = Fraction(b1 - b2, a1 - a2)
    q = Fraction(b1) - p * Fraction(a1)
    for _, a, b in pairs:
        if p * a + q != b:
            return None
    vals: list[int | None] = list(seq)
    for i in range(1, len(vals)):
        if vals[i] is None and vals[i - 1] is not None:
            v = p * vals[i - 1] + q
            if v.denominator != 1:
                return None
            vals[i] = int(v)
    return vals[gap_i]


def _fit_mul_increment(seq: list[int | None], gap_i: int):
    """a_{n+1} = p*a_n + (q0 + d*n) -- "x2 then +1, +2, +3 ...".

    A staple SSC family that a plain constant-offset recurrence cannot express,
    and whose absence made the solver reach for a spurious cubic instead.
    Solves q0 and d exactly from two consecutive known steps, then requires every
    other known step to hold.
    """
    for p in (2, 3, 4, 5):
        steps = [(i, seq[i], seq[i + 1]) for i in range(len(seq) - 1)
                 if seq[i] is not None and seq[i + 1] is not None]
        if len(steps) < 3:
            return None
        (i1, a1, b1), (i2, a2, b2) = steps[0], steps[1]
        if i1 == i2:
            continue
        # b = p*a + q0 + d*i
        r1, r2 = b1 - p * a1, b2 - p * a2
        d = Fraction(r2 - r1, i2 - i1)
        q0 = Fraction(r1) - d * i1
        if any(Fraction(b) != p * a + q0 + d * i for i, a, b in steps):
            continue
        vals: list[int | None] = list(seq)
        for i in range(1, len(vals)):
            if vals[i] is None and vals[i - 1] is not None:
                v = p * vals[i - 1] + q0 + d * (i - 1)
                if v.denominator != 1:
                    return None
                vals[i] = int(v)
        if vals[gap_i] is not None:
            return vals[gap_i]
    return None


def _fit_cyclic_diff(seq: list[int | None], gap_i: int):
    """Repeating difference cycle, e.g. -3, -8, -3, -8."""
    diffs = []
    for i in range(len(seq) - 1):
        if seq[i] is None or seq[i + 1] is None:
            diffs.append(None)
        else:
            diffs.append(seq[i + 1] - seq[i])
    known = [d for d in diffs if d is not None]
    if len(known) < 3:
        return None
    for period in (2, 3):
        if len(known) < period * 2:
            continue
        cyc = known[:period]
        if all(known[i] == cyc[i % period] for i in range(len(known))):
            if gap_i == 0:
                return None
            prev = seq[gap_i - 1]
            if prev is None:
                return None
            return prev + cyc[(gap_i - 1) % period]
    return None


def solve_number_series(stem: str, options) -> str | None:
    seq = _seq_from(stem)
    if not seq or seq.count(None) != 1:
        return None
    gap_i = seq.index(None)
    known = [(i, v) for i, v in enumerate(seq) if v is not None]
    if len(known) < 3:
        return None

    candidates = []
    for deg in (1, 2, 3):
        v = _fit_poly_predict(known, gap_i, deg)
        if v is not None:
            candidates.append(v)
            break
    for fn in (_fit_recurrence, _fit_mul_increment, _fit_cyclic_diff):
        v = fn(seq, gap_i)
        if v is not None:
            candidates.append(v)

    # Geometric
    ratios = []
    ok = True
    for i in range(len(seq) - 1):
        a, b = seq[i], seq[i + 1]
        if a is None or b is None:
            continue
        if a == 0:
            ok = False
            break
        ratios.append(Fraction(b, a))
    if ok and len(ratios) >= 2 and len(set(ratios)) == 1 and gap_i > 0:
        prev = seq[gap_i - 1]
        if prev is not None:
            v = Fraction(prev) * ratios[0]
            if v.denominator == 1:
                candidates.append(int(v))

    if not candidates:
        return None
    if len(set(candidates)) == 1:
        return match_option(candidates[0], options)
    # Competing rules disagree: let the options arbitrate, and stay silent if
    # more than one candidate is on offer (a genuinely under-determined series).
    return _disambiguate_by_options(set(candidates), options)


# ------------------------------------------------------------ letter-shift code
def _letters_only(w: str) -> str:
    return re.sub(r"[^A-Z]", "", w.upper())


def solve_letter_shift_code(stem: str, options) -> str | None:
    """"X is coded as Y, how is Z coded?" for a positional shift cipher."""
    s = norm(stem)
    if not re.search(r"code", s, re.I):
        return None
    pairs = re.findall(r"\b([A-Z]{3,})\b\s*(?:is (?:written|coded) as|as|->|=)\s*"
                       r"'?\b([A-Z]{3,})\b", s)
    pairs = [(_letters_only(a), _letters_only(b)) for a, b in pairs]
    pairs = [(a, b) for a, b in pairs if len(a) == len(b) and a != b]
    if not pairs:
        return None
    src, dst = pairs[0]
    shifts = [(ALPHA.index(b) - ALPHA.index(a)) % 26 for a, b in zip(src, dst)]

    caps = re.findall(r"\b([A-Z]{3,})\b", s)
    known = {src, dst}
    target = None
    for c in caps:
        cc = _letters_only(c)
        if cc not in known:
            target = cc
    if not target:
        return None

    if len(set(shifts)) == 1:
        k = shifts[0]
    elif len(shifts) == len(target):
        k = None
    else:
        return None

    if k is not None:
        coded = "".join(ALPHA[(ALPHA.index(ch) + k) % 26] for ch in target)
    else:
        coded = "".join(ALPHA[(ALPHA.index(ch) + s2) % 26]
                        for ch, s2 in zip(target, shifts))
    return match_option(coded, options)


# ----------------------------------------------------------- symbol arithmetic
SAFE = re.compile(r"^[\d\s+\-*/().]+$")


def _safe_eval(expr: str):
    if not SAFE.match(expr):
        return None
    try:
        v = eval(expr, {"__builtins__": {}}, {})  # noqa: S307 - guarded by SAFE
    except (ZeroDivisionError, SyntaxError, TypeError):
        return None
    return Fraction(v).limit_denominator(10**6) if isinstance(v, (int, float)) else None


def solve_symbol_definition(stem: str, options) -> str | None:
    """"If @ means +, # means -, ... evaluate <expr>"."""
    s = norm(stem)
    defs = dict(re.findall(r"'?([@#$%&*A-Za-z])'?\s*(?:means|=|stands for)\s*"
                            r"'?([+\-*/x])'?", s))
    if len(defs) < 2:
        return None
    trans = {k: ("*" if v.lower() == "x" else v) for k, v in defs.items()}
    m = re.search(r"(?:evaluate|value of|what is)\s*:?\s*(.+?)(?:\?|$)", s, re.I)
    if not m:
        return None
    expr = m.group(1)
    for k, v in trans.items():
        expr = expr.replace(k, v)
    expr = re.sub(r"[^0-9+\-*/(). ]", "", expr)
    val = _safe_eval(expr)
    if val is None:
        return None
    return match_option(val, options)


def solve_sign_interchange(stem: str, options) -> str | None:
    """Which pair of signs, when swapped, makes the equation true."""
    s = norm(stem)
    if not re.search(r"interchang", s, re.I):
        return None
    eq = re.search(r"([\d\s+\-*/().]+)=\s*(-?\d+(?:\.\d+)?)", s)
    if not eq:
        return None
    lhs, rhs = eq.group(1), Fraction(eq.group(2))
    good = []
    for letter, text in options:
        pair = re.findall(r"([+\-*/])", norm(text))
        if len(pair) != 2:
            continue
        a, b = pair
        swapped = "".join(b if ch == a else a if ch == b else ch for ch in lhs)
        v = _safe_eval(swapped)
        if v is not None and v == rhs:
            good.append(letter)
    return good[0] if len(good) == 1 else None


# --------------------------------------------------------------- number analogy
def _infer_op(a: int, b: int):
    """Candidate closed-form relations from a single pair, most specific first."""
    ops = []
    if a != 0 and b % a == 0:
        ops.append(("mul", b // a))
    ops.append(("add", b - a))
    if a > 0:
        for k in range(2, 5):
            if a ** k == b:
                ops.append(("pow", k))
        if a * a + a == b:
            ops.append(("sq_plus", 0))
    return ops


def _apply(op, a: int):
    kind, k = op
    if kind == "mul":
        return a * k
    if kind == "add":
        return a + k
    if kind == "pow":
        return a ** k
    if kind == "sq_plus":
        return a * a + a
    return None


def solve_number_analogy(stem: str, options) -> str | None:
    """A : B :: C : ? where a single relation explains the first pair."""
    s = norm(stem)
    nums = re.findall(r"(\d+)\s*:\s*(\d+)", s)
    if len(nums) < 1:
        return None
    a, b = int(nums[0][0]), int(nums[0][1])
    tail = re.search(r"(\d+)\s*:\s*\?", s)
    if not tail:
        return None
    c = int(tail.group(1))
    preds = set()
    for op in _infer_op(a, b):
        # Cross-check on a second complete pair when one exists.
        if len(nums) >= 2:
            a2, b2 = int(nums[1][0]), int(nums[1][1])
            if _apply(op, a2) != b2:
                continue
        v = _apply(op, c)
        if v is not None:
            preds.add(v)
    return _disambiguate_by_options(preds, options)


SOLVERS = {
    "position_unchanged": solve_position_unchanged,
    "alphabetical_arrangement": solve_dictionary_order,
    "number_or_letter_series": solve_number_series,
    "language_code": solve_letter_shift_code,
    "symbol_definition": solve_symbol_definition,
    "sign_interchange": solve_sign_interchange,
    "colon_notation": solve_number_analogy,
    "number_set_analogy": solve_number_analogy,
    "term_analogy": solve_number_analogy,
}

ALL_SOLVERS = [
    solve_position_unchanged, solve_dictionary_order, solve_number_series,
    solve_letter_shift_code, solve_symbol_definition, solve_sign_interchange,
    solve_number_analogy,
]


def solve_any(stem: str, options, subtopic: str | None = None) -> tuple[str | None, str | None]:
    """Try the subtopic's solver first, then every other. Returns (letter, which)."""
    if subtopic and subtopic in SOLVERS:
        r = SOLVERS[subtopic](stem, options)
        if r:
            return r, SOLVERS[subtopic].__name__
    for fn in ALL_SOLVERS:
        try:
            r = fn(stem, options)
        except Exception:
            continue
        if r:
            return r, fn.__name__
    return None, None
