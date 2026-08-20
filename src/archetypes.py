"""Mine the SOLUTION MECHANISM behind questions, not just their topic.

The topic forecast answers "how many series questions in 2026?". This answers the
question a candidate actually needs: "WHICH KIND of series, solved HOW?"

A number series is the cleanest case because the generating rule is recoverable
from the numbers themselves. `2, 4, 10, 28, ?, 244` is not merely "a series
question" -- it is a linear recurrence a(n) = 3a(n-1) - 2, and a candidate who
recognises that family solves it in seconds. If the family MIX is stable across
years, it is predictable in a way raw topic counts are not.

Families detected, in order of specificity (first match wins, because a GP is
also technically a linear recurrence and the simpler description is the useful
one):

    arithmetic          constant difference
    geometric           constant ratio
    linear_recurrence   a(n) = p*a(n-1) + q
    quadratic           second difference constant  (diffs form an AP)
    cubic               third difference constant
    alternating         two interleaved sub-series
    power_offset        n^2 or n^3 with a constant offset
    fibonacci_like      a(n) = a(n-1) + a(n-2)
    unknown             none of the above

Everything here is arithmetic on the extracted numbers -- no model, no guessing.
A sequence either satisfies a family's equation or it does not.
"""

from __future__ import annotations

import re
from fractions import Fraction

# 3+ comma-separated integers, optionally ending in ? or a blank
SEQ_RE = re.compile(r"(?:\d+\s*,\s*){2,}\d*\s*[?_]?(?:\s*,\s*(?:\d+|\?|_+))*")
NUM_RE = re.compile(r"\d+")


TOKEN_RE = re.compile(r"\d+|[?_]+")


def extract_sequence(text: str) -> list[int | None]:
    """Longest comma-separated run, with None marking the '?' to be solved.

    Keeping the gap as None rather than dropping it is essential: in
    "2, 4, 10, 28, ?, 244" the 244 is four steps from 28 only if the missing
    slot is counted. Collapsing to [2,4,10,28,244] makes every recurrence fail.
    """
    best: list[int | None] = []
    for m in SEQ_RE.finditer(text):
        toks = TOKEN_RE.findall(m.group())
        seq = [int(t) if t.isdigit() else None for t in toks]
        if len(seq) > len(best):
            best = seq
    return best


def _known(s: list) -> list[tuple[int, int]]:
    return [(i, v) for i, v in enumerate(s) if v is not None]


def _fits(s: list, gen) -> bool:
    """Does a generating function reproduce every KNOWN term at its own index?"""
    kn = _known(s)
    if len(kn) < 3:
        return False
    try:
        return all(gen(i) == v for i, v in kn)
    except (ZeroDivisionError, OverflowError, ValueError):
        return False


def _diffs(s: list[int]) -> list[int]:
    return [b - a for a, b in zip(s, s[1:])]


def _const(xs: list) -> bool:
    return len(xs) >= 2 and len(set(xs)) == 1


def classify_sequence(s: list) -> tuple[str, str]:
    """Return (family, human-readable rule). Handles a None gap for the '?' term."""
    if len(s) < 4 or len(_known(s)) < 3:
        return "too_short", ""

    # Gap-aware pass: fit a closed form against known terms at their own indices.
    kn = _known(s)
    (i0, v0), (i1, v1) = kn[0], kn[1]
    span = i1 - i0

    # arithmetic / geometric across the gap
    if span > 0 and (v1 - v0) % span == 0:
        d = (v1 - v0) // span
        if _fits(s, lambda i: v0 + d * (i - i0)) and d != 0:
            return "arithmetic", f"{'+' if d > 0 else ''}{d}"
    if span > 0 and v0 != 0:
        r = Fraction(v1, v0)
        if r.denominator == 1 or span == 1:
            rr = r ** Fraction(1, span) if span > 1 else r
            if isinstance(rr, Fraction) and rr.denominator == 1:
                if _fits(s, lambda i: v0 * int(rr) ** (i - i0)):
                    return "geometric", f"×{int(rr)}"
        else:
            try:
                root = round(float(r) ** (1 / span), 6)
                if abs(root - round(root)) < 1e-9 and round(root) > 1:
                    k = int(round(root))
                    if _fits(s, lambda i: v0 * k ** (i - i0)):
                        return "geometric", f"×{k}"
            except (ValueError, OverflowError):
                pass

    # linear recurrence from three CONSECUTIVE known terms
    for a in range(len(s) - 2):
        w = s[a:a + 3]
        if None in w or w[1] == w[0]:
            continue
        p = Fraction(w[2] - w[1], w[1] - w[0])
        q = Fraction(w[1]) - p * Fraction(w[0])

        def gen(i, a=a, w=w, p=p, q=q):
            v = Fraction(w[0])
            for _ in range(i - a):
                v = p * v + q
            return v if v.denominator != 1 else int(v)

        if p != 1 and _fits(s, gen):
            sign = "+" if q >= 0 else "−"
            return "linear_recurrence", f"×{p} {sign} {abs(q)}"

    # Gap-aware polynomial fit. Finite differences cannot cross a None, so a
    # quadratic with the '?' in the middle was previously unsolvable -- which was
    # the single largest source of "unknown" in the corpus. Fitting a degree-d
    # polynomial through the known (index, value) points handles any gap
    # position uniformly, and covers arithmetic/quadratic/cubic at once.
    for deg, name in ((2, "quadratic"), (3, "cubic")):
        if len(kn) >= deg + 2:
            poly = _fit_poly(kn, deg)
            if poly and _fits(s, poly):
                return name, f"degree-{deg} polynomial in n"

    # Cyclic differences: SSC uses repeating +a,+b,+a,+b patterns constantly.
    cyc = _cyclic_diff(s)
    if cyc:
        return "cyclic_difference", cyc

    # Prime-number differences -- a distinct family worth naming, because the
    # candidate must recognise the primes rather than compute a rule.
    pri = _prime_diff(s)
    if pri:
        return "prime_difference", pri

    # Alternating operations, e.g. ×2, +3, ×2, +3.
    alt = _alternating_ops(s)
    if alt:
        return "alternating_ops", alt

    if None in s:
        s = [v for v in s if v is not None]
        if len(s) < 4:
            return "too_short", ""

    d1 = _diffs(s)
    if _const(d1):
        return "arithmetic", f"+{d1[0]}" if d1[0] >= 0 else f"{d1[0]}"

    # geometric: exact integer or rational ratio held throughout
    if all(a != 0 for a in s[:-1]):
        ratios = [Fraction(b, a) for a, b in zip(s, s[1:])]
        if _const(ratios):
            r = ratios[0]
            return "geometric", f"×{r}" if r.denominator != 1 else f"×{r.numerator}"

    # linear recurrence a(n) = p*a(n-1) + q, solved from the first two steps
    if len(s) >= 4 and s[1] != s[0]:
        try:
            p = Fraction(s[2] - s[1], s[1] - s[0])
            q = Fraction(s[1]) - p * Fraction(s[0])
            if all(Fraction(s[i + 1]) == p * Fraction(s[i]) + q for i in range(len(s) - 1)):
                if p == 1:
                    return "arithmetic", f"+{q}"
                sign = "+" if q >= 0 else "−"
                return "linear_recurrence", f"×{p} {sign} {abs(q)}"
        except ZeroDivisionError:
            pass

    d2 = _diffs(d1)
    if _const(d2):
        return "quadratic", f"differences form an AP (+{d2[0]} each)"

    if len(d2) >= 2:
        d3 = _diffs(d2)
        if _const(d3):
            return "cubic", f"third difference constant ({d3[0]})"

    # alternating: odd- and even-indexed terms each follow their own simple rule
    if len(s) >= 6:
        a, b = s[0::2], s[1::2]
        fa, _ = classify_sequence(a + [0] * (4 - len(a))) if len(a) >= 4 else ("", "")
        fb, _ = classify_sequence(b + [0] * (4 - len(b))) if len(b) >= 4 else ("", "")
        if len(a) >= 3 and len(b) >= 3:
            da, db = _diffs(a), _diffs(b)
            if _const(da) and _const(db):
                return "alternating", f"two interleaved APs (+{da[0]}, +{db[0]})"

    # n^k with a constant offset
    for k, name in ((2, "square"), (3, "cube")):
        for start in (1, 2, 3):
            offs = {v - (i + start) ** k for i, v in enumerate(s)}
            if len(offs) == 1:
                o = offs.pop()
                sign = "+" if o >= 0 else "−"
                return "power_offset", f"n^{k} {sign} {abs(o)} (from n={start})"

    if len(s) >= 5 and all(s[i] == s[i - 1] + s[i - 2] for i in range(2, len(s))):
        return "fibonacci_like", "a(n) = a(n−1) + a(n−2)"

    # "Find the wrong term": SSC asks this constantly, and it is a distinct
    # archetype -- the sequence follows a clean rule EXCEPT at one position, and
    # the task is to spot the outlier rather than continue the pattern. Detect it
    # by removing each term in turn and re-classifying.
    if len(s) >= 5:
        for drop in range(len(s)):
            trimmed = s[:drop] + s[drop + 1:]
            fam, rule = _classify_clean(trimmed)
            if fam not in ("unknown", "too_short"):
                return f"wrong_term_{fam}", f"{rule} (broken at position {drop + 1})"

    return "unknown", ""


_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
           73, 79, 83, 89, 97, 101, 103, 107, 109, 113]


def _fit_poly(kn: list[tuple[int, int]], deg: int):
    """Exact polynomial through the first deg+1 known points, via Lagrange.

    Fractions, not floats: a float fit accumulates error and then 'verifies'
    against rounded values, which silently accepts sequences that do not fit.
    """
    pts = kn[: deg + 1]
    if len({i for i, _ in pts}) != len(pts):
        return None

    def poly(x: int):
        total = Fraction(0)
        for j, (xj, yj) in enumerate(pts):
            term = Fraction(yj)
            for m, (xm, _) in enumerate(pts):
                if m != j:
                    term *= Fraction(x - xm, xj - xm)
            total += term
        return int(total) if total.denominator == 1 else None

    return poly


def _prefix(s: list) -> list[int]:
    """Leading run of known terms — where a step rule can be read off directly."""
    out = []
    for v in s:
        if v is None:
            break
        out.append(v)
    return out


def _verify_steps(s: list, steps) -> bool:
    """Generate forward from s[0] using per-step deltas and check every known term.

    This is what makes the detectors gap-tolerant: the '?' is simply generated
    along with everything else, so a rule spanning the gap still verifies.
    """
    if s[0] is None:
        return False
    cur = s[0]
    for i in range(1, len(s)):
        d = steps(i - 1)
        if d is None:
            return False
        cur = cur + d
        if s[i] is not None and s[i] != cur:
            return False
    return True


def _cyclic_diff(s: list) -> str | None:
    """Repeating difference cycle, e.g. -3, -8, -3, -8."""
    pre = _prefix(s)
    if len(s) < 6 or len(pre) < 3:
        return None
    d = _diffs(pre)
    for period in (2, 3):
        if len(d) < period or len(set(d[:period])) < 2:
            continue
        cyc = d[:period]
        if _verify_steps(s, lambda i, c=cyc, p=period: c[i % p]):
            return "repeating " + ", ".join(f"{x:+d}" for x in cyc)
    return None


def _prime_diff(s: list) -> str | None:
    """Differences that walk consecutive primes, ascending or descending."""
    pre = _prefix(s)
    if len(s) < 5 or len(pre) < 3:
        return None
    d0 = _diffs(pre)
    sign = -1 if d0[0] < 0 else 1
    mags = [abs(x) for x in d0]
    if not all(m in _PRIMES for m in mags):
        return None
    idx = [_PRIMES.index(m) for m in mags]
    steps_idx = {b - a for a, b in zip(idx, idx[1:])}
    if len(steps_idx) != 1:
        return None
    stride = steps_idx.pop()
    start = idx[0]

    def gen(i, start=start, stride=stride, sign=sign):
        j = start + stride * i
        return sign * _PRIMES[j] if 0 <= j < len(_PRIMES) else None

    if _verify_steps(s, gen):
        seq = [abs(gen(i)) for i in range(min(4, len(s) - 1)) if gen(i) is not None]
        return "consecutive prime differences (" + ", ".join(map(str, seq)) + " …)"
    return None


def _alternating_ops(s: list) -> str | None:
    """Two operations applied in turn, e.g. ×2 then +3."""
    pre = _prefix(s)
    if len(s) < 5 or len(pre) < 3:
        return None

    def infer(a: int, b: int):
        if a != 0 and b % a == 0 and b // a != 1:
            return ("mul", b // a)
        return ("add", b - a)

    op0, op1 = infer(pre[0], pre[1]), infer(pre[1], pre[2])
    if op0 == op1:
        return None

    def apply(v, op):
        return v * op[1] if op[0] == "mul" else v + op[1]

    cur = s[0]
    for i in range(1, len(s)):
        cur = apply(cur, op0 if (i - 1) % 2 == 0 else op1)
        if s[i] is not None and s[i] != cur:
            return None

    def fmt(op):
        return f"×{op[1]}" if op[0] == "mul" else f"{op[1]:+d}"

    return f"alternating {fmt(op0)} then {fmt(op1)}"


def _classify_clean(s: list[int]) -> tuple[str, str]:
    """Family check on a gapless sequence. Split out so the wrong-term search
    cannot recurse into itself."""
    if len(s) < 4:
        return "too_short", ""
    d1 = _diffs(s)
    if _const(d1):
        return "arithmetic", f"+{d1[0]}"
    if all(a != 0 for a in s[:-1]):
        ratios = [Fraction(b, a) for a, b in zip(s, s[1:])]
        if _const(ratios) and ratios[0].denominator == 1:
            return "geometric", f"×{ratios[0].numerator}"
    d2 = _diffs(d1)
    if _const(d2):
        return "quadratic", f"differences form an AP (+{d2[0]} each)"
    for k in (2, 3):
        for start in (1, 2, 3):
            offs = {v - (i + start) ** k for i, v in enumerate(s)}
            if len(offs) == 1:
                o = offs.pop()
                return "power_offset", f"n^{k} {'+' if o >= 0 else '−'} {abs(o)}"
    return "unknown", ""


# ---------------------------------------------------------------- coding-decoding

PAIR_RE = re.compile(r"['\"‘’“”]?\b([A-Z]{3,8})\b['\"‘’“”]?\s*(?:is (?:written|coded)|→|->|as)\s*"
                     r"['\"‘’“”]?\b([A-Z]{3,8})\b", re.IGNORECASE)


def classify_coding(text: str) -> tuple[str, str]:
    """Infer the letter transform from a worked example in the stem."""
    m = PAIR_RE.search(text)
    if not m:
        return "unknown", ""
    a, b = m.group(1).upper(), m.group(2).upper()
    if len(a) != len(b):
        return "length_change", ""

    shifts = [(ord(y) - ord(x)) % 26 for x, y in zip(a, b)]
    if _const(shifts):
        k = shifts[0]
        k = k - 26 if k > 13 else k
        return "uniform_shift", f"{'+' if k >= 0 else ''}{k} per letter"
    if b == a[::-1]:
        return "reversal", "reverse the word"
    if sorted(a) == sorted(b):
        return "anagram_positional", "letters rearranged"
    # opposite letter (A<->Z)
    if all(ord(y) - ord("A") == 25 - (ord(x) - ord("A")) for x, y in zip(a, b)):
        return "opposite_letter", "A↔Z complement"
    if len(set(shifts)) == 2:
        return "alternating_shift", f"shifts {sorted(set(shifts))}"
    return "mixed_shift", f"shifts {shifts}"
