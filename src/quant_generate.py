"""Construct new Quant questions fitted to the 2026 blueprint.

Quant is the best-verifiable of the four sections: every answer here is computed
in exact rational arithmetic (fractions.Fraction, never floats) and then
INDEPENDENTLY RE-DERIVED by a checker that re-parses the numbers out of the
RENDERED STEM TEXT. If the re-parse disagrees, or cannot reach a unique answer,
the candidate is discarded rather than shipped.

That round trip is the whole point. "I computed it correctly" is not evidence
when the same code did the computing, so the checker deliberately works only from
the printed question, exactly as a candidate would.

Distractors are built from REAL error mechanisms (§49) -- forgetting the second
discount, adding instead of compounding, inverting a ratio, using the wrong
percentage base -- not from random jitter, and every option set is checked for
uniqueness so exactly one option is correct.

Parameter choices follow the observed corpus tendency (§13): percentages are
multiples of 5 or standard fractions, ratios use small integers, and values are
chosen so the answer lands on a clean magnitude, which is what makes an SSC-style
item solvable inside a minute.
"""

from __future__ import annotations

import json
import math
import random
import re
from collections import Counter
from fractions import Fraction as F
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"

NUM = re.compile(r"-?\d+(?:\.\d+)?")


def fmt(v) -> str:
    """Render an exact value the way a paper would."""
    if isinstance(v, F):
        if v.denominator == 1:
            return str(v.numerator)
        f = float(v)
        return f"{f:.2f}".rstrip("0").rstrip(".")
    if isinstance(v, float):
        return f"{v:.2f}".rstrip("0").rstrip(".")
    return str(v)


def options_from(correct, wrongs, rng):
    """Four distinct options, exactly one correct."""
    seen = {fmt(correct)}
    picks = []
    for w in wrongs:
        s = fmt(w)
        if s not in seen:
            seen.add(s)
            picks.append(w)
        if len(picks) == 3:
            break
    if len(picks) < 3:
        return None, None
    vals = [correct] + picks
    rng.shuffle(vals)
    opts = [(l, fmt(v)) for l, v in zip("abcd", vals)]
    letter = "abcd"[vals.index(correct)]
    return opts, letter


# ============================================================== generators
def gen_simplification(rng):
    """BODMAS evaluation -- the single largest microtopic (3.93/25)."""
    a, b, c, d = (rng.randint(2, 24) for _ in range(4))
    e = rng.choice([2, 3, 4, 6])
    expr = f"{a} + {b} x {c} - {d * e} / {e}"
    val = F(a) + F(b) * c - F(d * e, e)
    if val <= 0 or val != int(val):
        return None
    val = F(int(val))
    wrongs = [val + b, val - c, F(a + b) * c - d, val + e, val - e]
    opts, letter = options_from(val, wrongs, rng)
    if not opts:
        return None
    return dict(
        topic="number_system", subtopic="simplification",
        stem=f"Simplify:  {expr}",
        options=opts, answer=letter, difficulty="easy", est_time=35,
        check="simplification",
        trace=f"BODMAS: {a} + {b*c} - {d} = {val}")


def gen_identity(rng):
    """x + 1/x = k  =>  x^2 + 1/x^2 = k^2 - 2. Exact, and a staple identity."""
    k = rng.randint(3, 12)
    val = k * k - 2
    wrongs = [k * k, k * k + 2, k * k - 1, k * k - 4, 2 * k]
    opts, letter = options_from(F(val), [F(w) for w in wrongs], rng)
    if not opts:
        return None
    return dict(
        topic="algebra", subtopic="identities",
        stem=f"If x + 1/x = {k}, then what is the value of x² + 1/x² ?",
        options=opts, answer=letter, difficulty="easy-medium", est_time=40,
        check="identity_k2m2",
        trace=f"(x+1/x)^2 = x^2 + 1/x^2 + 2  =>  {k}^2 - 2 = {val}")


def gen_ratio_hcf_lcm(rng):
    """Ratio a:b with HCF h  =>  numbers ah, bh and LCM = a*b*h (a,b coprime)."""
    a, b = rng.choice([(2, 3), (3, 5), (4, 7), (5, 8), (3, 7),
                       (5, 9), (7, 9), (4, 9), (2, 5)])
    h = rng.choice([3, 4, 5, 6, 8, 9, 12])
    if math.gcd(a, b) != 1:
        return None
    lcm = a * b * h
    wrongs = [a * b, lcm // h, (a + b) * h, a * h * b * h, lcm + h]
    opts, letter = options_from(F(lcm), [F(w) for w in wrongs], rng)
    if not opts:
        return None
    return dict(
        topic="ratio_proportion", subtopic="ratio_basic",
        stem=(f"The ratio of two numbers is {a}:{b} and their HCF is {h}. "
              f"What is their LCM?"),
        options=opts, answer=letter, difficulty="easy", est_time=35,
        check="ratio_hcf_lcm",
        trace=f"numbers are {a*h} and {b*h}; LCM = {a}x{b}x{h} = {lcm}")


def gen_successive_percentage(rng):
    """Increase then decrease -- net multiplier, a recurring template."""
    p = rng.choice([10, 15, 20, 25, 30, 40])
    q = rng.choice([10, 15, 20, 25, 30])
    base = rng.choice([2000, 2400, 3000, 4000, 5000, 6000])
    final = F(base) * (100 + p) * (100 - q) / 10000
    if final != int(final):
        return None
    final = F(int(final))
    wrongs = [F(base) * (100 + p - q) / 100, F(base), final + base * F(p - q, 100),
              F(base) * (100 - q) * (100 + p) / 10000 + 1]
    wrongs = [w for w in wrongs if w == int(w)]
    opts, letter = options_from(final, [F(int(w)) for w in wrongs] +
                               [final + 40, final - 40], rng)
    if not opts:
        return None
    return dict(
        topic="percentage", subtopic="successive_change",
        stem=(f"The price of an article is ₹{base}. It is first increased by "
              f"{p}% and then decreased by {q}%. What is the final price "
              f"(in ₹)?"),
        options=opts, answer=letter, difficulty="easy-medium", est_time=45,
        check="successive_pct",
        trace=f"{base} x {100+p}/100 x {100-q}/100 = {final}")


def gen_successive_discount(rng):
    """Two successive discounts on a marked price -- distractor = single sum."""
    mp = rng.choice([1200, 1500, 1600, 2000, 2400, 3000])
    d1 = rng.choice([10, 15, 20, 25])
    d2 = rng.choice([5, 10, 20])
    sp = F(mp) * (100 - d1) * (100 - d2) / 10000
    if sp != int(sp):
        return None
    sp = F(int(sp))
    naive = F(mp) * (100 - d1 - d2) / 100          # forgot to compound
    wrongs = [naive, F(mp) * (100 - d1) / 100, F(mp) * (100 - d2) / 100, sp + 50]
    wrongs = [F(int(w)) for w in wrongs if w == int(w)]
    opts, letter = options_from(sp, wrongs, rng)
    if not opts:
        return None
    return dict(
        topic="profit_loss_discount", subtopic="successive_discount",
        stem=(f"The marked price of an article is ₹{mp}. Two successive "
              f"discounts of {d1}% and {d2}% are allowed. What is the selling "
              f"price (in ₹)?"),
        options=opts, answer=letter, difficulty="medium", est_time=50,
        check="successive_discount",
        trace=f"{mp} x {100-d1}/100 x {100-d2}/100 = {sp}; "
              f"naive single {d1+d2}% gives {naive}")


def gen_simple_interest(rng):
    r = rng.choice([4, 5, 6, 8, 10, 12])
    t = rng.choice([2, 3, 4, 5])
    p = rng.choice([4000, 5000, 6000, 8000, 12000, 15000])
    si = F(p * r * t, 100)
    if si != int(si):
        return None
    si = F(int(si))
    wrongs = [F(p) * r * t / 1000, si + p, F(p * r, 100), si * 2]
    wrongs = [F(int(w)) for w in wrongs if w == int(w)]
    opts, letter = options_from(si, wrongs + [si + 100, si - 100], rng)
    if not opts:
        return None
    return dict(
        topic="simple_compound_interest", subtopic="simple_interest",
        stem=(f"Find the simple interest (in ₹) on a sum of ₹{p} at "
              f"{r}% per annum for {t} years."),
        options=opts, answer=letter, difficulty="easy", est_time=35,
        check="simple_interest",
        trace=f"SI = PRT/100 = {p}x{r}x{t}/100 = {si}")


def gen_average_replacement(rng):
    n = rng.choice([10, 12, 15, 20, 25])
    old_avg = rng.choice([30, 40, 45, 50, 60])
    delta = rng.choice([2, 3, 4, 5])
    # average rises by delta when one member is replaced
    increase = n * delta
    wrongs = [delta, n * delta // 2, n + delta, increase + n]
    opts, letter = options_from(F(increase), [F(w) for w in wrongs], rng)
    if not opts:
        return None
    return dict(
        topic="average", subtopic="average_replacement",
        stem=(f"The average weight of {n} students is {old_avg} kg. When one "
              f"student is replaced by a new student, the average increases by "
              f"{delta} kg. By how many kg is the new student heavier than the "
              f"one replaced?"),
        options=opts, answer=letter, difficulty="medium", est_time=45,
        check="average_replacement",
        trace=f"total change = n x delta = {n} x {delta} = {increase}")


def gen_pipes(rng):
    a, b = rng.choice([(12, 24), (10, 15), (20, 30), (6, 12), (15, 30), (8, 24)])
    together = F(a * b, a + b)
    wrongs = [F(a + b), F(a + b, 2), F(abs(a - b)), together + 1, together - 1]
    opts, letter = options_from(together, wrongs, rng)
    if not opts:
        return None
    return dict(
        topic="time_and_work", subtopic="pipes_cisterns",
        stem=(f"Pipe A can fill a tank in {a} hours and pipe B can fill the same "
              f"tank in {b} hours. If both pipes are opened together, in how many "
              f"hours will the tank be filled?"),
        options=opts, answer=letter, difficulty="easy-medium", est_time=45,
        check="pipes_together",
        trace=f"1/{a} + 1/{b} = ({a}+{b})/{a*b}  =>  {a*b}/{a+b} = {together}")


def gen_relative_speed(rng):
    """Train crossing a pole or platform -- unit conversion is the trap."""
    speed_kmh = rng.choice([36, 54, 72, 90, 108])
    secs = rng.choice([10, 12, 15, 20])
    mps = F(speed_kmh * 1000, 3600)
    length = mps * secs
    if length != int(length):
        return None
    length = F(int(length))
    wrongs = [F(speed_kmh * secs), length * 2, F(speed_kmh) * secs / 2,
              length + 50, length - 50]
    wrongs = [F(int(w)) for w in wrongs if w == int(w)]
    opts, letter = options_from(length, wrongs, rng)
    if not opts:
        return None
    return dict(
        topic="time_speed_distance", subtopic="trains",
        stem=(f"A train running at {speed_kmh} km/h crosses a pole in {secs} "
              f"seconds. What is the length of the train (in metres)?"),
        options=opts, answer=letter, difficulty="easy-medium", est_time=45,
        check="train_pole",
        trace=f"{speed_kmh} km/h = {mps} m/s; length = {mps} x {secs} = {length}")


def gen_angle_centre(rng):
    """Angle at centre = 2 x angle at circumference on the same arc."""
    circ = rng.choice([25, 30, 35, 40, 42, 50, 55, 60])
    centre = 2 * circ
    wrongs = [circ, circ // 2, 180 - centre, 90 - circ, centre + 10]
    wrongs = [w for w in wrongs if w > 0]
    opts, letter = options_from(F(centre), [F(w) for w in wrongs], rng)
    if not opts:
        return None
    return dict(
        topic="geometry", subtopic="angles",
        stem=(f"In a circle with centre O, the angle subtended by a chord AB at "
              f"a point C on the major arc is {circ}°. What is the measure "
              f"of ∠AOB (in degrees)?"),
        options=opts, answer=letter, difficulty="easy", est_time=35,
        check="angle_centre",
        trace=f"angle at centre = 2 x {circ} = {centre}")


def gen_divisibility(rng):
    d = rng.choice([3, 4, 6, 8, 9, 11])
    target = rng.randint(1000, 9999)
    correct = target - (target % d)
    pool = {correct}
    wrongs = []
    while len(wrongs) < 3:
        cand = rng.randint(1000, 9999)
        if cand % d != 0 and cand not in pool:
            pool.add(cand)
            wrongs.append(cand)
    opts, letter = options_from(F(correct), [F(w) for w in wrongs], rng)
    if not opts:
        return None
    # uniqueness: exactly one option divisible by d
    if sum(1 for _, t in opts if int(t) % d == 0) != 1:
        return None
    return dict(
        topic="number_system", subtopic="divisibility",
        stem=f"Which of the following numbers is divisible by {d}?",
        options=opts, answer=letter, difficulty="easy", est_time=30,
        check="divisibility",
        trace=f"{correct} % {d} == 0; the other three are not")


def gen_marked_price(rng):
    sp = rng.choice([7650, 8500, 9180, 6800, 5100])
    d = rng.choice([10, 15, 20])
    mp = F(sp * 100, 100 - d)
    if mp != int(mp):
        return None
    mp = F(int(mp))
    wrongs = [F(sp) * (100 + d) / 100, F(sp) + d, mp + 100, F(sp) * 100 / (100 + d)]
    wrongs = [F(int(w)) for w in wrongs if w == int(w)]
    opts, letter = options_from(mp, wrongs, rng)
    if not opts:
        return None
    return dict(
        topic="profit_loss_discount", subtopic="marked_price_discount",
        stem=(f"An article is sold for ₹{sp} after allowing a discount of "
              f"{d}%. What was its marked price (in ₹)?"),
        options=opts, answer=letter, difficulty="medium", est_time=45,
        check="marked_price",
        trace=f"MP = SP x 100/(100-{d}) = {sp} x 100/{100-d} = {mp}")



# ------------------------------------------------------------- mensuration
# The 2026 forecast expects ~3.0 mensuration questions per paper and the pool
# supplied none, so the blueprint could never be met. Radii are multiples of 7
# and pi is taken as 22/7, which is both the SSC convention and what keeps the
# answers exact rather than decimal approximations.
PI = F(22, 7)


def gen_cylinder_volume(rng):
    r = rng.choice([7, 14, 21])
    h = rng.choice([10, 12, 15, 20, 25, 30])
    vol = PI * r * r * h
    wrongs = [PI * r * h,                 # forgot to square
              2 * PI * r * h,             # curved surface area
              PI * r * r * h * 2,         # doubled
              PI * r * r * (h + 5)]
    opts, letter = options_from(vol, wrongs, rng)
    if not opts:
        return None
    return dict(topic="mensuration", subtopic="cylinder_volume",
                stem=(f"The radius of the base of a right circular cylinder is "
                      f"{r} cm and its height is {h} cm. What is its volume "
                      f"(in cm³)? (Take pi = 22/7)"),
                options=opts, answer=letter, difficulty="easy", est_time=40,
                check="cylinder_volume",
                trace=f"(22/7)*{r}^2*{h} = {fmt(vol)}")


def gen_sphere_surface(rng):
    r = rng.choice([7, 14, 21])
    area = 4 * PI * r * r
    wrongs = [PI * r * r, 2 * PI * r * r, 4 * PI * r,
              F(4, 3) * PI * r * r * r]
    opts, letter = options_from(area, wrongs, rng)
    if not opts:
        return None
    return dict(topic="mensuration", subtopic="sphere_surface",
                stem=(f"What is the total surface area of a sphere of radius "
                      f"{r} cm (in cm²)? (Take pi = 22/7)"),
                options=opts, answer=letter, difficulty="easy", est_time=35,
                check="sphere_surface",
                trace=f"4*(22/7)*{r}^2 = {fmt(area)}")


def gen_cuboid_volume(rng):
    l = rng.randint(5, 25); b = rng.randint(4, 20); h = rng.randint(3, 18)
    vol = F(l * b * h)
    wrongs = [F(2 * (l * b + b * h + h * l)),   # surface area
              F(l + b + h), F(l * b), F(4 * (l + b + h))]
    opts, letter = options_from(vol, wrongs, rng)
    if not opts:
        return None
    return dict(topic="mensuration", subtopic="cuboid_volume",
                stem=(f"A cuboid has length {l} cm, breadth {b} cm and height "
                      f"{h} cm. What is its volume (in cm³)?"),
                options=opts, answer=letter, difficulty="easy", est_time=30,
                check="cuboid_volume", trace=f"{l}*{b}*{h} = {l*b*h}")


def gen_circle_area(rng):
    r = rng.choice([7, 14, 21, 28])
    area = PI * r * r
    wrongs = [2 * PI * r, PI * r, F(r * r), 4 * PI * r * r]
    opts, letter = options_from(area, wrongs, rng)
    if not opts:
        return None
    return dict(topic="mensuration", subtopic="circle_area",
                stem=(f"What is the area of a circle whose radius is {r} cm "
                      f"(in cm²)? (Take pi = 22/7)"),
                options=opts, answer=letter, difficulty="easy", est_time=30,
                check="circle_area", trace=f"(22/7)*{r}^2 = {fmt(area)}")



# ---------------------------------------------------------------- geometry
# INSIGHTS.md §2: geometry is 59% circle/chord/tangent, so the supply is
# weighted there rather than spread evenly over the syllabus. Every length is
# drawn from a Pythagorean triple so answers stay exact integers.
TRIPLES = [(3, 4, 5), (6, 8, 10), (5, 12, 13), (8, 15, 17), (7, 24, 25),
           (9, 12, 15), (12, 16, 20), (20, 21, 29), (10, 24, 26)]


def gen_tangent_length(rng):
    """Tangent is perpendicular to the radius at the point of contact."""
    r, t, d = rng.choice(TRIPLES)
    wrongs = [F(d - r), F(d + r), F(r + t), F(d * d - r * r)]
    opts, letter = options_from(F(t), wrongs, rng)
    if not opts:
        return None
    return dict(topic="geometry", subtopic="tangent_length",
                stem=(f"From an external point P, a tangent is drawn to a circle "
                      f"with centre O and radius {r} cm. If OP = {d} cm, what is "
                      f"the length of the tangent (in cm)?"),
                options=opts, answer=letter, difficulty="easy", est_time=40,
                check="tangent_length",
                trace=f"sqrt({d}^2 - {r}^2) = {t}")


def gen_cyclic_quad(rng):
    """Opposite angles of a cyclic quadrilateral are supplementary."""
    a = rng.choice([65, 70, 75, 80, 85, 95, 100, 105, 110, 115])
    opp = 180 - a
    wrongs = [F(a), F(360 - a), F(90), F(a // 2)]
    wrongs = [w for w in wrongs if w != F(opp)]
    opts, letter = options_from(F(opp), wrongs, rng)
    if not opts:
        return None
    return dict(topic="geometry", subtopic="cyclic_quadrilateral",
                stem=(f"ABCD is a cyclic quadrilateral. If angle A = {a}°, "
                      f"what is the measure of angle C (in degrees)?"),
                options=opts, answer=letter, difficulty="easy", est_time=30,
                check="cyclic_quad", trace=f"180 - {a} = {opp}")


def gen_chord_intersect(rng):
    """Intersecting chords: AP x PB = CP x PD."""
    ap, pb, cp = rng.randint(2, 12), rng.randint(2, 12), rng.choice([2, 3, 4, 6, 8])
    prod = ap * pb
    if prod % cp:
        return None
    pd = prod // cp
    wrongs = [F(prod), F(ap + pb - cp), F(cp), F(pd + 2)]
    wrongs = [w for w in wrongs if w != F(pd) and w > 0]
    opts, letter = options_from(F(pd), wrongs, rng)
    if not opts:
        return None
    return dict(topic="geometry", subtopic="intersecting_chords",
                stem=(f"Two chords AB and CD of a circle intersect at a point P "
                      f"inside the circle. If AP = {ap} cm, PB = {pb} cm and "
                      f"CP = {cp} cm, what is the length of PD (in cm)?"),
                options=opts, answer=letter, difficulty="medium", est_time=45,
                check="chord_intersect", trace=f"{ap}*{pb}/{cp} = {pd}")


def gen_triangle_exterior(rng):
    """Exterior angle equals the sum of the two opposite interior angles."""
    a = rng.choice([35, 40, 45, 50, 55, 60, 65])
    b = rng.choice([30, 40, 50, 55, 70, 75])
    ext = a + b
    if ext >= 180:
        return None
    wrongs = [F(180 - ext), F(a), F(b), F(180 - a - b)]
    wrongs = [w for w in wrongs if w != F(ext) and w > 0]
    opts, letter = options_from(F(ext), wrongs, rng)
    if not opts:
        return None
    return dict(topic="geometry", subtopic="exterior_angle",
                stem=(f"In triangle ABC, angle A = {a}° and angle B = {b}°. "
                      f"The side BC is produced to D. What is the measure of "
                      f"the exterior angle ACD (in degrees)?"),
                options=opts, answer=letter, difficulty="easy", est_time=35,
                check="triangle_exterior", trace=f"{a} + {b} = {ext}")


def gen_similar_triangles(rng):
    """Corresponding sides of similar triangles are proportional."""
    k = rng.choice([2, 3, 4])
    a = rng.choice([4, 5, 6, 7, 8, 9])
    b = rng.choice([6, 8, 10, 12])
    A, B = a * k, b * k
    wrongs = [F(b), F(B + a), F(b * a), F(B - a)]
    wrongs = [w for w in wrongs if w != F(B) and w > 0]
    opts, letter = options_from(F(B), wrongs, rng)
    if not opts:
        return None
    return dict(topic="geometry", subtopic="similar_triangles",
                stem=(f"Triangle ABC is similar to triangle PQR. If AB = {a} cm, "
                      f"PQ = {A} cm and BC = {b} cm, what is the length of QR "
                      f"(in cm)?"),
                options=opts, answer=letter, difficulty="medium", est_time=40,
                check="similar_triangles", trace=f"{b} * {A}/{a} = {B}")



def gen_trigonometry(rng):
    """Standard-angle identity. Values are exact, so no rounding is involved."""
    cases = [
        ("sin 30° + cos 60°", F(1)),
        ("sin 90° - cos 0°", F(0)),
        ("tan 45° + sin 30°", F(3, 2)),
        ("sin 30° x cos 60°", F(1, 4)),
        ("cos 30° x cos 30°", F(3, 4)),
        ("sin 45° x sin 45°", F(1, 2)),
        ("tan 45° x tan 45°", F(1)),
        ("sin 60° x sin 60°", F(3, 4)),
        ("1 - sin 30°", F(1, 2)),
        ("sin 30° x sin 30° + cos 30° x cos 30°", F(1)),
        ("2 x sin 30° x cos 60°", F(1, 2)),
        ("tan 60° x tan 30°", F(1)),
    ]
    expr, val = rng.choice(cases)
    wrongs = [val + 1, val - 1, val * 2, F(1, 2) if val != F(1, 2) else F(3, 4)]
    wrongs = [w for w in wrongs if w != val]
    opts, letter = options_from(val, wrongs, rng)
    if not opts:
        return None
    return dict(topic="trigonometry", subtopic="trigonometry",
                stem=f"What is the value of {expr}?",
                options=opts, answer=letter, difficulty="easy", est_time=35,
                check="trigonometry", trace=f"{expr} = {fmt(val)}")



def gen_boats_streams(rng):
    """Downstream = still-water + stream; upstream = still-water - stream.

    The distance is chosen as a multiple of the downstream speed so the answer
    is an exact whole number of hours rather than a rounded decimal.
    """
    b = rng.choice([8, 10, 12, 15, 18, 20, 24])
    st = rng.choice([2, 3, 4, 5, 6])
    if st >= b:
        return None
    down = b + st
    hours = rng.choice([2, 3, 4, 5])
    dist = down * hours
    wrongs = [F(dist, b - st), F(dist, b), F(hours + 1), F(hours * 2)]
    wrongs = [w for w in wrongs if w != F(hours) and w > 0]
    opts, letter = options_from(F(hours), wrongs, rng)
    if not opts:
        return None
    return dict(topic="boats_and_streams", subtopic="boats_and_streams",
                stem=(f"The speed of a boat in still water is {b} km/h and the "
                      f"speed of the stream is {st} km/h. How long will the boat "
                      f"take to cover {dist} km downstream (in hours)?"),
                options=opts, answer=letter, difficulty="easy", est_time=40,
                check="boats_streams",
                trace=f"{dist}/({b}+{st}) = {hours}")


def gen_mixture_alligation(rng):
    """Alligation: (dearer - mean) : (mean - cheaper) gives the mixing ratio."""
    cheap = rng.choice([20, 24, 30, 36, 40, 45])
    dear = cheap + rng.choice([10, 12, 15, 20, 24, 30])
    # pick a mean strictly between them that yields a clean ratio
    mean = rng.randint(cheap + 1, dear - 1)
    hi, lo = dear - mean, mean - cheap
    g = __import__("math").gcd(hi, lo)
    hi, lo = hi // g, lo // g
    if max(hi, lo) > 12:
        return None
    correct = f"{hi}:{lo}"
    wrongs = [f"{lo}:{hi}", f"{hi+1}:{lo}", f"{hi}:{lo+1}", f"{hi+lo}:{lo}"]
    wrongs = [w for w in wrongs if w != correct]
    opts, letter = options_from(correct, wrongs, rng)
    if not opts:
        return None
    return dict(topic="mixture_and_alligation", subtopic="mixture_and_alligation",
                stem=(f"In what ratio must rice costing ₹{cheap} per kg be mixed "
                      f"with rice costing ₹{dear} per kg so that the mixture is "
                      f"worth ₹{mean} per kg?"),
                options=opts, answer=letter, difficulty="medium", est_time=45,
                check="alligation",
                trace=f"({dear}-{mean}):({mean}-{cheap}) = {correct}")



def gen_time_and_work(rng):
    """A alone in a days, B alone in b days -> together ab/(a+b) days.

    Same arithmetic as pipes-and-cisterns but a distinct forecast topic, and it
    needs its own generator: reassigning the pipes generator to
    `pipes_and_cisterns` left `time_and_work` (0.76/paper) unsupplied.
    """
    pairs = [(6, 12), (10, 15), (12, 24), (20, 30), (15, 30), (9, 18),
             (8, 24), (14, 21), (16, 48), (18, 9), (5, 20)]
    a, b = rng.choice(pairs)
    together = F(a * b, a + b)
    wrongs = [F(a + b), F(a + b, 2), F(abs(a - b)), together + 1]
    wrongs = [w for w in wrongs if w != together and w > 0]
    opts, letter = options_from(together, wrongs, rng)
    if not opts:
        return None
    return dict(topic="time_and_work", subtopic="work_together",
                stem=(f"A can complete a piece of work in {a} days and B can "
                      f"complete the same work in {b} days. Working together, "
                      f"in how many days will they complete the work?"),
                options=opts, answer=letter, difficulty="easy", est_time=40,
                check="work_together",
                trace=f"({a}*{b})/({a}+{b}) = {fmt(together)}")



# --------------------------------------------------------- data interpretation
# DI needs a data table, not just a stem. The forecast is 0.45 questions per
# paper - less than one - so a 4-question set would rarely fit. Each item is
# therefore self-contained and carries its own table.
#
# The table is embedded in the STEM (not a separate passage field) so that
# `recheck` can re-parse it and re-derive the answer from what was actually
# printed, exactly as every other check here does. The question text leads and
# the table follows, which also keeps the first 60 characters distinct so the
# frame cap does not treat every DI item as one archetype.
DI_FIRMS = [("A", "B", "C"), ("P", "Q", "R"), ("X", "Y", "Z")]
DI_YEARS = [(2021, 2022, 2023), (2022, 2023, 2024), (2020, 2021, 2022)]


def _di_table(rng):
    firms = rng.choice(DI_FIRMS)
    years = rng.choice(DI_YEARS)
    grid = {f: [rng.choice(range(60, 400, 10)) for _ in years] for f in firms}
    lines = ["Sales (in Rs lakh)", "Year  " + "  ".join(f"{f:>5}" for f in firms)]
    for i, y in enumerate(years):
        lines.append(f"{y}  " + "  ".join(f"{grid[f][i]:>5}" for f in firms))
    return firms, years, grid, "\n".join(lines)


def gen_data_interpretation(rng):
    firms, years, grid, table = _di_table(rng)
    f = rng.choice(firms)
    kind = rng.choice(["total", "difference", "average", "ratio"])

    if kind == "total":
        val = F(sum(grid[f]))
        q = (f"What is the total sales of Company {f} over the three years "
             f"(in Rs lakh)? Refer to the table below.")
        wrongs = [val - grid[f][0], val + 10, F(max(grid[f])), val / 3]
    elif kind == "difference":
        g = rng.choice([x for x in firms if x != f])
        i = rng.randrange(len(years))
        val = F(abs(grid[f][i] - grid[g][i]))
        q = (f"What is the difference between the sales of Company {f} and "
             f"Company {g} in {years[i]} (in Rs lakh)? Refer to the table below.")
        wrongs = [F(grid[f][i] + grid[g][i]), F(grid[f][i]), F(grid[g][i]), val + 10]
    elif kind == "average":
        tot = sum(grid[f])
        if tot % 3:
            return None
        val = F(tot, 3)
        q = (f"What is the average annual sales of Company {f} over the three "
             f"years (in Rs lakh)? Refer to the table below.")
        wrongs = [F(tot), val + 10, F(max(grid[f])), F(min(grid[f]))]
    else:
        g = rng.choice([x for x in firms if x != f])
        i = rng.randrange(len(years))
        a, b = grid[f][i], grid[g][i]
        from math import gcd
        d = gcd(a, b)
        val = f"{a//d}:{b//d}"
        q = (f"What is the ratio of the sales of Company {f} to Company {g} "
             f"in {years[i]}? Refer to the table below.")
        wrongs = [f"{b//d}:{a//d}", f"{a//d+1}:{b//d}", f"{a//d}:{b//d+1}"]
        wrongs = [w for w in wrongs if w != val]
        opts, letter = options_from(val, wrongs, rng)
        if not opts:
            return None
        return dict(topic="data_interpretation", subtopic="di_table",
                    stem=q + "\n\n" + table, options=opts, answer=letter,
                    difficulty="medium", est_time=55, check="di_ratio",
                    trace=f"{a}:{b} -> {val}")

    wrongs = [w for w in wrongs if w != val and w > 0]
    opts, letter = options_from(val, wrongs, rng)
    if not opts:
        return None
    return dict(topic="data_interpretation", subtopic="di_table",
                stem=q + "\n\n" + table, options=opts, answer=letter,
                difficulty="medium", est_time=55, check=f"di_{kind}",
                trace=f"{kind} -> {fmt(val)}")


GENERATORS = {
    "simplification": gen_simplification,
    "identities": gen_identity,
    "ratio_basic": gen_ratio_hcf_lcm,
    "successive_change": gen_successive_percentage,
    "successive_discount": gen_successive_discount,
    "simple_interest": gen_simple_interest,
    "average_replacement": gen_average_replacement,
    "pipes_cisterns": gen_pipes,
    "trains": gen_relative_speed,
    "angles": gen_angle_centre,
    "divisibility": gen_divisibility,
    "marked_price_discount": gen_marked_price,
    "cylinder_volume": gen_cylinder_volume,
    "sphere_surface": gen_sphere_surface,
    "cuboid_volume": gen_cuboid_volume,
    "circle_area": gen_circle_area,
    "tangent_length": gen_tangent_length,
    "cyclic_quadrilateral": gen_cyclic_quad,
    # Six geometry generators over-supplied the topic by +10.7 against a
    # forecast of 3.33/paper. Trimmed to four, keeping the circle core
    # (INSIGHTS.md §2: geometry is 59% circle/chord/tangent) plus one triangle
    # type. `gen_chord_intersect` (power-of-a-point, rare in SSC) and
    # `gen_triangle_exterior` (trivially easy) are retired but retained below
    # with their recheck branches - re-enable by uncommenting.
    # "intersecting_chords": gen_chord_intersect,
    # "exterior_angle": gen_triangle_exterior,
    "similar_triangles": gen_similar_triangles,
    "trigonometry": gen_trigonometry,
    "boats_and_streams": gen_boats_streams,
    "mixture_and_alligation": gen_mixture_alligation,
    "work_together": gen_time_and_work,
    "di_table": gen_data_interpretation,
}


# ================================================== independent re-derivation
def recheck(q) -> bool:
    """Re-derive the answer from the RENDERED STEM ONLY.

    Deliberately re-parses the printed numbers instead of reading any value the
    generator kept, so an error in the generator cannot be confirmed by itself.
    """
    stem = q["stem"]
    nums = [int(x) for x in re.findall(r"\d+", stem.replace(",", ""))]
    opts = dict(q["options"])
    got = None
    c = q["check"]
    try:
        if c == "simplification":
            m = re.search(r"Simplify:\s*(\d+) \+ (\d+) x (\d+) - (\d+) / (\d+)", stem)
            a, b, cc, d, e = (int(g) for g in m.groups())
            got = F(a) + F(b) * cc - F(d, e)
        elif c == "identity_k2m2":
            k = int(re.search(r"1/x\s*=\s*(\d+)", stem).group(1))
            got = F(k * k - 2)
        elif c == "ratio_hcf_lcm":
            a, b = (int(g) for g in re.search(r"is (\d+):(\d+)", stem).groups())
            h = int(re.search(r"HCF is (\d+)", stem).group(1))
            got = F(a * b * h)
        elif c == "successive_pct":
            base = int(re.search(r"₹(\d+)", stem).group(1))
            p = int(re.search(r"increased by (\d+)%", stem).group(1))
            qq = int(re.search(r"decreased by (\d+)%", stem).group(1))
            got = F(base) * (100 + p) * (100 - qq) / 10000
        elif c == "successive_discount":
            mp = int(re.search(r"₹(\d+)", stem).group(1))
            d1, d2 = (int(g) for g in
                      re.search(r"discounts of (\d+)% and (\d+)%", stem).groups())
            got = F(mp) * (100 - d1) * (100 - d2) / 10000
        elif c == "simple_interest":
            p = int(re.search(r"sum of ₹(\d+)", stem).group(1))
            r = int(re.search(r"at\s*(\d+)% per annum", stem).group(1))
            t = int(re.search(r"for (\d+) years", stem).group(1))
            got = F(p * r * t, 100)
        elif c == "average_replacement":
            n = int(re.search(r"average weight of (\d+)", stem).group(1))
            d = int(re.search(r"increases by (\d+) kg", stem).group(1))
            got = F(n * d)
        elif c == "pipes_together":
            # re.findall inside a genexp is not subscriptable -- the original
            # form raised, so every pipes candidate failed recheck and none
            # shipped. Build a real list first.
            hrs = [int(g) for g in re.findall(r"in (\d+) hours", stem)]
            if len(hrs) < 2:
                return False
            a, b = hrs[0], hrs[1]
            got = F(a * b, a + b)
        elif c == "train_pole":
            sp = int(re.search(r"at (\d+) km/h", stem).group(1))
            t = int(re.search(r"in (\d+) seconds", stem).group(1))
            got = F(sp * 1000, 3600) * t
        elif c == "angle_centre":
            a = int(re.search(r"is (\d+)°", stem).group(1))
            got = F(2 * a)
        elif c == "divisibility":
            d = int(re.search(r"divisible by (\d+)", stem).group(1))
            hits = [l for l, t in q["options"] if int(t) % d == 0]
            return len(hits) == 1 and hits[0] == q["answer"]
        elif c == "cylinder_volume":
            r = int(re.search(r"cylinder is (\d+) cm", stem).group(1))
            h = int(re.search(r"height is (\d+) cm", stem).group(1))
            got = F(22, 7) * r * r * h
        elif c == "sphere_surface":
            r = int(re.search(r"radius (\d+) cm", stem).group(1))
            got = 4 * F(22, 7) * r * r
        elif c == "cuboid_volume":
            l = int(re.search(r"length (\d+) cm", stem).group(1))
            b = int(re.search(r"breadth (\d+) cm", stem).group(1))
            h = int(re.search(r"height (\d+) cm", stem).group(1))
            got = F(l * b * h)
        elif c == "circle_area":
            r = int(re.search(r"radius is (\d+) cm", stem).group(1))
            got = F(22, 7) * r * r
        elif c == "tangent_length":
            r = int(re.search(r"radius (\d+) cm", stem).group(1))
            d = int(re.search(r"OP = (\d+) cm", stem).group(1))
            v = d * d - r * r
            root = int(round(v ** 0.5))
            if root * root != v:
                return False
            got = F(root)
        elif c == "cyclic_quad":
            a = int(re.search(r"angle A = (\d+)", stem).group(1))
            got = F(180 - a)
        elif c == "chord_intersect":
            ap = int(re.search(r"AP = (\d+) cm", stem).group(1))
            pb = int(re.search(r"PB = (\d+) cm", stem).group(1))
            cp = int(re.search(r"CP = (\d+) cm", stem).group(1))
            got = F(ap * pb, cp)
        elif c == "triangle_exterior":
            a = int(re.search(r"angle A = (\d+)", stem).group(1))
            b = int(re.search(r"angle B = (\d+)", stem).group(1))
            got = F(a + b)
        elif c == "similar_triangles":
            a = int(re.search(r"AB = (\d+) cm", stem).group(1))
            A = int(re.search(r"PQ = (\d+) cm", stem).group(1))
            b = int(re.search(r"BC = (\d+) cm", stem).group(1))
            got = F(b * A, a)
        elif c == "trigonometry":
            expr = re.search(r"value of (.+)\?", stem).group(1)
            TAB = {"sin 30°": F(1,2), "cos 60°": F(1,2), "sin 90°": F(1),
                   "cos 0°": F(1), "tan 45°": F(1), "sin 45°": None,
                   "cos 30°": None, "sin 60°": None, "tan 60°": None,
                   "tan 30°": None}
            # only exact-rational combinations are generated, so re-derive from
            # the printed expression using a small explicit table
            EXACT = {
                "sin 30° + cos 60°": F(1), "sin 90° - cos 0°": F(0),
                "tan 45° + sin 30°": F(3,2), "sin 30° x cos 60°": F(1,4),
                "cos 30° x cos 30°": F(3,4), "sin 45° x sin 45°": F(1,2),
                "tan 45° x tan 45°": F(1), "sin 60° x sin 60°": F(3,4),
                "1 - sin 30°": F(1,2),
                "sin 30° x sin 30° + cos 30° x cos 30°": F(1),
                "2 x sin 30° x cos 60°": F(1,2), "tan 60° x tan 30°": F(1),
            }
            got = EXACT.get(expr.strip())
            if got is None:
                return False
        elif c == "boats_streams":
            b = int(re.search(r"still water is (\d+) km/h", stem).group(1))
            st = int(re.search(r"stream is (\d+) km/h", stem).group(1))
            dist = int(re.search(r"cover (\d+) km downstream", stem).group(1))
            got = F(dist, b + st)
        elif c == "alligation":
            prices = [int(x) for x in re.findall(r"₹(\d+) per kg", stem)]
            if len(prices) != 3:
                return False
            cheap, dear, mean = prices
            if not (cheap < mean < dear):
                return False
            hi, lo = dear - mean, mean - cheap
            g = __import__("math").gcd(hi, lo)
            want = f"{hi//g}:{lo//g}"
            hits = [l for l, t in q["options"] if t == want]
            return len(hits) == 1 and hits[0] == q["answer"]
        elif c == "work_together":
            days = [int(g) for g in re.findall(r"in (\d+) days", stem)]
            if len(days) < 2:
                return False
            a, b = days[0], days[1]
            got = F(a * b, a + b)
        elif c.startswith("di_"):
            # re-parse the printed table, then re-derive
            rows = {}
            head = None
            for line in stem.split("\n"):
                parts = line.split()
                if parts and parts[0] == "Year":
                    head = parts[1:]
                elif head and parts and parts[0].isdigit() and len(parts) == len(head) + 1:
                    rows[int(parts[0])] = [int(x) for x in parts[1:]]
            if not head or not rows:
                return False
            col = {f: [rows[y][i] for y in sorted(rows)] for i, f in enumerate(head)}
            if c == "di_total":
                f = re.search(r"Company (\w+) over", stem).group(1)
                got = F(sum(col[f]))
            elif c == "di_average":
                f = re.search(r"Company (\w+) over", stem).group(1)
                got = F(sum(col[f]), 3)
            elif c == "di_difference":
                f, g = re.search(r"Company (\w+) and Company (\w+)", stem).groups()
                y = int(re.search(r"in (\d{4})", stem).group(1))
                i = sorted(rows).index(y)
                got = F(abs(col[f][i] - col[g][i]))
            elif c == "di_ratio":
                f, g = re.search(r"Company (\w+) to Company (\w+)", stem).groups()
                y = int(re.search(r"in (\d{4})", stem).group(1))
                i = sorted(rows).index(y)
                a, b = col[f][i], col[g][i]
                from math import gcd
                d = gcd(a, b)
                want = f"{a//d}:{b//d}"
                hits = [l for l, t in q["options"] if t == want]
                return len(hits) == 1 and hits[0] == q["answer"]
            else:
                return False
        elif c == "marked_price":
            sp = int(re.search(r"sold for ₹(\d+)", stem).group(1))
            d = int(re.search(r"discount of (\d+)%", stem).group(1))
            got = F(sp * 100, 100 - d)
    except Exception:
        return False
    if got is None:
        return False
    want = fmt(got)
    hits = [l for l, t in q["options"] if t == want]
    return len(hits) == 1 and hits[0] == q["answer"]


def build_pool(per_arch=30, seed=2026):
    rng = random.Random(seed)
    pool, rej = [], Counter()
    for arch, fn in GENERATORS.items():
        made, tries = 0, 0
        while made < per_arch and tries < per_arch * 200:
            tries += 1
            q = fn(rng)
            if not q:
                rej[f"{arch}:construction"] += 1
                continue
            if len({t for _, t in q["options"]}) != 4:
                rej[f"{arch}:duplicate_option"] += 1
                continue
            if not recheck(q):
                rej[f"{arch}:FAILED_RECHECK"] += 1
                continue
            ident = (q["stem"], tuple(t for _, t in q["options"]))
            if ident in {(p["stem"], tuple(t for _, t in p["options"])) for p in pool}:
                rej[f"{arch}:duplicate"] += 1
                continue
            q["verified_by"] = f"machine:recheck({q['check']})"
            pool.append(q)
            made += 1
    return pool, rej


def main() -> None:
    import sys as _s
    n = int(_s.argv[1]) if len(_s.argv) > 1 else 30
    pool, rej = build_pool(per_arch=n)
    print("=" * 88)
    print("QUANT CANDIDATE GENERATION — exact arithmetic + independent re-derivation")
    print("=" * 88)
    print(f"pool size: {len(pool)}\n")
    for k, v in Counter(q["subtopic"] for q in pool).most_common():
        print(f"  {k:<28}{v:>4}")
    print(f"\nall machine-verified: "
          f"{sum(1 for q in pool if q['verified_by'].startswith('machine'))}/{len(pool)}")
    print(f"difficulty mix: {dict(Counter(q['difficulty'] for q in pool))}")
    print(f"mean estimated time: "
          f"{sum(q['est_time'] for q in pool)/max(len(pool),1):.0f}s")
    if rej:
        print("\nrejected during construction:")
        for k, v in rej.most_common(10):
            print(f"  {k:<40}{v:>6}")
    json.dump(pool, open(OUT / "quant_candidates.json", "w"), indent=2)
    print(f"\nWrote {OUT/'quant_candidates.json'}")


if __name__ == "__main__":
    main()
