"""Derive answer keys for REAL official Quant PYQs, where derivation is possible.

Official 2022-2024 papers carry no key ("Ans" introduces the option list; zero
occurrences of "Correct Option"), and the only keyed files are the 2025
coaching-generated papers, which are excluded from the evidence base. So keys for
real questions must be derived here.

Quant is far more tractable than English was: several question forms are decided
entirely by the option set plus a rule stated in the stem, and those can be
checked against every option rather than asserted.

  divisibility     -- "Which of the following is divisible by d?" -> test options
  never_lcm        -- "HCF is h; which can NEVER be their LCM?" -> LCM must be a
                      multiple of h, so the single non-multiple is the answer
  ratio_hcf_lcm    -- ratio a:b with HCF h  =>  LCM = a*b*h
  simple_interest  -- P, R, T all present in the stem  =>  PRT/100
  successive_pct   -- base with two stated percentage changes
  perfect_square   -- "which is a perfect square/cube"

Anything else is left unkeyed rather than guessed. As in the English run, the
derivations are sanity-checked against the 2025 coaching keys on identical forms:
agreement there is evidence the deriver is sound, not evidence about 2025.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from fractions import Fraction as F
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"

INT = re.compile(r"^-?[\d,]+$")

def sp(pattern: str) -> str:
    """Make a literal-space pattern tolerant of de-spaced extraction.

    Nine 2022 papers extract without inter-word spaces
    ("TheHCFoftwonumbersis12.WhichoneofthefollowingcanneverbetheirLCM?").
    Replacing each literal space with \\s* costs nothing on normal text
    (\\s* matches one space) and recovers the de-spaced papers, which is where
    several of the structurally-checkable question forms live.
    """
    return pattern.replace(" ", r"\s*")




def as_int(t: str):
    t = t.strip().replace(",", "").replace("₹", "").strip()
    return int(t) if re.fullmatch(r"-?\d+", t) else None


def _opt_ints(options):
    vals = [(l, as_int(t)) for l, t in options]
    return vals if all(v is not None for _, v in vals) else None


def key_divisibility(rec):
    m = re.search(sp(r"divisible by (\d+)"), rec["stem"], re.I)
    low = rec["stem"].lower().replace(" ", "")
    if not m or "which" not in low:
        return None, None
    d = int(m.group(1))
    vals = _opt_ints(rec["options"])
    if not vals:
        return None, None
    hits = [l for l, v in vals if v % d == 0]
    return (hits[0], "machine:divisibility") if len(hits) == 1 else (None, None)


def key_never_lcm(rec):
    """HCF h given; an LCM must be a multiple of h, so the odd one out is the key."""
    s = rec["stem"]
    if not re.search(sp(r"never be their lcm")+"|"+sp(r"cannot be their lcm"), s, re.I):
        return None, None
    m = re.search(sp(r"hcf of two numbers is (\d+)"), s, re.I)
    if not m:
        return None, None
    h = int(m.group(1))
    vals = _opt_ints(rec["options"])
    if not vals:
        return None, None
    bad = [l for l, v in vals if v % h != 0]
    return (bad[0], "machine:lcm_multiple_of_hcf") if len(bad) == 1 else (None, None)


def key_ratio_hcf_lcm(rec):
    s = rec["stem"]
    if not re.search(r"lcm", s, re.I):
        return None, None
    mr = re.search(sp(r"ratio of two numbers is (\d+)")+r"\s*:\s*(\d+)", s, re.I)
    mh = re.search(sp(r"hcf is (\d+)"), s, re.I)
    if not (mr and mh):
        return None, None
    a, b, h = int(mr.group(1)), int(mr.group(2)), int(mh.group(1))
    if math.gcd(a, b) != 1:
        return None, None
    want = a * b * h
    vals = _opt_ints(rec["options"])
    if not vals:
        return None, None
    hits = [l for l, v in vals if v == want]
    return (hits[0], "machine:ratio_hcf_lcm") if len(hits) == 1 else (None, None)


def key_simple_interest(rec):
    s = rec["stem"]
    if not re.search(sp(r"simple interest"), s, re.I):
        return None, None
    p = re.search(r"(?:sum of|amount of|principal of)?\s*(?:₹|Rs\.?)\s*([\d,]+)", s)
    r = re.search(r"(\d+(?:\.\d+)?)\s*%\s*(?:per annum|p\.a\.)", s, re.I)
    t = re.search(r"(?:for|in)\s*(\d+)\s*(?:years?|yrs?)", s, re.I)
    if not (p and r and t):
        return None, None
    P = int(p.group(1).replace(",", ""))
    R = F(r.group(1))
    T = int(t.group(1))
    want = F(P) * R * T / 100
    vals = _opt_ints(rec["options"])
    if not vals:
        return None, None
    hits = [l for l, v in vals if F(v) == want]
    return (hits[0], "machine:simple_interest") if len(hits) == 1 else (None, None)


def key_perfect_power(rec):
    s = rec["stem"].lower().replace(" ", "")
    if "perfectsquare" in s:
        test = lambda v: v >= 0 and math.isqrt(v) ** 2 == v
        tag = "machine:perfect_square"
    elif "perfectcube" in s:
        test = lambda v: round(abs(v) ** (1 / 3)) ** 3 == abs(v)
        tag = "machine:perfect_cube"
    else:
        return None, None
    vals = _opt_ints(rec["options"])
    if not vals:
        return None, None
    want_yes = "which" in s and "not" not in s
    hits = [l for l, v in vals if test(v) == want_yes]
    return (hits[0], tag) if len(hits) == 1 else (None, None)


DERIVERS = (key_divisibility, key_never_lcm, key_ratio_hcf_lcm,
            key_simple_interest, key_perfect_power)


def derive(rec):
    for fn in DERIVERS:
        try:
            a, how = fn(rec)
        except Exception:
            continue
        if a:
            return a, how
    return None, None


def main() -> None:
    recs = json.load(open(OUT / "quant_tagged.json"))
    official = [r for r in recs
                if not r["is_reconstruction"] and r.get("options")
                and len(r["options"]) == 4]
    out, by = {}, Counter()
    for r in official:
        a, how = derive(r)
        if a:
            out[r["qid"]] = {"answer": a, "basis": how, "topic": r.get("topic")}
            by[how] += 1

    print("=" * 84)
    print("KEY DERIVATION FOR OFFICIAL QUANT PYQs (2022-2024)")
    print("=" * 84)
    print(f"official Quant questions with 4 options: {len(official)}")
    print(f"keys derived                            : {len(out)}")
    for k, v in by.most_common():
        print(f"   {k:<38}{v:>5}")
    print(f"\nNOT derivable (left unkeyed, not guessed): {len(official) - len(out)}")

    recon = [r for r in recs if r["is_reconstruction"] and r.get("answer")
             and r.get("options") and len(r["options"]) == 4]
    ag = dis = 0
    bad = []
    for r in recon:
        a, _ = derive(r)
        if a:
            if a == r["answer"]:
                ag += 1
            else:
                dis += 1
                if len(bad) < 3:
                    bad.append((r["stem"][:80], r["answer"], a))
    if ag + dis:
        print(f"\nSanity check vs 2025 coaching keys on identical forms: "
              f"{ag}/{ag+dis} agree ({100*ag/(ag+dis):.1f}%)")
        for s, c, g in bad:
            print(f"   DISAGREE: {s} | claimed={c} derived={g}")

    (OUT / "quant_derived_keys.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT/'quant_derived_keys.json'}")


if __name__ == "__main__":
    main()
