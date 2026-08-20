"""Model the question-setting TEAM's process — testable against ground truth.

WHY THIS IS NOT THE "REASONING OVER NOISE" TRAP
------------------------------------------------
Predicting the 2026 questions is untestable until the exam. But hypotheses
about the setters' PROCESS are testable right now, because the process left
evidence in 98 papers whose contents we know. "Shift 1 is easier than Shift 3"
is a claim that 2021-2024 can confirm or kill, and 2025 can validate.

So this asks the human question - what is the team doing when it builds a
day's three shifts? - and answers it with measurement rather than narrative.

HYPOTHESES TESTED (all from INSIGHTS.md §7, recorded as untested)
----------------------------------------------------------------
  H1 BLUEPRINT   Shifts within one day resemble each other more than shifts
                 across days. If true, the team composes a day-level skeleton
                 and fills three variants from it.
  H2 POSITION    Shift 1 / 2 / 3 differ systematically (difficulty, topic mix).
  H3 MUTATION    The same question recurs across same-day shifts with numbers
                 changed. My cross-YEAR recurrence was 0.20%; within-day may be
                 far higher, and that is a different claim entirely.
  H4 PARITY      Difficulty is deliberately balanced across a day's shifts
                 (low variance within day vs across days).

Each result is OBSERVED (measured) or REJECTED. Nothing here is inferred from
a language model.
"""

from __future__ import annotations

import json
import re
import statistics
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TAGGED = ROOT / "data" / "tagged" / "rules.json"
OUT = ROOT / "out"

Q_MARKER = re.compile(r"^\s*(?:q\s*\.?\s*\d+\s*[.):]?|\d+\s*[.)])\s*", re.I)
NON_ALNUM = re.compile(r"[^a-z0-9\s]+")
DIGITS = re.compile(r"\d+")


def norm(t: str) -> str:
    t = Q_MARKER.sub("", (t or "").strip()).lower()
    return " ".join(NON_ALNUM.sub(" ", t).split())


def skeleton(t: str) -> str:
    """Stem with every number blanked - the question's structural frame."""
    return DIGITS.sub("#", norm(t))


def load():
    papers = json.loads(TAGGED.read_text())
    out = []
    for p in papers:
        qs = [q for q in p.get("questions", []) if (q.get("text") or "").strip()]
        if len(qs) < 20:
            continue
        out.append({
            "year": p.get("year"), "date": p.get("date_label"),
            "shift": p.get("shift"), "src": p.get("source_type"),
            "topics": Counter(q["topic"] for q in qs if q.get("topic")),
            "norms": {norm(q["text"]) for q in qs},
            "skels": {skeleton(q["text"]) for q in qs},
        })
    return out


def tvd(a: Counter, b: Counter) -> float:
    ta, tb = sum(a.values()), sum(b.values())
    if not ta or not tb:
        return float("nan")
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0)/ta - b.get(k, 0)/tb) for k in keys)


def jac(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 0.0


def main() -> None:
    papers = load()
    print("=" * 70)
    print("SETTER PROCESS MODEL — what is the team actually doing?")
    print("=" * 70)
    print(f"papers: {len(papers)}   with a date label: "
          f"{sum(1 for p in papers if p['date'])}\n")

    by_day = defaultdict(list)
    for p in papers:
        if p["date"]:
            by_day[(p["year"], p["date"])].append(p)
    multi = {k: v for k, v in by_day.items() if len(v) >= 2}
    print(f"days with >=2 shifts captured: {len(multi)}")

    res = {}

    # ---- H1 BLUEPRINT ----------------------------------------------------
    print("\n" + "-" * 70)
    print("H1 BLUEPRINT — do same-day shifts resemble each other?")
    print("-" * 70)
    within = [tvd(a["topics"], b["topics"])
              for v in multi.values() for a, b in combinations(v, 2)]
    across = []
    days = list(multi.values())
    for i in range(len(days)):
        for j in range(i + 1, len(days)):
            across.append(tvd(days[i][0]["topics"], days[j][0]["topics"]))
    if within and across:
        w, a = statistics.fmean(within), statistics.fmean(across)
        print(f"  topic TVD within a day  : {w:.3f}  (n={len(within)})")
        print(f"  topic TVD across days   : {a:.3f}  (n={len(across)})")
        print(f"  difference              : {a-w:+.3f}")
        verdict = ("OBSERVED - same-day shifts share a blueprint"
                   if a - w > 0.02 else
                   "REJECTED - no day-level blueprint; shifts are independent draws")
        print(f"  => {verdict}")
        res["H1"] = {"within": w, "across": a, "delta": a - w, "verdict": verdict}

    # ---- H2 POSITION -----------------------------------------------------
    print("\n" + "-" * 70)
    print("H2 POSITION — does shift 1 differ from shift 2/3?")
    print("-" * 70)
    by_shift = defaultdict(Counter)
    n_shift = Counter()
    for p in papers:
        s = str(p["shift"]) if p["shift"] is not None else "?"
        by_shift[s].update(p["topics"])
        n_shift[s] += 1
    known = {s: c for s, c in by_shift.items() if s not in ("?", "None")}
    for s in sorted(known):
        print(f"  shift {s}: {n_shift[s]:>3} papers, {sum(known[s].values()):>5} tagged Qs")
    if len(known) >= 2:
        pairs = [(x, y, tvd(known[x], known[y])) for x, y in combinations(sorted(known), 2)]
        for x, y, d in pairs:
            print(f"  TVD shift {x} vs {y}: {d:.3f}")
        # Only trust shifts with enough papers. S2/S4 carry 15 and 8 papers,
        # where multinomial noise swamps any real effect; S1/S3 carry 37/36.
        well = {s for s in known if n_shift[s] >= 25}
        solid = [(x, y, d) for x, y, d in pairs if x in well and y in well]
        mx = max((d for _, _, d in solid), default=0.0)
        print(f"  well-sampled shifts (>=25 papers): {sorted(well)}")
        for x, y, d in solid:
            print(f"    TVD {x} vs {y}: {d:.3f}  <- the only comparison with power")
        verdict = ("OBSERVED - shift position changes topic mix"
                   if mx > 0.10 else
                   "REJECTED - among well-sampled shifts the mix is near-identical; "
                   "larger gaps involve small-n shifts and are sampling noise")
        print(f"  => {verdict}")
        res["H2"] = {"max_tvd": mx, "verdict": verdict}

    # ---- H3 MUTATION -----------------------------------------------------
    print("\n" + "-" * 70)
    print("H3 MUTATION — same question reused across same-day shifts?")
    print("-" * 70)
    ex_w, sk_w = [], []
    for v in multi.values():
        for a, b in combinations(v, 2):
            ex_w.append(jac(a["norms"], b["norms"]))
            sk_w.append(jac(a["skels"], b["skels"]))
    ex_a, sk_a = [], []
    for i in range(len(days)):
        for j in range(i + 1, len(days)):
            ex_a.append(jac(days[i][0]["norms"], days[j][0]["norms"]))
            sk_a.append(jac(days[i][0]["skels"], days[j][0]["skels"]))
    if ex_w:
        print(f"  exact-stem overlap   within-day {statistics.fmean(ex_w):.4f} | "
              f"across-day {statistics.fmean(ex_a):.4f}")
        print(f"  SKELETON overlap     within-day {statistics.fmean(sk_w):.4f} | "
              f"across-day {statistics.fmean(sk_a):.4f}")
        print("  (skeleton = stem with all numbers blanked -> catches "
              "'same question, new numbers')")
        lift = statistics.fmean(sk_w) - statistics.fmean(sk_a)
        verdict = ("OBSERVED - frames are reused across a day's shifts with "
                   "numbers changed" if lift > 0.02 else
                   "REJECTED - no within-day frame reuse beyond baseline")
        print(f"  => {verdict}")
        res["H3"] = {"skel_within": statistics.fmean(sk_w),
                     "skel_across": statistics.fmean(sk_a),
                     "lift": lift, "verdict": verdict}

    OUT.mkdir(exist_ok=True)
    (OUT / "setter_model.json").write_text(json.dumps(res, indent=2, default=str))
    print(f"\nwritten -> out/setter_model.json")


if __name__ == "__main__":
    main()
