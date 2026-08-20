"""Candidate interview recollections — quarantined evidence, by construction.

WHY THIS FILE IS SO DEFENSIVE
-----------------------------
UPSC publishes nothing about what happens inside a Personality Test. What
circulates is candidate recollection posted to forums and coaching sites, and
it carries three defects that cannot be corrected after the fact:

    self-selection   people who cleared post; people who did not, mostly do not
    unverifiable     no official record exists to check any of it against
    post-hoc recall  written AFTER the result is known, so the memory is shaped
                     by the outcome it is being used to explain

That is the same defect class as the Tier-II contamination in HANDOVER.md §6,
which silently poisoned every analysis that ran before it was caught. The owner
asked for this store anyway, with the tradeoff visible. So it is built - and
built so it cannot leak into anything that claims to be measured.

THE QUARANTINE, ENFORCED IN CODE
--------------------------------
1. Separate store. Never the ledger, never `data/world/incoming.json`. Nothing
   here reaches `world_evidence.build()`.
2. `provenance` is MANDATORY on every record. A write without it raises.
3. `BASIS_CEILING = "WEAK"` is absolute. `effective_basis()` never returns more,
   whatever the volume - a million recollections are still recollections.
4. `assert_not_confirmable()` raises if anything tries to use this to support a
   CONFIRMED claim. Volume must never be mistaken for evidence.

WHAT IT IS LEGITIMATELY GOOD FOR
--------------------------------
Not "what the board asks" - the sample cannot support that. What it CAN do,
honestly, is describe the shape of the encounter as reported: how long, how
many questions, whether the DAF drove them, which broad areas recur. Treat
every output as "what candidates say happened", never "what happens".

    python src/interview_transcripts.py --stats
    python src/interview_transcripts.py --add-example
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

STORE = ROOT / "data" / "world" / "interview" / "transcripts.jsonl"

BASIS_CEILING = "WEAK"          # absolute. Not a default - a ceiling.
PROVENANCE_TAG = "UNVERIFIED_CANDIDATE_RECOLLECTION"

SOURCE_KINDS = ("forum", "coaching_site", "blog", "youtube", "book",
                "personal_communication", "other")


class QuarantineViolation(Exception):
    """Raised when quarantined data is used as if it were measured."""


@dataclass(frozen=True)
class Transcript:
    year: int
    board_chair: str | None          # Commission member, if the candidate named one
    source_kind: str
    source_ref: str                  # url or citation - must be traceable
    provenance: str = PROVENANCE_TAG
    outcome_known: bool = True       # recalled after the result: almost always True
    reported_marks: int | None = None
    duration_min: int | None = None
    daf_driven: bool | None = None   # did questions follow the application form
    areas: tuple[str, ...] = field(default=())
    notes: str = ""
    added: str = ""

    def __post_init__(self):
        if self.provenance != PROVENANCE_TAG:
            raise QuarantineViolation(
                f"provenance must be {PROVENANCE_TAG!r}; this store holds "
                f"nothing else. Verified material belongs elsewhere.")
        if not (self.source_ref or "").strip():
            raise QuarantineViolation(
                "source_ref is mandatory - an untraceable recollection cannot "
                "even be audited later, which makes it worthless here.")
        if self.source_kind not in SOURCE_KINDS:
            raise QuarantineViolation(f"unknown source_kind: {self.source_kind!r}")


def add(t: Transcript) -> None:
    STORE.parent.mkdir(parents=True, exist_ok=True)
    d = asdict(t)
    d["added"] = datetime.now().isoformat(timespec="seconds")
    with STORE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(d, ensure_ascii=False) + "\n")


def load() -> list[dict]:
    if not STORE.exists():
        return []
    out = []
    for line in STORE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def effective_basis(_rows: list[dict] | None = None) -> str:
    """Always WEAK. Deliberately ignores volume.

    A larger pile of self-selected, post-hoc recollections is a larger pile of
    self-selected, post-hoc recollections. Sample size does not repair a biased
    sampling frame, and pretending otherwise is how this store would poison the
    rest of the system.
    """
    return BASIS_CEILING


def assert_not_confirmable(purpose: str = "") -> None:
    raise QuarantineViolation(
        f"REFUSED: quarantined interview recollections may not support a "
        f"CONFIRMED claim{' (' + purpose + ')' if purpose else ''}. "
        f"Basis is capped at {BASIS_CEILING}. Use them to describe what "
        f"candidates REPORT, never to assert what boards DO.")


def stats() -> dict:
    rows = load()
    if not rows:
        return {"records": 0, "basis": BASIS_CEILING,
                "note": "store is empty - nothing has been added"}
    yrs, kinds, chairs = {}, {}, {}
    for r in rows:
        yrs[r.get("year")] = yrs.get(r.get("year"), 0) + 1
        kinds[r.get("source_kind")] = kinds.get(r.get("source_kind"), 0) + 1
        if r.get("board_chair"):
            chairs[r["board_chair"]] = chairs.get(r["board_chair"], 0) + 1
    with_marks = sum(1 for r in rows if r.get("reported_marks") is not None)
    return {
        "records": len(rows),
        "basis": effective_basis(rows),
        "by_year": dict(sorted(yrs.items())),
        "by_source_kind": kinds,
        "named_chairs": chairs,
        "with_reported_marks": with_marks,
        "selection_bias_warning": (
            f"{with_marks}/{len(rows)} report marks. Candidates who cleared "
            f"publish far more often than those who did not, so ANY summary of "
            f"this store describes successful candidates' recollections - not "
            f"the interview, and not the population."),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        print("quarantine self-test\n")
        try:
            Transcript(year=2024, board_chair=None, source_kind="forum",
                       source_ref="x", provenance="VERIFIED")
            print("  FAIL  forged provenance was accepted")
        except QuarantineViolation as e:
            print(f"  PASS  forged provenance refused")
        try:
            Transcript(year=2024, board_chair=None, source_kind="forum",
                       source_ref="")
            print("  FAIL  untraceable record accepted")
        except QuarantineViolation as e:
            print(f"  PASS  untraceable record refused")
        try:
            assert_not_confirmable("topic frequency")
            print("  FAIL  confirmable use allowed")
        except QuarantineViolation:
            print(f"  PASS  CONFIRMED use refused")
        print(f"  PASS  effective_basis() = {effective_basis()} (ignores volume)")
    else:
        print(json.dumps(stats(), indent=2))
