"""The actor registry - every mind the exam passes through, and what it emits.

WHY A REGISTRY AND NOT A NARRATIVE
----------------------------------
The goal is a model of the collective mind that produces a UPSC result: the
setting teams, the evaluators, the interview board, and the governmental and
international actors upstream of all of them. That ambition is legitimate, and
it is also the single most likely place for this system to start generating
confident nonsense.

cognition.py already fixed the rule that prevents it:

    "Where a construct has no measurable signature here, it is omitted rather
     than asserted."

So every actor below declares what it EMITS and what evidence stream, if any,
carries that emission. An actor with no observable emission is not deleted - it
is registered with evidence_basis NONE and is FORBIDDEN from shaping output. It
stays visible as a known gap rather than being quietly modelled anyway.

THE TWO AXES
------------
Upstream cascade - who influences the exam before it is written:

    international -> agenda -> subnational -> instrument -> discourse

Stage pipeline - who acts inside the exam itself:

    prelims_setters -> mains_setters -> mains_evaluators -> interview_board

The evaluators are listed as a distinct mind on purpose. What gets REWARDED
shapes an exam as much as what gets ASKED, and the marking scheme is a separate
decision by separate people.

THE OBSERVABILITY GRADIENT
--------------------------
This is the load-bearing honesty in the file. The stages are not equally
knowable, and pretending otherwise is how exam "analysis" becomes astrology:

    prelims_setters    STRONG   public paper + official key, many years
    mains_setters      PARTIAL  questions public, marking scheme is not
    mains_evaluators   NONE     no public trace of how marks are assigned
    interview_board    WEAK     board composition is public; nothing else is

A claim about an actor may not exceed that actor's evidence basis. The rule is
enforced in code by `can_support_claim`, not left to whoever writes the next
hypothesis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Evidence strength, ordered. An actor cannot support a claim requiring more
# than it has.
BASIS_RANK = {"NONE": 0, "WEAK": 1, "PARTIAL": 2, "STRONG": 3}


@dataclass(frozen=True)
class Actor:
    name: str
    axis: str                 # "cascade" | "stage"
    emits: tuple[str, ...]    # observable outputs, if any
    evidence_basis: str       # NONE | WEAK | PARTIAL | STRONG
    stream: str | None        # which evidence stream carries it
    note: str


CASCADE: tuple[Actor, ...] = (
    Actor("international", "cascade",
          ("treaties", "index rankings", "climate commitments", "sanctions"),
          "PARTIAL", "world_ledger",
          "Upstream of domestic agenda. Heavily represented in UPSC IR and "
          "environment. Observed only through Indian press, so coverage is "
          "filtered by what Indian editors chose to carry."),
    Actor("agenda", "cascade",
          ("Budget", "Economic Survey", "NITI reports", "Cabinet decisions"),
          "PARTIAL", "world_ledger",
          "Sets union direction. Primary documents are public but are currently "
          "reaching the ledger via newspapers rather than being read directly."),
    Actor("subnational", "cascade",
          ("state schemes", "assembly acts", "district orders"),
          "WEAK", "world_ledger",
          "State-level emission. Kept separate from agenda because a Chief "
          "Minister's scheme is not union policy direction."),
    Actor("instrument", "cascade",
          ("PIB releases", "gazette notifications", "judgments", "reports"),
          "PARTIAL", "world_ledger",
          "Converts direction into acts. Largest observed layer."),
    Actor("discourse", "cascade",
          ("editorials", "commentary", "op-eds"),
          "PARTIAL", "world_ledger",
          "Argues about the rest. Closest to what a setter actually reads, and "
          "the hardest to separate from the channel carrying it."),
)

STAGES: tuple[Actor, ...] = (
    Actor("prelims_setters", "stage",
          ("the Prelims paper", "official answer key"),
          "STRONG", "evidence.py",
          "The only stage with a corpus large enough for pooled estimation - "
          "and even then UPSC gives ~1 paper/year against SSC CGL's 156."),
    Actor("mains_setters", "stage",
          ("the Mains question papers",),
          "PARTIAL", "evidence.py",
          "Questions are public; what a good answer looks like is not. Claims "
          "about difficulty or intent are unsupported at this basis."),
    Actor("mains_evaluators", "stage",
          (),
          "NONE", None,
          "No public trace of how marks are assigned. Registered so the gap is "
          "explicit. FORBIDDEN from shaping output until an evidence stream "
          "exists - candidate-reported marks are self-selected and would need "
          "their own bias test before counting."),
    Actor("interview_board", "stage",
          ("board composition",),
          "WEAK", "data/world/interview/board_composition.json",
          "Now has a stream: UPSC's own published Commission composition "
          "(names, tenure dates), so the CHAIR POOL active on any date is "
          "derivable. Basis stays WEAK and that is not pedantry - a Personality "
          "Test board seats ~4 advisors besides the Commission chair, and those "
          "are never published. Composition is also not conduct: knowing who "
          "chaired says nothing measurable about what was asked. Candidate "
          "transcripts remain excluded - self-selected, unverifiable, recalled "
          "after the outcome is known - they live QUARANTINED in "
          "`src/interview_transcripts.py`, basis hard-capped at WEAK, and are "
          "structurally barred from supporting a CONFIRMED claim. That store "
          "does NOT raise this actor's basis however large it grows."),
)

ALL: tuple[Actor, ...] = CASCADE + STAGES
BY_NAME = {a.name: a for a in ALL}


def can_support_claim(actor: str, required: str = "PARTIAL") -> tuple[bool, str]:
    """May a claim about `actor` be made at `required` evidence strength?

    This is the guard that keeps the unobservable stages from being modelled
    anyway. It is enforced here rather than trusted to each new hypothesis.
    """
    a = BY_NAME.get(actor)
    if a is None:
        return False, f"unknown actor: {actor!r}"
    have, need = BASIS_RANK[a.evidence_basis], BASIS_RANK.get(required, 2)
    if have == 0:
        return False, (f"{actor} has evidence_basis NONE - FORBIDDEN from "
                       f"shaping output; it emits nothing observable")
    if have < need:
        return False, (f"{actor} has {a.evidence_basis} evidence, claim needs "
                       f"{required}. Demote the claim or find a stream.")
    return True, f"{actor}: {a.evidence_basis} evidence via {a.stream}"


def report() -> str:
    lines = ["ACTOR REGISTRY", ""]
    for axis, group in (("cascade (upstream influence)", CASCADE),
                        ("stage (inside the exam)", STAGES)):
        lines.append(f"  {axis}")
        for a in group:
            emits = ", ".join(a.emits) if a.emits else "- nothing observable -"
            lines.append(f"    {a.name:18} {a.evidence_basis:8} {emits[:52]}")
        lines.append("")
    lines.append("  guard checks")
    for who, need in (("prelims_setters", "STRONG"), ("mains_setters", "STRONG"),
                      ("mains_evaluators", "PARTIAL"),
                      ("interview_board", "PARTIAL"), ("agenda", "PARTIAL")):
        ok, why = can_support_claim(who, need)
        lines.append(f"    {'PASS' if ok else 'BLOCK'}  {why}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(report())
