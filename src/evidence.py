"""Evidence adapter — the ONLY component that knows what PYQ data looks like.

THE BOUNDARY THIS CREATES
-------------------------
The brain must be a pure reasoning unit: it models how the SSC question-setting
team behaves, and it must not care whether the evidence arrived as a PDF, a
tagged JSON, a scraped page, or a hand-typed note. Previously
`src/setter_brain.py` opened `data/tagged/rules.json` itself, which welded the
model of the TEAM to the shape of one file. Change the file and the brain
breaks; use the brain for another exam and it cannot run at all.

So all format knowledge stops here. This module reads whatever the corpus
currently is and emits ONE stable structure:

    Observation
        paper      opaque id
        date       exam date label, or None
        shift      shift label, or None
        section    reasoning | general_awareness | quant | english | None
        topic      taxonomy label, or None
        stem       the question text
        frame      stem with all digits blanked -> the reusable question shell
        numbers    integers appearing in the stem
        length     characters, a crude difficulty proxy

The brain consumes a list of these and nothing else. Swap the corpus, swap the
exam, swap the tagger - the brain is unchanged.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TAGGED = ROOT / "data" / "tagged" / "rules.json"

DIGITS = re.compile(r"\d+")
NUMS = re.compile(r"\b\d{1,6}\b")
Q_MARKER = re.compile(r"^\s*(?:q\s*\.?\s*\d+\s*[.):]?|\d+\s*[.)])\s*", re.I)
NON_ALNUM = re.compile(r"[^a-z0-9\s]+")


@dataclass(frozen=True)
class Observation:
    paper: str
    date: str | None
    shift: str | None
    section: str | None
    topic: str | None
    stem: str
    frame: str
    numbers: tuple[int, ...] = field(default=())
    length: int = 0


def _norm(t: str) -> str:
    t = Q_MARKER.sub("", (t or "").strip()).lower()
    return " ".join(NON_ALNUM.sub(" ", t).split())


def load(min_questions: int = 15) -> list[Observation]:
    """Read the current corpus and emit Observations. Format knowledge ends here."""
    if not TAGGED.exists():
        return []
    out: list[Observation] = []
    for idx, p in enumerate(json.loads(TAGGED.read_text())):
        qs = [q for q in p.get("questions", []) if (q.get("text") or "").strip()]
        if len(qs) < min_questions:
            continue
        # Paper identity must be UNIQUE per paper. `source_pdf` is absent from
        # the tagged corpus (0/156), so falling back to date_label collapsed
        # every shift on a date into one "paper" and silently zeroed both the
        # blueprint and frame-reuse tests. The list index guarantees uniqueness;
        # year/date/shift are kept in the id so it stays human-readable.
        pid = "|".join(str(x) for x in
                       (p.get("year"), p.get("date_label"), p.get("shift"), idx))
        for q in qs:
            text = q["text"]
            n = _norm(text)
            out.append(Observation(
                paper=pid,
                date=(f'{p.get("year")}-{p.get("date_label")}'
                      if p.get("date_label") else None),
                shift=str(p.get("shift")) if p.get("shift") is not None else None,
                section=q.get("section"),
                topic=q.get("topic"),
                stem=text,
                frame=DIGITS.sub("#", n),
                numbers=tuple(int(m) for m in NUMS.findall(text)
                              if 1 < int(m) < 10000),
                length=len(text),
            ))
    return out


def summary(obs: list[Observation]) -> str:
    papers = {o.paper for o in obs}
    tagged = sum(1 for o in obs if o.topic)
    days = {o.date for o in obs if o.date}
    return (f"{len(obs)} observations | {len(papers)} papers | {len(days)} dates | "
            f"{tagged} topic-labelled ({100*tagged/max(len(obs),1):.0f}%)")


if __name__ == "__main__":
    o = load()
    print(summary(o))
    if o:
        s = o[0]
        print(f"\nsample:\n  section={s.section} topic={s.topic}\n  frame={s.frame[:76]}")
