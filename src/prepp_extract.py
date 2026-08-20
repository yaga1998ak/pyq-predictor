"""Extract solved SSC CGL Tier-I questions from prepp practice papers.

WHY A SEPARATE EXTRACTOR
------------------------
Every other source in this project ships UNSOLVED papers. The official SSC PDFs
carry no key at all (not in text, not in bold, not as annotations); careerpower
and adda247 mirror those. The only answered questions in the corpus so far come
from 2025 memory-based reconstructions.

These papers are different: they carry a dedicated section at the end,

    Answers
    1. Answer: c
    Explanation: ...

which is clean ASCII and joinable to the questions by number.

TWO HAZARDS, BOTH HANDLED
-------------------------
1. BILINGUAL. Many questions are Hindi. The PDFs also decode Devanagari badly,
   leaving NUL bytes. Any stem or option carrying NUL / Devanagari is dropped —
   an unreadable question is worse than a missing one.
2. WRONG EXAM. The same directory holds mocks with (+3,-1) marking and
   cause-effect items, which are not SSC CGL Tier-I (+2,-0.5). Papers are
   accepted only if they show Tier-I section headers.

    python src/prepp_extract.py --dir data/raw/prepp --out out/prepp_questions.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import pypdf

logging.getLogger("pypdf").setLevel(logging.CRITICAL)

ROOT = Path(__file__).resolve().parent.parent
DEVA = re.compile(r"[ऀ-ॿ]")
NUL = "\x00"

SECTION_OF = [
    ("general_awareness", re.compile(r"(?i)general\s+awareness")),
    ("quant", re.compile(r"(?i)quantitative\s+aptitude")),
    ("english", re.compile(r"(?i)english\s+comprehension")),
    ("reasoning", re.compile(r"(?i)general\s+intelligence")),
]

Q_RE = re.compile(r"(?m)^\s*(\d{1,3})\.\s+(.{10,600}?)(?=\n\s*a\.\s)", re.S)
OPT_RE = re.compile(r"(?m)^\s*([abcd])\.\s*(.+?)\s*$")
ANS_RE = re.compile(r"(?m)^\s*(\d{1,3})\.\s*Answer\s*:\s*([abcd])\b")


# The per-question marking scheme is laid out inline and lands in the middle of
# stems ("...with the characters. (+2, -0.5)"). It is page furniture, not part
# of the question, and would otherwise be fed to the tagger and printed in the
# paper.
MARKS = re.compile(r"\(\s*\+\s*\d+\s*,\s*-\s*[\d.]+\s*\)")


def clean(s: str) -> str:
    return " ".join(MARKS.sub(" ", s.replace(NUL, "")).split())


def usable(s: str) -> bool:
    """Reject anything Hindi or encoding-damaged."""
    return bool(s) and NUL not in s and not DEVA.search(s)


def section_at(text: str, pos: int) -> str:
    """Nearest preceding section header wins."""
    best, bpos = "reasoning", -1
    for name, rx in SECTION_OF:
        for m in rx.finditer(text, 0, pos):
            if m.start() > bpos:
                best, bpos = name, m.start()
    return best


def extract(path: Path) -> list[dict]:
    try:
        text = "".join((p.extract_text() or "") for p in pypdf.PdfReader(str(path)).pages)
    except Exception:
        return []

    if not (re.search(r"(?i)general\s+intelligence", text)
            and re.search(r"(?i)english\s+comprehension", text)):
        return []            # not a Tier-I paper

    split = text.rfind("\nAnswers")
    if split < 0:
        return []
    body, keyblock = text[:split], text[split:]

    keys = {int(n): a for n, a in ANS_RE.findall(keyblock)}
    if len(keys) < 40:
        return []

    out = []
    for m in Q_RE.finditer(body):
        num = int(m.group(1))
        if num not in keys:
            continue
        stem = clean(m.group(2))
        tail = body[m.end(): m.end() + 900]
        opts = [(l, clean(t)) for l, t in OPT_RE.findall(tail)][:4]
        if len(opts) != 4:
            continue
        if not usable(stem) or not all(usable(t) for _, t in opts):
            continue         # Hindi or NUL-damaged
        if len(stem) < 15:
            continue
        out.append({
            "stem": stem, "options": [list(o) for o in opts],
            "answer": keys[num], "section": section_at(body, m.start()),
            "verified_by": "published_key:prepp", "source_pdf": path.name,
            "printed_number": num,
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "data" / "raw" / "prepp"))
    ap.add_argument("--out", default=str(ROOT / "out" / "prepp_questions.json"))
    a = ap.parse_args()

    files = sorted(Path(a.dir).glob("*.pdf"))
    allq: list[dict] = []
    for f in files:
        got = extract(f)
        allq += got
        print(f"  {f.name[:34]:<34} {len(got):>4} solved questions")
    from collections import Counter
    Path(a.out).write_text(json.dumps(allq, indent=1))
    print(f"\n  {len(allq)} questions from {len(files)} papers -> {a.out}")
    print(f"  by section: {dict(Counter(q['section'] for q in allq))}")
    print(f"  answer spread: {dict(Counter(q['answer'] for q in allq))}")


if __name__ == "__main__":
    main()
