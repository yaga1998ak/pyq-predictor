"""Reject Tier-II papers from the Tier-I corpus, by CONTENT not filename.

Six Tier-II papers reached data/raw/2021/ named `15-English.pdf`,
`16-Maths-English.pdf` etc. Nothing in the filename or the source page's link
text said "Tier-II", so a name-based filter passed them straight through. They
then parsed at ~200 questions each (Tier-II Paper-2 English has 200 questions,
against 100 for a whole Tier-I paper), which is exactly the "2021 extracts at
130%" anomaly.

The tell is the SECTION NAMES, which differ between tiers:

    Tier-I   General Intelligence and Reasoning | General Awareness
             Quantitative Aptitude | English Comprehension
    Tier-II  Mathematical Abilities | Quantitative Abilities
             English Language and Comprehension | Computer Knowledge

    python src/tier_guard.py --scan          # report, move nothing
    python src/tier_guard.py --quarantine    # move offenders to data/tier2/
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import pypdf

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
QUARANTINE = ROOT / "data" / "tier2"

T2_MARKERS = ("english language and comprehension", "quantitative abilities",
              "mathematical abilities", "computer knowledge test",
              "computer proficiency")
T1_MARKERS = ("general intelligence and reasoning", "general awareness",
              "quantitative aptitude", "english comprehension")


def classify(path: Path) -> tuple[str, str]:
    """Return (tier, why). Reads only the first pages - the header is enough."""
    try:
        pages = pypdf.PdfReader(str(path)).pages
        head = "".join((p.extract_text() or "") for p in pages[:6]).lower()
        # Page count is a cheap stand-in for question count: extracting every
        # page of 166 PDFs to count markers took minutes. Tier-II Paper-2
        # (200 questions) runs 70-85 pages; a Tier-I paper runs 30-45.
        npages = len(pages)
    except Exception as e:
        return "unknown", f"unreadable: {type(e).__name__}"

    secs = [s.strip().lower() for s in re.findall(r"section\s*:\s*([A-Za-z &]+)", head)]
    t2 = [m for m in T2_MARKERS if any(m in s for s in secs) or m in head]
    t1 = [m for m in T1_MARKERS if any(m in s for s in secs)]

    if t2 and not t1:
        return "tier2", f"section says {t2[0]!r}"
    if t2 and t1:
        # both appear; section headers are authoritative over stray prose
        if any(any(m in s for s in secs) for m in T2_MARKERS):
            return "tier2", f"section header says {t2[0]!r}"
    if npages > 60:
        return "tier2", f"{npages} pages (Tier-I runs 30-45)"
    if t1:
        return "tier1", f"section says {t1[0]!r}"
    return "tier1", f"{npages} pages, no Tier-II marker"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--quarantine", action="store_true")
    a = ap.parse_args()

    bad = []
    for pdf in sorted(RAW.rglob("*.pdf")):
        tier, why = classify(pdf)
        if tier == "tier2":
            bad.append((pdf, why))
            print(f"  TIER-2  {pdf.relative_to(RAW)}  ({why})")

    print(f"\n  {len(bad)} Tier-II paper(s) in the Tier-I corpus")
    if bad and a.quarantine:
        QUARANTINE.mkdir(parents=True, exist_ok=True)
        for pdf, _ in bad:
            shutil.move(str(pdf), str(QUARANTINE / pdf.name))
        print(f"  moved to {QUARANTINE.relative_to(ROOT)}/ (kept, not deleted)")
    elif bad:
        print("  run with --quarantine to move them out")


if __name__ == "__main__":
    main()
