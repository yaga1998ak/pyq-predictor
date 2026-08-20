"""Forensic inventory of the raw PYQ corpus.

Independently establishes what data actually exists: per-PDF identity, exam
date/shift, extraction quality, duplicate relationships and source tier.
Deliberately does not trust any previously reported counts.

Writes out/inventory.json and prints the summary tables.
"""

from __future__ import annotations

import hashlib
import json
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
BACKUP = ROOT / "data" / "raw_backup"
OUT = ROOT / "out"

MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
    "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}

# Source tiers per the project's confidence hierarchy.
# L1 official SSC, L2 established platform w/ shift-specific paper,
# L3 coaching reconstruction, L4 community memory-based.
RECONSTRUCTION_MARKERS = ("similar-paper", "similar_paper", "memory")


@dataclass
class PaperRecord:
    path: str
    filename: str
    year_dir: int
    sha256: str
    size_bytes: int
    pages: int | None = None
    exam_date: str | None = None          # ISO yyyy-mm-dd
    shift: int | None = None
    language: str = "english"
    source_tier: int = 2
    is_reconstruction: bool = False
    # extraction diagnostics
    chars_extracted: int = 0
    median_chunk_len: int = 0
    q_marker_dialect: str | None = None
    q_markers_found: int = 0
    text_layer: bool = False
    despaced: bool = False
    has_answer_key: bool = False
    reasoning_markers: int = 0
    is_tier2: bool = False
    eligible: bool = True          # Tier-1 English original, usable as evidence
    notes: list[str] = field(default_factory=list)


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def parse_date_shift(name: str, year_dir: int) -> tuple[str | None, int | None]:
    """Pull exam date and shift from the filename.

    Filenames are inconsistent across years: 'ssc-cgl-tier-1-13-august-2021-shift-1',
    'ssc-cgl-01-dec-2022-shift-1-question-paper',
    'SSC-CGL-T-I-Similar-Paper-Held-on-13-Sep-2025-S1-English'.
    """
    low = name.lower().replace("_", "-")

    shift = None
    m = re.search(r"shift[-\s]*(\d)", low) or re.search(r"[-\s]s(\d)[-\s.]", low)
    if m:
        shift = int(m.group(1))

    # day-month(-year) in either order.
    # Scan ALL matches and take the first with a real month token: the first
    # regex match is often spurious ('1-question', '23-tier'), which silently
    # dropped 16 papers' dates in the first run.
    day = month = None
    for m in re.finditer(r"(\d{1,2})(?:st|nd|rd|th)?[-\s]+([a-z]{3,9})", low):
        if m.group(2) in MONTHS:
            day, month = int(m.group(1)), MONTHS[m.group(2)]
            break
    if month is None:
        for m in re.finditer(r"([a-z]{3,9})[-\s]+(\d{1,2})(?:st|nd|rd|th)?", low):
            if m.group(1) in MONTHS:
                month, day = MONTHS[m.group(1)], int(m.group(2))
                break

    # prefer an explicit 4-digit year in the filename over the directory
    my = re.search(r"(20\d{2})", low)
    year = int(my.group(1)) if my else year_dir

    if day and month and 1 <= day <= 31:
        return f"{year:04d}-{month:02d}-{day:02d}", shift
    return None, shift


def extract_text(p: Path) -> tuple[str, int | None]:
    try:
        from pypdf import PdfReader
    except ImportError:
        return "", None
    try:
        r = PdfReader(str(p))
        pages = len(r.pages)
        parts = []
        for pg in r.pages:
            try:
                parts.append(pg.extract_text() or "")
            except Exception:
                parts.append("")
        return "\n".join(parts), pages
    except Exception:
        return "", None


# Question marker dialects seen in this corpus (see HANDOVER §2).
DIALECTS = {
    "Q.n": re.compile(r"Q\.\s*\d{1,3}"),
    "Qn.": re.compile(r"Q\s*\d{1,3}\s*\."),
}

REASONING_HINTS = (
    "in a certain code", "odd one out", "select the odd", "which number will",
    "complete the series", "missing number", "blood relation", "how is",
    "mirror image", "water image", "paper folded", "embedded", "syllogism",
    "conclusion", "venn", "dice", "direction", "facing", "arranged in",
    "rank", "matrix", "cluster of letters",
)

KEY_HINTS = ("answer key", "correct option", "ans.", "answer:")

# Tier-2 papers have a different section structure (Mathematical Abilities,
# Computer Knowledge) and different per-section counts, so they corrupt Tier-1
# topic proportions. Two slipped past the original filename-based filter.
#
# Word-boundary anchored: a bare 'mains' substring matches inside 'remains'
# ("the office remains open" in a syllogism stem), which false-positived a
# legitimate 2025 Tier-1 paper on the first pass.
TIER2_CONTENT = re.compile(
    r"\btier[\s\-]?(?:ii|2)\b"
    r"|\bmathematical\s+abilities\b"
    r"|\bcomputer\s+knowledge\s+and\s+module\b"
    r"|\bcgle\s*20\d{2}\s*tier\b",
    re.I,
)
TIER2_NAME = re.compile(r"\bmains\b|\btier[\s\-]?(?:ii|2)\b", re.I)


def diagnose(rec: PaperRecord, text: str) -> None:
    rec.chars_extracted = len(text)
    rec.text_layer = len(text) > 500

    low = text.lower()
    rec.has_answer_key = any(k in low for k in KEY_HINTS)
    rec.reasoning_markers = sum(low.count(h) for h in REASONING_HINTS)

    # Tier-2 detection: filename OR extracted content. Content wins, because
    # 'SSC-CGL-6-March-2023.pdf' carries no tier hint in its name at all.
    name_norm = rec.filename.replace("_", "-")
    if TIER2_NAME.search(name_norm) or TIER2_CONTENT.search(text[:4000]):
        rec.is_tier2 = True
        rec.eligible = False
        rec.notes.append("TIER2-EXCLUDE")

    if rec.language != "english":
        rec.eligible = False
        rec.notes.append("non-english")

    # de-spaced extraction: very few spaces relative to length
    if text:
        space_ratio = text.count(" ") / max(len(text), 1)
        rec.despaced = space_ratio < 0.05
        if rec.despaced:
            rec.notes.append("despaced-extraction")

    best, best_n = None, 0
    for label, rx in DIALECTS.items():
        n = len(rx.findall(text))
        if n > best_n:
            best, best_n = label, n
    rec.q_marker_dialect, rec.q_markers_found = best, best_n

    # median chunk length between question markers -> detects image-only questions
    if best:
        chunks = DIALECTS[best].split(text)
        lens = [len(c.strip()) for c in chunks[1:]]
        if lens:
            rec.median_chunk_len = int(statistics.median(lens))
            if rec.median_chunk_len < 40:
                rec.notes.append("questions-likely-images")

    if not rec.text_layer:
        rec.notes.append("no-text-layer-needs-ocr")


def build(paths: list[Path], label: str) -> list[PaperRecord]:
    out = []
    for p in sorted(paths):
        try:
            year_dir = int(p.parent.name)
        except ValueError:
            year_dir = 0
        rec = PaperRecord(
            path=str(p.relative_to(ROOT)),
            filename=p.name,
            year_dir=year_dir,
            sha256=sha256_of(p),
            size_bytes=p.stat().st_size,
        )
        low = p.name.lower()
        rec.is_reconstruction = any(m in low.replace("_", "-") for m in RECONSTRUCTION_MARKERS)
        rec.source_tier = 3 if rec.is_reconstruction else 2
        if "hindi" in low:
            rec.language = "hindi"
        rec.exam_date, rec.shift = parse_date_shift(p.name, year_dir)
        if rec.exam_date is None:
            rec.notes.append("date-unparsed")
        if rec.shift is None:
            rec.notes.append("shift-unparsed")

        text, pages = extract_text(p)
        rec.pages = pages
        diagnose(rec, text)
        out.append(rec)
        print(f"  [{label}] {p.name[:66]:<66} "
              f"{rec.pages or '?':>3}p {rec.q_markers_found:>3}q "
              f"{'|'.join(rec.notes) if rec.notes else 'ok'}")
    return out


def main() -> None:
    raw = sorted(RAW.rglob("*.pdf"))
    bak = sorted(BACKUP.rglob("*.pdf"))
    print(f"Scanning {len(raw)} PDFs in data/raw, {len(bak)} in data/raw_backup\n")

    recs = build(raw, "raw")
    print()
    brecs = build(bak, "bak")

    # ---- duplicate detection -------------------------------------------------
    by_hash: dict[str, list[PaperRecord]] = defaultdict(list)
    for r in recs + brecs:
        by_hash[r.sha256].append(r)
    exact_dupes = {h: [r.path for r in v] for h, v in by_hash.items() if len(v) > 1}

    # shift identity: (exam_date, shift) within data/raw only
    by_shift: dict[tuple, list[PaperRecord]] = defaultdict(list)
    unkeyed = []
    for r in recs:
        if r.exam_date and r.shift:
            by_shift[(r.exam_date, r.shift)].append(r)
        else:
            unkeyed.append(r)
    shift_collisions = {f"{k[0]}#S{k[1]}": [r.path for r in v]
                        for k, v in by_shift.items() if len(v) > 1}

    # ---- per-year summary ----------------------------------------------------
    years = sorted({r.year_dir for r in recs})
    print("\n" + "=" * 96)
    print("PER-YEAR INVENTORY (data/raw)")
    print("=" * 96)
    hdr = (f"{'Year':<6}{'PDFs':>6}{'Tier2':>7}{'Elig':>6}{'UniqShift':>11}"
           f"{'Recon':>7}{'Recon%':>8}{'Despaced':>10}{'Keys':>6}")
    print(hdr)
    print("-" * 96)
    totals = defaultdict(int)
    for y in years:
        rs = [r for r in recs if r.year_dir == y]
        el = [r for r in rs if r.eligible]
        us = {(r.exam_date, r.shift) for r in el if r.exam_date and r.shift}
        recon = sum(r.is_reconstruction for r in el)
        row = dict(
            pdfs=len(rs), tier2=sum(r.is_tier2 for r in rs), elig=len(el),
            uniq=len(us), recon=recon,
            desp=sum(r.despaced for r in el),
            keys=sum(r.has_answer_key for r in el),
        )
        for k, v in row.items():
            totals[k] += v
        pct = f"{100 * recon / len(el):.0f}%" if el else "-"
        print(f"{y:<6}{row['pdfs']:>6}{row['tier2']:>7}{row['elig']:>6}"
              f"{row['uniq']:>11}{row['recon']:>7}{pct:>8}"
              f"{row['desp']:>10}{row['keys']:>6}")
    print("-" * 96)
    tpct = f"{100 * totals['recon'] / totals['elig']:.0f}%" if totals['elig'] else "-"
    print(f"{'TOTAL':<6}{totals['pdfs']:>6}{totals['tier2']:>7}{totals['elig']:>6}"
          f"{totals['uniq']:>11}{totals['recon']:>7}{tpct:>8}"
          f"{totals['desp']:>10}{totals['keys']:>6}")

    excluded = [r for r in recs if not r.eligible]
    print(f"\nEXCLUDED from Tier-1 evidence base: {len(excluded)}")
    for r in excluded:
        print(f"  {r.path}  -> {'|'.join(r.notes)}")

    print(f"\nExact-hash duplicate groups (raw+backup): {len(exact_dupes)}")
    for h, paths in list(exact_dupes.items())[:12]:
        print(f"  {h[:12]}  " + "  <=>  ".join(paths))
    print(f"\nSame (date,shift) collisions inside data/raw: {len(shift_collisions)}")
    for k, paths in list(shift_collisions.items())[:12]:
        print(f"  {k}: {paths}")
    print(f"\nRecords with unparsed date or shift: {len(unkeyed)}")
    for r in unkeyed[:12]:
        print(f"  {r.path}  ({'|'.join(r.notes)})")

    OUT.mkdir(exist_ok=True)
    payload = {
        "raw": [asdict(r) for r in recs],
        "backup": [asdict(r) for r in brecs],
        "exact_duplicate_groups": exact_dupes,
        "shift_collisions": shift_collisions,
    }
    (OUT / "inventory.json").write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {OUT / 'inventory.json'}")


if __name__ == "__main__":
    sys.exit(main())
