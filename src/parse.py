"""PDF -> structured questions.

PYQ PDFs vary wildly: some are clean text, many are scans. This handles the text
case and tells you loudly when a file is image-only, because a silent zero-question
parse that flows into the tagger is how you end up backtesting on nothing.

For scanned PDFs run OCR first (ocrmypdf --force-ocr in.pdf out.pdf), then re-run.

Naming convention for input files:  <exam>_<year>[_<shift>].pdf
    ssc_cgl_2023.pdf
    ssc_cgl_2023_shift2.pdf

Run:  python src/parse.py --raw data/raw --out data/parsed/papers.json
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from pypdf import PdfReader

from schema import Paper, Question, save_papers, REPO

# "1." / "Q1)" / "Q.1" / "12 ." at the start of a line
Q_PATTERN = re.compile(r"^\s*(?:Q\s*\.?\s*)?(\d{1,3})\s*[\.\)]\s+(.+)", re.MULTILINE)

# Real-world PYQ filenames follow no single convention. Observed on careerpower:
#   ssc-cgl-tier-1-13-august-2021-shift-1.pdf
#   ssc-cgl-01-dec-2022-shift-1-question-paper.pdf
#   SSC-CGL-13-September-2024-Shift-1.pdf
#   SSC_CGL_2023_Tier_1_14th-July-2023-Shift-1.pdf
#   SSC-CGL-T-I-Similar-Paper-Held-on-12-Sep-2025-S1-English.pdf
# So extract each field independently rather than matching one rigid pattern.
YEAR_RE = re.compile(r"(20(?:1[5-9]|2[0-6]))")
SHIFT_RE = re.compile(r"(?:shift[-_ ]?|\bS)(\d)", re.I)
DATE_RE = re.compile(
    r"(\d{1,2})(?:st|nd|rd|th)?[-_ ]?"
    r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*",
    re.I,
)
SIMILAR_RE = re.compile(r"similar|memory[-_ ]?based", re.I)


def parse_filename(stem: str, folder_year: int | None = None):
    """Extract (exam, year, shift, date_label, source_type) from a filename.

    folder_year (from data/raw/<year>/) wins when present -- it was derived once,
    deliberately, when building the download list, rather than re-guessed here.
    """
    year = folder_year
    if year is None:
        m = YEAR_RE.search(stem)
        if not m:
            raise ValueError(
                f"no exam year in '{stem}' and no year folder. "
                "Put the file in data/raw/<year>/ or include the year in the name."
            )
        year = int(m.group(1))

    shift_m = SHIFT_RE.search(stem)
    shift = f"S{shift_m.group(1)}" if shift_m else None

    date_m = DATE_RE.search(stem)
    date_label = f"{int(date_m.group(1))}-{date_m.group(2).title()}" if date_m else None

    source_type = "memory_based" if SIMILAR_RE.search(stem) else "official"
    return "SSC_CGL", year, shift, date_label, source_type


def extract_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


# SSC papers mark questions "Q.1 / Q.2 ..." and answer OPTIONS as "1. 2. 3. 4.".
# Splitting on bare numbers therefore splits on every option as well, producing
# ~60% too many "questions", each one the tail of the previous answer glued to
# the real question. Always prefer the explicit Q. marker when the document has
# them; fall back to bare numbering only for papers that lack it.
# Two marker dialects appear across years and must BOTH be matched:
#   "Q.1 text"  (2022-2024 official papers)
#   "Q1. text"  (2025 memory-based papers)
# Matching only the first silently falls back to bare-number splitting, which
# splits on answer options and inflates counts ~20%.
# Trailing space is NOT guaranteed -- 2024 papers contain "Q.2How many squares".
# Requiring \s there drops the marker, falls back to bare-number splitting and
# inflates those papers to ~160 questions. Use a lookahead for the first text
# character instead.
QMARK_PATTERN = re.compile(r"\bQ\s*\.?\s*(\d{1,3})\s*\.?(?=\s*[A-Za-z(])")
OPTION_TAIL = re.compile(r"\bAns\b.*$", re.IGNORECASE | re.DOTALL)

# Response sheets (candidate answer records) contain question IDs and chosen
# options but NOT the question text -- the questions are images. They parse to
# plausible-looking fragments, which is worse than failing: they would silently
# contribute garbage topic labels to the corpus.
RESPONSE_SHEET = re.compile(r"Chosen Option|Status\s*:\s*(Answered|Not Answered)", re.I)

# PDF text extraction sometimes loses inter-word spacing entirely
# ("Afterinterchangingthegiven..."). Such text still tags, but far less
# reliably, so it is worth flagging rather than discovering later.
def spacing_ratio(text: str) -> float:
    """Fraction of characters that are spaces. Normal English prose is ~0.15."""
    return text.count(" ") / max(len(text), 1)


def _split_on(pattern: re.Pattern, text: str) -> list[str]:
    matches = list(pattern.finditer(text))
    out = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = re.sub(r"\s+", " ", text[m.start() : end]).strip()
        if len(body) > 15:  # drop fragments from headers/footers
            out.append(body)
    return out


def split_questions(text: str, expect: int = 100) -> list[str]:
    """Split into questions, choosing the marker style the document actually uses."""
    qmark_hits = len(QMARK_PATTERN.findall(text))

    # Require a plausible number of Q. markers before trusting them -- a stray
    # "Q.1" in a header should not override a differently-formatted paper.
    if qmark_hits >= expect * 0.5:
        chunks = _split_on(QMARK_PATTERN, text)
    else:
        chunks = _split_on(Q_PATTERN, text)

    # Trim each chunk at the answer marker: options and worked solutions add
    # tokens without adding topic signal, and long chunks slow tagging.
    cleaned = []
    for c in chunks:
        c = OPTION_TAIL.sub("", c).strip()
        if len(c) > 15:
            cleaned.append(c)
    return cleaned


def parse_pdf(pdf_path: Path, expect: int = 100) -> Paper:
    folder_year = None
    if pdf_path.parent.name.isdigit() and len(pdf_path.parent.name) == 4:
        folder_year = int(pdf_path.parent.name)

    exam, year, shift, date_label, source_type = parse_filename(pdf_path.stem, folder_year)
    text = extract_text(pdf_path)

    chunks = split_questions(text, expect)

    if not chunks:
        raise ValueError(
            f"no questions found -- likely a scanned/image PDF. "
            f"Fix with: ocrmypdf --force-ocr '{pdf_path}' '{pdf_path}'"
        )

    # Presence of response-sheet markers is NOT itself disqualifying: SSC's
    # official papers embed candidate responses alongside full question text.
    # What matters is whether question TEXT survived extraction. Response sheets
    # whose questions are images yield a stream of ~30-char fragments; real
    # papers yield ~100+ char questions.
    lengths = sorted(len(c) for c in chunks)
    median_len = lengths[len(lengths) // 2]
    if median_len < 40:
        raise ValueError(
            f"no question text (median chunk {median_len} chars) -- response "
            "sheet whose questions are images. Unusable."
        )

    # Missing word spacing degrades tagging but does not make the paper useless,
    # so record it rather than discarding data.
    ratio = spacing_ratio(text)
    text_quality = "degraded_spacing" if ratio < 0.08 else "ok"

    # qid must be unique across the corpus: several sittings share a year+shift,
    # separated only by date.
    tag = f"{year}-{date_label or 'nodate'}-{shift or 'S1'}"
    questions = [
        Question(
            qid=f"{tag}-Q{i+1:03d}",
            year=year,
            exam=exam,
            text=chunk,
            source_pdf=pdf_path.name,
        )
        for i, chunk in enumerate(chunks)
    ]
    return Paper(
        year=year,
        exam=exam,
        shift=shift,
        date_label=date_label,
        source_type=source_type,
        text_quality=text_quality,
        questions=questions,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(REPO / "data/raw"))
    ap.add_argument("--out", default=str(REPO / "data/parsed/papers.json"))
    ap.add_argument("--expect", type=int, default=100,
                    help="expected questions per paper; warns when far off")
    args = ap.parse_args()

    # rglob: papers live in data/raw/<year>/ subfolders
    pdfs = sorted(Path(args.raw).rglob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"no PDFs under {args.raw}. Run src/fetch.py first.")

    papers, failures = [], []
    for pdf in pdfs:
        try:
            paper = parse_pdf(pdf, args.expect)
        except Exception as exc:  # keep going; report everything at the end
            failures.append(f"  {pdf.parent.name}/{pdf.name}: {exc}")
            continue
        n = len(paper.questions)
        flag = "" if abs(n - args.expect) <= args.expect * 0.25 else "  <-- CHECK"
        papers.append(paper)
        if flag:
            print(f"  {pdf.name:<58} {n:>4} q{flag}")

    from collections import Counter
    by_year = Counter(p.year for p in papers)
    by_src = Counter(p.source_type for p in papers)
    by_q = Counter(p.text_quality for p in papers)
    print("\nparsed by year:")
    for y in sorted(by_year):
        q = sum(len(p.questions) for p in papers if p.year == y)
        print(f"  {y}: {by_year[y]:>3} papers  {q:>6} questions")
    print(f"\nsource: {dict(by_src)}   text_quality: {dict(by_q)}")

    if failures:
        print(f"\nFAILED ({len(failures)}):")
        print("\n".join(failures[:20]))
        if len(failures) > 20:
            print(f"  ... and {len(failures)-20} more")

    if papers:
        save_papers(papers, Path(args.out))
        total = sum(len(p.questions) for p in papers)
        print(f"\nparsed {len(papers)} papers ({total} questions) -> {args.out}")
        print(f"next: python src/tag.py --papers {args.out}")
    else:
        raise SystemExit("\nnothing parsed successfully.")


if __name__ == "__main__":
    main()
