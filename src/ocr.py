"""OCR the scanned PDFs that the text parser cannot read.

14 of the 84 downloaded papers are image-only scans -- almost all of 2023, which
is precisely the year the backtest needs: with only 2021-2025 available and three
years consumed by training, recovering 2023 adds a whole test year and roughly
halves the noise in every skill estimate. That is worth more than any further
tagger tuning.

Safety: ocrmypdf --force-ocr rewrites the file. A crash or a bad OCR pass midway
would destroy an original we would have to re-download, so this writes to a temp
file and only replaces the original after verifying the output actually parses.
Originals are also copied to data/raw_backup/ before anything is touched.

    python src/ocr.py --dry-run     # list what needs OCR, touch nothing
    python src/ocr.py               # OCR them
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

from parse import extract_text, split_questions, RESPONSE_SHEET
from schema import REPO


def needs_ocr(pdf: Path, expect: int = 100) -> tuple[bool, str]:
    """Decide whether a PDF is worth OCRing, and say why.

    Distinguishes three cases the parser lumps together: genuinely scanned pages
    (OCR helps), response sheets whose questions are images (OCR cannot recover
    question text that was never there), and files that already parse fine.
    """
    try:
        text = extract_text(pdf)
    except Exception as exc:
        return True, f"unreadable ({type(exc).__name__})"

    chunks = split_questions(text, expect)
    if not chunks:
        return True, "no extractable text"

    lengths = sorted(len(c) for c in chunks)
    if lengths[len(lengths) // 2] < 40:
        if len(RESPONSE_SHEET.findall(text)) >= 10:
            return False, "response sheet — questions are images, OCR won't help"
        return True, "text present but fragmentary"

    if len(chunks) < expect * 0.5:
        return True, f"only {len(chunks)} questions found"

    return False, f"{len(chunks)} questions — fine"


def ocr_one(pdf: Path, backup_dir: Path, timeout: int = 900) -> tuple[bool, str]:
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup = backup_dir / pdf.name
    if not backup.exists():
        shutil.copy2(pdf, backup)

    tmp = pdf.with_suffix(".ocr.tmp.pdf")
    cmd = [
        "ocrmypdf",
        "--force-ocr",        # pages carry a junk text layer; replace it
        "--optimize", "1",
        "--quiet",
        str(pdf),
        str(tmp),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        tmp.unlink(missing_ok=True)
        return False, f"timeout after {timeout}s"

    if r.returncode != 0:
        tmp.unlink(missing_ok=True)
        return False, (r.stderr or r.stdout).strip().splitlines()[-1][:120] if (r.stderr or r.stdout) else f"exit {r.returncode}"

    # Verify BEFORE replacing: an OCR run can succeed and still yield nothing
    # useful, and swapping in a file that parses worse than the original is a
    # silent regression.
    try:
        n_after = len(split_questions(extract_text(tmp)))
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        return False, f"output unreadable ({type(exc).__name__})"

    if n_after < 10:
        tmp.unlink(missing_ok=True)
        return False, f"OCR produced only {n_after} questions — keeping original"

    tmp.replace(pdf)
    return True, f"{n_after} questions"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(REPO / "data/raw"))
    ap.add_argument("--backup", default=str(REPO / "data/raw_backup"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    pdfs = sorted(Path(args.raw).rglob("*.pdf"))
    todo, skip = [], []
    for pdf in pdfs:
        need, why = needs_ocr(pdf)
        (todo if need else skip).append((pdf, why))

    print(f"{len(pdfs)} PDFs: {len(todo)} need OCR, {len(skip)} do not\n")
    for pdf, why in todo:
        print(f"  OCR   {pdf.parent.name}/{pdf.name:<52} {why}")
    unhelpable = [(p, w) for p, w in skip if "response sheet" in w]
    for pdf, why in unhelpable:
        print(f"  SKIP  {pdf.parent.name}/{pdf.name:<52} {why}")

    if args.dry_run or not todo:
        print("\n--dry-run: nothing modified." if args.dry_run else "\nnothing to do.")
        return

    if shutil.which("ocrmypdf") is None:
        raise SystemExit("ocrmypdf not installed: brew install ocrmypdf")

    print(f"\noriginals -> {args.backup}\n")
    ok = failed = 0
    t0 = time.time()
    for i, (pdf, _) in enumerate(todo, 1):
        success, msg = ocr_one(pdf, Path(args.backup), args.timeout)
        status = "ok  " if success else "FAIL"
        ok, failed = (ok + 1, failed) if success else (ok, failed + 1)
        print(f"  [{i}/{len(todo)}] {status} {pdf.name:<52} {msg}", flush=True)

    print(f"\nOCR'd {ok}, failed {failed}  ({time.time()-t0:.0f}s)")
    print("next: python src/parse.py   (re-parse to pick up the new text)")


if __name__ == "__main__":
    main()
