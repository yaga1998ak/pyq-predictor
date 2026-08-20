"""Reclaim disk the pipeline generates, without touching anything irreplaceable.

WHY THIS EXISTS
---------------
This system writes files every single day and never removed any of them. Left
alone until the exam that is roughly 300 daily PDFs plus their markdown, logs
and journals. It also learned the hard way that OCR with compression disabled
inflated the corpus 10-19x (2018: 51MB -> 870MB), so generated artefacts need
a janitor, not good intentions.

DELIBERATELY CONSERVATIVE. It only removes things that are either
regenerable from committed inputs or genuinely spent:

  daily markdown/logs   older than KEEP_DAYS, PDF kept (the PDF is the artefact)
  delivered PDFs        older than KEEP_PDF_DAYS AND already marked sent
  .sent_ markers        older than KEEP_DAYS
  stale grow_*.log      pool-growth logs, pure noise once read

It NEVER touches data/raw, data/official, data/tier2, out/*.json (backtest
artefacts the PDF quotes), or the Obsidian vault.

    python src/housekeeping.py --dry-run    # show what would go
    python src/housekeeping.py --run
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DAILY = ROOT / "out" / "daily"

KEEP_DAYS = 21        # markdown, logs, sent-markers
KEEP_PDF_DAYS = 60    # delivered PDFs

NEVER = ("data/raw", "data/official", "data/tier2", "out/forecast",
         "out/setter", "out/layer_", "out/question_level", "out/temporal")


def age_days(p: Path) -> float:
    return (time.time() - p.stat().st_mtime) / 86400


def collect() -> list[tuple[Path, str]]:
    doomed: list[tuple[Path, str]] = []
    if not DAILY.exists():
        return doomed
    for p in DAILY.iterdir():
        if p.is_dir():
            continue
        try:
            age = age_days(p)
        except OSError:
            continue
        n = p.name
        if n.endswith(".md") and age > KEEP_DAYS:
            doomed.append((p, f"markdown, {age:.0f}d (PDF retained)"))
        elif n.startswith("run_") and n.endswith(".log") and age > KEEP_DAYS:
            doomed.append((p, f"run log, {age:.0f}d"))
        elif n.startswith(".sent_") and age > KEEP_DAYS:
            doomed.append((p, f"sent marker, {age:.0f}d"))
        elif n.endswith(".pdf") and age > KEEP_PDF_DAYS:
            # only if it was actually delivered
            stamp = DAILY / f".sent_{n.replace('SSC_CGL_2026_Daily_','').replace('.pdf','')}"
            if stamp.exists():
                doomed.append((p, f"delivered PDF, {age:.0f}d"))
    for p in (ROOT / "out").glob("grow_*.log"):
        if age_days(p) > 7:
            doomed.append((p, "pool-growth log"))
    return doomed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()

    doomed = collect()
    total = sum(p.stat().st_size for p, _ in doomed if p.exists())
    print(f"HOUSEKEEPING — {len(doomed)} file(s), {total/1e6:.1f} MB")
    for p, why in doomed[:12]:
        print(f"  {p.name[:52]:<52} {why}")
    if len(doomed) > 12:
        print(f"  ... and {len(doomed)-12} more")

    if not a.run:
        print("\n  dry run — nothing deleted. Use --run to act.")
        return
    freed = 0
    for p, _ in doomed:
        try:
            freed += p.stat().st_size
            p.unlink()
        except OSError:
            pass
    print(f"\n  freed {freed/1e6:.1f} MB")


if __name__ == "__main__":
    main()
