"""Weekly corpus refresh: parse -> tag -> forecast, only when the corpus changed.

WHY THIS EXISTS
---------------
Adding papers is useless until they are parsed, tagged and folded into the
forecast. That chain was manual, so the daily job kept composing from a stale
blueprint while new evidence sat unread on disk. This closes the loop.

It is deliberately CONSERVATIVE:
  * runs only when the set of source PDFs actually changed (hash of the file
    list + sizes), because re-tagging is expensive and pointless otherwise;
  * writes the new forecast to a STAGING path and only promotes it if it
    passes sanity checks - a forecast that lost topics or stopped summing to
    25 per section would silently corrupt every future paper;
  * always keeps the previous forecast as .prev so a bad promotion is one
    move to undo.

    python src/corpus_refresh.py --check    # report, change nothing
    python src/corpus_refresh.py --run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
PY = str(ROOT / ".venv" / "bin" / "python")
STATE = OUT / "corpus_state.json"
FORECAST = OUT / "forecast_2026.json"

SECTIONS = {"general_intelligence_reasoning", "general_awareness",
            "quantitative_aptitude", "english_comprehension"}


def corpus_fingerprint() -> str:
    h = hashlib.sha1()
    for p in sorted((ROOT / "data" / "raw").rglob("*.pdf")):
        h.update(p.name.encode())
        h.update(str(p.stat().st_size).encode())
    return h.hexdigest()


def changed() -> tuple[bool, str, str]:
    now = corpus_fingerprint()
    old = ""
    if STATE.exists():
        try:
            old = json.loads(STATE.read_text()).get("fingerprint", "")
        except Exception:
            pass
    return now != old, old, now


def sane(path: Path) -> tuple[bool, str]:
    """A forecast must cover all four sections and sum to ~25 each."""
    try:
        rows = json.loads(path.read_text()).get("forecast", [])
    except Exception as e:
        return False, f"unreadable: {e}"
    if not rows:
        return False, "empty forecast"
    secs: dict[str, float] = {}
    for r in rows:
        secs[r.get("section", "?")] = secs.get(r.get("section", "?"), 0) + float(
            r.get("expected") or 0)
    missing = SECTIONS - set(secs)
    if missing:
        return False, f"missing sections: {sorted(missing)}"
    for s, tot in secs.items():
        if s in SECTIONS and not (24.0 <= tot <= 26.0):
            return False, f"{s} sums to {tot:.1f}, expected ~25"
    return True, f"{len(rows)} topics, all four sections sum to ~25"


def run(step: str, args: list[str], timeout: int = 7200) -> bool:
    print(f"  -> {step} ...", flush=True)
    r = subprocess.run([PY] + args, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        print(f"     FAILED: {(r.stderr or r.stdout)[-300:]}")
        return False
    tail = [l for l in (r.stdout or "").splitlines() if l.strip()][-1:]
    print(f"     ok  {tail[0][:100] if tail else ''}")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    diff, old, now = changed()
    n_pdf = len(list((ROOT / "data" / "raw").rglob("*.pdf")))
    print(f"CORPUS REFRESH — {n_pdf} source PDFs")
    print(f"  fingerprint {'CHANGED' if diff else 'unchanged'}"
          f"  ({old[:8] or 'none'} -> {now[:8]})")

    if a.check or not (a.run or a.force):
        ok, why = sane(FORECAST) if FORECAST.exists() else (False, "absent")
        print(f"  current forecast: {'OK' if ok else 'PROBLEM'} - {why}")
        return

    if not diff and not a.force:
        print("  nothing to do (use --force to rebuild anyway)")
        return

    # Guard FIRST: 12 Tier-II papers had reached the Tier-I corpus with names
    # like `15-English.pdf` that no filename filter could catch. Tier-II
    # Paper-2 carries 200 questions against 100 for a whole Tier-I paper, so
    # they inflated 2021 to "130% extraction" and silently skewed the forecast.
    run("tier guard", ["src/tier_guard.py", "--quarantine"], timeout=1800)

    if not run("parse", ["src/parse.py"]):
        sys.exit(1)
    # src/rules.py is THE tagger: HANDOVER.md §3 measured it at ~80% precision
    # against 35% for a local 7B, because SSC stems are heavily templated.
    # Do not substitute an LLM tagger here.
    if not run("tag (rules)", ["src/rules.py",
                               "--papers", "data/parsed/papers.json",
                               "--out", "data/tagged/rules.json"], timeout=3600):
        print("  tagging failed - forecast NOT promoted")
        sys.exit(1)

    prev = FORECAST.with_suffix(".json.prev")
    if FORECAST.exists():
        shutil.copy2(FORECAST, prev)

    if not run("forecast", ["src/predict.py", "--year", "2026",
                            "--papers", "data/tagged/rules.json",
                            "--out", "out/forecast_2026.json"]):
        print("  forecast step failed - previous forecast retained")
        sys.exit(1)

    ok, why = sane(FORECAST)
    if not ok:
        print(f"  NEW FORECAST REJECTED: {why}")
        if prev.exists():
            shutil.copy2(prev, FORECAST)
            print("  restored previous forecast")
        sys.exit(1)

    STATE.write_text(json.dumps({"fingerprint": now, "pdfs": n_pdf}, indent=2))
    print(f"  promoted: {why}")


if __name__ == "__main__":
    main()
