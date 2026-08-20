"""Pre-build a validated buffer of papers so delivery never depends on today.

THE FAILURE THIS REMOVES
------------------------
Generation and delivery were coupled: if the morning build broke, no paper was
emailed. A study habit that silently skips days is broken even when the code is
"correct" - fail-safe is not good enough, it has to be fail-operational.

So papers are built AHEAD, validated at build time, and queued. Delivery then
just picks the next validated PDF off the queue - a few lines with almost no
dependencies, which is the part that must never fail.

A break in generation now costs buffer depth, not a missed day. With a 7-day
buffer the whole pipeline can be down for a week before you notice a gap.

    python src/prebuild.py --days 7
    python src/prebuild.py --status
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
DAILY = OUT / "daily"
QUEUE = OUT / "delivery_queue.json"
PY = str(ROOT / ".venv" / "bin" / "python")
TARGET_DEPTH = 7


def load_q() -> dict:
    if QUEUE.exists():
        try:
            return json.loads(QUEUE.read_text())
        except Exception:
            pass
    return {"pending": [], "sent": []}


def save_q(q: dict) -> None:
    QUEUE.parent.mkdir(parents=True, exist_ok=True)
    QUEUE.write_text(json.dumps(q, indent=2))


def build_one(day: str) -> dict | None:
    """Build + validate + render one paper. Returns queue entry or None."""
    md = DAILY / f"SSC_CGL_2026_Daily_{day}.md"
    # Build directly - do NOT go through run_daily.sh, which runs the brain
    # (and can trigger pool growth) on every call. Buffering must be cheap.
    b = subprocess.run([PY, str(ROOT / "src" / "daily_run.py"), "--date", day],
                       capture_output=True, text=True, timeout=900)
    if b.returncode != 0:
        return None
    v = subprocess.run([PY, str(ROOT / "src" / "validate_daily.py"), "--md", str(md)],
                       capture_output=True, text=True, timeout=300)
    if v.returncode != 0:
        return None  # never queue a paper that failed validation
    r = subprocess.run([PY, str(ROOT / "src" / "daily_run.py"),
                        "--date", day, "--pdf"],
                       capture_output=True, text=True, timeout=900)
    pdf = next((l.split("=", 1)[1] for l in r.stdout.splitlines()
                if l.startswith("PDF_PATH=")), None)
    if not pdf or not Path(pdf).exists():
        return None
    return {"date": day, "pdf": pdf, "bytes": Path(pdf).stat().st_size}


def topup(depth: int = TARGET_DEPTH) -> dict:
    q = load_q()
    have = {e["date"] for e in q["pending"]} | {e["date"] for e in q["sent"]}
    made, failed = [], []
    d = date.today()
    while len(q["pending"]) < depth:
        d += timedelta(days=1)
        day = d.isoformat()
        if day in have:
            continue
        e = build_one(day)
        if e:
            q["pending"].append(e); made.append(day)
        else:
            failed.append(day)
        if len(made) + len(failed) > depth * 3:
            break
    q["pending"].sort(key=lambda e: e["date"])
    save_q(q)
    return {"built": made, "failed": failed, "depth": len(q["pending"])}


def status() -> str:
    q = load_q()
    pend = q["pending"]
    L = ["DELIVERY QUEUE", "=" * 50,
         f"  buffered : {len(pend)} papers",
         f"  sent     : {len(q['sent'])}"]
    if pend:
        L.append(f"  next     : {pend[0]['date']}")
        L.append(f"  covers   : {pend[0]['date']} -> {pend[-1]['date']}")
    ok = all(Path(e["pdf"]).exists() for e in pend)
    L.append(f"  files ok : {'yes' if ok else 'MISSING FILES'}")
    if len(pend) < 3:
        L.append("  WARNING: buffer low - run --days 7")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=0)
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    if a.days:
        r = topup(a.days)
        print(f"built {len(r['built'])}, failed {len(r['failed'])}, "
              f"depth now {r['depth']}")
        if r["failed"]:
            print("  failed:", ", ".join(r["failed"][:5]))
    print(status())


if __name__ == "__main__":
    main()
