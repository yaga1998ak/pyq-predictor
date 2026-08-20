"""Email an alert when the daily job cannot deliver a paper.

Without this, a failure only appends to out/daily/failures.log - a file nobody
opens. The watchdog will still deliver a paper from the buffer, so delivery is
safe; what goes unnoticed is that the PIPELINE is broken and the buffer is
draining. That is exactly the failure a study habit hides until it runs out.

Kept deliberately tiny and dependency-free: it must work when the rest of the
system does not.

    python src/notify_failure.py --reason "validation failed: ..." --date 2026-09-20
"""

from __future__ import annotations

import argparse
import json
import smtplib
import ssl
from datetime import date
from email.message import EmailMessage
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def env() -> dict:
    d = {}
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                d[k.strip()] = v.strip()
    return d


def buffer_depth() -> int:
    q = ROOT / "out" / "delivery_queue.json"
    try:
        return len(json.loads(q.read_text()).get("pending", []))
    except Exception:
        return -1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reason", default="unspecified")
    ap.add_argument("--date", default=date.today().isoformat())
    a = ap.parse_args()

    e = env()
    depth = buffer_depth()
    msg = EmailMessage()
    msg["From"] = e.get("SMTP_USER", "")
    msg["To"] = e.get("DAILY_RECIPIENT", "")
    msg["Subject"] = f"[ACTION NEEDED] SSC CGL daily build failed — {a.date}"
    msg.set_content(
        f"The daily build did not produce a valid paper on {a.date}.\n\n"
        f"Reason:\n  {a.reason[:600]}\n\n"
        f"Delivery buffer remaining: {depth} pre-validated papers.\n"
        + ("You will still receive papers from the buffer, but it is draining.\n"
           if depth > 0 else
           "THE BUFFER IS EMPTY — tomorrow there may be no paper.\n")
        + "\nDiagnose with:\n"
          f"  tail -40 {ROOT}/out/daily/run_{a.date}.log\n"
          f"  {ROOT}/.venv/bin/python src/brain.py\n"
          f"  {ROOT}/.venv/bin/python src/prebuild.py --status\n")
    try:
        with smtplib.SMTP_SSL(e.get("SMTP_HOST", "smtp.gmail.com"),
                              int(e.get("SMTP_PORT", 465)),
                              context=ssl.create_default_context(), timeout=60) as s:
            s.login(e["SMTP_USER"], e["SMTP_PASS"])
            s.send_message(msg)
        print(f"ALERT SENT -> {msg['To']} (buffer {depth})")
    except Exception as ex:  # never let the alerter itself crash the workflow
        print(f"alert failed: {type(ex).__name__}: {str(ex)[:120]}")


if __name__ == "__main__":
    main()
