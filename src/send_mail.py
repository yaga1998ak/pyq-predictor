"""Send the daily PDF over SMTP. Credentials come from .env, never inline."""
from __future__ import annotations
import argparse, mimetypes, smtplib, ssl, sys
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

def send(pdf: Path, to: str, date: str) -> None:
    e = env()
    msg = EmailMessage()
    msg["From"] = e["SMTP_USER"]
    msg["To"] = to
    msg["Subject"] = f"SSC CGL 2026 Tier-I — Daily Practice Paper — {date}"
    msg.set_content(
        f"Daily SSC CGL Tier-I 2026 practice paper attached ({date}).\n\n"
        "100 questions (25 per section) plus a 1-page General Awareness brief.\n\n"
        "The GA knowledge zones are blind-validated at 81.9% coverage against 624\n"
        "held-out 2025 questions. Specific-question prediction measured 0.00% and\n"
        "is not claimed - see the confidence page on page 1.\n\n"
        "Every answer is either published by SSC or computationally verified.\n")
    data = pdf.read_bytes()
    ctype, _ = mimetypes.guess_type(pdf.name)
    maintype, subtype = (ctype or "application/pdf").split("/", 1)
    msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=pdf.name)

    with smtplib.SMTP_SSL(e.get("SMTP_HOST", "smtp.gmail.com"),
                          int(e.get("SMTP_PORT", 465)),
                          context=ssl.create_default_context(), timeout=60) as s:
        s.login(e["SMTP_USER"], e["SMTP_PASS"])
        s.send_message(msg)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--to", default=env().get("DAILY_RECIPIENT", ""))
    ap.add_argument("--date", default="")
    a = ap.parse_args()
    p = Path(a.pdf)
    if not p.exists():
        print(f"MISSING PDF: {p}"); sys.exit(1)
    send(p, a.to, a.date or p.stem.split("_")[-1])
    print(f"SENT -> {a.to}  ({p.name}, {p.stat().st_size:,} bytes)")

if __name__ == "__main__":
    main()
