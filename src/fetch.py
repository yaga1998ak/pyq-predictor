"""Download PYQ PDFs into data/raw/<year>/.

Deliberately rate-limited (default 1.5s between requests). These are free public
resources on someone else's bandwidth; hammering them is both rude and a good way
to get blocked mid-run.

Resumable: already-downloaded files are skipped, so re-running after a failure
costs nothing.

Run:  python src/fetch.py --list urls.tsv          (year<TAB>filename<TAB>url)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import requests

from schema import REPO

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120 Safari/537.36"


def download(url: str, dest: Path, timeout: int = 90) -> tuple[bool, str]:
    if dest.exists() and dest.stat().st_size > 10_000:
        return True, "skip (exists)"
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=timeout, stream=True)
        r.raise_for_status()
    except requests.RequestException as exc:
        return False, f"ERROR {exc}"

    ctype = r.headers.get("content-type", "")
    body = r.content
    # A server returning an HTML error page with 200 is the classic silent
    # failure here -- it lands on disk as a .pdf and blows up at parse time.
    if b"%PDF" not in body[:1024] and "pdf" not in ctype.lower():
        return False, f"NOT A PDF (content-type: {ctype})"

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(body)
    return True, f"{len(body)/1_000_000:.1f} MB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True, help="TSV: year<TAB>filename<TAB>url")
    ap.add_argument("--out", default=str(REPO / "data/raw"))
    ap.add_argument("--delay", type=float, default=1.5)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = []
    for line in Path(args.list).read_text().splitlines():
        if not line.strip():
            continue
        year, fname, url = line.split("\t")
        rows.append((year, fname, url))
    if args.limit:
        rows = rows[: args.limit]

    root = Path(args.out)
    ok = failed = skipped = 0
    failures = []

    for i, (year, fname, url) in enumerate(rows, 1):
        dest = root / year / fname
        success, msg = download(url, dest)
        if success:
            if msg.startswith("skip"):
                skipped += 1
            else:
                ok += 1
                time.sleep(args.delay)
        else:
            failed += 1
            failures.append(f"  {year}/{fname}: {msg}")
        if i % 10 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)}  ok={ok} skip={skipped} failed={failed}", flush=True)

    print(f"\ndownloaded {ok}, skipped {skipped}, failed {failed}")
    if failures:
        print("\nFAILURES:")
        print("\n".join(failures))


if __name__ == "__main__":
    main()
