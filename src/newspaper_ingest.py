"""Extract exam-relevant FACTS from newspaper PDFs into zone-mapped items.

WHAT THIS TAKES AND WHAT IT LEAVES
----------------------------------
It takes *facts*: who was appointed, which scheme launched at what cost, who
won which award, which body published which index. Facts are what GA questions
are made of, and facts are not the newspaper's expression - they are the events
it reports.

It does NOT store or reproduce article text. Each item keeps one extracted
sentence as provenance so a claim can be traced, and nothing more. The output
is a study index, not a copy of the paper.

WHY SENTENCE-LEVEL, NOT ARTICLE-LEVEL
-------------------------------------
A newspaper is overwhelmingly not exam-relevant - opinion, sport, local crime,
markets. Article-level mapping would drag all of that in behind one keyword.
Working at sentence level with a two-signal gate keeps precision high and
deliberately discards most of each issue. INSIGHTS.md §4: a wrong label biases
every downstream count; a missing one only shrinks the sample.

    python src/newspaper_ingest.py --dir ~/Downloads --limit 5
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pypdf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
STORE = OUT / "current_affairs.json"

import sys
sys.path.insert(0, str(ROOT / "src"))
from current_affairs import ZONE_RULES, _kw  # reuse the validated rules

# PDF text artefacts seen in these files
FIXES = [("/uni20B9", "₹"), ("/f_i", "fi"), ("/f_l", "fl"), ("/f_f", "ff"),
         ("’", "'"), ("“", '"'), ("”", '"'), ("\xad", "")]

SENT = re.compile(r"(?<=[.!?])\s+")
WS = re.compile(r"\s+")

# A sentence must contain one of these to be a candidate FACT, not commentary.
FACT_VERBS = re.compile(
    r"\b(appointed|named|launched|inaugurated|approved|signed|won|awarded|"
    r"conferred|released|announced|passed|ratified|elected|sworn|unveiled|"
    r"topped|ranked|declared|allocated|set up|established)\b", re.I)

# Proper-noun-ish or numeric anchor - a fact usually names something.
ANCHOR = re.compile(r"(₹\s?[\d,.]+|\b\d{4}\b|\b[A-Z][a-z]+ [A-Z][a-z]+\b|"
                    r"\b[A-Z]{2,}\b)")

DATE_IN_NAME = re.compile(r"(\d{1,2})[~\-_.](\d{1,2})[~\-_.](\d{4})")



# Newspaper PDFs emit runs of repeated glyphs where graphics/rules were laid
# out ("Nature's signals BBBBBBBB..."). These pass every semantic filter, so
# they need an explicit structural check.
RUN = re.compile(r"(.)\1{6,}")

# "launched" is a genuine scheme signal but also matches military and sporting
# usage ("Russia launched an offensive"). Exclude those contexts outright
# rather than trying to disambiguate them per-verb.
EXCLUDE = re.compile(
    r"\b(offensive|missile|airstrike|drone attack|war on|troops|casualt|"
    r"killed|wounded|ceasefire|invasion|shelling|"
    r"match|innings|wicket|goal|striker|semi-final|quarter-final|"
    r"runner-up|triumph|title clash)\b", re.I)


def is_noise(s: str) -> bool:
    if RUN.search(s):
        return True
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return True
    # a real sentence is not dominated by one character
    from collections import Counter
    if Counter(letters).most_common(1)[0][1] / len(letters) > 0.35:
        return True
    # or by tokens with no vowels (column-break artefacts)
    toks = [t for t in s.split() if t.isalpha()]
    if toks and sum(1 for t in toks if not set(t.lower()) & set("aeiou")) / len(toks) > 0.4:
        return True
    return False


def clean(t: str) -> str:
    for a, b in FIXES:
        t = t.replace(a, b)
    # PDFs of newspapers break words across column line-wraps
    t = re.sub(r"-\n", "", t)
    t = t.replace("\n", " ")
    return WS.sub(" ", t).strip()


def issue_date(name: str) -> str | None:
    m = DATE_IN_NAME.search(name)
    if not m:
        return None
    d, mo, y = m.groups()
    try:
        return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"
    except Exception:
        return None


def zone_for_sentence(s: str) -> str | None:
    low = s.lower()
    scores = {z: sum(1 for k in kws if _kw(k).search(low))
              for z, kws in ZONE_RULES.items()}
    best = max(scores, key=lambda z: scores[z])
    return best if scores[best] > 0 else None


def facts_from(path: Path, max_pages: int = 20) -> list[dict]:
    try:
        r = pypdf.PdfReader(str(path))
    except Exception as e:
        print(f"  {path.name}: unreadable ({type(e).__name__})")
        return []
    text = clean("".join((p.extract_text() or "")
                         for p in r.pages[:max_pages]))
    day = issue_date(path.name)
    out, seen = [], set()
    for s in SENT.split(text):
        s = s.strip()
        if not (60 <= len(s) <= 320):
            continue
        if not FACT_VERBS.search(s):
            continue
        if not ANCHOR.search(s):
            continue
        if is_noise(s):
            continue
        if EXCLUDE.search(s):
            continue
        z = zone_for_sentence(s)
        if not z:
            continue
        key = s[:90].lower()
        if key in seen:
            continue
        seen.add(key)
        out.append({"headline": s, "summary": "", "zone": z,
                    "published": day or "", "source": f"TH/{path.name[:28]}",
                    "seen": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "data" / "newspapers"))
    ap.add_argument("--glob", default="THE HINDU UPSC IAS EDITION*.pdf")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--keep", type=int, default=600)
    a = ap.parse_args()

    files = sorted(glob.glob(str(Path(a.dir) / a.glob)))
    if a.limit:
        files = files[-a.limit:]
    print(f"NEWSPAPER INGEST — {len(files)} issue(s)")

    store = json.loads(STORE.read_text()) if STORE.exists() else {"items": []}
    known = {i["headline"][:90].lower() for i in store.get("items", [])}

    fresh = []
    for f in files:
        got = [x for x in facts_from(Path(f))
               if x["headline"][:90].lower() not in known]
        for x in got:
            known.add(x["headline"][:90].lower())
        fresh += got
        print(f"  {Path(f).name[:46]:<46} {len(got):>4} facts")

    store["items"] = (fresh + store.get("items", []))[: a.keep]
    STORE.write_text(json.dumps(store, indent=2))

    from collections import Counter
    print(f"\n  new facts: {len(fresh)}   stored total: {len(store['items'])}")
    if fresh:
        print("  by zone:")
        for z, n in Counter(i["zone"] for i in fresh).most_common(10):
            print(f"    {z:<28}{n:>4}")
        print("\n  samples:")
        for i in fresh[:5]:
            print(f"    [{i['zone']}] {i['headline'][:96]}")


if __name__ == "__main__":
    main()
