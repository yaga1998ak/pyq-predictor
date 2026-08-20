"""Test whether printed question numbers can assign sections deterministically.

SSC CGL Tier-1 is a fixed 100-question layout:
    Q1-25  General Intelligence & Reasoning
    Q26-50 General Awareness
    Q51-75 Quantitative Aptitude
    Q76-100 English Comprehension

If the printed numbers survive PDF extraction, the section of every question is
known exactly, with no reliance on the 68%-coverage regex tagger. This probe
validates the assumption instead of trusting it: it scores each quartile by
lexical fingerprints of the section it is *supposed* to be, so a paper with a
different ordering shows up as a mismatch rather than silently poisoning counts.
"""

from __future__ import annotations

import json
import re
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "out"

# Two marker dialects, plus the compact 'Q1.' style used by 2025 reconstructions.
Q_RX = re.compile(r"(?:^|\s)Q\.?\s*(\d{1,3})\s*\.?\s", re.M)

FINGERPRINTS = {
    "reasoning": (
        "certain code", "odd one out", "select the odd", "which number will replace",
        "complete the series", "missing number", "blood relation", "mirror image",
        "water image", "paper folded", "embedded", "syllogism", "conclusions",
        "venn", "dice", "direction", "facing", "rank", "matrix",
        "cluster of letters", "series", "analogy", "related to the",
        "same way as", "signs", "interchanged",
    ),
    "general_awareness": (
        "who among", "which of the following state", "capital of", "river",
        "dynasty", "article", "constitution", "scheme", "won the", "awarded",
        "minister", "founded", "festival", "dance", "temple", "vitamin",
        "chemical", "census", "olympic", "tournament", "governor",
    ),
    "quant": (
        "find the value", "how many", "percentage", "profit", "loss", "discount",
        "interest", "ratio", "average", "speed", "train", "pipe", "cistern",
        "triangle", "circle", "cylinder", "sphere", "sin", "cos", "tan",
        "simplify", "cm", "km/h", "rs.", "sum of money",
    ),
    "english": (
        "synonym", "antonym", "idiom", "one word substitution", "spelling",
        "sentence", "passage", "grammatically", "voice of", "narration",
        "para jumble", "fill in the blank", "most appropriate meaning",
        "error", "correctly spelt", "segment",
    ),
}

SECTION_OF_NUMBER = [
    (1, 25, "reasoning"),
    (26, 50, "general_awareness"),
    (51, 75, "quant"),
    (76, 100, "english"),
]


def section_for(n: int) -> str | None:
    for lo, hi, name in SECTION_OF_NUMBER:
        if lo <= n <= hi:
            return name
    return None


def extract_numbered(text: str) -> list[tuple[int, str]]:
    """Return (printed_number, body) pairs in document order."""
    hits = list(Q_RX.finditer(text))
    out = []
    for i, m in enumerate(hits):
        n = int(m.group(1))
        end = hits[i + 1].start() if i + 1 < len(hits) else len(text)
        out.append((n, text[m.end():end]))
    return out


def score(body: str) -> tuple[str | None, dict[str, int]]:
    low = body.lower()
    sc = {k: sum(low.count(w) for w in ws) for k, ws in FINGERPRINTS.items()}
    best = max(sc, key=sc.get)
    return (best if sc[best] > 0 else None), sc


def main() -> None:
    from pypdf import PdfReader

    inv = json.load(open(OUT / "inventory.json"))
    eligible = [r for r in inv["raw"] if r["eligible"]]

    rows = []
    agree_tot = Counter()
    seen_tot = Counter()

    for rec in eligible:
        p = ROOT / rec["path"]
        try:
            text = "".join(pg.extract_text() or "" for pg in PdfReader(str(p)).pages)
        except Exception:
            continue
        pairs = extract_numbered(text)
        nums = [n for n, _ in pairs]

        # Is this a clean 1..100 layout?
        uniq = sorted(set(n for n in nums if 1 <= n <= 100))
        monotonic = nums == sorted(nums)
        agree = seen = 0
        for n, body in pairs:
            exp = section_for(n)
            if exp is None or len(body.strip()) < 25:
                continue
            got, _ = score(body)
            if got is None:
                continue
            seen += 1
            seen_tot[exp] += 1
            if got == exp:
                agree += 1
                agree_tot[exp] += 1

        rows.append(dict(
            path=rec["path"], year=rec["year_dir"], date=rec["exam_date"],
            shift=rec["shift"], markers=len(pairs), uniq_1_100=len(uniq),
            max_num=max(nums) if nums else 0, monotonic=monotonic,
            scored=seen, agree=agree,
            agree_pct=round(100 * agree / seen, 1) if seen else None,
        ))

    print("=" * 100)
    print("SECTION-BY-QUESTION-NUMBER PROBE")
    print("=" * 100)
    print(f"{'Year':<6}{'Papers':>7}{'MedMarkers':>12}{'Med#1-100':>11}"
          f"{'Monotonic':>11}{'MedAgree%':>11}")
    print("-" * 100)
    for y in sorted({r["year"] for r in rows}):
        rs = [r for r in rows if r["year"] == y]
        med_m = int(statistics.median([r["markers"] for r in rs]))
        med_u = int(statistics.median([r["uniq_1_100"] for r in rs]))
        mono = sum(r["monotonic"] for r in rs)
        ags = [r["agree_pct"] for r in rs if r["agree_pct"] is not None]
        med_a = f"{statistics.median(ags):.0f}%" if ags else "-"
        print(f"{y:<6}{len(rs):>7}{med_m:>12}{med_u:>11}"
              f"{f'{mono}/{len(rs)}':>11}{med_a:>11}")

    print("\nPER-SECTION fingerprint agreement (all eligible papers):")
    for s in ("reasoning", "general_awareness", "quant", "english"):
        if seen_tot[s]:
            print(f"  Q-range {s:<18} {agree_tot[s]:>6}/{seen_tot[s]:<6} "
                  f"= {100*agree_tot[s]/seen_tot[s]:>5.1f}%")

    bad = [r for r in rows if r["uniq_1_100"] < 90]
    print(f"\nPapers with <90 distinct printed numbers in 1..100: {len(bad)}/{len(rows)}")
    for r in sorted(bad, key=lambda x: x["uniq_1_100"])[:15]:
        print(f"  {r['uniq_1_100']:>3} nums  max={r['max_num']:>3}  {r['path']}")

    (OUT / "section_probe.json").write_text(json.dumps(rows, indent=2))
    print(f"\nWrote {OUT / 'section_probe.json'}")


if __name__ == "__main__":
    main()
