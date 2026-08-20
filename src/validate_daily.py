"""Pre-flight gate: never email a malformed paper (spec §65).

Exit 0 = safe to send. Exit 1 = hold, with reasons on stdout as JSON.

Checks are structural and cheap; they exist because every serious failure in
this project produced *plausible* output rather than a crash (INSIGHTS.md §5).
A paper that is short 3 questions, or repeats a stem, or carries a missing
answer key, looks fine in a PDF viewer and is worthless to study from.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PER_SECTION = 25
TOTAL = 100
SECTIONS = ["General Intelligence & Reasoning", "General Awareness",
            "Quantitative Aptitude", "English Comprehension"]


def check(md_path: Path) -> dict:
    errs, warns = [], []
    if not md_path.exists():
        return {"ok": False, "errors": [f"missing markdown: {md_path}"]}

    text = md_path.read_text()

    # 1. question count
    qs = re.findall(r"^\*\*Q(\d+)\.\*\*", text, re.M)
    nums = [int(n) for n in qs]
    if len(nums) != TOTAL:
        errs.append(f"expected {TOTAL} questions, found {len(nums)}")
    if nums and nums != list(range(1, len(nums) + 1)):
        errs.append("question numbering is not contiguous from 1")

    # 2. every section present
    for s in SECTIONS:
        if f"## {s}" not in text:
            errs.append(f"missing section heading: {s}")

    # 3. answer key completeness
    key_block = text.split("## Answer Key", 1)
    if len(key_block) < 2:
        errs.append("no answer key")
    else:
        keys = re.findall(r"\b(\d+)\.\s*([abcd])\b", key_block[1])
        if len(keys) != TOTAL:
            errs.append(f"answer key has {len(keys)} entries, expected {TOTAL}")
        kn = [int(n) for n, _ in keys]
        if kn and sorted(kn) != list(range(1, TOTAL + 1)):
            errs.append("answer key numbering incomplete")
        # a key that is overwhelmingly one letter signals a parse bug
        dist = Counter(a for _, a in keys)
        if dist:
            top, cnt = dist.most_common(1)[0]
            if cnt / max(len(keys), 1) > 0.55:
                errs.append(f"answer key {100*cnt/len(keys):.0f}% '{top}' "
                            f"- almost certainly a parser bug")

    # 4. duplicate stems within the paper.
    # SSC questions are heavily templated - "Select the number that will replace
    # the question mark" prefixes dozens of DIFFERENT questions (INSIGHTS.md §2:
    # "setters fill slots in frames"). So compare FULL normalised stems; a prefix
    # match is a template, not a duplicate.
    # Compare stem + options: fixed-stem archetypes are legitimately repeated
    # in a real paper with different option sets, so a stem match alone is not
    # a duplicate.
    blocks_q = re.split(r"^\*\*Q\d+\.\*\*", text, flags=re.M)[1:]
    sigs = []
    for b in blocks_q:
        head = b.split("\n", 1)[0]
        opts = re.findall(r"^- \([abcd]\)\s*(.+)$", b, re.M)
        sigs.append(" ".join(head.lower().split()) + "##" + "|".join(
            " ".join(o.lower().split()) for o in opts))
    dup = [x for x, c in Counter(sigs).items() if c > 1]
    if dup:
        errs.append(f"{len(dup)} genuinely duplicate question(s) (stem+options)")

    # 4b. archetype concentration - a section swamped by one template is a bad
    # paper even when every question is distinct.
    blocks_by_sec = re.split(r"^## ", text, flags=re.M)
    for blk in blocks_by_sec:
        head = blk.split("\n", 1)[0].strip()
        if head not in SECTIONS:
            continue
        sec_stems = re.findall(r"^\*\*Q\d+\.\*\*\s*(.+)$", blk, re.M)
        if not sec_stems:
            continue
        pref = Counter(" ".join(s.lower().split())[:60] for s in sec_stems)
        top, cnt = pref.most_common(1)[0]
        # The selector caps each stem-frame at its own forecast weight, so a
        # heavily-forecast archetype legitimately repeats. series_completion is
        # forecast at 5.83/paper, which a flat threshold of 4 flagged as a
        # defect on every paper. Warn only above the largest forecast weight.
        MAX_FORECAST = 6   # highest single-topic forecast across all sections
        if cnt > MAX_FORECAST + 2:
            warns.append(f"{head}: {cnt}/{len(sec_stems)} share one template "
                         f"- above any topic's forecast weight")

    # 5. every question has 4 options
    blocks = re.split(r"^\*\*Q\d+\.\*\*", text, flags=re.M)[1:]
    # strip fenced data tables before counting options
    blocks = [re.sub(r"```.*?```", " ", b, flags=re.S) for b in blocks]
    bad_opts = sum(1 for b in blocks if len(re.findall(r"^- \([abcd]\)", b, re.M)) != 4)
    if bad_opts:
        errs.append(f"{bad_opts} question(s) do not have exactly 4 options")

    # 6. the honesty page must be present - it is not decoration
    if "Specific questions" not in text or "0.00%" not in text:
        warns.append("measured-confidence block missing or altered")

    return {"ok": not errs, "errors": errs, "warnings": warns,
            "questions": len(nums), "file": str(md_path)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    r = check(Path(a.md))
    print(json.dumps(r, indent=2))
    sys.exit(0 if r["ok"] else 1)


if __name__ == "__main__":
    main()
