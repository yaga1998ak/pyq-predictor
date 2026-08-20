"""Extract the Reasoning section from every eligible Tier-1 paper.

Section identity comes from question NUMBERING STRUCTURE, not from the topic
tagger. Two layouts exist in this corpus and both are handled explicitly:

  * continuous  -- numbers run 1..100 (2025 reconstructions).
                   Reasoning is Q1-25 by the fixed Tier-1 order.
  * per-section -- numbering restarts at 1 four times (2021-2024 official).
                   Reasoning is the FIRST block in document order.

Canonical Tier-1 order (validated in section_probe.py: quant fingerprints hit
98.6% in the Q51-75 slot):
    1-25 Reasoning | 26-50 General Awareness | 51-75 Quant | 76-100 English

Why this matters: the inherited regex tagger reached only ~68% coverage and
leaked Reasoning items into Quant (a '%'-means-'+' operator question was tagged
`percentage`). Deriving the section structurally removes that whole error class,
and lets a question be counted as Reasoning even when no topic rule fires --
which is exactly what made Direction Sense / Matrix / Ranking read 0.0.
"""

from __future__ import annotations

import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"

Q_RX = re.compile(r"(?:^|\s)Q\.?\s*(\d{1,3})\s*\.?\s", re.M)

SECTIONS = ("reasoning", "general_awareness", "quant", "english")

# Option dialects.
#   official:       "Ans 1. foo 2. bar 3. baz 4. qux"   (digits)
#   reconstruction: "(a) foo (b) bar (c) baz (d) qux"   (letters)
OPT_LETTER = re.compile(r"\(([a-d])\)\s*(.+?)(?=\([a-d]\)|ans\b|$)", re.I | re.S)
OPT_DIGIT = re.compile(r"(?:^|\s)([1-4])\s*\.\s*(.+?)(?=(?:\s[1-4]\s*\.)|$)", re.S)

# Correct-answer markers. "Chosen Option" is the CANDIDATE's pick on official
# response sheets and is NOT evidence of correctness -- excluded deliberately.
ANS_LETTER = re.compile(r"\bans(?:wer)?\b\s*[.:\-]?\s*\(?([a-d])\)?", re.I)
ANS_DIGIT = re.compile(r"\bans(?:wer)?\b\s*[.:\-]?\s*\(?([1-4])\)?", re.I)
CHOSEN_RX = re.compile(r"chosen\s*option", re.I)

DIGIT_TO_LETTER = {"1": "a", "2": "b", "3": "c", "4": "d"}

# Official response sheets append per-question candidate metadata immediately
# after the last option, so it gets swept into that option's text:
#   "(c) 3642 Question ID : 26433078953 Status : Not Answered Chosen Option : --"
# Truncate at the first such marker. Without this the trailing option is both
# unreadable and a giveaway that it is the one carrying the metadata.
OPT_TAIL = re.compile(
    r"\s*(?:"
    r"question\s*id"
    r"|status\s*:"
    r"|chosen\s*option"
    r"|comprehension\s*:"
    r"|section\s*:"
    r"|subquestion"
    r"|\bans(?:wer)?\b"
    r"|Q\s*\.\s*\d{1,3}\s*(?=[A-Z])"
    r").*$",
    re.I | re.S)


def clean_option(text: str) -> str:
    return re.sub(r"\s+", " ", OPT_TAIL.sub("", text)).strip(" .;,")


def numbered_chunks(text: str) -> list[tuple[int, str]]:
    hits = list(Q_RX.finditer(text))
    out = []
    for i, m in enumerate(hits):
        n = int(m.group(1))
        end = hits[i + 1].start() if i + 1 < len(hits) else len(text)
        out.append((n, text[m.end():end]))
    return out


def assign_sections(pairs: list[tuple[int, str]]) -> tuple[list[str | None], str]:
    """Label each chunk with a section. Returns (labels, layout_name)."""
    nums = [n for n, _ in pairs]
    if not nums:
        return [], "empty"

    # Continuous layout: numbers meaningfully exceed 25 and mostly ascend.
    if max(nums) > 60:
        labels = []
        for n in nums:
            if 1 <= n <= 25:
                labels.append("reasoning")
            elif 26 <= n <= 50:
                labels.append("general_awareness")
            elif 51 <= n <= 75:
                labels.append("quant")
            elif 76 <= n <= 100:
                labels.append("english")
            else:
                labels.append(None)
        return labels, "continuous"

    # Per-section layout: split on numbering resets, take blocks in order.
    labels: list[str | None] = []
    block = 0
    prev = 0
    for n in nums:
        if n <= prev:
            block += 1
        prev = n
        labels.append(SECTIONS[block] if block < 4 else None)
    return labels, "per_section"


def parse_options(chunk: str) -> tuple[list[tuple[str, str]], str | None]:
    """Return ([(letter, text)], dialect). Prefers the lettered dialect."""
    lets = [(l.lower(), clean_option(t)[:200])
            for l, t in OPT_LETTER.findall(chunk)]
    seen = {l for l, _ in lets}
    if len(seen) >= 4:
        best, out = {}, []
        for l, t in lets:
            if l not in best and t:
                best[l] = t
        if len(best) >= 4:
            return [(l, best[l]) for l in "abcd"], "letter"

    tail = chunk
    m = re.search(r"\bans\b", chunk, re.I)
    if m:
        tail = chunk[m.start():]
    digs = [(DIGIT_TO_LETTER[d], clean_option(t)[:200])
            for d, t in OPT_DIGIT.findall(tail)]
    best = {}
    for l, t in digs:
        if l not in best and t:
            best[l] = t
    if len(best) >= 4:
        return [(l, best[l]) for l in "abcd"], "digit"
    return [], None


def parse_answer(chunk: str, dialect: str | None) -> tuple[str | None, str]:
    """Return (answer_letter, provenance).

    CRITICAL: in official SSC papers "Ans" INTRODUCES THE OPTION LIST --
    "Ans 1. None 2. One 3. Two 4. Three" -- it does not mark the correct choice.
    An earlier version read the "1" after "Ans" as the key and assigned answer
    'a' to all 508 questions of 2024 without the skew being visible anywhere
    except the answer-letter histogram (63% 'a' corpus-wide).

    Official papers contain no correct-answer marker at all: across a 2024 shift
    there are zero occurrences of "Correct Option", "Answer Key" or "Status".
    So the digit dialect yields NO key, ever. Only the lettered reconstruction
    dialect ("... ans.(b)") carries one, and even that is a Tier-3 coaching
    claim to be verified by solver rather than trusted.
    """
    if CHOSEN_RX.search(chunk):
        # Official response sheet: only the candidate's own pick is present.
        return None, "chosen_option_only"
    if dialect == "letter":
        m = ANS_LETTER.search(chunk)
        if m:
            return m.group(1).lower(), "claimed_letter"
    if dialect == "digit":
        return None, "no_key_in_official_paper"
    return None, "absent"


def stem_of(chunk: str) -> str:
    cut = len(chunk)
    for rx in (re.compile(r"\(a\)", re.I), re.compile(r"\bans\b", re.I)):
        m = rx.search(chunk)
        if m:
            cut = min(cut, m.start())
    stem = re.sub(r"\s+", " ", chunk[:cut]).strip()
    return re.sub(r"^Q\s*\.?\s*\d+\s*\.?\s*", "", stem)


def main() -> None:
    from pypdf import PdfReader

    inv = json.load(open(OUT / "inventory.json"))
    eligible = [r for r in inv["raw"] if r["eligible"]]

    # Drop the byte-identical twin: same sha256 under two different date labels,
    # so one filename is wrong. Keep one, count one observation.
    seen_hash: set[str] = set()
    papers_meta = []
    dropped_dupe = []
    for r in sorted(eligible, key=lambda x: x["path"]):
        if r["sha256"] in seen_hash:
            dropped_dupe.append(r["path"])
            continue
        seen_hash.add(r["sha256"])
        papers_meta.append(r)

    records = []
    per_paper = []
    layout_count = Counter()

    for rec in papers_meta:
        p = ROOT / rec["path"]
        try:
            text = "".join(pg.extract_text() or "" for pg in PdfReader(str(p)).pages)
        except Exception:
            continue
        pairs = numbered_chunks(text)
        labels, layout = assign_sections(pairs)
        layout_count[layout] += 1

        n_reason = n_opts = n_ans = 0
        for (num, chunk), lab in zip(pairs, labels):
            if lab != "reasoning":
                continue
            n_reason += 1
            flat = re.sub(r"\s+", " ", chunk).strip()
            opts, dialect = parse_options(flat)
            ans, prov = parse_answer(flat, dialect)
            stem = stem_of(flat)
            if opts:
                n_opts += 1
            if ans:
                n_ans += 1
            records.append(dict(
                qid=f"{rec['exam_date'] or rec['year_dir']}-S{rec['shift']}-Q{num:03d}",
                year=rec["year_dir"], exam_date=rec["exam_date"], shift=rec["shift"],
                printed_number=num, layout=layout,
                source_pdf=Path(rec["path"]).name,
                source_tier=rec["source_tier"],
                is_reconstruction=rec["is_reconstruction"],
                stem=stem, options=opts, option_dialect=dialect,
                answer=ans, answer_provenance=prov,
                stem_len=len(stem),
            ))
        per_paper.append(dict(
            path=rec["path"], year=rec["year_dir"], date=rec["exam_date"],
            shift=rec["shift"], layout=layout,
            markers=len(pairs), reasoning=n_reason,
            with_options=n_opts, with_answer=n_ans,
        ))

    # ---- report --------------------------------------------------------------
    print("=" * 104)
    print("REASONING EXTRACTION — per year")
    print("=" * 104)
    print(f"{'Year':<6}{'Shifts':>7}{'Layout':>14}{'R found':>9}{'R/shift':>9}"
          f"{'Nominal':>9}{'Recall':>8}{'w/Options':>11}{'w/Answer':>10}")
    print("-" * 104)
    tot = defaultdict(int)
    for y in sorted({r["year"] for r in per_paper}):
        rs = [r for r in per_paper if r["year"] == y]
        lay = Counter(r["layout"] for r in rs).most_common(1)[0][0]
        R = sum(r["reasoning"] for r in rs)
        nom = 25 * len(rs)
        o = sum(r["with_options"] for r in rs)
        a = sum(r["with_answer"] for r in rs)
        tot["shifts"] += len(rs); tot["R"] += R; tot["nom"] += nom
        tot["o"] += o; tot["a"] += a
        print(f"{y:<6}{len(rs):>7}{lay:>14}{R:>9}{R/len(rs):>9.1f}{nom:>9}"
              f"{100*R/nom:>7.0f}%{o:>11}{a:>10}")
    print("-" * 104)
    print(f"{'ALL':<6}{tot['shifts']:>7}{'':>14}{tot['R']:>9}"
          f"{tot['R']/tot['shifts']:>9.1f}{tot['nom']:>9}"
          f"{100*tot['R']/tot['nom']:>7.0f}%{tot['o']:>11}{tot['a']:>10}")

    print(f"\nLayouts: {dict(layout_count)}")
    print(f"Dropped byte-identical duplicate: {dropped_dupe}")

    print("\nAnswer provenance across Reasoning questions:")
    for k, v in Counter(r["answer_provenance"] for r in records).most_common():
        print(f"  {k:<22}{v:>6}  ({100*v/len(records):.1f}%)")
    print("\nOption dialect:")
    for k, v in Counter(str(r["option_dialect"]) for r in records).most_common():
        print(f"  {k:<22}{v:>6}  ({100*v/len(records):.1f}%)")

    print("\nUsable-for-generation pool (options AND answer present), by year:")
    for y in sorted({r["year"] for r in records}):
        rs = [r for r in records if r["year"] == y]
        good = [r for r in rs if r["options"] and r["answer"]]
        print(f"  {y}: {len(good):>5} / {len(rs):<5} ({100*len(good)/len(rs):.0f}%)")

    outliers = [r for r in per_paper if not (20 <= r["reasoning"] <= 26)]
    print(f"\nShifts whose Reasoning count is outside 20-26: {len(outliers)}/{len(per_paper)}")
    for r in sorted(outliers, key=lambda x: x["reasoning"])[:14]:
        print(f"  R={r['reasoning']:>3} markers={r['markers']:>4} {r['layout']:<12} {r['path']}")

    (OUT / "reasoning_questions.json").write_text(json.dumps(records, indent=2))
    (OUT / "reasoning_per_paper.json").write_text(json.dumps(per_paper, indent=2))
    print(f"\nWrote {OUT/'reasoning_questions.json'} ({len(records)} questions)")


if __name__ == "__main__":
    main()
