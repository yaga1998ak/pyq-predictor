"""Extract the English Comprehension section from every eligible Tier-1 paper.

Reuses the section-assignment machinery validated in the Reasoning run
(src/reasoning_extract.py), which derives the section from question NUMBERING
STRUCTURE rather than from a topic tagger:

  * continuous  -- numbers run 1..100 (2025 reconstructions) -> English = Q76-100
  * per_section -- numbering restarts four times (2021-2024)  -> English = 4th block

Canonical Tier-1 order was validated empirically, not assumed: quant lexical
fingerprints hit 98.6% in the Q51-75 slot (src/section_probe.py).

ONE STRUCTURAL DIFFERENCE FROM REASONING, and it matters:
English contains PASSAGE-BASED GROUPS. A Cloze passage or RC passage is printed
once and serves 5+ questions. Consequences handled here:
  * the passage text lands in the FIRST question's chunk, inflating its stem and
    leaving later questions looking context-free;
  * counting those questions as independent observations is fine, but treating
    the passage as part of one question's stem corrupts both difficulty features
    and any length-based diagnostic.
So passages are detected, lifted into their own record, and linked by passage_id.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from reasoning_extract import (assign_sections, numbered_chunks, parse_answer,
                               parse_options, stem_of)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"

# Passage delimiter. Official SSC papers label the block explicitly
# ("Comprehension: In the following passage, some words have been deleted...")
# and REPEAT the whole passage after every question in the group, printed AFTER
# that question's options.
PASSAGE_MARK = re.compile(
    r"Comprehension\s*:|"
    r"(?:In|Read)\s+the\s+following\s+passage|"
    r"Read\s+the\s+(?:given\s+)?passage\s+(?:carefully|and)|"
    r"some\s+(?:of\s+the\s+)?words\s+have\s+been\s+deleted", re.I)

# Where the option block starts -- used to decide whether a passage precedes or
# trails the question, which differs between official and reconstruction papers.
OPT_START = re.compile(r"\(a\)|\bAns\b", re.I)

# Counting blanks is NOT a passage test. An earlier version treated ">=3 blank
# markers and length >= 260" as a passage, which classified 225 para-jumble
# questions as passages -- their segment blanks ("____/____/____/____") trip the
# same pattern. Every "passage" found in 2025 was one of those, while the real
# official passages were missed entirely. Detection now requires the explicit
# lead-in, and blanks are only counted once a passage is already identified.
BLANK_MARK = re.compile(r"_{2,}|\(\s*\d{1,2}\s*\)_*")


# The passage body must stop where the next question begins. Official papers use
# "SubQuestion No : 25" markers that the shared Q-marker regex does not split on,
# so several questions can share one chunk and the passage ran 1224 words into the
# following question's text ("... SubQuestion No : 25 Q.25 Select the most ...").
PASSAGE_END = re.compile(
    r"SubQuestion\s*No|Q\s*\.\s*\d{1,3}|"
    r"Select the most appropriate option to fill in blank", re.I)

# De-spaced extractions make word counts meaningless (a whole passage reads as a
# handful of "words"), so they are flagged and measured in characters instead.
def _despaced(text: str) -> bool:
    return len(text) > 40 and text.count(" ") / len(text) < 0.10


def find_passage(flat_chunk: str) -> tuple[str | None, bool]:
    """Return (passage_text, precedes_question).

    Official style trails the passage after the options; some reconstructions
    print it before the question. Both are handled so the stem stays clean.
    """
    m = PASSAGE_MARK.search(flat_chunk)
    if not m:
        return None, False
    opt = OPT_START.search(flat_chunk)
    precedes = bool(opt and m.start() < opt.start())
    body = flat_chunk[m.start():].strip()
    body = re.sub(r"^Comprehension\s*:\s*", "", body, flags=re.I)
    # Drop the lead-in sentence, then cut at the next question marker.
    lead = re.match(r".{0,200}?(?:each blank|questions? that follow|"
                    r"answer the questions?)\W*", body, re.I | re.S)
    if lead:
        body = body[lead.end():]
    stop = PASSAGE_END.search(body)
    if stop and stop.start() > 120:
        body = body[:stop.start()].strip()
    if len(body) < 200:
        return None, False
    return body, precedes


def passage_key(text: str) -> str:
    """Identity for de-duplication: the passage repeats once per question.

    Keyed on a normalised PREFIX, not the whole text. Hashing the full string
    failed: each reprint of a passage is cut at a slightly different point (the
    same 2025 passage appeared as 331, 341, 331, 337 and 348 words), so five
    copies of one passage hashed to five distinct keys and every 2025 shift
    reported 5 passages instead of 1. 416 of 420 within-shift pairs were >75%
    similar -- i.e. the same passage. The opening 150 alphanumeric characters are
    stable across reprints and unique across genuinely different passages.
    """
    norm = re.sub(r"[^a-z0-9]", "", text.lower())
    return hashlib.sha1(norm[:150].encode()).hexdigest()[:16]


def main() -> None:
    from pypdf import PdfReader

    inv = json.load(open(OUT / "inventory.json"))
    eligible = [r for r in inv["raw"] if r["eligible"]]

    # Drop the byte-identical twin found in the inventory audit: same sha256
    # under two different date labels, so one filename is wrong.
    seen, papers, dropped = set(), [], []
    for r in sorted(eligible, key=lambda x: x["path"]):
        if r["sha256"] in seen:
            dropped.append(r["path"])
            continue
        seen.add(r["sha256"])
        papers.append(r)

    records, per_paper, passages = [], [], []
    layout_count = Counter()

    for rec in papers:
        p = ROOT / rec["path"]
        try:
            text = "".join(pg.extract_text() or "" for pg in PdfReader(str(p)).pages)
        except Exception:
            continue
        pairs = numbered_chunks(text)
        labels, layout = assign_sections(pairs)
        layout_count[layout] += 1

        n_eng = n_opt = n_ans = 0
        # Per-shift passage registry: the same passage is reprinted once per
        # question in its group, so it must collapse to one observation.
        shift_passages: dict[str, str] = {}
        for (num, chunk), lab in zip(pairs, labels):
            if lab != "english":
                continue
            n_eng += 1
            flat = re.sub(r"\s+", " ", chunk).strip()
            opts, dialect = parse_options(flat)
            ans, prov = parse_answer(flat, dialect)
            stem = stem_of(flat)

            passage_id = None
            body, precedes = find_passage(flat)
            if body:
                key = passage_key(body)
                if key not in shift_passages:
                    pid = f"{rec['exam_date']}-S{rec['shift']}-P{len(shift_passages)+1}"
                    shift_passages[key] = pid
                    passages.append(dict(
                        passage_id=pid, year=rec["year_dir"],
                        exam_date=rec["exam_date"], shift=rec["shift"],
                        text=body, length_chars=len(body),
                        length_words=len(body.split()),
                        despaced=_despaced(body),
                        # ~5.5 chars/word is the usual English average; used only
                        # for de-spaced passages, where split() cannot work.
                        est_words=round(len(body) / 5.5),
                        n_blanks=len(BLANK_MARK.findall(body)),
                        source_pdf=Path(rec["path"]).name))
                passage_id = shift_passages[key]
                if precedes:
                    # Passage sits inside the stem region: strip it so the stem
                    # is the question, not the passage.
                    cut = PASSAGE_MARK.search(stem)
                    if cut:
                        stem = stem[:cut.start()].strip() or stem

            if opts:
                n_opt += 1
            if ans:
                n_ans += 1
            records.append(dict(
                qid=f"{rec['exam_date'] or rec['year_dir']}-S{rec['shift']}-EQ{num:03d}",
                year=rec["year_dir"], exam_date=rec["exam_date"],
                shift=rec["shift"], printed_number=num, layout=layout,
                source_pdf=Path(rec["path"]).name,
                source_tier=rec["source_tier"],
                is_reconstruction=rec["is_reconstruction"],
                stem=stem, options=opts, option_dialect=dialect,
                answer=ans, answer_provenance=prov,
                passage_id=passage_id, stem_len=len(stem)))
        per_paper.append(dict(
            path=rec["path"], year=rec["year_dir"], date=rec["exam_date"],
            shift=rec["shift"], layout=layout, markers=len(pairs),
            english=n_eng, with_options=n_opt, with_answer=n_ans,
            passages=len(shift_passages)))

    # ---------------------------------------------------------------- report
    print("=" * 104)
    print("ENGLISH SECTION INVENTORY")
    print("=" * 104)
    print(f"{'Year':<6}{'Shifts':>7}{'Layout':>14}{'E found':>9}{'E/shift':>9}"
          f"{'Nominal':>9}{'Recall':>8}{'w/Options':>11}{'w/Answer':>10}{'Passages':>10}")
    print("-" * 104)
    tot = defaultdict(int)
    for y in sorted({r["year"] for r in per_paper}):
        rs = [r for r in per_paper if r["year"] == y]
        lay = Counter(r["layout"] for r in rs).most_common(1)[0][0]
        E = sum(r["english"] for r in rs)
        nom = 25 * len(rs)
        o = sum(r["with_options"] for r in rs)
        a = sum(r["with_answer"] for r in rs)
        ps = sum(r["passages"] for r in rs)
        for k, v in (("shifts", len(rs)), ("E", E), ("nom", nom),
                     ("o", o), ("a", a), ("p", ps)):
            tot[k] += v
        print(f"{y:<6}{len(rs):>7}{lay:>14}{E:>9}{E/len(rs):>9.1f}{nom:>9}"
              f"{100*E/nom:>7.0f}%{o:>11}{a:>10}{ps:>10}")
    print("-" * 104)
    print(f"{'ALL':<6}{tot['shifts']:>7}{'':>14}{tot['E']:>9}"
          f"{tot['E']/tot['shifts']:>9.1f}{tot['nom']:>9}"
          f"{100*tot['E']/tot['nom']:>7.0f}%{tot['o']:>11}{tot['a']:>10}{tot['p']:>10}")

    print(f"\nLayouts: {dict(layout_count)}")
    print(f"Dropped byte-identical duplicate: {dropped}")
    print("\nAnswer provenance:")
    for k, v in Counter(r["answer_provenance"] for r in records).most_common():
        print(f"  {k:<28}{v:>6}  ({100*v/len(records):.1f}%)")
    print("\nOption dialect:")
    for k, v in Counter(str(r["option_dialect"]) for r in records).most_common():
        print(f"  {k:<28}{v:>6}  ({100*v/len(records):.1f}%)")

    print(f"\nPassages detected: {len(passages)}")
    if passages:
        wl = sorted(p["length_words"] for p in passages)
        print(f"  words: min {wl[0]}  median {wl[len(wl)//2]}  max {wl[-1]}")
        print(f"  with >=3 numbered blanks (cloze-like): "
              f"{sum(1 for p in passages if p['n_blanks'] >= 3)}")
        by_year = Counter(p["year"] for p in passages)
        print(f"  by year: {dict(sorted(by_year.items()))}")

    blank = sum(1 for r in records if len(r["stem"].strip()) < 25)
    print(f"\nBlank/near-empty stems: {blank} ({100*blank/len(records):.1f}%)")

    out = [r for r in per_paper if not (20 <= r["english"] <= 26)]
    print(f"Shifts whose English count is outside 20-26: {len(out)}/{len(per_paper)}")
    for r in sorted(out, key=lambda x: x["english"])[:12]:
        print(f"  E={r['english']:>3} markers={r['markers']:>4} {r['layout']:<12} {r['path']}")

    (OUT / "english_questions.json").write_text(json.dumps(records, indent=2))
    (OUT / "english_passages.json").write_text(json.dumps(passages, indent=2))
    (OUT / "english_per_paper.json").write_text(json.dumps(per_paper, indent=2))
    print(f"\nWrote {OUT/'english_questions.json'} ({len(records)} questions)")


if __name__ == "__main__":
    main()
