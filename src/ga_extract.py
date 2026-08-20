"""Extract the General Awareness section from every eligible Tier-1 paper.

Section identity comes from question NUMBERING STRUCTURE, as validated in the
three earlier runs (src/reasoning_extract.py):

  * continuous  -- numbers run 1..100 (2025 files)  -> GA = Q26-50
  * per_section -- numbering restarts four times     -> GA = 2nd block

GA differs from the other three sections in one way that governs everything
downstream: it has NO stem templates. "Who among the following..." says nothing
about the topic, so the signal lives in NAMED ENTITIES -- dynasties, rivers,
Articles, schemes, awards, organisations. This module therefore extracts entity
and year features alongside the stem, since those are what the tagger will use.

Also captured here, because §27 needs it: whether a question is static GK or
carries a temporal marker (an explicit year, "recently", "as of"), which is the
basis for the static-versus-current split.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from reasoning_extract import (assign_sections, numbered_chunks, parse_answer,
                              parse_options, stem_of)
from reasoning_tag import normalize

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"

# Temporal markers -> current-affairs linkage.
YEAR_RX = re.compile(r"\b(19\d{2}|20\d{2})\b")
RECENT_RX = re.compile(
    r"\brecently\b|\bas of\b|\bin 20(2[0-9])\b|\bcurrently\b|"
    r"\bwas (?:launched|appointed|inaugurated|signed|released|held)\b|"
    r"\bwon the\b|\bhosted\b", re.I)

# Capitalised multi-word entities: the primary GA signal.
ENTITY_RX = re.compile(r"\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){0,3})\b")
STOP_ENTITY = {
    "Which", "What", "Who", "When", "Where", "The", "In", "Of", "Select",
    "Identify", "Choose", "Following", "None", "Both", "All", "Ans",
    "Question", "Status", "Chosen", "Option", "Section", "General",
    "Awareness", "Comprehension", "As", "For", "With", "From", "By", "At",
    "This", "That", "These", "Those", "It", "He", "She", "They",
    # Instruction furniture, not entities. Statement-based GA items open with
    # "Consider the following", "Assertion (A)", "Read the statements", so these
    # words outranked real entities in the frequency table.
    "Consider", "Assertion", "Reason", "Read", "Under", "According", "Given",
    "Statements", "Statement", "Arrange", "Match", "List", "Column",
    "Codes", "Choose", "Below", "Above", "Also", "Among", "Only", "Not",
    "True", "False", "Correct", "Incorrect", "Wrong", "Right",
}

# "India"/"Indian" appear in a third of all GA stems and carry no topical
# information on their own, so they are excluded from the SIGNAL vocabulary while
# remaining part of longer entities ("Indian Constitution", "Reserve Bank").
LOW_INFO = {"India", "Indian", "Indias"}

ARTICLE_RX = re.compile(r"\bArticle\s+(\d+[A-Z]?)\b", re.I)
AMEND_RX = re.compile(r"\b(\d+)(?:st|nd|rd|th)\s+Amendment\b", re.I)


def entities(text: str) -> list[str]:
    out = []
    for m in ENTITY_RX.finditer(text):
        e = m.group(1).strip()
        head = e.split()[0]
        if head in STOP_ENTITY:
            continue
        if len(e) < 4 or e in LOW_INFO:
            continue
        out.append(e)
    # de-duplicate, keep order
    seen, uniq = set(), []
    for e in out:
        if e.lower() not in seen:
            seen.add(e.lower())
            uniq.append(e)
    return uniq[:12]


def temporal_profile(stem: str) -> dict:
    years = [int(y) for y in YEAR_RX.findall(stem)]
    return dict(
        years=years[:6],
        max_year=max(years) if years else None,
        has_recent_marker=bool(RECENT_RX.search(stem)),
        articles=ARTICLE_RX.findall(stem)[:4],
        amendments=[int(a) for a in AMEND_RX.findall(stem)][:3],
    )


def main() -> None:
    from pypdf import PdfReader

    inv = json.load(open(OUT / "inventory.json"))
    eligible = [r for r in inv["raw"] if r["eligible"]]

    seen, papers, dropped = set(), [], []
    for r in sorted(eligible, key=lambda x: x["path"]):
        if r["sha256"] in seen:
            dropped.append(r["path"])
            continue
        seen.add(r["sha256"])
        papers.append(r)

    records, per_paper = [], []
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

        n_q = n_opt = n_ans = 0
        for (num, chunk), lab in zip(pairs, labels):
            if lab != "general_awareness":
                continue
            n_q += 1
            flat = re.sub(r"\s+", " ", chunk).strip()
            opts, dialect = parse_options(flat)
            ans, prov = parse_answer(flat, dialect)
            stem = stem_of(flat)
            norm = normalize(stem)
            if opts:
                n_opt += 1
            if ans:
                n_ans += 1
            records.append(dict(
                qid=f"{rec['exam_date'] or rec['year_dir']}-S{rec['shift']}-GA{num:03d}",
                year=rec["year_dir"], exam_date=rec["exam_date"],
                shift=rec["shift"], printed_number=num, layout=layout,
                source_pdf=Path(rec["path"]).name,
                source_tier=rec["source_tier"],
                is_reconstruction=rec["is_reconstruction"],
                stem=stem, options=opts, option_dialect=dialect,
                answer=ans, answer_provenance=prov,
                stem_len=len(stem),
                entities=entities(norm),
                temporal=temporal_profile(norm)))
        per_paper.append(dict(
            path=rec["path"], year=rec["year_dir"], date=rec["exam_date"],
            shift=rec["shift"], layout=layout, markers=len(pairs),
            ga=n_q, with_options=n_opt, with_answer=n_ans))

    # ---------------------------------------------------------------- report
    print("=" * 104)
    print("GENERAL AWARENESS — SECTION INVENTORY")
    print("=" * 104)
    print(f"{'Year':<6}{'Shifts':>7}{'Layout':>14}{'GA found':>10}{'GA/shift':>10}"
          f"{'Nominal':>9}{'Recall':>8}{'w/Options':>11}{'w/Answer':>10}")
    print("-" * 104)
    tot = defaultdict(int)
    for y in sorted({r["year"] for r in per_paper}):
        rs = [r for r in per_paper if r["year"] == y]
        lay = Counter(r["layout"] for r in rs).most_common(1)[0][0]
        G = sum(r["ga"] for r in rs)
        nom = 25 * len(rs)
        o = sum(r["with_options"] for r in rs)
        a = sum(r["with_answer"] for r in rs)
        for k, v in (("shifts", len(rs)), ("G", G), ("nom", nom), ("o", o), ("a", a)):
            tot[k] += v
        print(f"{y:<6}{len(rs):>7}{lay:>14}{G:>10}{G/len(rs):>10.1f}{nom:>9}"
              f"{100*G/nom:>7.0f}%{o:>11}{a:>10}")
    print("-" * 104)
    print(f"{'ALL':<6}{tot['shifts']:>7}{'':>14}{tot['G']:>10}"
          f"{tot['G']/tot['shifts']:>10.1f}{tot['nom']:>9}"
          f"{100*tot['G']/tot['nom']:>7.0f}%{tot['o']:>11}{tot['a']:>10}")

    print(f"\nLayouts: {dict(layout_count)}")
    print(f"Dropped byte-identical duplicate: {dropped}")
    print("\nAnswer provenance:")
    for k, v in Counter(r["answer_provenance"] for r in records).most_common():
        print(f"  {k:<28}{v:>6}  ({100*v/len(records):.1f}%)")

    print("\nEXTRACTION QUALITY")
    print(f"{'Year':<6}{'GA':>6}{'blank':>7}{'blank%':>8}{'despaced':>10}"
          f"{'mean entities':>15}")
    print("-" * 104)
    for y in sorted({r["year"] for r in records}):
        rs = [r for r in records if r["year"] == y]
        blank = sum(1 for r in rs if len(r["stem"].strip()) < 25)
        dsp = sum(1 for r in rs if len(r["stem"]) > 40
                  and r["stem"].count(" ") / max(len(r["stem"]), 1) < 0.10)
        ent = sum(len(r["entities"]) for r in rs) / len(rs)
        print(f"{y:<6}{len(rs):>6}{blank:>7}{100*blank/len(rs):>7.0f}%{dsp:>10}"
              f"{ent:>15.1f}")

    print("\nSTATIC vs CURRENT-AFFAIRS MARKERS (§27)")
    print(f"{'Year':<6}{'GA':>6}{'has year':>10}{'recent kw':>11}"
          f"{'either':>8}{'static%':>9}")
    print("-" * 104)
    for y in sorted({r["year"] for r in records}):
        rs = [r for r in records if r["year"] == y]
        hy = sum(1 for r in rs if r["temporal"]["years"])
        hk = sum(1 for r in rs if r["temporal"]["has_recent_marker"])
        ei = sum(1 for r in rs if r["temporal"]["years"]
                 or r["temporal"]["has_recent_marker"])
        print(f"{y:<6}{len(rs):>6}{hy:>10}{hk:>11}{ei:>8}"
              f"{100*(len(rs)-ei)/len(rs):>8.0f}%")

    # Year-lag distribution: how far back do year-citing GA questions reach?
    print("\nYEAR-LAG of year-citing GA questions (exam year minus cited year)")
    lag = Counter()
    for r in records:
        my = r["temporal"]["max_year"]
        if my and my <= r["year"]:
            d = r["year"] - my
            lag["same year" if d == 0 else
                "1 year" if d == 1 else
                "2 years" if d == 2 else
                "3 years" if d == 3 else
                "4-8 years" if d <= 8 else "historical (>8y)"] += 1
    tl = sum(lag.values())
    for k in ("same year", "1 year", "2 years", "3 years", "4-8 years",
              "historical (>8y)"):
        if lag[k]:
            print(f"  {k:<20}{lag[k]:>6}  {100*lag[k]/tl:>5.1f}%")

    print("\nMOST FREQUENT ENTITIES (the primary GA signal)")
    ec = Counter()
    for r in records:
        for e in r["entities"]:
            ec[e] += 1
    for e, c in ec.most_common(25):
        print(f"  {c:>4}  {e}")

    out = [r for r in per_paper if not (20 <= r["ga"] <= 26)]
    print(f"\nShifts whose GA count is outside 20-26: {len(out)}/{len(per_paper)}")
    for r in sorted(out, key=lambda x: x["ga"])[:10]:
        print(f"  GA={r['ga']:>3} markers={r['markers']:>4} {r['layout']:<12} {r['path']}")

    (OUT / "ga_questions.json").write_text(json.dumps(records, indent=2))
    (OUT / "ga_per_paper.json").write_text(json.dumps(per_paper, indent=2))
    print(f"\nWrote {OUT/'ga_questions.json'} ({len(records)} questions)")


if __name__ == "__main__":
    main()
