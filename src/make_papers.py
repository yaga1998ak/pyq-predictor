"""Generate multiple SSC CGL Tier-1 mock papers as PDFs, by difficulty band.

Each paper is 100 real SSC questions (25 per section) in the 2026 forecast's
topic mix, rendered to a printable PDF with the answer key on its own page.

Difficulty is a within-topic tercile (see difficulty.py) -- a "hard" paper has
the same topic composition as an easy one, just the harder questions from each
topic. That keeps the papers comparable and preserves the forecast weighting.

Papers never share a question: the pool is consumed as papers are built, so a
set of six is six distinct sittings rather than six reshuffles of the same items.

    python src/make_papers.py                      # 6 papers: 2 easy, 2 medium, 2 hard
    python src/make_papers.py --per-band 3         # 9 papers
    python src/make_papers.py --bands hard --per-band 4
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

from difficulty import band_within_topic
from mock_paper import allocate, harvest
from schema import Taxonomy, REPO

# The corpus is full of ₹, ×, ÷, – and curly quotes. Core PDF fonts are Latin-1
# only and silently render those as black squares, so a Unicode TTF is required.
FONT_CANDIDATES = [
    ("SSCUni", "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
    ("SSCUni", "/Library/Fonts/Arial Unicode.ttf"),
]


def register_font() -> str:
    for name, path in FONT_CANDIDATES:
        if Path(path).exists():
            pdfmetrics.registerFont(TTFont(name, path))
            return name
    return "Helvetica"  # degrades on symbols, but still produces a readable PDF


def styles(font: str):
    ss = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("t", parent=ss["Title"], fontName=font, fontSize=17,
                                spaceAfter=2, textColor=colors.HexColor("#12263f")),
        "sub": ParagraphStyle("s", parent=ss["Normal"], fontName=font, fontSize=8.5,
                              alignment=TA_CENTER, textColor=colors.HexColor("#5b6b7f"),
                              spaceAfter=10),
        "sec": ParagraphStyle("h", parent=ss["Heading2"], fontName=font, fontSize=11.5,
                              spaceBefore=12, spaceAfter=6,
                              textColor=colors.HexColor("#1c3d5a")),
        "q": ParagraphStyle("q", parent=ss["Normal"], fontName=font, fontSize=9.5,
                            leading=13, spaceAfter=3),
        "opt": ParagraphStyle("o", parent=ss["Normal"], fontName=font, fontSize=9,
                              leading=12, leftIndent=14),
        "tag": ParagraphStyle("g", parent=ss["Normal"], fontName=font, fontSize=6.8,
                              textColor=colors.HexColor("#93a1b0"), spaceAfter=8),
    }


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_pdf(picked: list[dict], tax: Taxonomy, path: Path,
               paper_no: int, band: str, year: int) -> None:
    font = register_font()
    st = styles(font)
    doc = SimpleDocTemplate(
        str(path), pagesize=A4,
        leftMargin=16 * mm, rightMargin=16 * mm,
        topMargin=14 * mm, bottomMargin=14 * mm,
        title=f"SSC CGL Tier-1 Mock {year} — Paper {paper_no} ({band})",
    )
    flow = [
        Paragraph(f"SSC CGL Tier&#8209;1 &nbsp;·&nbsp; Mock Paper {paper_no}", st["title"]),
        Paragraph(
            f"{band.upper()} &nbsp;|&nbsp; 100 questions &nbsp;|&nbsp; 25 per section "
            f"&nbsp;|&nbsp; 60 minutes &nbsp;|&nbsp; +2 correct, &#8722;0.5 wrong<br/>"
            f"Real SSC questions (2023&#8211;2025) in the {year} forecast topic mix. "
            f"Answer key on the final page.",
            st["sub"]),
    ]

    n = 0
    for section in tax.sections:
        rows = [q for q in picked if tax.topic_to_section[q["topic"]] == section]
        if not rows:
            continue
        flow.append(Paragraph(
            f"{section.replace('_',' ').title()} &nbsp;({len(rows)} questions)", st["sec"]))
        for q in rows:
            n += 1
            q["_no"] = n
            block = [Paragraph(f"<b>{n}.</b> {esc(q['stem'])}", st["q"])]
            for letter, text in q["options"]:
                block.append(Paragraph(f"({letter}) &nbsp;{esc(text)}", st["opt"]))
            meta = q["topic"].replace("_", " ")
            if q.get("method"):
                meta += f" &nbsp;›&nbsp; <b>{q['method'].replace('_',' ')}</b>"
            if q.get("rule"):
                meta += f" &nbsp;[{esc(q['rule'])}]"
            block.append(Paragraph(f"{meta} · {q['year']}", st["tag"]))
            # KeepTogether stops a stem from being orphaned from its options
            # across a page break, which makes a printed paper unusable.
            flow.append(KeepTogether(block))

    flow.append(PageBreak())

    # Topics the pool could not supply. Silently omitting them would teach a
    # candidate that the exam has no direction-sense or matrix questions, which
    # is false -- the corpus simply has none in extractable form.
    present = {q["topic"] for q in picked}
    missing = [t for t in tax.topics if t not in present]
    if missing:
        flow.append(Paragraph("Not covered by this paper", st["sec"]))
        flow.append(Paragraph(
            "SSC asks these topics, but the source corpus has no extractable "
            "questions for them (they are figure-based, or were lost in PDF "
            "extraction). <b>Study them separately &mdash; their absence here is a "
            "limitation of this paper, not of the exam.</b>", st["q"]))
        by_sec = {}
        for t in missing:
            by_sec.setdefault(tax.topic_to_section[t], []).append(t.replace("_", " "))
        for sec, ts in by_sec.items():
            flow.append(Paragraph(
                f"<b>{sec.replace('_',' ').title()}:</b> {', '.join(ts)}", st["opt"]))
        flow.append(Spacer(1, 10))

    flow.append(Paragraph("Answer Key", st["sec"]))
    ordered = sorted(picked, key=lambda q: q["_no"])
    per_col = 25
    cols = [ordered[i:i + per_col] for i in range(0, len(ordered), per_col)]
    data = []
    for r in range(max(len(c) for c in cols)):
        row = []
        for c in cols:
            row += [str(c[r]["_no"]), c[r]["answer"].upper()] if r < len(c) else ["", ""]
        data.append(row)
    t = Table(data, colWidths=[11 * mm, 11 * mm] * len(cols))
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), font),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#12263f")),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#dbe3ec")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    flow.append(t)
    doc.build(flow)


def take(by_topic: dict, want: dict, used: set, tax: Taxonomy, pool: list) -> list[dict]:
    picked = []
    for topic, k in want.items():
        avail = [q for q in by_topic.get(topic, []) if id(q) not in used]
        got = avail[:k]
        for q in got:
            used.add(id(q))
        picked.extend(got)
        if len(got) < k:  # backfill from the same section so it still totals 25
            need = k - len(got)
            sec = tax.topic_to_section[topic]
            alt = [q for q in pool
                   if tax.topic_to_section[q["topic"]] == sec and id(q) not in used]
            for q in alt[:need]:
                used.add(id(q))
                picked.append(q)
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forecast", default=str(REPO / "out/forecast_2026.json"))
    ap.add_argument("--raw", default=str(REPO / "data/raw"))
    ap.add_argument("--outdir", default=str(REPO / "out/papers"))
    ap.add_argument("--bands", nargs="+", default=["easy", "medium", "hard"])
    ap.add_argument("--per-band", type=int, default=2)
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    tax = Taxonomy.load("ssc_cgl")
    forecast = {r["topic"]: r["expected"]
                for r in json.load(open(args.forecast))["forecast"]}

    print("harvesting ...")
    pool = harvest(Path(args.raw))
    band_within_topic(pool)
    print(f"  {len(pool)} questions  " +
          "  ".join(f"{b}={n}" for b, n in Counter(q["difficulty"] for q in pool).most_common()))

    want = allocate(forecast, tax)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    used: set = set()
    made = []

    paper_no = 0
    for band in args.bands:
        band_pool = [q for q in pool if q["difficulty"] == band]
        by_topic = defaultdict(list)
        for q in band_pool:
            by_topic[q["topic"]].append(q)
        for lst in by_topic.values():
            rng.shuffle(lst)
            lst.sort(key=lambda q: -q["year"])   # recent pattern first

        for _ in range(args.per_band):
            paper_no += 1
            picked = take(by_topic, want, used, tax, band_pool or pool)
            if len(picked) < 60:
                print(f"  paper {paper_no} ({band}): only {len(picked)} questions — skipped")
                paper_no -= 1
                continue
            path = outdir / f"SSC_CGL_2026_Mock_{paper_no:02d}_{band}.pdf"
            render_pdf(picked, tax, path, paper_no, band, args.year)
            shifts = len({q["source"] for q in picked})
            made.append((path, band, len(picked), shifts))
            print(f"  paper {paper_no:>2} [{band:<6}] {len(picked):>3} questions "
                  f"from {shifts} different shifts -> {path.name}")

    print(f"\n{len(made)} papers -> {outdir}")


if __name__ == "__main__":
    main()
