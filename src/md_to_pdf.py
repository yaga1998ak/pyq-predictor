"""Render the handover Markdown to a printable PDF.

Deliberately a small subset of Markdown -- headings, tables, lists, bold/italic,
code, blockquotes -- because the input is one known document, not arbitrary
Markdown. A full parser would be more code and more failure modes for no gain.

    python src/md_to_pdf.py HANDOVER.md out/HANDOVER.pdf
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

INK = colors.HexColor("#12263f")
MUTED = colors.HexColor("#5b6b7f")
RULE = colors.HexColor("#d8e0e9")
ACCENT = colors.HexColor("#1c5d99")
BAND = colors.HexColor("#eef3f8")


def font() -> tuple[str, str]:
    """Return (regular, bold). Unicode TTF required: the corpus has ₹ × ÷ – ‘ '."""
    uni = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
    if uni.exists():
        pdfmetrics.registerFont(TTFont("Uni", str(uni)))
        return "Uni", "Helvetica-Bold"
    return "Helvetica", "Helvetica-Bold"


def inline(s: str) -> str:
    """Markdown inline → ReportLab mini-HTML. Escape first, then add markup."""
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    s = re.sub(r"`([^`]+)`", r'<font face="Courier" size="8.5">\1</font>', s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", s)
    return s


def build(md_path: Path, pdf_path: Path) -> None:
    reg, bold = font()
    ss = getSampleStyleSheet()
    S = {
        "h1": ParagraphStyle("h1", parent=ss["Title"], fontName=bold, fontSize=19,
                             textColor=INK, spaceAfter=4, alignment=0),
        "h2": ParagraphStyle("h2", parent=ss["Heading2"], fontName=bold, fontSize=12.5,
                             textColor=ACCENT, spaceBefore=15, spaceAfter=6),
        "h3": ParagraphStyle("h3", parent=ss["Heading3"], fontName=bold, fontSize=10.5,
                             textColor=INK, spaceBefore=9, spaceAfter=4),
        "p": ParagraphStyle("p", parent=ss["Normal"], fontName=reg, fontSize=9.2,
                            leading=13.2, textColor=INK, spaceAfter=6),
        "li": ParagraphStyle("li", parent=ss["Normal"], fontName=reg, fontSize=9.2,
                             leading=13, textColor=INK, leftIndent=12, spaceAfter=3,
                             bulletIndent=3),
        "quote": ParagraphStyle("q", parent=ss["Normal"], fontName=reg, fontSize=9,
                                leading=13, textColor=colors.HexColor("#7a4b00"),
                                leftIndent=10, borderPadding=6, spaceAfter=7,
                                backColor=colors.HexColor("#fff8e6")),
        "code": ParagraphStyle("c", parent=ss["Normal"], fontName="Courier", fontSize=8,
                               leading=10.5, textColor=colors.HexColor("#1a3a52"),
                               backColor=colors.HexColor("#f4f7fa"), leftIndent=6,
                               borderPadding=6, spaceAfter=8),
        "cell": ParagraphStyle("cell", parent=ss["Normal"], fontName=reg, fontSize=8.2,
                               leading=11, textColor=INK),
        "cellh": ParagraphStyle("ch", parent=ss["Normal"], fontName=bold, fontSize=8.2,
                                leading=11, textColor=colors.white),
    }

    flow: list = []
    lines = md_path.read_text().splitlines()
    i = 0
    while i < len(lines):
        ln = lines[i]

        if ln.startswith("```"):
            buf = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                buf.append(lines[i].replace("&", "&amp;").replace("<", "&lt;"))
                i += 1
            flow.append(Paragraph("<br/>".join(buf) or " ", S["code"]))
            i += 1
            continue

        if ln.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:|-]+\|$", lines[i + 1]):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                if not re.match(r"^\|[\s:|-]+\|$", lines[i]):
                    rows.append([c.strip() for c in lines[i].strip("|").split("|")])
                i += 1
            ncol = max(len(r) for r in rows)
            data = [[Paragraph(inline(c), S["cellh"] if r == 0 else S["cell"])
                     for c in row + [""] * (ncol - len(row))]
                    for r, row in enumerate(rows)]
            avail = 178 * mm
            t = Table(data, colWidths=[avail / ncol] * ncol, repeatRows=1)
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BAND]),
                ("GRID", (0, 0), (-1, -1), 0.25, RULE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            flow += [t, Spacer(1, 8)]
            continue

        if ln.startswith("# "):
            flow.append(Paragraph(inline(ln[2:]), S["h1"]))
        elif ln.startswith("## "):
            flow.append(Paragraph(inline(ln[3:]), S["h2"]))
        elif ln.startswith("### "):
            flow.append(Paragraph(inline(ln[4:]), S["h3"]))
        elif ln.startswith("> "):
            flow.append(Paragraph(inline(ln[2:]), S["quote"]))
        elif re.match(r"^\s*[-*] ", ln):
            flow.append(Paragraph(inline(re.sub(r"^\s*[-*] ", "", ln)), S["li"], bulletText="•"))
        elif re.match(r"^\s*\d+\. ", ln):
            m = re.match(r"^\s*(\d+)\. (.*)", ln)
            flow.append(Paragraph(inline(m.group(2)), S["li"], bulletText=f"{m.group(1)}."))
        elif ln.strip() == "---":
            flow.append(Spacer(1, 6))
        elif ln.strip():
            flow.append(Paragraph(inline(ln), S["p"]))
        else:
            flow.append(Spacer(1, 3))
        i += 1

    doc = SimpleDocTemplate(
        str(pdf_path), pagesize=A4,
        leftMargin=16 * mm, rightMargin=16 * mm,
        topMargin=15 * mm, bottomMargin=15 * mm,
        title="SSC CGL Tier-1 2026 — Prediction Project Handover",
    )

    def footer(canvas, _doc):
        canvas.saveState()
        canvas.setFont(reg, 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(16 * mm, 9 * mm, "SSC CGL Tier-1 2026 — Prediction Project Handover")
        canvas.drawRightString(A4[0] - 16 * mm, 9 * mm, f"{canvas.getPageNumber()}")
        canvas.setStrokeColor(RULE)
        canvas.line(16 * mm, 12 * mm, A4[0] - 16 * mm, 12 * mm)
        canvas.restoreState()

    doc.build(flow, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "HANDOVER.md")
    dst = Path(sys.argv[2] if len(sys.argv) > 2 else "out/HANDOVER.pdf")
    dst.parent.mkdir(parents=True, exist_ok=True)
    build(src, dst)
    print(f"-> {dst}")
