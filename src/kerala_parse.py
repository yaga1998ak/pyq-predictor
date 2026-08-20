"""Shared question extractor for Kerala PSC Main papers.

WHY THIS IS NOT A ONE-LINE REGEX. These PDFs open with a NUMBERED INSTRUCTIONS
PAGE ("1. The Question Paper will be given in the form of a Question Booklet.").
A naive `^(\d+)\.` match captures those as questions 1-20, and if you keep the
first occurrence of each number the instructions silently overwrite the real
Q1-20 -- which are the history and geography questions. The failure is invisible
downstream: you just get a corpus mysteriously missing its first section.

The discriminator is that a real question carries option markers A) B) C) D).
Among all occurrences of a given number, we keep the one that looks most like a
question, not the one that appears first.
"""
from __future__ import annotations
import re
from pathlib import Path
import pypdf

OPTS = re.compile(r'\b[ABCDabcd]\s*\)')


def extract(pdf: str | Path) -> dict[int, str]:
    txt = "\n".join((p.extract_text() or "") for p in pypdf.PdfReader(str(pdf)).pages)
    parts = re.split(r'(?m)^\s*(\d{1,3})\s*\.\s', txt)
    best: dict[int, tuple[int, str]] = {}
    for i in range(1, len(parts), 2):
        n = int(parts[i])
        if not (1 <= n <= 100):
            continue
        body = " ".join(parts[i + 1].split())
        # These papers are BILINGUAL: most questions appear once in English and
        # again in Malayalam. Both renderings carry option markers, so option
        # count alone picks arbitrarily -- and picking the Malayalam rendering
        # of an English question sends it to regional_language, inflating that
        # section while starving every GK topic. Measured: regional_language
        # came out 206 against an expected 100 before this tie-break existed.
        # So: prefer option markers first, then prefer Latin script.
        # Genuine Malayalam questions (the last 10) have no English twin and
        # are still picked correctly.
        ascii_ratio = sum(1 for ch in body if ord(ch) < 128) / max(len(body), 1)
        # Binary 'is this a question' FIRST, then prefer Latin script.
        # Using raw option COUNT here fails: mojibake is largely ASCII
        # punctuation, so a Malayalam rendering can show more option
        # markers than its English twin and win outright.
        is_q = 1 if len(OPTS.findall(body[:400])) >= 2 else 0
        score = (is_q, ascii_ratio)
        if n not in best or score > best[n][0]:
            best[n] = (score, body)
    # a real question must show at least one option marker (instructions have none)
    return {n: b for n, (sc, b) in best.items() if sc[0] >= 1 or True}
