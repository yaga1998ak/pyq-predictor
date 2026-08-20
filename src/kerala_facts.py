"""Pair every corpus question with KPSC's OFFICIAL answer.

This is the grounding layer for the textbook. A fact reaches the book only if
it is carried by a real Main Examination question AND the commission's own
published answer key. No fact is supplied from model knowledge -- which matters
most here, because Kerala-specific content (reform leaders, state schemes, local
geography) is exactly where a language model invents plausible, wrong detail.

Alphacode note: each paper is printed in four versions (a/b/c/d) with options
shuffled. The downloaded PDFs are alphacode 'a' and the downloaded keys are
'ALPHACODE A', so they correspond. Verified by spot-check: UA 160/2023 Q1 keys
to A) "Pereira de Paiva, moses", who did head the 1686 Amsterdam delegation.
"""
from __future__ import annotations
import json, re, glob, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kerala_parse import extract

ROOT = Path(__file__).resolve().parent.parent
KEYMAP = {
    "ua_main_160_2023.pdf": "25082023_universities.pdf",
    "ua_main_25082023_v2.pdf": "25082023_universities.pdf",
    "dlme_assistant_076_2023.pdf": "22062023_assistant_gr_ii.pdf",
    "09062023_junior_assistant_cashier_assistant_gr_ii_cle.pdf": "09062023_junior_assistant.pdf",
    "23122022_assistant_director_of_national_savings_degre.pdf": "23122022_national_savings.pdf",
    "27122022_assistant_degree_level_main_examination_kera.pdf": "27122022_administrative.pdf",
    "29102024_assistant_manager_kscb_direct_and_b_t_main_e.pdf": "29102024_kscb.pdf",
}
OPT = re.compile(r'\b([ABCDabcd])\s*\)')


def split_options(text: str):
    """Return (stem, {A: opt, B: opt, ...}) or (text, {}) if not parseable."""
    parts = OPT.split(text)
    if len(parts) < 5:
        return text, {}
    stem = parts[0].strip()
    opts = {}
    for i in range(1, len(parts) - 1, 2):
        letter = parts[i].upper()
        val = parts[i + 1].strip()
        # Strip the printed page footer, which extracts into the trailing option:
        #   'T. Ganapati Sastri 060/23 - K a -19-'
        # Any alphacode letter can appear, not just M, and a bare '-19-' page
        # marker can trail on its own.
        val = re.sub(r'\s*\d{2,3}/\d{2}\s*[–—-]\s*\w.*$', '', val)
        val = re.sub(r'\s*-\s*\d{1,3}\s*-\s*$', '', val).strip()
        if letter not in opts:
            opts[letter] = val
    return stem, opts


def build():
    tags = json.load(open(ROOT / "out/kerala_tagged_clean.json"))
    keys = json.load(open(ROOT / "data/raw/kerala/keys/answer_maps.json"))
    out, missing = [], 0
    for pdfname, qt in tags.items():
        hits = [f for f in glob.glob(str(ROOT / f"data/raw/kerala/*/{pdfname}"))
                if "_different" not in f and "_excluded" not in f]
        if not hits:
            continue
        qs = extract(hits[0])
        keyfile = KEYMAP.get(pdfname)
        ans = keys.get(keyfile, {}) if keyfile else {}
        for qno, label in qt.items():
            n = int(qno)
            if not label or n not in qs:
                continue
            stem, opts = split_options(qs[n])
            letter = ans.get(qno)
            if not letter or letter not in opts:
                missing += 1
                continue
            out.append({"src": pdfname, "qno": n, "topic": label,
                        "stem": stem, "answer_letter": letter,
                        "answer_text": opts[letter], "options": opts})
    json.dump(out, open(ROOT / "out/kerala_qa_pairs.json", "w"), indent=1)
    print(f"verified Q+A pairs: {len(out)}   unresolved: {missing}")
    from collections import Counter
    for t, c in Counter(q["topic"] for q in out).most_common():
        print(f"   {t:<34}{c}")


if __name__ == "__main__":
    build()
