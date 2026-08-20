"""Complete question bank: every verified question, grouped by topic.

All 669 questions from the corpus with KPSC's official answers, ordered by the
topic's weight in the exam so the heaviest sections come first. Unlike the model
papers, nothing is sampled and nothing is withheld -- this is the whole corpus,
for drilling one topic at a time.
"""
from __future__ import annotations
import json, re
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
MERGE = {"indian_constitution": "governance_polity_acts",
         "kerala_governance_acts": "governance_polity_acts"}
ORDER = [
    ("governance_polity_acts", 20, "Kerala Governance, Constitution & Acts"),
    ("current_affairs", 15, "Current Affairs  (stale for 2026 — form practice only)"),
    ("science", 12, "Science — Life Science, Physics, Chemistry"),
    ("aptitude", 10, "Simple Arithmetic, Mental Ability & Reasoning"),
    ("general_english", 10, "General English"),
    ("regional_language", 10, "Regional Language (Malayalam) — legacy font, see original PDFs"),
    ("history", 5, "History"),
    ("geography", 5, "Geography"),
    ("economics", 5, "Economics"),
    ("arts_literature_culture_sports", 5, "Arts, Literature, Culture & Sports"),
    ("basics_of_computer", 3, "Basics of Computer"),
]
NON_LATIN = re.compile(r'[^\x00-\x7F]')


def main():
    pairs = json.load(open(ROOT / "out/kerala_qa_pairs.json"))
    by = defaultdict(list)
    seen = set()
    for q in pairs:
        k = re.sub(r'\W', '', q["stem"].lower())[:80]
        if k in seen:
            continue
        seen.add(k)
        by[MERGE.get(q["topic"], q["topic"])].append(q)

    out, A = [], None
    A = out.append
    total = sum(len(by.get(t, [])) for t, _, _ in ORDER)
    A("# Kerala PSC Degree Level Main — Complete Question Bank")
    A("")
    A(f"**{total} questions** from 7 official Degree Level Main Examination papers, "
      "each with the commission's published answer. Grouped by topic, heaviest "
      "section first.")
    A("")
    A("Duplicates removed (two source files are the same 25/08/2023 exam).")
    A("")
    A("---")
    A("")
    A("## Contents")
    A("")
    for t, marks, label in ORDER:
        A(f"- **{label}** — {marks} marks · {len(by.get(t, []))} questions")
    A("")
    A("---")
    A("")
    for t, marks, label in ORDER:
        qs = by.get(t, [])
        if not qs:
            continue
        A(f"## {label}")
        A(f"*{marks} marks in the exam · {len(qs)} questions here*")
        A("")
        if t == "regional_language":
            A("These extract as mojibake from the legacy non-Unicode font. They are "
              "listed for completeness; practise from the original PDFs.")
            A("")
        garbled = 0
        for i, q in enumerate(qs, 1):
            stem = q["stem"]
            if len(NON_LATIN.findall(stem)) / max(len(stem), 1) > 0.08:
                garbled += 1
                if t != "regional_language":
                    continue
            A(f"**{i}.** {stem}")
            A("")
            for L_ in ("A", "B", "C", "D"):
                if L_ in q["options"]:
                    mark = " ✓" if L_ == q["answer_letter"] else ""
                    A(f"   {L_}) {q['options'][L_][:170]}{mark}")
            A("")
            A(f"   <sub>Answer: **{q['answer_letter']}** · {q['src'][:30]} Q{q['qno']}</sub>")
            A("")
        if garbled and t != "regional_language":
            A(f"*({garbled} further questions in this topic are unreadable "
              "Malayalam renderings and are omitted.)*")
            A("")
        A("---")
        A("")
    p = ROOT / "out" / "KERALA_PSC_QUESTION_BANK.md"
    p.write_text("\n".join(out))
    print(f"{total} questions -> {p}")


if __name__ == "__main__":
    main()
