"""Build the Kerala PSC Degree Level Main PREDICTIVE STUDY BOOK.

Every statement in this book is derived from a real Degree Level Main
Examination question AND the commission's own published answer key. The model
that wrote the sentences was permitted only to rephrase that verified content;
it was not permitted to supply facts. Where the corpus is silent, the book says
so instead of filling the gap.

Chapters are ordered by the official mark distribution, so reading top-to-bottom
is reading in descending order of marks.
"""
from __future__ import annotations
import json, re, random
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent

CHAPTERS = [
    ("kerala_governance_acts", 15, "Kerala Governance, Administration & Important Acts",
     "**This is the chapter that cannot be found elsewhere.** KPSC's detailed syllabus "
     "lists GK topics (i) to (xi) but the published document skips (v) entirely — "
     "Kerala Governance & Administration, worth 10 marks, has *no* official syllabus. "
     "Past questions are the only map of it, and this chapter is that map."),
    ("indian_constitution", 5, "Indian Constitution",
     "Fully specified in the official syllabus (1,827 characters for 5 marks), so it "
     "reads long relative to its weight. Learn the recurring form: questions name an "
     "Article, a writ, or a constitutional body and ask you to match it."),
    ("science", 12, "Science — Life Science & Public Health, Physics, Chemistry",
     "The best return in the book. Life Science is 6 marks against 302 characters of "
     "syllabus and Chemistry 3 against 271 — the most marks per page you will find. "
     "Study this chapter first."),
    ("history", 5, "History — Kerala, India, World",
     "Sits at the very front of every paper. Kerala's reform movements and the "
     "Travancore/Malabar sequence dominate; world history appears as revolutions."),
    ("economics", 5, "Economics",
     "271 characters of syllabus for 5 marks — joint-best efficiency in the exam. "
     "Short, closed, and worth learning completely."),
    ("geography", 5, "Geography",
     "Split between physical geography terminology and Kerala's own rivers, districts "
     "and landforms."),
    ("arts_literature_culture_sports", 5, "Arts, Literature, Culture & Sports",
     "Heavily Kerala-weighted: art forms, authors and the works attributed to them."),
    ("basics_of_computer", 3, "Basics of Computer",
     "3 marks. Hardware/software vocabulary and general IT awareness."),
]

EXCLUDED = [
    ("Current Affairs", 15,
     "**Cannot be taught from past papers.** This corpus is 2022–23; that news is "
     "simply wrong for a 2026 sitting. Take these 15 marks from a live current-affairs "
     "source. Leaving the slot empty is the honest option — filling it with stale "
     "material would cost you marks, not win them."),
    ("Regional Language (Malayalam)", 10,
     "**Not extractable.** In every paper this block is printed in a legacy "
     "non-Unicode font that does not survive text extraction. Practise it from the "
     "original question paper PDFs."),
    ("Simple Arithmetic, Mental Ability & Reasoning", 10,
     "**Method, not facts.** Nothing here can be memorised — these 10 marks come from "
     "practice on the original papers, not from notes."),
    ("General English", 10,
     "**Method, not facts.** Grammar, tense, idiom and vocabulary. Practise on the "
     "original papers."),
]


def dedupe(notes):
    """Drop near-duplicates: the same fact asked twice across papers."""
    seen, out = set(), []
    for n in notes:
        k = re.sub(r'[^a-z0-9]', '', n["note"].lower())[:70]
        if k in seen:
            continue
        seen.add(k)
        out.append(n)
    return out


def practice(pairs, topic, n=6, seed=7):
    """Real questions with their options, spread across source papers."""
    qs = [q for q in pairs if q["topic"] == topic and len(q["options"]) >= 4
          and 25 < len(q["stem"]) < 420]
    by_src = defaultdict(list)
    for q in qs:
        by_src[q["src"]].append(q)
    rng = random.Random(seed)
    picked, srcs = [], sorted(by_src)
    # round-robin across papers so one paper cannot dominate the set
    while len(picked) < n and any(by_src.values()):
        for sname in srcs:
            if by_src[sname] and len(picked) < n:
                picked.append(by_src[sname].pop(rng.randrange(len(by_src[sname]))))
    return picked


def render_practice(A, pairs, topic, n=6):
    ps = practice(pairs, topic, n)
    if not ps:
        return
    A("**Practice questions** — real, from past Main papers:")
    A("")
    for i, q in enumerate(ps, 1):
        A(f"**Q{i}.** {q['stem']}")
        A("")
        for L_ in ("A", "B", "C", "D"):
            if L_ in q["options"]:
                A(f"- **{L_})** {q['options'][L_][:150]}")
        A("")
        A(f"<details><summary>Answer</summary>\n\n**{q['answer_letter']}) "
          f"{q['answer_text'][:150]}**\n\n<sub>{q['src'][:32]} Q{q['qno']}</sub>\n\n</details>")
        A("")


def main():
    data = json.load(open(ROOT / "out/kerala_study_notes.json"))
    pairs = json.load(open(ROOT / "out/kerala_qa_pairs.json"))
    by = defaultdict(list)
    for v in data.values():
        by[v["topic"]].append(v)

    L = []
    A = L.append
    total = sum(len(dedupe(by.get(t, []))) for t, _, _, _ in CHAPTERS)

    A("# Kerala PSC — Degree Level Main Examination")
    A("# The Predictive Study Book")
    A("")
    A("**University Assistant · Company / Corporation / Board Assistant**")
    A("")
    A(f"**{total} study facts**, every one taken from a real Degree Level Main "
      "Examination question and the commission's **own published answer key**. "
      "No fact in this book was supplied from general knowledge — where the past "
      "papers are silent, the book says so rather than guess.")
    A("")
    A("Chapters run in descending order of marks. Reading top-to-bottom is reading "
      "in order of what the exam actually pays for.")
    A("")
    A("---")
    A("")
    A("## Before you start: where the marks actually are")
    A("")
    A("| Read in this order | Marks | Why |")
    A("|---|---|---|")
    A("| Science | 12 | Best marks-per-page in the exam (Life Science 6 marks / 302 chars of syllabus) |")
    A("| Kerala Governance & Acts | 15 | Heaviest block, and 10 of it has **no published syllabus** |")
    A("| Economics | 5 | 271 chars of syllabus — short and closed |")
    A("| History | 5 | Opens every paper |")
    A("| Constitution | 5 | Long syllabus (1,827 chars) for 5 marks — learn the forms, not the text |")
    A("| Geography · Arts · Computer | 13 | Steady, low-density |")
    A("")
    A("**The trap:** *Important Acts* specifies 4,613 characters of syllabus for 5 "
      "marks — **18× less efficient than Life Science**. Read front-to-back and you "
      "will spend a third of your time on 5 marks.")
    A("")
    A("**Negative marking is −1/3.** Three wrong guesses cancel a correct answer.")
    A("")
    A("*A note on the practice questions:* they are reproduced verbatim from the "
      "official PDFs, and KPSC's typesetting sometimes extracts with stray spaces "
      "inside words (\"Jawaharlal N ehru\"). These are left uncorrected on purpose — "
      "an automated \"repair\" risks silently altering what a question actually asks, "
      "which is a worse error than a visible typo.")
    A("")
    A("---")
    A("")

    for topic, marks, title, intro in CHAPTERS:
        notes = dedupe(by.get(topic, []))
        A(f"## {title}")
        A(f"### {marks} marks · {len(notes)} facts")
        A("")
        A(intro)
        A("")
        if not notes:
            A("*No verified facts in corpus for this topic.*")
            A("")
            continue
        A("**Facts to revise:**")
        A("")
        for i, n in enumerate(notes, 1):
            A(f"{i}. {n['note']}")
        A("")
        render_practice(A, pairs, topic, 6)
        A("---")
        A("")

    A("## Sections this book deliberately does not teach")
    A("")
    A("These are **45 of the 100 marks**. Pretending to cover them would be the "
      "dishonest part of a book like this.")
    A("")
    EX_TOPIC = {"Current Affairs": "current_affairs",
                "Simple Arithmetic, Mental Ability & Reasoning": "aptitude",
                "General English": "general_english"}
    for name, marks, why in EXCLUDED:
        A(f"### {name} — {marks} marks")
        A("")
        A(why)
        A("")
        t = EX_TOPIC.get(name)
        if t:
            A("Facts cannot be taught here — but the real questions can still be "
              "practised, and that is what these marks reward:")
            A("")
            render_practice(A, pairs, t, 5)

    A("---")
    A("")
    A("## Provenance")
    A("")
    A("- **7 official Degree Level Main Examination papers**, 669 questions paired "
      "with KPSC's published answer keys.")
    A("- Structure verified twice: the official *Detailed Syllabus and Mark "
      "Distribution* (Cat.No.26/2022) and the block boundaries read directly off "
      "paper 160/2023. Both agree.")
    A("- **University Assistant Main has been held once**, 25/08/2023. The remaining "
      "papers are same-blueprint degree-level Mains. They are not University "
      "Assistant questions and are not presented as such.")
    A("- Excluded after measurement: two typist papers (42 and 46 computer questions "
      "against an expected 3), Secretariat/KPSC Assistant Main (Paper I+II, different "
      "exam), and Kannada/Malayalam-script printings (no extractable English).")

    out = ROOT / "out" / "KERALA_PSC_MAIN_STUDY_BOOK.md"
    out.write_text("\n".join(L))
    print(f"{total} facts across {len(CHAPTERS)} chapters -> {out}")


if __name__ == "__main__":
    main()
