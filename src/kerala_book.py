"""Build the Kerala PSC Degree Level Main filtered textbook.

WHAT THIS BOOK IS. The examination reduced to the smallest body of material that
covers most of it, using KPSC's own published mark distribution as the skeleton
and 700 real past questions as evidence for what sits inside each slot.

WHAT IT REFUSES TO BE. Not a fact list to memorise. Cross-paper analysis of the
corpus shows specific facts almost never repeat -- "Article ___ of the
Constitution" recurs in 6 of 7 papers but never the same Article. Only the FORM
recurs. A book promising the actual questions would be lying, and this project
already measured that failure mode on SSC (0.00% specific-question realisation).

HONEST MEASUREMENT NOTE. This book reports TOPIC coverage -- whether it tells
you to study the area a question came from. It does NOT claim SOLUTION coverage
(whether it teaches enough to answer it), because that cannot be measured
without hand-marking held-out papers. The SSC book separated these two numbers
and so does this one.
"""
from __future__ import annotations
import json, re, glob, sys
from pathlib import Path
from collections import Counter, defaultdict
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kerala_parse import extract

ROOT = Path(__file__).resolve().parent.parent
MERGE = {"indian_constitution": "governance_polity_acts",
         "kerala_governance_acts": "governance_polity_acts"}

# Official published distribution, Cat.No.26/2022
WEIGHTS = [
    ("governance_polity_acts", 20, "Kerala Governance & Administration (10) + Indian Constitution (5) + Important Acts (5)"),
    ("current_affairs", 15, "Current Affairs"),
    ("science", 12, "Life Science & Public Health (6) + Physics (3) + Chemistry (3)"),
    ("aptitude", 10, "Simple Arithmetic, Mental Ability & Reasoning"),
    ("general_english", 10, "General English"),
    ("regional_language", 10, "Regional Language (Malayalam / Kannada / Tamil)"),
    ("history", 5, "History — Kerala, India, World"),
    ("geography", 5, "Geography"),
    ("economics", 5, "Economics"),
    ("arts_literature_culture_sports", 5, "Arts, Literature, Culture, Sports"),
    ("basics_of_computer", 3, "Basics of Computer"),
]

EFFICIENCY = [  # marks per 1000 chars of official syllabus text
    ("Life Science & Public Health", 6, 302, 19.9), ("Economics", 5, 271, 18.5),
    ("Chemistry", 3, 271, 11.1), ("Geography", 5, 845, 5.9),
    ("History", 5, 915, 5.5), ("Arts/Sports/Literature", 5, 1636, 3.1),
    ("Basics of Computer", 3, 1021, 2.9), ("Indian Constitution", 5, 1827, 2.7),
    ("Physics", 3, 1156, 2.6), ("Important Acts", 5, 4613, 1.1),
]


def load_corpus():
    tags = json.load(open(ROOT / "out/kerala_tagged_clean.json"))
    by_topic = defaultdict(list)
    for pdfname, qt in tags.items():
        hits = [f for f in glob.glob(str(ROOT / f"data/raw/kerala/*/{pdfname}"))
                if "_different" not in f and "_excluded" not in f]
        if not hits:
            continue
        qs = extract(hits[0])
        for qno, lab in qt.items():
            if not lab or int(qno) not in qs:
                continue
            by_topic[MERGE.get(lab, lab)].append(
                {"src": pdfname, "qno": int(qno), "text": qs[int(qno)]})
    return by_topic


def question_forms(qs):
    """Recurring FORMS, which is what actually repeats -- not facts."""
    pats = [
        (r'\bArticle\s+\d+', "Article ___ of the Constitution"),
        (r'\bmatch\s+list|list\s*[-–]\s*I\b', "Match List-I with List-II"),
        (r'\bAct,?\s+\d{4}', "Named Act with year"),
        (r'\bwho (?:is|was|were)\b', "Who is/was — person identification"),
        (r'\bwhich year|\bin which year|\bwhen was', "Year identification"),
        (r'\bfirst\b.{0,26}\b(?:in|of)\b', "First/earliest superlative"),
        (r'\bcapital|\bheadquarters', "Capital / headquarters"),
    ]
    out = Counter()
    for q in qs:
        for rx, name in pats:
            if re.search(rx, q["text"], re.I):
                out[name] += 1
    return out


def main():
    by_topic = load_corpus()
    total_q = sum(len(v) for v in by_topic.values())
    n_papers = len({q["src"] for v in by_topic.values() for q in v})
    L = []
    A = L.append

    A("# Kerala PSC — Degree Level Main Examination")
    A("## The Filtered Examination Book")
    A("")
    A("**University Assistant · Company / Corporation / Board Assistant**")
    A("")
    A(f"Built from **{total_q} real questions** across **{n_papers} official Degree Level "
      f"Main Examination papers**, organised on KPSC's own published mark distribution. "
      "Nothing here is leaked, insider, or model-invented. The power is in the filtering, "
      "and every filter is stated with its evidence.")
    A("")
    A("---")
    A("")
    A("## What this book will not do")
    A("")
    A("It will not give you the questions. Across all 7 papers, specific facts almost "
      "never repeat — *\"Article ___ of the Constitution\"* appears in **6 of 7 papers "
      "and never once the same Article*. Only the **form** recurs. Any book claiming to "
      "predict the actual questions is selling you something; this project already "
      "measured that failure directly on SSC CGL, where specific-question prediction "
      "realised **0.00%**.")
    A("")
    A("What repeats, and therefore what can be taught, is: the **weighting**, the "
      "**question forms**, and the **topic boundaries**.")
    A("")
    A("---")
    A("")
    A("## 1. The structure — verified twice, independently")
    A("")
    A("| Section | Marks | Paper positions |")
    A("|---|---|---|")
    A("| General Knowledge | 55 | Q1–55 |")
    A("| Current Affairs | 15 | ~Q56–70 |")
    A("| Simple Arithmetic, Mental Ability & Reasoning | 10 | Q71–80 |")
    A("| General English | 10 | Q81–90 |")
    A("| Regional Language | 10 | Q91–100 |")
    A("")
    A("100 questions · 100 marks · **−1/3 per wrong answer**.")
    A("")
    A("This is confirmed two ways: KPSC's official *Detailed Syllabus and Mark "
      "Distribution* (Cat.No.26/2022), and the block boundaries read directly off paper "
      "160/2023. Both agree on all four blocks. Coaching sites widely publish a 5×20 "
      "split — **that is wrong**.")
    A("")
    A("---")
    A("")
    A("## 2. The 10 marks nobody can study for")
    A("")
    A("KPSC's detailed syllabus lists its General Knowledge topics as (i) through (xi). "
      "The published document contains (i), (ii), (iii), (iv) — then jumps directly to "
      "(vi).")
    A("")
    A("**Section (v) is Kerala Governance & Administration. It is worth 10 marks — the "
      "single heaviest topic on the paper — and no detailed syllabus exists for it.**")
    A("")
    A("Every candidate studying from the official syllabus is blind on 10% of the exam. "
      "Past questions are the only available map of it, which is the single strongest "
      "reason this book exists.")
    A("")
    A("---")
    A("")
    A("## 3. Study effort per mark varies 18×")
    A("")
    A("Marks divided by the volume of syllabus KPSC specifies for them:")
    A("")
    A("| Topic | Marks | Syllabus chars | Marks per 1k chars |")
    A("|---|---|---|---|")
    for name, mk, ch, eff in EFFICIENCY:
        A(f"| {name} | {mk} | {ch:,} | **{eff}** |")
    A("")
    A("**Important Acts demands 4,613 characters of specified content for 5 marks — 18× "
      "worse than Life Science.** A candidate reading the syllabus front-to-back sinks a "
      "third of their reading time into 5 marks. Read Life Science, Economics and "
      "Chemistry first; they are short, fully specified, and worth 14 marks together.")
    A("")
    A("---")
    A("")
    A("## 4. Weighting, and what sits inside each slot")
    A("")
    for topic, marks, label in WEIGHTS:
        qs = by_topic.get(topic, [])
        A(f"### {label} — {marks} marks")
        A("")
        if topic == "current_affairs":
            A("**Do not study this from past papers.** These questions are time-bound: the "
              "corpus is 2022–23, and that news is simply wrong for a 2026 sitting. "
              "15 marks must come from a live current-affairs source. This book "
              "deliberately leaves the slot empty rather than fill it with stale material.")
            A("")
            continue
        if topic == "regional_language":
            A("**Not extractable.** In every paper this block is printed in a legacy "
              "non-Unicode font and does not survive text extraction. It cannot be taught "
              "from this corpus — practise from the original PDFs directly.")
            A("")
            continue
        A(f"*{len(qs)} questions in corpus.*")
        A("")
        forms = question_forms(qs)
        if forms:
            A("Recurring question forms:")
            A("")
            for f, c in forms.most_common(4):
                A(f"- {f} — *{c} of {len(qs)} questions*")
            A("")
        # Real questions, spread across different source papers so the sample
        # shows the topic's range rather than one paper's quirks.
        seen_src, samples = set(), []
        for q in sorted(qs, key=lambda x: (x["src"], x["qno"])):
            if q["src"] in seen_src and len(samples) < 4:
                continue
            seen_src.add(q["src"]); samples.append(q)
            if len(samples) == 4:
                break
        if samples:
            A("**Actual questions from this slot:**")
            A("")
            for q in samples:
                t = q["text"]
                t = t[:300] + ("..." if len(t) > 300 else "")
                A(f"> {t}")
                A(f">")
                A(f"> <sub>{q['src'][:34]} Q{q['qno']}</sub>")
                A("")
    A("---")
    A("")
    A("## 5. Honest limits")
    A("")
    A("- **University Assistant Main has been held once**, on 25/08/2023. The other six "
      "papers are same-blueprint degree-level Mains used to model the shared design. "
      "They are not University Assistant questions and are not presented as such.")
    A("- **Two typist papers were excluded** after measurement: they carried 42 and 46 "
      "computer questions against an expected 3. Typist posts follow a different "
      "blueprint despite sitting at the same tier.")
    A("- **Secretariat / KPSC Assistant Main is excluded** — it is Paper I + Paper II "
      "with no Current Affairs block. Different exam.")
    A("- **Topic labelling is ~83% accurate**, measured against KPSC's published "
      "distribution. Current Affairs is systematically undercounted because a 2023 news "
      "item reads as static fact to a classifier.")
    A("- This book reports **topic coverage, not solution coverage**. It tells you where "
      "the marks are; it does not claim to teach enough to answer every question there.")

    out = ROOT / "out" / "KERALA_PSC_MAIN_FILTERED_BOOK.md"
    out.write_text("\n".join(L))
    print(f"corpus: {total_q} questions, {n_papers} papers")
    print("written ->", out)


if __name__ == "__main__":
    main()
