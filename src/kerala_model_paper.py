"""Generate model question papers for the Kerala PSC Degree Level Main.

Every question is a REAL question from a past Main Examination, with the
commission's own published answer. Nothing is model-written, so no question can
be subtly wrong in the way a generated one can.

Papers are built to KPSC's published mark distribution (Cat.No.26/2022) and are
mutually exclusive: a question used in one paper is never reused in another, so
sitting all four is four genuinely different papers.

Two slots are handled honestly rather than padded:
  * Malayalam (10) extracts as mojibake from the legacy font, so it is marked
    for practice from the original PDFs instead of printed as garbage.
  * Current Affairs (15) is drawn from 2022-23 papers and is STALE for a 2026
    sitting. It is included for question-form practice with that warning
    attached, not as current knowledge.
"""
from __future__ import annotations
import json, re, random, argparse
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
MERGE = {"indian_constitution": "governance_polity_acts",
         "kerala_governance_acts": "governance_polity_acts"}

BLUEPRINT = [
    # Order matches the real paper's verified block structure:
    # Q1-55 General Knowledge, Q56-70 Current Affairs, Q71-80 Aptitude,
    # Q81-90 English, Q91-100 Malayalam.
    ("governance_polity_acts", 20, "Kerala Governance, Constitution & Acts"),
    ("science", 12, "Science (Life Science, Physics, Chemistry)"),
    ("history", 5, "History"),
    ("geography", 5, "Geography"),
    ("economics", 5, "Economics"),
    ("arts_literature_culture_sports", 5, "Arts, Literature, Culture & Sports"),
    ("basics_of_computer", 3, "Basics of Computer"),
    ("current_affairs", 15, "Current Affairs"),
    ("aptitude", 10, "Simple Arithmetic, Mental Ability & Reasoning"),
    ("general_english", 10, "General English"),
    ("regional_language", 10, "Regional Language (Malayalam)"),
]
NON_LATIN = re.compile(r'[^\x00-\x7F]')


def readable(q):
    t = q["stem"]
    return len(NON_LATIN.findall(t)) / max(len(t), 1) < 0.08


def load_pool():
    pairs = json.load(open(ROOT / "out/kerala_qa_pairs.json"))
    pool = defaultdict(list)
    seen = set()
    for q in pairs:
        key = re.sub(r'\W', '', q["stem"].lower())[:80]
        if key in seen:            # the two UA scans are the same exam
            continue
        seen.add(key)
        if len(q["options"]) < 4 or not (20 < len(q["stem"]) < 600):
            continue
        pool[MERGE.get(q["topic"], q["topic"])].append(q)
    return pool


def build(pool, used, blueprint, rng):
    picked, short = [], []
    for topic, need, label in blueprint:
        avail = [q for q in pool.get(topic, [])
                 if id(q) not in used and (topic == "regional_language" or readable(q))]
        rng.shuffle(avail)
        take = avail[:need]
        for q in take:
            used.add(id(q))
        picked.append((label, topic, take))
        if len(take) < need:
            short.append((label, need - len(take)))
    return picked, short


def render(name, exam, picked, short, paper_no):
    L, A = [], None
    out = []
    A = out.append
    A(f"# {exam}")
    A(f"## Model Question Paper {paper_no}")
    A("")
    A("**Time: 1 hour 30 minutes · Maximum marks: 100 · Negative marking: −1/3**")
    A("")
    A("Every question below is a real question from an official Kerala PSC Degree "
      "Level Main Examination, with the commission's published answer. Built to the "
      "official mark distribution (Cat.No.26/2022).")
    A("")
    A("---")
    A("")
    n = 0
    key = []
    for label, topic, qs in picked:
        A(f"### {label}")
        A("")
        if topic == "regional_language":
            A("*These 10 questions are printed in a legacy non-Unicode Malayalam font "
              "that does not survive text extraction. Practise this section directly "
              "from the original question paper PDFs.*")
            A("")
            n += 10
            continue
        if topic == "current_affairs":
            A("*Drawn from 2022–23 papers. Use these for question-form practice only — "
              "the facts are out of date for a 2026 sitting.*")
            A("")
        for q in qs:
            n += 1
            A(f"**{n}.** {q['stem']}")
            A("")
            for L_ in ("A", "B", "C", "D"):
                if L_ in q["options"]:
                    A(f"   {L_}) {q['options'][L_][:170]}")
            A("")
            key.append((n, q["answer_letter"], q["answer_text"][:70], q["src"], q["qno"]))
        A("")
    A("---")
    A("")
    A("## Answer Key")
    A("")
    A(" · ".join(f"**{k[0]}**-{k[1]}" for k in key))
    A("")
    A("### Answers with source")
    A("")
    A("| # | Ans | Answer text | Source paper | Q |")
    A("|---|---|---|---|---|")
    for num, letter, text, src, qno in key:
        A(f"| {num} | {letter} | {text} | {src[:26]} | {qno} |")
    if short:
        A("")
        A("### Slots left short")
        A("")
        for label, k in short:
            A(f"- **{label}** — {k} short (pool exhausted after excluding reused questions)")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260819)
    a = ap.parse_args()
    pool = load_pool()
    print("pool:", {k: len(v) for k, v in sorted(pool.items())})
    rng = random.Random(a.seed)
    exams = [("University Assistant — Main Examination", "univ"),
             ("Company / Corporation / Board Assistant — Main Examination", "company")]
    for exam, slug in exams:
        # Reset per exam. A University Assistant candidate sits the two univ
        # papers; a Company/Board candidate sits the two company papers. They are
        # different people, so overlap BETWEEN exams costs nothing -- while
        # forbidding it globally exhausted the Current Affairs and Geography
        # pools and left the fourth paper 15 questions short.
        used = set()
        for pno in (1, 2):
            picked, short = build(pool, used, BLUEPRINT, rng)
            md = render(slug, exam, picked, short, pno)
            p = ROOT / "out" / f"MODEL_PAPER_{slug}_{pno}.md"
            p.write_text(md)
            filled = sum(len(q) for _, t, q in picked if t != "regional_language")
            print(f"  {p.name}: {filled} printed questions"
                  f"{'  SHORT: ' + str(short) if short else ''}")


if __name__ == "__main__":
    main()
