"""Held-out coverage test: how much of an UNSEEN paper do the notes reach?

Method. Take one paper out. Build the note set from the remaining six. Then ask,
for every question in the held-out paper, whether the notes contain the fact
needed to answer it. Repeat for each paper (leave-one-out).

Two numbers are reported separately, because conflating them is how study guides
overstate themselves:

  TOPIC coverage    - the notes cover the area the question came from.
  ANSWER coverage   - the notes actually contain the fact that answers it.

A book saying "study Kerala geography" has the first and not the second. Only
the second is worth anything on exam day, and it is always the smaller number.

Matching is deterministic: the held-out question's official answer text is
compared against note text by token overlap. No model judges its own work.
"""
from __future__ import annotations
import json, re
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
STOP = set("""the a an of in on at to for and or is are was were be been which what who
whom whose how when where why not with by from as that this these those it its their his
her all any both each more most other some such only own same than too very can will just
following correct answer statement given below""".split())


def toks(s):
    return {w for w in re.findall(r'[a-z]{3,}', s.lower()) if w not in STOP}


def main():
    notes = json.load(open(ROOT / "out/kerala_study_notes.json"))
    pairs = json.load(open(ROOT / "out/kerala_qa_pairs.json"))
    TEACH = {n["topic"] for n in notes.values()}

    by_src = defaultdict(list)
    for uid, n in notes.items():
        by_src[n["src"]].append(n)

    # ua_main_160_2023 and ua_main_25082023_v2 are the SAME exam (25/08/2023):
    # 86% of stems are byte-identical at the same question number. Holding out
    # one leaves its twin in training, which scored a false 65-68% answer
    # coverage against 0-2% for genuinely unseen papers. They are held out as
    # one unit. This is the leakage the >63% ceiling was set to catch.
    GROUP = {"ua_main_160_2023.pdf": "UA_25082023",
             "ua_main_25082023_v2.pdf": "UA_25082023"}
    groups = defaultdict(list)
    for sname in by_src:
        groups[GROUP.get(sname, sname)].append(sname)
    papers = sorted(groups)
    print(f"{'held-out paper':<44}{'Qs':>4}{'topic%':>8}{'answer%':>9}")
    print("-" * 66)
    tt = ta = tq = 0
    for held in papers:
        held_srcs = set(groups[held])
        train = [n for s, ns in by_src.items() if s not in held_srcs for n in ns]
        train_topics = {n["topic"] for n in train}
        train_tok = [toks(n["note"]) for n in train]
        test = [q for q in pairs if q["src"] in held_srcs and q["topic"] in TEACH]
        if not test:
            continue
        topic_hit = answer_hit = 0
        for q in test:
            if q["topic"] in train_topics:
                topic_hit += 1
            at = toks(q["answer_text"] + " " + q["stem"])
            if not at:
                continue
            # answered if some note shares most of the answer's distinctive tokens
            best = max((len(at & nt) / len(at) for nt in train_tok), default=0)
            if best >= 0.60:
                answer_hit += 1
        n = len(test)
        tt += topic_hit; ta += answer_hit; tq += n
        print(f"{held[:42]:<44}{n:>4}{100*topic_hit/n:>7.0f}%{100*answer_hit/n:>8.0f}%")
    print("-" * 66)
    print(f"{'WEIGHTED MEAN':<44}{tq:>4}{100*tt/tq:>7.0f}%{100*ta/tq:>8.0f}%")
    print()
    print("TOPIC coverage  = the notes point you at the right area.")
    print("ANSWER coverage = the notes actually contain the fact. This is the honest one.")
    print()
    print("Scope: taught chapters only. Current Affairs, Malayalam, Aptitude and")
    print("English (45 of 100 marks) are excluded -- the notes never claimed them.")


if __name__ == "__main__":
    main()
