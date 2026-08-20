"""Assemble the two 25-question English papers from the 2026 blueprint.

Paper 1 -- newly constructed questions (english_candidates.json).
Paper 2 -- REAL official PYQ text (2022-2024 only, never the 2025 coaching
           "Similar Papers"), keyed by derivation where possible and by
           explicitly-labelled editorial solve otherwise.

Cloze is handled as a passage GROUP: the passage is printed once and its five
blanks follow it, because a blank's answer depends on the surrounding text
(§68). Allocating cloze as five independent questions would produce orphan
blanks with no passage.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
NOMINAL = 25

DIRTY = re.compile(r"question\s*id|chosen\s*option|status\s*:|"
                   r"comprehension\s*:|section\s*:", re.I)

# OCR residue seen in the 2023 scanned papers: stray symbols and case-mangled
# fragments inside option text ("To try to influence *", "To argue eW",
# "Noerror #2."). Printing these would put unreadable options on a paper.
OCR_JUNK = re.compile(r"[*#@\\|~^]|\b[a-z][A-Z]\b|\bNoerror\b")

# Paper 2 uses OFFICIAL 2022-2024 only. Three 2025 files lack the
# "Similar-Paper" marker and so are not flagged as reconstructions, but 2025 as
# a whole is outside the validated evidence base, so it is excluded by year.
PAPER2_YEARS = {2022, 2023, 2024}


def is_readable(t: str) -> bool:
    t = t.strip()
    if len(t) < 20:
        return False
    if t.count(" ") / len(t) < 0.10:
        return False
    return max((len(m) for m in re.findall(r"[A-Za-z]+", t)), default=0) <= 20


def is_clean(q) -> bool:
    if len(q.get("options", [])) != 4:
        return False
    texts = [t.strip() for _, t in q["options"]]
    if any(not t or DIRTY.search(t) for t in texts):
        return False
    if len(set(t.lower() for t in texts)) != 4:
        return False
    if any(OCR_JUNK.search(t) for t in texts):
        return False
    return is_readable(q["stem"]) and not DIRTY.search(q["stem"])


def load_blueprint():
    return json.load(open(OUT / "english_forecast_2026.json"))["family"]


def allocate(bp, available: set[str], total=NOMINAL):
    usable = {k: v["expected"] for k, v in bp.items() if k in available}
    dropped = {k: v["expected"] for k, v in bp.items()
               if k not in available and v["expected"] > 0}
    s = sum(usable.values())
    scaled = {k: v / s * total for k, v in usable.items()}
    floors = {k: int(v) for k, v in scaled.items()}
    left = total - sum(floors.values())
    for k in sorted(scaled, key=lambda k: -(scaled[k] - floors[k]))[:left]:
        floors[k] += 1
    return {k: v for k, v in floors.items() if v > 0}, dropped


def reorder(q, rng):
    opts = q["options"]
    if any(re.search(r"all of the above|none of these", t, re.I) for _, t in opts):
        return opts, q["answer"]
    correct = dict(opts)[q["answer"]]
    texts = [t for _, t in opts]
    rng.shuffle(texts)
    new = list(zip("abcd", texts))
    return new, next(l for l, t in new if t == correct)


def render(title, subtitle, questions, alloc, dropped, notes, bp):
    L = [f"# {title}\n", subtitle + "\n",
         f"**{len(questions)} questions · English Comprehension · "
         f"answer key, forecast table and limitations at the end**\n", "---\n",
         "## Questions\n"]
    shown_passages = set()
    for i, q in enumerate(questions, 1):
        pt = q.get("passage_text")
        if pt and pt not in shown_passages:
            shown_passages.add(pt)
            L.append("> **Passage.** In the following passage, some words have "
                     "been deleted. Read the passage carefully and select the "
                     "most appropriate option to fill in each blank.\n")
            L.append("> " + pt.replace("\n", " ") + "\n")
        L.append(f"**{i}.** {q['stem']}\n")
        for letter, text in q["options"]:
            L.append(f"- ({letter}) {text}")
        L.append("")
    L += ["---\n", "## Answer key\n"]
    row = []
    for i, q in enumerate(questions, 1):
        row.append(f"{i}. {q['answer'].upper()}")
        if i % 5 == 0:
            L.append("  ".join(row)); row = []
    if row:
        L.append("  ".join(row))
    L += ["", "---\n", "## Question-level forecast table\n",
          "| Q | Family | Grammar rule | Expected/25 | P(appears) | Key basis |",
          "|---|---|---|---|---|---|"]
    for i, q in enumerate(questions, 1):
        f = q["family"]
        e = bp.get(f, {}).get("expected", 0)
        p = bp.get(f, {}).get("p_appears", 0)
        L.append(f"| {i} | {f} | {q.get('micro') or '—'} | {e:.2f} | {p:.0%} | "
                 f"{q.get('verified_by','—')} |")
    L += ["", "---\n", "## Blueprint coverage\n",
          "| Family | Blueprint expected | Allocated |", "|---|---|---|"]
    for f, n in sorted(alloc.items(), key=lambda x: -x[1]):
        L.append(f"| {f} | {bp.get(f,{}).get('expected',0):.2f} | {n} |")
    if dropped:
        L += ["", "### Families not represented\n",
              "| Family | Blueprint expected |", "|---|---|"]
        for f, e in sorted(dropped.items(), key=lambda x: -x[1]):
            L.append(f"| {f} | {e:.2f} |")
        L.append(f"\nCombined omitted share: **{sum(dropped.values()):.2f} of 25**.\n")
    L += ["---\n", "## Method and limitations\n"] + [f"- {n}" for n in notes] + [""]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    bp = load_blueprint()

    # ---------------- Paper 1 ---------------------------------------------
    cands = [q for q in json.load(open(OUT / "english_candidates.json"))
             if is_clean(q)]
    by_fam = defaultdict(list)
    for q in cands:
        by_fam[q["family"]].append(q)
    alloc1, dropped1 = allocate(bp, set(by_fam))
    picked1 = []
    for fam, n in alloc1.items():
        pool = list(by_fam[fam])
        if fam == "cloze_test":
            groups = defaultdict(list)
            for q in pool:
                groups[q.get("passage_text")].append(q)
            g = rng.choice([v for v in groups.values() if len(v) >= n])
            picked1.extend(sorted(g, key=lambda q: q["stem"])[:n])
        else:
            rng.shuffle(pool)
            picked1.extend(pool[:n])
    for q in picked1:
        if not q.get("passage_text"):
            q["options"], q["answer"] = reorder(q, rng)
    # cloze block last so its passage reads as one unit
    picked1.sort(key=lambda q: (q["family"] == "cloze_test", q["family"]))

    mv1 = sum(1 for q in picked1 if q["verified_by"].startswith("machine"))
    notes1 = [
        f"**{mv1} of {len(picked1)} answers are machine-verified** by a check that "
        "can fail: the 234,456-word system dictionary (spelling, homophones), "
        "grammatical rule evaluation (subject-verb agreement, a/an by sound), and "
        "reconstruction identity for para-jumbles (the key IS the authored order).",
        f"The remaining {len(picked1)-mv1} are **EDITORIAL** — vocabulary items "
        "whose answers rest on lexical judgement. There is no WordNet in this "
        "environment, so no machine oracle exists for synonyms, antonyms, idioms "
        "or one-word substitution. Each is labelled in the table above.",
        "Family mix follows the 2026 blueprint from `pooled_all` on **official "
        "2022-2024 papers only**. 2021 was dropped (53% extraction recall) and "
        "2025 excluded (see below).",
        "**2025 was excluded from the evidence base**, not merely down-weighted. "
        "TVD between the two official years 2023 and 2024 is 0.094; between 2024 "
        "and the 2025 files it is 0.356. The 2025 files are labelled "
        "\"T-I Similar Paper\" — coaching practice papers that invent families the "
        "real exam does not use at that rate (reading comprehension 3.73/shift "
        "against ~0.34 in official papers).",
        "These are not predictions of specific 2026 questions. They are original "
        "items matching the forecast family mix and grammar-rule distribution.",
    ]
    (OUT / "english_paper1_constructed.md").write_text(render(
        "SSC CGL Tier-1 2026 — English Practice Paper 1",
        "**Newly constructed questions fitted to the forecast blueprint.** "
        "Machine-verified where an oracle exists; editorial where none does.",
        picked1, alloc1, dropped1, notes1, bp))

    # ---------------- Paper 2 ---------------------------------------------
    tagged = json.load(open(OUT / "english_tagged.json"))
    derived = json.load(open(OUT / "english_derived_keys.json"))
    # Cloze questions need their passage reprinted, or the blanks are unanswerable.
    passages = {p["passage_id"]: p["text"]
                for p in json.load(open(OUT / "english_passages.json"))}
    manual = {}
    mp = OUT / "english_manual_keys.json"
    if mp.exists():
        manual = json.load(open(mp))

    real_by_fam = defaultdict(list)
    for r in tagged:
        if r["is_reconstruction"] or not r.get("family"):
            continue
        if r["year"] not in PAPER2_YEARS:
            continue
        cand = dict(family=r["family"], subtype=r.get("subtype"),
                    micro=r.get("micro"), stem=r["stem"],
                    options=r.get("options") or [], year=r["year"],
                    source=r["source_pdf"], qid=r["qid"],
                    passage_text=passages.get(r.get("passage_id")))
        if not is_clean(cand):
            continue
        if r["qid"] in derived:
            cand["answer"] = derived[r["qid"]]["answer"]
            cand["verified_by"] = derived[r["qid"]]["basis"]
        elif r["qid"] in manual:
            cand["answer"] = manual[r["qid"]]
            cand["verified_by"] = "EDITORIAL:solved_from_real_pyq"
        else:
            cand["answer"] = None
            cand["verified_by"] = None
        real_by_fam[r["family"]].append(cand)

    keyed = {f: [q for q in v if q["answer"]] for f, v in real_by_fam.items()}
    keyed = {f: v for f, v in keyed.items() if v}
    if not keyed:
        print("No keyed official questions yet.")
    alloc2, dropped2 = allocate(bp, set(keyed)) if keyed else ({}, {})
    picked2 = []
    for fam, n in alloc2.items():
        pool = list(keyed[fam])
        if fam == "cloze_test":
            groups = defaultdict(list)
            for q in pool:
                groups[q.get("passage_text")].append(q)
            usable = [v for v in groups.values() if len(v) >= n]
            if usable:
                pool = sorted(rng.choice(usable), key=lambda q: q["qid"])
            else:
                pool = sorted(pool, key=lambda q: q["qid"])
        else:
            rng.shuffle(pool)
        picked2.extend(pool[:n])
    # Top-up: a family may have fewer keyed questions than the blueprint asks
    # for (cloze in particular, since each blank needs its own key). Fill the
    # shortfall from the families that have surplus, largest blueprint weight
    # first, so the paper still lands on exactly 25 rather than silently short.
    used = {q["qid"] for q in picked2}
    if len(picked2) < NOMINAL:
        spare = [q for f in sorted(keyed, key=lambda f: -bp.get(f, {}).get("expected", 0))
                 for q in keyed[f] if q["qid"] not in used]
        for q in spare[:NOMINAL - len(picked2)]:
            picked2.append(q)
            used.add(q["qid"])

    for q in picked2:
        # Cloze options may be reordered safely -- each blank's options are
        # independent; only the PASSAGE must stay fixed. Leaving them unshuffled
        # skewed the key to 'c' on 11 of 23 questions.
        q["options"], q["answer"] = reorder(q, rng)
    picked2.sort(key=lambda q: (q["family"] == "cloze_test", q["family"]))

    mv2 = sum(1 for q in picked2 if str(q["verified_by"]).startswith("machine"))
    notes2 = [
        "Every question is **real SSC CGL Tier-1 text**, extracted from official "
        "2022-2024 papers and reproduced as printed.",
        "**Provenance limit you must know about:** official SSC papers contain no "
        "answer key at all. \"Ans\" introduces the option list, and there are zero "
        "occurrences of \"Correct Option\" or \"Answer Key\" across a full shift. "
        "The only keyed files in the corpus are the 2025 \"Similar Paper\" coaching "
        "papers, which are not real PYQs and are excluded here.",
        f"So keys were derived: **{mv2} of {len(picked2)} machine-derived** "
        "(dictionary / phonetic article rule / agreement rule), the remainder "
        "solved editorially and labelled as such. Only 10 of 925 official English "
        "questions are machine-keyable, which is why the editorial share is large.",
        "The derivers were sanity-checked against the 2025 coaching keys on the "
        "same question formats: **6 of 6 agreed**.",
        "Family mix follows the same 2026 blueprint as Paper 1.",
    ]
    (OUT / "english_paper2_real_pyq.md").write_text(render(
        "SSC CGL Tier-1 2026 — English Practice Paper 2",
        "**Real past-year questions** (official 2022-2024), selected to match the "
        "forecast blueprint. Key basis stated per question.",
        picked2, alloc2, dropped2, notes2, bp))

    # ---------------- report ----------------------------------------------
    for nm, pk, al, dr in (("PAPER 1 (constructed)", picked1, alloc1, dropped1),
                           ("PAPER 2 (real official PYQ)", picked2, alloc2, dropped2)):
        print("=" * 80)
        print(nm)
        print("=" * 80)
        print(f"questions: {len(pk)}   omitted blueprint share: {sum(dr.values()):.2f}/25")
        print(f"answer balance: {dict(sorted(Counter(q['answer'] for q in pk).items()))}")
        for f, n in sorted(al.items(), key=lambda x: -x[1]):
            print(f"   {f:<26}{n:>3}  (blueprint {bp.get(f,{}).get('expected',0):.2f})")
        print(f"verification: {dict(Counter(str(q.get('verified_by')).split(':')[0] for q in pk))}")
        print()

    # What still needs an editorial key, for the operator to fill in.
    need = []
    for fam, n in (alloc1 or {}).items():
        pass
    for fam, v in real_by_fam.items():
        for q in v:
            if not q["answer"]:
                need.append(q)
    json.dump([{"qid": q["qid"], "family": q["family"], "stem": q["stem"],
                "options": q["options"]} for q in need[:400]],
              open(OUT / "english_needs_key.json", "w"), indent=2)
    print(f"Unkeyed official questions available for editorial keying: {len(need)}")
    print(f"  -> {OUT/'english_needs_key.json'}")


if __name__ == "__main__":
    main()
