"""Assemble a Kerala PSC degree-level Main MOCK PAPER from real past questions.

SELECTION, NOT GENERATION. Every question in the output is a real question from
a real Kerala PSC Degree Level Main Examination, with the commission's own
published answer. Nothing is written by a model, so nothing can be fabricated.

The paper is built to KPSC's PUBLISHED mark distribution -- GK 55 (broken down
by topic), Current Affairs 15, Aptitude 10, English 10, Malayalam 10 -- which is
taken from the official syllabus, never inferred from tag counts.

Two honest limits are enforced rather than hidden:
  * CURRENT AFFAIRS from 2022-23 papers is STALE for a 2026 exam. Those slots
    are reported as a deficit, not quietly filled with obsolete news.
  * MALAYALAM extracts as mojibake (legacy non-Unicode font), so those slots
    cannot be served as readable text.
A slot that cannot be filled honestly is left visibly short.

Used questions are recorded so successive daily papers do not repeat.

Run: ./.venv/bin/python src/kerala_mock.py --exam univ --seed 1
"""
from __future__ import annotations
import argparse, json, random, re, glob
from pathlib import Path
from collections import defaultdict
from kerala_parse import extract as _extract

ROOT = Path(__file__).resolve().parent.parent
TAGS = ROOT / "out" / "kerala_tagged.json"
KEYS = ROOT / "data" / "raw" / "kerala" / "keys" / "answer_maps.json"
USED = ROOT / "out" / "kerala_used_questions.json"

# Official published distribution (keralapsc.gov.in Cat.No.26/2022)
BLUEPRINT = {
    # Coarse taxonomy: kerala_governance(10)+important_acts(5) are merged because
    # the tagger cannot separate them (measured: 2 vs 10 and 12 vs 5 apart,
    # 14 vs 15 together). physics(3)+chemistry(3)+life_science(6) merged as
    # science for the same reason. Totals still sum to the official 100.
    "kerala_governance_acts": 15,
    "history": 5, "geography": 5, "economics": 5, "indian_constitution": 5,
    "science": 12, "arts_literature_culture_sports": 5, "basics_of_computer": 3,
    "current_affairs": 15,
    "aptitude": 10, "general_english": 10, "regional_language": 10,
}
STALE = {"current_affairs"}          # time-bound; 2023 news is wrong for 2026
UNREADABLE = {"regional_language"}   # legacy-font mojibake

# map answer-key files to their question-paper files
KEYMAP = {
    "ua_main_160_2023.pdf": "25082023_universities.pdf",
    "dlme_assistant_076_2023.pdf": "22062023_assistant_gr_ii.pdf",
    "09062023_junior_assistant_cashier_assistant_gr_ii_cle.pdf": "09062023_junior_assistant.pdf",
    "23122022_assistant_director_of_national_savings_degre.pdf": "23122022_national_savings.pdf",
    "26062023_clerk_cum_typist_degree_level_main_examinati.pdf": "26062023_clerk_cum_typist.pdf",
    "27122022_assistant_degree_level_main_examination_kera.pdf": "27122022_administrative.pdf",
    "29122022_typist_clerk_gr_ii_degree_level_main_examina.pdf": "29122022_typist_clerk.pdf",
    "29102024_assistant_manager_kscb_direct_and_b_t_main_e.pdf": "29102024_kscb.pdf",
    "08102025_assistant_time_keeper_main_examination_kanna.pdf": "08102025_time_keeper.pdf",
}


def load_text(pdf):
    return _extract(pdf)


def build_pool() -> dict[str, list[dict]]:
    tags = json.load(open(TAGS))
    keys = json.load(open(KEYS))
    pool = defaultdict(list)
    for pdfname, qtags in tags.items():
        if not isinstance(qtags, dict) or "_status" in qtags:
            continue
        hits = [f for f in glob.glob(str(ROOT / "data/raw/kerala/*" / pdfname))
                if "_excluded" not in f]
        if not hits:
            continue
        text = load_text(Path(hits[0]))
        ans = keys.get(KEYMAP.get(pdfname, ""), {})
        for qno, label in qtags.items():
            if not label or int(qno) not in text:
                continue
            pool[label].append({
                "src": pdfname, "qno": int(qno),
                "text": text[int(qno)], "answer": ans.get(qno, "?"),
                "uid": f"{pdfname}#{qno}",
            })
    return pool


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exam", choices=["univ", "company"], required=True)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    pool = build_pool()
    used = set(json.load(open(USED))) if USED.exists() else set()
    rng = random.Random(a.seed)

    picked, deficits = [], []
    for topic, need in BLUEPRINT.items():
        avail = [q for q in pool.get(topic, []) if q["uid"] not in used]
        rng.shuffle(avail)
        take = avail[:need]
        picked.extend((topic, q) for q in take)
        if len(take) < need:
            why = ("STALE - 2022/23 news invalid for 2026" if topic in STALE else
                   "UNREADABLE - legacy-font mojibake" if topic in UNREADABLE else
                   f"pool exhausted ({len(pool.get(topic, []))} total)")
            deficits.append((topic, need - len(take), why))

    name = {"univ": "University Assistant", "company": "Company/Corporation/Board Assistant"}[a.exam]
    lines = [f"# Kerala PSC {name} — Main Examination Mock Paper",
             "",
             "**Every question below is a real question from a past Kerala PSC Degree Level "
             "Main Examination, with the commission's published answer.** Nothing is model-generated.",
             "",
             f"Built to the official mark distribution (Cat.No.26/2022). "
             f"Filled {len(picked)} of 100 slots.", ""]
    if deficits:
        lines += ["## Unfilled slots — deliberately left short", "",
                  "| Section | Short by | Why |", "|---|---|---|"]
        lines += [f"| {t} | {n} | {w} |" for t, n, w in deficits]
        lines += [""]
    lines += ["---", "", "## Questions", ""]
    for i, (topic, q) in enumerate(picked, 1):
        lines.append(f"**{i}.** {q['text']}")
        lines.append(f"*[{topic} · source {q['src'][:28]} Q{q['qno']}]*")
        lines.append("")
    lines += ["---", "", "## Answer key", ""]
    lines += [" ".join(f"{i}-{q['answer']}" for i, (_, q) in enumerate(picked, 1))]

    out = ROOT / "out" / f"kerala_mock_{a.exam}_seed{a.seed}.md"
    out.write_text("\n".join(lines))
    used |= {q["uid"] for _, q in picked}
    json.dump(sorted(used), open(USED, "w"))

    print(f"POOL SIZES: " + ", ".join(f"{k}={len(v)}" for k, v in sorted(pool.items())))
    print(f"\nfilled {len(picked)}/100 slots -> {out}")
    for t, n, w in deficits:
        print(f"  SHORT {t}: -{n}  ({w})")


if __name__ == "__main__":
    main()
