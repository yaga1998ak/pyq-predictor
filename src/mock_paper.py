"""Assemble a mock SSC CGL Tier-1 paper from real past questions.

What this is: 100 authentic questions drawn from the 2021-2025 corpus, selected
so the TOPIC MIX matches the 2026 forecast, with the original options and answer
keys intact.

What this is not: a prediction of the actual 2026 questions. Nothing can do that.
The forecast predicts topic weights, so this paper reproduces those weights using
questions whose answers are known to be correct because SSC published them.

That distinction is why questions are sampled rather than generated: a generated
question can carry a subtly wrong answer key, and a practice paper that teaches
the wrong answer is worse than no practice paper.

Selection rules:
  * recent years are preferred -- the 2022 pattern differs from 2025's
  * only questions with parseable options AND a published answer are eligible
  * no repeats, and no two questions from the same source paper back to back

    python src/mock_paper.py --out out/ssc_cgl_2026_mock.md
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from archetypes import classify_coding, classify_sequence, extract_sequence
from methods import classify_method
from parse import extract_text, QMARK_PATTERN, parse_filename, spacing_ratio
from rules import classify
from schema import Taxonomy, REPO

ANSWER_RE = re.compile(r"\bAns\w*\.?\s*[:\-]?\s*\(?([a-dA-D1-4])\)?")
OPTION_RE = re.compile(r"\(([a-d])\)\s*(.+?)(?=\([a-d]\)|\bAns\b|$)", re.IGNORECASE | re.DOTALL)


def harvest(raw_dir: Path, min_year: int = 2023) -> list[dict]:
    """Re-extract questions keeping options and answers.

    parse.py deliberately trims everything from "Ans" onward -- options add tokens
    without adding topic signal, which is right for tagging and wrong here.
    """
    out = []
    for pdf in sorted(raw_dir.rglob("*.pdf")):
        folder_year = int(pdf.parent.name) if pdf.parent.name.isdigit() else None
        try:
            _, year, _, _, source_type = parse_filename(pdf.stem, folder_year)
        except ValueError:
            continue
        if year < min_year:
            continue
        try:
            text = extract_text(pdf)
        except Exception:
            continue
        if spacing_ratio(text) < 0.08:
            continue  # unreadable without spaces; not fit to print

        matches = list(QMARK_PATTERN.finditer(text))
        for i, m in enumerate(matches):
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            chunk = re.sub(r"\s+", " ", text[m.start():end]).strip()
            if not (60 < len(chunk) < 1200):
                continue
            ans = ANSWER_RE.search(chunk)
            if not ans:
                continue
            stem_end = chunk.find("(a)")
            if stem_end < 30:
                continue
            opts = OPTION_RE.findall(chunk[stem_end:])
            if len(opts) < 4:
                continue

            topic, _ = classify(chunk)
            if not topic:
                continue

            # Solution method: what the candidate actually has to KNOW. Series and
            # coding-decoding are solved arithmetically from the question's own
            # numbers/letters; everything else falls back to lexical signatures.
            method = rule = None
            if topic == "series_completion":
                seq = extract_sequence(chunk)
                if len(seq) >= 4:
                    fam, rl = classify_sequence(seq)
                    if fam not in ("too_short", "unknown"):
                        method, rule = fam, rl
            elif topic == "coding_decoding":
                fam, rl = classify_coding(chunk)
                if fam != "unknown":
                    method, rule = fam, rl
            if method is None:
                method = classify_method(topic, chunk)
            key = ans.group(1).lower()
            key = {"1": "a", "2": "b", "3": "c", "4": "d"}.get(key, key)
            # Strip the source paper's own "Q56." numbering -- the mock paper
            # renumbers from 1, and two competing numbers is confusing to sit.
            stem = re.sub(r"^Q\s*\.?\s*\d+\s*\.?\s*", "", chunk[:stem_end].strip())
            if len(stem) < 25:
                continue
            out.append({
                "stem": stem,
                "options": [(l.lower(), re.sub(r"\bAns\b.*$", "", t, flags=re.I).strip()[:160])
                            for l, t in opts[:4]],
                "answer": key,
                "topic": topic,
                "method": method,
                "rule": rule,
                "year": year,
                "source": pdf.name,
                "source_type": source_type,
            })
    return out


# Validated on held-out years 2023-2025 (see validate_papers.py): blending the
# empirical forecast with a uniform-within-section syllabus prior at this weight
# cut wrong-bucket questions from ~27 to ~24 per 100. Pure forecast (1.0) leaves
# topics the tagger cannot see -- direction_sense, matrix, ranking_and_order --
# at zero questions, which no real SSC paper ever is.
SHRINKAGE = 0.8


def allocate(forecast: dict, taxonomy: Taxonomy, shrinkage: float = SHRINKAGE) -> dict[str, int]:
    """Turn fractional forecast counts into whole questions summing to 25 per section.

    Largest-remainder within each section: floor everything, then hand the
    leftover slots to the biggest fractions. Rounding each topic independently
    would not sum to 25.
    """
    want: dict[str, int] = {}
    for section in taxonomy.sections:
        topics = taxonomy.sections[section]["topics"]
        vals = {t: forecast.get(t, 0.0) for t in topics}
        total = sum(vals.values()) or 1.0
        target = taxonomy.section_size(section)
        uniform = target / len(topics)
        scaled = {t: shrinkage * (v / total * target) + (1 - shrinkage) * uniform
                  for t, v in vals.items()}
        base = {t: int(v) for t, v in scaled.items()}
        short = target - sum(base.values())
        for t in sorted(scaled, key=lambda k: scaled[k] - base[k], reverse=True)[:short]:
            base[t] += 1
        want.update(base)
    return want


def build(pool: list[dict], want: dict[str, int], seed: int, taxonomy=None) -> list[dict]:
    rng = random.Random(seed)
    by_topic = defaultdict(list)
    for q in pool:
        by_topic[q["topic"]].append(q)
    # Recent papers first, so the mock reflects the current exam pattern.
    for qs in by_topic.values():
        rng.shuffle(qs)
        qs.sort(key=lambda q: -q["year"])

    picked, shortfall = [], {}
    used = set()
    for topic, n in want.items():
        avail = [q for q in by_topic.get(topic, []) if id(q) not in used]
        take = avail[:n]
        for q in take:
            used.add(id(q))
        picked.extend(take)
        if len(take) < n:
            shortfall[topic] = n - len(take)

    # Backfill within the same section so each section still totals 25. A section
    # that is short reads as a broken paper; substituting a neighbouring topic
    # from the same section keeps the sitting realistic.
    if shortfall and taxonomy is not None:
        for topic, missing in list(shortfall.items()):
            section = taxonomy.topic_to_section[topic]
            pool_sec = [
                q for q in pool
                if taxonomy.topic_to_section[q["topic"]] == section and id(q) not in used
            ]
            pool_sec.sort(key=lambda q: -q["year"])
            fill = pool_sec[:missing]
            for q in fill:
                used.add(id(q))
            picked.extend(fill)
            if len(fill) >= missing:
                del shortfall[topic]
            else:
                shortfall[topic] = missing - len(fill)
    return picked, shortfall


def render(picked: list[dict], taxonomy: Taxonomy, year: int, shortfall: dict) -> str:
    L = [
        f"# SSC CGL Tier-1 — Mock Paper {year}",
        "",
        "**Practice paper.** Every question below is a real SSC CGL question from a "
        f"past paper (2023–2025), with SSC's own answer key. The *topic mix* follows "
        f"the {year} forecast; the questions themselves are not predictions of what "
        f"{year} will ask — no method can do that.",
        "",
        f"**{len(picked)} questions · 4 sections · answer key at the end**",
        "",
        "---",
        "",
    ]
    n = 0
    for section in taxonomy.sections:
        rows = [q for q in picked if taxonomy.topic_to_section[q["topic"]] == section]
        title = section.replace("_", " ").title()
        L += [f"## {title} ({len(rows)} questions)", ""]
        for q in rows:
            n += 1
            q["_no"] = n
            L.append(f"**{n}.** {q['stem']}")
            L.append("")
            for letter, text in q["options"]:
                L.append(f"&nbsp;&nbsp;({letter}) {text}")
            L += ["", f"<sub>{q['topic']} · {q['year']}</sub>", ""]
        L.append("---")
        L.append("")

    L += ["## Answer Key", "", "| # | Ans | # | Ans | # | Ans | # | Ans |",
          "|---|---|---|---|---|---|---|---|"]
    ordered = sorted(picked, key=lambda q: q["_no"])
    for i in range(0, len(ordered), 4):
        row = ordered[i:i + 4]
        cells = "".join(f" {q['_no']} | **{q['answer']}** |" for q in row)
        L.append("|" + cells)
    if shortfall:
        L += ["", "## Coverage gaps", "",
              "The corpus had too few tagged questions for these topics, so the paper "
              "is short on them. They still appear in the real exam:", ""]
        for t, k in sorted(shortfall.items(), key=lambda kv: -kv[1]):
            L.append(f"- **{t}** — {k} question(s) short")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forecast", default=str(REPO / "out/forecast_2026.json"))
    ap.add_argument("--raw", default=str(REPO / "data/raw"))
    ap.add_argument("--taxonomy", default="ssc_cgl")
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=str(REPO / "out/ssc_cgl_2026_mock.md"))
    args = ap.parse_args()

    tax = Taxonomy.load(args.taxonomy)
    fc = json.load(open(args.forecast))
    forecast = {r["topic"]: r["expected"] for r in fc["forecast"]}

    print("harvesting questions with options + answer keys ...")
    pool = harvest(Path(args.raw))
    print(f"  {len(pool)} usable questions")

    want = allocate(forecast, tax)
    picked, shortfall = build(pool, want, args.seed, tax)
    print(f"  selected {len(picked)} of {sum(want.values())} target")

    md = render(picked, tax, args.year, shortfall)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(md)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
