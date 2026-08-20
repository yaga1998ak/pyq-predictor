"""Assemble the single final SSC CGL Tier-I 2026 predicted paper.

MODE DECISION (§3, §33, §66) — ONE PAPER, decided by measurement:

  * No adaptation signature. Official-to-official consecutive-year TVD is FLAT
    or FALLING (2022->23 mean 0.261, 2023->24 mean 0.136). Deliberate
    anti-prediction mutation would push it UP. Recency beat pooled history in
    only 1 of 4 sections, and the arms-race hypothesis predicts the opposite.
  * The posterior is UNIMODAL BUT WIDE. Real shifts scatter around a stable
    centroid (median TVD to centroid 0.216-0.269) without clustering into
    distinct modes. That is within-mode sampling noise, not between-mode
    scenario structure.
  * Therefore a 3-paper "Continuation / Anti-Prediction / New Regime" portfolio
    would attach invented weights to regimes for which this corpus contains no
    observable (§75), and building papers by resampling within-mode variance is
    exactly what §34 forbids.

Sectional blueprint follows each section's validated forecast. The one genuine
2026 regime change is Tier-2 official evidence: a 15-MINUTE SECTIONAL TIMER,
i.e. ~36 s per question, which is a hard constraint on time load (§48) and is
enforced here per section.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from ga_questions import as_records as ga_records

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
PER_SECTION = 25
SECTION_TIME_LIMIT = 900  # 15 minutes, official 2026 sectional timer

DIRTY = re.compile(r"question\s*id|chosen\s*option|status\s*:|section\s*:|"
                   r"comprehension\s*:", re.I)
OCR_JUNK = re.compile(r"[*#@\\|~^®]|\bJf\b|\bZF\b")



# Per-archetype solving-time estimates, in seconds. The 2026 sectional timer
# gives 900 s for 25 questions -- an average of 36 s -- so a flat default is not
# good enough: a 45 s default put Reasoning and English 25% OVER the limit, which
# no real setter would ship because the section becomes unfinishable.
# Values are ordered by the work each archetype actually requires.
TIME_MODEL = {
    # Reasoning
    "language_code": 40, "number_or_letter_series": 45, "colon_notation": 25,
    "number_set_analogy": 40, "term_analogy": 30, "generic_odd": 30,
    "coded_relation": 50, "symbolic_operator": 50, "generation_chain": 45,
    "sign_interchange": 45, "sign_substitution": 40, "symbol_definition": 35,
    "statements_conclusions": 55, "position_unchanged": 40,
    "alphabetical_arrangement": 35, "number_pair_operation": 45,
    "letter_blank_completion": 40, "image": 25, "dice_faces": 40,
    "figure_series": 30, "fold_punch": 30, "hidden_figure": 25,
    "count_shapes": 40, "pattern_completion": 40, "day_offset": 40,
    "letter_substitution": 35, "identical_strings": 30,
    # English -- vocabulary is fast, transformation and ordering are slow
    "synonym": 15, "antonym": 15, "group_of_words": 20, "idiom_meaning": 20,
    "incorrectly_spelt": 15, "correctly_spelt": 15, "homonym_in_sentence": 25,
    "single_blank": 25, "underlined_segment": 30, "underlined_word": 25,
    "four_segments": 40, "parts_as_options": 40, "no_error_option": 40,
    "to_passive": 35, "to_active": 35, "reported_speech": 40,
    "sentence_order": 45, "part_order": 40, "numbered_blank": 25,
    "passage_question": 40, "targeted_rule": 30,
}


def est_time_for(q) -> int:
    """Archetype-based estimate, falling back to a mid value."""
    if q.get("est_time"):
        return int(q["est_time"])
    key = q.get("subtopic") or q.get("topic")
    return TIME_MODEL.get(key, 35)



def build_subtopic_to_topic():
    """Map subtopic -> topic from the tagged corpora.

    The Reasoning and English generators emit only a subtopic, so `topic` was
    None on every generated question. That silently broke the §60 realism test:
    comparing an all-None distribution against the historical topic centroid
    yields TVD 0.500 and looked like a genuine outlier paper rather than a
    missing field.
    """
    m = {}
    for f, tk, sk in (("reasoning_tagged.json", "topic", "subtopic"),
                      ("quant_tagged.json", "topic", "subtopic"),
                      ("english_tagged.json", "family", "subtype")):
        for r in json.load(open(OUT / f)):
            t, sub = r.get(tk), r.get(sk)
            if t and sub and sub not in m:
                m[sub] = t
    return m


SUB2TOP = build_subtopic_to_topic()


def clean(q) -> bool:
    opts = q.get("options") or []
    if len(opts) != 4:
        return False
    t = [str(x[1]).strip() for x in opts]
    if any(not x for x in t) or len(set(t)) != 4:
        return False
    if any(DIRTY.search(x) or OCR_JUNK.search(x) for x in t):
        return False
    s = q.get("stem", "")
    if len(s) < 20 or DIRTY.search(s):
        return False
    if s.count(" ") / max(len(s), 1) < 0.10:
        return False
    return True


def grade(q, blueprint, level_key):
    """S/A/B/C per §36, from the section's own validated blueprint.

    S -- the microtopic is in the blueprint's top band (expected >= 1.0 per 25)
    A -- present in the blueprint with expected >= 0.35, i.e. structurally
         predicted though not top-ranked
    B -- in the blueprint but low expectation: scenario-supported only
    C -- not in the blueprint: controlled novelty
    """
    key = q.get(level_key) or q.get("subtopic") or q.get("topic")
    e = blueprint.get(key, {}).get("expected", 0.0) if blueprint else 0.0
    if e >= 1.0:
        return "S", f"blueprint expected {e:.2f}/25 (top band)"
    if e >= 0.35:
        return "A", f"blueprint expected {e:.2f}/25 (structurally predicted)"
    if e > 0:
        return "B", f"blueprint expected {e:.2f}/25 (scenario-supported)"
    return "C", "outside blueprint (controlled novelty)"


def pick(pool, blueprint, level_key, n, rng, micro_cap=3):
    """Blueprint-proportional selection with a per-microtopic cap for realism."""
    by = defaultdict(list)
    for q in pool:
        by[q.get(level_key) or q.get("subtopic") or q.get("topic")].append(q)
    weights = {k: blueprint.get(k, {}).get("expected", 0.0) for k in by}
    tot = sum(weights.values()) or 1.0
    alloc = {k: v / tot * n for k, v in weights.items()}
    floors = {k: int(v) for k, v in alloc.items()}
    left = n - sum(floors.values())
    for k in sorted(alloc, key=lambda k: -(alloc[k] - floors[k]))[:left]:
        floors[k] += 1
    chosen, used = [], Counter()
    for k, cnt in sorted(floors.items(), key=lambda x: -x[1]):
        cands = by[k]
        rng.shuffle(cands)
        take = min(cnt, micro_cap, len(cands))
        chosen.extend(cands[:take])
        used[k] += take
    # top-up respecting the cap
    if len(chosen) < n:
        spare = [q for k in sorted(by, key=lambda k: -weights.get(k, 0))
                 for q in by[k]
                 if q not in chosen and used[k] < micro_cap]
        rng.shuffle(spare)
        for q in spare:
            if len(chosen) >= n:
                break
            k = q.get(level_key) or q.get("subtopic") or q.get("topic")
            if used[k] < micro_cap:
                chosen.append(q)
                used[k] += 1
    return chosen[:n]


def reorder(q, rng):
    opts = q["options"]
    correct = dict(opts)[q["answer"]]
    texts = [t for _, t in opts]
    rng.shuffle(texts)
    new = list(zip("abcd", texts))
    return new, next(l for l, t in new if t == correct)


def balance(qs, rng, cap=9, tries=400):
    for _ in range(tries):
        c = Counter(q["answer"] for q in qs)
        worst, n = c.most_common(1)[0]
        if n <= cap:
            return
        for q in qs:
            if q["answer"] == worst:
                q["options"], q["answer"] = reorder(q, rng)


def load_section(cand_file, forecast_file, forecast_key, level_key, section, rng):
    pool = [q for q in json.load(open(OUT / cand_file)) if clean(q)]
    bp = json.load(open(OUT / forecast_file))
    bp = bp[forecast_key] if forecast_key else bp
    sel = pick(pool, bp, level_key, PER_SECTION, rng)
    for q in sel:
        q["section"] = section
        q["options"], q["answer"] = reorder(q, rng)
        g, why = grade(q, bp, level_key)
        q["grade"], q["grade_reason"] = g, why
        q["est_time"] = est_time_for(q)
        q.setdefault("difficulty", "medium")
        if not q.get("topic"):
            # generators use different key names: `subtopic` (Reasoning, Quant)
            # and `subtype` (English). Reading only one leaves topic None and
            # reproduces the same false-outlier reading.
            q["topic"] = (q.get("family")
                          or SUB2TOP.get(q.get("subtopic"))
                          or SUB2TOP.get(q.get("subtype")))

    # Enforce the 15-minute sectional budget (§1 hard constraint, §48 realism).
    # Swap the slowest items for faster same-grade alternatives until the
    # section fits, rather than shipping an unfinishable section.
    def load(qs):
        return sum(x["est_time"] for x in qs)

    if load(sel) > SECTION_TIME_LIMIT:
        chosen_ids = {id(x) for x in sel}
        spare = [q for q in pool if id(q) not in chosen_ids]
        for q in spare:
            q["est_time"] = est_time_for(q)
            g, why = grade(q, bp, level_key)
            q["grade"], q["grade_reason"] = g, why
        # prefer fast, high-grade replacements
        spare.sort(key=lambda q: (q["est_time"], {"S": 0, "A": 1, "B": 2, "C": 3}[q["grade"]]))
        used_micro = Counter(x.get(level_key) or x.get("subtopic") for x in sel)
        guard = 0
        while load(sel) > SECTION_TIME_LIMIT and spare and guard < 400:
            guard += 1
            sel.sort(key=lambda x: -x["est_time"])
            slow = sel[0]
            repl = None
            for cand in spare:
                k = cand.get(level_key) or cand.get("subtopic")
                if cand["est_time"] < slow["est_time"] and used_micro[k] < 4:
                    repl = cand
                    break
            if repl is None:
                break
            spare.remove(repl)
            ks = slow.get(level_key) or slow.get("subtopic")
            kr = repl.get(level_key) or repl.get("subtopic")
            used_micro[ks] -= 1
            used_micro[kr] += 1
            sel.remove(slow)
            repl["section"] = section
            repl["options"], repl["answer"] = reorder(repl, rng)
            repl.setdefault("difficulty", "medium")
            if not repl.get("topic"):
                repl["topic"] = (repl.get("family")
                                 or SUB2TOP.get(repl.get("subtopic"))
                                 or SUB2TOP.get(repl.get("subtype")))
            sel.append(repl)

    balance(sel, rng)
    return sel, bp


def main() -> None:
    rng = random.Random(2026)

    reasoning, bp_r = load_section(
        "candidates.json", "reasoning_forecast_2026_subtopic.json",
        "blueprint", "subtopic", "reasoning", rng)
    quant, bp_q = load_section(
        "quant_candidates.json", "quant_forecast_2026.json",
        "subtopic_corrected", "subtopic", "quant", rng)
    english, bp_e = load_section(
        "english_candidates.json", "english_forecast_2026.json",
        "subtype", "subtype", "english", rng)

    ga = ga_records()
    bp_g = json.load(open(OUT / "ga_zones_2026.json"))
    ga_freq = bp_g["final_freq"]
    for q in ga:
        f = ga_freq.get(q["topic"], 0)
        # GA grading uses the zone's observed per-shift frequency: every
        # question is answerable from a selected note, so S requires the zone
        # to be a top-band zone as well.
        if f >= 0.45:
            q["grade"], q["grade_reason"] = "S", f"zone {f:.2f}/shift (top band)"
        elif f >= 0.20:
            q["grade"], q["grade_reason"] = "A", f"zone {f:.2f}/shift (predicted zone)"
        else:
            q["grade"], q["grade_reason"] = "B", f"zone {f:.2f}/shift (lower band)"
    balance(ga, rng)

    sections = [("General Intelligence & Reasoning", reasoning),
                ("General Awareness", ga),
                ("Quantitative Aptitude", quant),
                ("English Comprehension", english)]

    # ---------------------------------------------------------------- audits
    print("=" * 92)
    print("FINAL PAPER — SECTION AUDIT")
    print("=" * 92)
    print(f"{'section':<34}{'n':>4}{'S':>4}{'A':>4}{'B':>4}{'C':>4}"
          f"{'S+A%':>7}{'time(s)':>9}{'limit':>7}")
    print("-" * 92)
    all_q = []
    for name, qs in sections:
        c = Counter(q["grade"] for q in qs)
        sa = 100 * (c["S"] + c["A"]) / len(qs)
        t = sum(q["est_time"] for q in qs)
        flag = "OK" if t <= SECTION_TIME_LIMIT else "OVER"
        print(f"{name:<34}{len(qs):>4}{c['S']:>4}{c['A']:>4}{c['B']:>4}"
              f"{c['C']:>4}{sa:>6.0f}%{t:>9}{flag:>7}")
        all_q.extend(qs)
    c = Counter(q["grade"] for q in all_q)
    sa = 100 * (c["S"] + c["A"]) / len(all_q)
    print("-" * 92)
    print(f"{'WHOLE PAPER':<34}{len(all_q):>4}{c['S']:>4}{c['A']:>4}"
          f"{c['B']:>4}{c['C']:>4}{sa:>6.0f}%")
    print(f"\nS+A directness: {sa:.1f}%   (§37 target >=75%, preferred >=80%)")
    print(f"answer spread: {dict(sorted(Counter(q['answer'] for q in all_q).items()))}")
    verified = sum(1 for q in all_q
                   if str(q.get("verified_by", "")).startswith("machine"))
    print(f"machine-verified answers: {verified}/{len(all_q)} "
          f"({100*verified/len(all_q):.0f}%)")

    json.dump([{k: v for k, v in q.items() if k != "trace"} for q in all_q],
              open(OUT / "final_paper_questions.json", "w"), indent=2)

    # ---------------------------------------------------------------- render
    L = ["# SSC CGL Tier-I 2026 — Final Predicted Paper\n",
         "**Single paper.** The posterior over 2026 paper configurations is "
         "unimodal but wide: consecutive-year composition drift is flat or "
         "falling, and real shifts scatter around a stable centroid without "
         "forming distinct modes. A scenario portfolio would require inventing "
         "weights for regimes this corpus cannot observe.\n",
         "**Official 2026 structure enforced:** 100 questions · 200 marks · "
         "4 sections of 25 · **15-minute sectional timer** · "
         "−0.50 negative marking.\n", "---\n"]

    qno = 0
    for name, qs in sections:
        t = sum(q["est_time"] for q in qs)
        L.append(f"## {name}")
        L.append(f"*25 questions · estimated load {t//60} min {t%60} s "
                 f"against the 15-minute sectional limit*\n")
        for q in qs:
            qno += 1
            L.append(f"**Q{qno}.** {q['stem']}\n")
            for letter, text in q["options"]:
                L.append(f"- ({letter}) {text}")
            L.append("")
        L.append("---\n")

    L.append("## Answer Key\n")
    row = []
    for i, q in enumerate(all_q, 1):
        row.append(f"{i}. {q['answer'].upper()}")
        if i % 10 == 0:
            L.append("  ".join(row)); row = []
    if row:
        L.append("  ".join(row))

    L += ["", "---\n", "## Question-Level Traceability (§71)\n",
          "| Q | Section | Micro-archetype | Grade | Basis | Diff | Time |",
          "|---|---|---|---|---|---|---|"]
    for i, q in enumerate(all_q, 1):
        sec = {"reasoning": "Reasoning", "general_awareness": "GA",
               "quant": "Quant", "english": "English"}.get(q["section"], q["section"])
        micro = q.get("subtopic") or q.get("topic") or "—"
        L.append(f"| {i} | {sec} | {micro} | **{q['grade']}** | "
                 f"{q['grade_reason']} | {q.get('difficulty','—')} | "
                 f"{q.get('est_time','—')}s |")

    L += ["", "---\n", "## Direct-Prediction Audit (§62)\n",
          "| Section | n | S | A | B | C | S+A | Time load | Limit |",
          "|---|---|---|---|---|---|---|---|---|"]
    for name, qs in sections:
        cc = Counter(q["grade"] for q in qs)
        tt = sum(q["est_time"] for q in qs)
        L.append(f"| {name} | {len(qs)} | {cc['S']} | {cc['A']} | {cc['B']} | "
                 f"{cc['C']} | {100*(cc['S']+cc['A'])/len(qs):.0f}% | "
                 f"{tt//60}m {tt%60}s | 15m |")
    L.append(f"| **Whole paper** | 100 | {c['S']} | {c['A']} | {c['B']} | "
             f"{c['C']} | **{sa:.0f}%** | — | — |")

    (OUT / "final_paper.md").write_text("\n".join(L))
    print(f"\nWrote {OUT/'final_paper.md'}")
    return sa


if __name__ == "__main__":
    main()
