"""Assemble the two 25-question Reasoning papers from the 2026 blueprint.

Paper 1 -- newly constructed questions (out/candidates.json), each answer
           re-derived independently of the generator that produced it.
Paper 2 -- real PYQ text from the corpus, with each key either solver-verified
           or flagged as a source claim.

Allocation uses largest-remainder rounding on the archetype blueprint so the
paper lands on exactly 25. Archetypes with no available questions are dropped
BEFORE renormalising, and the dropped share is reported rather than buried --
mostly figure-based archetypes that cannot exist as text.
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

FIGURE_ARCHETYPES = {"image", "figure_series", "fold_punch", "hidden_figure",
                     "count_shapes", "dice_faces", "pattern_completion"}

NO_SHUFFLE = re.compile(r"all of the above|none of these|both .* and|"
                        r"none of the above", re.I)

# Belt-and-braces guard. Extraction now strips response-sheet metadata from
# option text, but a paper is the last place to discover it did not: a leaked
# "Question ID ... Chosen Option : --" tail both garbles the option and marks it
# as the last one. Any item still carrying such text is dropped, not printed.
DIRTY = re.compile(r"question\s*id|chosen\s*option|status\s*:|^\s*$", re.I)


def is_readable(text: str) -> bool:
    """Reject space-degraded extractions.

    Nine 2022 papers and some 2023 ones extract without inter-word spacing
    ("Afterarrangingthegivenwordsaccordingtodictionaryorder"). That text is fine
    to TAG -- the squashed matching tier handles it -- but it must never be
    printed, because a candidate cannot read it. Two tests: overall space ratio,
    and the longest unbroken alphabetic run.
    """
    t = text.strip()
    if len(t) < 25:
        return False
    if t.count(" ") / len(t) < 0.10:
        return False
    longest = max((len(m) for m in re.findall(r"[A-Za-z]+", t)), default=0)
    return longest <= 20


def is_clean(q) -> bool:
    if len(q["options"]) != 4:
        return False
    if not is_readable(q["stem"]):
        return False
    if any(not is_readable(t) and len(t) > 25 for _, t in q["options"]):
        return False
    texts = [t.strip() for _, t in q["options"]]
    if any(DIRTY.search(t) or not t for t in texts):
        return False
    if len(set(t.lower() for t in texts)) != 4:
        return False
    if DIRTY.search(q["stem"]):
        return False
    return True


def load_blueprint(level="subtopic"):
    p = OUT / f"reasoning_forecast_2026_{level}.json"
    return json.load(open(p))["blueprint"]


def allocate(blueprint, available: set[str], total=NOMINAL):
    """Largest-remainder allocation over the archetypes we can actually fill."""
    usable = {k: v["expected"] for k, v in blueprint.items() if k in available}
    dropped = {k: v["expected"] for k, v in blueprint.items()
               if k not in available and v["expected"] > 0}
    s = sum(usable.values())
    if s == 0:
        raise SystemExit("no usable archetypes")
    scaled = {k: v / s * total for k, v in usable.items()}
    floors = {k: int(v) for k, v in scaled.items()}
    left = total - sum(floors.values())
    order = sorted(scaled, key=lambda k: -(scaled[k] - floors[k]))
    for k in order[:left]:
        floors[k] += 1
    return {k: v for k, v in floors.items() if v > 0}, dropped, scaled


def pick(pool_by_arch, alloc, rng, prefer_verified=True):
    chosen = []
    for arch, n in alloc.items():
        cands = list(pool_by_arch.get(arch, []))
        if prefer_verified:
            cands.sort(key=lambda q: 0 if str(q.get("verified_by", "")).startswith(
                ("solver", "model_check", "relation", "property")) else 1)
            head = [c for c in cands if str(c.get("verified_by", "")).startswith(
                ("solver", "model_check", "relation", "property"))]
            tail = [c for c in cands if c not in head]
            rng.shuffle(head)
            rng.shuffle(tail)
            cands = head + tail
        else:
            rng.shuffle(cands)
        chosen.extend(cands[:n])
    return chosen


def interleave(questions, rng):
    """Spread archetypes through the paper instead of grouping them.

    Selection naturally emits all 7 series questions together, then all 4 coding
    questions. A real shift interleaves, and a grouped paper both reads wrong and
    lets a candidate coast on momentum within a block (the realism constraint in
    §53). Round-robin across archetypes, largest group first, with the order
    inside each group randomised.
    """
    buckets = defaultdict(list)
    for q in questions:
        buckets[q["subtopic"]].append(q)
    for b in buckets.values():
        rng.shuffle(b)
    order = sorted(buckets, key=lambda k: -len(buckets[k]))
    out = []
    while any(buckets[k] for k in order):
        for k in order:
            if buckets[k]:
                out.append(buckets[k].pop())
    return out


def reorder_options(q, rng):
    """Shuffle option order, preserving which text is correct.

    A practice paper inheriting the corpus's answer-letter skew (36% 'a') would
    train candidates to guess 'a'. Items whose options reference each other
    ("All of the above") are left untouched, since their order carries meaning.
    """
    opts = q["options"]
    if any(NO_SHUFFLE.search(t) for _, t in opts):
        return opts, q["answer"]
    correct_text = dict(opts)[q["answer"]]
    texts = [t for _, t in opts]
    rng.shuffle(texts)
    letters = "abcd"[:len(texts)]
    new = list(zip(letters, texts))
    new_ans = next(l for l, t in new if t == correct_text)
    return new, new_ans


def render(title, subtitle, questions, alloc, dropped, scaled, notes):
    L = []
    L.append(f"# {title}\n")
    L.append(subtitle + "\n")
    L.append(f"**{len(questions)} questions · General Intelligence & Reasoning "
             f"· answer key and evidence map at the end**\n")
    L.append("---\n")
    L.append("## Questions\n")
    for i, q in enumerate(questions, 1):
        L.append(f"**{i}.** {q['stem']}\n")
        for letter, text in q["options"]:
            L.append(f"- ({letter}) {text}")
        L.append("")
    L.append("---\n")
    L.append("## Answer key\n")
    row = []
    for i, q in enumerate(questions, 1):
        row.append(f"{i}. {q['answer'].upper()}")
        if i % 5 == 0:
            L.append("  ".join(row))
            row = []
    if row:
        L.append("  ".join(row))
    L.append("")
    L.append("---\n")
    L.append("## Question-level forecast table\n")
    L.append("| Q | Archetype | Expected/25 | P(appears) | Verification |")
    L.append("|---|---|---|---|---|")
    bp = load_blueprint()
    for i, q in enumerate(questions, 1):
        a = q["subtopic"]
        e = bp.get(a, {}).get("expected", 0)
        p = bp.get(a, {}).get("p_appears", 0)
        L.append(f"| {i} | {a} | {e:.2f} | {p:.0%} | {q.get('verified_by','—')} |")
    L.append("")
    L.append("---\n")
    L.append("## Why these archetypes\n")
    L.append("| Archetype | Blueprint expected | Allocated here |")
    L.append("|---|---|---|")
    for a, n in sorted(alloc.items(), key=lambda x: -x[1]):
        L.append(f"| {a} | {bp.get(a,{}).get('expected',0):.2f} | {n} |")
    L.append("")
    if dropped:
        L.append("### Archetypes deliberately not represented\n")
        L.append("| Archetype | Blueprint expected | Reason |")
        L.append("|---|---|---|")
        for a, e in sorted(dropped.items(), key=lambda x: -x[1]):
            why = ("needs a real figure" if a in FIGURE_ARCHETYPES
                   else "no verified question available")
            L.append(f"| {a} | {e:.2f} | {why} |")
        L.append(f"\nCombined omitted share: "
                 f"**{sum(dropped.values()):.2f} of 25 questions "
                 f"({100*sum(dropped.values())/25:.0f}%)**. "
                 f"Their weight was redistributed across the archetypes above, "
                 f"so this paper is denser in text-based reasoning than a real "
                 f"shift will be.\n")
    L.append("---\n")
    L.append("## Method and limitations\n")
    for n in notes:
        L.append(f"- {n}")
    L.append("")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    bp = load_blueprint()

    # ---------------- Paper 1: newly constructed ---------------------------
    cands = json.load(open(OUT / "candidates.json"))
    by_arch = defaultdict(list)
    for q in cands:
        by_arch[q["subtopic"]].append(q)
    alloc1, dropped1, scaled1 = allocate(bp, set(by_arch))
    picked1 = interleave(pick(by_arch, alloc1, rng), rng)
    for q in picked1:
        q["options"], q["answer"] = reorder_options(q, rng)

    notes1 = [
        "Every question was constructed programmatically, then its answer was "
        "**re-derived from the rendered question text by an independent solver** "
        "(`src/solvers.py`). Any candidate the solver could not confirm was "
        "discarded, not shipped.",
        "Syllogism items are verified by brute-force model checking over a small "
        "universe; relation-chain, set-relation and odd-one-out items are verified "
        "by construction proof (the property must hold for exactly the intended "
        "split). The verification route is printed per question above.",
        "Archetype mix follows the 2026 blueprint from `dirichlet_a2`, the model "
        "selected by walk-forward backtesting on official papers (Δ MAE −0.0101 "
        "vs naive averaging, 95% CI [−0.0199, −0.0012], p=0.023).",
        "These questions are **not predictions of specific 2026 questions**. No "
        "one can predict those. They are original items matching the forecast "
        "archetype mix, difficulty and structure.",
        "Option order is randomised so the answer key is not letter-skewed.",
    ]
    md1 = render(
        "SSC CGL Tier-1 2026 — Reasoning Practice Paper 1",
        "**Newly constructed questions fitted to the forecast blueprint.** "
        "Every answer independently verified; no answer is taken on trust.",
        picked1, alloc1, dropped1, scaled1, notes1)
    (OUT / "paper1_constructed.md").write_text(md1)

    # ---------------- Paper 2: real PYQs -----------------------------------
    src = {r["qid"]: r for r in json.load(open(OUT / "reasoning_questions.json"))}
    tagged = json.load(open(OUT / "reasoning_tagged.json"))
    ver = json.load(open(OUT / "verified_keys.json"))

    real_by_arch = defaultdict(list)
    for r in tagged:
        s = src.get(r["qid"])
        if not (s and len(s["options"]) == 4 and r.get("subtopic")):
            continue
        v = ver.get(r["qid"])
        answer = v["answer"] if v else s["answer"]
        if not answer:
            continue
        if len(r["stem"]) < 40 or len(r["stem"]) > 400:
            continue
        cand = dict(subtopic=r["subtopic"], stem=r["stem"], options=s["options"])
        if not is_clean(cand):
            continue
        real_by_arch[r["subtopic"]].append(dict(
            subtopic=r["subtopic"], stem=r["stem"], options=s["options"],
            answer=answer,
            verified_by=("solver:" + v["solver"]) if v else "source key (unverified)",
            year=r["year"], source=s["source_pdf"]))

    alloc2, dropped2, scaled2 = allocate(bp, set(real_by_arch))
    picked2 = interleave(pick(real_by_arch, alloc2, rng), rng)
    for q in picked2:
        q["options"], q["answer"] = reorder_options(q, rng)

    n_solver = sum(1 for q in picked2 if q["verified_by"].startswith("solver"))
    notes2 = [
        "Every question is **real SSC CGL Tier-1 text** extracted from the "
        "past-paper corpus (2021–2025), reproduced as printed.",
        f"**{n_solver} of {len(picked2)} keys were independently re-derived** by "
        "`src/solvers.py` from the question text. The remainder carry the source "
        "paper's key, marked as unverified in the table above.",
        "**Important provenance limit:** official SSC papers in this corpus "
        "contain no answer key at all — 'Ans' introduces the option list, and "
        "there are zero occurrences of 'Correct Option' or 'Answer Key'. The only "
        "keys available come from Tier-3 coaching reconstructions of 2025. Those "
        "keys agreed with the independent solver on **125 of 125** checkable "
        "questions, which is why they are used at all.",
        "Archetype mix follows the same 2026 blueprint as Paper 1.",
        "Option order is randomised: the source pool's keys are 36% 'a', which "
        "would otherwise train a guessing heuristic.",
    ]
    md2 = render(
        "SSC CGL Tier-1 2026 — Reasoning Practice Paper 2",
        "**Real past-year questions**, selected to match the forecast blueprint. "
        "Keys independently re-derived where computable.",
        picked2, alloc2, dropped2, scaled2, notes2)
    (OUT / "paper2_real_pyq.md").write_text(md2)

    # ---------------- report ----------------------------------------------
    for name, picked, alloc, dropped in (("PAPER 1 (constructed)", picked1, alloc1, dropped1),
                                          ("PAPER 2 (real PYQ)", picked2, alloc2, dropped2)):
        print("=" * 84)
        print(name)
        print("=" * 84)
        print(f"questions: {len(picked)}")
        print(f"answer-letter balance: {dict(sorted(Counter(q['answer'] for q in picked).items()))}")
        print(f"archetypes used: {len(alloc)}   omitted share: "
              f"{sum(dropped.values()):.2f}/25")
        for a, n in sorted(alloc.items(), key=lambda x: -x[1]):
            print(f"   {a:<30}{n:>3}   (blueprint {bp.get(a,{}).get('expected',0):.2f})")
        vb = Counter(q.get("verified_by", "—").split(":")[0] for q in picked)
        print(f"verification: {dict(vb)}")
        print()
    print(f"Wrote {OUT/'paper1_constructed.md'}")
    print(f"Wrote {OUT/'paper2_real_pyq.md'}")


if __name__ == "__main__":
    main()
