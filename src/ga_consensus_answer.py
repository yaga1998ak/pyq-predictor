"""Recover GA answers by multi-model consensus — VALIDATED BEFORE USE.

The problem this solves
-----------------------
1,028 official GA questions carry no usable answer:

    no_key_in_official_paper  473   the PDF has no key at all
    chosen_option_only        550   the PDF records the CANDIDATE'S marked
                                    response, not the correct answer

No parser can fix this; the information is absent from the source. But GA
questions are *closed-book factual* — the answer is recoverable from world
knowledge, which is the one job `INSIGHTS.md` §4 identifies models as suited
for ("the signal is in named entities rather than phrasing").

The discipline
--------------
`INSIGHTS.md` §5: "after every pipeline stage, check the output against an
*independent* expectation - not the stage's own success signal." A model that
reports high confidence while being wrong is this project's documented failure
mode (a 35%-accuracy tagger reported 96% mean confidence).

So this module MEASURES accuracy on questions whose answers are already known,
BEFORE any recovered answer is allowed into the question pool.

Gating rule: an answer is accepted only on UNANIMOUS agreement across all
responding models. Disagreement means drop, not majority-vote — `INSIGHTS.md`
§4: "low coverage beats wrong labels. A wrong label *biases* every downstream
count; a missing label only *shrinks the sample*."

    python src/ga_consensus_answer.py --validate --sample 60   # measure first
    python src/ga_consensus_answer.py --run                    # then apply
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_registry import available  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"

LETTER = re.compile(r"\b([abcd])\b", re.I)

# How many models must actually answer before their agreement counts.
# Accuracy is the priority here, not coverage: a missing label only
# shrinks the sample, a wrong one biases every downstream count
# (INSIGHTS.md §4).
MIN_AGREE = 3

PROMPT = """You are answering a question from the SSC CGL exam (India).

{stem}

{options}

Reply with ONLY the letter of the correct option: a, b, c, or d.
No explanation. No punctuation. Just the single letter."""


def fmt_options(opts) -> str:
    lines = []
    for o in opts:
        if isinstance(o, (list, tuple)) and len(o) >= 2:
            lines.append(f"{o[0]}) {o[1]}")
        else:
            lines.append(str(o))
    return "\n".join(lines)


def parse_letter(raw: str) -> str | None:
    """Pull a single option letter out of a model reply, tolerating chatter."""
    if not raw:
        return None
    txt = raw.strip().lower()
    # strip <think>...</think> blocks emitted by reasoning models
    txt = re.sub(r"<think>.*?</think>", " ", txt, flags=re.S)
    txt = txt.strip()
    if not txt:
        return None
    if txt[0] in "abcd" and (len(txt) == 1 or not txt[1].isalpha()):
        return txt[0]
    m = LETTER.search(txt)
    return m.group(1).lower() if m else None


def ask_all(q: dict, models) -> dict:
    prompt = PROMPT.format(stem=q.get("stem", ""),
                           options=fmt_options(q.get("options", [])))
    votes = {}
    for m in models:
        if not m.available:
            continue
        try:
            # reasoning models need headroom for their thinking tokens
            budget = 900 if "deepseek" in m.model_id else 24
            votes[m.name] = parse_letter(
                m.generate(prompt, max_tokens=budget, timeout=120))
        except Exception:  # noqa: BLE001 - one model failing must not stop the run
            votes[m.name] = None
    good = [v for v in votes.values() if v]
    # STRICT unanimity: every model that ANSWERED must agree, and at least
    # MIN_AGREE of them must have answered.
    #
    # The old rule accepted 2 agreeing models regardless of how many were
    # asked, so adding a 4th model RAISED coverage 60%->80% while DROPPING
    # accuracy 88.9%->78.1% — the opposite of what consensus should do. A
    # 2-of-4 agreement is not a consensus, it is two models and two abstentions.
    unanimous = (len(set(good)) == 1 and len(good) >= MIN_AGREE)
    return {"votes": votes,
            "consensus": good[0] if unanimous else None,
            "unanimous": unanimous,
            "n_responded": len(good)}


def load_ga() -> list[dict]:
    return json.loads((OUT / "ga_questions.json").read_text())


def validate(sample: int) -> dict:
    """Measure consensus accuracy against questions whose answer is known."""
    ga = load_ga()
    known = [q for q in ga
             if q.get("answer") in {"a", "b", "c", "d"}
             and q.get("options") and len(q.get("options", [])) == 4
             and q.get("stem")]
    random.seed(20260818)
    pick = random.sample(known, min(sample, len(known)))

    models = available()
    live = [m for m in models if m.available]
    print("=" * 68)
    print("GA CONSENSUS — VALIDATION (measure before trusting)")
    print("=" * 68)
    print(f"models live : {', '.join(m.name for m in live)}")
    print(f"sample      : {len(pick)} questions with known answers\n")

    per_model = Counter()
    per_model_n = Counter()
    unan_correct = unan_total = 0
    covered = 0

    for i, q in enumerate(pick, 1):
        truth = q["answer"]
        r = ask_all(q, live)
        for name, v in r["votes"].items():
            if v:
                per_model_n[name] += 1
                if v == truth:
                    per_model[name] += 1
        if r["unanimous"]:
            unan_total += 1
            covered += 1
            if r["consensus"] == truth:
                unan_correct += 1
        if i % 10 == 0:
            print(f"  ...{i}/{len(pick)}")

    print("\n" + "-" * 68)
    print("PER-MODEL ACCURACY (all answers, ungated)")
    print("-" * 68)
    for name in per_model_n:
        acc = per_model[name] / per_model_n[name]
        print(f"  {name:<10} {per_model[name]:>3}/{per_model_n[name]:<3} = {100*acc:5.1f}%")

    cov = covered / len(pick) if pick else 0
    acc = unan_correct / unan_total if unan_total else 0
    print("\n" + "-" * 68)
    print("UNANIMOUS-GATED (what would actually enter the pool)")
    print("-" * 68)
    print(f"  coverage  : {covered}/{len(pick)} = {100*cov:.1f}%")
    print(f"  accuracy  : {unan_correct}/{unan_total} = {100*acc:.1f}%")
    print(f"\n  => gating trades {100*(1-cov):.0f}% coverage for accuracy")

    res = {"sample": len(pick), "models": [m.name for m in live],
           "per_model": {k: {"correct": per_model[k], "n": per_model_n[k],
                             "acc": per_model[k]/per_model_n[k]} for k in per_model_n},
           "gated": {"coverage": cov, "accuracy": acc,
                     "n_unanimous": unan_total, "n_correct": unan_correct}}
    (OUT / "ga_consensus_validation.json").write_text(json.dumps(res, indent=2))
    print(f"\n  written -> out/ga_consensus_validation.json")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--sample", type=int, default=60)
    a = ap.parse_args()
    if a.validate:
        validate(a.sample)
    else:
        print("run with --validate first; --run is gated on the measured accuracy")


if __name__ == "__main__":
    main()
