"""Compose the daily paper from the VALIDATED model of the setting team.

This is the link that makes the register mean something. Everything the setter
brain has confirmed becomes a constraint on how the paper is built; everything
it has rejected is explicitly NOT modelled.

    out/setter_register.json  ->  composition constraints  ->  daily paper

CURRENT CONSTRAINTS DERIVED FROM THE REGISTER
---------------------------------------------
  topic_floor        CONFIRMED  guarantee the contractual topics every paper
  numeric_habits     CONFIRMED  prefer items whose numbers are round
  day_blueprint      OBSERVED   compose to one day-level topic skeleton
  shift_position     REJECTED   do NOT vary anything by shift
  frame_reuse        REJECTED   never reuse a question frame; fresh each time
  cross_section_bal  REJECTED   sections composed independently, no compensation
  difficulty_parity  REJECTED   do not equalise difficulty across shifts
  temporal drift     REJECTED   pooled average, NOT recency weighting

Each constraint cites the finding that justifies it, so a change in the
register changes the paper - which is what "becoming the team's brain" means
operationally.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
NUMS = re.compile(r"\b\d{1,6}\b")


def register() -> dict:
    p = OUT / "setter_register.json"
    return json.loads(p.read_text()).get("tested", {}) if p.exists() else {}


def confirmed(key: str) -> bool:
    return register().get(key, {}).get("verdict") == "CONFIRMED"


# ------------------------------------------------------------ constraints ---
def floor_topics() -> list[str]:
    """Topics the team has no discretion over - guarantee them."""
    r = register().get("topic_floor", {})
    if r.get("verdict") != "CONFIRMED":
        return []
    return list(r.get("evidence", {}).get("in_95pct", []))


def roundness(q: dict) -> float:
    """How well an item matches the team's confirmed numeric habit."""
    if not confirmed("numeric_habits"):
        return 0.0
    vals = [int(m) for m in NUMS.findall(q.get("stem") or "")
            if 1 < int(m) < 10000]
    if not vals:
        return 0.0
    return sum(1 for v in vals if v % 5 == 0) / len(vals)


# Corpus topic labels differ from generated-pool labels for the same concept.
ALIASES = {
    "series_completion": {"series_completion", "number_or_letter_series"},
    "percentage": {"percentage"},
    "one_word_substitution": {"one_word_substitution"},
}


def _matches(topic: str, floor: str) -> bool:
    return topic in ALIASES.get(floor, {floor})


def apply(pool: list[dict], weights: dict[str, float]) -> dict[str, float]:
    """Weights are untouched by the floor.

    An earlier version raised each contractual topic's weight to 1.0. Weights
    are RELATIVE and their scale differs per section, so that handed
    `percentage` 20 of 25 quant questions - the floor swamped the blueprint.
    Presence is guaranteed after selection instead (see `ensure_floor`), which
    is what "appears in every paper" actually means.
    """
    return dict(weights)


def ensure_floor(chosen: list[dict], pool: list[dict]) -> list[dict]:
    """Guarantee >=1 of each contractual topic, swapping out an over-represented one."""
    floors = floor_topics()
    if not floors or not chosen:
        return chosen
    have = Counter(q.get("topic") for q in chosen)
    for f in floors:
        if any(_matches(t, f) for t in have):
            continue
        cand = next((q for q in pool
                     if _matches(q.get("topic") or "", f)
                     and q not in chosen), None)
        if not cand:
            continue  # section genuinely has no such item; do not fake one
        # displace one question from the most over-represented topic
        top = have.most_common(1)[0][0]
        victim = next((q for q in reversed(chosen) if q.get("topic") == top), None)
        if victim:
            chosen[chosen.index(victim)] = cand
            have[top] -= 1
            have[cand.get("topic")] += 1
    return chosen


def rank_key(q: dict) -> float:
    """Tie-breaker within a topic: prefer items matching setter habits."""
    return roundness(q)


def summary() -> str:
    reg = register()
    L = ["MODEL-DRIVEN COMPOSITION", "=" * 62]
    conf = [k for k, v in reg.items() if v["verdict"] == "CONFIRMED"]
    rej = [k for k, v in reg.items() if v["verdict"] == "REJECTED"]
    L.append(f"  constraints applied  : {', '.join(conf) or 'none'}")
    L.append(f"  explicitly NOT modelled: {', '.join(rej) or 'none'}")
    ft = floor_topics()
    if ft:
        L.append(f"  guaranteed topics    : {', '.join(ft)}")
    L.append("")
    L.append("  A change in out/setter_register.json changes the paper.")
    return "\n".join(L)


if __name__ == "__main__":
    print(summary())
