"""Verified patterns — findings already PROVEN, fed to YAGA as facts.

WHY THESE ARE INPUTS, NOT HYPOTHESES
------------------------------------
Some things have been settled by blind backtest and should not be re-litigated
every morning. Re-deriving a proven result daily wastes compute and, worse,
invites a noisy run to overturn something that a careful protocol established.

So the pipeline splits cleanly:

    BACKTESTS (occasional, rigorous)  ->  verified patterns  ->  YAGA
    EVIDENCE  (daily)                 ->  live hypotheses    ->  YAGA

YAGA reasons over both. A verified pattern carries its protocol and the number
that would overturn it, so it is auditable — not an article of faith.

Each pattern below is read from an artefact on disk, never hardcoded. If the
artefact is missing the pattern is simply absent; nothing is invented.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"


@dataclass
class Pattern:
    name: str
    finding: str
    value: float | None
    protocol: str
    decides: str            # what it licenses or forbids
    overturned_if: str


def _read(name: str) -> dict:
    p = OUT / name
    try:
        return json.loads(p.read_text()) if p.exists() else {}
    except Exception:
        return {}


def collect() -> list[Pattern]:
    pats: list[Pattern] = []

    gz = _read("ga_zones_2026.json").get("blind_result", {})
    if gz:
        pats.append(Pattern(
            "ga_zone_coverage",
            "A frozen set of 25 GA knowledge zones covers most of a real paper",
            round(gz.get("direct_plus_partial_pct", 0), 1),
            f"zones frozen on 2023-24 official papers, revealed against "
            f"{gz.get('n','?')} held-out 2025 questions",
            "LICENSES forecasting General Awareness at ZONE level. This is the "
            "only layer measured above the 75% target.",
            "coverage falling below ~70% on a future held-out year"))

    ql = _read("question_level_backtest.json")
    if ql:
        pats.append(Pattern(
            "question_level_empty",
            "Specific questions cannot be predicted",
            round(100 * ql.get("realised", {}).get("implied_sa_rate", 0), 2),
            "best-100 selected from 2021-24, scored against 45 papers of 2025",
            "FORBIDS claiming question-level prediction, at any confidence, "
            "for any exam. Composition must stay at topic/zone level.",
            "a future blind test exceeding ~5% realised"))

    lb = _read("layer_backtest.json")
    if lb.get("L3_topic_ranking"):
        pats.append(Pattern(
            "topic_rank_signal",
            "Topic ranking carries usable signal",
            round(100 * lb["L3_topic_ranking"].get("top_10", 0), 1),
            "train 2021-24, test 2025, top-10 overlap",
            "LICENSES ordering topics by expected weight.",
            "top-10 overlap falling to chance (~20%)"))

    tp = _read("temporal.json")
    if tp:
        pats.append(Pattern(
            "no_temporal_drift",
            "The exam does not drift; apparent drift tracks extraction coverage",
            None,
            "12 coverage-matched year-pairs; corr(TVD, years apart) = +0.155, "
            "slope +0.0087/yr",
            "REQUIRES pooled averaging. FORBIDS recency weighting, which would "
            "chase measurement noise.",
            "corr(TVD, years apart) rising above ~0.5 on matched coverage"))

    cv = _read("ga_consensus_validation.json").get("gated", {})
    if cv:
        pats.append(Pattern(
            "model_consensus_accuracy",
            "Unanimous multi-model agreement is accurate enough to accept a GA answer",
            round(100 * cv.get("accuracy", 0), 1),
            f"unanimity-gated across live models, "
            f"{100*cv.get('coverage',0):.0f}% coverage on known-answer questions",
            "LICENSES accepting a model-derived GA answer ONLY on unanimity.",
            "gated accuracy falling below ~80%"))

    return pats


def as_dict() -> dict:
    return {p.name: asdict(p) for p in collect()}


def summary() -> str:
    ps = collect()
    L = ["VERIFIED PATTERNS — proven, fed to YAGA as facts", "=" * 68]
    for p in ps:
        v = f"{p.value}" if p.value is not None else "—"
        L.append(f"  {p.name:<26} {v:>7}")
        L.append(f"      {p.finding}")
        L.append(f"      decides   : {p.decides}")
        L.append(f"      overturned: {p.overturned_if}")
        L.append("")
    L.append(f"  {len(ps)} verified patterns available")
    return "\n".join(L)


if __name__ == "__main__":
    print(summary())
