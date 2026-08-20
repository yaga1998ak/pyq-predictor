"""Difficulty scoring for SSC CGL questions.

There is no ground truth here. SSC does not publish per-question difficulty and
the corpus carries no candidate response data, so nothing in this file is a
*measurement* -- it is a set of structural proxies that correlate with how much
work a question takes:

  steps      multi-stage wording ("successive", "then", "respectively") means the
             solver cannot answer in one operation
  numbers    more distinct quantities to track
  length     longer stems carry more conditions to hold in mind
  operations arithmetic/relational operators in the stem

Questions are ranked WITHIN their own topic and split into terciles, which
matters: a hard percentage question and a hard syllogism have nothing in common
on an absolute scale, but "hard for its topic" is meaningful and keeps every
paper's topic mix intact regardless of the difficulty setting.

Treat easy/medium/hard as *relative bands*, not calibrated levels.
"""

from __future__ import annotations

import re

MULTI_STEP = re.compile(
    r"\bsuccessive|\brespectively\b|\bthen\b|\bif\b.{0,60}\bthen\b|"
    r"\bafter (that|which)\b|\bfurther\b|\bremaining\b|\balso\b|"
    r"\bboth\b|\beach of\b|\bin the ratio\b|\bwhat percent",
    re.IGNORECASE,
)
NUMBER = re.compile(r"\d+(?:\.\d+)?")
OPERATOR = re.compile(r"[+\-×÷*/%=<>]|\bper cent\b|\bpercent")
NEGATION = re.compile(r"\bNOT\b|\bincorrect\b|\bexcept\b|\bfalse\b", re.IGNORECASE)


def score(stem: str, options: list[tuple[str, str]] | None = None) -> float:
    """Higher = harder. Unitless; only the ordering within a topic is meaningful."""
    words = len(stem.split())
    numbers = len(set(NUMBER.findall(stem)))
    steps = len(MULTI_STEP.findall(stem))
    ops = len(OPERATOR.findall(stem))

    s = 0.0
    s += min(words / 12.0, 5.0)          # length, capped so essays don't dominate
    s += min(numbers * 0.8, 4.0)         # distinct quantities to track
    s += steps * 1.6                     # each stage compounds the work
    s += min(ops * 0.4, 2.5)
    if NEGATION.search(stem):
        s += 1.2                         # "which is NOT" forces checking every option

    # Numeric options spanning a wide range usually mean a computed answer rather
    # than a recall answer; near-identical options mean a discrimination task.
    if options:
        vals = []
        for _, text in options:
            m = NUMBER.search(text)
            if m:
                try:
                    vals.append(float(m.group()))
                except ValueError:
                    pass
        if len(vals) == 4:
            lo, hi = min(vals), max(vals)
            if lo > 0 and hi / lo < 1.5:
                s += 1.0                 # close options: fine discrimination needed
    return round(s, 2)


def band_within_topic(questions: list[dict]) -> None:
    """Assign each question a 'difficulty' of easy/medium/hard, in place.

    Banding is per topic and by tercile, so every band still contains every topic
    and the paper's composition does not drift when the user picks 'hard'.
    """
    by_topic: dict[str, list[dict]] = {}
    for q in questions:
        q["diff_score"] = score(q["stem"], q.get("options"))
        by_topic.setdefault(q["topic"], []).append(q)

    for group in by_topic.values():
        group.sort(key=lambda q: q["diff_score"])
        n = len(group)
        if n < 3:
            for q in group:
                q["difficulty"] = "medium"
            continue
        cut1, cut2 = n // 3, 2 * n // 3
        for i, q in enumerate(group):
            q["difficulty"] = "easy" if i < cut1 else ("medium" if i < cut2 else "hard")
