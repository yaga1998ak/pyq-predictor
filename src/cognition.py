"""Cognitive layer — the setting team as HUMANS under constraint.

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
It is not a model of consciousness. Nothing here claims to know what a setter
experiences, intends or feels; those are unfalsifiable and have no place in a
brain that audits itself.

It IS a model of documented human decision-making biases. A question paper is
written by a small group, under deadline, inside an institution, from a finite
question bank. Cognitive science has well-established findings about how people
choose under exactly those conditions, and each one leaves a MEASURABLE trace.

Every hypothesis below names the cognitive construct it operationalises, states
the signature it should leave in the data, and states what would refute it.
Where a construct has no measurable signature here, it is omitted rather than
asserted.

CONSTRUCTS OPERATIONALISED
--------------------------
  anchoring          (Tversky & Kahneman) - judgements pull toward a reference
                     point. Signature: this year's composition sits closer to
                     recent papers than to the long-run mean.
  availability       (Tversky & Kahneman) - recent/salient material comes to
                     mind first. Signature: dated references cluster in the
                     years just before the exam.
  satisficing        (Simon) - people take the first adequate option, not the
                     optimal one. Signature: heavy reuse of a few question
                     frames rather than maximally varied construction.
  effort economy     constructing a novel item costs more than instantiating a
                     template. Signature: template-heavy topics carry MORE
                     questions than bespoke ones.
  round-number bias  humans generate and prefer round quantities.
  serial position    (Ebbinghaus / Murdock) - order is not arbitrary; openers
                     and closers get treated differently. Signature: systematic
                     difficulty or length gradient across a section.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter, defaultdict

from brain_core import Claim, _by_paper, _tvd

YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")


def h_anchoring(obs) -> Claim:
    """Do recent papers pull composition more than the long-run mean?"""
    papers = _by_paper(obs)
    order = sorted(papers)
    if len(order) < 30:
        return Claim("cog_anchoring", "Composition anchors on recent papers",
                     "UNDERPOWERED", {"papers": len(order)}, "", {})
    k = max(5, len(order) // 6)
    recent, old = order[-k:], order[:k]
    latest = Counter()
    for pid in order[-1:]:
        latest.update(o.topic for o in papers[pid] if o.topic)
    rc, oc = Counter(), Counter()
    for pid in recent[:-1]:
        rc.update(o.topic for o in papers[pid] if o.topic)
    for pid in old:
        oc.update(o.topic for o in papers[pid] if o.topic)
    d_recent, d_old = _tvd(latest, rc), _tvd(latest, oc)
    ok = d_recent < d_old - 0.02
    return Claim(
        "cog_anchoring",
        "ANCHORING: a new paper resembles recent papers more than older ones",
        "CONFIRMED" if ok else "REJECTED",
        {"tvd_to_recent": round(d_recent, 3), "tvd_to_old": round(d_old, 3),
         "window": k, "papers": len(order)},
        "Setters work from what is in front of them. Weight the recent window "
        "above the deep archive — but only as far as this gap justifies.",
        {"power": f"{len(order)} papers, window {k}",
         "control": "same paper compared against both windows",
         "constraint": "TVD to two disjoint sets is not forced",
         "refuted_by": "tvd_to_recent meeting or exceeding tvd_to_old"})


def h_availability(obs) -> Claim:
    """Do dated references cluster in the years just before the exam?"""
    hits = []
    for o in obs:
        for y in YEAR_RE.findall(o.stem):
            yi = int(y)
            if 1900 <= yi <= 2026:
                hits.append(yi)
    if len(hits) < 300:
        return Claim("cog_availability", "Recent events dominate dated references",
                     "UNDERPOWERED", {"n": len(hits)}, "", {})
    recent = sum(1 for y in hits if y >= 2018) / len(hits)
    ok = recent > 0.25
    return Claim(
        "cog_availability",
        "AVAILABILITY: dated references cluster in the years just before the exam",
        "CONFIRMED" if ok else "REJECTED",
        {"n": len(hits), "share_since_2018": round(100*recent, 1),
         "median_year": int(statistics.median(hits))},
        "What is recent is what comes to mind. Current affairs should be "
        "weighted to the last few years, not spread evenly over history.",
        {"power": f"{len(hits):,} dated references",
         "control": "share measured against all dated references, not a subset",
         "constraint": "year distribution is not compositionally forced",
         "refuted_by": "recent share falling to the base rate of an even spread"})


def h_satisficing(obs) -> Claim:
    """Do setters reuse a few frames rather than construct maximally varied ones?"""
    frames = Counter(o.frame[:60] for o in obs if o.frame)
    if len(frames) < 100:
        return Claim("cog_satisficing", "Frames are reused rather than varied",
                     "UNDERPOWERED", {"frames": len(frames)}, "", {})
    total = sum(frames.values())
    top = sum(c for _, c in frames.most_common(max(1, len(frames)//20)))
    conc = top / total
    ok = conc > 0.25
    return Claim(
        "cog_satisficing",
        "SATISFICING: a small set of frames carries a large share of questions",
        "CONFIRMED" if ok else "REJECTED",
        {"distinct_frames": len(frames), "questions": total,
         "top_5pct_share": round(100*conc, 1)},
        "Under deadline people take the first adequate option. Frames repeat, "
        "so the STRUCTURE of a question is predictable even though its content "
        "is not.",
        {"power": f"{total:,} questions over {len(frames):,} frames",
         "control": "concentration compared against an even spread",
         "constraint": "share of top frames is not simplex-forced",
         "refuted_by": "top-5% share approaching 5%, i.e. no concentration"})


def h_effort_economy(obs) -> Claim:
    """Do cheap-to-construct topics carry more questions than expensive ones?"""
    by_topic = defaultdict(list)
    for o in obs:
        if o.topic:
            by_topic[o.topic].append(o)
    if len(by_topic) < 15:
        return Claim("cog_effort_economy", "Cheap-to-build topics carry more load",
                     "UNDERPOWERED", {"topics": len(by_topic)}, "", {})
    rows = []
    for t, qs in by_topic.items():
        distinct = len({o.frame[:60] for o in qs})
        reuse = len(qs) / max(distinct, 1)      # higher = more templated
        rows.append((reuse, len(qs)))
    rows.sort()
    half = len(rows)//2
    low = statistics.fmean(n for _, n in rows[:half])       # bespoke topics
    high = statistics.fmean(n for _, n in rows[half:])      # templated topics
    ok = high > low * 1.2
    return Claim(
        "cog_effort_economy",
        "EFFORT ECONOMY: templated topics carry more questions than bespoke ones",
        "CONFIRMED" if ok else "REJECTED",
        {"topics": len(rows), "mean_questions_templated": round(high, 1),
         "mean_questions_bespoke": round(low, 1),
         "ratio": round(high/max(low, 1e-9), 2)},
        "A paper is built under a time budget. Topics that can be instantiated "
        "from a template are cheaper to produce, and cheaper topics get used "
        "more.",
        {"power": f"{len(rows)} topics",
         "control": "topics split at the median reuse rate, same corpus",
         "constraint": "question counts are not forced by reuse rate",
         "refuted_by": "ratio falling to ~1.0"})


def h_serial_position(obs) -> Claim:
    """Is question order arbitrary, or is there a gradient across a section?"""
    papers = _by_paper(obs)
    firsts, lasts = [], []
    for qs in papers.values():
        if len(qs) < 20:
            continue
        n = len(qs)//4
        firsts += [o.length for o in qs[:n]]
        lasts += [o.length for o in qs[-n:]]
    if len(firsts) < 300:
        return Claim("cog_serial_position", "Question order carries a gradient",
                     "UNDERPOWERED", {"n": len(firsts)}, "", {})
    f, l = statistics.fmean(firsts), statistics.fmean(lasts)
    ratio = l / max(f, 1e-9)
    ok = abs(ratio - 1.0) > 0.15
    return Claim(
        "cog_serial_position",
        "SERIAL POSITION: stem length differs systematically between the "
        "opening and closing quarter of a paper",
        "CONFIRMED" if ok else "REJECTED",
        {"mean_len_first_quarter": round(f, 1),
         "mean_len_last_quarter": round(l, 1), "ratio": round(ratio, 2)},
        "Order is a construction artefact, not a difficulty ramp — treat any "
        "gradient as weak evidence and never as a scoring strategy.",
        {"power": f"{len(firsts):,} opening vs {len(lasts):,} closing questions",
         "control": "same papers contribute to both groups",
         "constraint": "stem length is not compositionally bound",
         "refuted_by": "ratio within 15% of 1.0"})


COGNITIVE = {
    "cog_anchoring": h_anchoring,
    "cog_availability": h_availability,
    "cog_satisficing": h_satisficing,
    "cog_effort_economy": h_effort_economy,
    "cog_serial_position": h_serial_position,
}
