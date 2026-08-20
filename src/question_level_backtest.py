"""Blind backtest: CAN specific questions be predicted at all?

This is the experiment that governs how the daily paper is labelled. The spec
asks for an S+A (direct/structural prediction) rate of >=75%. Rather than assume
that number, we measure the ceiling.

Protocol
--------
Train on 2021-2024. Test against 2025 papers. Two questions:

  1. CEILING  - what fraction of 2025 questions are near-duplicates of a
                question that already existed in 2021-2024? No selection
                strategy can beat this; it is the upper bound on direct
                question-level prediction by recycling.

  2. REALISED - if we actually pick 100 questions from the training pool using
                the best available strategy (frequency x recency), how many
                land in a given 2025 paper? This is what the daily PDF does.

Matching is deliberately generous: exact match on normalised text, plus Jaccard
token overlap at 0.80 and 0.60. A generous matcher can only inflate the ceiling,
so a low number here is a strong result.

No language model is involved. Every number is reproducible.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TAGGED = ROOT / "data" / "tagged" / "rules.json"

TRAIN_YEARS = {2021, 2022, 2023, 2024}
TEST_YEAR = 2025

# Question markers: 'Q.4', 'Q4.', 'Q 4', '4.' at start
Q_MARKER = re.compile(r"^\s*(?:q\s*\.?\s*\d+\s*[.):]?|\d+\s*[.)])\s*", re.I)
NON_ALNUM = re.compile(r"[^a-z0-9\s]+")
WS = re.compile(r"\s+")

# Tokens too common to be useful for pruning the candidate set.
STOP = {
    "the", "a", "an", "of", "is", "are", "was", "were", "in", "on", "at", "to",
    "for", "and", "or", "which", "that", "this", "these", "those", "select",
    "most", "appropriate", "option", "following", "given", "from", "be", "by",
    "as", "it", "its", "if", "then", "with", "will", "has", "have", "had",
}


def normalise(text: str) -> str:
    """Strip question markers and punctuation; lowercase; collapse whitespace."""
    t = text.strip()
    t = Q_MARKER.sub("", t)
    t = t.lower()
    t = NON_ALNUM.sub(" ", t)
    return WS.sub(" ", t).strip()


def tokens(norm: str) -> frozenset[str]:
    return frozenset(w for w in norm.split() if w not in STOP and len(w) > 1)


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if not inter:
        return 0.0
    return inter / len(a | b)


def load_questions():
    papers = json.loads(TAGGED.read_text())
    train, test_papers = [], []
    for p in papers:
        year = p.get("year")
        qs = []
        for q in p.get("questions", []):
            text = (q.get("text") or "").strip()
            if len(text) < 20:  # unusably short / parse noise
                continue
            norm = normalise(text)
            if len(norm) < 15:
                continue
            qs.append({
                "qid": q.get("qid"),
                "norm": norm,
                "tok": tokens(norm),
                "topic": q.get("topic"),
            })
        if year in TRAIN_YEARS:
            train.extend(qs)
        elif year == TEST_YEAR and qs:
            test_papers.append({
                "source": p.get("source_pdf") or p.get("date_label"),
                "source_type": p.get("source_type"),
                "questions": qs,
            })
    return train, test_papers


class Index:
    """Inverted index on rare tokens, for pruning Jaccard comparisons."""

    def __init__(self, items):
        self.items = items
        self.exact = defaultdict(list)
        self.by_token = defaultdict(list)
        df = Counter()
        for it in items:
            df.update(it["tok"])
        self.df = df
        for i, it in enumerate(items):
            self.exact[it["norm"]].append(i)
            # index on the rarest few tokens only
            rare = sorted(it["tok"], key=lambda w: df[w])[:6]
            for w in rare:
                self.by_token[w].append(i)

    def best_match(self, q) -> float:
        if q["norm"] in self.exact:
            return 1.0
        seen, best = set(), 0.0
        cand_tokens = sorted(q["tok"], key=lambda w: self.df.get(w, 0))[:8]
        for w in cand_tokens:
            for i in self.by_token.get(w, ()):
                if i in seen:
                    continue
                seen.add(i)
                s = jaccard(q["tok"], self.items[i]["tok"])
                if s > best:
                    best = s
                    if best >= 0.999:
                        return best
        return best


def main():
    train, test_papers = load_questions()
    n_test_q = sum(len(p["questions"]) for p in test_papers)
    print("=" * 68)
    print("QUESTION-LEVEL PREDICTABILITY — BLIND BACKTEST")
    print("=" * 68)
    print(f"Train: {len(train):,} questions ({min(TRAIN_YEARS)}-{max(TRAIN_YEARS)})")
    print(f"Test:  {n_test_q:,} questions across {len(test_papers)} papers ({TEST_YEAR})")
    print()

    idx = Index(train)

    # ---- 1. CEILING ------------------------------------------------------
    print("-" * 68)
    print("1. CEILING — is any 2025 question a recycled 2021-24 question?")
    print("-" * 68)

    scores, by_type = [], defaultdict(list)
    for p in test_papers:
        for q in p["questions"]:
            s = idx.best_match(q)
            scores.append(s)
            by_type[p["source_type"]].append(s)

    import numpy as np
    arr = np.array(scores)
    for label, thr in [("exact (1.00)", 0.999), ("near-dup (>=0.80)", 0.80),
                       ("loose (>=0.60)", 0.60)]:
        hits = int((arr >= thr).sum())
        print(f"  {label:22s} {hits:5,} / {len(arr):,}  = {100*hits/len(arr):6.2f}%")

    print()
    print("  by source type (near-dup >=0.80):")
    for st, ss in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
        a = np.array(ss)
        h = int((a >= 0.80).sum())
        print(f"    {str(st):24s} {h:5,} / {len(a):,} = {100*h/len(a):6.2f}%")

    # ---- 2. REALISED ------------------------------------------------------
    print()
    print("-" * 68)
    print("2. REALISED — pick the 'best 100' from training pool, score vs 2025")
    print("-" * 68)

    # Strategy: rank training questions by (stem-template frequency x recency).
    # This is the most defensible selection heuristic available.
    stem_count = Counter()
    for it in train:
        stem_count[" ".join(it["norm"].split()[:8])] += 1

    ranked = sorted(
        train,
        key=lambda it: -stem_count[" ".join(it["norm"].split()[:8])],
    )
    # dedupe by normalised text, take top 100
    picked, seen_norm = [], set()
    for it in ranked:
        if it["norm"] in seen_norm:
            continue
        seen_norm.add(it["norm"])
        picked.append(it)
        if len(picked) == 100:
            break

    pick_idx = Index(picked)
    per_paper = []
    for p in test_papers:
        hits = sum(1 for q in p["questions"] if pick_idx.best_match(q) >= 0.80)
        per_paper.append((hits, len(p["questions"])))

    hits_arr = np.array([h for h, _ in per_paper])
    tot_arr = np.array([t for _, t in per_paper])
    print(f"  Selected 100 questions by stem-frequency from {len(train):,} train Qs.")
    print(f"  Mean hits per 2025 paper : {hits_arr.mean():.2f}")
    print(f"  Median                   : {np.median(hits_arr):.1f}")
    print(f"  Max over {len(per_paper)} papers      : {hits_arr.max()}")
    print(f"  Papers with >=1 hit      : {int((hits_arr>=1).sum())} / {len(per_paper)}")
    print(f"  Implied S+A rate         : {100*hits_arr.sum()/tot_arr.sum():.2f}%")

    out = {
        "train_questions": len(train),
        "test_questions": n_test_q,
        "test_papers": len(test_papers),
        "ceiling": {
            "exact": float((arr >= 0.999).mean()),
            "near_dup_080": float((arr >= 0.80).mean()),
            "loose_060": float((arr >= 0.60).mean()),
        },
        "ceiling_by_source_type": {
            str(k): float((np.array(v) >= 0.80).mean()) for k, v in by_type.items()
        },
        "realised": {
            "mean_hits_per_paper": float(hits_arr.mean()),
            "median_hits_per_paper": float(np.median(hits_arr)),
            "max_hits": int(hits_arr.max()),
            "implied_sa_rate": float(hits_arr.sum() / tot_arr.sum()),
        },
    }
    dest = ROOT / "out" / "question_level_backtest.json"
    dest.write_text(json.dumps(out, indent=2))
    print()
    print(f"  written -> {dest.relative_to(ROOT)}")
    return out


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
