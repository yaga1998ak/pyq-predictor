"""Data model and taxonomy loading.

One rule governs this whole project: every number that reaches a report comes
from code in this repo, never from a language model. The LLM's only job is
turning question text into a topic label (see tag.py). Counting, weighting and
forecasting happen here, deterministically, so they can be re-run and audited.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
TAXONOMY_DIR = REPO / "taxonomy"
DATA = REPO / "data"


@dataclass
class Question:
    """A single question from a past paper."""

    qid: str
    year: int
    exam: str
    text: str
    section: str | None = None
    topic: str | None = None          # assigned by the tagger
    tagger_confidence: float | None = None
    options: list[str] = field(default_factory=list)
    answer: str | None = None
    source_pdf: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Paper:
    """One exam sitting: a year's worth of questions."""

    year: int
    exam: str
    questions: list[Question]
    shift: str | None = None          # SSC runs multiple shifts per day
    # "official" or "memory_based". Coaching sites publish reconstructed
    # ("similar"/memory-based) papers alongside official ones. They are useful
    # signal but noisier, and mixing them in silently means you cannot tell
    # whether a trend is real or an artefact of reconstruction quality.
    source_type: str = "official"
    date_label: str | None = None     # e.g. "12-Sep" -- distinguishes sittings
    # "ok" or "degraded_spacing". Degraded papers still tag, just less reliably;
    # keep them and MEASURE their effect rather than discarding data on a hunch.
    text_quality: str = "ok"

    @property
    def topic_counts(self) -> Counter:
        return Counter(q.topic for q in self.questions if q.topic)

    @property
    def n_tagged(self) -> int:
        return sum(1 for q in self.questions if q.topic)


class Taxonomy:
    """Topic list for an exam, loaded from taxonomy/*.yaml."""

    def __init__(self, spec: dict):
        self.exam = spec["exam"]
        self.questions_per_paper = spec.get("questions_per_paper")
        self.sections = spec["sections"]
        self.topics: list[str] = []
        self.topic_to_section: dict[str, str] = {}
        for section, body in self.sections.items():
            for topic in body["topics"]:
                self.topics.append(topic)
                self.topic_to_section[topic] = section

    @classmethod
    def load(cls, name: str = "ssc_cgl") -> "Taxonomy":
        with open(TAXONOMY_DIR / f"{name}.yaml") as fh:
            return cls(yaml.safe_load(fh))

    def section_size(self, section: str) -> int:
        return self.sections[section]["questions"]

    def validate(self, topic: str) -> bool:
        return topic in self.topic_to_section

    def __len__(self) -> int:
        return len(self.topics)


def save_papers(papers: list[Paper], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "year": p.year,
            "exam": p.exam,
            "shift": p.shift,
            "source_type": p.source_type,
            "text_quality": p.text_quality,
            "date_label": p.date_label,
            "questions": [q.to_dict() for q in p.questions],
        }
        for p in papers
    ]
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)


def load_papers(path: Path) -> list[Paper]:
    with open(path) as fh:
        payload = json.load(fh)
    return [
        Paper(
            year=item["year"],
            exam=item["exam"],
            shift=item.get("shift"),
            source_type=item.get("source_type", "official"),
            text_quality=item.get("text_quality", "ok"),
            date_label=item.get("date_label"),
            questions=[Question(**q) for q in item["questions"]],
        )
        for item in payload
    ]
