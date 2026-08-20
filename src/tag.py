"""Topic tagging via a local Ollama model.

This is the ONLY place a language model touches the pipeline, and its output is
constrained to a label from the taxonomy -- never a number, never a judgement
about what will appear. Everything downstream is arithmetic on these labels.

Two safeguards matter:

  1. Any label outside the taxonomy is rejected and retried, then left untagged.
     An invented topic silently becomes a new category and corrupts every
     year-over-year comparison.
  2. Checkpointing after each batch. Tagging thousands of questions on an 8B
     model takes hours; losing it to a crash is avoidable.

Reasoning models (deepseek-r1) emit <think>...</think> before answering; that is
stripped before JSON parsing.

Run:  python src/tag.py --papers data/parsed/papers.json --model deepseek-r1:8b
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests

from schema import Taxonomy, load_papers, save_papers, REPO

OLLAMA = "http://localhost:11434"
THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)


def build_prompt(question: str, taxonomy: Taxonomy) -> str:
    """Ask for a NUMBER, not a string.

    Presenting topics grouped under section headings invites the model to answer
    with the heading ("quantitative_aptitude") rather than a topic, which then
    fails validation -- that alone accounted for most "invalid" replies in
    benchmarking, on models that were otherwise classifying correctly.

    A flat numbered list removes the failure mode entirely: an integer either
    indexes a real topic or it does not, and there is no near-miss string to
    adjudicate.
    """
    lines = [f"{i+1}. {t}" for i, t in enumerate(taxonomy.topics)]
    numbered = "\n".join(lines)
    return f"""Classify this exam question by choosing ONE topic number from the list.

TOPICS:
{numbered}

QUESTION:
{question[:1200]}

Reply with JSON only, using the NUMBER of the best matching topic:
{{"topic_number": <1-{len(taxonomy.topics)}>, "confidence": <0.0-1.0>}}"""


def strip_reasoning(text: str) -> str:
    return THINK_BLOCK.sub("", text).strip()


def call_ollama(prompt: str, model: str, timeout: int = 180) -> str:
    resp = requests.post(
        f"{OLLAMA}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 512},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json().get("response", "")


def parse_reply(raw: str, taxonomy: Taxonomy) -> tuple[str | None, float]:
    cleaned = strip_reasoning(raw)
    match = JSON_OBJ.search(cleaned)
    if not match:
        return None, 0.0
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None, 0.0

    conf = float(obj.get("confidence", 0.0) or 0.0)

    # Preferred path: an index into the taxonomy, which cannot be a near-miss.
    num = obj.get("topic_number")
    if num is not None:
        try:
            i = int(num)
        except (TypeError, ValueError):
            return None, conf
        if 1 <= i <= len(taxonomy.topics):
            return taxonomy.topics[i - 1], conf
        return None, conf

    # Fallback for models that answer with a name anyway.
    topic = str(obj.get("topic", "")).strip().lower().replace(" ", "_")
    if taxonomy.validate(topic):
        return topic, conf
    return None, conf


def tag_question(text: str, taxonomy: Taxonomy, model: str, retries: int = 2):
    prompt = build_prompt(text, taxonomy)
    for attempt in range(retries + 1):
        try:
            topic, conf = parse_reply(call_ollama(prompt, model), taxonomy)
        except requests.RequestException as exc:
            if attempt == retries:
                print(f"    ollama error: {exc}", file=sys.stderr)
                return None, 0.0
            time.sleep(2)
            continue
        if topic:
            return topic, conf
    return None, 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", default=str(REPO / "data/parsed/papers.json"))
    ap.add_argument("--out", default=str(REPO / "data/tagged/papers.json"))
    ap.add_argument("--model", default="deepseek-r1:8b")
    ap.add_argument("--taxonomy", default="ssc_cgl")
    ap.add_argument("--limit", type=int, default=0, help="tag only N questions (smoke test)")
    ap.add_argument("--checkpoint-every", type=int, default=25)
    args = ap.parse_args()

    try:
        requests.get(f"{OLLAMA}/api/tags", timeout=5).raise_for_status()
    except Exception:
        raise SystemExit(f"Ollama not reachable at {OLLAMA}. Start it: brew services start ollama")

    tax = Taxonomy.load(args.taxonomy)
    papers = load_papers(Path(args.papers))

    pending = [q for p in papers for q in p.questions if not q.topic]
    if args.limit:
        pending = pending[: args.limit]
    print(f"tagging {len(pending)} questions with {args.model}\n")

    t0 = time.time()
    ok = failed = 0
    for i, q in enumerate(pending, 1):
        topic, conf = tag_question(q.text, tax, args.model)
        if topic:
            q.topic, q.tagger_confidence, ok = topic, conf, ok + 1
        else:
            failed += 1
        if i % args.checkpoint_every == 0 or i == len(pending):
            save_papers(papers, Path(args.out))
            rate = i / (time.time() - t0)
            eta = (len(pending) - i) / rate / 60 if rate else 0
            print(
                f"  {i}/{len(pending)}  ok={ok} failed={failed}  "
                f"{rate:.2f} q/s  eta {eta:.1f} min"
            )

    save_papers(papers, Path(args.out))
    print(f"\ntagged {ok}, failed {failed} -> {args.out}")
    if failed:
        print(
            f"WARNING: {failed} questions untagged. They are excluded from all counts,\n"
            "         which biases results if failures cluster in one topic. Investigate\n"
            "         before trusting the backtest."
        )
    print("next: python src/eval_tagger.py  (measure accuracy before trusting these labels)")


if __name__ == "__main__":
    main()
