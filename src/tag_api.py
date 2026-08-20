"""Topic tagging via the Claude API — the accuracy path.

Why this exists alongside tag.py (local Ollama): a 7B/8B local model has to be
coaxed into emitting a valid label at all, and its accuracy on Indian-exam
content is unmeasured. This path removes both problems:

  structured outputs   output_config.format pins the reply to a JSON schema, so
                       a malformed reply is impossible -- not "rare", impossible.
                       Combined with a topic INDEX rather than a topic name,
                       an invalid label cannot be represented.
  prompt caching       the taxonomy prefix is identical for every question, so
                       it is written once and read at ~0.1x for the rest of the run.
  batch API            50% off list price, and 7,861 questions is far inside the
                       100k-per-batch ceiling.

Cost is dominated by INPUT tokens (a long taxonomy, a short answer), which is
exactly the shape prompt caching is for -- so run the estimate before assuming
this is expensive.

Auth: export ANTHROPIC_API_KEY, or `ant auth login` (the SDK reads the profile).

    python src/tag_api.py --estimate                    # cost only, no API calls
    python src/tag_api.py --papers data/parsed/papers.json --limit 200
    python src/tag_api.py --papers data/parsed/papers.json --batch
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from schema import Taxonomy, load_papers, save_papers, REPO

# List price per MTok (input, output). Batch API halves both.
PRICING = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}
DEFAULT_MODEL = "claude-opus-5"


def build_system(taxonomy: Taxonomy) -> list[dict]:
    """Stable prefix, cached across every question in the run.

    Everything that does not vary per question belongs here. The question itself
    goes in the user turn, after the cache breakpoint -- putting it in the system
    prompt would change the prefix on every request and cache nothing.
    """
    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(taxonomy.topics))
    return [
        {
            "type": "text",
            "text": (
                "You classify questions from Indian competitive exams (SSC CGL) "
                "into a fixed topic taxonomy.\n\n"
                f"TOPICS:\n{numbered}\n\n"
                "Choose the single best matching topic number. Judge by the skill "
                "the question actually tests, not by surface vocabulary: a word "
                "problem about trains tests time-speed-distance, not vocabulary. "
                "Set confidence below 0.5 when the question is genuinely ambiguous "
                "or spans two topics."
            ),
            "cache_control": {"type": "ephemeral"},
        }
    ]


def output_schema(n_topics: int) -> dict:
    """A topic INDEX, not a name.

    An out-of-range integer is detectable; a plausible-but-wrong topic string is
    not. Schema-constrained decoding plus an index means every reply maps onto a
    real taxonomy entry or fails loudly.
    """
    return {
        "type": "json_schema",
        "schema": {
            "type": "object",
            "properties": {
                "topic_number": {"type": "integer"},
                "confidence": {"type": "number"},
            },
            "required": ["topic_number", "confidence"],
            "additionalProperties": False,
        },
    }


def estimate_cost(n_questions: int, taxonomy: Taxonomy, model: str, batch: bool) -> dict:
    """Rough cost projection. Token counts are estimates, not measurements.

    Call --estimate before a full run: with caching, the taxonomy is billed once
    at the write rate and then at ~0.1x, so the naive
    (questions x full-prompt) figure overstates the real cost several-fold.
    """
    topics_tokens = sum(len(t) // 3 + 2 for t in taxonomy.topics)
    system_tokens = topics_tokens + 120
    per_question_tokens = 180          # question text, after the cache breakpoint
    output_tokens = 20                 # {"topic_number": N, "confidence": X}

    in_rate, out_rate = PRICING.get(model, PRICING[DEFAULT_MODEL])
    if batch:
        in_rate, out_rate = in_rate / 2, out_rate / 2

    cache_write = system_tokens * 1.25 * in_rate / 1e6
    cache_read = system_tokens * 0.1 * in_rate / 1e6 * (n_questions - 1)
    fresh_in = per_question_tokens * in_rate / 1e6 * n_questions
    out = output_tokens * out_rate / 1e6 * n_questions

    return {
        "model": model,
        "batch": batch,
        "questions": n_questions,
        "cached_prefix_tokens": system_tokens,
        "cost_usd": round(cache_write + cache_read + fresh_in + out, 2),
        "cost_no_cache_usd": round(
            (system_tokens + per_question_tokens) * in_rate / 1e6 * n_questions + out, 2
        ),
    }


def tag_sync(client, questions, taxonomy: Taxonomy, model: str, out_path: Path, papers):
    """Sequential path -- for smoke tests and small runs. Use --batch for a full run."""
    system = build_system(taxonomy)
    fmt = output_schema(len(taxonomy.topics))
    ok = failed = 0
    t0 = time.time()

    for i, q in enumerate(questions, 1):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=64,
                system=system,
                output_config={"format": fmt},
                messages=[{"role": "user", "content": q.text[:1500]}],
            )
            if resp.stop_reason == "refusal":
                failed += 1
                continue
            text = next(b.text for b in resp.content if b.type == "text")
            obj = json.loads(text)
            idx = int(obj["topic_number"])
            if 1 <= idx <= len(taxonomy.topics):
                q.topic = taxonomy.topics[idx - 1]
                q.tagger_confidence = float(obj.get("confidence", 0))
                ok += 1
            else:
                failed += 1
        except Exception as exc:
            print(f"  error on {q.qid}: {exc}", file=sys.stderr)
            failed += 1

        if i % 25 == 0 or i == len(questions):
            save_papers(papers, out_path)
            rate = i / (time.time() - t0)
            print(f"  {i}/{len(questions)}  ok={ok} failed={failed}  {rate:.1f} q/s")

    return ok, failed


def tag_batch(client, questions, taxonomy: Taxonomy, model: str, out_path: Path, papers):
    """Batch API -- 50% off, results in under an hour for a run this size.

    Results come back in arbitrary order, so they are keyed by custom_id (the
    qid). Never zip them against the input list.
    """
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    system = build_system(taxonomy)
    fmt = output_schema(len(taxonomy.topics))

    requests = [
        Request(
            custom_id=q.qid,
            params=MessageCreateParamsNonStreaming(
                model=model,
                max_tokens=64,
                system=system,
                output_config={"format": fmt},
                messages=[{"role": "user", "content": q.text[:1500]}],
            ),
        )
        for q in questions
    ]

    batch = client.messages.batches.create(requests=requests)
    print(f"batch {batch.id} submitted ({len(requests)} requests)")
    print("polling; most batches finish well under the 24h ceiling ...")

    while True:
        batch = client.messages.batches.retrieve(batch.id)
        if batch.processing_status == "ended":
            break
        c = batch.request_counts
        print(f"  {batch.processing_status}: processing={c.processing} "
              f"succeeded={c.succeeded} errored={c.errored}", flush=True)
        time.sleep(30)

    by_qid = {q.qid: q for q in questions}
    ok = failed = 0
    for result in client.messages.batches.results(batch.id):
        q = by_qid.get(result.custom_id)
        if q is None or result.result.type != "succeeded":
            failed += 1
            continue
        msg = result.result.message
        if msg.stop_reason == "refusal":
            failed += 1
            continue
        try:
            obj = json.loads(next(b.text for b in msg.content if b.type == "text"))
            idx = int(obj["topic_number"])
        except Exception:
            failed += 1
            continue
        if 1 <= idx <= len(taxonomy.topics):
            q.topic = taxonomy.topics[idx - 1]
            q.tagger_confidence = float(obj.get("confidence", 0))
            ok += 1
        else:
            failed += 1

    save_papers(papers, out_path)
    return ok, failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", default=str(REPO / "data/parsed/papers.json"))
    ap.add_argument("--out", default=str(REPO / "data/tagged/papers.json"))
    ap.add_argument("--taxonomy", default="ssc_cgl")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"default {DEFAULT_MODEL}; cheaper: claude-sonnet-5, claude-haiku-4-5")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", action="store_true", help="Batch API (50%% off)")
    ap.add_argument("--estimate", action="store_true", help="print cost and exit")
    args = ap.parse_args()

    tax = Taxonomy.load(args.taxonomy)
    papers = load_papers(Path(args.papers))
    pending = [q for p in papers for q in p.questions if not q.topic]
    if args.limit:
        pending = pending[: args.limit]

    if args.estimate:
        print(f"\n{len(pending)} untagged questions, taxonomy = {len(tax)} topics\n")
        print(f"{'model':<20}{'sync':>10}{'batch':>10}{'no cache':>12}")
        print("-" * 52)
        for m in PRICING:
            s = estimate_cost(len(pending), tax, m, batch=False)
            b = estimate_cost(len(pending), tax, m, batch=True)
            print(f"{m:<20}{'$'+str(s['cost_usd']):>10}{'$'+str(b['cost_usd']):>10}"
                  f"{'$'+str(s['cost_no_cache_usd']):>12}")
        print("\nEstimates, not measurements -- token counts are approximated. The "
              "'no cache' column\nis what the same run costs without the cached "
              "taxonomy prefix, i.e. the cost of\ngetting the prompt structure wrong.")
        return

    try:
        import anthropic
    except ImportError:
        raise SystemExit("pip install anthropic")

    client = anthropic.Anthropic()  # ANTHROPIC_API_KEY, or an `ant auth login` profile

    est = estimate_cost(len(pending), tax, args.model, args.batch)
    print(f"tagging {len(pending)} questions with {args.model}"
          f"{' (batch)' if args.batch else ''}  est. ${est['cost_usd']}\n")

    runner = tag_batch if args.batch else tag_sync
    ok, failed = runner(client, pending, tax, args.model, Path(args.out), papers)

    print(f"\ntagged {ok}, failed {failed} -> {args.out}")
    if failed:
        print("WARNING: failures are excluded from all counts. If they cluster in "
              "one topic\n         they bias the analysis -- inspect before trusting "
              "the backtest.")
    print("next: python src/eval_tagger.py  (measure accuracy before trusting these labels)")


if __name__ == "__main__":
    main()
