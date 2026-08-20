"""Tag Kerala PSC degree-level Main questions into the published taxonomy.

The LLM assigns a LABEL and nothing else. Every count, weight and forecast
downstream is arithmetic on these labels -- and crucially, the SECTION WEIGHTS
ARE NEVER INFERRED FROM THEM: KPSC publishes the mark distribution
(GK 55 / Current Affairs 15 / Aptitude 10 / English 10 / Malayalam 10), so
observed tag counts inform WHICH ARCHETYPES to teach, never how much a section
is worth.

MEASURED BIAS (out/kerala_tagger_eval.json): the tagger undercounts
current_affairs by 6-8 per paper, because a 2023 news item reads as static fact
to a model that already knows it. That bias is recorded, not corrected by hand,
and it is the reason topic counts must not drive section weighting.

Run:  ./.venv/bin/python src/kerala_tag.py --model qwen2.5:14b
"""
from __future__ import annotations
import argparse, json, re, sys, time, glob
from pathlib import Path
import requests
from kerala_parse import extract

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "out" / "kerala_tagged.json"
OLLAMA = "http://localhost:11434/api/generate"

LABELS = [
    "kerala_governance_acts", "history", "geography", "economics",
    "indian_constitution", "science", "arts_literature_culture_sports",
    "basics_of_computer", "current_affairs", "aptitude", "general_english",
    "regional_language",
]

PROMPT = """Classify this Kerala PSC Degree Level Main Examination question into exactly ONE category. Reply with ONLY the number.

1 kerala_governance_acts (Kerala government, administration, local self government, state schemes, Kerala institutions, AND named Acts/legal provisions - these overlap heavily, treat as one)
2 history (Kerala/India/World history, freedom struggle, reform movements, dynasties, renaissance leaders)
3 geography (physical, Indian, Kerala, world geography, rivers, climate, soils)
4 economics (economy, banking, budget, planning, poverty, tax)
5 indian_constitution (articles, fundamental rights, Parliament, judiciary, amendments)
6 science (physics, chemistry, biology, life science, medicine, disease, nutrition, public health)
7 arts_literature_culture_sports (art forms, literature, authors, festivals, sports, awards for these)
8 basics_of_computer (computer, IT, internet, cyber, software)
9 current_affairs (events, awards, appointments, launches, summits from the ~2 years before the exam)
10 aptitude (arithmetic, mental ability, reasoning, series, puzzles, percentages, ratios)
11 general_english (English grammar, vocabulary, tense, idiom, comprehension)
12 regional_language (Malayalam language question; may appear as garbled non-Latin characters)

Question: {q}

Number:"""





def tag_one(q: str, model: str) -> str | None:
    """Return a label from LABELS, or None. An out-of-taxonomy reply is
    rejected rather than invented into a new category."""
    for _ in range(2):
        try:
            r = requests.post(OLLAMA, json={
                "model": model, "prompt": PROMPT.format(q=q[:420]), "stream": False,
                "options": {"temperature": 0, "num_predict": 8}}, timeout=30)
            m = re.search(r'\b([1-9]|1[0-2])\b', r.json().get("response", ""))
            if m:
                return LABELS[int(m.group(1)) - 1]
        except Exception as e:
            print("  retry:", e, file=sys.stderr); time.sleep(2)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:14b")
    a = ap.parse_args()

    pdfs = [f for f in sorted(glob.glob(str(ROOT / "data/raw/kerala/*/*.pdf")))
            if "_excluded" not in f]   # skip non-Malayalam-medium papers
    done = json.load(open(OUT)) if OUT.exists() else {}

    for pdf in pdfs:
        key = Path(pdf).name
        if key in done:
            print(f"skip (done): {key}"); continue
        qs = extract(pdf)
        if len(qs) < 50:
            print(f"skip (only {len(qs)} questions, needs OCR): {key}")
            done[key] = {"_status": "needs_ocr", "_n": len(qs)}
            json.dump(done, open(OUT, "w"), indent=1); continue

        print(f"tagging {key} ({len(qs)} questions)...", flush=True)
        t0 = time.time(); tags = {}
        for i, n in enumerate(sorted(qs), 1):
            tags[str(n)] = tag_one(qs[n], a.model)
            if i % 10 == 0:
                print(f"    {i}/{len(qs)} ({time.time()-t0:.0f}s)", flush=True)
        done[key] = tags
        json.dump(done, open(OUT, "w"), indent=1)   # checkpoint per paper
        bad = sum(1 for v in tags.values() if v is None)
        print(f"  done in {time.time()-t0:.0f}s, untagged={bad}", flush=True)

    print("ALL DONE ->", OUT)


if __name__ == "__main__":
    main()
