"""Turn verified Q+A pairs into study statements.

THE MODEL'S ONLY JOB IS REPHRASING. It is given a question and the commission's
official answer, and must state that as a fact. It is forbidden from adding
detail. This is the same constraint the tagger works under: transform verified
content, never supply knowledge. It matters because Kerala-specific detail --
reform leaders, state schemes, local geography -- is precisely where a model
produces confident, plausible, wrong answers.

A statement that cannot be formed from the question and answer alone is dropped,
not guessed at.
"""
from __future__ import annotations
import json, re, sys, time, argparse
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out" / "kerala_study_notes.json"
OLLAMA = "http://localhost:11434/api/generate"

# taught topics only: method-based and time-bound sections are excluded
TEACH = {"kerala_governance_acts", "indian_constitution", "science", "history",
         "economics", "arts_literature_culture_sports", "basics_of_computer",
         "geography"}

PROMPT = """You are converting an exam question and its OFFICIAL correct answer into a single factual study statement.

RULES:
- Use ONLY facts contained in the question and the correct answer.
- Do NOT add dates, names, or detail that are not present.
- Write one clear declarative sentence a student can revise from.
- No preamble. Output the sentence only.

QUESTION: {stem}
OFFICIAL CORRECT ANSWER: {ans}

Study statement:"""


def make(stem, ans, model):
    try:
        r = requests.post(OLLAMA, json={
            "model": model,
            "prompt": PROMPT.format(stem=stem[:520], ans=ans[:180]),
            "stream": False,
            "options": {"temperature": 0, "num_predict": 90}}, timeout=45)
        s = " ".join(r.json().get("response", "").split())
        s = re.sub(r'^(Study statement:|Statement:)\s*', '', s, flags=re.I).strip().strip('"')
        return s if 15 < len(s) < 400 else None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:14b")
    a = ap.parse_args()
    pairs = [q for q in json.load(open(ROOT / "out/kerala_qa_pairs.json"))
             if q["topic"] in TEACH]
    done = json.load(open(OUT)) if OUT.exists() else {}
    print(f"{len(pairs)} facts to write; {len(done)} already done", flush=True)
    t0 = time.time()
    for i, q in enumerate(pairs, 1):
        uid = f"{q['src']}#{q['qno']}"
        if uid in done:
            continue
        s = make(q["stem"], q["answer_text"], a.model)
        if s:
            done[uid] = {"topic": q["topic"], "note": s,
                         "src": q["src"], "qno": q["qno"],
                         "answer": q["answer_text"]}
        if i % 25 == 0:
            json.dump(done, open(OUT, "w"), indent=1)
            print(f"  {i}/{len(pairs)}  ({time.time()-t0:.0f}s)", flush=True)
    json.dump(done, open(OUT, "w"), indent=1)
    print(f"DONE {len(done)} study statements -> {OUT}")


if __name__ == "__main__":
    main()
