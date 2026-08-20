"""Forecast the topic distribution for a future paper.

Read the output as what it is: a probability-weighted topic map with honest
uncertainty. It says "expect roughly this many questions from this topic, give
or take this much". It does NOT predict specific questions, and any use of it
that assumes otherwise is misuse.

The model used here should be whichever one WON your backtest -- not whichever
sounds most sophisticated. Pass --model to override.

Run:  python src/predict.py --papers data/tagged/papers.json --year 2026
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from backtest import papers_to_history, counts_vector
from models import all_models
from schema import Taxonomy, load_papers, REPO


def forecast(papers, taxonomy: Taxonomy, model_name: str | None, n_questions: int):
    history, _ = papers_to_history(papers)
    candidates = {m.name: m for m in all_models(taxonomy)}

    if model_name and model_name not in candidates:
        raise SystemExit(f"unknown model '{model_name}'. Options: {sorted(candidates)}")
    model = candidates[model_name] if model_name else candidates["mean_last_3"]

    # Forecasting a real future paper: 25 per section, not the tagger's coverage.
    model.nominal_sections = True
    model.fit(history)
    pred = model.predict(n_questions)
    interval = model.predict_interval(n_questions, n_papers=1)
    return model, pred, interval, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", default=str(REPO / "data/tagged/papers.json"))
    ap.add_argument("--taxonomy", default="ssc_cgl")
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--model", default=None,
                    help="model name that won your backtest; defaults to mean_last_3")
    ap.add_argument("--questions", type=int, default=None)
    ap.add_argument("--out", default=str(REPO / "out/forecast.json"))
    args = ap.parse_args()

    tax = Taxonomy.load(args.taxonomy)
    papers = load_papers(Path(args.papers))
    n_q = args.questions or tax.questions_per_paper

    model, pred, interval, history = forecast(papers, tax, args.model, n_q)
    lo, hi = interval if interval else (None, None)

    print(f"\nForecast for {tax.exam} {args.year}   ({n_q} questions)")
    print(f"model: {model.name}   trained on {len(history)} years "
          f"({history[0][0]}-{history[-1][0]})\n")

    order = np.argsort(-pred)
    current_section = None
    rows = []
    for i in order:
        topic = tax.topics[i]
        section = tax.topic_to_section[topic]
        rows.append({
            "topic": topic,
            "section": section,
            "expected": round(float(pred[i]), 2),
            "lo90": None if lo is None else round(float(lo[i]), 1),
            "hi90": None if hi is None else round(float(hi[i]), 1),
        })

    for section in tax.sections:
        print(f"--- {section}  ({tax.section_size(section)} questions) ---")
        sec_rows = [r for r in rows if r["section"] == section]
        for r in sec_rows:
            rng = "" if r["lo90"] is None else f"   [{r['lo90']:.0f}-{r['hi90']:.0f}]"
            bar = "#" * int(round(r["expected"] * 2))
            print(f"  {r['topic']:<32}{r['expected']:>5.1f}{rng:<12} {bar}")
        print()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(
            {"exam": tax.exam, "year": args.year, "model": model.name,
             "trained_years": [y for y, _ in history], "forecast": rows},
            fh, indent=2,
        )
    print(f"-> {args.out}")
    print(
        "\nHow to use this: allocate study time by expected count, and treat the\n"
        "[lo-hi] range as the real spread. A topic at 2.0 [0-5] is far less certain\n"
        "than one at 2.0 [1-3]. This forecasts TOPIC WEIGHT, never specific questions."
    )


if __name__ == "__main__":
    main()
