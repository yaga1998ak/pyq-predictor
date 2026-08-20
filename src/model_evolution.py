"""Model evolution — YAGA re-measures its own models and adapts.

Accuracy is a moving target. Local models get replaced, cloud quotas change,
a new model appears. A system that benchmarks once and trusts that number
forever is not evolving; it is coasting on a stale measurement.

So this re-benchmarks periodically, keeps the HISTORY, and promotes whichever
configuration measures best. The history matters as much as the current number:
a model that silently degrades should be visible as a trend, not discovered
when the output is already wrong.

Feeds `model_consensus_accuracy` into YAGA's verified patterns, so the brain
always reasons with the CURRENT measured accuracy rather than a remembered one.

    python src/model_evolution.py --status
    python src/model_evolution.py --benchmark --sample 40
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
HISTORY = OUT / "model_history.json"
LATEST = OUT / "ga_consensus_validation.json"
PY = str(ROOT / ".venv" / "bin" / "python")

STALE_DAYS = 14          # re-benchmark if the measurement is older than this
MIN_ACCEPTABLE = 0.75    # below this, model-derived answers are NOT trusted


def history() -> list[dict]:
    if HISTORY.exists():
        try:
            return json.loads(HISTORY.read_text())
        except Exception:
            pass
    return []


def record(res: dict) -> None:
    h = history()
    g = res.get("gated", {})
    h.append({
        "when": datetime.now().isoformat(timespec="seconds"),
        "models": res.get("models", []),
        "sample": res.get("sample"),
        "accuracy": round(g.get("accuracy", 0), 4),
        "coverage": round(g.get("coverage", 0), 4),
        "per_model": {k: round(v.get("acc", 0), 4)
                      for k, v in (res.get("per_model") or {}).items()},
    })
    HISTORY.write_text(json.dumps(h, indent=2))


def is_stale() -> bool:
    if not LATEST.exists():
        return True
    h = history()
    if not h:
        return True
    try:
        last = datetime.fromisoformat(h[-1]["when"]).date()
    except Exception:
        return True
    return (date.today() - last).days >= STALE_DAYS


def trustworthy() -> tuple[bool, str]:
    """May model-derived GA answers be used at all?"""
    h = history()
    if not h:
        return False, "never benchmarked"
    acc = h[-1]["accuracy"]
    if acc < MIN_ACCEPTABLE:
        return False, (f"gated accuracy {100*acc:.1f}% is below the "
                       f"{100*MIN_ACCEPTABLE:.0f}% floor — answers NOT trusted")
    return True, f"gated accuracy {100*acc:.1f}%"


def benchmark(sample: int = 40) -> dict:
    print(f"  re-benchmarking on {sample} known-answer questions ...", flush=True)
    subprocess.run([PY, str(ROOT / "src" / "ga_consensus_answer.py"),
                    "--validate", "--sample", str(sample)],
                   capture_output=True, text=True, timeout=7200)
    if not LATEST.exists():
        print("  benchmark produced no result"); return {}
    res = json.loads(LATEST.read_text())
    record(res)
    return res


def status() -> str:
    h = history()
    ok, why = trustworthy()
    L = ["MODEL EVOLUTION", "=" * 62,
         f"  benchmarks recorded : {len(h)}",
         f"  measurement stale?  : {'YES — due a re-run' if is_stale() else 'no'}",
         f"  answers trusted?    : {'YES' if ok else 'NO'} ({why})"]
    if h:
        L.append("")
        L.append(f"  {'when':<20}{'acc':>7}{'cov':>7}   models")
        for e in h[-6:]:
            L.append(f"  {e['when'][:16]:<20}{100*e['accuracy']:6.1f}%"
                     f"{100*e['coverage']:6.1f}%   {', '.join(e['models'])}")
        if len(h) >= 2:
            d = h[-1]["accuracy"] - h[-2]["accuracy"]
            L.append("")
            L.append(f"  trend since last: {100*d:+.1f} points")
        L.append("")
        L.append("  per-model (latest):")
        for k, v in (h[-1].get("per_model") or {}).items():
            L.append(f"    {k:<12}{100*v:5.1f}%")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--if-stale", action="store_true",
                    help="benchmark only when the measurement has aged out")
    ap.add_argument("--sample", type=int, default=40)
    a = ap.parse_args()
    if a.benchmark or (a.if_stale and is_stale()):
        benchmark(a.sample)
    print(status())


if __name__ == "__main__":
    main()
