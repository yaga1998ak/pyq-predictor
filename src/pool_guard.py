"""Keep the question pools ahead of consumption — no human in the loop.

The daily job consumes 25 questions per section per day and never repeats one.
Without this, the system silently starves: papers keep building until a pool
empties, then quality degrades before anything visibly fails.

This runs before each build. If any section's runway drops below THRESHOLD
days, it regenerates that section's solver-verified pool at a larger target.
Reasoning and quant are synthesised + independently re-derived, so they can be
grown without bound. English and GA come from the real corpus and cannot be
grown here - for those it warns, since the fix is more PYQ data.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
PY = str(ROOT / ".venv" / "bin" / "python")
PER_DAY = 25
THRESHOLD_DAYS = 21          # regenerate when under three weeks of runway
GROW_STEP = 250              # per-archetype target increment

GROWABLE = {
    "reasoning": ("candidates.json", "generate_questions.py"),
    "quant": ("quant_candidates.json", "quant_generate.py"),
}


def runway() -> dict[str, int]:
    sys.path.insert(0, str(ROOT / "src"))
    import daily_run
    return {k: len(v) // PER_DAY for k, v in daily_run.load_pools().items()}


def grow(section: str, target: int) -> bool:
    _, script = GROWABLE[section]
    print(f"  growing {section} -> {target}/archetype ...", flush=True)
    r = subprocess.run([PY, str(ROOT / "src" / script), str(target)],
                       capture_output=True, text=True, timeout=3600)
    return r.returncode == 0


def main() -> None:
    days = runway()
    print("POOL GUARD")
    for k, v in sorted(days.items(), key=lambda kv: kv[1]):
        flag = "  <-- LOW" if v < THRESHOLD_DAYS else ""
        print(f"  {k:<20}{v:>4} days{flag}")

    for section, d in days.items():
        if d >= THRESHOLD_DAYS:
            continue
        if section not in GROWABLE:
            print(f"  WARN {section} low ({d}d) and not growable - needs more PYQ data")
            continue
        path = OUT / GROWABLE[section][0]
        cur = len(json.loads(path.read_text())) if path.exists() else 0
        # infer current per-archetype target and step it up
        target = max(GROW_STEP, (cur // 11) + GROW_STEP)
        grow(section, target)

    print("  done")


if __name__ == "__main__":
    main()
