"""Model roster — add a new model to YAGA without touching code.

Everything in YAGA that should evolve is declared as DATA: hypotheses in
`yaga_proposals.json`, exams in `exam_profile.py`, and models here. When a
better model appears, you add a line; YAGA benchmarks it against the same
known-answer questions as everything else and promotes it only if it measures
better. No model is trusted because it is new.

    out/yaga_models.json
    [
      {"name": "qwen14", "provider": "ollama", "model": "qwen2.5:14b",
       "role": "recall", "enabled": true},
      ...
    ]

  role  recall   bulk local work, unlimited calls
        judgment scarce cloud calls, budget-capped
        probe    benchmark-only; never used in production until promoted

A model added as `probe` participates in benchmarks but not in answering, so a
newcomer can be measured safely before it influences anything.

    python src/model_roster.py --list
    python src/model_roster.py --add qwen3 ollama qwen3:14b --role probe
    python src/model_roster.py --sync          # write defaults from .env
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILE = ROOT / "out" / "yaga_models.json"

VALID_ROLES = ("recall", "judgment", "probe")
VALID_PROVIDERS = ("ollama", "gemini", "openai", "anthropic")


def load() -> list[dict]:
    if not FILE.exists():
        return []
    try:
        raw = json.loads(FILE.read_text())
    except Exception:
        return []
    out = []
    for m in raw if isinstance(raw, list) else []:
        if m.get("provider") not in VALID_PROVIDERS:
            m["_invalid"] = f"unknown provider {m.get('provider')!r}"
        elif m.get("role") not in VALID_ROLES:
            m["_invalid"] = f"unknown role {m.get('role')!r}"
        out.append(m)
    return out


def save(models: list[dict]) -> None:
    FILE.parent.mkdir(parents=True, exist_ok=True)
    FILE.write_text(json.dumps(models, indent=2))


def active(role: str | None = None) -> list[dict]:
    ms = [m for m in load() if m.get("enabled") and not m.get("_invalid")]
    if role:
        ms = [m for m in ms if m.get("role") == role]
    return ms


def sync_from_env() -> list[dict]:
    """Seed the roster from what .env already configures. Idempotent."""
    import model_registry
    env = model_registry.load_env()
    seed = []
    if env.get("OLLAMA_BASE_URL"):
        seed.append({"name": "qwen", "provider": "ollama",
                     "model": env.get("OLLAMA_MODEL", "qwen2.5:7b"),
                     "role": "recall", "enabled": True})
        if env.get("OLLAMA_LARGE"):
            seed.append({"name": "qwen14", "provider": "ollama",
                         "model": env["OLLAMA_LARGE"], "role": "recall",
                         "enabled": True})
        if env.get("OLLAMA_REASONER"):
            seed.append({"name": "deepseek", "provider": "ollama",
                         "model": env["OLLAMA_REASONER"], "role": "recall",
                         "enabled": True})
    if env.get("GOOGLE_API_KEY"):
        seed.append({"name": "gemini", "provider": "gemini",
                     "model": env.get("GEMINI_MODEL", "gemini-flash-latest"),
                     "role": "judgment", "enabled": True})
    if env.get("OPENAI_API_KEY"):
        seed.append({"name": "openai", "provider": "openai",
                     "model": env.get("OPENAI_MODEL", "gpt-4o"),
                     "role": "judgment", "enabled": True})
    cur = {m["name"]: m for m in load()}
    for m in seed:
        cur.setdefault(m["name"], m)      # never overwrite a human edit
    out = list(cur.values())
    save(out)
    return out


def accuracy_table() -> dict:
    """Latest measured accuracy per model, from the benchmark history."""
    h = ROOT / "out" / "model_history.json"
    try:
        hist = json.loads(h.read_text())
        return hist[-1].get("per_model", {}) if hist else {}
    except Exception:
        return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--sync", action="store_true")
    ap.add_argument("--add", nargs=3, metavar=("NAME", "PROVIDER", "MODEL"))
    ap.add_argument("--role", default="probe", choices=VALID_ROLES)
    ap.add_argument("--disable", metavar="NAME")
    a = ap.parse_args()

    if a.sync:
        ms = sync_from_env()
        print(f"  roster seeded from .env: {len(ms)} models")
    elif a.add:
        name, provider, model = a.add
        ms = load()
        if any(m.get("name") == name for m in ms):
            print(f"  {name} already in the roster")
        else:
            ms.append({"name": name, "provider": provider, "model": model,
                       "role": a.role, "enabled": True})
            save(ms)
            print(f"  added {name} ({model}) as {a.role}")
            if a.role == "probe":
                print("  -> it will be BENCHMARKED but not used in answering")
                print("     until it measures better than what is in place")
    elif a.disable:
        ms = load()
        for m in ms:
            if m.get("name") == a.disable:
                m["enabled"] = False
        save(ms)
        print(f"  disabled {a.disable}")

    acc = accuracy_table()
    print("\nROSTER")
    print(f"  {'name':<12}{'role':<11}{'model':<22}{'enabled':<9}{'measured':>9}")
    for m in load():
        bad = m.get("_invalid")
        a_ = acc.get(m.get("name"))
        print(f"  {m.get('name',''):<12}{m.get('role',''):<11}{m.get('model',''):<22}"
              f"{str(m.get('enabled')):<9}"
              f"{(f'{100*a_:.1f}%' if a_ is not None else '—'):>9}"
              + (f"   INVALID: {bad}" if bad else ""))


if __name__ == "__main__":
    main()
