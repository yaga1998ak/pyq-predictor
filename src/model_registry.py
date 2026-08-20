"""Unified access to every model available, with health checks and fallback.

MODEL ALLOCATION IS BY MEASURED COMPETENCE, NOT BY VOTING (spec §13, §69).

The backtests in this repo already settled what models may and may not do:

  * `HANDOVER.md` §3 - a local 7B tagger scored 35% precision against ~80% for
    regex rules, and failed *systematically* (872 questions dumped into one
    label, `coding_decoding` never assigned once in 7,858). Models do not tag
    reasoning / quant / english.
  * `README.md` - "No number in any output comes from a language model."
    Counting, weighting and forecasting are numpy. Models never see a number
    they could hallucinate a replacement for.
  * `INSIGHTS.md` §4 - the one place models genuinely add value is GENERAL
    AWARENESS, "where the signal is in named entities rather than phrasing,
    and where rules only reach 9.9 of 25 questions per paper."

So models are used for exactly three jobs:

    1. GA current-affairs extraction  (genuinely daily-varying input)
    2. GA entity / knowledge-zone labelling
    3. Cross-model fact-checking, where disagreement is the confidence signal

Everything else is deterministic. A model being unavailable degrades those
three jobs and nothing else - the paper still builds.
"""

from __future__ import annotations

import io
import json
import os
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"


def load_env() -> dict:
    """Read .env without exporting secrets into the process env of children."""
    env = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    for k, v in os.environ.items():  # real env wins
        if k in ("GOOGLE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
            env[k] = v
    return env


# Transient upstream conditions. 503 in particular is routine on the Gemini
# free tier and must not fail an unattended daily run.
RETRY_CODES = {429, 500, 502, 503, 504}
MAX_TRIES = 4


def _post(url: str, payload: dict, headers: dict, timeout: int) -> dict:
    body = json.dumps(payload).encode()
    last: Exception | None = None
    for attempt in range(MAX_TRIES):
        try:
            req = urllib.request.Request(url, data=body, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            # A quota exhaustion (insufficient_quota) will never recover by
            # retrying, so surface it immediately rather than burning 4 tries.
            if e.code == 429:
                peek = e.read()
                if b"insufficient_quota" in peek:
                    raise urllib.error.HTTPError(
                        e.url, e.code, e.reason, e.headers, io.BytesIO(peek))
                last = urllib.error.HTTPError(
                    e.url, e.code, e.reason, e.headers, io.BytesIO(peek))
            elif e.code in RETRY_CODES:
                last = e
            else:
                raise
        except (TimeoutError, urllib.error.URLError, socket.timeout) as e:
            last = e
        if attempt < MAX_TRIES - 1:
            time.sleep(2 ** attempt * 1.5)  # 1.5s, 3s, 6s
    raise last  # type: ignore[misc]


@dataclass
class Model:
    name: str
    kind: str          # gemini | openai | anthropic | ollama
    model_id: str
    available: bool = False
    throttled: bool = False
    note: str = ""

    def generate(self, prompt: str, max_tokens: int = 1024,
                 timeout: int = 120) -> str:
        env = load_env()
        if self.kind == "gemini":
            url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
                   f"{self.model_id}:generateContent?key={env['GOOGLE_API_KEY']}")
            r = _post(url, {"contents": [{"parts": [{"text": prompt}]}],
                            "generationConfig": {"maxOutputTokens": max_tokens}},
                      {"Content-Type": "application/json"}, timeout)
            return r["candidates"][0]["content"]["parts"][0]["text"]

        if self.kind == "openai":
            r = _post("https://api.openai.com/v1/chat/completions",
                      {"model": self.model_id, "max_tokens": max_tokens,
                       "messages": [{"role": "user", "content": prompt}]},
                      {"Authorization": f"Bearer {env['OPENAI_API_KEY']}",
                       "Content-Type": "application/json"}, timeout)
            return r["choices"][0]["message"]["content"]

        if self.kind == "anthropic":
            r = _post("https://api.anthropic.com/v1/messages",
                      {"model": self.model_id, "max_tokens": max_tokens,
                       "messages": [{"role": "user", "content": prompt}]},
                      {"x-api-key": env["ANTHROPIC_API_KEY"],
                       "anthropic-version": "2023-06-01",
                       "Content-Type": "application/json"}, timeout)
            return r["content"][0]["text"]

        if self.kind == "ollama":
            base = env.get("OLLAMA_BASE_URL", "http://localhost:11434")
            r = _post(f"{base}/api/generate",
                      {"model": self.model_id, "prompt": prompt, "stream": False,
                       "options": {"num_predict": max_tokens}},
                      {"Content-Type": "application/json"}, timeout)
            return r.get("response", "")

        raise ValueError(f"unknown kind {self.kind}")

    def health(self, timeout: int = 60) -> "Model":
        try:
            out = self.generate("Reply with exactly: OK", max_tokens=256,
                                timeout=timeout)
            self.available = bool(out.strip())
            self.note = "ok" if self.available else "empty response"
        except urllib.error.HTTPError as e:
            body = e.read().decode()[:120].replace("\n", " ")
            # 429 = quota/rate wall. The key is VALID and the quota resets, so
            # the model stays available and the budget ledger paces it. Only a
            # real auth/permission failure disables a model.
            if e.code == 429:
                self.available = True
                self.throttled = True
                self.note = f"throttled (HTTP 429) - key valid, quota resets"
            else:
                self.available = False
                self.note = f"HTTP {e.code}: {body}"
        except Exception as e:  # noqa: BLE001 - health check must never raise
            self.available = False
            self.note = f"{type(e).__name__}: {str(e)[:90]}"
        return self


def registry() -> list[Model]:
    env = load_env()
    models: list[Model] = []
    if env.get("GOOGLE_API_KEY"):
        models.append(Model("gemini", "gemini",
                            env.get("GEMINI_MODEL", "gemini-flash-latest")))
    if env.get("OPENAI_API_KEY"):
        models.append(Model("openai", "openai", env.get("OPENAI_MODEL", "gpt-4o")))
    if env.get("ANTHROPIC_API_KEY"):
        models.append(Model("claude", "anthropic",
                            env.get("ANTHROPIC_MODEL", "claude-sonnet-5")))
    if env.get("OLLAMA_BASE_URL"):
        models.append(Model("qwen", "ollama", env.get("OLLAMA_MODEL", "qwen2.5:7b")))
        if env.get("OLLAMA_LARGE"):
            models.append(Model("qwen14", "ollama", env["OLLAMA_LARGE"]))
        if env.get("OLLAMA_REASONER"):
            models.append(Model("deepseek", "ollama", env["OLLAMA_REASONER"]))
    return models


def available(check: bool = True) -> list[Model]:
    ms = registry()
    if not check:
        return ms
    return [m.health() for m in ms]


def consensus(prompt: str, models: list[Model], max_tokens: int = 512) -> dict:
    """Ask every available model the same question. Disagreement is the signal.

    Used ONLY for GA fact-checking. Never for producing a number.
    """
    answers = {}
    for m in models:
        if not m.available:
            continue
        try:
            answers[m.name] = m.generate(prompt, max_tokens=max_tokens).strip()
        except Exception as e:  # noqa: BLE001
            answers[m.name] = f"__ERROR__ {type(e).__name__}"
    norm = [a.lower().strip(" .\n") for a in answers.values()
            if not a.startswith("__ERROR__")]
    agree = len(set(norm)) == 1 if norm else False
    return {"answers": answers, "unanimous": agree, "n_responded": len(norm)}


if __name__ == "__main__":
    print("=" * 62)
    print("MODEL REGISTRY — health check")
    print("=" * 62)
    for m in available():
        flag = "OK " if m.available else "DOWN"
        print(f"  [{flag}] {m.name:<10} {m.model_id:<22} {m.note[:60]}")
    live = [m for m in available(check=False)]
    print(f"\n  {len(live)} configured")
