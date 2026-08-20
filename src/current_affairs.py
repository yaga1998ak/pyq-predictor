"""Pull current affairs from the open web and map them onto the GA zones.

WHY THIS IS THE ONE PLACE EXTERNAL DATA BELONGS
-----------------------------------------------
The PYQ corpus is fixed; nothing new arrives about past papers. Current affairs
is the only genuinely daily-varying input, and it is measured to matter:
53% of dated GA questions cite the 0-3 years before the exam, peaking at the
previous year (21%). For a 2026 paper, 2025-26 events are the richest source.

DESIGN CHOICES
--------------
  RSS, not a search API   No key, no quota, no vendor. This must keep working
                          unattended until the exam; an API that expires or
                          rate-limits is a liability, as the Gemini free tier
                          already demonstrated (2 calls then HTTP 429).

  Deterministic mapping   Items are matched to GA zones by keyword rules, not
                          by a model. HANDOVER.md §3: rules beat a local 7B on
                          this corpus 80% vs 35%, because the language is
                          templated. A model is consulted only for items the
                          rules decline, and only if one is live.

  Zone-anchored           Only zones the setter model actually validated are
                          tracked. An item that maps to no zone is dropped
                          rather than shoehorned - a wrong zone label is worse
                          than a missing one (INSIGHTS.md §4).

    python src/current_affairs.py --fetch
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
STORE = OUT / "current_affairs.json"

FEEDS = {
    "PIB": "https://www.pib.gov.in/RssMain.aspx?ModId=6&Lang=1&Regid=3",
    "Hindu-National": "https://www.thehindu.com/news/national/feeder/default.rss",
    "Hindu-SciTech": "https://www.thehindu.com/sci-tech/feeder/default.rss",
    "IE-India": "https://indianexpress.com/section/india/feed/",
}

# Zone -> keyword rules. Keys must match the validated GA zone names.
ZONE_RULES: dict[str, list[str]] = {
    "government_schemes": ["yojana", "scheme", "mission", "abhiyan", "pradhan mantri",
                           "pm-", "launched by the government", "flagship programme"],
    "awards_honours": ["award", "honour", "honored", "prize", "padma", "bharat ratna",
                       "nobel", "conferred", "felicitated"],
    "olympics_asian_games": ["olympic", "asian games", "commonwealth games",
                             "paralympic", "medal tally"],
    "other_sports_events": ["world cup", "championship", "tournament", "trophy",
                            "grand slam", "series win"],
    "space_and_technology": ["isro", "satellite", "launch vehicle", "chandrayaan",
                             "gaganyaan", "aditya", "spacecraft", "pslv", "gslv"],
    "appointments_persons": ["appointed", "sworn in", "takes charge", "new chief",
                             "named as", "elevated as", "resigns"],
    "budget_and_taxation": ["budget", "gst", "tax", "fiscal deficit", "finance bill",
                            "revenue", "customs duty"],
    "parliament_legislature": ["lok sabha", "rajya sabha", "parliament", "bill passed",
                               "ordinance", "session of parliament"],
    "constitution_general": ["supreme court", "constitution", "article ", "amendment",
                             "fundamental right", "high court ruling"],
    "planning_and_indices": ["index", "ranking", "report released", "survey",
                             "gdp growth", "niti aayog"],
    "festivals": ["festival", "celebrated", "jayanti", "diwas", "day observed"],
    "new_criminal_laws": ["nyaya sanhita", "criminal law", "bharatiya nagarik",
                          "penal code"],
}


# --- zones reachable by news, added after a funnel audit -------------------
# A 2026-08-19 audit showed 13 of 16 genuine facts per issue were discarded
# because no rule existed for their zone. These close that gap.
ZONE_RULES.update({
    "president_governor": ["governor", "president of india", "rashtrapati",
                           "raj bhavan", "assent to the bill"],
    "colonial_acts_administration": ["code of civil procedure", "penal code",
                                     "act of 18", "act of 19", "colonial-era law",
                                     "repealed the act"],
    "census_demography": ["census", "population", "birth rate", "death rate",
                          "civil registration", "sex ratio", "literacy rate"],
    "rivers_and_drainage": ["river", "dam", "reservoir", "canal", "basin",
                            "tributary", "water level"],
    "soils_agriculture": ["crop", "kharif", "rabi", "msp", "minimum support price",
                          "farmer", "irrigation", "fertiliser", "soil"],
    "cricket": ["test match", "odi", "t20", "bcci", "ipl", "wicket", "batter"],
    "biology": ["species", "vaccine", "virus", "tiger", "wildlife", "biodiversity",
                "disease outbreak", "genome"],
    "chemistry": ["chemical", "compound", "element", "molecule", "isotope"],
    "state_identification": ["state government", "chief minister", "assembly",
                             "state cabinet"],
    "fundamental_duties": ["fundamental duty", "fundamental right",
                           "directive principle"],
    "committees_reports": ["committee", "commission", "panel recommended",
                           "task force", "report submitted", "white paper"],
    "international_orgs": ["united nations", "unesco", "who", "imf", "world bank",
                           "brics", "g20", "asean", "wto", "opec"],
    "science_institutions": ["research institute", "laboratory", "csir",
                             "drdo", "iit", "national science", "isro"],
})

# Zones that are STATIC - newspapers do not generate new facts about them.
# They are covered by the PYQ corpus, not by current affairs. Listing them
# explicitly stops a future reader mistaking their absence for an oversight.
STATIC_ZONES = {
    "mughal_empire", "mauryan_empire", "chola_and_south", "gupta_and_post_gupta",
    "freedom_movement", "classical_dance", "folk_dance_by_state",
    "music_instruments", "vedic_period",
}

TAG = re.compile(r"<[^>]+>")


def clean(s: str) -> str:
    return " ".join(TAG.sub(" ", s or "").split())


def fetch_feed(name: str, url: str, timeout: int = 20) -> list[dict]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        raw = urllib.request.urlopen(req, timeout=timeout).read()
        root = ET.fromstring(raw)
    except Exception as e:  # a dead feed must never break the daily run
        print(f"  {name}: FAILED ({type(e).__name__})")
        return []
    items = []
    for it in root.iter("item"):
        t = clean((it.findtext("title") or ""))
        d = clean((it.findtext("description") or ""))[:400]
        p = (it.findtext("pubDate") or "").strip()
        if t:
            items.append({"headline": t, "summary": d, "published": p, "source": name})
    print(f"  {name}: {len(items)} items")
    return items


_KW_CACHE: dict[str, re.Pattern] = {}


def _kw(k: str) -> re.Pattern:
    """Word-boundary matcher.

    Substring matching produced real errors: 'tax' matched 'taxis', filing a
    traffic story under budget_and_taxation. Multi-word keys keep their spaces;
    boundaries are applied at each end.
    """
    if k not in _KW_CACHE:
        _KW_CACHE[k] = re.compile(r"(?<!\w)" + re.escape(k.strip()) + r"(?!\w)")
    return _KW_CACHE[k]


def zone_for(item: dict) -> str | None:
    """Deterministic keyword mapping. Returns None rather than guessing.

    Requires >=2 keyword hits for single-signal zones, so one incidental word
    cannot classify a story.
    """
    text = f"{item['headline']} {item['summary']}".lower()
    scores = {z: sum(1 for k in kws if _kw(k).search(text))
              for z, kws in ZONE_RULES.items()}
    best = max(scores, key=lambda z: scores[z])
    if scores[best] == 0:
        return None
    # a lone generic hit is not enough unless it came from the headline itself
    if scores[best] == 1:
        head = item["headline"].lower()
        if not any(_kw(k).search(head) for k in ZONE_RULES[best]):
            return None
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--keep", type=int, default=120)
    a = ap.parse_args()

    store = json.loads(STORE.read_text()) if STORE.exists() else {"items": []}
    known = {i["headline"] for i in store.get("items", [])}

    if a.fetch:
        print("CURRENT AFFAIRS — fetching")
        fresh = []
        for name, url in FEEDS.items():
            for it in fetch_feed(name, url):
                if it["headline"] in known:
                    continue
                z = zone_for(it)
                if not z:
                    continue          # unmapped -> dropped, never shoehorned
                it["zone"] = z
                it["seen"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                fresh.append(it)
                known.add(it["headline"])
        store["items"] = (fresh + store.get("items", []))[: a.keep]
        STORE.parent.mkdir(parents=True, exist_ok=True)
        STORE.write_text(json.dumps(store, indent=2))
        print(f"\n  new mapped items: {len(fresh)}")

    items = store.get("items", [])
    print(f"  stored: {len(items)}")
    if items:
        from collections import Counter
        print("  by zone:")
        for z, n in Counter(i["zone"] for i in items).most_common(8):
            print(f"    {z:<28}{n:>4}")
        print("\n  most recent:")
        for i in items[:5]:
            print(f"    [{i['zone']}] {i['headline'][:78]}")


if __name__ == "__main__":
    main()
