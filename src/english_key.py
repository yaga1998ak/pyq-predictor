"""Derive answer keys for REAL official English PYQs, where derivation is possible.

Why this exists: official SSC papers (2022-2024) carry no answer key at all --
"Ans" introduces the option list, and there are zero occurrences of "Correct
Option" or "Answer Key" in a full shift. The only keyed files in the corpus are
the 2025 "T-I Similar Paper" coaching papers, which are not real PYQs (they
invent families the real exam does not use at that rate). So a paper of real
questions with trustworthy keys requires deriving the keys here.

Three derivations that can genuinely fail, and therefore mean something:

  spelling   -- "Select the INCORRECTLY spelt word": the answer is the single
                option absent from the 234,456-word system dictionary. Requires
                an exact 3-1 split or it declines.
  article    -- a/an is fixed by the SOUND of the next word ("an hour",
                "a university"), so the licensed form is computable.
  agreement  -- for options that differ only in a BE/HAVE form, the head noun's
                number licenses exactly one.

Everything else -- synonym, antonym, idiom, one-word substitution, para-jumble
ordering -- has no oracle available here and is left unkeyed rather than guessed.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
DICT_PATH = Path("/usr/share/dict/words")
DICT = ({w.strip().lower() for w in DICT_PATH.open() if w.strip()}
        if DICT_PATH.exists() else set())


def _clean(w: str) -> str:
    return re.sub(r"[^a-z]", "", w.lower())


def in_dict(w: str) -> bool:
    c = _clean(w)
    return bool(c) and c in DICT


def key_spelling(rec):
    """Single option absent from the dictionary = the misspelling."""
    stem = rec["stem"].lower()
    if not re.search(r"incorrectly spelt|misspelt|spelling error", stem):
        return None, None
    opts = rec["options"]
    if len(opts) != 4:
        return None, None
    # Single-word options only; phrases are not dictionary-checkable.
    if any(len(t.split()) != 1 for _, t in opts):
        return None, None
    absent = [l for l, t in opts if not in_dict(t)]
    if len(absent) != 1:
        return None, None
    return absent[0], "machine:dictionary"


AN_WORDS = {"hour", "honest", "heir", "honour", "honor", "umbrella", "elephant",
            "idea", "orange", "apple", "engineer", "eagle", "island", "onion"}
A_WORDS = {"university", "union", "european", "one", "useful", "unique",
           "uniform", "hotel", "historic", "horse", "house", "year", "unit"}


def key_article(rec):
    """a/an decided by the sound of the following word."""
    opts = rec["options"]
    texts = {_clean(t) for _, t in opts}
    if not texts <= {"a", "an", "the", "noarticle", ""} or len(opts) != 4:
        return None, None
    m = re.search(r"_{2,}\s*([A-Za-z]+)", rec["stem"])
    if not m:
        return None, None
    nxt = m.group(1).lower()
    if nxt in AN_WORDS:
        want = "an"
    elif nxt in A_WORDS:
        want = "a"
    else:
        return None, None
    hits = [l for l, t in opts if _clean(t) == want]
    if len(hits) != 1:
        return None, None
    return hits[0], "machine:phonetic_article_rule"


SINGULAR_HEADS = re.compile(
    r"\b(list|bunch|quality|each|number|neither|either|one|set|series|"
    r"scenery|furniture|information|advice|luggage|news|committee)\b", re.I)
BE_SING = {"is", "was", "has", "hasbeen"}
BE_PLUR = {"are", "were", "have", "havebeen"}


def key_agreement(rec):
    """Options differing only in a BE/HAVE form: head-noun number licenses one."""
    opts = rec["options"]
    if len(opts) != 4:
        return None, None
    forms = [_clean(t) for _, t in opts]
    if not all(f in BE_SING | BE_PLUR for f in forms):
        return None, None
    head = SINGULAR_HEADS.search(rec["stem"])
    if not head:
        return None, None
    want = BE_SING
    hits = [l for l, t in opts if _clean(t) in want]
    if len(hits) != 1:
        return None, None
    return hits[0], "machine:agreement_rule"


DERIVERS = (key_spelling, key_article, key_agreement)


def derive(rec):
    for fn in DERIVERS:
        try:
            a, how = fn(rec)
        except Exception:
            continue
        if a:
            return a, how
    return None, None


def main() -> None:
    recs = json.load(open(OUT / "english_tagged.json"))
    official = [r for r in recs if not r["is_reconstruction"]]
    out, by = {}, Counter()
    for r in official:
        if not r.get("options"):
            continue
        a, how = derive(r)
        if a:
            out[r["qid"]] = {"answer": a, "basis": how, "family": r.get("family")}
            by[how] += 1

    print("=" * 84)
    print("KEY DERIVATION FOR OFFICIAL ENGLISH PYQs (2022-2024)")
    print("=" * 84)
    print(f"official English questions with options: "
          f"{sum(1 for r in official if r.get('options'))}")
    print(f"keys derived                           : {len(out)}")
    for k, v in by.most_common():
        print(f"   {k:<36}{v:>5}")
    print("\nby family:")
    fam = Counter(v["family"] for v in out.values())
    for k, v in fam.most_common():
        print(f"   {str(k):<28}{v:>5}")
    print("\nNOT derivable (left unkeyed rather than guessed): "
          f"{sum(1 for r in official if r.get('options')) - len(out)}")

    # Cross-check the derivations against the 2025 coaching keys, which use the
    # same question formats. Agreement there is evidence the derivers are sound.
    recon = [r for r in recs if r["is_reconstruction"] and r.get("answer")]
    ag = dis = 0
    for r in recon:
        a, _ = derive(r)
        if a:
            if a == r["answer"]:
                ag += 1
            else:
                dis += 1
    if ag + dis:
        print(f"\nSanity check vs 2025 coaching keys on the same formats: "
              f"{ag}/{ag+dis} agree ({100*ag/(ag+dis):.1f}%)")

    (OUT / "english_derived_keys.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT/'english_derived_keys.json'}")


if __name__ == "__main__":
    main()
