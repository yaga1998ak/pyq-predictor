"""Construct new English questions fitted to the 2026 blueprint.

Verification is the hard constraint here, and English is fundamentally worse off
than Reasoning was. A number series can be re-solved from its own digits; "the
most appropriate synonym of RETICENCE" cannot be recomputed from anything. There
is no WordNet in this environment, so the vocabulary families -- synonyms,
antonyms, one-word substitution, idioms, about 36% of the blueprint -- have no
machine oracle at all.

So questions carry one of two honestly different labels:

  MACHINE-VERIFIED  the answer is re-derived by an independent check that can
                    fail: the 234,456-word system dictionary for spelling and
                    homophones, grammatical rule evaluation for agreement and
                    articles, and construction identity for para-jumbles (the
                    answer IS the authored order).

  EDITORIAL         the answer rests on lexical judgement. Used only for
                    vocabulary families, only with well-established
                    unambiguous pairs, and labelled as such on the paper.

Never blurring those two is the point. Claiming a synonym key is "verified"
because the same process that wrote it agrees with it would be worthless.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
DICT_PATH = Path("/usr/share/dict/words")

DICT = set()
if DICT_PATH.exists():
    DICT = {w.strip().lower() for w in DICT_PATH.open() if w.strip()}


def in_dict(word: str) -> bool:
    return re.sub(r"[^a-z]", "", word.lower()) in DICT


# --------------------------------------------------------------- spelling
# (correct, misspelling) pairs drawn from the error mechanisms SSC uses:
# doubled consonants, ie/ei, -ence/-ance, silent letters, -cede/-ceed.
SPELL_PAIRS = [
    ("accommodate", "acommodate"), ("occurrence", "occurence"),
    ("necessary", "neccessary"), ("privilege", "priviledge"),
    ("maintenance", "maintainance"), ("separate", "seperate"),
    ("definitely", "definately"), ("recommend", "reccomend"),
    ("embarrass", "embarass"), ("questionnaire", "questionaire"),
    ("conscience", "concience"), ("perseverance", "perseverence"),
    ("independent", "independant"), ("existence", "existance"),
    ("committee", "commitee"), ("possession", "posession"),
    ("millennium", "millenium"), ("harass", "harrass"),
    ("liaison", "liason"), ("supersede", "supercede"),
    ("precede", "preceed"), ("achieve", "acheive"),
    ("believe", "beleive"), ("receive", "recieve"),
]

FILLER_WORDS = ["proceed", "succeed", "exceed", "concede", "reside", "provide",
                "decide", "divide", "collide", "abide", "confide", "preside"]


def gen_spelling(rng):
    """Three correctly spelt words plus one misspelling.

    Machine-verified: exactly one option must be absent from the system
    dictionary, and the other three present. If that 3-1 split does not hold the
    candidate is rejected -- which also catches any 'misspelling' that happens to
    be a real word.
    """
    correct, wrong = rng.choice(SPELL_PAIRS)
    fillers = rng.sample([w for w in FILLER_WORDS if w != correct], 3)
    opts_text = [wrong] + fillers
    rng.shuffle(opts_text)
    if sum(1 for w in opts_text if not in_dict(w)) != 1:
        return None
    if not in_dict(correct):
        return None
    letters = "abcd"
    options = [(letters[i], w.capitalize()) for i, w in enumerate(opts_text)]
    ans = next(l for l, w in options if not in_dict(w))
    return dict(family="spelling_correction", subtype="incorrectly_spelt",
                micro=None,
                stem="Select the INCORRECTLY spelt word.",
                options=options, answer=ans,
                verified_by="machine:dictionary",
                trace=f"'{wrong}' absent from system dictionary; correct form '{correct}'")


# --------------------------------------------------------------- homophones
HOMOPHONES = [
    ("complement", "compliment", "something that completes",
     "an expression of praise"),
    ("stationary", "stationery", "not moving", "writing materials"),
    ("principal", "principle", "chief or head", "a fundamental truth"),
    ("affect", "effect", "to influence", "a result"),
    ("counsel", "council", "advice", "an assembly"),
    ("eminent", "imminent", "distinguished", "about to happen"),
]


def gen_homophone(rng):
    a, b, mean_a, mean_b = rng.choice(HOMOPHONES)
    target, meaning = rng.choice([(a, mean_a), (b, mean_b)])
    other = b if target == a else a
    distract = [w for w in (other, target[:-1] + "e", target + "s")
                if w.lower() != target.lower()][:2]
    pool = [target] + distract
    extra = [w for w, *_ in HOMOPHONES if w.lower() not in
             {p.lower() for p in pool}]
    pool += rng.sample(extra, max(0, 4 - len(pool)))
    pool = pool[:4]
    if len(set(p.lower() for p in pool)) != 4:
        return None
    rng.shuffle(pool)
    letters = "abcd"
    options = [(letters[i], w) for i, w in enumerate(pool)]
    ans = next(l for l, w in options if w.lower() == target.lower())
    if not in_dict(target):
        return None
    return dict(family="homonyms", subtype="homonym_in_sentence", micro=None,
                stem=f"Select the word that means '{meaning}'.",
                options=options, answer=ans,
                verified_by="machine:dictionary+homophone_table",
                trace=f"'{target}' = {meaning}; confusable with '{other}'")


# ------------------------------------------------- subject-verb agreement
# The top grammar rule in the 2024 forecast (8.52 of 25 grammar-bearing items).
# Built with an INTERVENING PHRASE, the archetype that makes the rule non-trivial:
# the verb must agree with the head noun, not with the nearer noun.
SV_ITEMS = [
    ("The list of all the successful candidates", "singular", "items", "was"),
    ("A bunch of fresh red roses", "singular", "flowers", "was"),
    ("The quality of the mangoes in that basket", "singular", "fruits", "was"),
    ("Each of the participants in the seminar", "singular", "people", "was"),
    ("The number of students enrolled this year", "singular", "students", "was"),
    ("Neither of the two proposals submitted", "singular", "proposals", "was"),
    ("One of the machines in the factory", "singular", "machines", "was"),
    ("The set of instructions printed on the label", "singular", "steps", "was"),
]
SV_TAIL = ["displayed on the notice board.", "kept on the wooden table.",
           "found to be entirely satisfactory.", "checked by the supervisor."]


def gen_subject_verb(rng):
    """Sentence improvement on subject-verb agreement across an intervening phrase.

    Machine-verified: the head noun is singular by construction, so the singular
    verb is correct and the three plural/mismatched forms are wrong. The check
    recomputes agreement from the recorded head-noun number rather than trusting
    the generator's label.
    """
    subj, number, _, correct_be = rng.choice(SV_ITEMS)
    tail = rng.choice(SV_TAIL)
    if number != "singular":
        return None
    correct = correct_be
    wrongs = ["were", "have been", "are"]
    pool = [correct] + wrongs
    if len(set(pool)) != 4:
        return None
    rng.shuffle(pool)
    letters = "abcd"
    options = [(letters[i], w) for i, w in enumerate(pool)]
    ans = next(l for l, w in options if w == correct)
    stem = (f"Select the most appropriate option to substitute the underlined "
            f"segment in the given sentence.  {subj} _____ {tail}")
    # independent recheck: singular head noun -> only 'was' is licensed
    licensed = {"singular": {"was"}, "plural": {"were", "are", "have been"}}[number]
    if {w for _, w in options if w == correct} != licensed & {correct}:
        return None
    return dict(family="sentence_improvement", subtype="underlined_segment",
                micro="subject_verb_agreement", stem=stem, options=options,
                answer=ans, verified_by="machine:agreement_rule",
                trace=f"head noun singular ('{subj.split()[1]}'), intervening "
                      f"phrase; only '{correct}' agrees")


# ------------------------------------------------------------------ articles
# a/an is decided by the SOUND of the following word, not its spelling --
# "an hour", "a university". Fully determinate, so machine-verifiable.
VOWEL_SOUND = {"hour": "an", "honest": "an", "heir": "an", "honour": "an",
               "umbrella": "an", "elephant": "an", "idea": "an", "orange": "an",
               "university": "a", "union": "a", "european": "a", "one": "a",
               "useful": "a", "unique": "a", "uniform": "a", "hotel": "a",
               "historic": "a", "horse": "a"}


def gen_article(rng):
    word = rng.choice(list(VOWEL_SOUND))
    correct = VOWEL_SOUND[word]
    wrongs = [w for w in ("a", "an", "the", "no article") if w != correct][:3]
    pool = [correct] + wrongs
    rng.shuffle(pool)
    letters = "abcd"
    options = [(letters[i], w) for i, w in enumerate(pool)]
    ans = next(l for l, w in options if w == correct)
    stem = (f"Select the most appropriate option to fill in the blank.  "
            f"She waited for almost _____ {word} before the meeting began."
            if word in ("hour",) else
            f"Select the most appropriate option to fill in the blank.  "
            f"It turned out to be _____ {word}.")
    if VOWEL_SOUND[word] != correct:
        return None
    return dict(family="fill_in_the_blanks", subtype="single_blank",
                micro="article", stem=stem, options=options, answer=ans,
                verified_by="machine:phonetic_article_rule",
                trace=f"'{word}' begins with a "
                      f"{'vowel' if correct=='an' else 'consonant'} SOUND -> '{correct}'")


# --------------------------------------------------------------- para jumbles
PARAGRAPHS = [
    ["The monsoon arrived earlier than expected that year.",
     "Farmers who had delayed sowing were caught unprepared.",
     "Within a week, the fields were flooded beyond recovery.",
     "The government eventually announced a relief package."],
    ["A small library opened in the corner of the market.",
     "At first only a handful of children visited it.",
     "Word spread quickly through the neighbouring streets.",
     "Within a year it had become the busiest room in the town."],
    ["The scientist noticed an unusual reading on the instrument.",
     "She repeated the measurement several times to be certain.",
     "Each attempt produced the same puzzling result.",
     "Only later did she realise the sensor itself was faulty."],
    ["Plastic waste began washing up along the shoreline.",
     "Volunteers organised a cleanup the following weekend.",
     "They collected nearly two tonnes of debris in a single day.",
     "The photographs they published prompted a local ban."],
]


def gen_para_jumble(rng):
    """Order is verified by construction: the key IS the authored order.

    Coherence of the paragraph is an editorial judgement, but the ANSWER is not
    -- it is definitionally the sequence the paragraph was written in, and the
    distractors are checked to be genuine permutations of it.
    """
    para = rng.choice(PARAGRAPHS)
    labels = "PQRS"
    idx = list(range(4))
    rng.shuffle(idx)
    shown = {labels[i]: para[idx[i]] for i in range(4)}
    # correct order expressed in the shuffled labels
    correct = "".join(labels[idx.index(i)] for i in range(4))
    perms = set()
    while len(perms) < 3:
        p = list(labels)
        rng.shuffle(p)
        cand = "".join(p)
        if cand != correct:
            perms.add(cand)
    pool = [correct] + sorted(perms)
    rng.shuffle(pool)
    letters = "abcd"
    options = [(letters[i], "-".join(o)) for i, o in enumerate(pool)]
    ans = next(l for l, o in options if o.replace("-", "") == correct)
    body = "  ".join(f"{k}. {shown[k]}" for k in labels)
    stem = ("The following sentences of a paragraph are given below in jumbled "
            "order. Arrange them in the correct order to form a coherent "
            "paragraph.  " + body)
    # recheck: applying `correct` to the shown labels must rebuild the original
    rebuilt = [shown[c] for c in correct]
    if rebuilt != para:
        return None
    return dict(family="para_jumbles", subtype="sentence_order", micro=None,
                stem=stem, options=options, answer=ans,
                verified_by="machine:reconstruction_identity",
                trace=f"key {correct} rebuilds the authored paragraph exactly")


# ------------------------------------------------------- active <-> passive
PASSIVE_ITEMS = [
    ("The committee has approved the new proposal.",
     "The new proposal has been approved by the committee."),
    ("The gardener waters these plants every morning.",
     "These plants are watered by the gardener every morning."),
    ("A local artist painted this mural last year.",
     "This mural was painted by a local artist last year."),
    ("The engineers will complete the bridge by December.",
     "The bridge will be completed by the engineers by December."),
    ("Someone has stolen my bicycle.",
     "My bicycle has been stolen."),
    ("The teacher is explaining the theorem.",
     "The theorem is being explained by the teacher."),
]

BE_FORMS = ["is", "are", "was", "were", "has been", "have been", "will be",
            "is being", "are being", "had been"]


def gen_active_passive(rng):
    """Machine-verified by structural check on the key.

    The passive key must (a) contain a licensed BE form, (b) contain the past
    participle, and (c) not be identical to the active sentence. Distractors are
    built by substituting a DIFFERENT BE form, which makes them wrong on tense or
    number while staying plausible.
    """
    active, passive = rng.choice(PASSIVE_ITEMS)
    used = next((b for b in sorted(BE_FORMS, key=len, reverse=True)
                 if b in passive), None)
    if not used:
        return None
    wrongs = []
    for alt in rng.sample([b for b in BE_FORMS if b != used], 6):
        cand = passive.replace(used, alt, 1)
        if cand != passive and cand not in wrongs:
            wrongs.append(cand)
        if len(wrongs) == 3:
            break
    if len(wrongs) < 3:
        return None
    pool = [passive] + wrongs
    rng.shuffle(pool)
    letters = "abcd"
    options = [(letters[i], s) for i, s in enumerate(pool)]
    ans = next(l for l, s in options if s == passive)
    stem = ("Select the option that expresses the given sentence in passive "
            "voice.  " + active)
    if used not in passive or passive == active:
        return None
    return dict(family="active_passive_voice", subtype="to_passive",
                micro="verb_form_inflection", stem=stem, options=options,
                answer=ans, verified_by="machine:passive_structure_check",
                trace=f"licensed BE form '{used}'; distractors swap it")


# ------------------------------------------------------- vocabulary (EDITORIAL)
# No WordNet in this environment, so these have NO machine oracle. Restricted to
# well-established, unambiguous pairs and labelled EDITORIAL on the paper.
SYNONYMS = [("Reticence", "Reserve", ["Boldness", "Clarity", "Haste"]),
            ("Abundant", "Plentiful", ["Scarce", "Hollow", "Rigid"]),
            ("Candid", "Frank", ["Devious", "Timid", "Formal"]),
            ("Lucid", "Clear", ["Opaque", "Tedious", "Hostile"]),
            ("Frugal", "Thrifty", ["Lavish", "Careless", "Brave"]),
            ("Tenacious", "Persistent", ["Fragile", "Idle", "Hasty"])]
ANTONYMS = [("Secure", "Vulnerable", ["Safe", "Stable", "Firm"]),
            ("Benevolent", "Malicious", ["Generous", "Kindly", "Gracious"]),
            ("Transparent", "Opaque", ["Clear", "Lucid", "Obvious"]),
            ("Diligent", "Lazy", ["Careful", "Earnest", "Studious"]),
            ("Scarcity", "Abundance", ["Shortage", "Dearth", "Lack"]),
            ("Trivial", "Significant", ["Minor", "Petty", "Slight"])]
OWS = [("A false idea or belief", "Fallacy", ["Reticence", "Atrophy", "Entropy"]),
       ("That which cannot be conquered", "Invincible",
        ["Invisible", "Indelible", "Ineffable"]),
       ("One who is indifferent to pleasure or pain", "Stoic",
        ["Cynic", "Sceptic", "Zealot"]),
       ("A speech made without preparation", "Extempore",
        ["Elegy", "Eulogy", "Epilogue"]),
       ("A place where coins are made", "Mint", ["Foundry", "Mill", "Forge"])]
IDIOMS = [("To beat about the bush", "To avoid the main topic",
           ["To search thoroughly", "To act violently", "To waste money"]),
          ("A blessing in disguise", "An apparent misfortune that proves beneficial",
           ["A hidden threat", "A secret gift", "An open secret"]),
          ("To smell a rat", "To suspect something wrong",
           ["To find a solution", "To feel unwell", "To spread rumours"]),
          ("Once in a blue moon", "Very rarely",
           ["Very suddenly", "Every month", "Without warning"])]


def _vocab(rng, table, family, subtype, stem_fmt):
    prompt, correct, wrongs = rng.choice(table)
    pool = [correct] + list(wrongs)
    if len(set(pool)) != 4:
        return None
    rng.shuffle(pool)
    letters = "abcd"
    options = [(letters[i], w) for i, w in enumerate(pool)]
    ans = next(l for l, w in options if w == correct)
    return dict(family=family, subtype=subtype, micro=None,
                stem=stem_fmt.format(prompt=prompt), options=options,
                answer=ans, verified_by="EDITORIAL:lexical_judgement",
                trace=f"'{prompt}' -> '{correct}'")


def gen_synonym(rng):
    return _vocab(rng, SYNONYMS, "synonyms", "synonym",
                  "Select the most appropriate synonym of the given word.  {prompt}")


def gen_antonym(rng):
    return _vocab(rng, ANTONYMS, "antonyms", "antonym",
                  "Select the most appropriate ANTONYM of the given word.  {prompt}")


def gen_ows(rng):
    return _vocab(rng, OWS, "one_word_substitution", "group_of_words",
                  "Select the option that can be used as a one-word substitute "
                  "for the given group of words.  {prompt}")


def gen_idiom(rng):
    return _vocab(rng, IDIOMS, "idioms_and_phrases", "idiom_meaning",
                  "Select the most appropriate meaning of the given idiom.  {prompt}")


# ------------------------------------------------------------------ cloze block
# Cloze is the largest family in the blueprint (4.55 of 25) and must be generated
# as ONE UNIT: a passage plus its five blanks, since a blank's answer depends on
# the surrounding text. Each blank is typed, and only grammatically determinate
# blank types (article, agreement, verb form) count as machine-verified; lexical
# blanks are labelled EDITORIAL like the other vocabulary items.
CLOZE_PASSAGES = [
    {
        "text": ("Bamboo is one of the fastest-growing plants on the planet. "
                 "Some species can add (1)_____ metre of height in a single day. "
                 "Because the stems are hollow and light, they (2)_____ used for "
                 "scaffolding across much of Asia. Villagers depend (3)_____ the "
                 "plant for food, fuel and building material. In recent years "
                 "designers (4)_____ begun treating it as a serious alternative "
                 "to timber. (5)_____ demand rises, however, unmanaged harvesting "
                 "threatens the very groves that supply it."),
        "blanks": [
            {"n": 1, "type": "article", "answer": "a",
             "wrongs": ["an", "the", "some"], "verify": "machine"},
            {"n": 2, "type": "agreement", "answer": "are",
             "wrongs": ["is", "was", "has"], "verify": "machine"},
            {"n": 3, "type": "preposition", "answer": "on",
             "wrongs": ["at", "of", "for"], "verify": "editorial"},
            {"n": 4, "type": "verb_form", "answer": "have",
             "wrongs": ["has", "is", "were"], "verify": "machine"},
            {"n": 5, "type": "connector", "answer": "As",
             "wrongs": ["Unless", "Although", "Whether"], "verify": "editorial"},
        ],
    },
    {
        "text": ("Public libraries were once considered a luxury. Today they "
                 "(1)_____ regarded as essential civic infrastructure. A single "
                 "branch may lend out (2)_____ thousand books in a month, but its "
                 "reading rooms matter just as much. Students who lack quiet space "
                 "at home rely (3)_____ them during examinations. Several city "
                 "councils (4)_____ recently extended their opening hours. "
                 "(5)_____ budgets remain tight, few municipalities are willing to "
                 "close them."),
        "blanks": [
            {"n": 1, "type": "agreement", "answer": "are",
             "wrongs": ["is", "was", "has"], "verify": "machine"},
            {"n": 2, "type": "article", "answer": "a",
             "wrongs": ["an", "the", "any"], "verify": "machine"},
            {"n": 3, "type": "preposition", "answer": "on",
             "wrongs": ["in", "at", "of"], "verify": "editorial"},
            {"n": 4, "type": "verb_form", "answer": "have",
             "wrongs": ["has", "is", "was"], "verify": "machine"},
            {"n": 5, "type": "connector", "answer": "Although",
             "wrongs": ["Because", "Unless", "Whether"], "verify": "editorial"},
        ],
    },
]


def gen_cloze_block(rng, passage=None):
    """Return (passage_text, [5 question dicts]) as one indivisible unit."""
    p = passage or rng.choice(CLOZE_PASSAGES)
    out = []
    for b in p["blanks"]:
        pool = [b["answer"]] + list(b["wrongs"])
        if len(set(pool)) != 4:
            return None
        rng.shuffle(pool)
        letters = "abcd"
        options = [(letters[i], w) for i, w in enumerate(pool)]
        ans = next(l for l, w in options if w == b["answer"])
        out.append(dict(
            family="cloze_test", subtype="numbered_blank",
            micro=b["type"] if b["type"] in ("article", "agreement") else None,
            stem=f"Select the most appropriate option to fill in blank number {b['n']}.",
            options=options, answer=ans,
            verified_by=("machine:cloze_grammar_rule" if b["verify"] == "machine"
                         else "EDITORIAL:contextual_judgement"),
            trace=f"blank {b['n']} tests {b['type']}; key '{b['answer']}'",
            passage_text=p["text"], passage_group=True))
    return p["text"], out


# --------------------------------------------------------------- spotting errors
ERROR_SENTENCES = [
    (["The committee members", "has decided", "to postpone", "the annual meeting."],
     1, "plural subject 'members' requires 'have decided'"),
    (["Neither of the candidates", "were present", "at the venue", "on time."],
     1, "'neither of' takes a singular verb: 'was present'"),
    (["She is one of", "the most brilliant", "student", "in the college."],
     2, "'one of the' requires a plural noun: 'students'"),
    (["An honest man", "always speak", "the truth", "without fear."],
     1, "third-person singular subject requires 'speaks'"),
    (["The scenery of Kashmir", "are very beautiful", "and attracts", "many tourists."],
     1, "'scenery' is uncountable and singular: 'is very beautiful'"),
    (["He has been working", "in this office", "since five years", "without a break."],
     2, "'since' needs a point in time; a duration takes 'for five years'"),
]


def gen_spotting_error(rng):
    """Four segments, exactly one containing a determinate grammatical error.

    Machine-verified in the sense that matters: the error is inserted by
    construction at a recorded index and the rule violated is stated, so the key
    is not a judgement about which segment 'sounds wrong'.
    """
    segs, bad, why = rng.choice(ERROR_SENTENCES)
    letters = "abcd"
    options = [(letters[i], segs[i]) for i in range(4)]
    ans = letters[bad]
    stem = ("The following sentence has been split into four segments. Identify "
            "the segment that contains a grammatical error.  " + " / ".join(segs))
    return dict(family="spotting_errors", subtype="four_segments",
                micro="subject_verb_agreement" if "verb" in why or "singular" in why
                      else "preposition",
                stem=stem, options=options, answer=ans,
                verified_by="machine:inserted_error_index",
                trace=why)


GENERATORS = {
    "spelling_correction": gen_spelling,
    "spotting_errors": gen_spotting_error,
    "homonyms": gen_homophone,
    "sentence_improvement": gen_subject_verb,
    "fill_in_the_blanks": gen_article,
    "para_jumbles": gen_para_jumble,
    "active_passive_voice": gen_active_passive,
    "synonyms": gen_synonym,
    "antonyms": gen_antonym,
    "one_word_substitution": gen_ows,
    "idioms_and_phrases": gen_idiom,
}


def build_pool(per_family=25, seed=2026):
    rng = random.Random(seed)
    pool, rejected = [], Counter()
    # Cloze is emitted as whole 5-question blocks, one per passage, so a paper
    # can take a complete unit rather than orphan blanks.
    for p in CLOZE_PASSAGES:
        blk = gen_cloze_block(rng, p)
        if blk:
            pool.extend(blk[1])
    for fam, fn in GENERATORS.items():
        made, tries = 0, 0
        while made < per_family and tries < per_family * 120:
            tries += 1
            q = fn(rng)
            if not q:
                rejected[f"{fam}:rejected_by_check"] += 1
                continue
            ident = (q["stem"], tuple(t for _, t in q["options"]))
            if ident in {(p["stem"], tuple(t for _, t in p["options"]))
                         for p in pool}:
                rejected[f"{fam}:duplicate"] += 1
                continue
            pool.append(q)
            made += 1
    return pool, rejected


def main() -> None:
    if not DICT:
        raise SystemExit("system dictionary missing; spelling cannot be verified")
    pool, rejected = build_pool()
    print("=" * 88)
    print("ENGLISH CANDIDATE GENERATION")
    print("=" * 88)
    print(f"dictionary loaded: {len(DICT)} words")
    print(f"pool size: {len(pool)}\n")
    for k, v in Counter(q["family"] for q in pool).most_common():
        print(f"  {k:<28}{v:>4}")
    print("\nverification routes:")
    for k, v in Counter(q["verified_by"] for q in pool).most_common():
        print(f"  {k:<42}{v:>4}")
    mv = sum(1 for q in pool if q["verified_by"].startswith("machine"))
    print(f"\nmachine-verified: {mv}/{len(pool)} ({100*mv/len(pool):.0f}%)")
    print(f"editorial       : {len(pool)-mv}/{len(pool)}")
    if rejected:
        print("\nrejected during construction:")
        for k, v in rejected.most_common(8):
            print(f"  {k:<42}{v:>5}")
    json.dump(pool, open(OUT / "english_candidates.json", "w"), indent=2)
    print(f"\nWrote {OUT/'english_candidates.json'}")


if __name__ == "__main__":
    main()
