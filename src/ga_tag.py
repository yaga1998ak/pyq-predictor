"""GA tagger: subject -> topic -> micro-subtopic, driven by an ENTITY GAZETTEER.

GA has no stem templates. "Who among the following..." carries no topical
information, so the three earlier taggers' approach (match the instruction
phrasing) cannot work here. The signal is in named entities -- Articles,
dynasties, rivers, schemes, awards, organisations, sports events -- so this
tagger is a gazetteer with an ordered disambiguation policy.

Consequence for the evidence base, established in ga_extract.py: de-spaced
extraction destroys the GA signal specifically. Entity detection needs word
boundaries, and 126 of 2022's 157 GA stems have none, giving 0.5 entities per
question against 2.1 in 2023. 2021 is separately unusable at 44% recall. Both are
excluded, which is a harder cut than the other sections needed.

Micro-subtopics are the prediction unit (§10), chosen to be narrow enough to
revise and broad enough to generate several question forms (§13).
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from reasoning_tag import normalize

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
DROP_YEARS = {2021, 2022}

# (subject, micro_subtopic, pattern). Order matters: specific institutions and
# Articles must be claimed before the generic subject words that also match them.
RULES: list[tuple[str, str, str]] = [
    # ---------------- POLITY ------------------------------------------------
    ("polity", "fundamental_rights",
     r"fundamental rights?|article 1[2-9]\b|article 2[0-9]\b|right to (equality|"
     r"freedom|life|education|constitutional remedies)|writ of|habeas corpus"),
    ("polity", "fundamental_duties",
     r"fundamental duties|article 51a|swaran singh"),
    ("polity", "dpsp",
     r"directive principles|\bdpsp\b|article 3[6-9]\b|article 4[0-9]\b"),
    ("polity", "constitutional_amendments",
     r"\d+(?:st|nd|rd|th) amendment|amendment act|constitution \(.*amendment"),
    ("polity", "parliament_legislature",
     r"\bparliament\b|lok sabha|rajya sabha|money bill|speaker of|"
     r"session of parliament|no[- ]confidence|quorum|state legislature|"
     r"legislative (assembly|council)"),
    ("polity", "president_governor",
     r"\bpresident of\b|\bpresident\b.{0,30}(elect|power|term|impeach)|"
     r"\bgovernor\b|vice[- ]president|article 5[2-9]\b|article 7[0-9]\b"),
    ("polity", "judiciary",
     r"supreme court|high court|chief justice|judicial review|"
     r"public interest litigation|collegium"),
    ("polity", "constitutional_bodies",
     r"election commission|comptroller and auditor|finance commission|"
     r"\bupsc\b|union public service|attorney general|\bcag\b"),
    ("polity", "schedules_and_lists",
     r"schedule of the constitution|\d+(?:st|nd|rd|th) schedule|"
     r"union list|state list|concurrent list|seventh schedule"),
    ("polity", "local_government",
     r"panchayati raj|municipal|\b73rd\b|\b74th\b|gram sabha|nagar palika"),
    ("polity", "constitution_general",
     r"constituent assembly|preamble|indian constitution|"
     r"\bconstitution\b|\barticle \d+"),

    # ---------------- HISTORY ----------------------------------------------
    ("history", "indus_valley",
     r"indus valley|harappa|mohenjo|lothal|dholavira|kalibangan|"
     r"\bharappan\b"),
    ("history", "vedic_period",
     r"\bvedic\b|\brigveda\b|\bveda\b|\bupanishad|\baryan"),
    ("history", "buddhism_jainism",
     r"\bbuddha\b|buddhis|\bjain\b|jainis|mahavira|"
     r"buddhist council|\bstupa\b|\bsangha\b|tirthankara"),
    ("history", "mauryan_empire",
     r"\bmaurya|\bashoka|chandragupta maurya|kautilya|arthashastra|"
     r"kalinga war|\bedicts?\b"),
    ("history", "gupta_and_post_gupta",
     r"\bgupta\b|samudragupta|chandragupta ii|harsha|\bpallava|\bchalukya"),
    ("history", "chola_and_south",
     r"\bchola\b|vijayalaya|rajaraja|\bpandya\b|\brashtrakuta|"
     r"vijayanagara|krishnadeva"),
    ("history", "delhi_sultanate",
     r"delhi sultanate|slave dynasty|\bkhilji|\bkhalji|\btughlaq|"
     r"\blodi\b|iltutmish|razia|balban|alauddin"),
    ("history", "mughal_empire",
     r"\bmughal|\bakbar\b|\bbabur\b|humayun|jahangir|shah jahan|aurangzeb|"
     r"\bdin[- ]i[- ]ilahi|mansabdari|\bpanipat\b"),
    ("history", "maratha_and_regional",
     r"\bmaratha|shivaji|peshwa|\bsikh\b.{0,30}(guru|empire)|ranjit singh|"
     r"\btipu\b|hyder ali"),
    ("history", "revolt_1857",
     r"revolt of 1857|first war of independence|mangal pandey|rani lakshmibai|"
     r"\bsepoy\b"),
    ("history", "freedom_movement",
     r"indian national congress|non[- ]cooperation|civil disobedience|"
     r"quit india|dandi march|salt satyagraha|\bgandhi\b|jallianwala|"
     r"rowlatt|simon commission|cabinet mission|swadeshi|home rule|"
     r"partition of bengal|muslim league|subhas chandra bose|\bazad hind"),
    ("history", "governors_general_viceroys",
     r"governor[- ]general|\bviceroy\b|lord (dalhousie|curzon|ripon|"
     r"cornwallis|wellesley|mountbatten|canning|hastings)"),
    ("history", "socio_religious_reform",
     r"brahmo samaj|arya samaj|ram mohan|vivekananda|ramakrishna|"
     r"jyotiba phule|\bsati\b|widow remarriage|dayanand"),

    # ---------------- GEOGRAPHY -------------------------------------------
    ("geography", "rivers_and_drainage",
     r"\briver\b|\btributary\b|tributaries|\bganga\b|brahmaputra|godavari|"
     r"krishna river|\bnarmada|\btapi\b|\bkaveri|\bcauvery|\bindus\b|\bsutlej|"
     r"\bdelta\b|\bconfluence"),
    ("geography", "mountains_and_relief",
     r"himalaya|\bghats\b|aravalli|vindhya|satpura|\bplateau\b|"
     r"\bpeak\b|\bpass\b.{0,20}(himalaya|india)|karakoram|shivalik"),
    ("geography", "monsoon_and_climate",
     r"\bmonsoon\b|\brainfall\b|\bclimate\b|\bcyclone\b|"
     r"western disturbance|\bel nino|rain shadow"),
    ("geography", "soils_agriculture",
     r"\bsoil\b|\balluvial\b|black soil|laterite|\bcrop\b|\bkharif\b|"
     r"\brabi\b|green revolution|\bcultivation\b|largest producer"),
    ("geography", "minerals_industry",
     r"\bmineral\b|\bore\b|\bcoal\b|\bbauxite|\bmica\b|iron ore|"
     r"\brefinery\b|\bsteel plant"),
    ("geography", "states_and_boundaries",
     r"shares? (a |its )?border|\bboundary\b|largest state|smallest state|"
     r"capital of.{0,20}state|newest state|union territor"),
    ("geography", "world_geography",
     r"\bcontinent\b|\bsahara\b|\bamazon\b|\bnile\b|\bstrait\b|"
     r"\blatitude\b|\blongitude\b|\bequator\b|tropic of|\bocean\b"),

    # ---------------- ENVIRONMENT ----------------------------------------
    ("environment", "protected_areas",
     r"national park|wildlife sanctuar|biosphere reserve|tiger reserve|"
     r"\bramsar\b|bird sanctuar|elephant reserve"),
    ("environment", "conventions_and_reports",
     r"\bunfccc\b|paris agreement|kyoto|montreal protocol|\bcop\d+|"
     r"convention on biological|\bcites\b|climate summit"),
    ("environment", "ecology_biodiversity",
     r"\becosystem\b|\bbiodiversity\b|\bfood chain\b|\bendangered\b|"
     r"\biucn\b|\bspecies\b.{0,25}(status|list)"),

    # ---------------- ECONOMY --------------------------------------------
    ("economy", "rbi_and_banking",
     r"reserve bank|\brbi\b|\brepo rate\b|\bcrr\b|\bslr\b|monetary policy|"
     r"\bnpa\b|\bbasel\b|\bnabard\b|\bsebi\b|\birdai\b"),
    ("economy", "budget_and_taxation",
     r"union budget|\bgst\b|direct tax|indirect tax|\bfiscal deficit\b|"
     r"\bincome tax\b|\bcustoms duty\b|finance bill"),
    ("economy", "national_income",
     r"\bgdp\b|\bgnp\b|gross domestic|national income|per capita income|"
     r"\bnso\b|base year"),
    ("economy", "planning_and_indices",
     r"niti aayog|planning commission|five year plan|"
     r"human development index|\bhdi\b|inflation index|\bcpi\b|\bwpi\b"),
    ("economy", "government_schemes",
     r"\byojana\b|\bmission\b.{0,25}(launch|scheme)|pradhan mantri|"
     r"\bmgnrega\b|\bayushman\b|jan dhan|ujjwala|\bpmay\b|swachh bharat|"
     r"make in india|atmanirbhar|\bscheme\b"),

    # ---------------- SCIENCE ---------------------------------------------
    ("science", "physics",
     r"\bnewton\b|\bforce\b|\bvelocity\b|\bacceleration\b|\bfriction\b|"
     r"\boptics\b|\brefraction\b|\bmagnet\b|\belectric current\b|\bohm\b|"
     r"\bsi unit\b|\bwatt\b|\bjoule\b|\bpascal\b|\blens\b|\bsound wave"),
    ("science", "chemistry",
     r"\batomic number\b|\bperiodic table\b|\bvalency\b|\bacid\b|\bbase\b|"
     r"\bchemical formula\b|\bcatalyst\b|\balloy\b|\boxidation\b|"
     r"\bisotope\b|\bnoble gas\b|\bcompound\b.{0,20}formula"),
    ("science", "biology",
     r"\bvitamin\b|\bdeficiency\b|\bhormone\b|\benzyme\b|\bblood group\b|"
     r"\bphotosynthesis\b|\bcell\b.{0,20}(organelle|wall|membrane)|"
     r"\bchromosome\b|\bdna\b|\bdisease\b|\bmalaria\b|\btuberculosis\b|"
     r"\bhuman body\b|\brespiration\b|\bdigestion"),
    ("science", "space_and_technology",
     r"\bisro\b|\bnasa\b|\bchandrayaan\b|\bmangalyaan\b|\bgaganyaan\b|"
     r"\bpslv\b|\bgslv\b|\bsatellite\b|\bspace (mission|station)\b|"
     r"\baditya[- ]l1\b"),

    # ---------------- ART & CULTURE ---------------------------------------
    ("art_culture", "classical_dance",
     r"\bbharatanatyam\b|\bkathak\b|\bkathakali\b|\bodissi\b|\bodissi\b|"
     r"\bkuchipudi\b|\bmanipuri\b|\bmohiniyattam\b|\bsattriya\b|"
     r"classical dance|\bdance form\b|\bfolk dance\b"),
    ("art_culture", "music_instruments",
     r"\bhindustani\b|\bcarnatic\b|\braga\b|\btabla\b|\bsitar\b|\bsarod\b|"
     r"\bveena\b|\bshehnai\b|\bmusical instrument\b|\bflute\b"),
    ("art_culture", "festivals",
     r"\bfestival\b|\bbihu\b|\bonam\b|\bpongal\b|\bbaisakhi\b|\bhornbill\b|"
     r"\bthrissur\b|\brath yatra\b|\blosar\b|\bwangala\b"),
    ("art_culture", "monuments_heritage",
     r"world heritage|\bunesco\b|\btaj mahal\b|\bqutb\b|\bqutub\b|"
     r"\bfort\b|\bcave\b.{0,20}(ajanta|ellora|elephanta)|\btemple\b|"
     r"\bstupa\b|\bmonument\b"),
    ("art_culture", "painting_crafts",
     r"\bpainting\b|\bmadhubani\b|\bwarli\b|\bpattachitra\b|\bkalamkari\b|"
     r"\bhandicraft\b|\bgi tag\b|geographical indication"),

    # ---------------- SPORTS ----------------------------------------------
    ("sports", "olympics_asian_games",
     r"\bolympic\b|\bolympics\b|asian games|commonwealth games|"
     r"\bparalympic\b|\bnational games\b"),
    ("sports", "cricket",
     r"\bcricket\b|\bipl\b|\bodi\b|\btest match\b|\branji\b|\bicc\b|"
     r"\bworld cup\b.{0,20}cricket"),
    ("sports", "trophies_and_terms",
     r"\btrophy\b|\bcup\b.{0,20}(associated|related|sport)|"
     r"\bterm\b.{0,25}(associated with|used in).{0,20}(sport|game)|"
     r"\bduranD\b|\bsantosh trophy\b|\bthomas cup\b"),
    ("sports", "other_sports_events",
     r"\bbadminton\b|\bhockey\b|\btennis\b|\bchess\b|\bwrestling\b|"
     r"\bboxing\b|\bshooting\b|\barchery\b|\bkabaddi\b|\bathletics\b"),

    # ---------------- MISCELLANEOUS ---------------------------------------
    ("misc", "awards_honours",
     r"\bpadma\b|bharat ratna|\bnobel\b|\bjnanpith\b|\bdadasaheb\b|"
     r"\barjuna award\b|\bkhel ratna\b|\bdronacharya\b|\baward\b"),
    ("misc", "books_authors",
     r"\bauthor of\b|\bwritten by\b|\bbook\b.{0,25}(written|authored)|"
     r"\bautobiography\b|\bnovel\b"),
    ("misc", "organisations_hq",
     r"\bheadquarters\b|\bunited nations\b|\bwho\b.{0,15}headquarter|"
     r"\bimf\b|\bworld bank\b|\bwto\b|\bunesco\b|\bopec\b|\basean\b|"
     r"\bbrics\b|\bsaarc\b|\bg20\b"),
    ("misc", "important_days",
     r"\bobserved (on|every)\b|\bworld .{0,20}day\b|\bnational .{0,20}day\b|"
     r"\bcelebrated on\b"),
    ("misc", "census_demography",
     r"\bcensus\b|\bsex ratio\b|\bliteracy rate\b|\bpopulation density\b|"
     r"\bdecadal growth\b"),
    ("misc", "appointments_persons",
     r"\bappointed as\b|\bchief minister\b|\bchairman of\b|"
     r"\bsworn in\b|\bnew (chief|director|governor)\b"),

    # ---- gazetteer expansion, driven by the entities dominating the untagged
    # set. GA coverage was 51% and the misses were not random: colonial-era
    # Acts, the 2023 criminal-law replacements, named folk dances and bare state
    # names accounted for most of them.
    ("polity", "new_criminal_laws",
     r"bharatiya nyaya sanhita|bharatiya nagarik suraksha|bharatiya sakshya|"
     r"\bbns\b|\bbnss\b|replaced the indian penal code"),
    ("history", "colonial_acts_administration",
     r"regulating act|charter act|pitt'?s india act|government of india act|"
     r"east india company|morley[- ]minto|montagu[- ]chelmsford|"
     r"\bindia act\b|\bdoctrine of lapse\b|permanent settlement|"
     r"ryotwari|mahalwari|\bzamindar"),
    ("history", "peasant_tribal_movements",
     r"\bmoplah\b|\bmappila\b|\bsanthal\b|\bmunda\b|\bbirsa\b|"
     r"\bchamparan\b|\bkheda\b|\bbardoli\b|\btebhaga\b|\btelangana\b"
     r".{0,20}movement|peasant movement|tribal (revolt|uprising)"),
    ("art_culture", "folk_dance_by_state",
     r"\bkalbelia\b|\bghoomar\b|\blavani\b|\bgarba\b|\bbhangra\b|"
     r"\bgiddha\b|\bchhau\b|\bbihu\b|\bdollu\b|\byakshagana\b|"
     r"\bkarma\b.{0,12}dance|\bcheraw\b|\bthang[- ]ta\b|\bfugdi\b|"
     r"\bpanihari\b|\btamasha\b|\bnautanki\b|\bbaul\b"),
    ("science", "scientists_discoveries",
     r"\bknown for the discovery\b|\bdiscovered by\b|\bnobel prize in\b|"
     r"\bcavendish\b|\brutherford\b|\bbohr\b|\bcurie\b|\bhess\b|"
     r"\braman\b|\bbose\b.{0,20}(statistic|particle)|\bchandrasekhar\b|"
     r"\bgravitational constant\b|\bcosmic ray"),
    ("economy", "reports_and_indices",
     r"\bindex\b|\branking\b.{0,25}(report|index)|\breport\b.{0,20}released|"
     r"ease of doing business|global hunger|world happiness"),
    ("sports", "world_cups_championships",
     r"\bworld cup\b|\bworld championship\b|\bgrand slam\b|"
     r"\bfifa\b|\bwimbledon\b"),
    ("geography", "state_identification",
     r"\b(maharashtra|rajasthan|karnataka|kerala|odisha|jharkhand|"
     r"chhattisgarh|gujarat|punjab|haryana|bihar|assam|manipur|mizoram|"
     r"nagaland|tripura|meghalaya|sikkim|uttarakhand|telangana|goa|"
     r"madhya pradesh|uttar pradesh|tamil nadu|andhra pradesh|"
     r"west bengal|himachal pradesh|arunachal pradesh)\b"),
]

COMPILED = [(s, m, re.compile(p, re.I)) for s, m, p in RULES]


def _despace(pattern: str) -> str | None:
    s = pattern
    # Same comma caveat as the other taggers: stripping "," turns "{0,25}" into
    # "{025}", which compiles as "exactly 25 chars" and silently kills the rule.
    for tok in (r"[\s\-]?", r"[\s\-]", r"\s", r"\b", " ?", " "):
        s = s.replace(tok, "")
    try:
        re.compile(s)
    except re.error:
        return None
    return s


DESPACED = [(s, m, re.compile(b, re.I))
            for s, m, p in RULES if (b := _despace(p))]
_SQUASH = re.compile(r"[\s,\-.'’]+")
MIN_SQUASHED = 10


def classify(stem: str) -> tuple[str | None, str | None, str]:
    text = normalize(stem)
    for s, m, rx in COMPILED:
        if rx.search(text):
            return s, m, "direct"
    sq = _SQUASH.sub("", text)
    for s, m, rx in DESPACED:
        mm = rx.search(sq)
        if mm and len(mm.group(0)) >= MIN_SQUASHED:
            return s, m, "squashed"
    return None, None, "none"


def main() -> None:
    recs = [r for r in json.load(open(OUT / "ga_questions.json"))
            if r["year"] not in DROP_YEARS]

    kept, blank = [], []
    for r in recs:
        stem = normalize(r["stem"])
        if len(stem) < 25:
            r["subject"] = r["micro"] = None
            r["tag_route"] = "blank"
            blank.append(r)
            continue
        s, m, how = classify(r["stem"])
        r["subject"], r["micro"], r["tag_route"] = s, m, how
        kept.append(r)

    usable = kept + blank
    shifts = len({(r["exam_date"], r["shift"]) for r in usable})
    tagged = [r for r in kept if r["subject"]]

    print("=" * 96)
    print("GA TAGGER — gazetteer coverage (2021 and 2022 excluded)")
    print("=" * 96)
    print(f"questions           : {len(usable)}  across {shifts} shifts")
    print(f"blank stems         : {len(blank)}")
    print(f"tagged              : {len(tagged)}")
    print(f"coverage (non-blank): {100*len(tagged)/max(len(kept),1):.1f}%")
    print(f"route: {dict(Counter(r['tag_route'] for r in kept))}")

    print("\n" + "=" * 96)
    print("SUBJECT x MICRO-SUBTOPIC (per shift)")
    print("=" * 96)
    by = defaultdict(Counter)
    for r in tagged:
        by[r["subject"]][r["micro"]] += 1
    for s in sorted(by, key=lambda k: -sum(by[k].values())):
        tot = sum(by[s].values())
        print(f"\n{s:<22}{tot:>5}{tot/shifts:>8.2f}/shift")
        for m, c in by[s].most_common():
            print(f"    {m:<32}{c:>5}{c/shifts:>8.2f}")

    print("\n" + "=" * 96)
    print("PER-YEAR")
    print("=" * 96)
    print(f"{'Year':<6}{'Shifts':>7}{'GA':>6}{'tagged':>8}{'cov':>7}{'blank':>7}")
    print("-" * 96)
    for y in sorted({r["year"] for r in usable}):
        u = [r for r in usable if r["year"] == y]
        k = [r for r in kept if r["year"] == y]
        t = [r for r in k if r["subject"]]
        sh = len({(r["exam_date"], r["shift"]) for r in u})
        b = sum(1 for r in blank if r["year"] == y)
        print(f"{y:<6}{sh:>7}{len(u):>6}{len(t):>8}"
              f"{100*len(t)/max(len(k),1):>6.0f}%{b:>7}")

    # Instrument comparability: official 2024 vs coaching 2025 (§ same test as
    # the English run, where reconstruction TVD was 4x the official year-to-year).
    def dist(pred):
        c = Counter(r["subject"] for r in tagged if pred(r))
        n = sum(c.values())
        return {k: v / n for k, v in c.items()} if n else {}

    d23, d24 = dist(lambda r: r["year"] == 2023), dist(lambda r: r["year"] == 2024)
    d25 = dist(lambda r: r["year"] == 2025)
    tvd = lambda a, b: sum(abs(a.get(k, 0) - b.get(k, 0))
                           for k in set(a) | set(b)) / 2
    print("\n" + "=" * 96)
    print("INSTRUMENT COMPARABILITY (subject mix)")
    print("=" * 96)
    print(f"  TVD(2023 vs 2024, both official)      = {tvd(d23, d24):.3f}")
    print(f"  TVD(2024 official vs 2025 coaching)   = {tvd(d24, d25):.3f}")
    print(f"\n{'subject':<22}{'2023':>9}{'2024':>9}{'2025':>9}")
    print("-" * 96)
    for s in sorted(set(d23) | set(d24) | set(d25),
                    key=lambda k: -d24.get(k, 0)):
        print(f"{s:<22}{d23.get(s,0):>9.3f}{d24.get(s,0):>9.3f}{d25.get(s,0):>9.3f}")

    untag = [r for r in kept if not r["subject"]]
    print(f"\nUntagged (non-blank): {len(untag)}")
    for r in untag[:10]:
        print("  -", normalize(r["stem"])[:98])

    (OUT / "ga_tagged.json").write_text(json.dumps(usable, indent=2))
    print(f"\nWrote {OUT/'ga_tagged.json'}")


if __name__ == "__main__":
    main()
