"""25 GA questions, one per predicted knowledge zone.

Every question is answerable from the corresponding self-contained note in
out/ga_zones_notes.md, which is what makes them grade S under §40: the student
who studied the note can answer them. Facts were verified when the notes were
written; the 2025 current-affairs items were checked against reporting.

Distractors are built from the "Common Confusions" section of each note -- the
confusions SSC actually exploits (red sandstone vs white marble, Fa-Hien vs
Hiuen Tsang, baking soda vs washing soda), not arbitrary wrong options.
"""

from __future__ import annotations

# (zone, stem, options a-d, answer, why_this_grade, difficulty, est_time_s)
QUESTIONS = [
    ("olympics_asian_games",
     "India's total medal tally at the Paris 2024 Olympic Games was:",
     ["5 medals", "6 medals", "7 medals", "4 medals"], "b",
     "Note 01 states the tally and the 1 silver + 5 bronze split.", "easy", 30),

    ("awards_honours",
     "How many Padma Vibhushan awards were conferred in 2025?",
     ["Five", "Six", "Seven", "Nine"], "c",
     "Note 05 lists all seven 2025 Padma Vibhushan recipients.", "medium", 35),

    ("constitution_general",
     "The words 'Socialist' and 'Secular' were inserted into the Preamble by which "
     "Constitutional Amendment?",
     ["42nd Amendment, 1976", "44th Amendment, 1978",
      "24th Amendment, 1971", "52nd Amendment, 1985"], "a",
     "Note 06: Preamble amended only once, by the 42nd.", "easy", 30),

    ("parliament_legislature",
     "A Money Bill is dealt with under which Article of the Constitution?",
     ["Article 108", "Article 110", "Article 112", "Article 117"], "b",
     "Note 03 gives Article 110 for Money Bill, 108 for joint sitting.",
     "medium", 35),

    ("president_governor",
     "The power of the Governor to grant pardons is contained in which Article?",
     ["Article 72", "Article 143", "Article 161", "Article 213"], "c",
     "Note 21: Article 72 is the President, 161 the Governor.", "medium", 35),

    ("new_criminal_laws",
     "The Bharatiya Nagarik Suraksha Sanhita, 2023 replaced which of the following?",
     ["Indian Penal Code, 1860", "Indian Evidence Act, 1872",
      "Code of Criminal Procedure, 1973", "Indian Contract Act, 1872"], "c",
     "Note 16: BNSS replaces the CrPC; BNS replaces the IPC.", "medium", 35),

    ("mughal_empire",
     "Which Mughal ruler is associated with the construction of the Buland "
     "Darwaza at Fatehpur Sikri?",
     ["Babur", "Akbar", "Jahangir", "Shah Jahan"], "b",
     "Note 07: Akbar built Fatehpur Sikri in red sandstone.", "easy", 30),

    ("mauryan_empire",
     "The Lion Capital adopted as India's National Emblem is located at:",
     ["Sanchi", "Sarnath", "Bodh Gaya", "Vaishali"], "b",
     "Note 17: Sarnath Lion Capital = National Emblem.", "easy", 25),

    ("gupta_and_post_gupta",
     "The Chinese traveller Fa-Hien visited India during the reign of:",
     ["Chandragupta Maurya", "Samudragupta",
      "Chandragupta II", "Harshavardhana"], "c",
     "Note 19: Fa-Hien under Chandragupta II; Hiuen Tsang under Harsha.",
     "medium", 35),

    ("colonial_acts_administration",
     "Dyarchy in the provinces was introduced by which Act?",
     ["Government of India Act, 1919", "Government of India Act, 1935",
      "Indian Councils Act, 1909", "Charter Act, 1833"], "a",
     "Note 18: 1919 introduced dyarchy; 1935 abolished it in provinces.",
     "medium", 35),

    ("freedom_movement",
     "The Purna Swaraj resolution was adopted at which session of the Indian "
     "National Congress?",
     ["Calcutta, 1928", "Lahore, 1929", "Karachi, 1931", "Lucknow, 1916"], "b",
     "Note 22: Lahore 1929, Nehru presiding.", "easy", 30),

    ("rivers_and_drainage",
     "Which of the following rivers flows through a rift valley and forms an "
     "estuary rather than a delta?",
     ["Godavari", "Mahanadi", "Narmada", "Krishna"], "c",
     "Note 08: Narmada and Tapi are west-flowing rift-valley rivers.",
     "medium", 35),

    ("soils_agriculture",
     "Black soil, also called regur soil, is most suited to the cultivation of:",
     ["Tea", "Cotton", "Rice", "Jute"], "b",
     "Note 12: black soil = cotton soil.", "easy", 25),

    ("state_identification",
     "The Tropic of Cancer passes through how many Indian states?",
     ["Six", "Seven", "Eight", "Nine"], "c",
     "Note 14: eight states.", "medium", 30),

    ("classical_dance",
     "Sattriya, one of the eight classical dance forms, originated in:",
     ["Odisha", "Manipur", "Assam", "Kerala"], "c",
     "Note 09: Sattriya, founded by Srimanta Sankaradeva, Assam.",
     "easy", 30),

    ("festivals",
     "The Wangala festival, also known as the 'Hundred Drums' festival, is "
     "celebrated by which community?",
     ["Bhils of Rajasthan", "Garos of Meghalaya",
      "Todas of Tamil Nadu", "Santhals of Jharkhand"], "b",
     "Note 11: Wangala = Garo, Meghalaya.", "medium", 35),

    ("music_instruments",
     "Ustad Bismillah Khan, a Bharat Ratna awardee, was a maestro of which "
     "instrument?",
     ["Sarod", "Shehnai", "Santoor", "Sitar"], "b",
     "Note 23: Bismillah Khan ↔ shehnai.", "easy", 25),

    ("folk_dance_by_state",
     "The Kalbelia dance, inscribed on UNESCO's list of Intangible Cultural "
     "Heritage, belongs to:",
     ["Gujarat", "Rajasthan", "Haryana", "Punjab"], "b",
     "Note 02: Kalbelia ↔ Rajasthan, UNESCO-listed.", "medium", 30),

    ("government_schemes",
     "Under Ayushman Bharat PM-JAY, the health cover provided per family per "
     "year is:",
     ["₹2 lakh", "₹3 lakh", "₹5 lakh", "₹10 lakh"], "c",
     "Note 04: ₹5 lakh per family per year.", "easy", 25),

    ("planning_and_indices",
     "The Human Development Index is published by:",
     ["World Bank", "IMF", "UNDP", "WEF"], "c",
     "Note 15: HDI ↔ UNDP, first published 1990.", "easy", 25),

    ("space_and_technology",
     "Chandrayaan-3 achieved its soft landing near the lunar south pole on:",
     ["14 July 2023", "23 August 2023",
      "2 September 2023", "22 October 2023"], "b",
     "Note 10: launched 14 Jul, landed 23 Aug — National Space Day.",
     "easy", 30),

    ("chemistry",
     "The chemical formula of washing soda is:",
     ["NaHCO₃", "Na₂CO₃·10H₂O", "NaOH", "CaSO₄·½H₂O"], "b",
     "Note 20: baking soda NaHCO₃ vs washing soda Na₂CO₃·10H₂O.",
     "medium", 30),

    ("biology",
     "Deficiency of vitamin B₁ (thiamine) causes which disease?",
     ["Pellagra", "Scurvy", "Beriberi", "Rickets"], "c",
     "Note 24: B₁ → beriberi; B₃ → pellagra; C → scurvy.", "easy", 25),

    ("census_demography",
     "According to Census 2011, the state with the highest population density is:",
     ["West Bengal", "Bihar", "Kerala", "Uttar Pradesh"], "b",
     "Note 25: Bihar 1,106 per km² — highest.", "medium", 35),

    ("other_sports_events",
     "Which Indian became the World Chess Champion in 2024?",
     ["R. Praggnanandhaa", "D. Gukesh", "Vidit Gujrathi", "Arjun Erigaisi"],
     "b",
     "Note 13: Gukesh, youngest ever world champion.", "easy", 25),
]


def as_records():
    out = []
    for zone, stem, opts, ans, why, diff, t in QUESTIONS:
        out.append(dict(
            section="general_awareness", topic=zone, subtopic=zone,
            stem=stem,
            options=[(l, o) for l, o in zip("abcd", opts)],
            answer=ans, grade="S", grade_reason=why,
            difficulty=diff, est_time=t,
            verified_by="EDITORIAL:fact-checked against zone note"))
    return out


if __name__ == "__main__":
    import json
    from collections import Counter
    from pathlib import Path
    recs = as_records()
    print(f"GA questions: {len(recs)}")
    print("answer spread:", dict(sorted(Counter(r["answer"] for r in recs).items())))
    print("difficulty:", dict(Counter(r["difficulty"] for r in recs)))
    print(f"total est. time: {sum(r['est_time'] for r in recs)}s "
          f"(sectional limit 900s)")
    zones = {r["topic"] for r in recs}
    print(f"distinct zones covered: {len(zones)} of 25")
    Path("../out/ga_paper_questions.json").write_text(json.dumps(recs, indent=2))
    print("Wrote ../out/ga_paper_questions.json")
