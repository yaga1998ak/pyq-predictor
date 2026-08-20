"""Maximum-coverage selection of 25 GA knowledge zones, with blind 2025 validation.

The objective is §2: maximise the expected number of 2026 GA questions a student
could answer from ONLY the selected notes. That is a weighted maximum-coverage
problem, not a top-25 frequency list, because a note covers more than its own
label:

    coverage(S) = sum_over_micro  freq[micro] * max_{z in S} M[z][micro]

M is the coverage matrix. M[z][z] = 1.0; explicitly adjacent clusters get
PARTIAL (a note on Fundamental Duties genuinely helps on a Fundamental Rights vs
DPSP distinction question); same-subject-but-unrelated gets SAME_SUBJECT, since a
polity note gives some grounding but will not answer a specific Article question.

The objective is monotone submodular, so greedy selection carries the standard
(1 - 1/e) approximation guarantee and is the right algorithm here -- no annealing
or quantum-inspired search is warranted for a 60-element ground set, and claiming
otherwise would be decoration (§38, §81).

VALIDATION: zones are selected from 2023-2024 ONLY, frozen, and then scored
against actual 2025 questions. Baselines are scored on the identical test set.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
K = 25
PARTIAL = 0.45
SAME_SUBJECT = 0.10

# Adjacency asserted from the taxonomy's own structure: clusters SSC tests
# against each other, so a note on one materially helps on the other. Kept
# explicit and small rather than inferred from thin co-occurrence counts.
ADJACENT = [
    ("fundamental_rights", "fundamental_duties"),
    ("fundamental_rights", "dpsp"),
    ("fundamental_duties", "dpsp"),
    ("fundamental_duties", "constitutional_amendments"),
    ("constitution_general", "constituent_assembly"),
    ("constitution_general", "schedules_and_lists"),
    ("constitution_general", "constitutional_amendments"),
    ("parliament_legislature", "president_governor"),
    ("parliament_legislature", "constitutional_bodies"),
    ("judiciary", "constitutional_bodies"),
    ("new_criminal_laws", "judiciary"),
    ("mauryan_empire", "buddhism_jainism"),
    ("indus_valley", "vedic_period"),
    ("delhi_sultanate", "mughal_empire"),
    ("mughal_empire", "monuments_heritage"),
    ("colonial_acts_administration", "governors_general_viceroys"),
    ("freedom_movement", "peasant_tribal_movements"),
    ("freedom_movement", "socio_religious_reform"),
    ("gupta_and_post_gupta", "chola_and_south"),
    ("rivers_and_drainage", "mountains_and_relief"),
    ("monsoon_and_climate", "soils_agriculture"),
    ("soils_agriculture", "minerals_industry"),
    ("state_identification", "states_and_boundaries"),
    ("state_identification", "folk_dance_by_state"),
    ("classical_dance", "folk_dance_by_state"),
    ("classical_dance", "music_instruments"),
    ("festivals", "folk_dance_by_state"),
    ("government_schemes", "planning_and_indices"),
    ("planning_and_indices", "reports_and_indices"),
    ("national_income", "budget_and_taxation"),
    ("rbi_and_banking", "budget_and_taxation"),
    ("olympics_asian_games", "world_cups_championships"),
    ("olympics_asian_games", "other_sports_events"),
    ("trophies_and_terms", "other_sports_events"),
    ("awards_honours", "books_authors"),
    ("protected_areas", "ecology_biodiversity"),
    ("conventions_and_reports", "ecology_biodiversity"),
    ("physics", "scientists_discoveries"),
    ("chemistry", "scientists_discoveries"),
    ("biology", "ecology_biodiversity"),
    ("space_and_technology", "scientists_discoveries"),
]


def load(years):
    recs = json.load(open(OUT / "ga_tagged.json"))
    sel = [r for r in recs if r["year"] in years and r.get("micro")]
    return sel


def freq_per_shift(recs):
    shifts = len({(r["exam_date"], r["shift"]) for r in recs})
    c = Counter(r["micro"] for r in recs)
    return {m: v / shifts for m, v in c.items()}, shifts, c


def subject_of(recs):
    out = {}
    for r in recs:
        out.setdefault(r["micro"], r["subject"])
    return out


def build_matrix(micros, subj):
    adj = defaultdict(set)
    for a, b in ADJACENT:
        adj[a].add(b)
        adj[b].add(a)
    M = {}
    for z in micros:
        row = {}
        for m in micros:
            if z == m:
                row[m] = 1.0
            elif m in adj[z]:
                row[m] = PARTIAL
            elif subj.get(z) and subj.get(z) == subj.get(m):
                row[m] = SAME_SUBJECT
            else:
                row[m] = 0.0
        M[z] = row
    return M


def coverage(S, freq, M):
    total = 0.0
    for m, f in freq.items():
        best = 0.0
        for z in S:
            v = M.get(z, {}).get(m, 0.0)
            if v > best:
                best = v
        total += f * best
    return total


def greedy(freq, M, k=K):
    """Submodular greedy: repeatedly add the zone with the largest marginal gain."""
    chosen, gains = [], []
    pool = list(M)
    while len(chosen) < k and pool:
        best_z, best_gain = None, -1.0
        base = coverage(chosen, freq, M)
        for z in pool:
            g = coverage(chosen + [z], freq, M) - base
            if g > best_gain:
                best_z, best_gain = z, g
        if best_z is None:
            break
        chosen.append(best_z)
        gains.append(best_gain)
        pool.remove(best_z)
    return chosen, gains


def top_k_frequency(freq, k=K):
    return [m for m, _ in sorted(freq.items(), key=lambda x: -x[1])[:k]]


def evaluate(zones, test_recs, M):
    """Coverage on ACTUAL test questions, at the levels §43 asks for."""
    direct = partial = related = none = 0
    for r in test_recs:
        m = r["micro"]
        best = max((M.get(z, {}).get(m, 0.0) for z in zones), default=0.0)
        if best >= 1.0:
            direct += 1
        elif best >= PARTIAL:
            partial += 1
        elif best > 0:
            related += 1
        else:
            none += 1
    n = len(test_recs)
    return dict(n=n, direct=direct, partial=partial, related=related, none=none,
                direct_pct=100 * direct / n,
                direct_plus_partial_pct=100 * (direct + partial) / n,
                any_pct=100 * (direct + partial + related) / n)


def main() -> None:
    # ---------------- BLIND: select from 2023-2024, test on 2025 -------------
    train = load({2023, 2024})
    test = load({2025})
    freq_tr, sh_tr, _ = freq_per_shift(train)
    subj = subject_of(train + test)
    micros = sorted(set(freq_tr) | {m for r in test for m in [r["micro"]]})
    M = build_matrix(micros, subj)

    print("=" * 100)
    print("BLIND ZONE SELECTION — trained on official 2023-2024 only")
    print("=" * 100)
    print(f"training questions {len(train)} across {sh_tr} shifts; "
          f"ground set {len(micros)} micro-subtopics")

    zones, gains = greedy(freq_tr, M)
    print(f"\nFrozen 25 zones (greedy max-coverage), with marginal gain:")
    for i, (z, g) in enumerate(zip(zones, gains), 1):
        print(f"  {i:>2}. {z:<34}{freq_tr.get(z,0):>6.2f}/shift   +{g:.3f}")

    print("\n--- revealing actual 2025 ---")
    res = evaluate(zones, test, M)
    print(f"2025 tagged GA questions: {res['n']}")
    print(f"  directly covered            {res['direct']:>5}  {res['direct_pct']:.1f}%")
    print(f"  partially (adjacent cluster){res['partial']:>5}  "
          f"{100*res['partial']/res['n']:.1f}%")
    print(f"  same-subject only           {res['related']:>5}  "
          f"{100*res['related']/res['n']:.1f}%")
    print(f"  not covered                 {res['none']:>5}  "
          f"{100*res['none']/res['n']:.1f}%")
    print(f"  => DIRECT+PARTIAL coverage  {res['direct_plus_partial_pct']:.1f}%")

    # ---------------- baselines on the identical test set -------------------
    print("\n" + "=" * 100)
    print("BASELINE COMPARISON (same frozen test set)")
    print("=" * 100)
    rng = random.Random(7)
    rows = []
    rows.append(("greedy_max_coverage", zones))
    rows.append(("top25_frequency", top_k_frequency(freq_tr)))
    fr_last, _, _ = freq_per_shift([r for r in train if r["year"] == 2024])
    rows.append(("top25_last_year_only", top_k_frequency(fr_last)))
    rnd_scores = []
    for _ in range(200):
        rnd_scores.append(evaluate(rng.sample(micros, min(K, len(micros))),
                                   test, M)["direct_plus_partial_pct"])
    print(f"{'selector':<26}{'direct%':>9}{'direct+partial%':>18}{'uncovered%':>12}")
    print("-" * 100)
    for name, zs in rows:
        r = evaluate(zs, test, M)
        print(f"{name:<26}{r['direct_pct']:>9.1f}{r['direct_plus_partial_pct']:>18.1f}"
              f"{100*r['none']/r['n']:>12.1f}")
    print(f"{'random25 (mean of 200)':<26}{'—':>9}"
          f"{sum(rnd_scores)/len(rnd_scores):>18.1f}{'—':>12}")

    # ---------------- FINAL: retrain on 2023-2025 ---------------------------
    allr = load({2023, 2024, 2025})
    freq_all, sh_all, cnt_all = freq_per_shift(allr)
    subj_all = subject_of(allr)
    micros_all = sorted(freq_all)
    M_all = build_matrix(micros_all, subj_all)
    final, fgains = greedy(freq_all, M_all)

    print("\n" + "=" * 100)
    print(f"FINAL 2026 ZONES — retrained on 2023-2025 ({sh_all} shifts, "
          f"{len(allr)} labelled questions)")
    print("=" * 100)
    print(f"{'#':>3} {'zone':<34}{'subject':<16}{'/shift':>8}{'exp/25':>8}{'gain':>8}")
    print("-" * 100)
    tot_exp = 0.0
    for i, (z, g) in enumerate(zip(final, fgains), 1):
        f = freq_all.get(z, 0)
        tot_exp += f
        print(f"{i:>3} {z:<34}{subj_all.get(z,'—'):<16}{f:>8.2f}"
              f"{f:>8.2f}{g:>8.3f}")
    print("-" * 100)
    cov_final = coverage(final, freq_all, M_all)
    tot_freq = sum(freq_all.values())
    print(f"Expected questions directly on a selected zone : {tot_exp:.2f} per shift")
    print(f"Weighted coverage incl. partial credit         : {cov_final:.2f} "
          f"of {tot_freq:.2f} labelled/shift ({100*cov_final/tot_freq:.1f}%)")

    json.dump({"blind_zones": zones, "blind_result": res,
               "final_zones": final,
               "final_freq": {z: freq_all.get(z, 0) for z in final},
               "subject": {z: subj_all.get(z) for z in final},
               "n_shifts": sh_all, "n_labelled": len(allr),
               "coverage_weighted_pct": 100 * cov_final / tot_freq},
              open(OUT / "ga_zones_2026.json", "w"), indent=2)
    print(f"\nWrote {OUT/'ga_zones_2026.json'}")


if __name__ == "__main__":
    main()
