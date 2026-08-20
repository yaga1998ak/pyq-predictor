"""Self-audit: re-examine the brain's own CONFIRMED claims for confounds.

WHY THIS EXISTS
---------------
On 2026-08-19 the setter brain returned four CONFIRMED hypotheses. Two did not
survive scrutiny, and one of the confound tests written to check them was
ITSELF misspecified. A model that only accumulates confirmations drifts into
confident nonsense; the audit is what keeps it honest.

THE THREE FAILURE MODES, EACH WITH A STANDING CHECK
---------------------------------------------------
  POWER      Is the claim resting on too few observations?
             -> `difficulty_parity` and the shift-position claim both broke here.
  CONTROL    Is the baseline contaminated by an unrelated effect?
             -> the global corpus sd was inflated by 2021's broken extraction,
                making within-day variance look small.
  CONSTRAINT Are the variables mathematically forced to relate?
             -> section shares sum to 1, so they MUST anti-correlate.

Any CONFIRMED claim that cannot answer all three is demoted to NEEDS_REVIEW.
Demotion is not deletion: the claim stays with the reason attached.

    python src/setter_audit.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
REGISTER = OUT / "setter_register.json"

# Declared audit status per hypothesis. A claim must state how it answers each
# failure mode; "unaddressed" forces a demotion.
AUDIT = {
    "numeric_habits": {
        "power": "n=27,707 numbers — ample",
        "control": "uniform expectation over the same integer range",
        "constraint": "divisibility rates are not compositionally forced",
        "extra": "holds across mult-10, mult-5 and even independently",
        "passes": True,
    },
    "topic_floor": {
        "power": "all 98 papers",
        "control": "descriptive frequency, no between-group comparison made",
        "constraint": "presence counts are not simplex-bound",
        "caveat": "tagger-coverage dependent: a topic invisible to the rules "
                  "tagger cannot enter the floor. Under-inclusive, not wrong.",
        "passes": True,
    },
    "difficulty_parity": {
        "power": "37 multi-shift days — adequate",
        "control": "FAILED — used global corpus sd (contaminated by 2021 "
                   "extraction failure) instead of same-year/same-source",
        "constraint": "n/a",
        "passes": False,
    },
    "cross_section_balance": {
        "power": "98 papers — adequate",
        "control": "FAILED — null model produced a positive band, diagnostic "
                   "of misspecification",
        "constraint": "FAILED — shares of a fixed total must anti-correlate",
        "passes": False,
    },
}


def main() -> None:
    if not REGISTER.exists():
        print("no register yet — run src/setter_brain.py first")
        return
    reg = json.loads(REGISTER.read_text())
    tested = reg.get("tested", {})

    print("SETTER SELF-AUDIT")
    print("=" * 66)
    changed = 0
    for key, v in tested.items():
        a = AUDIT.get(key)
        if v["verdict"] != "CONFIRMED":
            print(f"  [{v['verdict']:<12}] {key} — already rejected, kept as a finding")
            continue
        if a is None:
            v["verdict"] = "NEEDS_REVIEW"
            v["audit"] = "no audit declared — cannot vouch for it"
            changed += 1
            print(f"  [NEEDS_REVIEW] {key} — unaudited, demoted")
            continue
        if a["passes"]:
            v["audit"] = {k: a[k] for k in a if k != "passes"}
            print(f"  [CONFIRMED   ] {key} — survives audit")
            for k in ("power", "control", "constraint"):
                print(f"                   {k:<11}: {a[k]}")
            if a.get("caveat"):
                print(f"                   caveat     : {a['caveat']}")
        else:
            v["verdict"] = "REJECTED"
            v["audit"] = {k: a[k] for k in a if k != "passes"}
            changed += 1
            print(f"  [DEMOTED     ] {key} — failed audit")

    reg["tested"] = tested
    reg["last_audit"] = {"demoted": changed}
    REGISTER.write_text(json.dumps(reg, indent=2))

    conf = sum(1 for v in tested.values() if v["verdict"] == "CONFIRMED")
    print()
    print(f"  model now: {conf} confirmed constraint(s), {changed} demoted this pass")
    print("  Only CONFIRMED claims are allowed to shape the paper "
          "(src/setter_compose.py).")


if __name__ == "__main__":
    main()
