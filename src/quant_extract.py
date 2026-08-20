"""Extract the Quantitative Aptitude section from every eligible Tier-1 paper.

Reuses the section-assignment machinery validated in the Reasoning and English
runs (src/reasoning_extract.py): the section comes from question NUMBERING
STRUCTURE, not from a topic tagger.

  * continuous  -- numbers run 1..100 (2025 files)     -> Quant = Q51-75
  * per_section -- numbering restarts four times        -> Quant = 3rd block

Quant is the best-validated slot of the four: quant lexical fingerprints hit
98.6% inside Q51-75 in src/section_probe.py, which is what confirmed the
canonical Tier-1 order in the first place.

DI SETS are the structural wrinkle here, analogous to English cloze passages: one
table or chart serves several questions, printed once. They are lifted into their
own records and linked by di_id, with per-shift de-duplication keyed on a
normalised PREFIX -- the same fix the English passages needed, because each
reprint of a shared block is cut at a slightly different point.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from reasoning_extract import (assign_sections, numbered_chunks, parse_answer,
                               parse_options, stem_of)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"

# DI lead-ins. SSC prints an instruction plus the data block.
DI_MARK = re.compile(
    r"study the (following|given) (table|bar|pie|graph|chart|data|information)"
    r"|(?:based on|according to) the (following|given) (table|bar|pie|graph|chart)"
    r"|the (table|bar graph|pie chart|line graph) (below |given )?shows"
    r"|answer the questions? based on (the )?(following|given)"
    r"|the following (table|bar graph|pie chart|line graph)", re.I)

OPT_START = re.compile(r"\(a\)|\bAns\b", re.I)


def find_di(flat_chunk: str) -> tuple[str | None, bool]:
    m = DI_MARK.search(flat_chunk)
    if not m:
        return None, False
    opt = OPT_START.search(flat_chunk)
    precedes = bool(opt and m.start() < opt.start())
    body = flat_chunk[m.start():].strip()
    if len(body) < 120:
        return None, False
    return body, precedes


def di_key(text: str) -> str:
    norm = re.sub(r"[^a-z0-9]", "", text.lower())
    return hashlib.sha1(norm[:150].encode()).hexdigest()[:16]


# ---------------------------------------------------------- numeric features
NUM_RX = re.compile(r"-?\d+(?:\.\d+)?")
PCT_RX = re.compile(r"(\d+(?:\.\d+)?)\s*%")
RATIO_RX = re.compile(r"(\d+)\s*:\s*(\d+)(?:\s*:\s*(\d+))?")
FRAC_RX = re.compile(r"(\d+)\s*/\s*(\d+)")


def numeric_profile(stem: str, options) -> dict:
    """Extract the numerical parameters a Quant question is built from.

    This is the raw material for the parameter-distribution work: percentages,
    ratios, fractions, magnitudes and whether the option set is integer-valued.
    """
    nums = [float(x) for x in NUM_RX.findall(stem)]
    pcts = [float(x) for x in PCT_RX.findall(stem)]
    ratios = [tuple(int(g) for g in m if g) for m in RATIO_RX.findall(stem)]
    fracs = [(int(a), int(b)) for a, b in FRAC_RX.findall(stem)]
    opt_nums = []
    for _, t in options or []:
        f = NUM_RX.findall(t)
        if len(f) == 1:
            opt_nums.append(float(f[0]))
    return dict(
        n_numbers=len(nums),
        numbers=nums[:12],
        pcts=pcts[:6],
        pct_multiple_of_5=(bool(pcts) and all(p % 5 == 0 for p in pcts)),
        ratios=[list(r) for r in ratios[:4]],
        fractions=[list(f) for f in fracs[:4]],
        max_magnitude=max([abs(x) for x in nums], default=0.0),
        options_numeric=len(opt_nums) == len(options or []) and bool(opt_nums),
        options_all_integer=(bool(opt_nums)
                             and all(float(v).is_integer() for v in opt_nums)),
        option_values=opt_nums[:4],
    )


def main() -> None:
    from pypdf import PdfReader

    inv = json.load(open(OUT / "inventory.json"))
    eligible = [r for r in inv["raw"] if r["eligible"]]

    seen, papers, dropped = set(), [], []
    for r in sorted(eligible, key=lambda x: x["path"]):
        if r["sha256"] in seen:
            dropped.append(r["path"])
            continue
        seen.add(r["sha256"])
        papers.append(r)

    records, per_paper, di_sets = [], [], []
    layout_count = Counter()

    for rec in papers:
        p = ROOT / rec["path"]
        try:
            text = "".join(pg.extract_text() or "" for pg in PdfReader(str(p)).pages)
        except Exception:
            continue
        pairs = numbered_chunks(text)
        labels, layout = assign_sections(pairs)
        layout_count[layout] += 1

        n_q = n_opt = n_ans = 0
        shift_di: dict[str, str] = {}
        for (num, chunk), lab in zip(pairs, labels):
            if lab != "quant":
                continue
            n_q += 1
            flat = re.sub(r"\s+", " ", chunk).strip()
            opts, dialect = parse_options(flat)
            ans, prov = parse_answer(flat, dialect)
            stem = stem_of(flat)

            di_id = None
            body, precedes = find_di(flat)
            if body:
                k = di_key(body)
                if k not in shift_di:
                    did = f"{rec['exam_date']}-S{rec['shift']}-DI{len(shift_di)+1}"
                    shift_di[k] = did
                    di_sets.append(dict(
                        di_id=did, year=rec["year_dir"],
                        exam_date=rec["exam_date"], shift=rec["shift"],
                        text=body[:1500], length_chars=len(body),
                        source_pdf=Path(rec["path"]).name))
                di_id = shift_di[k]
                if precedes:
                    cut = DI_MARK.search(stem)
                    if cut:
                        stem = stem[:cut.start()].strip() or stem

            if opts:
                n_opt += 1
            if ans:
                n_ans += 1
            records.append(dict(
                qid=f"{rec['exam_date'] or rec['year_dir']}-S{rec['shift']}-QQ{num:03d}",
                year=rec["year_dir"], exam_date=rec["exam_date"],
                shift=rec["shift"], printed_number=num, layout=layout,
                source_pdf=Path(rec["path"]).name,
                source_tier=rec["source_tier"],
                is_reconstruction=rec["is_reconstruction"],
                stem=stem, options=opts, option_dialect=dialect,
                answer=ans, answer_provenance=prov,
                di_id=di_id, stem_len=len(stem),
                numeric=numeric_profile(stem, opts)))
        per_paper.append(dict(
            path=rec["path"], year=rec["year_dir"], date=rec["exam_date"],
            shift=rec["shift"], layout=layout, markers=len(pairs),
            quant=n_q, with_options=n_opt, with_answer=n_ans,
            di_sets=len(shift_di)))

    # ---------------------------------------------------------------- report
    print("=" * 104)
    print("QUANTITATIVE APTITUDE — SECTION INVENTORY")
    print("=" * 104)
    print(f"{'Year':<6}{'Shifts':>7}{'Layout':>14}{'Q found':>9}{'Q/shift':>9}"
          f"{'Nominal':>9}{'Recall':>8}{'w/Options':>11}{'w/Answer':>10}{'DI sets':>9}")
    print("-" * 104)
    tot = defaultdict(int)
    for y in sorted({r["year"] for r in per_paper}):
        rs = [r for r in per_paper if r["year"] == y]
        lay = Counter(r["layout"] for r in rs).most_common(1)[0][0]
        Q = sum(r["quant"] for r in rs)
        nom = 25 * len(rs)
        o = sum(r["with_options"] for r in rs)
        a = sum(r["with_answer"] for r in rs)
        d = sum(r["di_sets"] for r in rs)
        for k, v in (("shifts", len(rs)), ("Q", Q), ("nom", nom),
                     ("o", o), ("a", a), ("d", d)):
            tot[k] += v
        print(f"{y:<6}{len(rs):>7}{lay:>14}{Q:>9}{Q/len(rs):>9.1f}{nom:>9}"
              f"{100*Q/nom:>7.0f}%{o:>11}{a:>10}{d:>9}")
    print("-" * 104)
    print(f"{'ALL':<6}{tot['shifts']:>7}{'':>14}{tot['Q']:>9}"
          f"{tot['Q']/tot['shifts']:>9.1f}{tot['nom']:>9}"
          f"{100*tot['Q']/tot['nom']:>7.0f}%{tot['o']:>11}{tot['a']:>10}{tot['d']:>9}")

    print(f"\nLayouts: {dict(layout_count)}")
    print(f"Dropped byte-identical duplicate: {dropped}")
    print("\nAnswer provenance:")
    for k, v in Counter(r["answer_provenance"] for r in records).most_common():
        print(f"  {k:<28}{v:>6}  ({100*v/len(records):.1f}%)")

    print("\nEXTRACTION QUALITY")
    print(f"{'Year':<6}{'Q':>6}{'blank':>7}{'blank%':>8}{'despaced':>10}"
          f"{'numeric opts':>14}{'int opts':>10}")
    print("-" * 104)
    for y in sorted({r["year"] for r in records}):
        rs = [r for r in records if r["year"] == y]
        blank = sum(1 for r in rs if len(r["stem"].strip()) < 25)
        dsp = sum(1 for r in rs if len(r["stem"]) > 40
                  and r["stem"].count(" ") / max(len(r["stem"]), 1) < 0.10)
        numo = sum(1 for r in rs if r["numeric"]["options_numeric"])
        into = sum(1 for r in rs if r["numeric"]["options_all_integer"])
        print(f"{y:<6}{len(rs):>6}{blank:>7}{100*blank/len(rs):>7.0f}%{dsp:>10}"
              f"{100*numo/len(rs):>13.0f}%{100*into/len(rs):>9.0f}%")

    print(f"\nDI sets: {len(di_sets)}   linked questions: "
          f"{sum(1 for r in records if r['di_id'])}")
    if di_sets:
        by = Counter(d["year"] for d in di_sets)
        print(f"  by year: {dict(sorted(by.items()))}")
        per = Counter((d["exam_date"], d["shift"]) for d in di_sets)
        vals = sorted(per.values())
        print(f"  per shift: median {vals[len(vals)//2]}  max {vals[-1]}")

    out = [r for r in per_paper if not (20 <= r["quant"] <= 26)]
    print(f"\nShifts whose Quant count is outside 20-26: {len(out)}/{len(per_paper)}")
    for r in sorted(out, key=lambda x: x["quant"])[:12]:
        print(f"  Q={r['quant']:>3} markers={r['markers']:>4} {r['layout']:<12} {r['path']}")

    (OUT / "quant_questions.json").write_text(json.dumps(records, indent=2))
    (OUT / "quant_di_sets.json").write_text(json.dumps(di_sets, indent=2))
    (OUT / "quant_per_paper.json").write_text(json.dumps(per_paper, indent=2))
    print(f"\nWrote {OUT/'quant_questions.json'} ({len(records)} questions)")


if __name__ == "__main__":
    main()
