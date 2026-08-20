# Kerala PSC — Degree Level Main, question tooling

Tools for working with **Kerala Public Service Commission Degree Level Main
Examination** past papers: parsing them, tagging questions by topic, measuring
syllabus coverage, and assembling practice papers to the commission's own
published mark distribution.

Built for aspirants who cannot afford coaching. Everything here works from
papers KPSC has already published.

---

## The one rule

**No question and no answer in any output is written by a language model.**

Every question a paper emits is a real question from a real Kerala PSC Degree
Level Main Examination, carrying the commission's own published answer. A
generated question can look perfect and carry a subtly wrong key, and a practice
paper that teaches the wrong answer is worse than no practice paper at all.

The language model's only job is turning existing question text into a topic
label from a fixed taxonomy. Selection, counting and mark distribution are
deterministic Python you can re-run and audit.

| Job | How |
|---|---|
| PDF → structured questions | `kerala_parse.py` |
| Question → topic label | `kerala_tag.py` (LLM, constrained to the taxonomy) |
| Topic → syllabus coverage | `kerala_coverage.py` |
| Questions → practice paper | `kerala_mock.py`, `kerala_model_paper.py` |
| Questions → question bank | `kerala_qbank.py` |
| Questions → study notes | `kerala_notes.py`, `kerala_studybook.py`, `kerala_book.py` |

## Two limits it refuses to hide

A paper is built to KPSC's published distribution (Cat.No.26/2022): General
Knowledge 55, Current Affairs 15, Aptitude 10, English 10, Malayalam 10. Two
slot types cannot be filled honestly from an older corpus, and the tools report
them as deficits rather than padding them:

- **Current affairs** drawn from 2022–23 papers is stale for a later exam.
  Those slots are reported short, not quietly filled with obsolete news.
- **Malayalam** extracts as mojibake from the legacy non-Unicode font, so those
  slots are marked for practice from the original PDFs instead of printed as
  garbage.

A paper that says "94 of 100 slots filled, here is why" is more useful than one
that silently invents six questions.

## Requirements

- Python 3.11+
- A local [Ollama](https://ollama.com) model for tagging (qwen2.5:7b is enough —
  topic labelling is a constrained classification task, not a reasoning one)
- `pip install -r requirements.txt`

## Corpus

**Question papers are not included in this repository.** They are KPSC's
publications, and are downloadable from
[keralapsc.gov.in](https://www.keralapsc.gov.in). Point the tools at your own
copies.

## Status

Working, and used in practice. The tagger and taxonomy cover the Degree Level
Main syllabus; paper assembly fills 100 of 100 slots on a current corpus.

This is a personal project by a Kerala PSC tutor, published in case it saves
someone else the parsing work. Issues and pull requests welcome.

## Licence

GNU Affero General Public License v3.0 — see [LICENSE](LICENSE).

You may use, modify and redistribute this, including running it as a service,
provided you publish your source under the same licence. The intent is that
improvements to free exam preparation stay free.
