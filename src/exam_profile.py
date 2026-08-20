"""Exam profiles — what the brain needs to know about ANY exam.

The brain models how a question-setting team behaves. That model is not
SSC-specific, but it does need to know the shape of the exam it is looking at:
how many sections, how they are weighted, whether the paper is timed per
section. Everything here is declarative. Adding an exam is adding a profile,
not writing code.

Domain hypotheses are opt-in. "Government-policy content is over-represented"
is a real, measured effect for an Indian public-service recruiter and is
meaningless for, say, a private certification exam — so a profile declares
which domain hypotheses apply to it.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ExamProfile:
    key: str
    name: str
    sections: tuple[str, ...]
    questions_per_section: int
    marks_per_question: float
    negative_per_wrong: float
    sectional_seconds: int | None       # None = single whole-paper budget
    source: str                         # where the structure was verified
    domain_hypotheses: tuple[str, ...] = field(default=())
    # Non-uniform papers: per-section counts, aligned with `sections`.
    # None means every section carries `questions_per_section`.
    section_questions: tuple[int, ...] | None = None

    @property
    def total_questions(self) -> int:
        if self.section_questions is not None:
            return sum(self.section_questions)
        return len(self.sections) * self.questions_per_section

    @property
    def total_marks(self) -> float:
        return self.total_questions * self.marks_per_question


SSC_CGL_T1 = ExamProfile(
    key="ssc_cgl_t1",
    name="SSC CGL Tier-I 2026",
    sections=("reasoning", "general_awareness", "quant", "english"),
    questions_per_section=25,
    marks_per_question=2.0,
    negative_per_wrong=0.50,
    sectional_seconds=900,              # 15 min per subject, new in 2026
    source="Notice_of_adv_cgl_2026.pdf §13.8 (data/official/)",
    domain_hypotheses=("policy_pressure", "civic_core"),
)

# A second profile, to keep the brain honest about being general. Structure
# per SSC's CHSL notice; it is NOT verified against a 2026 notification and is
# marked as such, because an unverified structure printed as fact is exactly
# the error this project already made once with sectional timing.
SSC_CHSL_T1 = ExamProfile(
    key="ssc_chsl_t1",
    name="SSC CHSL Tier-I (UNVERIFIED structure)",
    sections=("reasoning", "general_awareness", "quant", "english"),
    questions_per_section=25,
    marks_per_question=2.0,
    negative_per_wrong=0.50,
    sectional_seconds=None,
    source="UNVERIFIED — confirm against the current notification before use",
    domain_hypotheses=("policy_pressure", "civic_core"),
)

# --- Kerala PSC degree-level Main exams ---------------------------------
# STRUCTURE UNVERIFIED. Aggregator sources contradict each other on duration
# (75 vs 90 min) and on the section list. Confirm against the Kerala PSC
# notification PDF before any of these numbers reach a printed page.
#
# Domain hypotheses: policy_pressure and civic_core carry over (public-service
# recruiter). kerala_local is new -- state-specific content is a large, stable
# share here and has no SSC analogue.

KERALA_UNIV_ASSISTANT_MAIN = ExamProfile(
    key="kerala_univ_assistant_main",
    name="Kerala PSC University Assistant Main",
    sections=("general_knowledge", "current_affairs", "aptitude",
              "general_english", "regional_language"),
    questions_per_section=0,            # unused; see section_questions
    section_questions=(55, 15, 10, 10, 10),
    marks_per_question=1.0,
    negative_per_wrong=0.33,
    sectional_seconds=None,             # whole-paper budget
    source=("VERIFIED. Official 'Detailed Syllabus and Mark Distribution (Degree "
           "Level Main Examination)' Cat.No.26/2022 -- see "
           "data/official/kerala/degree_main_syllabus_2022_ksfe.pdf: GK 55 + "
           "Current Affairs 15 + Arithmetic/Reasoning 10 + English 10 + "
           "Malayalam 10. Cross-checked independently against block "
           "boundaries in paper 160/2023 -- all four match. Duration "
           "UNVERIFIED. NOTE: Secretariat/KPSC Assistant Main is a DIFFERENT "
           "paper (Paper I+II, no Current Affairs block) -- do not pool it."),
    domain_hypotheses=("policy_pressure", "civic_core", "kerala_local"),
)

KERALA_COMPANY_BOARD_ASSISTANT_MAIN = ExamProfile(
    key="kerala_company_board_main",
    name="Kerala PSC Company/Corporation/Board Assistant Main",
    sections=("general_knowledge", "current_affairs", "aptitude",
              "general_english", "regional_language"),
    questions_per_section=0,
    section_questions=(55, 15, 10, 10, 10),
    marks_per_question=1.0,
    negative_per_wrong=0.33,
    sectional_seconds=None,
    source=("VERIFIED. Official 'Detailed Syllabus and Mark Distribution (Degree "
           "Level Main Examination)' Cat.No.26/2022 -- see "
           "data/official/kerala/degree_main_syllabus_2022_ksfe.pdf: GK 55 + "
           "Current Affairs 15 + Arithmetic/Reasoning 10 + English 10 + "
           "Malayalam 10. Cross-checked independently against block "
           "boundaries in paper 160/2023 -- all four match. Duration "
           "UNVERIFIED. NOTE: Secretariat/KPSC Assistant Main is a DIFFERENT "
           "paper (Paper I+II, no Current Affairs block) -- do not pool it."
           " Cat.No.26/2022 covers assistant-grade posts in "
           "KSFE/KSEB/KMML/KELTRON etc -- the govt-owned company/corporation/"
           "board family. Syllabus verified; no Company/Board Main PAPER yet."),
    domain_hypotheses=("policy_pressure", "civic_core", "kerala_local"),
)


PROFILES = {p.key: p for p in (SSC_CGL_T1, SSC_CHSL_T1,
                               KERALA_UNIV_ASSISTANT_MAIN,
                               KERALA_COMPANY_BOARD_ASSISTANT_MAIN)}


def get(key: str) -> ExamProfile:
    if key not in PROFILES:
        raise KeyError(f"unknown exam {key!r}; known: {sorted(PROFILES)}")
    return PROFILES[key]


if __name__ == "__main__":
    for k, p in PROFILES.items():
        print(f"  {k:<14} {p.name}")
        print(f"     {p.total_questions} questions, {p.total_marks:.0f} marks, "
              f"-{p.negative_per_wrong}/wrong, "
              f"{'sectional '+str(p.sectional_seconds)+'s' if p.sectional_seconds else 'whole-paper timing'}")
        print(f"     domain hypotheses: {list(p.domain_hypotheses) or 'none'}")
        print(f"     source: {p.source}")
