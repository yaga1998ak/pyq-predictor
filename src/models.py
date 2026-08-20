"""Forecasting models for per-topic question counts.

Every model exposes the same interface:

    model.fit(history)                 history = [(year, Counter(topic -> count)), ...]
    model.predict(n_questions)         -> np.ndarray over taxonomy.topics
    model.predict_interval(n, level)   -> (lo, hi) arrays, or None if the model
                                          has no notion of uncertainty

The baselines exist to be beaten. If DirichletMultinomial cannot beat
MeanLastK on walk-forward backtests, the extra machinery is buying nothing and
you should ship the baseline -- that judgement is the entire point of the
backtest harness.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from schema import Taxonomy


class Predictor:
    name = "base"

    def __init__(self, taxonomy: Taxonomy, section_aware: bool = True):
        self.tax = taxonomy
        self.topics = taxonomy.topics
        self.idx = {t: i for i, t in enumerate(self.topics)}
        self.section_aware = section_aware
        # False = scale to observed (tagged) totals, for backtesting.
        # True  = scale to the real 25-per-section exam pattern, for forecasting.
        self.nominal_sections = False
        self.history: list[tuple[int, Counter]] = []

    def fit(self, history: list[tuple[int, Counter]]) -> "Predictor":
        self.history = sorted(history, key=lambda x: x[0])
        return self

    def _counts_matrix(self) -> np.ndarray:
        """rows = years, cols = topics."""
        m = np.zeros((len(self.history), len(self.topics)))
        for r, (_, counts) in enumerate(self.history):
            for topic, c in counts.items():
                if topic in self.idx:
                    m[r, self.idx[topic]] = c
        return m

    def _section_masks(self) -> dict[str, np.ndarray]:
        masks = {}
        for section in self.tax.sections:
            mask = np.array(
                [self.tax.topic_to_section[t] == section for t in self.topics]
            )
            masks[section] = mask
        return masks

    def _section_targets(self, n_questions: int) -> dict[str, float]:
        """How many questions to allocate to each section, summing to n_questions.

        The nominal exam pattern is 25 per section, and using it directly was a
        bug: it only holds when every question was extracted AND tagged. Under
        partial coverage the shortfall is not uniform -- rule tagging reaches
        18.7 quant questions per paper but only 6.2 general-awareness ones -- so
        forcing 25 each made predictions total 100 against actuals totalling 61,
        and every error metric measured that scale gap instead of forecast skill.

        Deriving targets from the TRAINING history's observed section shares
        keeps the section-proportion prior while matching the actual scale, and
        adapts automatically as tagger coverage changes.

        Two modes, because backtesting and forecasting need opposite scaling:

          observed (default)  Match the partially-covered data. Used when scoring
                              against actual tagged counts -- predicting a full
                              25-per-section paper against a 61-question actual
                              measures the coverage gap, not forecast skill.
          nominal             Match the real exam pattern: 25 per section. Used
                              when forecasting a FUTURE paper, which will have 25
                              per section regardless of what your tagger reached.
                              Scaling by observed shares here would bake tagger
                              coverage into the prediction -- general awareness is
                              under-tagged (6.2/paper vs quant's 18.7), so it
                              would be under-forecast for a reason that has
                              nothing to do with the exam.

        Within-section proportions are used in both modes; only the section
        totals differ. That assumes coverage is roughly random WITHIN a section,
        which is weaker for general awareness than elsewhere.
        """
        masks = self._section_masks()
        uniform = {s: n_questions / len(masks) for s in masks}

        if self.nominal_sections:
            per_paper = self.tax.questions_per_paper or n_questions
            scale = n_questions / per_paper
            return {s: self.tax.section_size(s) * scale for s in masks}

        if not self.history:
            return uniform
        m = self._counts_matrix()
        totals = {s: float(m[:, mask].sum()) for s, mask in masks.items()}
        grand = sum(totals.values())
        if grand <= 0:
            return uniform
        return {s: n_questions * totals[s] / grand for s in masks}

    def _apply_section_constraints(self, raw: np.ndarray, n_questions: int) -> np.ndarray:
        """Rescale within each section so section totals match the observed scale."""
        if not self.section_aware:
            return raw
        targets = self._section_targets(n_questions)
        out = np.zeros_like(raw)
        for section, mask in self._section_masks().items():
            target = targets[section]
            block = raw[mask]
            total = block.sum()
            out[mask] = block / total * target if total > 0 else target / mask.sum()
        return out

    def predict(self, n_questions: int) -> np.ndarray:
        raise NotImplementedError

    def predict_interval(self, n_questions: int, level: float = 0.90, **kwargs):
        """Point-estimate models have no uncertainty model; None means 'not offered'.

        Accepts **kwargs so callers can pass n_papers uniformly without having to
        know which models support intervals.
        """
        return None


class LastYear(Predictor):
    """Whatever appeared last year will appear again. Naive but surprisingly stiff."""

    name = "last_year"

    def predict(self, n_questions: int) -> np.ndarray:
        if not self.history:
            return np.full(len(self.topics), n_questions / len(self.topics))
        return self._apply_section_constraints(self._counts_matrix()[-1].astype(float), n_questions)


class MeanLastK(Predictor):
    """Mean of the last k papers. THE baseline to beat -- it is hard to beat."""

    name = "mean_last_3"

    def __init__(self, taxonomy, k: int = 3, section_aware: bool = True):
        super().__init__(taxonomy, section_aware)
        self.k = k
        self.name = f"mean_last_{k}"

    def predict(self, n_questions: int) -> np.ndarray:
        if not self.history:
            return np.full(len(self.topics), n_questions / len(self.topics))
        m = self._counts_matrix()[-self.k :]
        return self._apply_section_constraints(m.mean(axis=0), n_questions)


class DirichletMultinomial(Predictor):
    """Bayesian topic-proportion model with exponential recency weighting.

    Topic counts in a paper are multinomial draws; the proportions themselves
    drift year to year. A Dirichlet prior updated by recency-weighted historical
    counts captures both, and -- unlike the point-estimate baselines -- yields
    calibrated credible intervals via the posterior predictive.

    half_life: years for a paper's influence to halve. None = no decay.
    alpha0:    prior strength per topic. Higher = more shrinkage toward uniform,
               which protects against overreacting to one anomalous paper.
    """

    name = "dirichlet_multinomial"

    def __init__(
        self,
        taxonomy: Taxonomy,
        alpha0: float = 0.8,
        half_life: float | None = 3.0,
        section_aware: bool = True,
    ):
        super().__init__(taxonomy, section_aware)
        self.alpha0 = alpha0
        self.half_life = half_life
        self.alpha: np.ndarray | None = None
        # Distinct name per hyperparameter set. Without this every variant keys
        # into the same results bucket and the backtest averages them together,
        # silently reporting a blend of models that does not exist.
        hl = "inf" if half_life is None else f"{half_life:g}"
        self.name = f"dirichlet_a{alpha0:g}_hl{hl}"

    def fit(self, history):
        super().fit(history)
        alpha = np.full(len(self.topics), self.alpha0, dtype=float)
        if self.history:
            latest = self.history[-1][0]
            for year, counts in self.history:
                if self.half_life:
                    w = 0.5 ** ((latest - year) / self.half_life)
                else:
                    w = 1.0
                for topic, c in counts.items():
                    if topic in self.idx:
                        alpha[self.idx[topic]] += w * c
        self.alpha = alpha
        return self

    def predict(self, n_questions: int) -> np.ndarray:
        if self.alpha is None:
            raise RuntimeError("call fit() first")
        return self._apply_section_constraints(
            n_questions * self.alpha / self.alpha.sum(), n_questions
        )

    def predict_interval(
        self,
        n_questions: int,
        level: float = 0.90,
        draws: int = 4000,
        n_papers: int = 1,
    ):
        """Posterior predictive interval.

        Samples proportions from the Dirichlet posterior, then counts from the
        multinomial -- so the interval carries both parameter uncertainty and
        sampling noise, which is why it is wider (and more honest) than a
        bootstrap over historical counts alone.

        n_papers matters and is easy to get wrong. When a year's observed counts
        are the MEAN over several shifts, that mean has variance reduced by a
        factor of n_papers. Generating single-paper intervals and comparing them
        to a multi-shift average produces ~100% coverage -- intervals that look
        reassuring and mean nothing. Pass the number of shifts averaged.
        """
        if self.alpha is None:
            raise RuntimeError("call fit() first")
        rng = np.random.default_rng(0)
        masks = (
            self._section_masks()
            if self.section_aware
            else {"all": np.ones(len(self.topics), bool)}
        )
        targets = self._section_targets(n_questions)
        samples = np.zeros((draws, len(self.topics)))
        for section, mask in masks.items():
            n_sec = targets[section] if self.section_aware else n_questions
            p = rng.dirichlet(self.alpha[mask], size=draws)
            for d in range(draws):
                # average n_papers independent sittings, matching how the
                # observed counts for this year were themselves aggregated
                acc = np.zeros(int(mask.sum()))
                for _ in range(n_papers):
                    acc += rng.multinomial(int(round(n_sec)), p[d])
                samples[d, mask] = acc / n_papers
        lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
        return np.quantile(samples, lo_q, axis=0), np.quantile(samples, hi_q, axis=0)


def all_models(taxonomy: Taxonomy) -> list[Predictor]:
    """The standard comparison set used by the backtest."""
    return [
        LastYear(taxonomy),
        MeanLastK(taxonomy, k=3),
        MeanLastK(taxonomy, k=5),
        DirichletMultinomial(taxonomy, alpha0=0.8, half_life=3.0),
        DirichletMultinomial(taxonomy, alpha0=0.5, half_life=2.0),
        DirichletMultinomial(taxonomy, alpha0=2.0, half_life=None),
    ]
