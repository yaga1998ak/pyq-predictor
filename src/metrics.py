"""Scoring functions for backtests.

Chosen so that gaming one metric shows up as damage in another:

  mae/rmse       accuracy of the count estimates themselves
  top_k_hit      whether the topics you would actually study rank correctly
  coverage       whether stated confidence is real -- a model with tight
                 intervals and poor coverage is lying, and only this catches it
"""

from __future__ import annotations

import numpy as np


def mae(pred: np.ndarray, actual: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - actual)))


def rmse(pred: np.ndarray, actual: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - actual) ** 2)))


def top_k_hit_rate(pred: np.ndarray, actual: np.ndarray, k: int = 10) -> float:
    """Overlap between predicted and actual top-k topics.

    This is the metric that maps onto behaviour: if you study the model's top 10
    topics, how many of the genuinely heaviest 10 did you cover?
    """
    pred_top = set(np.argsort(-pred)[:k])
    actual_top = set(np.argsort(-actual)[:k])
    return len(pred_top & actual_top) / k


def interval_coverage(lo: np.ndarray, hi: np.ndarray, actual: np.ndarray) -> float:
    """Fraction of topics whose true count fell inside the stated interval.

    For a 90% interval this should land near 0.90. Much lower means the model is
    overconfident; much higher means the intervals are uselessly wide.
    """
    return float(np.mean((actual >= lo) & (actual <= hi)))


def skill_score(model_err: float, baseline_err: float) -> float:
    """Fractional error reduction against a baseline.

    > 0 means the model beats the baseline; <= 0 means ship the baseline instead.
    """
    if baseline_err == 0:
        return 0.0
    return (baseline_err - model_err) / baseline_err


def evaluate(pred, actual, interval=None, k: int = 10) -> dict:
    out = {
        "mae": mae(pred, actual),
        "rmse": rmse(pred, actual),
        f"top_{k}_hit": top_k_hit_rate(pred, actual, k),
    }
    if interval is not None:
        lo, hi = interval
        out["coverage_90"] = interval_coverage(lo, hi, actual)
        out["mean_interval_width"] = float(np.mean(hi - lo))
    return out
