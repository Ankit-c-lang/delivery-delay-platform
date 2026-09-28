"""Evaluation metrics (PLAN.md §4.6).

**PR-AUC leads.** With roughly 7% positives, ROC-AUC is optimistic and flattering: a model
can look excellent on ROC while being useless at the operating point anyone would actually
run. Average precision integrates precision over recall and is dominated by the positive
class, which is the class that matters here.

The trivial baseline is reported alongside everything else and is worth stating out loud:
predicting never-late gives ~93% accuracy and **0% recall**. That is the cleanest way to show
why accuracy is the wrong metric for this problem, and it pre-empts the obvious trap question.

``recall_at_capacity`` is the operational metric: *if ops can review the top k% of orders each
day, what share of late orders do they catch?* It converts a ranking into a staffing decision,
which is the form the business actually consumes.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

logger = logging.getLogger(__name__)

#: Default review capacity: ops look at the top 10% of orders (PLAN.md §4.6).
DEFAULT_CAPACITY = 0.10


def _as_arrays(y_true: ArrayLike, y_score: ArrayLike) -> tuple[NDArray, NDArray]:
    """Validate and coerce a label/score pair.

    Raises:
        ValueError: On length mismatch, an empty input, non-binary labels, or non-finite
            scores. Every one of these would otherwise produce a plausible number from
            nonsense, which is worse than an error.
    """
    truth = np.asarray(y_true)
    score = np.asarray(y_score, dtype=float)
    if truth.shape != score.shape:
        raise ValueError(f"length mismatch: y_true {truth.shape} vs y_score {score.shape}")
    if truth.size == 0:
        raise ValueError("cannot compute metrics on an empty input")
    truth = truth.astype(bool) if truth.dtype == bool else truth.astype(float)
    unique = set(np.unique(truth).tolist())
    if not unique <= {0.0, 1.0}:
        raise ValueError(f"y_true must be binary, found values {sorted(unique)}")
    if not np.isfinite(score).all():
        raise ValueError("y_score contains NaN or infinity")
    return truth.astype(int), score


def pr_auc(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Area under the precision-recall curve (average precision). **The primary metric.**

    Returns:
        Average precision in ``[0, 1]``. A random ranking scores about the positive rate, so
        read it against that rather than against 0.5.
    """
    truth, score = _as_arrays(y_true, y_score)
    if truth.sum() == 0:
        raise ValueError("pr_auc is undefined with no positive labels")
    return float(average_precision_score(truth, score))


def roc_auc(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Area under the ROC curve. Secondary — see the module docstring for why."""
    truth, score = _as_arrays(y_true, y_score)
    if truth.sum() in (0, truth.size):
        raise ValueError("roc_auc is undefined when one class is absent")
    return float(roc_auc_score(truth, score))


def brier(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Brier score: mean squared error of the predicted probabilities.

    Lower is better. This is the calibration metric, and it is the one that will look poor on
    the promotion evaluation set for reasons that are **not** a bug — the calibration windows
    carry 2.2-2.7x its base rate (DECISIONS.md D20).
    """
    truth, score = _as_arrays(y_true, y_score)
    if ((score < 0) | (score > 1)).any():
        raise ValueError("brier expects probabilities in [0, 1]")
    return float(brier_score_loss(truth, score))


def recall_at_capacity(
    y_true: ArrayLike, y_score: ArrayLike, capacity: float = DEFAULT_CAPACITY
) -> float:
    """Share of late orders caught when reviewing the top ``capacity`` fraction by score.

    Args:
        y_true: Binary labels.
        y_score: Predicted scores, higher meaning more likely late.
        capacity: Fraction of orders reviewed, in ``(0, 1]``.

    Returns:
        Recall within the reviewed set, in ``[0, 1]``.

    The reviewed count is ``ceil(n * capacity)``, so a non-zero capacity always reviews at
    least one order rather than silently rounding down to none. Ties are broken by taking the
    highest scores first with a **stable** sort, so the result is reproducible; where many
    orders share a score the metric is therefore mildly optimistic, in the same way a real
    queue would be.
    """
    truth, score = _as_arrays(y_true, y_score)
    if not 0.0 < capacity <= 1.0:
        raise ValueError(f"capacity must be in (0, 1], got {capacity}")
    positives = int(truth.sum())
    if positives == 0:
        raise ValueError("recall_at_capacity is undefined with no positive labels")

    reviewed = math.ceil(truth.size * capacity)
    order = np.argsort(-score, kind="stable")[:reviewed]
    return float(truth[order].sum() / positives)


def precision_at_capacity(
    y_true: ArrayLike, y_score: ArrayLike, capacity: float = DEFAULT_CAPACITY
) -> float:
    """Share of the reviewed orders that really were late — the flip side of the staffing
    question, i.e. how much of the reviewers' time is well spent."""
    truth, score = _as_arrays(y_true, y_score)
    if not 0.0 < capacity <= 1.0:
        raise ValueError(f"capacity must be in (0, 1], got {capacity}")
    reviewed = math.ceil(truth.size * capacity)
    order = np.argsort(-score, kind="stable")[:reviewed]
    return float(truth[order].sum() / reviewed)


def positive_rate(y_true: ArrayLike) -> float:
    """Base rate. Read every other number against this, not against 0.5."""
    truth = np.asarray(y_true)
    if truth.size == 0:
        raise ValueError("cannot compute a positive rate on an empty input")
    return float(np.asarray(truth, dtype=float).mean())


def summarize(
    y_true: ArrayLike,
    y_score: ArrayLike,
    capacity: float = DEFAULT_CAPACITY,
    prefix: str = "",
) -> dict[str, float]:
    """Every metric at once, ready to log to MLflow as a metrics dict.

    Args:
        y_true: Binary labels.
        y_score: Predicted probabilities.
        capacity: Review capacity for the recall/precision pair.
        prefix: Prepended to each key, e.g. ``"val_"``.

    Returns:
        ``pr_auc``, ``roc_auc``, ``brier``, ``recall_at_capacity``,
        ``precision_at_capacity``, ``positive_rate``, ``lift_at_capacity`` and ``n``.

    ``lift_at_capacity`` is precision at capacity divided by the base rate: "reviewing this
    top slice finds late orders N times more often than reviewing at random". It is the
    number a non-technical stakeholder understands immediately.
    """
    base = positive_rate(y_true)
    precision = precision_at_capacity(y_true, y_score, capacity)
    metrics = {
        "pr_auc": pr_auc(y_true, y_score),
        "roc_auc": roc_auc(y_true, y_score),
        "brier": brier(y_true, y_score),
        "recall_at_capacity": recall_at_capacity(y_true, y_score, capacity),
        "precision_at_capacity": precision,
        "lift_at_capacity": precision / base if base > 0 else float("nan"),
        "positive_rate": base,
        "capacity": float(capacity),
        "n": float(np.asarray(y_true).size),
    }
    return {f"{prefix}{key}": value for key, value in metrics.items()}


def trivial_baseline(y_true: ArrayLike) -> dict[str, Any]:
    """The never-late baseline, stated explicitly (PLAN.md §4.6).

    Returns:
        Its accuracy and recall. Accuracy near 93% with recall of exactly 0 is the point: it
        is why accuracy is not reported anywhere else in this project.
    """
    truth = np.asarray(y_true, dtype=float)
    return {
        "strategy": "predict never late",
        "accuracy": float((truth == 0).mean()),
        "recall": 0.0,
        "precision": float("nan"),
        "n": int(truth.size),
    }
