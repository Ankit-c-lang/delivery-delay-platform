"""Scoring the promotion evaluation window — **the only module permitted to load it**.

    THIS MODULE IS THE SOLE READER OF 2018-05 -> 2018-08.
    ====================================================
    ``src/splits.py`` enforces that at runtime: :func:`~src.splits.promotion_evaluation_window`
    raises :class:`~src.splits.PromotionEvaluationLockedError` unless the caller has entered
    :func:`~src.splits.unlock_promotion_evaluation` and named itself, and
    ``AUTHORISED_UNLOCKERS`` lists exactly this module and ``src.registry.promote``. A test
    asserts by import inspection that nothing else reaches the window (PLAN.md §13 Phase 9).

    The guard protects the *window definition*, not just the loading code, because the way this
    leaks in practice is not a rogue import — it is a stray
    ``orders[orders.purchase >= "2018-05-01"]`` in a notebook, written by someone who only
    wanted "recent data".

**Call it a promotion evaluation set, never a test set** (§18 A3). Two gate runs read it, so it
has influenced selection — weakly, but it has. With two evaluations the selection bias is
negligible, and the honest move is to name the window accurately rather than to imply it was
never used. Carving out a third period was considered and rejected: it would cost evaluation-set
size to remove a bias two evaluations cannot meaningfully induce.

**Nothing here fits anything.** §13 names re-fitting on the holdout as the phase's first common
mistake. Every model arrives already fitted, with its own preprocessing artifact bundled inside
its pyfunc; this module loads rows, calls ``predict``, and computes metrics. There is no
``fit`` call in this file and there must never be one.

**Each version scores through its own artifact, and that is deliberate.** §8 requires "same rows,
same code path, no re-fitting" — all three hold. What differs is the as-of snapshot each version
bundles: v1 carries 2018-03-01, v2 carries 2018-05-01 (DECISIONS.md D33). Forcing both onto one
snapshot would compare v2's weights against v1's on v1's stale features, which is not what either
would do in production. Part of what a retrain buys *is* fresher aggregates, and a gate that
denied the challenger its own snapshot would measure less than the retrain actually changes.
Neither snapshot reaches into the window: both contain only outcomes resolved before 2018-05-01,
while every evaluation-window order is purchased on or after it.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.metrics import brier, pr_auc, roc_auc
from src.registry.pyfunc_wrapper import RAW_INPUT_COLUMNS

logger = logging.getLogger(__name__)

#: This module's own name, passed to the unlock so the log names the culprit if it is ever misused.
_THIS_MODULE = "src.evaluation.score_holdout"


@dataclass(frozen=True)
class HoldoutScores:
    """What one model did on the evaluation window.

    Attributes:
        model_version: Registry version scored.
        n_rows: Rows scored. Identical across models by construction.
        base_rate: Positive rate of those rows — the number every other metric must be read
            against (DECISIONS.md D20).
        pr_auc: Average precision.
        lift: ``pr_auc / base_rate``. **The comparable figure across windows**, because PR-AUC is
            bounded below by the base rate, and this window's is 4.40% against the calibration
            window's ~9.75%.
        roc_auc: Reported as secondary; flattering at a 4% positive rate (§4.6).
        brier: Calibration quality.
        threshold: The operating point the model carries.
        recall: Share of late orders caught at that threshold.
        precision: Share of flagged orders that were late.
        flag_rate: Share of all orders flagged.
        p95_latency_ms: p95 single-prediction latency, measured through this model's own pyfunc.
        warnings_seen: Unseen-category reports encountered while scoring, as a count per feature.
    """

    model_version: str
    n_rows: int
    base_rate: float
    pr_auc: float
    lift: float
    roc_auc: float
    brier: float
    threshold: float
    recall: float
    precision: float
    flag_rate: float
    p95_latency_ms: float
    warnings_seen: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Flat form, for logging as MLflow metrics or writing into a report."""
        return {
            "model_version": self.model_version,
            "n_rows": self.n_rows,
            "base_rate": round(self.base_rate, 5),
            "pr_auc": round(self.pr_auc, 5),
            "lift": round(self.lift, 3),
            "roc_auc": round(self.roc_auc, 5),
            "brier": round(self.brier, 6),
            "threshold": round(self.threshold, 5),
            "recall": round(self.recall, 5),
            "precision": round(self.precision, 5),
            "flag_rate": round(self.flag_rate, 5),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
        }


def load_holdout() -> tuple[pd.DataFrame, np.ndarray]:
    """Load the evaluation window's raw records and labels.

    Returns:
        ``(records, labels)``. ``records`` carries exactly
        :data:`~src.registry.pyfunc_wrapper.RAW_INPUT_COLUMNS`, so it is what a client would send;
        ``labels`` is the ``is_late`` outcome.

    Raises:
        PromotionEvaluationLockedError: Propagated from ``src.splits`` if the lock is held
            elsewhere.

    The rows are ordered by ``(order_purchase_timestamp, order_id)``, the same total order the
    training matrix uses (DECISIONS.md D35), so two runs score the rows in the same sequence and a
    latency percentile is measured over the same work.
    """
    from src.db import get_engine
    from src.splits import promotion_evaluation_window, unlock_promotion_evaluation

    with unlock_promotion_evaluation(_THIS_MODULE):
        window = promotion_evaluation_window()

    columns = ", ".join(RAW_INPUT_COLUMNS)
    query = f"""
        SELECT o.order_id, {columns}, l.is_late
        FROM features.orders_analytical o
        JOIN features.order_outcomes l USING (order_id)
        WHERE o.order_purchase_timestamp >= '{window.start}'
          AND o.order_purchase_timestamp < '{window.end_exclusive}'
        ORDER BY o.order_purchase_timestamp, o.order_id
    """
    frame = pd.read_sql(query, get_engine())
    if frame.empty:
        raise ValueError(
            f"the evaluation window {window} returned no rows; the ETL has not run or the "
            "window has drifted out of the data"
        )

    labels = frame["is_late"].to_numpy().astype(int)
    records = frame.loc[:, list(RAW_INPUT_COLUMNS)].copy()
    logger.info(
        "holdout: %d rows, %s .. %s, base rate %.4f",
        len(records),
        window.start,
        window.end,
        labels.mean(),
    )
    return records, labels


def measure_p95_latency(model: Any, records: pd.DataFrame, samples: int, warmup: int) -> float:
    """p95 single-prediction latency in milliseconds, through the model's own pyfunc.

    Args:
        model: An ``mlflow.pyfunc`` model, or anything with ``predict``.
        records: Rows to draw single predictions from.
        samples: Timed calls.
        warmup: Untimed calls first.

    Returns:
        The 95th percentile in milliseconds.

    **Single predictions, not a batch.** §8's budget is about the checkout path, where a request
    carries one order. Measuring a batch and dividing would report roughly 0.6 ms per record and
    hide the ~50 ms of fixed overhead every real request pays (Phase 8's three-way measurement).
    """
    if warmup + samples > len(records):
        samples = max(len(records) - warmup, 1)
    timings: list[float] = []
    for index in range(warmup + samples):
        row = records.iloc[[index % len(records)]]
        started = time.perf_counter()
        model.predict(row)
        elapsed = (time.perf_counter() - started) * 1000.0
        if index >= warmup:
            timings.append(elapsed)
    timings.sort()
    return timings[max(int(len(timings) * 0.95) - 1, 0)]


def score_model(
    model: Any,
    model_version: str,
    records: pd.DataFrame,
    labels: np.ndarray,
    *,
    latency_samples: int = 60,
    latency_warmup: int = 5,
) -> HoldoutScores:
    """Score one already-fitted model on the holdout.

    Args:
        model: A loaded pyfunc. **Already fitted** — nothing here fits.
        model_version: Registry version, for the record.
        records: Raw holdout records.
        labels: Their outcomes.
        latency_samples: Timed single predictions.
        latency_warmup: Untimed calls first.

    Returns:
        A :class:`HoldoutScores`.

    Raises:
        ValueError: If the model returns a row count that does not match the input, which would
            mean a silent reindexing somewhere and make every metric meaningless.
    """
    predictions = model.predict(records)
    if len(predictions) != len(records):
        raise ValueError(
            f"model {model_version} returned {len(predictions)} rows for {len(records)} inputs"
        )

    probabilities = np.asarray(predictions["probability"], dtype=float)
    threshold = float(predictions["threshold"].iloc[0])
    flagged = probabilities >= threshold
    positives = labels.astype(bool)

    true_positives = int((flagged & positives).sum())
    base_rate = float(labels.mean())
    score = pr_auc(labels, probabilities)

    return HoldoutScores(
        model_version=model_version,
        n_rows=len(records),
        base_rate=base_rate,
        pr_auc=score,
        # Guarded: a window with no positives would make lift meaningless rather than infinite.
        lift=float(score / base_rate) if base_rate > 0 else float("nan"),
        roc_auc=roc_auc(labels, probabilities),
        brier=brier(labels, probabilities),
        threshold=threshold,
        recall=true_positives / max(int(positives.sum()), 1),
        precision=true_positives / max(int(flagged.sum()), 1),
        flag_rate=float(flagged.mean()),
        p95_latency_ms=measure_p95_latency(model, records, latency_samples, latency_warmup),
    )


def describe(scores: HoldoutScores) -> str:
    """One line for a log or a report, with the base rate attached to the PR-AUC.

    The base rate travels with the number on purpose. A PR-AUC of 0.09 reads as a broken model
    until you know the window's positive rate is 4.40%, at which point it is a 2x lift.
    """
    return (
        f"v{scores.model_version}: PR-AUC {scores.pr_auc:.5f} "
        f"({scores.lift:.2f}x the {scores.base_rate:.2%} base rate), "
        f"Brier {scores.brier:.6f}, recall {scores.recall:.1%} at a {scores.flag_rate:.1%} "
        f"flag rate, p95 {scores.p95_latency_ms:.1f} ms"
    )
