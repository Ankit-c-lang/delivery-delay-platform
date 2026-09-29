"""Blending, calibration and the cost-derived threshold (PLAN.md §6.5, §4.7, §13 Phase 5).

Three post-processing steps, each fitted on the **calibration window** — never on the fit
window, where the models were trained, and never on the promotion evaluation window, which is
locked (`src/splits.py`).

**The window is split, because the phase's own warning requires it.** §13 lists "calibrating on
data used to fit the blend" as a common mistake, while asking for both the blend weights and the
calibrator to be fitted on validation. Those instructions collide, so the calibration window is
divided in two: weights are fitted on one half, the calibrator on the other. See
:func:`split_calibration_window` for why the split is interleaved rather than temporal —
briefly, a temporal split of v1's window leaves halves at 5.74% and 13.77% positives.

**Do not inflate the ensemble result.** §13 is explicit: if the gain is under 0.005 PR-AUC, say
plainly that the single model should ship. Three GBDTs on identical tabular features produce
highly correlated predictions, so a small gain is the expected outcome, not a failure. The
threshold lives in ``configs/base.yaml`` so the decision is a rule rather than a mood.

**The cost ratio is an assumption.** §4.7 is emphatic: never present the cost figures as if they
were measured. Every function here that touches cost takes the ratio as an argument and reports
it alongside the result.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss

from src.evaluation.metrics import brier, pr_auc

logger = logging.getLogger(__name__)

#: Probabilities are clipped away from 0 and 1 before log-loss, which is otherwise infinite.
EPSILON = 1e-7

#: Slack on the selection band's boundary comparison, absorbing float64 subtraction error only.
#: Twelve orders of magnitude below the band itself, so it can never widen the rule in practice.
BAND_EPSILON = 1e-12


@dataclass(frozen=True)
class BlendResult:
    """Blend weights and an honest comparison against the best single model.

    Attributes:
        weights: Non-negative, summing to 1, keyed by model name.
        blend_pr_auc: PR-AUC of the blended probabilities on the weight-fitting half.
        single_pr_auc: PR-AUC of each model on the same rows.
        best_single: Name of the best single model.
        delta: ``blend_pr_auc`` minus the best single model's PR-AUC. **May be negative** — a
            blend fitted by log-loss is not guaranteed to improve PR-AUC.
        min_gain: The gain required to prefer the blend.
        ship_ensemble: Whether ``delta`` clears ``min_gain``.
    """

    weights: dict[str, float]
    blend_pr_auc: float
    single_pr_auc: dict[str, float]
    best_single: str
    delta: float
    min_gain: float
    ship_ensemble: bool

    def summary(self) -> dict[str, Any]:
        return {
            "weights": {k: round(v, 4) for k, v in self.weights.items()},
            "blend_pr_auc": round(self.blend_pr_auc, 5),
            "single_pr_auc": {k: round(v, 5) for k, v in self.single_pr_auc.items()},
            "best_single": self.best_single,
            "delta": round(self.delta, 5),
            "min_gain": self.min_gain,
            "ship_ensemble": self.ship_ensemble,
        }


@dataclass(frozen=True)
class CalibrationResult:
    """What isotonic calibration did to the probabilities.

    Attributes:
        calibrator: The fitted isotonic model.
        brier_before: Brier score of the raw probabilities on the calibration half.
        brier_after: Brier score after calibration, on the same rows.
        improvement: ``brier_before - brier_after``; positive means better.
        n_fit: Rows the calibrator was fitted on.
        positives_fit: Positive labels among them — isotonic regression is non-parametric and
            gets unstable when this is small, so it is reported rather than assumed adequate.
    """

    calibrator: IsotonicRegression
    brier_before: float
    brier_after: float
    improvement: float
    n_fit: int
    positives_fit: int

    def summary(self) -> dict[str, Any]:
        return {
            "brier_before": round(self.brier_before, 6),
            "brier_after": round(self.brier_after, 6),
            "improvement": round(self.improvement, 6),
            "n_fit": self.n_fit,
            "positives_fit": self.positives_fit,
        }


@dataclass(frozen=True)
class ThresholdResult:
    """The cost-minimising operating point under a stated cost assumption.

    Attributes:
        threshold: Probability above which an order is flagged.
        fn_fp_ratio: The **assumed** cost of a false negative relative to a false positive.
        expected_cost_per_order: Cost at ``threshold``, in false-positive units.
        precision: Of the flagged orders, the share really late.
        recall: Of the late orders, the share flagged.
        flag_rate: Share of all orders flagged — the operational load.
        confusion: ``tp``, ``fp``, ``fn``, ``tn`` at ``threshold``.
        sweep: ``(threshold, expected_cost)`` pairs evaluated, for plotting.
    """

    threshold: float
    fn_fp_ratio: float
    expected_cost_per_order: float
    precision: float
    recall: float
    flag_rate: float
    confusion: dict[str, int]
    sweep: list[tuple[float, float]] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "threshold": round(self.threshold, 4),
            "fn_fp_ratio": self.fn_fp_ratio,
            "expected_cost_per_order": round(self.expected_cost_per_order, 5),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "flag_rate": round(self.flag_rate, 4),
            **{f"n_{k}": v for k, v in self.confusion.items()},
        }

    def sentence(self) -> str:
        """The §4.7 phrasing, with the assumption stated in the sentence itself."""
        return (
            f"Under an assumed {self.fn_fp_ratio:g}:1 FN:FP cost ratio, the optimal threshold "
            f"is {self.threshold:.2f}, catching {self.recall:.1%} of late orders at a "
            f"{self.flag_rate:.1%} flag rate (precision {self.precision:.1%})."
        )


def split_calibration_window(
    timestamps: pd.Series, how: str = "interleaved"
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Divide the calibration window into a weight-fitting half and a calibrator-fitting half.

    Args:
        timestamps: Purchase timestamps of the calibration rows.
        how: ``interleaved`` or ``temporal``.

    Returns:
        Two boolean masks over ``timestamps``, disjoint and covering every row.

    Raises:
        ValueError: On an unknown strategy.

    **Why interleaved is the default.** A temporal split is the instinctive choice — it mirrors
    production, where you calibrate on the past and apply forward. Measured on v1's window it
    is a bad trade: the halves come out at **5.74% and 13.77% positives**, so the calibrator
    would be fitted at 2.4x the base rate of the data the weights were fitted on, and 3.1x the
    evaluation window's 4.40%. That is DECISIONS.md D20 reappearing inside a two-month window.

    Interleaving by time order keeps both halves at the window's own base rate. It does not
    reintroduce the leak CLAUDE.md invariant 5 guards against: that invariant is about
    cross-validation for *model* fitting, where random folds let future outcomes into as-of
    features. Nothing is recomputed here — the features are already fixed, both halves sit
    inside the same two months, and what is being fitted is a set of blend weights and a
    one-dimensional monotone function. ``temporal`` stays available so the comparison can be
    measured rather than argued.
    """
    order = np.argsort(pd.to_datetime(timestamps).to_numpy(), kind="stable")
    n = len(order)
    first = np.zeros(n, dtype=bool)

    if how == "interleaved":
        first[order[::2]] = True
    elif how == "temporal":
        first[order[: n // 2]] = True
    else:
        raise ValueError(f"unknown calibration split {how!r}; expected interleaved or temporal")
    return first, ~first


def fit_blend_weights(
    probabilities: dict[str, NDArray[np.float64]], y: NDArray[np.int_]
) -> dict[str, float]:
    """Fit non-negative weights summing to 1 by minimising log-loss.

    Args:
        probabilities: Predicted positive-class probability per model, all the same length.
        y: Binary labels.

    Returns:
        Weights keyed by model name, non-negative and summing to 1.

    Raises:
        ValueError: If the models disagree on length, or there is only one model.

    Log-loss rather than PR-AUC because it is smooth and differentiable, so SLSQP can optimise
    it under the simplex constraint; PR-AUC is a step function of the ranking and would need a
    different optimiser. The consequence is stated plainly in :class:`BlendResult`: weights
    fitted for log-loss can make PR-AUC *worse*, and the delta is reported either way.

    §6.5 also explains why it is fitted here rather than on the training rows — on train the
    weights collapse onto whichever model overfits hardest.
    """
    names = sorted(probabilities)
    if len(names) < 2:
        raise ValueError(f"need at least two models to blend, got {names}")
    lengths = {len(probabilities[name]) for name in names}
    if len(lengths) != 1 or lengths != {len(y)}:
        raise ValueError(f"model probabilities and labels disagree on length: {lengths}, {len(y)}")

    stacked = np.column_stack([np.asarray(probabilities[name], dtype=float) for name in names])

    def objective(weights: NDArray[np.float64]) -> float:
        blended = np.clip(stacked @ weights, EPSILON, 1.0 - EPSILON)
        return float(log_loss(y, blended))

    start = np.full(len(names), 1.0 / len(names))
    result = minimize(
        objective,
        start,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(names),
        constraints=[{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}],
        options={"maxiter": 300, "ftol": 1e-10},
    )
    if not result.success:
        logger.warning("blend weight optimisation did not converge: %s", result.message)

    # Clean up the tiny negatives and sum drift SLSQP leaves behind, so the invariant the tests
    # assert holds exactly rather than to within a tolerance.
    weights = np.clip(result.x, 0.0, None)
    total = weights.sum()
    weights = weights / total if total > 0 else start
    return dict(zip(names, (float(w) for w in weights), strict=True))


def blend_probabilities(
    probabilities: dict[str, NDArray[np.float64]], weights: dict[str, float]
) -> NDArray[np.float64]:
    """Weighted average of per-model probabilities.

    Raises:
        KeyError: If a weight has no matching model, or vice versa. Silently ignoring a
            mismatch would produce a blend that is not the one whose weights were fitted.
    """
    if set(probabilities) != set(weights):
        raise KeyError(f"models {sorted(probabilities)} do not match weights {sorted(weights)}")
    stacked = np.column_stack([probabilities[name] for name in sorted(probabilities)])
    vector = np.array([weights[name] for name in sorted(weights)], dtype=float)
    return np.clip(stacked @ vector, 0.0, 1.0)


def evaluate_blend(
    probabilities: dict[str, NDArray[np.float64]],
    y: NDArray[np.int_],
    weights: dict[str, float],
    min_gain: float,
) -> BlendResult:
    """Compare the blend against the best single model on the same rows."""
    blended = blend_probabilities(probabilities, weights)
    singles = {name: pr_auc(y, probs) for name, probs in probabilities.items()}
    best_single = max(singles, key=lambda name: singles[name])
    blend_score = pr_auc(y, blended)
    delta = blend_score - singles[best_single]
    return BlendResult(
        weights=weights,
        blend_pr_auc=blend_score,
        single_pr_auc=singles,
        best_single=best_single,
        delta=delta,
        min_gain=min_gain,
        ship_ensemble=delta >= min_gain,
    )


@dataclass(frozen=True)
class ModelSelection:
    """Which single model ships, and why it rather than the top of the PR-AUC table.

    Attributes:
        chosen: The model that ships.
        best_by_pr_auc: The highest-scoring model. Differs from ``chosen`` only when the rule fired.
        scores: PR-AUC per model, on the rows the comparison used.
        costs_mb: Installed size of each model's library, from ``configs/base.yaml``.
        band: How far below the best a model may score and still be considered equivalent.
        contenders: Models inside the band, cheapest first.
        rule_changed_the_outcome: Whether the tie-break moved the decision.
    """

    chosen: str
    best_by_pr_auc: str
    scores: dict[str, float]
    costs_mb: dict[str, int]
    band: float
    contenders: tuple[str, ...]
    rule_changed_the_outcome: bool

    def sentence(self) -> str:
        """One line for the log and the report."""
        if not self.rule_changed_the_outcome:
            others = [name for name in self.contenders if name != self.chosen]
            if others:
                # Accuracy matters here: "nothing else was close" and "others were close
                # but this one is also cheapest" are different facts; do not blur them.
                return (
                    f"ship {self.chosen}: highest PR-AUC ({self.scores[self.chosen]:.5f}) and also "
                    f"the cheapest of the {len(self.contenders)} models inside the "
                    f"{self.band:.3f} equivalence band"
                )
            return (
                f"ship {self.chosen}: highest PR-AUC ({self.scores[self.chosen]:.5f}), and no "
                f"other model is within the {self.band:.3f} equivalence band"
            )
        gap = self.scores[self.best_by_pr_auc] - self.scores[self.chosen]
        saved = self.costs_mb[self.best_by_pr_auc] - self.costs_mb[self.chosen]
        return (
            f"ship {self.chosen} ({self.scores[self.chosen]:.5f}) over {self.best_by_pr_auc} "
            f"({self.scores[self.best_by_pr_auc]:.5f}): the {gap:.5f} gap is inside the "
            f"{self.band:.3f} equivalence band, and {self.chosen} costs "
            f"{self.costs_mb[self.chosen]} MB against {self.costs_mb[self.best_by_pr_auc]} MB "
            f"— {saved} MB less in the serving image"
        )


def select_single_model(
    scores: dict[str, float], band: float, costs_mb: dict[str, int]
) -> ModelSelection:
    """Pick the single model to ship: best PR-AUC, then cheapest inside the equivalence band.

    Args:
        scores: PR-AUC per model, all measured on the same rows.
        band: Width of the equivalence band. A model scoring within ``band`` of the best is
            treated as indistinguishable from it.
        costs_mb: Installed size of each model's library in MB.

    Returns:
        A :class:`ModelSelection` carrying the choice and the reasoning.

    Raises:
        ValueError: If ``scores`` is empty, ``band`` is negative, or a scored model has no cost
            entry. A missing cost is refused rather than defaulted: a model whose cost is unknown
            would silently win every tie by looking free.

    **Why a band at all.** DECISIONS.md D35 measured this pipeline's reproducibility floor: thread
    count alone moved CatBoost's calibration-window PR-AUC by 0.005 on identical data and
    identical hyperparameters. A gap smaller than that is not a result, it is the environment — so
    declaring a winner inside the band claims a difference already shown to be irreproducible.

    **Why serving cost is the tie-break.** The shipped model's library has to be installed in the
    API image, measured at 10 MB (LightGBM) to 269 MB (CatBoost). Inside the band that is the only
    difference between the candidates that anyone can observe. Ordering by cost keeps the choice a
    rule rather than a preference — the same reason ``min_ensemble_gain_pr_auc`` is a number in
    ``configs/base.yaml`` instead of a judgement made per run.

    The band never overrides a real difference: a model more than ``band`` below the best is not a
    contender at any price.
    """
    if not scores:
        raise ValueError("no model scores to select from")
    if band < 0:
        raise ValueError(f"selection band must be non-negative, got {band}")
    missing = sorted(set(scores) - set(costs_mb))
    if missing:
        raise ValueError(
            f"no serving cost configured for {missing}; add it to decision.serving_cost_mb in "
            "configs/base.yaml. Defaulting would let an unmeasured model win every tie by "
            "appearing to cost nothing."
        )

    best = max(scores, key=lambda name: scores[name])
    # BAND_EPSILON, not a bare <=. Subtracting two float64 PR-AUCs that differ by exactly the band
    # can land a hair above it — 0.2200 - 0.2150 evaluates to 0.005000000000000004 — so a model
    # sitting precisely on the documented boundary would be excluded by float representation alone.
    # A rule about ignoring differences too small to reproduce should not itself turn on one.
    within = [name for name in scores if scores[best] - scores[name] <= band + BAND_EPSILON]
    # Cheapest first; ties on cost broken by PR-AUC, then by name so the result is deterministic
    # whatever order the dict arrived in.
    contenders = sorted(within, key=lambda name: (costs_mb[name], -scores[name], name))
    chosen = contenders[0]
    return ModelSelection(
        chosen=chosen,
        best_by_pr_auc=best,
        scores=dict(scores),
        costs_mb={name: costs_mb[name] for name in scores},
        band=float(band),
        contenders=tuple(contenders),
        rule_changed_the_outcome=chosen != best,
    )


def fit_calibrator(probabilities: NDArray[np.float64], y: NDArray[np.int_]) -> CalibrationResult:
    """Fit isotonic regression and report Brier before and after, on the fitting rows.

    ``out_of_bounds="clip"`` so a serving probability outside the fitted range maps to the
    nearest endpoint instead of returning NaN — an uncalibrated NaN reaching the API would be
    a hard failure at the worst moment.

    The Brier improvement here is **in-sample** and will flatter the calibrator; the honest
    figure comes from applying it to held-out rows, which :func:`build_decision` does and the
    tests require.
    """
    truth = np.asarray(y, dtype=int)
    raw = np.asarray(probabilities, dtype=float)
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(raw, truth)
    calibrated = calibrator.predict(raw)
    before, after = brier(truth, raw), brier(truth, calibrated)
    return CalibrationResult(
        calibrator=calibrator,
        brier_before=before,
        brier_after=after,
        improvement=before - after,
        n_fit=int(truth.size),
        positives_fit=int(truth.sum()),
    )


def sweep_threshold(
    probabilities: NDArray[np.float64],
    y: NDArray[np.int_],
    fn_fp_ratio: float = 5.0,
    grid: NDArray[np.float64] | None = None,
) -> ThresholdResult:
    """Find the cost-minimising threshold under an assumed FN:FP cost ratio.

    Args:
        probabilities: Calibrated probabilities.
        y: Binary labels.
        fn_fp_ratio: Assumed cost of a false negative relative to a false positive. **An
            assumption, not a measurement** (§4.7).
        grid: Candidate thresholds. Defaults to every distinct predicted probability, which is
            exact for a finite sample — the cost function only changes where a prediction
            crosses the threshold, so no finer grid can find a better point.

    Returns:
        The argmin and its operating characteristics.

    Raises:
        ValueError: On a non-positive ratio, or labels of only one class.
    """
    truth = np.asarray(y, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    if fn_fp_ratio <= 0:
        raise ValueError(f"fn_fp_ratio must be positive, got {fn_fp_ratio}")
    if truth.sum() in (0, truth.size):
        raise ValueError("threshold sweep needs both classes present")

    if grid is None:
        grid = np.unique(scores)
    grid = np.asarray(grid, dtype=float)

    positives = truth == 1
    sweep: list[tuple[float, float]] = []
    for threshold in grid:
        flagged = scores >= threshold
        false_negatives = int(np.sum(positives & ~flagged))
        false_positives = int(np.sum(~positives & flagged))
        cost = (fn_fp_ratio * false_negatives + false_positives) / truth.size
        sweep.append((float(threshold), float(cost)))

    best_index = int(np.argmin([cost for _, cost in sweep]))
    threshold = sweep[best_index][0]

    flagged = scores >= threshold
    tp = int(np.sum(positives & flagged))
    fp = int(np.sum(~positives & flagged))
    fn = int(np.sum(positives & ~flagged))
    tn = int(np.sum(~positives & ~flagged))
    return ThresholdResult(
        threshold=threshold,
        fn_fp_ratio=float(fn_fp_ratio),
        expected_cost_per_order=sweep[best_index][1],
        precision=tp / (tp + fp) if (tp + fp) else 0.0,
        recall=tp / (tp + fn) if (tp + fn) else 0.0,
        flag_rate=float(flagged.mean()),
        confusion={"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        sweep=sweep,
    )


# ==================================================================================
# Phase 5 driver: refit, blend, calibrate, threshold, business impact, report.
# ==================================================================================

REPORT_PATH = Path("reports/model_selection.md")
DECISION_ARTIFACT_PATH = Path("artifacts/decision.joblib")

REVIEW_IMPACT_SQL = """
WITH per_order AS (
    -- Reviews are not unique per order (review_id has 814 duplicates and some orders carry
    -- more than one review), so collapse to one mean score per order before grouping.
    SELECT order_id, avg(review_score)::double precision AS review_score
    FROM raw.order_reviews
    GROUP BY order_id
)
SELECT o.is_late,
       count(*)                                   AS orders,
       count(p.review_score)                      AS with_review,
       avg(p.review_score)                        AS mean_review_score,
       avg((p.review_score <= 2)::int)            AS share_one_or_two
FROM features.order_outcomes o
LEFT JOIN per_order p USING (order_id)
GROUP BY o.is_late
ORDER BY o.is_late
"""


def review_impact(conn) -> dict[str, Any]:
    """Mean review score for on-time against late orders (PLAN.md §4.7).

    **Analysis only.** Reviews are written after delivery, so every review column is on the
    §4.2 denylist and none may enter the feature matrix. This function asserts that before
    returning, so the analysis cannot quietly become a feature source.

    Returns:
        Per-class counts and mean scores, plus the gap that converts recall into "reviews
        protected".

    Raises:
        RuntimeError: If any review column has reached the feature matrix.
    """
    from sqlalchemy import text

    from src.etl.schema import DENYLIST
    from src.features.build import FEATURE_NAMES

    leaked = {name for name in FEATURE_NAMES if name in DENYLIST or name.startswith("review")}
    if leaked:
        raise RuntimeError(f"review data reached the feature matrix: {sorted(leaked)}")

    rows = pd.read_sql(text(REVIEW_IMPACT_SQL), conn)
    by_class = {bool(row.is_late): row for row in rows.itertuples()}
    on_time, late = by_class[False], by_class[True]
    return {
        "on_time_mean": float(on_time.mean_review_score),
        "late_mean": float(late.mean_review_score),
        "gap": float(on_time.mean_review_score - late.mean_review_score),
        "on_time_orders": int(on_time.orders),
        "late_orders": int(late.orders),
        "on_time_share_1_or_2": float(on_time.share_one_or_two),
        "late_share_1_or_2": float(late.share_one_or_two),
        "reviews_missing": int(on_time.orders - on_time.with_review)
        + int(late.orders - late.with_review),
    }


def audit_feature(
    feature: str,
    model_names: list[str],
    X_fit: pd.DataFrame,
    y_fit: NDArray[np.int_],
    X_cal: pd.DataFrame,
    y_cal: NDArray[np.int_],
    tuned: dict[str, Any],
    model_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Refit each model with and without one feature and compare on the calibration window.

    §5 asks for a feature audit after the first model, to "drop 1-3 dead or redundant
    features". A leave-one-out refit is the honest form of that audit: permutation importance
    on a trained model tells you what the model *used*, whereas this tells you what the model
    is *worth* without it, which is the question when deciding to remove it.

    Also reports whether the feature's values at scoring time were ever seen in training. A
    feature with no overlap cannot have been learned — a tree routes unseen values to whichever
    branch the last split happened to leave open, so any apparent benefit is an accident of
    geometry rather than a learned effect.
    """
    from src.training.tune import fit_final_model, predict_positive

    fit_values = set(np.unique(X_fit[feature]).tolist())
    cal_values = set(np.unique(X_cal[feature]).tolist())
    audit: dict[str, Any] = {
        "feature": feature,
        "fit_values": sorted(fit_values),
        "calibration_values": sorted(cal_values),
        "overlap": sorted(fit_values & cal_values),
        "models": {},
    }
    dropped_fit, dropped_cal = X_fit.drop(columns=[feature]), X_cal.drop(columns=[feature])

    for name in model_names:
        params, fixed = tuned["models"][name]["best_params"], model_cfg["models"][name]["fixed"]
        with_it = predict_positive(name, fit_final_model(name, params, fixed, X_fit, y_fit), X_cal)
        without = predict_positive(
            name, fit_final_model(name, params, fixed, dropped_fit, y_fit), dropped_cal
        )
        audit["models"][name] = {
            "pr_auc_with": pr_auc(y_cal, with_it),
            "pr_auc_without": pr_auc(y_cal, without),
            "pr_auc_delta": pr_auc(y_cal, without) - pr_auc(y_cal, with_it),
            "brier_with": brier(y_cal, with_it),
            "brier_without": brier(y_cal, without),
        }
        logger.info(
            "audit %-9s without %s: PR-AUC %+.5f, Brier %+.5f",
            name,
            feature,
            audit["models"][name]["pr_auc_delta"],
            audit["models"][name]["brier_without"] - audit["models"][name]["brier_with"],
        )
    return audit


def build_decision(version: str = "v1", run_shap: bool = True) -> dict[str, Any]:
    """The whole Phase 5 pipeline.

    Refits each tuned model on the fit window, predicts on the calibration window, fits blend
    weights on one half and the calibrator on the other, sweeps the threshold, measures review
    impact and runs SHAP.

    Returns:
        Everything the report needs, plus the chosen scorer's name and the artifact updated
        with the calibrator and threshold.
    """
    import yaml

    from src.db import get_engine
    from src.features.build import build_matrix, window_slice
    from src.splits import version_splits
    from src.training.tune import (
        fit_final_model,
        load_model_config,
        load_tuned_params,
        predict_positive,
    )

    base = yaml.safe_load(Path("configs/base.yaml").read_text(encoding="utf-8"))
    decision_cfg = base["decision"]
    model_cfg = load_model_config()
    tuned = load_tuned_params()

    built = build_matrix(version)
    orders, matrix, labels, artifact = (
        built["orders"],
        built["matrix"],
        built["labels"],
        built["artifact"],
    )
    splits = version_splits(version)

    # window_slice, not an inline argsort: the ordering rule lives in exactly one place because
    # getting it wrong is invisible — see DECISIONS.md D35.
    X_fit, y_fit, _ = window_slice(splits.fit, orders, matrix, labels)
    X_cal, y_cal, stamps_cal = window_slice(splits.calibrate, orders, matrix, labels)
    logger.info(
        "fit %d rows (%.4f late) | calibrate %d rows (%.4f late)",
        len(X_fit),
        y_fit.mean(),
        len(X_cal),
        y_cal.mean(),
    )

    models: dict[str, Any] = {}
    probabilities: dict[str, NDArray[np.float64]] = {}
    for name, entry in tuned["models"].items():
        models[name] = fit_final_model(
            name, entry["best_params"], model_cfg["models"][name]["fixed"], X_fit, y_fit
        )
        probabilities[name] = predict_positive(name, models[name], X_cal)
        logger.info("%-9s refit on the fit window, scored the calibration window", name)

    weight_half, calib_half = split_calibration_window(
        stamps_cal, decision_cfg["calibration_split"]
    )
    logger.info(
        "calibration window split %s: weights on %d rows (%.4f late), calibrator on %d (%.4f)",
        decision_cfg["calibration_split"],
        weight_half.sum(),
        y_cal[weight_half].mean(),
        calib_half.sum(),
        y_cal[calib_half].mean(),
    )

    weights = fit_blend_weights(
        {name: p[weight_half] for name, p in probabilities.items()}, y_cal[weight_half]
    )
    blend = evaluate_blend(
        {name: p[weight_half] for name, p in probabilities.items()},
        y_cal[weight_half],
        weights,
        decision_cfg["min_ensemble_gain_pr_auc"],
    )
    logger.info(
        "blend PR-AUC %.5f vs best single %s %.5f -> delta %+.5f (ship=%s)",
        blend.blend_pr_auc,
        blend.best_single,
        blend.single_pr_auc[blend.best_single],
        blend.delta,
        blend.ship_ensemble,
    )

    # The blend-vs-single comparison above stays on pure PR-AUC — that is the honest answer to
    # "did blending help". Which SINGLE model ships is a separate question, and the equivalence
    # band answers it (DECISIONS.md D41).
    selection = None
    if blend.ship_ensemble:
        chosen = "blend"
        scores_cal = blend_probabilities(probabilities, weights)
    else:
        selection = select_single_model(
            blend.single_pr_auc,
            decision_cfg["selection_band_pr_auc"],
            decision_cfg["serving_cost_mb"],
        )
        chosen = selection.chosen
        logger.info("selection: %s", selection.sentence())
        scores_cal = probabilities[chosen]

    calibration = fit_calibrator(scores_cal[calib_half], y_cal[calib_half])
    # The honest figure: the calibrator fitted on one half, applied to the other.
    held_out_before = brier(y_cal[weight_half], scores_cal[weight_half])
    held_out_after = brier(
        y_cal[weight_half], calibration.calibrator.predict(scores_cal[weight_half])
    )

    calibrated = calibration.calibrator.predict(scores_cal[calib_half])
    threshold = sweep_threshold(calibrated, y_cal[calib_half], decision_cfg["fn_fp_cost_ratio"])
    logger.info(threshold.sentence())

    with get_engine().connect() as conn:
        impact = review_impact(conn)
    logger.info(
        "review impact: on-time %.3f vs late %.3f (gap %.3f)",
        impact["on_time_mean"],
        impact["late_mean"],
        impact["gap"],
    )

    explanation = None
    if run_shap:
        from src.evaluation.explain import explain_model, sample_rows

        shap_model = models[blend.best_single]
        sample = sample_rows(X_cal)
        if blend.best_single == "catboost":
            sample = sample.copy()
            for column in sample.columns:
                if isinstance(sample[column].dtype, pd.CategoricalDtype):
                    sample[column] = sample[column].astype(str)
        explanation = explain_model(shap_model, sample, blend.best_single)

    audit = None
    if explanation is not None:
        top_feature = str(explanation.mean_abs_shap.index[0])
        audit = audit_feature(
            top_feature, sorted(models), X_fit, y_fit, X_cal, y_cal, tuned, model_cfg
        )

    decided = artifact.with_decision(
        calibrator=calibration.calibrator, threshold=threshold.threshold
    )
    decided.save(DECISION_ARTIFACT_PATH)

    return {
        "version": version,
        "blend": blend,
        "calibration": calibration,
        "held_out_brier": {"before": held_out_before, "after": held_out_after},
        "threshold": threshold,
        "impact": impact,
        "explanation": explanation,
        "chosen": chosen,
        "audit": audit,
        "split": decision_cfg["calibration_split"],
        "n_weight_half": int(weight_half.sum()),
        "n_calib_half": int(calib_half.sum()),
        "rate_weight_half": float(y_cal[weight_half].mean()),
        "rate_calib_half": float(y_cal[calib_half].mean()),
    }


def write_report(result: dict[str, Any], path: Path = REPORT_PATH) -> None:
    """Write `reports/model_selection.md` — the ensemble delta and an explicit ship decision."""
    blend: BlendResult = result["blend"]
    calibration: CalibrationResult = result["calibration"]
    threshold: ThresholdResult = result["threshold"]
    impact = result["impact"]
    held = result["held_out_brier"]

    verdict = (
        f"**Ship the ensemble.** The blend gained {blend.delta:+.4f} PR-AUC over "
        f"{blend.best_single}, clearing the {blend.min_gain} bar."
        if blend.ship_ensemble
        else (
            f"**Ship the single model: `{blend.best_single}`.** The blend gained "
            f"{blend.delta:+.4f} PR-AUC over it, which does not clear the "
            f"{blend.min_gain} bar set in `configs/base.yaml`."
        )
    )

    lines = [
        "# Model selection: blend, calibration, threshold",
        "",
        "Generated by `python -m src.training.ensemble`. Every quantity here is fitted on the",
        f"**calibration window** of `{result['version']}` — never the fit window the models were",
        "trained on, and never the promotion evaluation window, which is locked (§18 A3).",
        "",
        "## The decision",
        "",
        verdict,
        "",
        "| | PR-AUC |",
        "|---|---:|",
    ]
    for name, score in sorted(blend.single_pr_auc.items(), key=lambda kv: -kv[1]):
        marker = " **(best single)**" if name == blend.best_single else ""
        lines.append(f"| `{name}`{marker} | {score:.5f} |")
    lines += [
        f"| **blend** | **{blend.blend_pr_auc:.5f}** |",
        f"| delta vs best single | **{blend.delta:+.5f}** |",
        "",
        "Blend weights, fitted by minimising log-loss (non-negative, summing to 1):",
        "",
        "| Model | Weight |",
        "|---|---:|",
    ]
    for name, weight in sorted(blend.weights.items(), key=lambda kv: -kv[1]):
        lines.append(f"| `{name}` | {weight:.4f} |")

    lines += [
        "",
        "### Why the gain is small, and why that was expected",
        "",
        "Three GBDTs on identical tabular features produce highly correlated predictions, so",
        "§6.5 predicts roughly +0.002 to +0.01 PR-AUC. Phase 4 measured the three models within",
        "**0.006 PR-AUC of each other** (DECISIONS.md D27), which bounds how much a blend of",
        "them can add. The weights are fitted by log-loss because it is smooth and",
        "differentiable under the simplex constraint; PR-AUC is a step function of the ranking.",
        "A log-loss-optimal blend can therefore be marginally *worse* on PR-AUC, and the delta",
        "is reported as measured either way.",
        "",
        "## Calibration",
        "",
        f"Isotonic regression, fitted on **{result['n_calib_half']:,}** rows "
        f"({result['rate_calib_half']:.2%} late) and evaluated on the "
        f"**{result['n_weight_half']:,}** rows held back for the weights "
        f"({result['rate_weight_half']:.2%} late).",
        "",
        "| Brier | Before | After | Change |",
        "|---|---:|---:|---:|",
        f"| in-sample (flatters) | {calibration.brier_before:.6f} | "
        f"{calibration.brier_after:.6f} | {-calibration.improvement:+.6f} |",
        f"| **held out** | {held['before']:.6f} | {held['after']:.6f} | "
        f"{held['after'] - held['before']:+.6f} |",
        "",
        "The held-out row is the honest one. The in-sample figure is reported only to make the",
        "difference between the two visible.",
        "",
        "**Expect this to look worse on the promotion evaluation set, and expect that to be",
        "correct.** The calibration window sits at roughly 9.75% positives against the",
        "evaluation window's 4.40% (DECISIONS.md D20), so a calibrator fitted here will",
        "over-predict there. That is a base-rate shift, not a defect in the calibrator.",
        "",
        f"The window was split **{result['split']}** rather than by date. A temporal split of",
        "this window leaves halves at 5.74% and 13.77% positives — the late rate climbs sharply",
        "through February — which would fit the calibrator on an unrepresentative slice. See",
        "`DECISIONS.md` D28.",
        "",
    ]

    selection = result.get("selection")
    if selection is not None:
        best_score = selection.scores[selection.best_by_pr_auc]
        lines += [
            "## Which single model ships (the equivalence band)",
            "",
            selection.sentence() + ".",
            "",
            "| Model | PR-AUC | gap to best | library | inside the band |",
            "|---|---:|---:|---:|:--:|",
        ]
        for name in sorted(selection.scores, key=lambda n: -selection.scores[n]):
            marker = " **(ships)**" if name == selection.chosen else ""
            lines.append(
                f"| `{name}`{marker} | {selection.scores[name]:.5f} | "
                f"{best_score - selection.scores[name]:.5f} | {selection.costs_mb[name]} MB | "
                f"{'yes' if name in selection.contenders else 'no'} |"
            )
        lines += [
            "",
            f"The band is **{selection.band:.3f} PR-AUC**, and it is not an arbitrary tolerance:",
            "it is the amount **thread count alone** moved CatBoost's score in `DECISIONS.md` D35,",
            "on identical data with identical hyperparameters. That is this pipeline's measured",
            "reproducibility floor, so a gap narrower than it is the environment rather than a",
            "result — declaring a winner inside it would claim a difference already shown to be",
            "irreproducible.",
            "",
            "Inside the band the tie is broken on **serving cost**, the only remaining",
            "difference anyone can observe: the shipped model's library is installed in the",
            "API image, 10 MB for LightGBM against 269 MB for CatBoost (`DECISIONS.md` D38).",
            "The rule lives in `configs/base.yaml` so the choice is a threshold, not a mood —",
            "the same reason `min_ensemble_gain_pr_auc` is a number in that file.",
            "",
            "A model more than the band below the best is **not** a contender at any price:",
            "the rule never trades accuracy for size, it settles ties accuracy cannot settle.",
            "",
        ]

    lines += [
        "## Threshold",
        "",
        threshold.sentence(),
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| threshold | **{threshold.threshold:.4f}** |",
        f"| assumed FN:FP cost ratio | {threshold.fn_fp_ratio:g}:1 |",
        f"| expected cost per order (FP units) | {threshold.expected_cost_per_order:.4f} |",
        f"| precision | {threshold.precision:.2%} |",
        f"| recall | {threshold.recall:.2%} |",
        f"| flag rate | {threshold.flag_rate:.2%} |",
        f"| true positives / false positives | {threshold.confusion['tp']:,} / "
        f"{threshold.confusion['fp']:,} |",
        f"| false negatives / true negatives | {threshold.confusion['fn']:,} / "
        f"{threshold.confusion['tn']:,} |",
        "",
        "**The cost ratio is an assumption, not a measurement.** Nothing here was derived from",
        "Olist's actual support costs; the ratio is a stated premise and the threshold is its",
        "consequence. Change the premise and the threshold moves.",
        "",
        "A second caveat on the threshold specifically: it is swept on a window whose base rate",
        "is roughly twice the evaluation window's, so the cost-minimising point here is lower",
        "than the one that would be chosen on deployment-era data. Phase 9 scores it on the",
        "evaluation set and that is where the number should be judged.",
        "",
        "## Business impact: review-score damage (§4.7)",
        "",
        f"- On-time orders average **{impact['on_time_mean']:.3f}** out of 5 "
        f"({impact['on_time_orders']:,} orders)",
        f"- Late orders average **{impact['late_mean']:.3f}** "
        f"({impact['late_orders']:,} orders)",
        f"- **Gap: {impact['gap']:.3f} review-score points**",
        f"- Share scoring 1 or 2 stars: {impact['on_time_share_1_or_2']:.1%} on-time against "
        f"**{impact['late_share_1_or_2']:.1%}** late",
        "",
        "At the threshold above, catching "
        f"{threshold.recall:.1%} of late orders puts roughly "
        f"{threshold.confusion['tp']:,} of the calibration window's late deliveries "
        "in reach of a proactive intervention. Whether that protects the review is an",
        "assumption this project does not test — it has no experiment, only the correlation.",
        "",
        "**Reviews are used for this analysis only.** Every review column is on the §4.2",
        "denylist; `review_impact()` asserts none has reached the feature matrix before it",
        "returns a single number.",
        "",
    ]

    explanation = result.get("explanation")
    if explanation is not None:
        lines += [
            "## SHAP (§6.7)",
            "",
            f"`TreeExplainer` on **{explanation.n_rows:,}** rows of `{explanation.model_name}`. "
            f"Additivity check: max |base + sum(shap) - model output| = "
            f"{explanation.additivity_error:.2e}.",
            "",
            "| Rank | Feature | Mean \\|SHAP\\| |",
            "|---:|---|---:|",
        ]
        for rank, (name, value) in enumerate(explanation.top(10), start=1):
            lines.append(f"| {rank} | `{name}` | {value:.5f} |")
        lines += [
            "",
            "Plots in `reports/shap/`: beeswarm, mean-|SHAP| bar and the top-3 dependence plots.",
            "",
        ]
        if explanation.alarms:
            lines += ["**Leakage alarms fired:**", ""]
            lines += [f"- {alarm}" for alarm in explanation.alarms]
            lines.append("")
        else:
            lines += [
                "No leakage alarm fired. §6.7 warns that a feature you thought was harmless",
                "showing up implausibly high is an alarm rather than a win, and D22 adds the",
                "mirror for this project: `seller_late_rate_hist` correlates only +0.0197 with",
                "the target under the §18 A1 rule, so its dominating would mean the snapshot",
                "join had regressed. Neither happened.",
                "",
            ]
        lines += [
            "The blend's SHAP values, had the blend shipped, would be the same weighted average",
            "of the per-model SHAP values: attributions are additive and a weighted average of",
            "models is a linear combination, so the attribution of the average is the average of",
            "the attributions. `blended_shap_values()` is that one line.",
            "",
        ]

    audit = result.get("audit")
    if audit:

        def describe(values: list[Any], limit: int = 12) -> str:
            """Summarise a value list: high-cardinality features would otherwise print 99
            levels twice and bury the one number that matters."""
            if not values:
                return "**none**"
            shown = ", ".join(f"`{v}`" for v in values[:limit])
            more = f" … ({len(values)} values)" if len(values) > limit else ""
            return shown + more

        overlap = audit["overlap"]
        lines += [
            f"## Feature audit: `{audit['feature']}` (§5)",
            "",
            f"SHAP ranked `{audit['feature']}` first, so it is the feature the audit examines.",
            "",
            f"- Values in the **fit** window: {describe(audit['fit_values'])}",
            f"- Values at **calibration** time: {describe(audit['calibration_values'])}",
            f"- **Overlap: {len(overlap)} of {len(set(audit['calibration_values']))}** "
            f"calibration-time values were seen in training",
            "",
            "| Model | PR-AUC with | without | delta | Brier with | without |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for name, m in sorted(audit["models"].items()):
            lines.append(
                f"| `{name}` | {m['pr_auc_with']:.5f} | {m['pr_auc_without']:.5f} | "
                f"**{m['pr_auc_delta']:+.5f}** | {m['brier_with']:.5f} | "
                f"{m['brier_without']:.5f} |"
            )
        lines += [
            "",
            "Each row is a genuine leave-one-out refit on the fit window, scored on the whole",
            "calibration window — not permutation importance. Permutation tells you what the",
            "model *used*; this tells you what it is *worth* without the feature, which is the",
            "question when deciding to remove it.",
            "",
        ]
        if not audit["overlap"]:
            lines += [
                "**This feature has no value overlap between training and scoring.** It cannot",
                "have been learned: a tree routes an unseen value to whichever branch the last",
                "split left open, so any apparent benefit is an accident of geometry rather than",
                "a learned effect. See `DECISIONS.md` D29 — this needs a decision before Phase 6",
                "registers a model.",
                "",
            ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Report written to %s", path)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for Phase 5."""
    import argparse

    parser = argparse.ArgumentParser(description="Phase 5: blend, calibrate, threshold")
    parser.add_argument("--version", default="v1")
    parser.add_argument("--no-shap", action="store_true", help="skip SHAP (it is the slow part)")
    parser.add_argument("--no-report", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    result = build_decision(args.version, run_shap=not args.no_shap)
    if not args.no_report:
        write_report(result)

    blend: BlendResult = result["blend"]
    logger.info(
        "DECISION: ship %s (blend delta %+.5f against a %.3f bar)",
        result["chosen"],
        blend.delta,
        blend.min_gain,
    )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
