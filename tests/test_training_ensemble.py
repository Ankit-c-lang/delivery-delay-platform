"""Blend, calibration and threshold tests (PLAN.md §13 Phase 5, §6.5, §4.7).

The phase names three required tests, and each guards a failure that would otherwise look
like a result:

1. **Blend weights non-negative and summing to 1.** A negative weight is not a blend, it is an
   unconstrained regression that will not generalise; weights that do not sum to 1 shift every
   probability and silently destroy calibration.
2. **The calibrator improves Brier on held-out data.** In-sample it always appears to help,
   because isotonic regression can fit the training points arbitrarily closely.
3. **The threshold sweep returns the actual argmin**, checked against an independent brute
   force rather than against what the function returned last time.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import brier
from src.training.ensemble import (
    EPSILON,
    blend_probabilities,
    evaluate_blend,
    fit_blend_weights,
    fit_calibrator,
    select_single_model,
    split_calibration_window,
    sweep_threshold,
)


@pytest.fixture
def three_models():
    """Three correlated scorers of differing quality, with a 7% positive rate."""
    rng = np.random.default_rng(11)
    n = 4000
    signal = rng.normal(size=n)
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-(signal - 2.6)))).astype(int)

    def scorer(noise: float) -> np.ndarray:
        logit = signal - 2.6 + rng.normal(scale=noise, size=n)
        return 1.0 / (1.0 + np.exp(-logit))

    return {"good": scorer(0.3), "fair": scorer(0.8), "poor": scorer(2.0)}, y


class TestBlendWeights:
    """The phase's first required test, and the shape of the constraint around it."""

    def test_weights_are_non_negative_and_sum_to_one(self, three_models):
        probabilities, y = three_models
        weights = fit_blend_weights(probabilities, y)
        assert set(weights) == set(probabilities)
        assert all(w >= 0.0 for w in weights.values()), weights
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-9)

    def test_the_sum_is_exact_not_merely_close(self, three_models):
        """SLSQP leaves tiny drift; the implementation renormalises so the invariant holds
        exactly. Asserted tightly because a 1e-6 drift applied to every probability is a
        systematic bias, not noise."""
        probabilities, y = three_models
        weights = fit_blend_weights(probabilities, y)
        assert abs(sum(weights.values()) - 1.0) < 1e-12

    def test_the_better_model_gets_more_weight(self, three_models):
        probabilities, y = three_models
        weights = fit_blend_weights(probabilities, y)
        assert weights["good"] > weights["poor"]

    def test_a_useless_model_can_be_driven_to_zero(self, three_models):
        probabilities, y = three_models
        rng = np.random.default_rng(3)
        probabilities = {**probabilities, "noise": rng.random(len(y))}
        weights = fit_blend_weights(probabilities, y)
        assert weights["noise"] < 0.1
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-9)

    def test_blending_a_single_model_is_refused(self, three_models):
        probabilities, y = three_models
        with pytest.raises(ValueError, match="at least two models"):
            fit_blend_weights({"good": probabilities["good"]}, y)

    def test_length_mismatch_is_refused(self, three_models):
        probabilities, y = three_models
        broken = {**probabilities, "short": probabilities["good"][:-5]}
        with pytest.raises(ValueError, match="disagree on length"):
            fit_blend_weights(broken, y)

    def test_blend_probabilities_rejects_a_model_weight_mismatch(self, three_models):
        probabilities, y = three_models
        weights = fit_blend_weights(probabilities, y)
        del weights["poor"]
        with pytest.raises(KeyError, match="do not match"):
            blend_probabilities(probabilities, weights)

    def test_equal_weights_give_the_arithmetic_mean(self):
        probabilities = {"a": np.array([0.2, 0.8]), "b": np.array([0.4, 0.6])}
        blended = blend_probabilities(probabilities, {"a": 0.5, "b": 0.5})
        np.testing.assert_allclose(blended, [0.3, 0.7])

    def test_blended_probabilities_stay_in_range(self, three_models):
        probabilities, y = three_models
        blended = blend_probabilities(probabilities, fit_blend_weights(probabilities, y))
        assert blended.min() >= 0.0
        assert blended.max() <= 1.0


class TestBlendHonesty:
    """§13: "Do not inflate the ensemble result"."""

    def test_the_delta_is_reported_against_the_best_single_model(self, three_models):
        probabilities, y = three_models
        weights = fit_blend_weights(probabilities, y)
        result = evaluate_blend(probabilities, y, weights, min_gain=0.005)
        assert result.best_single in probabilities
        assert result.best_single == max(result.single_pr_auc, key=result.single_pr_auc.get)
        assert result.delta == pytest.approx(
            result.blend_pr_auc - result.single_pr_auc[result.best_single]
        )

    def test_a_gain_below_the_minimum_does_not_ship(self, three_models):
        probabilities, y = three_models
        weights = fit_blend_weights(probabilities, y)
        strict = evaluate_blend(probabilities, y, weights, min_gain=1.0)
        assert strict.ship_ensemble is False

    def test_a_negative_delta_is_representable_and_does_not_ship(self):
        """A blend fitted by log-loss can make PR-AUC worse. The result type must be able to
        say so rather than clamping at zero."""
        y = np.array([0, 0, 1, 1] * 50)
        rng = np.random.default_rng(5)
        good = np.where(y == 1, 0.9, 0.1) + rng.normal(scale=0.01, size=len(y))
        probabilities = {"good": np.clip(good, 0, 1), "bad": rng.random(len(y))}
        weights = {"good": 0.5, "bad": 0.5}
        result = evaluate_blend(probabilities, y, weights, min_gain=0.005)
        assert result.delta < 0
        assert result.ship_ensemble is False

    def test_summary_is_roundable_and_json_friendly(self, three_models):
        probabilities, y = three_models
        weights = fit_blend_weights(probabilities, y)
        summary = evaluate_blend(probabilities, y, weights, min_gain=0.005).summary()
        assert isinstance(summary["ship_ensemble"], bool)
        assert set(summary["weights"]) == set(probabilities)


class TestCalibration:
    """The phase's second required test: **held-out** Brier, not in-sample."""

    def test_the_calibrator_improves_brier_on_held_out_data(self, three_models):
        probabilities, y = three_models
        raw = probabilities["fair"]
        # Deliberately miscalibrate: squash toward 0.5 so the probabilities rank well but are
        # badly scaled, which is exactly what calibration is for.
        skewed = 0.5 + (raw - 0.5) * 0.35

        half = len(y) // 2
        fitted = fit_calibrator(skewed[:half], y[:half])
        held_out_before = brier(y[half:], skewed[half:])
        held_out_after = brier(y[half:], fitted.calibrator.predict(skewed[half:]))
        assert held_out_after < held_out_before, (
            f"calibration made held-out Brier worse: {held_out_before:.6f} -> "
            f"{held_out_after:.6f}"
        )

    def test_in_sample_improvement_is_reported_separately(self, three_models):
        """It flatters, so it is labelled rather than presented as the result."""
        probabilities, y = three_models
        result = fit_calibrator(probabilities["fair"], y)
        assert result.brier_after <= result.brier_before
        assert result.improvement == pytest.approx(result.brier_before - result.brier_after)

    def test_the_fitting_sample_size_is_reported(self, three_models):
        """Isotonic regression is non-parametric and unstable on few positives, so the count
        is surfaced rather than assumed adequate."""
        probabilities, y = three_models
        result = fit_calibrator(probabilities["good"], y)
        assert result.n_fit == len(y)
        assert result.positives_fit == int(y.sum())

    def test_output_stays_in_range_and_is_monotone(self, three_models):
        probabilities, y = three_models
        result = fit_calibrator(probabilities["good"], y)
        grid = np.linspace(0.0, 1.0, 50)
        out = result.calibrator.predict(grid)
        assert out.min() >= 0.0
        assert out.max() <= 1.0
        assert np.all(np.diff(out) >= -1e-12), "isotonic output must be non-decreasing"

    def test_probabilities_outside_the_fitted_range_clip_rather_than_return_nan(self):
        """An uncalibrated NaN reaching the API is a hard failure at the worst moment."""
        rng = np.random.default_rng(0)
        raw = rng.uniform(0.2, 0.8, 500)
        y = (rng.random(500) < raw).astype(int)
        result = fit_calibrator(raw, y)
        out = result.calibrator.predict(np.array([0.0, 1.0]))
        assert np.isfinite(out).all()


class TestThresholdSweep:
    """The phase's third required test: the actual argmin."""

    def test_the_sweep_returns_the_true_argmin(self, three_models):
        """Brute-forced independently: for every candidate threshold, recompute the cost from
        the definition and confirm nothing beats what was returned."""
        probabilities, y = three_models
        scores = probabilities["good"]
        result = sweep_threshold(scores, y, fn_fp_ratio=5.0)

        best_cost = np.inf
        for threshold in np.unique(scores):
            flagged = scores >= threshold
            fn = int(np.sum((y == 1) & ~flagged))
            fp = int(np.sum((y == 0) & flagged))
            best_cost = min(best_cost, (5.0 * fn + fp) / len(y))
        assert result.expected_cost_per_order == pytest.approx(best_cost)

    def test_the_reported_cost_matches_the_reported_threshold(self, three_models):
        probabilities, y = three_models
        scores = probabilities["good"]
        result = sweep_threshold(scores, y, fn_fp_ratio=5.0)
        flagged = scores >= result.threshold
        fn = int(np.sum((y == 1) & ~flagged))
        fp = int(np.sum((y == 0) & flagged))
        assert result.expected_cost_per_order == pytest.approx((5.0 * fn + fp) / len(y))

    def test_confusion_counts_are_consistent(self, three_models):
        probabilities, y = three_models
        result = sweep_threshold(probabilities["good"], y, fn_fp_ratio=5.0)
        c = result.confusion
        assert c["tp"] + c["fp"] + c["fn"] + c["tn"] == len(y)
        assert c["tp"] + c["fn"] == int(y.sum())
        assert result.precision == pytest.approx(c["tp"] / (c["tp"] + c["fp"]))
        assert result.recall == pytest.approx(c["tp"] / (c["tp"] + c["fn"]))
        assert result.flag_rate == pytest.approx((c["tp"] + c["fp"]) / len(y))

    def test_a_harsher_false_negative_cost_lowers_the_threshold(self, three_models):
        """The economics must move the operating point in the right direction: if missing a
        late order costs more, flag more aggressively."""
        probabilities, y = three_models
        cheap = sweep_threshold(probabilities["good"], y, fn_fp_ratio=2.0)
        harsh = sweep_threshold(probabilities["good"], y, fn_fp_ratio=20.0)
        assert harsh.threshold <= cheap.threshold
        assert harsh.recall >= cheap.recall
        assert harsh.flag_rate >= cheap.flag_rate

    def test_an_extreme_ratio_flags_almost_everything(self, three_models):
        probabilities, y = three_models
        result = sweep_threshold(probabilities["good"], y, fn_fp_ratio=1000.0)
        assert result.recall > 0.95

    def test_a_custom_grid_is_honoured(self, three_models):
        probabilities, y = three_models
        grid = np.array([0.05, 0.10, 0.20])
        result = sweep_threshold(probabilities["good"], y, 5.0, grid=grid)
        assert result.threshold in set(grid.tolist())
        assert len(result.sweep) == 3

    def test_the_sentence_states_the_assumption(self, three_models):
        """§4.7: never present the cost figures as if they were measured."""
        probabilities, y = three_models
        sentence = sweep_threshold(probabilities["good"], y, 5.0).sentence()
        assert "assumed" in sentence.lower()
        assert "5:1" in sentence

    @pytest.mark.parametrize("ratio", [0.0, -1.0])
    def test_a_non_positive_ratio_is_refused(self, three_models, ratio):
        probabilities, y = three_models
        with pytest.raises(ValueError, match="must be positive"):
            sweep_threshold(probabilities["good"], y, ratio)

    def test_a_single_class_is_refused(self):
        with pytest.raises(ValueError, match="both classes"):
            sweep_threshold(np.array([0.1, 0.2, 0.3]), np.array([0, 0, 0]), 5.0)


class TestCalibrationWindowSplit:
    """Blend weights and the calibrator must not be fitted on the same rows (§13)."""

    @pytest.fixture
    def stamps(self) -> pd.Series:
        return pd.Series(
            pd.Timestamp("2018-01-01") + pd.to_timedelta(np.arange(1000) * 90, unit="min")
        )

    def test_the_halves_are_disjoint_and_complete(self, stamps):
        first, second = split_calibration_window(stamps)
        assert not (first & second).any()
        assert (first | second).all()

    def test_interleaving_preserves_the_base_rate_in_both_halves(self, stamps):
        """The reason interleaved is the default. A temporal split of v1's real window leaves
        halves at 5.74% and 13.77% positives, because the late rate rises sharply through
        February (DECISIONS.md D20)."""
        rng = np.random.default_rng(2)
        # Positive probability rising over time, as it really does in the window.
        y = (rng.random(len(stamps)) < np.linspace(0.02, 0.16, len(stamps))).astype(int)

        first, second = split_calibration_window(stamps, "interleaved")
        assert abs(y[first].mean() - y[second].mean()) < 0.02

        early, late = split_calibration_window(stamps, "temporal")
        assert y[late].mean() > y[early].mean() * 1.5, "temporal split should be imbalanced here"

    def test_temporal_split_is_ordered_in_time(self, stamps):
        early, late = split_calibration_window(stamps, "temporal")
        assert stamps[early].max() <= stamps[late].min()

    def test_an_unknown_strategy_is_refused(self, stamps):
        with pytest.raises(ValueError, match="unknown calibration split"):
            split_calibration_window(stamps, "random")

    def test_epsilon_keeps_log_loss_finite(self):
        assert 0 < EPSILON < 1e-3


class TestTheSelectionBand:
    """DECISIONS.md D41: which single model ships, when PR-AUC cannot tell them apart.

    The edge cases matter more than the happy path here, because this rule decides what gets
    deployed. The one thing it must never do is trade real accuracy for a smaller image.
    """

    COSTS: ClassVar[dict[str, int]] = {"lightgbm": 10, "xgboost": 85, "catboost": 269}

    def test_the_cheapest_model_wins_inside_the_band(self):
        """The real case, with the measured numbers from `make decide`."""
        selection = select_single_model(
            {"catboost": 0.21604, "lightgbm": 0.21585, "xgboost": 0.21237}, 0.005, self.COSTS
        )
        assert selection.chosen == "lightgbm"
        assert selection.best_by_pr_auc == "catboost"
        assert selection.rule_changed_the_outcome is True
        assert "259 MB less" in selection.sentence()

    def test_a_real_difference_is_never_traded_for_a_smaller_image(self):
        """The property that makes the rule safe. A gap wider than the band is decisive."""
        selection = select_single_model(
            {"catboost": 0.2600, "lightgbm": 0.2100, "xgboost": 0.2000}, 0.005, self.COSTS
        )
        assert selection.chosen == "catboost"
        assert selection.rule_changed_the_outcome is False
        assert selection.contenders == ("catboost",)

    def test_the_band_boundary_is_inclusive(self):
        """A model exactly `band` below the best is equivalent, not excluded.

        Pinned because an off-by-one here silently changes what ships: exclusive would make the
        rule fire less often than the config says, and nothing else would reveal it.
        """
        inside = select_single_model({"catboost": 0.2200, "lightgbm": 0.2150}, 0.005, self.COSTS)
        assert inside.chosen == "lightgbm"
        outside = select_single_model({"catboost": 0.2200, "lightgbm": 0.21499}, 0.005, self.COSTS)
        assert outside.chosen == "catboost"

    def test_a_zero_band_disables_the_rule_entirely(self):
        """So the behaviour can be turned off from config without touching code."""
        selection = select_single_model({"catboost": 0.21604, "lightgbm": 0.21585}, 0.0, self.COSTS)
        assert selection.chosen == "catboost"
        assert selection.rule_changed_the_outcome is False

    def test_the_cheapest_model_winning_on_its_own_merit_is_not_a_rule_change(self):
        """If the cheapest is also the best, nothing was traded and the report must not imply it."""
        selection = select_single_model({"lightgbm": 0.2200, "catboost": 0.2190}, 0.005, self.COSTS)
        assert selection.chosen == "lightgbm"
        assert selection.rule_changed_the_outcome is False
        assert "also" in selection.sentence()

    def test_an_unpriced_model_is_refused_rather_than_treated_as_free(self):
        """Defaulting would let an unmeasured library win every tie by appearing to cost nothing."""
        with pytest.raises(ValueError, match="no serving cost configured"):
            select_single_model({"catboost": 0.21, "newthing": 0.21}, 0.005, self.COSTS)

    def test_a_negative_band_is_refused(self):
        with pytest.raises(ValueError, match="non-negative"):
            select_single_model({"catboost": 0.21}, -0.001, self.COSTS)

    def test_no_scores_is_refused(self):
        with pytest.raises(ValueError, match="no model scores"):
            select_single_model({}, 0.005, self.COSTS)

    def test_the_result_is_deterministic_whatever_order_the_scores_arrive_in(self):
        """Dict order must not decide what ships."""
        forward = {"catboost": 0.2160, "lightgbm": 0.2160, "xgboost": 0.2160}
        backward = dict(reversed(list(forward.items())))
        assert (
            select_single_model(forward, 0.005, self.COSTS).chosen
            == select_single_model(backward, 0.005, self.COSTS).chosen
            == "lightgbm"
        )

    def test_the_band_matches_the_reproducibility_floor_measured_in_d35(self):
        """The band's value is an argument, not a preference, so it is pinned to its source.

        D35 measured CatBoost moving 0.21344 -> 0.20850 on identical data and hyperparameters when
        only the thread count changed. If someone widens the band, this test makes them confront
        that the justification no longer holds.
        """
        import yaml

        config = yaml.safe_load(Path("configs/base.yaml").read_text(encoding="utf-8"))
        band = config["decision"]["selection_band_pr_auc"]
        measured_thread_count_spread = 0.21344 - 0.20850
        assert band == pytest.approx(measured_thread_count_spread, abs=1e-4), (
            "the band is justified as D35's measured reproducibility floor; changing it needs a "
            "new measurement, not a new preference"
        )

    def test_every_supported_model_has_a_configured_cost(self):
        """Otherwise the rule raises at decision time, after the expensive part has already run."""
        import yaml

        from src.training.tune import SUPPORTED

        config = yaml.safe_load(Path("configs/base.yaml").read_text(encoding="utf-8"))
        costs = config["decision"]["serving_cost_mb"]
        assert set(SUPPORTED) <= set(costs), f"missing: {sorted(set(SUPPORTED) - set(costs))}"
