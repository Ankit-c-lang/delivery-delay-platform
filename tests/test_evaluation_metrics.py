"""Metrics, checked against values computed by hand (PLAN.md §13 Phase 4, §4.6).

Every expected number below is derived in a comment rather than copied from a previous run of
the code. A metric test that asserts what the code currently returns is a change detector, not
a correctness test, and it will happily lock in a bug.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics import (
    DEFAULT_CAPACITY,
    brier,
    positive_rate,
    pr_auc,
    precision_at_capacity,
    recall_at_capacity,
    roc_auc,
    summarize,
    trivial_baseline,
)


class TestPrAuc:
    def test_a_perfect_ranking_scores_one(self):
        assert pr_auc([1, 1, 0, 0], [4.0, 3.0, 2.0, 1.0]) == pytest.approx(1.0)

    def test_hand_computed_interleaved_case(self):
        """y = [1, 0, 1, 0] with descending scores, so positives sit at ranks 1 and 3.

        Average precision is the sum over positives of precision-at-that-rank times the
        recall increment:
            rank 1: precision 1/1, recall goes 0 -> 1/2  ->  1.000 * 0.5 = 0.50000
            rank 3: precision 2/3, recall goes 1/2 -> 1  ->  0.667 * 0.5 = 0.33333
            total = 0.83333
        """
        assert pr_auc([1, 0, 1, 0], [4.0, 3.0, 2.0, 1.0]) == pytest.approx(0.833333, abs=1e-5)

    def test_a_single_positive_ranked_last(self):
        """One positive at rank 2 of 2: precision there is 1/2, recall increment 1."""
        assert pr_auc([0, 1], [1.0, 0.0]) == pytest.approx(0.5)

    def test_undefined_without_positives(self):
        with pytest.raises(ValueError, match="no positive labels"):
            pr_auc([0, 0, 0], [0.1, 0.2, 0.3])


class TestRocAuc:
    def test_hand_computed_case(self):
        """y = [1, 0, 1, 0], scores [4, 3, 2, 1]. ROC-AUC is the probability a random
        positive outranks a random negative. The four pairs are
        (4>3) yes, (4>1) yes, (2>3) no, (2>1) yes  ->  3/4.
        """
        assert roc_auc([1, 0, 1, 0], [4.0, 3.0, 2.0, 1.0]) == pytest.approx(0.75)

    def test_perfect_separation_is_one(self):
        assert roc_auc([1, 1, 0, 0], [4.0, 3.0, 2.0, 1.0]) == pytest.approx(1.0)

    def test_inverted_ranking_is_zero(self):
        assert roc_auc([0, 0, 1, 1], [4.0, 3.0, 2.0, 1.0]) == pytest.approx(0.0)

    def test_undefined_with_one_class(self):
        with pytest.raises(ValueError, match="one class is absent"):
            roc_auc([1, 1], [0.9, 0.8])


class TestBrier:
    def test_hand_computed_case(self):
        """((1 - 0.8)^2 + (0 - 0.3)^2) / 2 = (0.04 + 0.09) / 2 = 0.065."""
        assert brier([1, 0], [0.8, 0.3]) == pytest.approx(0.065)

    def test_perfect_predictions_are_zero(self):
        assert brier([1, 0, 1], [1.0, 0.0, 1.0]) == pytest.approx(0.0)

    def test_confidently_wrong_is_one(self):
        assert brier([1, 0], [0.0, 1.0]) == pytest.approx(1.0)

    def test_rejects_scores_outside_zero_one(self):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            brier([1, 0], [1.4, 0.3])


class TestRecallAtCapacity:
    def test_hand_computed_case(self):
        """10 orders, descending scores, positives at ranks 1, 2 and 5.

        capacity 0.3 reviews ceil(10 * 0.3) = 3 orders, catching the positives at ranks 1
        and 2 but not the one at rank 5  ->  recall 2/3.
        """
        labels = [1, 1, 0, 0, 1, 0, 0, 0, 0, 0]
        scores = list(range(10, 0, -1))
        assert recall_at_capacity(labels, scores, 0.3) == pytest.approx(2 / 3)

    def test_precision_on_the_same_case(self):
        """2 of the 3 reviewed orders were really late."""
        labels = [1, 1, 0, 0, 1, 0, 0, 0, 0, 0]
        scores = list(range(10, 0, -1))
        assert precision_at_capacity(labels, scores, 0.3) == pytest.approx(2 / 3)

    def test_full_capacity_catches_everything(self):
        assert recall_at_capacity([1, 0, 1], [0.1, 0.9, 0.2], 1.0) == pytest.approx(1.0)

    def test_the_reviewed_count_rounds_up_so_it_is_never_empty(self):
        """ceil(10 * 0.01) = 1. Rounding down would review nobody and report 0 recall for a
        perfect model, which is a metric bug rather than a model result."""
        labels = [1] + [0] * 9
        scores = list(range(10, 0, -1))
        assert recall_at_capacity(labels, scores, 0.01) == pytest.approx(1.0)

    def test_ties_are_broken_stably_and_reproducibly(self):
        labels = [0, 1, 0, 1]
        scores = [0.5, 0.5, 0.5, 0.5]
        first = recall_at_capacity(labels, scores, 0.5)
        assert first == recall_at_capacity(labels, scores, 0.5)
        assert first == pytest.approx(0.5)

    @pytest.mark.parametrize("capacity", [0.0, -0.1, 1.5])
    def test_rejects_an_impossible_capacity(self, capacity):
        with pytest.raises(ValueError, match=r"capacity must be in \(0, 1\]"):
            recall_at_capacity([1, 0], [0.9, 0.1], capacity)

    def test_undefined_without_positives(self):
        with pytest.raises(ValueError, match="no positive labels"):
            recall_at_capacity([0, 0], [0.9, 0.1], 0.5)

    def test_the_default_capacity_is_the_plans_ten_percent(self):
        assert DEFAULT_CAPACITY == 0.10


class TestValidation:
    """Each of these would otherwise produce a plausible number from nonsense."""

    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="length mismatch"):
            pr_auc([1, 0, 1], [0.9, 0.1])

    def test_empty_input(self):
        with pytest.raises(ValueError, match="empty input"):
            pr_auc([], [])

    def test_non_binary_labels(self):
        with pytest.raises(ValueError, match="must be binary"):
            pr_auc([0, 1, 2], [0.1, 0.2, 0.3])

    def test_nan_scores(self):
        with pytest.raises(ValueError, match="NaN or infinity"):
            pr_auc([1, 0], [0.5, np.nan])

    def test_infinite_scores(self):
        with pytest.raises(ValueError, match="NaN or infinity"):
            pr_auc([1, 0], [np.inf, 0.1])

    def test_boolean_labels_are_accepted(self):
        assert pr_auc([True, False], [0.9, 0.1]) == pytest.approx(1.0)


class TestSummarize:
    def test_contains_every_metric_and_the_base_rate(self):
        labels = [1, 0, 1, 0]
        scores = [0.9, 0.6, 0.4, 0.1]
        out = summarize(labels, scores, capacity=0.5)
        assert set(out) == {
            "pr_auc",
            "roc_auc",
            "brier",
            "recall_at_capacity",
            "precision_at_capacity",
            "lift_at_capacity",
            "positive_rate",
            "capacity",
            "n",
        }
        assert out["positive_rate"] == pytest.approx(0.5)
        assert out["n"] == 4

    def test_lift_is_precision_over_the_base_rate(self):
        """Base rate 0.5; capacity 0.5 reviews the top 2, which are labels [1, 0], so
        precision is 0.5 and lift is exactly 1.0 — this ranking is no better than random at
        that operating point."""
        out = summarize([1, 0, 1, 0], [0.9, 0.6, 0.4, 0.1], capacity=0.5)
        assert out["precision_at_capacity"] == pytest.approx(0.5)
        assert out["lift_at_capacity"] == pytest.approx(1.0)

    def test_lift_above_one_for_a_good_ranking(self):
        out = summarize([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1], capacity=0.5)
        assert out["lift_at_capacity"] == pytest.approx(2.0)

    def test_prefix_is_applied_to_every_key(self):
        out = summarize([1, 0], [0.9, 0.1], prefix="val_")
        assert all(key.startswith("val_") for key in out)
        assert "val_pr_auc" in out

    def test_values_are_plain_floats_so_mlflow_can_log_them(self):
        for value in summarize([1, 0], [0.9, 0.1]).values():
            assert isinstance(value, float)


class TestTrivialBaseline:
    def test_accuracy_is_the_negative_rate_and_recall_is_zero(self):
        """PLAN.md §4.6's point in one assertion: 93% accuracy, 0% recall."""
        labels = [1] * 7 + [0] * 93
        out = trivial_baseline(labels)
        assert out["accuracy"] == pytest.approx(0.93)
        assert out["recall"] == 0.0
        assert out["n"] == 100

    def test_positive_rate_helper(self):
        assert positive_rate([1] * 7 + [0] * 93) == pytest.approx(0.07)
