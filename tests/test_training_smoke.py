"""Smoke training on the synthetic fixture (PLAN.md §13 Phase 4, §10.3).

These are the tests CI will actually be able to run from Phase 10: they need neither Postgres
nor the real dataset, because the dataset is 121 MB and CC BY-NC-SA and will never be in a
workflow. They use the **real search spaces** from ``configs/models.yaml`` with the budget cut
to a couple of trials, so a malformed search space or a library API change is caught here
rather than 15 minutes into a tuning run.

The phase's stated requirement is that a smoke train finishes in **under 30 seconds**.
"""

from __future__ import annotations

import copy
import time

import numpy as np
import pytest
from tests.fixtures.generate_synthetic import synthetic_matrix

from src.evaluation.metrics import pr_auc, summarize
from src.training.baselines import MajorityClassBaseline, build_logistic_baseline
from src.training.tune import cross_validate_baselines, load_model_config, tune


@pytest.fixture(scope="module")
def fixture_data():
    """600 synthetic rows with the real 39-column schema."""
    return synthetic_matrix()


def tiny_config(model: str, trials: int = 2, timeout: int = 25) -> dict:
    """The real search space on a token budget."""
    config = copy.deepcopy(load_model_config())
    config["cv"]["n_splits"] = 3
    spec = config["models"][model]
    spec["trials"] = trials
    spec["timeout_seconds"] = timeout
    # Keep the smoke run honest but quick: the point is that the plumbing works, not that the
    # model is good.
    for key, value in (("n_estimators", 60), ("iterations", 60)):
        if key in spec["fixed"]:
            spec["fixed"][key] = value
    return config


class TestSchemaFidelity:
    """A fixture that has drifted from the real schema tests nothing."""

    def test_the_fixture_matches_the_real_feature_contract(self, fixture_data):
        from src.features.build import CATEGORICAL_FEATURES, FEATURE_NAMES

        X, y, stamps = fixture_data
        assert list(X.columns) == list(FEATURE_NAMES)
        assert len(X) == len(y) == len(stamps)
        for column in CATEGORICAL_FEATURES:
            assert str(X[column].dtype) == "category", column

    def test_the_positive_rate_is_realistic(self, fixture_data):
        _, y, _ = fixture_data
        assert 0.04 <= y.mean() <= 0.12

    def test_timestamps_are_already_sorted(self, fixture_data):
        from src.training.tune import assert_time_sorted

        _, _, stamps = fixture_data
        assert_time_sorted(stamps)

    def test_generation_is_deterministic(self):
        first, y1, _ = synthetic_matrix(seed=7)
        second, y2, _ = synthetic_matrix(seed=7)
        assert first.equals(second)
        assert (y1 == y2).all()

    def test_no_sentinel_level_is_ever_drawn_as_a_value(self, fixture_data):
        """The sentinels must exist as levels without appearing as values, which is how the
        real matrix behaves when training saw every level."""
        from src.features.build import CATEGORICAL_FEATURES, MISSING, UNKNOWN

        X, _, _ = fixture_data
        for column in CATEGORICAL_FEATURES:
            values = set(X[column].astype(str))
            assert MISSING not in values and UNKNOWN not in values, column
            assert MISSING in set(X[column].cat.categories), column


class TestSmokeTrain:
    def test_lightgbm_smoke_train_finishes_under_thirty_seconds(self, fixture_data):
        """The phase's required test.

        Measured at ~3.9 s on an idle machine, so the 30 s bound has an 8x margin. It does
        assume the machine is not otherwise saturated: running this while a full Optuna study
        occupied all 12 threads pushed it past 30 s. That is CPU contention rather than a
        regression, and worth knowing before treating a failure here as a code problem.
        """
        X, y, stamps = fixture_data
        started = time.perf_counter()
        result = tune("lightgbm", X, y, stamps, tiny_config("lightgbm"))
        elapsed = time.perf_counter() - started
        assert elapsed < 30, f"smoke train took {elapsed:.1f}s"
        assert result.trials_completed >= 1
        assert result.best_score > float(y.mean()), "did not beat the base rate"

    @pytest.mark.parametrize("model", ["xgboost", "catboost"])
    def test_the_other_libraries_fit_on_the_real_search_space(self, fixture_data, model):
        """Catches an API change or a malformed space before a 15-minute run does."""
        X, y, stamps = fixture_data
        result = tune(model, X, y, stamps, tiny_config(model, trials=1, timeout=60))
        assert result.trials_completed == 1
        assert 0.0 <= result.best_score <= 1.0

    def test_tuning_rejects_an_unknown_model(self, fixture_data):
        X, y, stamps = fixture_data
        with pytest.raises(ValueError, match="unsupported model"):
            tune("randomforest", X, y, stamps, load_model_config())

    def test_tuning_rejects_unsorted_rows(self, fixture_data):
        X, y, stamps = fixture_data
        shuffled = stamps.sample(frac=1.0, random_state=1)
        with pytest.raises(ValueError, match="not sorted"):
            tune("lightgbm", X, y, shuffled, tiny_config("lightgbm"))

    def test_the_result_reports_what_it_actually_did(self, fixture_data):
        """A study that ran 1 of 40 trials must say so rather than be reported as tuned."""
        X, y, stamps = fixture_data
        result = tune("lightgbm", X, y, stamps, tiny_config("lightgbm", trials=2))
        summary = result.summary()
        assert summary["trials_requested"] == 2
        assert summary["trials_completed"] <= 2
        assert summary["seconds"] > 0
        assert len(summary["fold_scores"]) >= 1


class TestBaselinesOnTheFixture:
    def test_majority_class_scores_about_the_base_rate(self, fixture_data):
        """A constant score ranks nothing, so PR-AUC lands at the base rate. That is the floor
        every real model must clear visibly."""
        X, y, _ = fixture_data
        baseline = MajorityClassBaseline().fit(X, y)
        scores = baseline.predict_proba_positive(X)
        assert len(set(scores)) == 1
        assert pr_auc(y, scores) == pytest.approx(float(y.mean()), abs=0.02)

    def test_majority_class_roc_auc_is_exactly_one_half(self, fixture_data):
        X, y, _ = fixture_data
        baseline = MajorityClassBaseline().fit(X, y)
        out = summarize(y, baseline.predict_proba_positive(X))
        assert out["roc_auc"] == pytest.approx(0.5)
        assert out["recall_at_capacity"] > 0  # ties still fill the review queue

    def test_majority_class_refuses_to_predict_before_fitting(self, fixture_data):
        X, _, _ = fixture_data
        with pytest.raises(RuntimeError, match="fit the baseline"):
            MajorityClassBaseline().predict_proba_positive(X)

    def test_logistic_baseline_beats_the_floor(self, fixture_data):
        X, y, _ = fixture_data
        pipeline = build_logistic_baseline(X)
        pipeline.fit(X, y)
        score = pr_auc(y, pipeline.predict_proba(X)[:, 1])
        assert score > float(y.mean())

    def test_logistic_baseline_drops_the_high_cardinality_categoricals(self, fixture_data):
        """A 100-level zip prefix does not earn 100 sparse columns in a linear model. The
        GBDTs see it natively."""
        from src.training.baselines import one_hot_columns

        X, _, _ = fixture_data
        chosen = one_hot_columns(X)
        assert "customer_state" in chosen
        assert all(len(X[c].cat.categories) <= 30 for c in chosen)

    def test_both_baselines_cross_validate_on_the_same_folds(self, fixture_data):
        X, y, _ = fixture_data
        scores = cross_validate_baselines(X, y, n_splits=3)
        assert set(scores) == {"majority_class", "logistic_regression"}
        assert len(scores["majority_class"]) == len(scores["logistic_regression"]) == 3
        assert np.mean(scores["logistic_regression"]) > np.mean(scores["majority_class"])
