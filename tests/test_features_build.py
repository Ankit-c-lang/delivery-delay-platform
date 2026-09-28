"""Tests for feature construction and the preprocessing artifact (PLAN.md §13 Phase 3 part B).

Ranked by what the failure would cost:

1. A §4.2 denylist column reaching the feature matrix — invalidates every downstream number.
2. Imputation medians or category levels fitted on anything but training rows (§6.6).
3. Column order drifting between fit and transform — the model silently reads the wrong
   column, and nothing errors.
4. ``transform()`` not being deterministic, which makes the Phase 7 parity test meaningless.
5. An unseen category level crashing at serve time instead of becoming ``__unknown__``.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from src.etl.schema import DENYLIST
from src.features.artifact import (
    ARTIFACT_VERSION,
    CATEGORY_CAP,
    WINSOR_QUANTILE,
    PreprocessingArtifact,
    input_schema_hash,
)
from src.features.build import (
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_NAMES,
    MISSING,
    NUMERIC_FEATURES,
    REQUIRED_INPUT_COLUMNS,
    STATE_TO_REGION,
    UNKNOWN,
    cap_categories,
    construct_features,
)
from src.features.history import ENTITIES, GLOBAL_LEVEL, build_all_snapshots

N = 60


def make_orders(n: int = N, **overrides) -> pd.DataFrame:
    """An orders frame carrying every column construct_features requires."""
    purchased = pd.Timestamp("2017-06-01 09:30:00") + pd.to_timedelta(np.arange(n), unit="D")
    states = np.array(["SP", "RJ", "MG", "BA"])[np.arange(n) % 4]
    data = {
        "order_id": [f"o{i:04d}" for i in range(n)],
        "order_purchase_timestamp": purchased,
        "purchase_month": purchased.to_period("M").to_timestamp(),
        "promised_days": np.full(n, 20),
        "days_to_shipping_limit": np.full(n, 5.0),
        "customer_state": states,
        "seller_state": np.where(np.arange(n) % 2 == 0, "SP", "RJ"),
        "customer_zip_code_prefix": np.array(["01001", "20010", "30110", "40010"])[
            np.arange(n) % 4
        ],
        "customer_seller_distance_km": np.full(n, 360.7),
        "n_seller_states": np.ones(n, dtype=int),
        "n_items": np.ones(n, dtype=int),
        "n_distinct_products": np.ones(n, dtype=int),
        "n_distinct_sellers": np.ones(n, dtype=int),
        "total_price": np.full(n, 100.0),
        "total_freight": np.full(n, 10.0),
        "total_weight_g": np.full(n, 500.0),
        "max_item_weight_g": np.full(n, 500.0),
        "total_volume_cm3": np.full(n, 1000.0),
        "max_item_volume_cm3": np.full(n, 1000.0),
        "dominant_category": np.array(["cama_mesa_banho", "beleza_saude"])[np.arange(n) % 2],
        "dominant_payment_type": np.array(["credit_card", "boleto"])[np.arange(n) % 2],
        "max_installments": np.full(n, 3),
        "total_payment_value": np.full(n, 110.0),
        "n_payment_methods": np.ones(n, dtype=int),
        "seller_id": [f"s{i % 5:03d}" for i in range(n)],
        "route": np.where(np.arange(n) % 2 == 0, "SP->SP", "RJ->RJ"),
        "seller_order_count_hist": np.full(n, 10.0),
        "seller_late_rate_hist": np.full(n, 0.07),
        "seller_avg_delivery_days_hist": np.full(n, 11.0),
        "seller_avg_handling_days_hist": np.full(n, 2.2),
        "seller_is_new": np.zeros(n, dtype=bool),
        "route_order_count_hist": np.full(n, 50.0),
        "route_late_rate_hist": np.full(n, 0.06),
        "route_avg_delivery_days_hist": np.full(n, 10.0),
        "category_late_rate_hist": np.full(n, 0.065),
        "category_avg_delivery_days_hist": np.full(n, 10.5),
    }
    data.update(overrides)
    return pd.DataFrame(data)


@pytest.fixture
def orders() -> pd.DataFrame:
    return make_orders()


@pytest.fixture
def levels(orders) -> dict[str, tuple[str, ...]]:
    return PreprocessingArtifact._fit_category_levels(orders)


@pytest.fixture
def bounds() -> dict[str, float]:
    return {"days_to_shipping_limit": 30.0}


@pytest.fixture
def snapshots() -> dict[str, pd.DataFrame]:
    """Snapshots covering the months the fixture orders were purchased in."""
    resolved = pd.DataFrame(
        {
            "seller_id": [f"s{i % 5:03d}" for i in range(40)],
            "seller_state": ["SP", "RJ"] * 20,
            "route": ["SP->SP", "RJ->RJ"] * 20,
            "customer_state": ["SP", "RJ", "MG", "BA"] * 10,
            "dominant_category": ["cama_mesa_banho", "beleza_saude"] * 20,
            "order_delivered_customer_date": pd.to_datetime("2017-04-01")
            + pd.to_timedelta(np.arange(40), unit="D"),
            "is_late": [True, False, False, False] * 10,
            "delivery_days": np.full(40, 10.0),
            "handling_days": np.full(40, 2.0),
        }
    )
    months = list(pd.date_range("2017-05-01", "2018-08-01", freq="MS"))
    return build_all_snapshots(resolved, months)


@pytest.fixture
def artifact(orders, snapshots) -> PreprocessingArtifact:
    return PreprocessingArtifact.fit(
        orders,
        snapshots=snapshots,
        zip_centroids=pd.DataFrame(
            {
                "zip_code_prefix": ["01001"],
                "lat": [-23.55],
                "lng": [-46.63],
                "state": ["SP"],
                "n_points": [5],
            }
        ),
        state_centroids=pd.DataFrame(
            {"state": ["SP"], "lat": [-23.55], "lng": [-46.63], "n_points": [100]}
        ),
        train_cutoff=orders["order_purchase_timestamp"].max(),
    )


class TestTheFeatureContract:
    """The §5 inventory, and the §4.2 denylist."""

    def test_thirty_nine_features_in_eight_families(self):
        assert len(FEATURE_NAMES) == 39
        assert len(set(FEATURE_NAMES)) == 39

    def test_every_feature_is_categorical_boolean_or_numeric_exactly_once(self):
        kinds = list(CATEGORICAL_FEATURES) + list(BOOLEAN_FEATURES) + list(NUMERIC_FEATURES)
        assert sorted(kinds) == sorted(FEATURE_NAMES)
        assert len(kinds) == len(set(kinds))

    def test_no_feature_name_is_on_the_denylist(self):
        assert not (DENYLIST & set(FEATURE_NAMES))

    def test_no_required_input_is_on_the_denylist(self):
        assert not (DENYLIST & set(REQUIRED_INPUT_COLUMNS))

    def test_construction_refuses_a_frame_carrying_a_denylist_column(self, orders, levels, bounds):
        """The guard that matters most. A join upstream could pull the delivery date back in,
        and by then nothing else would notice."""
        poisoned = orders.copy()
        poisoned["order_delivered_customer_date"] = pd.Timestamp("2017-07-01")
        with pytest.raises(ValueError, match="denylist"):
            construct_features(poisoned, category_levels=levels, winsor_bounds=bounds)

    def test_construction_refuses_a_frame_missing_an_input(self, orders, levels, bounds):
        with pytest.raises(ValueError, match="missing input columns"):
            construct_features(
                orders.drop(columns=["promised_days"]),
                category_levels=levels,
                winsor_bounds=bounds,
            )

    def test_output_columns_are_exactly_the_feature_names_in_order(self, orders, levels, bounds):
        matrix = construct_features(orders, category_levels=levels, winsor_bounds=bounds)
        assert list(matrix.columns) == list(FEATURE_NAMES)

    def test_construction_does_not_mutate_its_input(self, orders, levels, bounds):
        before = orders.copy()
        construct_features(orders, category_levels=levels, winsor_bounds=bounds)
        pd.testing.assert_frame_equal(orders, before)


class TestDerivedFeatures:
    def test_time_features_come_from_the_purchase_timestamp(self, levels, bounds):
        # 2017-06-03 is a Saturday.
        orders = make_orders(2)
        orders["order_purchase_timestamp"] = pd.to_datetime(
            ["2017-06-03 14:05:00", "2017-06-05 08:00:00"]
        )
        matrix = construct_features(orders, category_levels=levels, winsor_bounds=bounds)
        assert matrix["purchase_hour"].tolist() == [14, 8]
        assert matrix["purchase_dayofweek"].tolist() == [5, 0]
        assert matrix["is_weekend"].tolist() == [True, False]

    def test_purchase_month_is_the_calendar_month_not_the_snapshot_key(self, levels, bounds):
        """`purchase_month` is 1-12 here, a seasonality proxy. `orders_analytical` has a
        column of the same name that is a first-of-month *timestamp* used to join snapshots.
        Confusing the two would put a nanosecond epoch into the model."""
        orders = make_orders(2)
        orders["order_purchase_timestamp"] = pd.to_datetime(["2017-03-15", "2018-11-02"])
        matrix = construct_features(orders, category_levels=levels, winsor_bounds=bounds)
        assert matrix["purchase_month"].tolist() == [3, 11]

    def test_same_state_and_region(self, levels, bounds):
        orders = make_orders(2)
        orders["customer_state"] = ["SP", "BA"]
        orders["seller_state"] = ["SP", "SP"]
        matrix = construct_features(orders, category_levels=levels, winsor_bounds=bounds)
        assert matrix["same_state"].tolist() == [True, False]
        assert matrix["customer_region"].astype(str).tolist() == ["SE", "NE"]

    def test_zip_prefix_2_keeps_the_leading_zero(self, levels, bounds):
        orders = make_orders(2)
        orders["customer_zip_code_prefix"] = ["01001", "99999"]
        matrix = construct_features(orders, category_levels=levels, winsor_bounds=bounds)
        assert matrix["customer_zip_prefix_2"].astype(str).tolist() == ["01", UNKNOWN]

    def test_ratios_guard_division_by_zero_with_nan_not_inf(self, levels, bounds):
        """§5 asks for a price-zero guard. An inf would poison a tree split silently."""
        orders = make_orders(2)
        orders["total_price"] = [100.0, 0.0]
        orders["total_volume_cm3"] = [1000.0, 0.0]
        matrix = construct_features(orders, category_levels=levels, winsor_bounds=bounds)
        assert matrix["freight_ratio"].tolist()[0] == pytest.approx(0.1)
        assert np.isnan(matrix["freight_ratio"].tolist()[1])
        assert np.isnan(matrix["avg_density"].tolist()[1])
        assert not np.isinf(matrix[["freight_ratio", "avg_density"]].to_numpy()).any()

    def test_days_to_shipping_limit_is_winsorised_at_the_fitted_bound(self, levels):
        orders = make_orders(3)
        orders["days_to_shipping_limit"] = [2.0, 25.0, 1052.0]
        matrix = construct_features(
            orders, category_levels=levels, winsor_bounds={"days_to_shipping_limit": 21.22}
        )
        assert matrix["days_to_shipping_limit"].tolist() == pytest.approx([2.0, 21.22, 21.22])

    def test_all_twenty_seven_states_map_to_a_region(self):
        assert len(STATE_TO_REGION) == 27
        assert set(STATE_TO_REGION.values()) == {"N", "NE", "CO", "SE", "S"}


class TestCategoryCapping:
    def test_unseen_and_missing_are_kept_apart(self):
        levels = ("a", "b", MISSING, UNKNOWN)
        capped = cap_categories(pd.Series(["a", "zzz", None]), levels)
        assert list(capped.astype(str)) == ["a", UNKNOWN, MISSING]

    def test_the_level_set_is_exactly_what_was_fitted(self):
        levels = ("a", "b", MISSING, UNKNOWN)
        capped = cap_categories(pd.Series(["a", "zzz", None]), levels)
        assert list(capped.categories) == list(levels)

    def test_product_category_is_capped_and_always_carries_both_sentinels(self):
        orders = make_orders(200)
        orders["dominant_category"] = [f"cat_{i % 90}" for i in range(200)]
        levels = PreprocessingArtifact._fit_category_levels(orders)["product_category"]
        assert len(levels) == CATEGORY_CAP + 2
        assert levels[-2:] == (MISSING, UNKNOWN)

    def test_region_levels_are_the_fixed_five_not_whatever_train_contained(self, orders):
        """A training window with no northern customer must still be able to encode one."""
        sp_only = orders.copy()
        sp_only["customer_state"] = "SP"
        levels = PreprocessingArtifact._fit_category_levels(sp_only)["customer_region"]
        assert set(levels) == {"N", "NE", "CO", "SE", "S", MISSING, UNKNOWN}


class TestFittedOnTrainOnly:
    """§6.6: imputation values and category levels come from training rows only."""

    def test_the_median_used_is_the_training_median_not_the_transformed_frames(
        self, orders, snapshots
    ):
        train = orders.copy()
        train["total_price"] = 100.0
        art = PreprocessingArtifact.fit(
            train,
            snapshots=snapshots,
            zip_centroids=pd.DataFrame(
                {
                    "zip_code_prefix": ["01001"],
                    "lat": [-23.5],
                    "lng": [-46.6],
                    "state": ["SP"],
                    "n_points": [1],
                }
            ),
            state_centroids=pd.DataFrame(
                {"state": ["SP"], "lat": [-23.5], "lng": [-46.6], "n_points": [1]}
            ),
            train_cutoff=train["order_purchase_timestamp"].max(),
        )
        assert art.numeric_medians["total_price"] == 100.0

        # A later frame with wildly different prices and one missing value must still be
        # imputed with 100.0, the training median.
        later = orders.copy()
        later["total_price"] = 9999.0
        later.loc[0, "total_weight_g"] = np.nan
        out = art.transform_with_snapshots(later, snapshots)
        assert out.loc[0, "total_weight_g"] == art.numeric_medians["total_weight_g"]

    def test_a_category_absent_from_training_becomes_unknown(self, artifact, orders, snapshots):
        later = orders.copy()
        later["dominant_payment_type"] = "crypto"
        out = artifact.transform_with_snapshots(later, snapshots)
        assert set(out["payment_type"].astype(str)) == {UNKNOWN}

    def test_fitting_on_an_empty_frame_is_refused(self, orders, snapshots):
        with pytest.raises(ValueError, match="empty training frame"):
            PreprocessingArtifact.fit(
                orders.head(0),
                snapshots=snapshots,
                zip_centroids=pd.DataFrame(),
                state_centroids=pd.DataFrame(),
                train_cutoff=pd.Timestamp("2017-12-31"),
            )

    def test_winsor_bound_comes_from_the_training_quantile(self, artifact, orders):
        expected = float(orders["days_to_shipping_limit"].quantile(WINSOR_QUANTILE))
        assert artifact.winsor_bounds["days_to_shipping_limit"] == pytest.approx(expected)


class TestTransform:
    def test_output_has_no_nulls_after_imputation(self, artifact, orders, snapshots):
        gappy = orders.copy()
        for column in ["total_weight_g", "total_volume_cm3", "seller_late_rate_hist"]:
            gappy.loc[0, column] = np.nan
        out = artifact.transform_with_snapshots(gappy, snapshots)
        assert out[list(NUMERIC_FEATURES)].isna().sum().sum() == 0

    def test_column_order_matches_the_artifacts_stored_list(self, artifact, orders, snapshots):
        out = artifact.transform_with_snapshots(orders, snapshots)
        assert list(out.columns) == list(artifact.feature_names)

    def test_transforming_twice_is_identical(self, artifact, orders, snapshots):
        """§13's requirement, and the precondition for Phase 7's parity test to mean anything."""
        first = artifact.transform_with_snapshots(orders, snapshots)
        second = artifact.transform_with_snapshots(orders, snapshots)
        pd.testing.assert_frame_equal(first, second)

    def test_serving_transform_twice_is_identical(self, artifact, orders):
        pd.testing.assert_frame_equal(artifact.transform(orders), artifact.transform(orders))

    def test_transform_does_not_mutate_its_input(self, artifact, orders, snapshots):
        before = orders.copy()
        artifact.transform_with_snapshots(orders, snapshots)
        artifact.transform(orders)
        pd.testing.assert_frame_equal(orders, before)

    def test_dtypes_are_what_the_tree_libraries_expect(self, artifact, orders, snapshots):
        out = artifact.transform_with_snapshots(orders, snapshots)
        for column in artifact.categorical_features:
            assert isinstance(out[column].dtype, pd.CategoricalDtype), column
        for column in artifact.boolean_features:
            assert out[column].dtype == bool, column
        for column in artifact.numeric_features:
            assert np.issubdtype(out[column].dtype, np.number), column

    def test_row_count_and_order_are_preserved(self, artifact, orders, snapshots):
        out = artifact.transform_with_snapshots(orders, snapshots)
        assert len(out) == len(orders)
        assert out.index.tolist() == orders.index.tolist()

    def test_the_two_paths_differ_only_in_snapshot_selection(self, artifact, orders, snapshots):
        """The §18 A6 claim, asserted.

        Force every order's purchase month to the artifact's bundled snapshot month. The
        training path and the serving path must then agree exactly — which is what "snapshot
        selection is the only difference" means. If they disagree here, the asymmetry is
        wider than A6 says it is.
        """
        aligned = orders.copy()
        aligned["purchase_month"] = artifact.latest_snapshot_month
        training_path = artifact.transform_with_snapshots(aligned, snapshots)
        serving_path = artifact.transform(aligned)
        pd.testing.assert_frame_equal(training_path, serving_path)

    def test_the_two_paths_do_differ_when_months_differ(self, artifact, orders, snapshots):
        """The mirror: if they never differed, the test above would be vacuous."""
        early = orders.copy()
        early["purchase_month"] = pd.Timestamp("2017-05-01")
        training_path = artifact.transform_with_snapshots(early, snapshots)
        serving_path = artifact.transform(early)
        assert not training_path["seller_order_count_hist"].equals(
            serving_path["seller_order_count_hist"]
        )


class TestPersistence:
    def test_save_and_load_round_trip_transforms_identically(
        self, artifact, orders, snapshots, tmp_path
    ):
        before = artifact.transform_with_snapshots(orders, snapshots)
        path = artifact.save(tmp_path / "a.joblib")
        reloaded = PreprocessingArtifact.load(path)
        pd.testing.assert_frame_equal(reloaded.transform_with_snapshots(orders, snapshots), before)

    def test_load_preserves_every_fitted_quantity(self, artifact, tmp_path):
        reloaded = PreprocessingArtifact.load(artifact.save(tmp_path / "a.joblib"))
        assert reloaded.numeric_medians == artifact.numeric_medians
        assert reloaded.category_levels == artifact.category_levels
        assert reloaded.winsor_bounds == artifact.winsor_bounds
        assert reloaded.feature_names == artifact.feature_names
        assert reloaded.train_cutoff == artifact.train_cutoff
        assert reloaded.latest_snapshot_month == artifact.latest_snapshot_month

    def test_load_rejects_a_different_artifact_version(self, artifact, tmp_path):
        import dataclasses

        import joblib

        stale = dataclasses.replace(artifact, version="0")
        path = tmp_path / "stale.joblib"
        joblib.dump(stale, path)
        with pytest.raises(ValueError, match="artifact version"):
            PreprocessingArtifact.load(path)

    def test_load_rejects_a_changed_feature_contract(self, artifact, tmp_path):
        """A hash mismatch means the code and the artifact disagree about the features —
        precisely the skew this class exists to prevent, so it must not load."""
        import dataclasses

        import joblib

        path = tmp_path / "drifted.joblib"
        joblib.dump(dataclasses.replace(artifact, input_schema_hash="deadbeef"), path)
        with pytest.raises(ValueError, match="input schema hash"):
            PreprocessingArtifact.load(path)

    def test_the_schema_hash_is_stable_and_short(self):
        assert input_schema_hash() == input_schema_hash()
        assert len(input_schema_hash()) == 16


class TestImmutability:
    """§6.2: "give it transform() and no public fit_transform() after serialization"."""

    def test_there_is_no_fit_transform(self):
        assert not hasattr(PreprocessingArtifact, "fit_transform")

    def test_fit_is_a_classmethod_so_an_instance_cannot_refit_itself(self):
        assert isinstance(
            PreprocessingArtifact.__dict__["fit"], classmethod
        ), "fit must be a classmethod: an instance that can refit can drift from what was logged"

    def test_the_artifact_is_frozen(self, artifact):
        with pytest.raises(Exception):  # noqa: B017 - dataclasses raises FrozenInstanceError
            artifact.threshold = 0.5

    def test_with_decision_returns_a_new_artifact(self, artifact):
        updated = artifact.with_decision(calibrator="fake", threshold=0.23)
        assert updated is not artifact
        assert artifact.threshold is None, "the original artifact was mutated"
        assert updated.threshold == 0.23
        assert updated.calibrator == "fake"

    def test_with_decision_rejects_an_out_of_range_threshold(self, artifact):
        for bad in (0.0, 1.0, -0.1, 1.5):
            with pytest.raises(ValueError, match="threshold must be"):
                artifact.with_decision(calibrator="fake", threshold=bad)

    def test_summary_is_json_serializable(self, artifact):
        import json

        assert json.loads(json.dumps(artifact.summary()))["n_features"] == 39


class TestAgainstTheRealData:
    pytestmark: ClassVar = [pytest.mark.integration, pytest.mark.slow]

    @pytest.fixture(scope="class")
    @staticmethod
    def built():
        """Built once per class: fitting the artifact and transforming 96,203 rows is ~4 s.

        A staticmethod because pytest removes class-scoped fixtures defined as instance
        methods in pytest 10 — each test gets a fresh instance while the fixture runs once,
        so `self` was never meaningful here.
        """
        from src.features.build import build_matrix

        return build_matrix("v1")

    def test_matrix_shape_and_no_nulls(self, built):
        matrix = built["matrix"]
        assert matrix.shape == (96_203, 39)
        assert matrix[list(NUMERIC_FEATURES)].isna().sum().sum() == 0

    def test_no_denylist_column_in_the_matrix(self, built):
        assert not (DENYLIST & set(built["matrix"].columns))

    def test_artifact_was_fitted_on_the_v1_window_only(self, built):
        artifact = built["artifact"]
        assert artifact.train_rows == 36_174
        assert artifact.train_cutoff < pd.Timestamp("2018-01-01")
        assert artifact.version == ARTIFACT_VERSION

    def test_category_levels_are_bounded(self, built):
        levels = built["artifact"].category_levels
        assert len(levels["product_category"]) == CATEGORY_CAP + 2
        assert len(levels["customer_region"]) == 7
        # Sellers are concentrated: fewer seller states appear in training than customer
        # states, so an order from an unusual seller state legitimately encodes as unknown.
        assert len(levels["seller_state"]) < len(levels["customer_state"])

    def test_the_real_matrix_transforms_identically_twice(self, built):
        artifact, orders, snapshots = built["artifact"], built["orders"], built["snapshots"]
        sample = orders.head(500)
        first = artifact.transform_with_snapshots(sample, snapshots)
        second = artifact.transform_with_snapshots(sample, snapshots)
        pd.testing.assert_frame_equal(first, second)

    def test_labels_align_with_the_matrix(self, built):
        assert len(built["labels"]) == len(built["matrix"])
        assert float(built["labels"].mean()) == pytest.approx(0.0679, abs=1e-3)

    def test_the_bundled_snapshot_is_the_latest_month(self, built):
        artifact = built["artifact"]
        assert artifact.latest_snapshot_month == pd.Timestamp("2018-08-01")
        for spec in ENTITIES:
            frame = artifact.latest_snapshots[spec.table]
            assert (frame["snapshot_month"] == artifact.latest_snapshot_month).all()
        assert GLOBAL_LEVEL.table in artifact.latest_snapshots

    def test_only_warm_up_rows_lack_history(self, built):
        """The 750 pre-imputation nulls in the history features are exactly 2017-01's orders:
        that month has no resolved history at any level, which is why the warm-up exists."""
        from src.features.history import attach_history

        attached = attach_history(built["orders"], built["snapshots"])
        no_history = attached["seller_late_rate_hist"].isna()
        assert int(no_history.sum()) == 750
        assert (attached.loc[no_history, "purchase_month"] == pd.Timestamp("2017-01-01")).all()
