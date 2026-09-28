"""Tests for the analytical table and the geolocation reduction (PLAN.md §13 Phase 2).

Ranked by the damage the failure would do:

1. A target computed at timestamp granularity — silently inflates the positive rate from
   6.79% to 8.13% and poisons every downstream metric (§4.3).
2. A §4.2 denylist column reaching the feature-safe table.
3. A fan-out join producing more than one row per order. Note that the positive-rate band
   check does **not** catch this: measured on the item-level fan-out the rate is 6.61%,
   still inside the band. Only the row-count test catches it.
4. A mean instead of a median zip centroid, which puts some Brazilian orders in Portugal.
5. The DDL and the Pandera contract drifting apart, which would let a new column skip
   validation entirely.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
from pandera.errors import SchemaErrors
from sqlalchemy import text

from src.etl.build_orders import (
    ANALYTICAL_COLUMNS,
    ANALYTICAL_TABLE,
    COLUMN_NAMES,
    OUTCOMES_TABLE,
    add_distance,
    analysis_window,
    build,
    measure_funnel,
    write_funnel_report,
)
from src.etl.geolocation import LAT_MAX, LAT_MIN, LNG_MAX, LNG_MIN, haversine_km
from src.etl.schema import DENYLIST, ORDERS_ANALYTICAL_SCHEMA

# Measured on the real population of 96,203 orders.
EXPECTED_ROWS = 96_203
EXPECTED_RATE = 0.0679
# What the timestamp-granularity bug would produce. Asserted against so the test fails if
# anyone "fixes" the target by removing the date casts.
BUGGY_TIMESTAMP_RATE = 0.0813


@pytest.fixture(scope="session")
def analytical(engine) -> pd.DataFrame:
    """The built table, read once."""
    with engine.connect() as conn:
        exists = conn.execute(
            text(
                "select count(*) from information_schema.tables "
                "where table_schema='features' and table_name='orders_analytical'"
            )
        ).scalar_one()
        if not exists:
            pytest.skip(f"{ANALYTICAL_TABLE} not built; run `make build-orders` first")
        return pd.read_sql(text(f"select * from {ANALYTICAL_TABLE}"), conn)


class TestHaversine:
    """Unit tests. No Postgres, no dataset — these run in CI."""

    # Great-circle distance for these exact coordinates on a sphere of radius 6371.0088 km.
    # Cross-checked by hand: dlat 0.6437 deg = 71.6 km, dlng 3.4604 deg at latitude -23.2
    # = 353.6 km, and sqrt(71.6^2 + 353.6^2) = 360.8.
    SAO_PAULO: ClassVar = (-23.5505, -46.6333)
    RIO: ClassVar = (-22.9068, -43.1729)
    MANAUS: ClassVar = (-3.1190, -60.0217)

    def test_sao_paulo_to_rio(self):
        got = float(haversine_km(*self.SAO_PAULO, *self.RIO))
        assert got == pytest.approx(360.7, abs=1.0)

    def test_sao_paulo_to_manaus(self):
        got = float(haversine_km(*self.SAO_PAULO, *self.MANAUS))
        assert got == pytest.approx(2689.5, abs=2.0)

    def test_identical_points_are_zero(self):
        assert float(haversine_km(*self.SAO_PAULO, *self.SAO_PAULO)) == pytest.approx(0.0)

    def test_symmetric(self):
        there = float(haversine_km(*self.SAO_PAULO, *self.RIO))
        back = float(haversine_km(*self.RIO, *self.SAO_PAULO))
        assert there == pytest.approx(back)

    def test_one_degree_at_the_equator(self):
        """A degree of great circle is 2*pi*R/360 = 111.195 km. Catches a radians/degrees mixup."""
        assert float(haversine_km(0.0, 0.0, 0.0, 1.0)) == pytest.approx(111.195, abs=0.01)

    def test_one_degree_of_latitude_matches_one_degree_of_longitude_at_the_equator(self):
        lat = float(haversine_km(0.0, 0.0, 1.0, 0.0))
        lng = float(haversine_km(0.0, 0.0, 0.0, 1.0))
        assert lat == pytest.approx(lng, abs=1e-9)

    def test_antipodal_is_half_the_circumference(self):
        assert float(haversine_km(0.0, 0.0, 0.0, 180.0)) == pytest.approx(20015.0, abs=1.0)

    def test_broadcasts_over_arrays(self):
        got = haversine_km(
            np.array([-23.5505, 0.0]),
            np.array([-46.6333, 0.0]),
            np.array([-22.9068, 0.0]),
            np.array([-43.1729, 1.0]),
        )
        assert got.shape == (2,)
        assert got[0] == pytest.approx(360.7, abs=1.0)
        assert got[1] == pytest.approx(111.195, abs=0.01)

    def test_nan_propagates_rather_than_inventing_a_distance(self):
        assert np.isnan(float(haversine_km(np.nan, -46.6, -22.9, -43.2)))


class TestContract:
    """Unit tests on the Pandera schema itself."""

    def test_schema_covers_exactly_the_ddl_columns(self):
        """The DDL and the contract must not drift.

        If a column can be added to the table without being added to the schema, it skips
        validation, and the denylist guarantee becomes decorative.
        """
        assert set(ORDERS_ANALYTICAL_SCHEMA.columns) == set(COLUMN_NAMES)
        assert len(ANALYTICAL_COLUMNS) == len(COLUMN_NAMES)

    def test_schema_is_strict(self):
        assert ORDERS_ANALYTICAL_SCHEMA.strict is True

    def test_no_declared_column_is_on_the_denylist(self):
        assert not (DENYLIST & set(COLUMN_NAMES))

    def test_denylist_names_the_columns_section_4_2_forbids(self):
        for name in (
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_status",
        ):
            assert name in DENYLIST
        assert any(n.startswith("review") for n in DENYLIST)

    def test_strictness_rejects_a_smuggled_denylist_column(self, minimal_frame):
        frame = minimal_frame.copy()
        frame["order_status"] = "delivered"
        with pytest.raises(SchemaErrors) as info:
            ORDERS_ANALYTICAL_SCHEMA.validate(frame, lazy=True)
        assert "order_status" in str(info.value)

    def test_rejects_an_out_of_band_positive_rate(self, minimal_frame):
        """A 45% rate must not validate: it would mean the target is definitionally wrong."""
        frame = minimal_frame.copy()
        frame["is_late"] = [True] * len(frame)
        with pytest.raises(SchemaErrors) as info:
            ORDERS_ANALYTICAL_SCHEMA.validate(frame, lazy=True)
        assert "positive_rate_in_band" in str(info.value)

    def test_rejects_duplicate_order_ids(self, minimal_frame):
        frame = pd.concat([minimal_frame, minimal_frame.head(1)], ignore_index=True)
        with pytest.raises(SchemaErrors):
            ORDERS_ANALYTICAL_SCHEMA.validate(frame, lazy=True)

    def test_rejects_a_coordinate_outside_brazil(self, minimal_frame):
        frame = minimal_frame.copy()
        frame.loc[0, "customer_lat"] = 41.15  # the Portugal geocoding failure
        with pytest.raises(SchemaErrors) as info:
            ORDERS_ANALYTICAL_SCHEMA.validate(frame, lazy=True)
        assert "customer_lat" in str(info.value)

    def test_accepts_the_minimal_valid_frame(self, minimal_frame):
        ORDERS_ANALYTICAL_SCHEMA.validate(minimal_frame, lazy=True)


class TestAddDistance:
    """Unit tests for the one pandas step in the build."""

    def test_adds_the_column_without_mutating_the_input(self, minimal_frame):
        staged = minimal_frame.drop(columns=["customer_seller_distance_km"])
        out = add_distance(staged)
        assert "customer_seller_distance_km" not in staged.columns
        assert "customer_seller_distance_km" in out.columns

    def test_distance_matches_haversine(self, minimal_frame):
        staged = minimal_frame.drop(columns=["customer_seller_distance_km"])
        out = add_distance(staged)
        expected = haversine_km(
            staged["customer_lat"],
            staged["customer_lng"],
            staged["seller_lat"],
            staged["seller_lng"],
        )
        np.testing.assert_allclose(out["customer_seller_distance_km"].to_numpy(), expected)

    def test_missing_coordinates_give_nan_not_zero(self, minimal_frame):
        staged = minimal_frame.drop(columns=["customer_seller_distance_km"]).copy()
        staged.loc[0, "seller_lat"] = np.nan
        out = add_distance(staged)
        assert np.isnan(out.loc[0, "customer_seller_distance_km"])


class TestAnalysisWindow:
    """Unit test: the inclusive/exclusive boundary is easy to get wrong by a whole day."""

    def test_end_bound_is_exclusive_and_one_day_past_the_config(self):
        start, end_exclusive = analysis_window()
        assert str(start) == "2017-01-01"
        # configs/splits.yaml says end 2018-08-31 inclusive.
        assert str(end_exclusive) == "2018-09-01"


class TestFunnelReport:
    """Unit tests on the report writer."""

    def test_renders_steps_and_drops(self, tmp_path):
        funnel = [("all raw orders", 100), ("status = 'delivered'", 90), ("+ in window", 80)]
        path = tmp_path / "funnel.md"
        write_funnel_report(funnel, 0.068, {"zip_centroids": 19011, "state_centroids": 27}, path)
        body = path.read_text(encoding="utf-8")
        assert "all raw orders" in body
        assert "| 10 |" in body  # 100 -> 90
        assert "19,011" in body
        assert "6.80%" in body


class TestAnalyticalTable:
    """Integration: the table as actually built."""

    pytestmark: ClassVar = [pytest.mark.integration, pytest.mark.slow]

    def test_exactly_one_row_per_order(self, analytical):
        assert len(analytical) == EXPECTED_ROWS
        assert analytical["order_id"].nunique() == EXPECTED_ROWS

    def test_positive_rate_in_band(self, analytical):
        rate = float(analytical["is_late"].mean())
        assert 0.05 <= rate <= 0.10
        assert rate == pytest.approx(EXPECTED_RATE, abs=1e-4)

    def test_positive_rate_is_not_the_timestamp_granularity_rate(self, analytical):
        """Pins the §4.3 date cast. The buggy value is inside no plausible tolerance of the
        correct one, so this fails loudly if the casts are removed."""
        rate = float(analytical["is_late"].mean())
        assert abs(rate - BUGGY_TIMESTAMP_RATE) > 0.01

    def test_no_denylist_column_in_the_table(self, analytical):
        assert not (DENYLIST & set(analytical.columns))

    def test_table_columns_are_exactly_the_declared_columns(self, analytical):
        assert set(analytical.columns) == set(COLUMN_NAMES)

    def test_the_built_table_satisfies_its_own_contract(self, analytical):
        ORDERS_ANALYTICAL_SCHEMA.validate(analytical, lazy=True)

    def test_purchase_month_is_the_first_of_the_month(self, analytical):
        assert (analytical["purchase_month"].dt.day == 1).all()
        assert (
            analytical["purchase_month"].dt.to_period("M")
            == analytical["order_purchase_timestamp"].dt.to_period("M")
        ).all()

    def test_every_order_is_inside_the_analysis_window(self, analytical):
        start, end_exclusive = analysis_window()
        purchased = analytical["order_purchase_timestamp"]
        assert purchased.min() >= pd.Timestamp(start)
        assert purchased.max() < pd.Timestamp(end_exclusive)

    def test_route_is_two_states_joined(self, analytical):
        assert (analytical["route"].str.len() == 6).all()
        rebuilt = analytical["seller_state"] + "->" + analytical["customer_state"]
        assert (analytical["route"] == rebuilt).all()

    def test_coordinates_are_inside_brazil(self, analytical):
        for column, lo, hi in (
            ("customer_lat", LAT_MIN, LAT_MAX),
            ("seller_lat", LAT_MIN, LAT_MAX),
            ("customer_lng", LNG_MIN, LNG_MAX),
            ("seller_lng", LNG_MIN, LNG_MAX),
        ):
            values = analytical[column].dropna()
            assert values.between(lo, hi).all(), f"{column} escaped the bounds filter"

    def test_distance_is_resolved_for_every_order(self, analytical):
        assert analytical["customer_seller_distance_km"].notna().all()
        assert (analytical["customer_seller_distance_km"] >= 0).all()

    def test_same_state_orders_are_much_closer_than_cross_state_orders(self, analytical):
        """A real signal that the distance means something, not just that it computes."""
        same = analytical["seller_state"] == analytical["customer_state"]
        assert analytical.loc[same, "customer_seller_distance_km"].mean() < 300
        assert analytical.loc[~same, "customer_seller_distance_km"].mean() > 600

    def test_the_state_centroid_fallback_was_actually_used(self, analytical):
        """157 customer and 7 seller prefixes are absent from raw.geolocation, so a build
        where the fallback never fires means the fallback is broken, not unnecessary."""
        assert (analytical["customer_geo_source"] == "state").sum() > 0
        assert (analytical["seller_geo_source"] == "state").sum() > 0
        assert (analytical["customer_geo_source"] == "none").sum() == 0
        assert (analytical["seller_geo_source"] == "none").sum() == 0

    def test_aggregates_are_internally_consistent(self, analytical):
        assert (analytical["n_distinct_products"] <= analytical["n_items"]).all()
        assert (analytical["n_distinct_sellers"] <= analytical["n_items"]).all()
        assert (analytical["n_seller_states"] <= analytical["n_distinct_sellers"]).all()
        assert (analytical["max_price"] <= analytical["total_price"] + 1e-9).all()

    def test_shipping_limit_never_precedes_purchase(self, analytical):
        """§18 A4's evidence, asserted: the minimum offset is a hard 2.00 days."""
        assert (analytical["days_to_shipping_limit"] >= 2.0 - 1e-9).all()


class TestOutcomesQuarantine:
    """`order_delivered_customer_date` must exist for §4.5 but only in the outcome table."""

    pytestmark: ClassVar = [pytest.mark.integration, pytest.mark.slow]

    def test_outcomes_table_holds_exactly_the_quarantined_columns(self, engine):
        """The quarantine table's column list is a deliberate, closed set.

        It carries the post-purchase facts the §4.5 as-of aggregates need about *already
        resolved* orders, and nothing else. `order_delivered_carrier_date`, `delivery_days`
        and `handling_days` were added for `avg_handling_days` and `avg_delivery_days`
        (PLAN.md §5 F6/F7). Asserting equality rather than containment is the point: a new
        post-purchase column cannot be parked here without this test being changed in the
        open.
        """
        with engine.connect() as conn:
            columns = {
                row[0]
                for row in conn.execute(
                    text(
                        "select column_name from information_schema.columns "
                        "where table_schema='features' and table_name='order_outcomes'"
                    )
                )
            }
        assert columns == {
            "order_id",
            "order_delivered_customer_date",
            "order_delivered_carrier_date",
            "is_late",
            "delivery_days",
            "handling_days",
        }

    def test_handling_days_excludes_self_contradictory_rows_rather_than_clipping(self, engine):
        """165 orders record a carrier handoff before their own purchase (down to -171 days)
        and 19 record it after customer delivery. Those are recording errors, so
        `handling_days` is NULL; clipping to 0 would make the seller look instant."""
        with engine.connect() as conn:
            total, nulls, minimum = conn.execute(
                text(
                    f"select count(*), count(*) filter (where handling_days is null), "
                    f"min(handling_days) from {OUTCOMES_TABLE}"
                )
            ).one()
        assert total == EXPECTED_ROWS
        assert nulls == 185  # 1 missing carrier date + 184 contradictory
        assert float(minimum) >= 0.0

    def test_delivery_days_is_always_positive_and_never_null(self, engine):
        with engine.connect() as conn:
            nulls, minimum = conn.execute(
                text(
                    f"select count(*) filter (where delivery_days is null), min(delivery_days) "
                    f"from {OUTCOMES_TABLE}"
                )
            ).one()
        assert nulls == 0
        assert float(minimum) > 0.0

    def test_outcomes_row_count_matches_the_analytical_table(self, engine):
        with engine.connect() as conn:
            analytical = conn.execute(text(f"select count(*) from {ANALYTICAL_TABLE}")).scalar_one()
            outcomes = conn.execute(text(f"select count(*) from {OUTCOMES_TABLE}")).scalar_one()
        assert analytical == outcomes == EXPECTED_ROWS

    def test_the_two_tables_agree_on_the_label(self, engine):
        with engine.connect() as conn:
            mismatches = conn.execute(
                text(
                    f"select count(*) from {ANALYTICAL_TABLE} a "
                    f"join {OUTCOMES_TABLE} o using (order_id) where a.is_late <> o.is_late"
                )
            ).scalar_one()
        assert mismatches == 0

    def test_every_outcome_resolves_after_its_purchase(self, engine):
        """Sanity on the column §4.5 will filter on: a delivery cannot precede its order."""
        with engine.connect() as conn:
            impossible = conn.execute(
                text(
                    f"select count(*) from {OUTCOMES_TABLE} o "
                    f"join {ANALYTICAL_TABLE} a using (order_id) "
                    "where o.order_delivered_customer_date < a.order_purchase_timestamp"
                )
            ).scalar_one()
        assert impossible == 0


class TestFunnel:
    """The funnel arithmetic, against the live raw schema."""

    pytestmark: ClassVar = [pytest.mark.integration, pytest.mark.slow]

    def test_funnel_is_monotonically_decreasing(self, engine):
        start, end_exclusive = analysis_window()
        with engine.connect() as conn:
            funnel = measure_funnel(
                conn, {"window_start": start, "window_end_exclusive": end_exclusive}
            )
        counts = [count for _, count in funnel]
        assert counts == sorted(counts, reverse=True)

    def test_funnel_drops_are_the_measured_ones(self, engine):
        start, end_exclusive = analysis_window()
        with engine.connect() as conn:
            funnel = measure_funnel(
                conn, {"window_start": start, "window_end_exclusive": end_exclusive}
            )
        assert [count for _, count in funnel] == [99_441, 96_478, 96_470, 96_203]

    def test_the_final_funnel_step_equals_the_table(self, engine, analytical):
        start, end_exclusive = analysis_window()
        with engine.connect() as conn:
            funnel = measure_funnel(
                conn, {"window_start": start, "window_end_exclusive": end_exclusive}
            )
        assert funnel[-1][1] == len(analytical)


class TestIdempotence:
    """A rebuild must produce the same table."""

    pytestmark: ClassVar = [pytest.mark.integration, pytest.mark.slow]

    def test_rebuild_reproduces_the_same_counts_and_rate(self, analytical):
        before_rows = len(analytical)
        before_rate = float(analytical["is_late"].mean())
        summary = build()
        assert summary["rows"] == before_rows
        assert summary["outcome_rows"] == before_rows
        assert float(summary["positive_rate"]) == pytest.approx(before_rate, abs=1e-12)

    def test_dominant_picks_are_deterministic(self, engine, analytical):
        """Ties are broken alphabetically, so a rebuild cannot reshuffle multi-seller orders."""
        before = analytical.set_index("order_id")[
            ["seller_id", "dominant_category", "dominant_payment_type"]
        ].sort_index()
        build()
        with engine.connect() as conn:
            after = (
                pd.read_sql(
                    text(
                        "select order_id, seller_id, dominant_category, dominant_payment_type "
                        f"from {ANALYTICAL_TABLE}"
                    ),
                    conn,
                )
                .set_index("order_id")
                .sort_index()
            )
        pd.testing.assert_frame_equal(before, after)
