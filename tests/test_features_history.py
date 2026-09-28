"""Leakage tests for the as-of monthly snapshots (PLAN.md §13 Phase 3 part A, §4.5, §18 A1).

**These are the tests the whole project rests on.** If a snapshot for month *M* can see an
outcome that was not yet known at *M*, every metric downstream is optimistic and no amount of
careful modelling afterwards recovers it.

The rule, stated once: a snapshot for month *M* aggregates only orders whose
``order_delivered_customer_date < M``. Not ``order_purchase_timestamp < M``.

Why that distinction is not pedantic, measured on the real data:

| Snapshot *M* | resolved by *M* | late rate | still in flight at *M* | late rate | ratio |
|---|---:|---:|---:|---:|---:|
| 2017-07 | 12,839 | 3.24% | 1,093 | 11.25% | 3.5x |
| 2018-01 | 40,930 | 4.30% | 2,496 | 27.68% | 6.4x |
| 2018-04 | 60,201 | 5.99% | 3,852 | 39.10% | 6.5x |

Orders still in flight at a cutoff are **3.5-6.5x more likely to end up late**, because late
orders take longer to resolve — that is what makes them late. The purchase-time rule admits
exactly that population, and it can only know their outcome by reading the future. At the first
training month, 2017-05, the leaky global late rate is 4.66% against a true 2.37%: nearly
double. The distortion is worst where history is thinnest, which is also where each observation
carries the most weight.

Most tests here are synthetic and need no database, because a hand-built entity with a known
history is the only way to prove the boundary is exactly right rather than approximately right.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text

from src.features.history import (
    ENTITIES,
    GLOBAL_LEVEL,
    METRIC_COLUMNS,
    attach_history,
    build_all_snapshots,
    build_snapshots,
    snapshot_months,
    write_snapshots,
)

RESOLVED_COLUMNS = [
    "seller_id",
    "seller_state",
    "route",
    "customer_state",
    "dominant_category",
    "order_delivered_customer_date",
    "is_late",
    "delivery_days",
    "handling_days",
]


def resolved(rows: list[dict]) -> pd.DataFrame:
    """Build a resolved-orders frame, filling unspecified columns with harmless defaults."""
    defaults = {
        "seller_id": "s1",
        "seller_state": "SP",
        "route": "SP->SP",
        "customer_state": "SP",
        "dominant_category": "cat_a",
        "is_late": False,
        "delivery_days": 10.0,
        "handling_days": 2.0,
    }
    frame = pd.DataFrame([{**defaults, **row} for row in rows])
    frame["order_delivered_customer_date"] = pd.to_datetime(frame["order_delivered_customer_date"])
    return frame[RESOLVED_COLUMNS]


def orders(rows: list[dict]) -> pd.DataFrame:
    """Build an orders frame to attach history to."""
    defaults = {
        "seller_id": "s1",
        "seller_state": "SP",
        "route": "SP->SP",
        "customer_state": "SP",
        "dominant_category": "cat_a",
    }
    frame = pd.DataFrame([{**defaults, **row} for row in rows])
    frame["purchase_month"] = pd.to_datetime(frame["purchase_month"])
    return frame


class TestTheLeakageBoundary:
    """Test 1 from the plan: no contributing order is resolved on or after *M*."""

    def test_an_order_resolved_on_the_first_of_the_month_is_excluded(self):
        """The boundary is strict. ``< M``, never ``<= M``.

        An order delivered at 00:00:00 on the first of *M* was not known before *M*, so a
        model predicting on the first of *M* could not have seen it.
        """
        history = resolved(
            [
                {"order_delivered_customer_date": "2017-03-31 23:59:59", "is_late": False},
                {"order_delivered_customer_date": "2017-04-01 00:00:00", "is_late": True},
            ]
        )
        snap = build_snapshots(history, [pd.Timestamp("2017-04-01")], key="seller_id")
        assert snap["order_count"].tolist() == [1]
        assert snap["late_rate"].tolist() == [0.0]

    def test_purchase_time_is_never_consulted(self):
        """The frame the builder receives has no purchase column at all.

        Stated as a test so the contract cannot be widened by accident: if a future change
        needs purchase time, it has to add the column, and this test is where the argument
        has to be made.
        """
        history = resolved([{"order_delivered_customer_date": "2017-02-01"}])
        assert "order_purchase_timestamp" not in history.columns
        snap = build_snapshots(history, [pd.Timestamp("2017-03-01")], key="seller_id")
        assert snap["order_count"].tolist() == [1]

    def test_an_order_resolved_after_the_month_never_contributes_to_it(self):
        history = resolved(
            [
                {"order_delivered_customer_date": "2017-01-15", "is_late": True},
                {"order_delivered_customer_date": "2017-06-15", "is_late": False},
            ]
        )
        snap = build_snapshots(
            history, [pd.Timestamp("2017-02-01"), pd.Timestamp("2017-07-01")], key="seller_id"
        )
        by_month = snap.set_index("snapshot_month")
        assert by_month.loc[pd.Timestamp("2017-02-01"), "order_count"] == 1
        assert by_month.loc[pd.Timestamp("2017-02-01"), "late_rate"] == 1.0
        assert by_month.loc[pd.Timestamp("2017-07-01"), "order_count"] == 2
        assert by_month.loc[pd.Timestamp("2017-07-01"), "late_rate"] == 0.5

    def test_the_earliest_month_has_no_history_at_all(self):
        """2017-01 genuinely has zero resolved orders, so its snapshot must be empty rather
        than zero-filled. A zero-filled row would look like a seller with a 0% late rate."""
        history = resolved([{"order_delivered_customer_date": "2017-05-01"}])
        snap = build_snapshots(history, [pd.Timestamp("2017-01-01")], key="seller_id")
        assert snap.empty

    def test_snapshots_are_expanding_not_rolling(self):
        """order_count for a fixed entity must never decrease as *M* advances."""
        history = resolved(
            [{"order_delivered_customer_date": f"2017-0{m}-15"} for m in range(1, 7)]
        )
        months = [pd.Timestamp(f"2017-{m:02d}-01") for m in range(2, 9)]
        snap = build_snapshots(history, months, key="seller_id").sort_values("snapshot_month")
        counts = snap["order_count"].tolist()
        assert counts == sorted(counts)
        assert counts[-1] == 6


class TestTheSharpChangeSeller:
    """Test 2 from the plan, the one that would catch a blended window."""

    def test_a_seller_who_turns_bad_in_month_four_still_reads_clean_at_month_four(self):
        """late_rate 0.0 across months 1-3, then 1.0 in month 4.

        The month-4 snapshot must report **0.0**. Reporting 0.25 would mean month 4's own
        resolved order leaked into the snapshot a month-4 prediction would use.
        """
        history = resolved(
            [
                {"order_delivered_customer_date": "2017-01-10", "is_late": False},
                {"order_delivered_customer_date": "2017-02-10", "is_late": False},
                {"order_delivered_customer_date": "2017-03-10", "is_late": False},
                {"order_delivered_customer_date": "2017-04-10", "is_late": True},
            ]
        )
        snap = build_snapshots(history, [pd.Timestamp("2017-04-01")], key="seller_id")
        assert snap["order_count"].tolist() == [3]
        assert snap["late_rate"].tolist() == [0.0], "month 4's own outcome leaked in"

    def test_the_change_does_show_up_the_following_month(self):
        """The mirror image: the information must arrive, just not early."""
        history = resolved(
            [
                {"order_delivered_customer_date": "2017-01-10", "is_late": False},
                {"order_delivered_customer_date": "2017-02-10", "is_late": False},
                {"order_delivered_customer_date": "2017-03-10", "is_late": False},
                {"order_delivered_customer_date": "2017-04-10", "is_late": True},
            ]
        )
        snap = build_snapshots(history, [pd.Timestamp("2017-05-01")], key="seller_id")
        assert snap["order_count"].tolist() == [4]
        assert snap["late_rate"].tolist() == [0.25]


class TestColdStart:
    """Tests 3 and 4 from the plan: flag, never impute silently."""

    def test_a_sellers_first_ever_order_is_flagged_new_and_gets_the_state_fallback(self):
        history = resolved(
            [
                {
                    "seller_id": "established",
                    "seller_state": "SP",
                    "order_delivered_customer_date": "2017-01-10",
                    "is_late": True,
                    "delivery_days": 30.0,
                }
            ]
        )
        snaps = build_all_snapshots(history, [pd.Timestamp("2017-02-01")])
        attached = attach_history(
            orders([{"order_id": "o1", "purchase_month": "2017-02-01", "seller_id": "brand_new"}]),
            snaps,
        )
        row = attached.iloc[0]
        assert bool(row["seller_is_new"]) is True
        # Falls back to SP, whose only resolved order was late.
        assert row["seller_late_rate_hist"] == 1.0
        assert row["seller_avg_delivery_days_hist"] == 30.0

    def test_an_established_seller_is_not_flagged_new(self):
        history = resolved(
            [{"seller_id": "s1", "order_delivered_customer_date": "2017-01-10", "is_late": False}]
        )
        snaps = build_all_snapshots(history, [pd.Timestamp("2017-02-01")])
        attached = attach_history(
            orders([{"order_id": "o1", "purchase_month": "2017-02-01", "seller_id": "s1"}]), snaps
        )
        assert bool(attached.iloc[0]["seller_is_new"]) is False
        assert attached.iloc[0]["seller_late_rate_hist"] == 0.0

    def test_the_global_fallback_is_used_when_the_state_is_also_empty(self):
        """Test 4: a new seller in a state with no resolved history at all."""
        history = resolved(
            [
                {
                    "seller_id": "s_sp",
                    "seller_state": "SP",
                    "order_delivered_customer_date": "2017-01-10",
                    "is_late": True,
                    "delivery_days": 20.0,
                }
            ]
        )
        snaps = build_all_snapshots(history, [pd.Timestamp("2017-02-01")])
        attached = attach_history(
            orders(
                [
                    {
                        "order_id": "o1",
                        "purchase_month": "2017-02-01",
                        "seller_id": "s_am",
                        "seller_state": "AM",  # no resolved history in AM
                    }
                ]
            ),
            snaps,
        )
        row = attached.iloc[0]
        assert bool(row["seller_is_new"]) is True
        # AM has nothing, so the global level answers: the one resolved order, which was late.
        assert row["seller_late_rate_hist"] == 1.0
        assert row["seller_avg_delivery_days_hist"] == 20.0

    def test_a_cold_entitys_own_order_count_is_zero_not_the_fallbacks_count(self):
        """The fallback supplies *rates*, not volumes.

        A new seller in Sao Paulo should read "0 orders of my own, and SP's late rate is my
        best guess". Inheriting SP's order_count would tell the model this brand-new seller
        has thousands of deliveries behind it, which is the opposite of true and would make
        `order_count` an anti-signal exactly where `is_new` is trying to help.
        """
        history = resolved(
            [
                {
                    "seller_id": "established",
                    "order_delivered_customer_date": "2017-01-10",
                    "is_late": True,
                }
            ]
        )
        snaps = build_all_snapshots(history, [pd.Timestamp("2017-02-01")])
        attached = attach_history(
            orders([{"order_id": "o1", "purchase_month": "2017-02-01", "seller_id": "new"}]),
            snaps,
        )
        row = attached.iloc[0]
        assert row["seller_order_count_hist"] == 0
        assert row["seller_late_rate_hist"] == 1.0
        assert bool(row["seller_is_new"]) is True

    def test_nothing_resolved_anywhere_yields_nan_not_a_fabricated_zero(self):
        """The first month of the project has no history at any level.

        NaN is the honest answer. A zero would say "this seller is never late", which is the
        opposite of "we do not know yet", and gradient-boosted trees handle NaN natively.
        """
        history = resolved([{"order_delivered_customer_date": "2018-01-01"}])
        snaps = build_all_snapshots(history, [pd.Timestamp("2017-01-01")])
        attached = attach_history(
            orders([{"order_id": "o1", "purchase_month": "2017-01-01"}]), snaps
        )
        row = attached.iloc[0]
        assert bool(row["seller_is_new"]) is True
        assert np.isnan(row["seller_late_rate_hist"])

    def test_every_entity_family_gets_its_own_is_new_flag(self):
        history = resolved([{"order_delivered_customer_date": "2017-01-10"}])
        snaps = build_all_snapshots(history, [pd.Timestamp("2017-02-01")])
        attached = attach_history(
            orders([{"order_id": "o1", "purchase_month": "2017-02-01"}]), snaps
        )
        for spec in ENTITIES:
            assert f"{spec.name}_is_new" in attached.columns


class TestMetrics:
    """The four aggregates, and how they treat missing input."""

    def test_avg_handling_days_ignores_nulls_rather_than_treating_them_as_zero(self):
        """184 real orders record a carrier handoff before the purchase; Phase 2 stores those
        as NULL. Counting them as 0.0 would make the seller look instant."""
        history = resolved(
            [
                {"order_delivered_customer_date": "2017-01-10", "handling_days": 4.0},
                {"order_delivered_customer_date": "2017-01-11", "handling_days": None},
            ]
        )
        snap = build_snapshots(history, [pd.Timestamp("2017-02-01")], key="seller_id")
        assert snap["order_count"].tolist() == [2]
        assert snap["avg_handling_days"].tolist() == [4.0]

    def test_all_metric_columns_are_produced(self):
        history = resolved([{"order_delivered_customer_date": "2017-01-10"}])
        snap = build_snapshots(history, [pd.Timestamp("2017-02-01")], key="seller_id")
        for column in METRIC_COLUMNS:
            assert column in snap.columns

    def test_late_rate_is_a_proportion(self):
        history = resolved(
            [
                {"order_delivered_customer_date": "2017-01-10", "is_late": True},
                {"order_delivered_customer_date": "2017-01-11", "is_late": False},
                {"order_delivered_customer_date": "2017-01-12", "is_late": True},
            ]
        )
        snap = build_snapshots(history, [pd.Timestamp("2017-02-01")], key="seller_id")
        assert snap["late_rate"].tolist() == [pytest.approx(2 / 3)]

    def test_entities_are_kept_separate(self):
        history = resolved(
            [
                {"seller_id": "a", "order_delivered_customer_date": "2017-01-10", "is_late": True},
                {"seller_id": "b", "order_delivered_customer_date": "2017-01-10", "is_late": False},
            ]
        )
        snap = build_snapshots(history, [pd.Timestamp("2017-02-01")], key="seller_id").set_index(
            "seller_id"
        )
        assert snap.loc["a", "late_rate"] == 1.0
        assert snap.loc["b", "late_rate"] == 0.0


class TestDeterminism:
    def test_building_twice_gives_an_identical_frame(self):
        history = resolved(
            [{"order_delivered_customer_date": f"2017-0{m}-15"} for m in range(1, 7)]
        )
        months = [pd.Timestamp(f"2017-{m:02d}-01") for m in range(2, 8)]
        first = build_snapshots(history, months, key="seller_id")
        second = build_snapshots(history, months, key="seller_id")
        pd.testing.assert_frame_equal(first, second)

    def test_attaching_twice_gives_an_identical_frame(self):
        history = resolved([{"order_delivered_customer_date": "2017-01-10"}])
        snaps = build_all_snapshots(history, [pd.Timestamp("2017-02-01")])
        frame = orders([{"order_id": "o1", "purchase_month": "2017-02-01"}])
        pd.testing.assert_frame_equal(attach_history(frame, snaps), attach_history(frame, snaps))

    def test_attach_does_not_mutate_its_input(self):
        history = resolved([{"order_delivered_customer_date": "2017-01-10"}])
        snaps = build_all_snapshots(history, [pd.Timestamp("2017-02-01")])
        frame = orders([{"order_id": "o1", "purchase_month": "2017-02-01"}])
        before = frame.copy()
        attach_history(frame, snaps)
        pd.testing.assert_frame_equal(frame, before)

    def test_attach_preserves_row_count_and_order(self):
        history = resolved([{"order_delivered_customer_date": "2017-01-10"}])
        snaps = build_all_snapshots(history, [pd.Timestamp("2017-02-01")])
        frame = orders([{"order_id": f"o{i}", "purchase_month": "2017-02-01"} for i in range(5)])
        attached = attach_history(frame, snaps)
        assert len(attached) == 5
        assert attached["order_id"].tolist() == frame["order_id"].tolist()


class TestSnapshotMonths:
    def test_covers_every_month_of_the_analysis_window_inclusive(self):
        months = snapshot_months(pd.Timestamp("2017-01-01"), pd.Timestamp("2018-09-01"))
        assert months[0] == pd.Timestamp("2017-01-01")
        assert months[-1] == pd.Timestamp("2018-08-01")
        assert len(months) == 20
        assert all(m.day == 1 for m in months)


class TestAgainstTheRealData:
    """Integration: the snapshots as actually built, cross-checked in SQL.

    The value here is that the check is written in a *different language* against a
    *different code path*: pandas builds the snapshot, SQL recounts it. A shared
    misunderstanding of the boundary would have to be made twice, in two idioms, to slip
    through.
    """

    pytestmark: ClassVar = [pytest.mark.integration, pytest.mark.slow]

    def test_snapshots_match_an_independent_set_based_sql_recount(self, engine):
        """Recompute every (seller, month) snapshot in SQL and demand an exact match.

        A FULL OUTER JOIN, so this catches three failure modes at once: a snapshot row whose
        metrics are wrong, a snapshot row that should not exist, and a snapshot row that is
        missing. Written set-based rather than as a correlated subquery — the correlated
        version ran for over nine minutes against 26,730 snapshot rows and would have made
        the suite unusable for the remaining nine phases.
        """
        with engine.connect() as conn:
            mismatches = conn.execute(text("""
                    WITH months AS (
                        SELECT DISTINCT snapshot_month FROM features.seller_monthly
                    ),
                    expected AS (
                        SELECT a.seller_id,
                               mo.snapshot_month,
                               count(*)                     AS order_count,
                               avg(o.is_late::int)          AS late_rate,
                               avg(o.delivery_days)         AS avg_delivery_days
                        FROM months mo
                        JOIN features.order_outcomes o
                          ON o.order_delivered_customer_date < mo.snapshot_month
                        JOIN features.orders_analytical a ON a.order_id = o.order_id
                        GROUP BY a.seller_id, mo.snapshot_month
                    )
                    SELECT count(*)
                    FROM features.seller_monthly sm
                    FULL OUTER JOIN expected e USING (seller_id, snapshot_month)
                    WHERE sm.order_count IS DISTINCT FROM e.order_count
                       OR round(sm.late_rate::numeric, 10)
                          IS DISTINCT FROM round(e.late_rate::numeric, 10)
                       OR round(sm.avg_delivery_days::numeric, 8)
                          IS DISTINCT FROM round(e.avg_delivery_days::numeric, 8)
                    """)).scalar_one()
        assert mismatches == 0

    def test_the_leaky_rule_would_have_produced_different_snapshots(self, engine):
        """The same recount under the *purchase-time* rule must disagree with what was built.

        This is the test that would fail if someone switched the filter column. It is not
        enough to assert the correct rule matches: the wrong rule has to be shown to give a
        different answer, or the test proves nothing about which column was used.
        """
        with engine.connect() as conn:
            differing, extra_orders = conn.execute(text("""
                    WITH months AS (
                        SELECT DISTINCT snapshot_month FROM features.seller_monthly
                    ),
                    leaky AS (
                        SELECT a.seller_id,
                               mo.snapshot_month,
                               count(*) AS order_count
                        FROM months mo
                        JOIN features.orders_analytical a
                          ON a.order_purchase_timestamp < mo.snapshot_month
                        GROUP BY a.seller_id, mo.snapshot_month
                    )
                    SELECT count(*) FILTER (WHERE sm.order_count <> l.order_count),
                           sum(l.order_count - sm.order_count)
                    FROM features.seller_monthly sm
                    JOIN leaky l USING (seller_id, snapshot_month)
                    """)).one()
        # Measured: 9,245 of the 26,730 seller-month snapshots would change, and the leaky
        # rule would admit 36,549 order-contributions whose outcome was not yet known at the
        # snapshot month. Pinned exactly: a drift in either number means the population or
        # the filter column changed, and both deserve a look.
        assert differing == 9_245, (
            "the purchase-time rule no longer disagrees as expected; either the snapshots "
            "are built with the leaky column now, or the population changed"
        )
        assert extra_orders == 36_549

    def test_write_snapshots_round_trips_through_postgres(self, engine):
        """The persistence layer, on a throwaway schema.

        Covers the COPY path that `make snapshots` uses but no other test reaches, and proves
        NaN `avg_handling_days` survives as NULL rather than becoming 0.0.
        """
        history = resolved(
            [
                {"order_delivered_customer_date": "2017-01-10", "handling_days": None},
                {"order_delivered_customer_date": "2017-01-11", "handling_days": None},
            ]
        )
        snaps = build_all_snapshots(history, [pd.Timestamp("2017-02-01")])

        # write_snapshots DROPs and recreates the real tables, so this must run inside a
        # transaction that is *always* rolled back. Postgres DDL is transactional and MVCC
        # keeps the synthetic tables invisible to every other connection meanwhile, so no
        # other test can observe them.
        with engine.connect() as conn:
            transaction = conn.begin()
            try:
                written = write_snapshots(conn, snaps)
                assert written["seller_monthly"] == 1
                handling = conn.execute(
                    text("SELECT avg_handling_days FROM features.seller_monthly")
                ).scalar_one()
                assert handling is None, "NaN became a real number on the way to Postgres"
            finally:
                transaction.rollback()

        # Belt and braces: the real snapshots must be exactly as they were.
        with engine.connect() as conn:
            assert (
                conn.execute(text("SELECT count(*) FROM features.seller_monthly")).scalar_one()
                == 26_730
            )

    def test_every_snapshot_table_exists_and_is_keyed(self, engine):
        expected = (
            {spec.table for spec in ENTITIES}
            | {spec.fallback_table for spec in ENTITIES if spec.fallback_table}
            | {GLOBAL_LEVEL.table}
        )
        with engine.connect() as conn:
            present = {
                row[0]
                for row in conn.execute(
                    text(
                        "SELECT table_name FROM information_schema.tables "
                        "WHERE table_schema = 'features'"
                    )
                )
            }
        assert expected <= present, f"missing snapshot tables: {sorted(expected - present)}"

    def test_late_rates_are_in_range_everywhere(self, engine):
        with engine.connect() as conn:
            for spec in ENTITIES:
                bad = conn.execute(
                    text(
                        f"SELECT count(*) FROM features.{spec.table} "
                        "WHERE late_rate < 0 OR late_rate > 1"
                    )
                ).scalar_one()
                assert bad == 0, spec.table

    def test_cold_start_prevalence_is_what_was_measured(self, engine, attached_features):
        """7.56% of training rows have a cold-start seller, and it does **not** fade after the
        warm-up — new sellers keep joining. A sudden drop to ~0 would mean the flag broke."""
        train = attached_features[attached_features["purchase_month"] >= "2017-05-01"]
        rate = float(train["seller_is_new"].mean())
        assert rate == pytest.approx(0.0756, abs=0.005)
        assert float(train["route_is_new"].mean()) < 0.01

    def test_no_denylist_column_survives_the_attach(self, attached_features):
        from src.etl.schema import DENYLIST

        assert not (DENYLIST & set(attached_features.columns))
