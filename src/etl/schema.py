"""The data contract for ``features.orders_analytical``.

Two things live here, and they are the same idea from two directions:

* :data:`DENYLIST` — the §4.2 columns that may never reach a feature matrix, because every
  one of them is populated *after* the prediction moment.
* :data:`ORDERS_ANALYTICAL_SCHEMA` — a Pandera schema with ``strict=True``, so the table is
  allowed exactly the columns listed and nothing else. That is the mechanism that makes
  denylist leakage impossible rather than merely discouraged: a forbidden column cannot be
  added to the table without also being added here, in the open.

Every numeric bound below was measured against the real data before it was written, so a
failure means the ETL changed, not that the bound was a guess.
"""

from __future__ import annotations

import pandera.pandas as pa
from pandera.dtypes import Timestamp

# PLAN.md §4.2. `order_status` is included even though the population is filtered to
# 'delivered' — it is then a constant, and a constant that encodes the filter is exactly
# the shape of a leak that survives review.
DENYLIST: frozenset[str] = frozenset(
    {
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_status",
        "review_id",
        "review_score",
        "review_comment_title",
        "review_comment_message",
        "review_creation_date",
        "review_answer_timestamp",
    }
)

# `order_delivered_customer_date` is genuinely needed by the §4.5 as-of snapshot rule, which
# §18 A1 rewrote to filter on delivery *outcome* time. It does not live in
# orders_analytical: it is quarantined in `features.order_outcomes`, which only the snapshot
# builder and the target join may read. See DECISIONS.md D19.
OUTCOME_TABLE = "order_outcomes"

# Measured on the population of 96,203 orders (2017-01-01 .. 2018-08-31).
POSITIVE_RATE_MIN = 0.05
POSITIVE_RATE_MAX = 0.10

_LAT = pa.Column(float, pa.Check.between(-34.0, 6.0), nullable=True)
_LNG = pa.Column(float, pa.Check.between(-74.0, -28.0), nullable=True)
_GEO_SOURCE = pa.Column(str, pa.Check.isin(["zip", "state", "none"]), nullable=False)


def _no_denylist_column(df) -> bool:
    """Dataframe-level check: none of the §4.2 forbidden columns is present."""
    return not (DENYLIST & set(df.columns))


def _positive_rate_in_band(df) -> bool:
    """Dataframe-level check: the target rate lands where §4.3 says it must.

    A rate near 8.1% means the timestamp-vs-date bug; a rate near 92% means the comparison
    is inverted; a rate near 38% means the target was defined against a duration rather
    than against the promised date.
    """
    rate = float(df["is_late"].mean())
    return POSITIVE_RATE_MIN <= rate <= POSITIVE_RATE_MAX


ORDERS_ANALYTICAL_SCHEMA = pa.DataFrameSchema(
    columns={
        # --- identity and time (all known at checkout) ---------------------------------
        "order_id": pa.Column(str, nullable=False, unique=True),
        "customer_id": pa.Column(str, nullable=False),
        "customer_unique_id": pa.Column(str, nullable=False),
        "order_purchase_timestamp": pa.Column(Timestamp, nullable=False),
        "purchase_month": pa.Column(Timestamp, nullable=False),
        "order_estimated_delivery_date": pa.Column(Timestamp, nullable=False),
        # Measured range 3..156 days. `promised_days` is the strongest single feature
        # available (§4.2); a negative value would mean the dates were parsed wrongly.
        "promised_days": pa.Column(int, pa.Check.between(1, 365), nullable=False),
        # --- the label ------------------------------------------------------------------
        "is_late": pa.Column(bool, nullable=False),
        # --- order_items aggregates -----------------------------------------------------
        "n_items": pa.Column(int, pa.Check.between(1, 50), nullable=False),
        "n_distinct_products": pa.Column(int, pa.Check.ge(1), nullable=False),
        "n_distinct_sellers": pa.Column(int, pa.Check.ge(1), nullable=False),
        "n_seller_states": pa.Column(int, pa.Check.ge(1), nullable=False),
        "total_price": pa.Column(float, pa.Check.gt(0), nullable=False),
        "max_price": pa.Column(float, pa.Check.gt(0), nullable=False),
        "total_freight": pa.Column(float, pa.Check.ge(0), nullable=False),
        # Nullable: 18 item rows carry no weight or dimensions at all.
        "total_weight_g": pa.Column(float, pa.Check.ge(0), nullable=True),
        "total_volume_cm3": pa.Column(float, pa.Check.ge(0), nullable=True),
        # Measured: some products record 0 g, so the bound is ge(0) not gt(0).
        "max_item_weight_g": pa.Column(float, pa.Check.ge(0), nullable=True),
        "max_item_volume_cm3": pa.Column(float, pa.Check.gt(0), nullable=True),
        # Nullable: 1,535 item rows have no product category.
        "dominant_category": pa.Column(str, nullable=True),
        "dominant_category_english": pa.Column(str, nullable=True),
        "max_shipping_limit_date": pa.Column(Timestamp, nullable=False),
        # Measured 2.00 .. 1052.00 days. Not a leak (§18 A4): a contractual seller deadline
        # set at order time. The 1052-day tail needs winsorizing and the near-collinearity
        # with promised_days needs auditing — both are Phase 3 decisions, so the value is
        # stored raw here.
        "days_to_shipping_limit": pa.Column(float, pa.Check.ge(0), nullable=False),
        # --- payment aggregates ---------------------------------------------------------
        "dominant_payment_type": pa.Column(str, nullable=False),
        # Measured 0..24. Zero is real: some payments record no instalment plan.
        "max_installments": pa.Column(int, pa.Check.between(0, 36), nullable=False),
        "total_payment_value": pa.Column(float, pa.Check.gt(0), nullable=False),
        "n_payment_methods": pa.Column(int, pa.Check.between(1, 10), nullable=False),
        # --- geography ------------------------------------------------------------------
        "customer_state": pa.Column(str, pa.Check.str_length(2, 2), nullable=False),
        "customer_zip_code_prefix": pa.Column(str, pa.Check.str_length(5, 5), nullable=False),
        "customer_lat": _LAT,
        "customer_lng": _LNG,
        "customer_geo_source": _GEO_SOURCE,
        "seller_id": pa.Column(str, nullable=False),
        "seller_state": pa.Column(str, pa.Check.str_length(2, 2), nullable=False),
        "seller_zip_code_prefix": pa.Column(str, pa.Check.str_length(5, 5), nullable=False),
        "seller_lat": _LAT,
        "seller_lng": _LNG,
        "seller_geo_source": _GEO_SOURCE,
        # Brazil's longest internal distance is ~4,300 km; the cap allows headroom without
        # admitting a coordinate that escaped the bounds filter.
        "customer_seller_distance_km": pa.Column(
            float, pa.Check.between(0.0, 5000.0), nullable=True
        ),
        # seller_state -> customer_state, the §4.5 route entity.
        "route": pa.Column(str, pa.Check.str_length(6, 6), nullable=False),
    },
    checks=[
        pa.Check(
            _no_denylist_column,
            name="no_denylist_column",
            error="§4.2 denylist column present",
        ),
        pa.Check(
            _positive_rate_in_band,
            name="positive_rate_in_band",
            error=f"is_late rate outside [{POSITIVE_RATE_MIN}, {POSITIVE_RATE_MAX}] (§4.3)",
        ),
    ],
    # No extra columns, no missing columns. This is what makes the denylist enforceable.
    strict=True,
    ordered=False,
    name="features.orders_analytical",
)
