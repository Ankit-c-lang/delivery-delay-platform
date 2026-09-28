"""Build ``features.orders_analytical`` — one row per delivered order, target derived.

**Prediction is made at ``order_purchase_timestamp``.** Only information available at the
moment the customer completes checkout may become a feature (PLAN.md §4.2). This table is
the feature-safe base: the Pandera contract in :mod:`src.etl.schema` is ``strict=True``, so
it holds exactly the declared columns and a §4.2 denylist column cannot be added to it
without being declared in the open.

``order_delivered_customer_date`` is the one denylist column the project genuinely needs
later: §18 A1 rewrote the §4.5 snapshot rule to filter on *delivery outcome* time. It is
therefore quarantined in a separate table, ``features.order_outcomes``, which only the
Phase 3 snapshot builder and the target join may read. See DECISIONS.md D19.

Division of labour: SQL does the set-based aggregation (it is a join over 1.55 M raw rows),
pandas does the great-circle distance and holds the frame for validation. The distance is
computed with :func:`src.etl.geolocation.haversine_km` rather than in SQL so there is one
implementation of the formula, and it is the one the tests exercise.

Validation runs on the assembled frame *before* the final table is written, and the whole
build is one transaction, so a contract failure leaves no table behind.

Run with::

    python -m src.etl.build_orders
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yaml
from pandera.errors import SchemaError, SchemaErrors
from sqlalchemy import text
from sqlalchemy.engine import Connection

from src.db import get_engine
from src.etl.geolocation import build_centroid_tables, haversine_km
from src.etl.schema import ORDERS_ANALYTICAL_SCHEMA

logger = logging.getLogger(__name__)

FEATURES_SCHEMA = "features"
ANALYTICAL_TABLE = f"{FEATURES_SCHEMA}.orders_analytical"
OUTCOMES_TABLE = f"{FEATURES_SCHEMA}.order_outcomes"
STAGE_TABLE = f"{FEATURES_SCHEMA}._orders_analytical_stage"
SPLITS_PATH = Path("configs/splits.yaml")
FUNNEL_REPORT = Path("reports/etl_funnel.md")

# (column, sql_type) in table order. The COPY column list and the DataFrame column order are
# both derived from this, so they cannot drift apart.
ANALYTICAL_COLUMNS: tuple[tuple[str, str], ...] = (
    ("order_id", "TEXT NOT NULL"),
    ("customer_id", "TEXT NOT NULL"),
    ("customer_unique_id", "TEXT NOT NULL"),
    ("order_purchase_timestamp", "TIMESTAMP NOT NULL"),
    ("purchase_month", "TIMESTAMP NOT NULL"),
    ("order_estimated_delivery_date", "TIMESTAMP NOT NULL"),
    ("promised_days", "INTEGER NOT NULL"),
    ("is_late", "BOOLEAN NOT NULL"),
    ("n_items", "INTEGER NOT NULL"),
    ("n_distinct_products", "INTEGER NOT NULL"),
    ("n_distinct_sellers", "INTEGER NOT NULL"),
    ("n_seller_states", "INTEGER NOT NULL"),
    ("total_price", "NUMERIC(12, 2) NOT NULL"),
    ("max_price", "NUMERIC(10, 2) NOT NULL"),
    ("total_freight", "NUMERIC(12, 2) NOT NULL"),
    ("total_weight_g", "DOUBLE PRECISION"),
    ("total_volume_cm3", "DOUBLE PRECISION"),
    ("dominant_category", "TEXT"),
    ("dominant_category_english", "TEXT"),
    ("max_shipping_limit_date", "TIMESTAMP NOT NULL"),
    ("days_to_shipping_limit", "DOUBLE PRECISION NOT NULL"),
    ("dominant_payment_type", "TEXT NOT NULL"),
    ("max_installments", "INTEGER NOT NULL"),
    ("total_payment_value", "NUMERIC(12, 2) NOT NULL"),
    ("n_payment_methods", "INTEGER NOT NULL"),
    ("customer_state", "TEXT NOT NULL"),
    ("customer_zip_code_prefix", "TEXT NOT NULL"),
    ("customer_lat", "DOUBLE PRECISION"),
    ("customer_lng", "DOUBLE PRECISION"),
    ("customer_geo_source", "TEXT NOT NULL"),
    ("seller_id", "TEXT NOT NULL"),
    ("seller_state", "TEXT NOT NULL"),
    ("seller_zip_code_prefix", "TEXT NOT NULL"),
    ("seller_lat", "DOUBLE PRECISION"),
    ("seller_lng", "DOUBLE PRECISION"),
    ("seller_geo_source", "TEXT NOT NULL"),
    ("customer_seller_distance_km", "DOUBLE PRECISION"),
    ("route", "TEXT NOT NULL"),
)

COLUMN_NAMES: tuple[str, ...] = tuple(name for name, _ in ANALYTICAL_COLUMNS)

# Everything except the distance, which pandas adds.
STAGE_SQL = f"""
CREATE TABLE {STAGE_TABLE} AS
WITH population AS (
    SELECT o.order_id,
           o.customer_id,
           o.order_purchase_timestamp,
           o.order_estimated_delivery_date,
           o.order_delivered_customer_date
    FROM raw.orders o
    WHERE o.order_status = 'delivered'
      AND o.order_delivered_customer_date IS NOT NULL
      AND o.order_purchase_timestamp >= :window_start
      AND o.order_purchase_timestamp <  :window_end_exclusive
),
item_agg AS (
    SELECT i.order_id,
           count(*)::int                       AS n_items,
           count(DISTINCT i.product_id)::int   AS n_distinct_products,
           count(DISTINCT i.seller_id)::int    AS n_distinct_sellers,
           count(DISTINCT s.seller_state)::int AS n_seller_states,
           sum(i.price)::numeric(12, 2)        AS total_price,
           max(i.price)::numeric(10, 2)        AS max_price,
           sum(i.freight_value)::numeric(12, 2) AS total_freight,
           sum(pr.product_weight_g)::double precision AS total_weight_g,
           sum(
               pr.product_length_cm::double precision
               * pr.product_height_cm
               * pr.product_width_cm
           )                                   AS total_volume_cm3,
           max(i.shipping_limit_date)          AS max_shipping_limit_date
    FROM population p
    JOIN raw.order_items i  ON i.order_id   = p.order_id
    JOIN raw.products    pr ON pr.product_id = i.product_id
    JOIN raw.sellers     s  ON s.seller_id   = i.seller_id
    GROUP BY i.order_id
),
-- "Dominant" = largest share of the order's item value, tie-broken alphabetically so the
-- result is deterministic across runs. It matters for only 1.32% of orders (1,272 are
-- multi-seller, 721 multi-category); n_distinct_sellers and n_seller_states stay in the
-- table so a model can still see that an order was split.
category_rank AS (
    SELECT i.order_id,
           pr.product_category_name AS category,
           row_number() OVER (
               PARTITION BY i.order_id
               ORDER BY sum(i.price) DESC, pr.product_category_name ASC
           ) AS rn
    FROM population p
    JOIN raw.order_items i  ON i.order_id    = p.order_id
    JOIN raw.products    pr ON pr.product_id = i.product_id
    WHERE pr.product_category_name IS NOT NULL
    GROUP BY i.order_id, pr.product_category_name
),
seller_rank AS (
    SELECT i.order_id,
           i.seller_id,
           row_number() OVER (
               PARTITION BY i.order_id
               ORDER BY sum(i.price) DESC, i.seller_id ASC
           ) AS rn
    FROM population p
    JOIN raw.order_items i ON i.order_id = p.order_id
    GROUP BY i.order_id, i.seller_id
),
payment_agg AS (
    SELECT pay.order_id,
           max(pay.payment_installments)::int    AS max_installments,
           sum(pay.payment_value)::numeric(12, 2) AS total_payment_value,
           count(DISTINCT pay.payment_type)::int AS n_payment_methods
    FROM population p
    JOIN raw.order_payments pay ON pay.order_id = p.order_id
    GROUP BY pay.order_id
),
payment_rank AS (
    SELECT pay.order_id,
           pay.payment_type,
           row_number() OVER (
               PARTITION BY pay.order_id
               ORDER BY sum(pay.payment_value) DESC, pay.payment_type ASC
           ) AS rn
    FROM population p
    JOIN raw.order_payments pay ON pay.order_id = p.order_id
    GROUP BY pay.order_id, pay.payment_type
),
-- Zip centroid first, state centroid as fallback. Needed, not defensive: 157 customer and
-- 7 seller prefixes never appear in raw.geolocation, and 4 more lose every point to the
-- out-of-Brazil bounds filter.
customer_geo AS (
    SELECT c.customer_id,
           c.customer_unique_id,
           c.customer_state,
           c.customer_zip_code_prefix,
           COALESCE(z.lat, st.lat) AS lat,
           COALESCE(z.lng, st.lng) AS lng,
           CASE
               WHEN z.lat IS NOT NULL  THEN 'zip'
               WHEN st.lat IS NOT NULL THEN 'state'
               ELSE 'none'
           END AS geo_source
    FROM raw.customers c
    LEFT JOIN {FEATURES_SCHEMA}.zip_centroids   z  ON z.zip_code_prefix = c.customer_zip_code_prefix
    LEFT JOIN {FEATURES_SCHEMA}.state_centroids st ON st.state = c.customer_state
),
seller_geo AS (
    SELECT s.seller_id,
           s.seller_state,
           s.seller_zip_code_prefix,
           COALESCE(z.lat, st.lat) AS lat,
           COALESCE(z.lng, st.lng) AS lng,
           CASE
               WHEN z.lat IS NOT NULL  THEN 'zip'
               WHEN st.lat IS NOT NULL THEN 'state'
               ELSE 'none'
           END AS geo_source
    FROM raw.sellers s
    LEFT JOIN {FEATURES_SCHEMA}.zip_centroids   z  ON z.zip_code_prefix = s.seller_zip_code_prefix
    LEFT JOIN {FEATURES_SCHEMA}.state_centroids st ON st.state = s.seller_state
)
SELECT p.order_id,
       p.customer_id,
       cg.customer_unique_id,
       p.order_purchase_timestamp,
       date_trunc('month', p.order_purchase_timestamp)         AS purchase_month,
       p.order_estimated_delivery_date,
       (p.order_estimated_delivery_date::date
        - p.order_purchase_timestamp::date)::int               AS promised_days,
       -- §4.3: DATE granularity on BOTH sides. order_estimated_delivery_date is always
       -- midnight, so a timestamp comparison marks every delivery on the promised day as
       -- late and inflates the rate from 6.79% to 8.13%.
       (p.order_delivered_customer_date::date
        > p.order_estimated_delivery_date::date)               AS is_late,
       ia.n_items,
       ia.n_distinct_products,
       ia.n_distinct_sellers,
       ia.n_seller_states,
       ia.total_price,
       ia.max_price,
       ia.total_freight,
       ia.total_weight_g,
       ia.total_volume_cm3,
       dc.category                                            AS dominant_category,
       COALESCE(tr.product_category_name_english, dc.category) AS dominant_category_english,
       ia.max_shipping_limit_date,
       (extract(epoch FROM (ia.max_shipping_limit_date - p.order_purchase_timestamp))
        / 86400.0)                                            AS days_to_shipping_limit,
       dp.payment_type                                        AS dominant_payment_type,
       pa.max_installments,
       pa.total_payment_value,
       pa.n_payment_methods,
       cg.customer_state,
       cg.customer_zip_code_prefix,
       cg.lat                                                 AS customer_lat,
       cg.lng                                                 AS customer_lng,
       cg.geo_source                                          AS customer_geo_source,
       ds.seller_id,
       sg.seller_state,
       sg.seller_zip_code_prefix,
       sg.lat                                                 AS seller_lat,
       sg.lng                                                 AS seller_lng,
       sg.geo_source                                          AS seller_geo_source,
       (sg.seller_state || '->' || cg.customer_state)          AS route
FROM population p
JOIN item_agg     ia ON ia.order_id = p.order_id
JOIN payment_agg  pa ON pa.order_id = p.order_id
JOIN customer_geo cg ON cg.customer_id = p.customer_id
JOIN (SELECT order_id, seller_id    FROM seller_rank  WHERE rn = 1) ds ON ds.order_id = p.order_id
JOIN (SELECT order_id, payment_type FROM payment_rank WHERE rn = 1) dp ON dp.order_id = p.order_id
JOIN seller_geo   sg ON sg.seller_id = ds.seller_id
-- LEFT: 1,535 item rows have no category at all, and 13 products carry a category with no
-- translation row (pc_gamer, portateis_cozinha_e_preparadores_de_alimentos). An inner join
-- would drop those orders silently.
LEFT JOIN (SELECT order_id, category FROM category_rank WHERE rn = 1) dc ON dc.order_id = p.order_id
LEFT JOIN raw.product_category_name_translation tr ON tr.product_category_name = dc.category
"""

OUTCOMES_SQL = f"""
INSERT INTO {OUTCOMES_TABLE} (order_id, order_delivered_customer_date, is_late)
SELECT o.order_id,
       o.order_delivered_customer_date,
       (o.order_delivered_customer_date::date > o.order_estimated_delivery_date::date)
FROM raw.orders o
WHERE o.order_status = 'delivered'
  AND o.order_delivered_customer_date IS NOT NULL
  AND o.order_purchase_timestamp >= :window_start
  AND o.order_purchase_timestamp <  :window_end_exclusive
"""

# Each step is cumulative: it applies its own predicate and every predicate above it.
FUNNEL_STEPS: tuple[tuple[str, str], ...] = (
    ("all raw orders", "TRUE"),
    ("status = 'delivered'", "order_status = 'delivered'"),
    (
        "+ delivery date not null",
        "order_status = 'delivered' AND order_delivered_customer_date IS NOT NULL",
    ),
    (
        "+ purchased inside the analysis window",
        "order_status = 'delivered' AND order_delivered_customer_date IS NOT NULL "
        "AND order_purchase_timestamp >= :window_start "
        "AND order_purchase_timestamp < :window_end_exclusive",
    ),
)


def analysis_window(splits_path: Path = SPLITS_PATH) -> tuple[date, date]:
    """Read the analysis window from ``configs/splits.yaml``.

    Returns:
        ``(start, end_exclusive)``. The config's ``end`` is inclusive (PLAN.md §4.4), so
        the exclusive bound returned here is one day later — comparing a timestamp against
        an inclusive date bound would silently drop everything after midnight on the last
        day.
    """
    splits = yaml.safe_load(splits_path.read_text(encoding="utf-8"))
    window = splits["analysis_window"]
    start = date.fromisoformat(str(window["start"]))
    end_inclusive = date.fromisoformat(str(window["end"]))
    return start, end_inclusive + timedelta(days=1)


def create_tables(conn: Connection) -> None:
    """Create the ``features`` schema and both output tables, replacing any existing ones."""
    conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {FEATURES_SCHEMA}"))
    conn.execute(text(f"DROP TABLE IF EXISTS {STAGE_TABLE}"))
    conn.execute(text(f"DROP TABLE IF EXISTS {ANALYTICAL_TABLE}"))
    conn.execute(text(f"DROP TABLE IF EXISTS {OUTCOMES_TABLE}"))

    body = ",\n".join(f"    {name} {sql_type}" for name, sql_type in ANALYTICAL_COLUMNS)
    conn.execute(
        text(
            f"CREATE TABLE {ANALYTICAL_TABLE} (\n{body},\n"
            f"    CONSTRAINT orders_analytical_pkey PRIMARY KEY (order_id)\n)"
        )
    )
    conn.execute(
        text(
            f"CREATE TABLE {OUTCOMES_TABLE} (\n"
            "    order_id TEXT NOT NULL,\n"
            "    order_delivered_customer_date TIMESTAMP NOT NULL,\n"
            "    is_late BOOLEAN NOT NULL,\n"
            "    CONSTRAINT order_outcomes_pkey PRIMARY KEY (order_id)\n)"
        )
    )


def measure_funnel(conn: Connection, params: dict[str, object]) -> list[tuple[str, int]]:
    """Count surviving orders after each population filter (PLAN.md §4.3)."""
    funnel: list[tuple[str, int]] = []
    for label, predicate in FUNNEL_STEPS:
        step_params = {k: v for k, v in params.items() if f":{k}" in predicate}
        count = conn.execute(
            text(f"SELECT count(*) FROM raw.orders WHERE {predicate}"), step_params
        ).scalar_one()
        funnel.append((label, count))
    return funnel


def add_distance(frame: pd.DataFrame) -> pd.DataFrame:
    """Add ``customer_seller_distance_km`` using the shared haversine implementation.

    Args:
        frame: The staged frame, with customer and seller coordinates.

    Returns:
        A new frame with the distance column. Never mutates the input — pandas 3
        copy-on-write makes in-place mutation of a slice an error waiting to happen
        (CLAUDE.md invariant 9).
    """
    out = frame.copy()
    out["customer_seller_distance_km"] = haversine_km(
        out["customer_lat"], out["customer_lng"], out["seller_lat"], out["seller_lng"]
    )
    return out


def copy_frame(conn: Connection, frame: pd.DataFrame, table: str, columns: tuple[str, ...]) -> int:
    """Write a DataFrame into ``table`` with ``COPY FROM STDIN``.

    Args:
        conn: Open connection inside a transaction.
        frame: Rows to write. Reindexed to ``columns``, so column order cannot drift.
        table: Fully qualified destination.
        columns: Destination columns, in table order.

    Returns:
        Rows written.
    """
    buffer = io.StringIO()
    frame.reindex(columns=list(columns)).to_csv(
        buffer, index=False, header=False, na_rep="", quoting=csv.QUOTE_MINIMAL
    )
    buffer.seek(0)
    cursor = conn.connection.cursor()
    try:
        cursor.copy_expert(
            f"COPY {table} ({', '.join(columns)}) FROM STDIN WITH (FORMAT csv, ENCODING 'UTF8')",
            buffer,
        )
        return cursor.rowcount
    finally:
        cursor.close()


def write_funnel_report(
    funnel: list[tuple[str, int]],
    positive_rate: float,
    centroids: dict[str, int],
    path: Path = FUNNEL_REPORT,
) -> None:
    """Write the population funnel to ``reports/etl_funnel.md``."""
    lines = [
        "# ETL funnel: raw.orders -> features.orders_analytical",
        "",
        "Generated by `python -m src.etl.build_orders`. Population filter per PLAN.md §4.3.",
        "",
        "| Step | Orders | Dropped |",
        "|---|---:|---:|",
    ]
    previous: int | None = None
    for label, count in funnel:
        dropped = "-" if previous is None else f"{previous - count:,}"
        lines.append(f"| {label} | {count:,} | {dropped} |")
        previous = count
    kept = funnel[-1][1]
    total = funnel[0][1]
    lines += [
        "",
        f"**Kept {kept:,} of {total:,} orders ({kept / total:.2%}).**",
        "",
        "## Target",
        "",
        f"- `is_late` positive rate: **{positive_rate:.4f}** ({positive_rate:.2%})",
        "- Compared at **date** granularity on both sides (§4.3). A timestamp comparison",
        "  would give 0.0813 by marking all 1,291 deliveries on the promised day as late.",
        "",
        "## Geolocation reduction",
        "",
        f"- `features.zip_centroids`: **{centroids.get('zip_centroids', 0):,}** prefixes",
        f"- `features.state_centroids`: **{centroids.get('state_centroids', 0):,}** states",
        "- Median, after dropping points outside Brazil. See `src/etl/geolocation.py`.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Funnel report written to %s", path)


def build() -> dict[str, object]:
    """Build both output tables in one transaction.

    Returns:
        A summary with the funnel, row counts, positive rate and centroid counts.

    Raises:
        SchemaError, SchemaErrors: If the assembled frame violates the §4.2/§4.3 contract.
            The transaction rolls back, so no table is left behind.
    """
    start, end_exclusive = analysis_window()
    params: dict[str, object] = {"window_start": start, "window_end_exclusive": end_exclusive}
    logger.info("Analysis window: %s .. %s (exclusive)", start, end_exclusive)

    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {FEATURES_SCHEMA}"))
        centroids = build_centroid_tables(conn)

        create_tables(conn)
        funnel = measure_funnel(conn, params)
        for label, count in funnel:
            logger.info("funnel  %-42s %7d", label, count)

        conn.execute(text(STAGE_SQL), params)
        staged = pd.read_sql(text(f"SELECT * FROM {STAGE_TABLE}"), conn)
        logger.info("Staged %d rows, %d columns", len(staged), staged.shape[1])

        frame = add_distance(staged)

        try:
            ORDERS_ANALYTICAL_SCHEMA.validate(frame, lazy=True)
        except (SchemaError, SchemaErrors) as exc:
            logger.error("Pandera contract failed; rolling back. %s", exc)
            raise

        written = copy_frame(conn, frame, ANALYTICAL_TABLE, COLUMN_NAMES)
        outcomes = conn.execute(text(OUTCOMES_SQL), params).rowcount
        conn.execute(text(f"DROP TABLE {STAGE_TABLE}"))

        if written != funnel[-1][1]:
            raise RuntimeError(
                f"wrote {written} rows but the funnel expected {funnel[-1][1]}; rolling back"
            )
        if outcomes != written:
            raise RuntimeError(
                f"{OUTCOMES_TABLE} has {outcomes} rows but {ANALYTICAL_TABLE} has {written}"
            )

        positive_rate = float(frame["is_late"].mean())
        logger.info("%s  %d rows, is_late rate %.4f", ANALYTICAL_TABLE, written, positive_rate)
        logger.info("%s  %d rows (quarantined outcome columns)", OUTCOMES_TABLE, outcomes)

    return {
        "funnel": funnel,
        "rows": written,
        "outcome_rows": outcomes,
        "positive_rate": positive_rate,
        "centroids": centroids,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-report", action="store_true", help="Skip the funnel report")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    started = time.perf_counter()
    summary = build()
    if not args.no_report:
        write_funnel_report(summary["funnel"], summary["positive_rate"], summary["centroids"])
    logger.info("Done in %.1fs", time.perf_counter() - started)
    return 0


if __name__ == "__main__":
    sys.exit(main())
