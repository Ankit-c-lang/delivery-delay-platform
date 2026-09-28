"""Load the nine raw Olist CSVs into the Postgres ``raw`` schema.

This is a landing zone, not a model. The job is to get the CSVs into Postgres
*faithfully* — same values, same nullability, no type inference, no cleaning. Every
transformation, rename and filter belongs in Phase 2 (``build_orders.py``).

Three decisions carry the correctness of this module:

1. **Timestamps are ``TIMESTAMP``, never ``TIMESTAMPTZ``.** The CSVs carry naive local
   times with no offset. ``TIMESTAMPTZ`` would make Postgres interpret them in the session
   ``TimeZone`` and shift them on read, which silently corrupts the date-granularity
   comparison in PLAN.md §4.3 (see DECISIONS.md D17).
2. **``*_zip_code_prefix`` is ``TEXT``.** 24% of customer and 33% of seller prefixes begin
   with a zero. ``INTEGER`` would destroy them.
3. **Money is ``NUMERIC(10,2)``.** ``price``, ``freight_value`` and ``payment_value`` are
   exactly two decimal places throughout; binary floats do not represent them exactly and
   the error compounds under the aggregation Phase 3 does.

Column names are kept exactly as Olist ships them, typos included
(``product_name_lenght``). Renaming here would mean the raw table no longer reconciles
against the file.

Run with::

    python -m src.etl.load_raw            # truncate and reload
    python -m src.etl.load_raw --recreate # drop and rebuild the DDL first
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import yaml
from sqlalchemy import text
from sqlalchemy.engine import Connection

from src.db import get_engine

logger = logging.getLogger(__name__)

RAW_SCHEMA = "raw"
CONFIG_PATH = Path("configs/base.yaml")
RECONCILIATION_REPORT = Path("reports/raw_load_reconciliation.md")

# review_comment_message holds free text with embedded newlines; the default field limit
# (128 KiB) is generous enough today, but raise it so a longer comment in a future
# refresh cannot make the *counting* step disagree with what COPY actually loaded.
csv.field_size_limit(10 * 1024 * 1024)


@dataclass(frozen=True)
class TableSpec:
    """DDL and provenance for one raw table.

    Attributes:
        table: Table name inside the ``raw`` schema.
        csv_name: File in the raw data directory.
        columns: ``(column_name, sql_type)`` in **CSV column order**. The order is load
            bearing: it is both the ``CREATE TABLE`` order and the ``COPY`` column list,
            and it is checked against the file's header before any data moves.
        primary_key: Columns forming the primary key, or empty when the table has no
            natural key. Verified against the data, never assumed.
    """

    table: str
    csv_name: str
    columns: tuple[tuple[str, str], ...]
    primary_key: tuple[str, ...] = ()

    @property
    def qualified(self) -> str:
        return f"{RAW_SCHEMA}.{self.table}"

    @property
    def column_names(self) -> list[str]:
        return [name for name, _ in self.columns]

    def create_table_sql(self) -> str:
        """Render the ``CREATE TABLE`` statement."""
        lines = [f"    {name} {sql_type}" for name, sql_type in self.columns]
        if self.primary_key:
            lines.append(
                f"    CONSTRAINT {self.table}_pkey PRIMARY KEY ({', '.join(self.primary_key)})"
            )
        body = ",\n".join(lines)
        return f"CREATE TABLE IF NOT EXISTS {self.qualified} (\n{body}\n)"

    def copy_sql(self) -> str:
        """Render the ``COPY ... FROM STDIN`` statement.

        ``FORMAT csv`` (not ``text``) is required: 3,852 review comments contain embedded
        newlines inside quoted fields and 172 contain escaped double quotes. ``HEADER
        true`` skips the first line, which also discards the UTF-8 BOM on
        ``product_category_name_translation.csv``. An unquoted empty field becomes NULL,
        which is exactly how this dataset encodes its nulls (bare ``,,``), so no
        ``FORCE_NULL`` is needed.
        """
        cols = ", ".join(self.column_names)
        return (
            f"COPY {self.qualified} ({cols}) FROM STDIN "
            f"WITH (FORMAT csv, HEADER true, ENCODING 'UTF8')"
        )


# Nullability below is measured, not guessed: every column was profiled against the real
# files before this list was written. `NOT NULL` appears only where the file has zero
# empty values, so a future refresh that introduces nulls fails loudly here rather than
# silently changing the meaning of a feature downstream.
SPECS: tuple[TableSpec, ...] = (
    TableSpec(
        table="orders",
        csv_name="olist_orders_dataset.csv",
        columns=(
            ("order_id", "TEXT NOT NULL"),
            ("customer_id", "TEXT NOT NULL"),
            ("order_status", "TEXT NOT NULL"),
            ("order_purchase_timestamp", "TIMESTAMP NOT NULL"),
            # 160 nulls: approval can be missing even on delivered orders.
            ("order_approved_at", "TIMESTAMP"),
            # 1,783 nulls.
            ("order_delivered_carrier_date", "TIMESTAMP"),
            # 2,965 nulls. The target column. 8 rows have status 'delivered' yet no
            # delivery date, and 6 'canceled' rows do have one — §4.3's population filter
            # requires BOTH conditions for exactly this reason.
            ("order_delivered_customer_date", "TIMESTAMP"),
            # Date-only in practice: all 99,441 values are exactly midnight. Kept as
            # TIMESTAMP so raw mirrors the file; Phase 2 must cast BOTH sides of the
            # comparison to date (§4.3).
            ("order_estimated_delivery_date", "TIMESTAMP NOT NULL"),
        ),
        primary_key=("order_id",),
    ),
    TableSpec(
        table="customers",
        csv_name="olist_customers_dataset.csv",
        columns=(
            ("customer_id", "TEXT NOT NULL"),
            ("customer_unique_id", "TEXT NOT NULL"),
            # 24.13% begin with '0'. TEXT is mandatory.
            ("customer_zip_code_prefix", "TEXT NOT NULL"),
            ("customer_city", "TEXT NOT NULL"),
            ("customer_state", "TEXT NOT NULL"),
        ),
        primary_key=("customer_id",),
    ),
    TableSpec(
        table="order_items",
        csv_name="olist_order_items_dataset.csv",
        columns=(
            ("order_id", "TEXT NOT NULL"),
            ("order_item_id", "INTEGER NOT NULL"),
            ("product_id", "TEXT NOT NULL"),
            ("seller_id", "TEXT NOT NULL"),
            ("shipping_limit_date", "TIMESTAMP NOT NULL"),
            ("price", "NUMERIC(10, 2) NOT NULL"),
            ("freight_value", "NUMERIC(10, 2) NOT NULL"),
        ),
        primary_key=("order_id", "order_item_id"),
    ),
    TableSpec(
        table="order_payments",
        csv_name="olist_order_payments_dataset.csv",
        columns=(
            ("order_id", "TEXT NOT NULL"),
            ("payment_sequential", "INTEGER NOT NULL"),
            ("payment_type", "TEXT NOT NULL"),
            ("payment_installments", "INTEGER NOT NULL"),
            ("payment_value", "NUMERIC(10, 2) NOT NULL"),
        ),
        primary_key=("order_id", "payment_sequential"),
    ),
    TableSpec(
        table="order_reviews",
        csv_name="olist_order_reviews_dataset.csv",
        columns=(
            ("review_id", "TEXT NOT NULL"),
            ("order_id", "TEXT NOT NULL"),
            ("review_score", "INTEGER NOT NULL"),
            ("review_comment_title", "TEXT"),
            ("review_comment_message", "TEXT"),
            ("review_creation_date", "TIMESTAMP NOT NULL"),
            ("review_answer_timestamp", "TIMESTAMP NOT NULL"),
        ),
        # review_id alone is NOT unique — 814 duplicates, where one review covers several
        # orders. (review_id, order_id) is. Reviews are used for business-impact
        # quantification only (§4.7), never as features.
        primary_key=("review_id", "order_id"),
    ),
    TableSpec(
        table="products",
        csv_name="olist_products_dataset.csv",
        columns=(
            ("product_id", "TEXT NOT NULL"),
            ("product_category_name", "TEXT"),
            # "lenght" is Olist's typo. Preserved so the table reconciles with the file;
            # Phase 2 renames it.
            ("product_name_lenght", "INTEGER"),
            ("product_description_lenght", "INTEGER"),
            ("product_photos_qty", "INTEGER"),
            ("product_weight_g", "INTEGER"),
            ("product_length_cm", "INTEGER"),
            ("product_height_cm", "INTEGER"),
            ("product_width_cm", "INTEGER"),
        ),
        primary_key=("product_id",),
    ),
    TableSpec(
        table="sellers",
        csv_name="olist_sellers_dataset.csv",
        columns=(
            ("seller_id", "TEXT NOT NULL"),
            # 33.18% begin with '0'.
            ("seller_zip_code_prefix", "TEXT NOT NULL"),
            ("seller_city", "TEXT NOT NULL"),
            ("seller_state", "TEXT NOT NULL"),
        ),
        primary_key=("seller_id",),
    ),
    TableSpec(
        table="geolocation",
        csv_name="olist_geolocation_dataset.csv",
        columns=(
            ("geolocation_zip_code_prefix", "TEXT NOT NULL"),
            # DOUBLE PRECISION, not NUMERIC: coordinates are measurements, not money, and
            # the source carries up to 20 spurious decimal places. Phase 2 reduces these
            # to one centroid per prefix.
            ("geolocation_lat", "DOUBLE PRECISION NOT NULL"),
            ("geolocation_lng", "DOUBLE PRECISION NOT NULL"),
            ("geolocation_city", "TEXT NOT NULL"),
            ("geolocation_state", "TEXT NOT NULL"),
        ),
        # No key of any kind: 19,015 prefixes over 1,000,163 rows, up to 1,146 rows for a
        # single prefix. Declaring one would reject the file.
        primary_key=(),
    ),
    TableSpec(
        table="product_category_name_translation",
        csv_name="product_category_name_translation.csv",
        columns=(
            ("product_category_name", "TEXT NOT NULL"),
            ("product_category_name_english", "TEXT NOT NULL"),
        ),
        primary_key=("product_category_name",),
    ),
)


def raw_data_dir(config_path: Path = CONFIG_PATH) -> Path:
    """Read the raw data directory from ``configs/base.yaml``.

    Read from config rather than hard-coded so there is one source of truth for the path.
    Phase 2 should promote this to a proper config module once a second module needs it.
    """
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return Path(config["paths"]["raw_data"])


def csv_header(path: Path) -> list[str]:
    """Return the file's header row.

    ``utf-8-sig`` strips the UTF-8 BOM that ``product_category_name_translation.csv``
    carries, so its first column name compares equal to the spec.
    """
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return next(csv.reader(handle))


def count_csv_rows(path: Path) -> int:
    """Count data rows with a CSV parser.

    **Never use ``wc -l``.** It overstates ``olist_order_reviews_dataset.csv`` by 5,495
    (embedded newlines inside quoted review text) and understates
    ``product_category_name_translation.csv`` by 1 (no trailing newline). A test built on
    line counts fails spuriously, and the tempting fix is to break the loader to match the
    wrong number (CLAUDE.md invariant 11).
    """
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return sum(1 for _ in reader)


def validate_header(spec: TableSpec, path: Path) -> None:
    """Fail if the file's columns are not the spec's columns, in order.

    Raises:
        ValueError: On any name or order mismatch. A re-download with reordered columns
            would otherwise load ``freight_value`` into ``price`` without complaint,
            because ``COPY`` matches by position.
    """
    header = csv_header(path)
    if header != spec.column_names:
        raise ValueError(
            f"{path.name}: header does not match the spec for {spec.qualified}.\n"
            f"  file: {header}\n"
            f"  spec: {spec.column_names}"
        )


def load_table(conn: Connection, spec: TableSpec, data_dir: Path) -> tuple[int, int, float]:
    """Truncate and reload one table.

    Args:
        conn: An open connection inside a transaction. Not committed here — the caller
            owns the transaction so a failure on file nine rolls back files one to eight.
        spec: The table to load.
        data_dir: Directory holding the CSVs.

    Returns:
        ``(rows_read, rows_inserted, seconds)``.

    Raises:
        FileNotFoundError: If the CSV is missing.
        ValueError: If the header does not match the spec.
    """
    path = data_dir / spec.csv_name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run scripts/download_data.py for instructions.")

    validate_header(spec, path)
    rows_read = count_csv_rows(path)

    started = time.perf_counter()
    conn.execute(text(f"TRUNCATE TABLE {spec.qualified}"))

    # psycopg2's COPY needs the DBAPI cursor. It shares this transaction, so it must not
    # be committed independently.
    cursor = conn.connection.cursor()
    try:
        with path.open("rb") as handle:
            cursor.copy_expert(spec.copy_sql(), handle)
        rows_inserted = cursor.rowcount
    finally:
        cursor.close()

    elapsed = time.perf_counter() - started
    logger.info(
        "%-36s read=%9d inserted=%9d %s (%.2fs)",
        spec.qualified,
        rows_read,
        rows_inserted,
        "OK" if rows_read == rows_inserted else "MISMATCH",
        elapsed,
    )
    return rows_read, rows_inserted, elapsed


def write_reconciliation_report(
    results: list[tuple[TableSpec, int, int, float]],
    path: Path = RECONCILIATION_REPORT,
) -> None:
    """Write the row-count reconciliation table to ``reports/``."""
    total_read = sum(read for _, read, _, _ in results)
    total_inserted = sum(ins for _, _, ins, _ in results)
    lines = [
        "# Raw load reconciliation",
        "",
        "Generated by `python -m src.etl.load_raw`. Counts come from a CSV parser, never",
        "`wc -l` (CLAUDE.md invariant 11).",
        "",
        "| CSV | Table | Rows read | Rows inserted | Match | Seconds |",
        "|---|---|---:|---:|:-:|---:|",
    ]
    for spec, read, inserted, secs in results:
        mark = "yes" if read == inserted else "**NO**"
        lines.append(
            f"| `{spec.csv_name}` | `{spec.qualified}` | {read:,} | {inserted:,} "
            f"| {mark} | {secs:.2f} |"
        )
    lines += [
        f"| **total** | | **{total_read:,}** | **{total_inserted:,}** "
        f"| {'yes' if total_read == total_inserted else '**NO**'} | |",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Reconciliation report written to %s", path)


def load_all(
    data_dir: Path | None = None, *, recreate: bool = False
) -> list[tuple[TableSpec, int, int, float]]:
    """Create the schema if needed and reload every table in one transaction.

    Args:
        data_dir: Directory holding the CSVs. Defaults to ``configs/base.yaml``.
        recreate: Drop and rebuild the tables before loading. Use after changing the DDL;
            ``TRUNCATE`` alone leaves an out-of-date table definition in place.

    Returns:
        One ``(spec, rows_read, rows_inserted, seconds)`` per table.

    Raises:
        RuntimeError: If any table's read and inserted counts disagree. The whole
            transaction rolls back, so the schema is never left half-loaded.
    """
    data_dir = data_dir or raw_data_dir()
    engine = get_engine()
    results: list[tuple[TableSpec, int, int, float]] = []

    with engine.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {RAW_SCHEMA}"))
        if recreate:
            for spec in SPECS:
                logger.warning("Dropping %s", spec.qualified)
                conn.execute(text(f"DROP TABLE IF EXISTS {spec.qualified}"))
        for spec in SPECS:
            conn.execute(text(spec.create_table_sql()))
        for spec in SPECS:
            read, inserted, secs = load_table(conn, spec, data_dir)
            results.append((spec, read, inserted, secs))

        mismatched = [s.qualified for s, read, ins, _ in results if read != ins]
        if mismatched:
            raise RuntimeError(
                f"Row counts disagree for {mismatched}; rolling back the entire load."
            )

    return results


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory holding the 9 CSVs (default: paths.raw_data in configs/base.yaml)",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="DROP and recreate the raw tables before loading (use after a DDL change)",
    )
    parser.add_argument("--no-report", action="store_true", help="Skip the reconciliation report")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    started = time.perf_counter()
    results = load_all(args.data_dir, recreate=args.recreate)
    if not args.no_report:
        write_reconciliation_report(results)

    total = sum(inserted for _, _, inserted, _ in results)
    logger.info(
        "Loaded %d tables, %d rows, in %.1fs", len(results), total, time.perf_counter() - started
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
