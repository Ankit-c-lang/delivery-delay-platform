"""The real ETL against a throwaway database, on synthetic CSVs (PLAN.md §10.3, §13 Phase 10).

    WHY THIS EXISTS SEPARATELY FROM THE OTHER ETL TESTS
    ==================================================
    The existing `integration` tests assert facts about the **real** Olist data — 99,224 review
    rows, exact leading-zero counts, the 96,203-row analytical table. They need a Postgres that has
    already been loaded with it, so they cannot run in CI, which will never have the 121 MB
    CC BY-NC-SA dataset.

    This module runs the **same loader and the same transform** against committed synthetic CSVs,
    so CI exercises the ETL code path for real rather than skipping it. §10.3 calls a Postgres
    service container "the part that looks like real CI, and it's cheap", and a green run that
    skipped the ETL would say nothing about the code that meets real data.

**It builds and drops its own database.** Marked ``ci_etl`` rather than ``integration`` for that
reason: the `integration` tests read whatever is already loaded, while these create
``delay_ci_smoke_<pid>``, load into it, and drop it in teardown. Running the ETL with
``recreate=True`` against the configured database would destroy the project's own loaded data on a
developer machine — an acceptable outcome in a disposable CI container, and completely unacceptable
locally. Isolation is what makes the same test safe in both places.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import ClassVar

import pytest
from sqlalchemy import create_engine, text
from tests.fixtures.synthetic_olist import LATE_RATE, N_ORDERS, generate

pytestmark = pytest.mark.ci_etl


@pytest.fixture(scope="module")
def throwaway_database():
    """Create a scratch database, point ``src.db`` at it, and drop it afterwards.

    Yields:
        The database name.

    Skips when Postgres is unreachable — checked first and on its own, so an unreachable server is
    the only thing that can produce a skip and a genuine ETL failure can never hide behind one
    (the Phase 6 lesson, DECISIONS.md D32).
    """
    import src.db as db

    admin_url = None
    try:
        settings = db.DatabaseSettings()
        admin_url = settings.url.set(database="postgres")
        probe = create_engine(admin_url, isolation_level="AUTOCOMMIT")
        with probe.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:
        pytest.skip(f"Postgres unreachable: {type(exc).__name__}: {exc}")

    name = f"delay_ci_smoke_{os.getpid()}"
    with probe.connect() as conn:
        conn.execute(text(f'DROP DATABASE IF EXISTS "{name}"'))
        conn.execute(text(f'CREATE DATABASE "{name}"'))

    original = os.environ.get("POSTGRES_DB")
    os.environ["POSTGRES_DB"] = name
    db.get_settings.cache_clear()
    db.get_engine.cache_clear()
    try:
        yield name
    finally:
        # Restore first, so a failure in teardown cannot leave the process pointed at a database
        # that is about to disappear.
        if original is None:
            os.environ.pop("POSTGRES_DB", None)
        else:
            os.environ["POSTGRES_DB"] = original
        db.get_engine.cache_clear()
        db.get_settings.cache_clear()
        db.get_engine().dispose()
        with probe.connect() as conn:
            conn.execute(
                text(
                    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                    f"WHERE datname = '{name}' AND pid <> pg_backend_pid()"
                )
            )
            conn.execute(text(f'DROP DATABASE IF EXISTS "{name}"'))


@pytest.fixture(scope="module")
def loaded(throwaway_database, tmp_path_factory):
    """Generate the CSVs, run the real loader, and return what it reported."""
    from src.etl.load_raw import load_all

    del throwaway_database
    directory = tmp_path_factory.mktemp("olist_csv")
    data = generate(directory)
    results = load_all(directory, recreate=True)
    return data, {spec.table: (read, inserted) for spec, read, inserted, _ in results}


class TestTheLoaderRunsForReal:
    def test_every_table_loaded_and_the_counts_reconcile(self, loaded):
        """The loader's own invariant: rows read must equal rows inserted, per table."""
        _, results = loaded
        assert len(results) == 9
        for table, (read, inserted) in results.items():
            assert read == inserted, f"{table}: read {read}, inserted {inserted}"

    def test_the_row_counts_match_a_csv_parser_not_wc_l(self, loaded):
        """CLAUDE.md invariant 11, exercised rather than asserted.

        The fixture puts a newline inside one review comment in ten, so the reviews file has 265
        physical lines and 240 logical rows. A loader counting lines would disagree with Postgres
        here, exactly as it would on the real file's 104,719 lines against 99,224 rows.
        """
        data, results = loaded
        reviews_csv = data.directory / "olist_order_reviews_dataset.csv"
        physical_lines = reviews_csv.read_text(encoding="utf-8").count("\n")
        logical_rows = results["order_reviews"][1]

        assert physical_lines > logical_rows, (
            "the fixture no longer contains an embedded newline, so this test would pass for a "
            "loader that counts lines"
        )
        assert logical_rows == data.row_counts["olist_order_reviews_dataset.csv"]

    def test_leading_zeros_survive_as_text(self, loaded):
        """A regression to INTEGER zip columns would silently drop the leading zero."""
        del loaded
        from src.db import get_engine

        with get_engine().connect() as conn:
            zeros = conn.execute(
                text("SELECT count(*) FROM raw.customers WHERE customer_zip_code_prefix LIKE '0%'")
            ).scalar_one()
        assert zeros > 0, "no zip prefix starts with 0; the fixture or the column type changed"

    def test_nulls_in_nullable_product_columns_arrived_as_nulls(self, loaded):
        """Empty CSV fields must become NULL, not the empty string or zero."""
        del loaded
        from src.db import get_engine

        with get_engine().connect() as conn:
            nulls = conn.execute(
                text("SELECT count(*) FROM raw.products WHERE product_weight_g IS NULL")
            ).scalar_one()
        assert nulls > 0


class TestTheTransformRunsForReal:
    pytestmark: ClassVar = [pytest.mark.ci_etl]

    @pytest.fixture(scope="class")
    @staticmethod
    def built(loaded):
        """Run the real `build_orders` against the synthetic raw tables."""
        from src.etl.build_orders import build

        del loaded
        return build()

    def test_the_analytical_table_is_populated(self, built):
        del built
        from src.db import get_engine

        with get_engine().connect() as conn:
            rows = conn.execute(
                text("SELECT count(*) FROM features.orders_analytical")
            ).scalar_one()
        assert 0 < rows <= N_ORDERS

    def test_the_pandera_contract_holds_on_synthetic_data(self, built):
        """The `strict=True` schema is what stops a denylist column entering the table.

        Running it on synthetic data is the point: a contract that has only ever been checked
        against one dataset is a contract that has only ever been checked once.
        """
        del built
        import pandas as pd

        from src.db import get_engine
        from src.etl.schema import ORDERS_ANALYTICAL_SCHEMA

        frame = pd.read_sql("SELECT * FROM features.orders_analytical", get_engine())
        ORDERS_ANALYTICAL_SCHEMA.validate(frame, lazy=True)

    def test_no_denylisted_column_reached_the_analytical_table(self, built):
        del built
        import pandas as pd

        from src.contract import DENYLIST
        from src.db import get_engine

        frame = pd.read_sql("SELECT * FROM features.orders_analytical LIMIT 1", get_engine())
        assert not (DENYLIST & set(frame.columns))

    def test_the_late_rate_is_in_the_configured_band(self, built):
        """The fixture targets ~7%, matching the real 6.79%, so the band in base.yaml applies.

        A rate far outside it would mean the DATE-granularity comparison broke (§4.3) rather than
        that the data is interesting — which is exactly what the band is there to catch.
        """
        del built
        import yaml

        from src.db import get_engine

        band = yaml.safe_load(Path("configs/base.yaml").read_text(encoding="utf-8"))["target"][
            "expected_positive_rate"
        ]
        with get_engine().connect() as conn:
            rate = float(
                conn.execute(
                    text("SELECT avg(is_late::int) FROM features.orders_analytical")
                ).scalar_one()
            )
        assert (
            band["min"] <= rate <= band["max"]
        ), f"late rate {rate:.4f} outside the configured band; the fixture targets {LATE_RATE:.0%}"

    def test_the_outcome_column_is_quarantined(self, built):
        """Invariant 12: `order_delivered_customer_date` lives only in `features.order_outcomes`."""
        del built
        from src.db import get_engine

        with get_engine().connect() as conn:
            columns = {
                row[0]
                for row in conn.execute(
                    text(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_schema='features' AND table_name='orders_analytical'"
                    )
                )
            }
        assert "order_delivered_customer_date" not in columns
        with get_engine().connect() as conn:
            outcomes = {
                row[0]
                for row in conn.execute(
                    text(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_schema='features' AND table_name='order_outcomes'"
                    )
                )
            }
        assert "order_delivered_customer_date" in outcomes
