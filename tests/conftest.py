"""Shared fixtures.

Two kinds of test live in this repo and they need different guards:

* **unit** — pure Python, no Postgres, no dataset. These must run anywhere, including CI.
* **integration** — needs the live Compose stack, and sometimes the real Olist download.
  CI has neither until Phase 10, and it will never have the dataset: it is 121 MB and
  CC BY-NC-SA (PLAN.md §10.3). These skip cleanly rather than fail.

Skipping is done by *probing*, not by checking an environment variable, so a developer who
forgets ``make up`` gets a skip with a reason instead of a connection traceback.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from src.etl.load_raw import SPECS, count_csv_rows, raw_data_dir


@pytest.fixture(scope="session")
def data_dir() -> Path:
    """Directory holding the nine raw CSVs, skipping if the download is absent."""
    directory = raw_data_dir()
    missing = [s.csv_name for s in SPECS if not (directory / s.csv_name).exists()]
    if missing:
        pytest.skip(
            f"real Olist dataset not present in {directory}/ "
            f"(missing {len(missing)} of {len(SPECS)} files); "
            "run python -m scripts.download_data for instructions"
        )
    return directory


@pytest.fixture(scope="session")
def engine():
    """A SQLAlchemy engine, skipping if Postgres is not reachable."""
    from src.db import get_engine

    try:
        eng = get_engine()
        with eng.connect() as conn:
            conn.execute(text("select 1"))
    except (SQLAlchemyError, OSError) as exc:
        pytest.skip(f"Postgres not reachable ({type(exc).__name__}); run `make up` first")
    return eng


@pytest.fixture(scope="session")
def csv_row_counts(data_dir: Path) -> dict[str, int]:
    """Authoritative row count per table, parsed from the CSVs.

    Session-scoped because parsing 1.55 M rows — the geolocation file alone is 1 M — is
    not something to repeat per test. Counted with :mod:`csv`, never ``wc -l``
    (CLAUDE.md invariant 11).
    """
    return {spec.table: count_csv_rows(data_dir / spec.csv_name) for spec in SPECS}


@pytest.fixture(scope="session")
def db_row_counts(engine, csv_row_counts) -> dict[str, int]:
    """Row count per table as Postgres currently holds it.

    Depends on ``csv_row_counts`` only to guarantee the dataset guard runs first, so a
    machine without the download skips rather than reporting confusing zeros.
    """
    _ = csv_row_counts
    with engine.connect() as conn:
        return {
            spec.table: conn.execute(text(f"select count(*) from {spec.qualified}")).scalar_one()
            for spec in SPECS
        }


@pytest.fixture
def minimal_frame() -> pd.DataFrame:
    """A small frame that satisfies ORDERS_ANALYTICAL_SCHEMA, for contract unit tests.

    100 rows with 7 positives, so the §4.3 positive-rate band is comfortably satisfied and a
    test can break it deliberately. Columns come from ``COLUMN_NAMES``, so adding a column to
    the table without adding it here fails immediately rather than silently skipping
    validation.
    """
    from src.etl.build_orders import COLUMN_NAMES

    n = 100
    purchase = pd.to_datetime("2017-06-15") + pd.to_timedelta(np.arange(n), unit="D")
    data = {
        "order_id": [f"order{i:04d}" for i in range(n)],
        "customer_id": [f"cust{i:04d}" for i in range(n)],
        "customer_unique_id": [f"uniq{i:04d}" for i in range(n)],
        "order_purchase_timestamp": purchase,
        "purchase_month": purchase.to_period("M").to_timestamp(),
        "order_estimated_delivery_date": purchase + pd.Timedelta(days=20),
        "promised_days": np.full(n, 20),
        "is_late": np.array([True] * 7 + [False] * (n - 7)),
        "n_items": np.ones(n, dtype=int),
        "n_distinct_products": np.ones(n, dtype=int),
        "n_distinct_sellers": np.ones(n, dtype=int),
        "n_seller_states": np.ones(n, dtype=int),
        "total_price": np.full(n, 100.0),
        "max_price": np.full(n, 100.0),
        "total_freight": np.full(n, 10.0),
        "total_weight_g": np.full(n, 500.0),
        "total_volume_cm3": np.full(n, 1000.0),
        "dominant_category": ["cama_mesa_banho"] * n,
        "dominant_category_english": ["bed_bath_table"] * n,
        "max_shipping_limit_date": purchase + pd.Timedelta(days=5),
        "days_to_shipping_limit": np.full(n, 5.0),
        "dominant_payment_type": ["credit_card"] * n,
        "max_installments": np.full(n, 3),
        "total_payment_value": np.full(n, 110.0),
        "n_payment_methods": np.ones(n, dtype=int),
        "customer_state": ["SP"] * n,
        "customer_zip_code_prefix": ["01001"] * n,
        "customer_lat": np.full(n, -23.55),
        "customer_lng": np.full(n, -46.63),
        "customer_geo_source": ["zip"] * n,
        "seller_id": [f"sell{i:04d}" for i in range(n)],
        "seller_state": ["RJ"] * n,
        "seller_zip_code_prefix": ["20010"] * n,
        "seller_lat": np.full(n, -22.91),
        "seller_lng": np.full(n, -43.17),
        "seller_geo_source": ["zip"] * n,
        "customer_seller_distance_km": np.full(n, 360.7),
        "route": ["RJ->SP"] * n,
    }
    missing = set(COLUMN_NAMES) - set(data)
    extra = set(data) - set(COLUMN_NAMES)
    assert not missing, f"minimal_frame is missing declared columns: {sorted(missing)}"
    assert not extra, f"minimal_frame has undeclared columns: {sorted(extra)}"
    return pd.DataFrame(data)[list(COLUMN_NAMES)]


@pytest.fixture
def tiny_csv(tmp_path: Path):
    """Factory writing a small CSV, for tests that must not touch the real files."""

    def _write(name: str, header: list[str], rows: list[list[str]]) -> Path:
        path = tmp_path / name
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)
        return path

    return _write
