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
