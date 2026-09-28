"""Tests for the raw CSV -> Postgres load (PLAN.md §13 Phase 1).

What these actually defend against, in order of how much damage each would do:

1. A row-count expectation taken from ``wc -l``, which is wrong for two of the nine files.
2. ``TIMESTAMPTZ`` instead of ``TIMESTAMP``, which shifts every Olist timestamp by the
   session offset and silently corrupts the §4.3 target.
3. ``INTEGER`` zip prefixes, which destroy the leading zero on a quarter of all rows.
4. A non-idempotent load that duplicates rows on re-run.
5. A re-download with reordered columns, which ``COPY`` would happily load into the wrong
   columns because it matches by position.
"""

from __future__ import annotations

from typing import ClassVar

import pytest
from sqlalchemy import text

from src.etl.load_raw import (
    SPECS,
    count_csv_rows,
    csv_header,
    load_all,
    validate_header,
    write_reconciliation_report,
)


# Marks are applied per class, not per module: TestHeaderValidation and TestSpecIntegrity
# are pure unit tests and must keep running in CI, which has neither Postgres nor the
# dataset.
#
# Published counts for the Olist release. Hard-coded on purpose: if a re-download differs,
# these tests should fail loudly rather than quietly re-derive a new "expected" value.
def wc_l(path) -> int:
    """Replicate ``wc -l``: count newline bytes, nothing else.

    Deliberately not ``sum(1 for _ in open(path))`` — Python's line iterator yields the
    final newline-less line as a line, so it silently disagrees with ``wc -l`` on exactly
    the files this test is about.
    """
    return path.read_bytes().count(b"\n")


PUBLISHED_COUNTS = {
    "orders": 99_441,
    "customers": 99_441,
    "order_items": 112_650,
    "order_payments": 103_886,
    "order_reviews": 99_224,
    "products": 32_951,
    "sellers": 3_095,
    "geolocation": 1_000_163,
    "product_category_name_translation": 71,
}


class TestSchema:
    """The tables exist, with the types the DDL promised."""

    pytestmark = pytest.mark.integration

    def test_every_expected_table_exists(self, engine):
        with engine.connect() as conn:
            present = {
                row[0]
                for row in conn.execute(
                    text(
                        "select table_name from information_schema.tables "
                        "where table_schema='raw'"
                    )
                )
            }
        expected = {spec.table for spec in SPECS}
        assert expected <= present, f"missing raw tables: {sorted(expected - present)}"

    def test_timestamps_are_naive(self, engine):
        """No ``timestamptz`` anywhere in the raw schema.

        The Olist CSVs carry naive local times with no offset. Postgres would interpret a
        ``timestamptz`` in the session ``TimeZone`` and shift it on read, which flips
        borderline orders between late and on-time because
        ``order_estimated_delivery_date`` is always midnight.
        """
        with engine.connect() as conn:
            offenders = conn.execute(
                text(
                    "select table_name, column_name from information_schema.columns "
                    "where table_schema='raw' and data_type = 'timestamp with time zone'"
                )
            ).all()
        assert offenders == [], f"timestamptz columns found: {offenders}"

    def test_timestamp_columns_are_all_present(self, engine):
        with engine.connect() as conn:
            found = conn.execute(
                text(
                    "select count(*) from information_schema.columns where table_schema='raw' "
                    "and data_type = 'timestamp without time zone'"
                )
            ).scalar_one()
        assert found == 8

    def test_money_is_numeric_not_float(self, engine):
        """Money columns must be exact. ``price`` sums feed order value features."""
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    "select table_name || '.' || column_name, data_type, numeric_scale "
                    "from information_schema.columns where table_schema='raw' "
                    "and column_name in ('price','freight_value','payment_value')"
                )
            ).all()
        assert len(rows) == 3
        for name, data_type, scale in rows:
            assert data_type == "numeric", f"{name} is {data_type}, not numeric"
            assert scale == 2, f"{name} has scale {scale}, not 2"

    def test_zip_prefixes_are_text(self, engine):
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    "select table_name || '.' || column_name, data_type "
                    "from information_schema.columns where table_schema='raw' "
                    "and column_name like '%zip_code_prefix'"
                )
            ).all()
        assert len(rows) == 3
        for name, data_type in rows:
            assert data_type == "text", f"{name} is {data_type}; leading zeros would be lost"


class TestRowCounts:
    """Counts reconcile three ways: the CSV parser, the database, and Olist's published
    figures."""

    pytestmark: ClassVar = [pytest.mark.integration, pytest.mark.slow]

    def test_db_counts_match_csv_counts(self, db_row_counts, csv_row_counts):
        assert db_row_counts == csv_row_counts

    def test_csv_counts_match_published_counts(self, csv_row_counts):
        assert csv_row_counts == PUBLISHED_COUNTS

    def test_reviews_count_is_not_the_line_count(self, data_dir):
        """The ``wc -l`` trap, stated explicitly.

        ``olist_order_reviews_dataset.csv`` has 104,719 newline-delimited lines after the
        header but only 99,224 records: 3,852 review comments contain newlines inside
        quoted fields. Expecting 104,719 here fails, and the tempting fix is to break the
        loader.
        """
        path = data_dir / "olist_order_reviews_dataset.csv"
        parsed = count_csv_rows(path)
        wc_data_rows = wc_l(path) - 1  # -1 for the header
        assert parsed == 99_224
        assert wc_data_rows == 104_719
        assert parsed < wc_data_rows, "wc -l OVERSTATES this file by 5,495"

    def test_translation_count_is_not_the_line_count(self, data_dir):
        """The opposite trap: this file has no trailing newline, so lines understate by 1."""
        path = data_dir / "product_category_name_translation.csv"
        parsed = count_csv_rows(path)
        wc_data_rows = wc_l(path) - 1  # -1 for the header
        assert parsed == 71
        assert wc_data_rows == 70
        assert parsed > wc_data_rows, "wc -l UNDERSTATES this file by 1"


class TestFidelity:
    """The values in Postgres are the values in the files."""

    pytestmark = pytest.mark.integration

    def test_leading_zeros_preserved(self, engine):
        with engine.connect() as conn:
            counts = conn.execute(
                text(
                    "select (select count(*) from raw.customers "
                    "          where customer_zip_code_prefix like '0%'),"
                    "       (select count(*) from raw.sellers "
                    "          where seller_zip_code_prefix like '0%'),"
                    "       (select count(*) from raw.geolocation "
                    "          where geolocation_zip_code_prefix like '0%')"
                )
            ).one()
        assert counts == (23_995, 1_027, 245_733)

    def test_zip_prefixes_are_all_five_characters(self, engine):
        with engine.connect() as conn:
            bad = conn.execute(
                text(
                    "select count(*) from raw.customers "
                    "where length(customer_zip_code_prefix) <> 5"
                )
            ).scalar_one()
        assert bad == 0

    def test_embedded_newlines_survived_copy(self, engine):
        with engine.connect() as conn:
            found = conn.execute(
                text(
                    r"select count(*) from raw.order_reviews "
                    r"where review_comment_message like E'%\n%'"
                )
            ).scalar_one()
        assert found == 3_852

    def test_bom_did_not_leak_into_the_first_column(self, engine):
        """``product_category_name_translation.csv`` starts with a UTF-8 BOM.

        ``COPY ... HEADER true`` discards the header line, and the BOM with it. If the BOM
        ever reached a value, every join on ``product_category_name`` would miss one
        category silently.
        """
        with engine.connect() as conn:
            rows, with_bom = conn.execute(
                text(
                    "select count(*), count(*) filter (where product_category_name like '%'||"
                    "chr(65279)||'%') from raw.product_category_name_translation"
                )
            ).one()
        assert rows == 71
        assert with_bom == 0

    def test_nulls_are_null_not_empty_string(self, engine):
        """Olist writes nulls as bare ``,,`` so ``FORMAT csv`` yields NULL, not ``''``.

        An empty string in ``review_comment_title`` would make "has a comment" logic wrong,
        and an empty string could never have reached a ``TIMESTAMP`` column at all.
        """
        with engine.connect() as conn:
            delivered_null, approved_null = conn.execute(
                text(
                    "select count(*) filter (where order_delivered_customer_date is null),"
                    "       count(*) filter (where order_approved_at is null) from raw.orders"
                )
            ).one()
            title_null, title_blank = conn.execute(
                text(
                    "select count(*) filter (where review_comment_title is null),"
                    "       count(*) filter (where review_comment_title = '') "
                    "from raw.order_reviews"
                )
            ).one()
        assert (delivered_null, approved_null) == (2_965, 160)
        assert title_null == 87_656
        assert title_blank == 0

    def test_purchase_timestamp_span(self, engine):
        """Guards against a truncated download, which counts alone would not catch."""
        with engine.connect() as conn:
            lo, hi = conn.execute(
                text(
                    "select min(order_purchase_timestamp), max(order_purchase_timestamp) "
                    "from raw.orders"
                )
            ).one()
        assert str(lo) == "2016-09-04 21:15:19"
        assert str(hi) == "2018-10-17 17:30:18"


class TestIdempotence:
    """``make load-raw`` twice in a row produces identical counts (PLAN.md Phase 1
    completion)."""

    pytestmark: ClassVar = [pytest.mark.integration, pytest.mark.slow]

    def test_reload_leaves_counts_unchanged(self, engine, data_dir, db_row_counts):
        before = dict(db_row_counts)
        load_all(data_dir)
        with engine.connect() as conn:
            after = {
                spec.table: conn.execute(
                    text(f"select count(*) from {spec.qualified}")
                ).scalar_one()
                for spec in SPECS
            }
        assert after == before

    def test_reload_is_a_replacement_not_an_append(self, engine, data_dir):
        """A missing ``TRUNCATE`` would double the rows, and a primary key would only catch
        it on eight of the nine tables — ``geolocation`` has no key at all."""
        with engine.connect() as conn:
            before = conn.execute(text("select count(*) from raw.geolocation")).scalar_one()
        load_all(data_dir)
        with engine.connect() as conn:
            after = conn.execute(text("select count(*) from raw.geolocation")).scalar_one()
        assert after == before == 1_000_163


class TestHeaderValidation:
    """Unit tests: no Postgres, no dataset. These run in CI."""

    def test_accepts_a_matching_header(self, tiny_csv):
        spec = next(s for s in SPECS if s.table == "sellers")
        path = tiny_csv("sellers.csv", spec.column_names, [["s1", "01001", "sao paulo", "SP"]])
        validate_header(spec, path)

    def test_rejects_reordered_columns(self, tiny_csv):
        """``COPY`` matches by position, so reordering silently loads the wrong data."""
        spec = next(s for s in SPECS if s.table == "sellers")
        swapped = [spec.column_names[1], spec.column_names[0], *spec.column_names[2:]]
        path = tiny_csv("sellers.csv", swapped, [["01001", "s1", "sao paulo", "SP"]])
        with pytest.raises(ValueError, match="header does not match"):
            validate_header(spec, path)

    def test_rejects_a_renamed_column(self, tiny_csv):
        spec = next(s for s in SPECS if s.table == "sellers")
        renamed = ["seller_uuid", *spec.column_names[1:]]
        path = tiny_csv("sellers.csv", renamed, [["s1", "01001", "sao paulo", "SP"]])
        with pytest.raises(ValueError, match="header does not match"):
            validate_header(spec, path)

    def test_strips_the_utf8_bom_from_the_header(self, tmp_path):
        path = tmp_path / "bom.csv"
        path.write_bytes("﻿a,b\n1,2".encode())
        assert csv_header(path) == ["a", "b"]

    def test_counts_a_final_row_without_a_trailing_newline(self, tmp_path):
        """Every Olist file ends without a newline. A hand-rolled counter would miss the
        last row."""
        path = tmp_path / "no_trailing.csv"
        path.write_text("a,b\n1,2\n3,4", encoding="utf-8")
        assert count_csv_rows(path) == 2

    def test_wc_l_undercounts_a_file_with_no_trailing_newline(self, tmp_path):
        """Pins the helper, so the two row-count tests keep comparing like with like."""
        with_newline = tmp_path / "with.csv"
        with_newline.write_text("a,b\n1,2\n", encoding="utf-8")
        without = tmp_path / "without.csv"
        without.write_text("a,b\n1,2", encoding="utf-8")
        assert wc_l(with_newline) == 2
        assert wc_l(without) == 1
        assert count_csv_rows(with_newline) == count_csv_rows(without) == 1

    def test_counts_records_not_lines(self, tmp_path):
        path = tmp_path / "embedded.csv"
        path.write_text('a,b\n1,"line one\nline two"\n2,plain\n', encoding="utf-8")
        assert count_csv_rows(path) == 2


class TestFailureHandling:
    """The loader must fail loudly and leave nothing half-written (CLAUDE.md conventions)."""

    pytestmark: ClassVar = [pytest.mark.integration, pytest.mark.slow]

    def test_a_missing_csv_rolls_back_the_whole_load(self, engine, data_dir, tmp_path):
        """One transaction means a failure on file nine undoes files one to eight.

        Built by symlinking eight of the nine files into a temp directory and omitting the
        last one in ``SPECS`` order. Eight tables therefore get TRUNCATEd and reloaded
        before the missing file raises, so the row counts afterwards prove the rollback
        rather than merely proving nothing was attempted.
        """
        omitted = SPECS[-1]
        for spec in SPECS[:-1]:
            (tmp_path / spec.csv_name).symlink_to((data_dir / spec.csv_name).resolve())
        assert not (tmp_path / omitted.csv_name).exists()

        with engine.connect() as conn:
            before = {
                spec.table: conn.execute(
                    text(f"select count(*) from {spec.qualified}")
                ).scalar_one()
                for spec in SPECS
            }

        with pytest.raises(FileNotFoundError, match=omitted.csv_name):
            load_all(tmp_path)

        with engine.connect() as conn:
            after = {
                spec.table: conn.execute(
                    text(f"select count(*) from {spec.qualified}")
                ).scalar_one()
                for spec in SPECS
            }
        assert after == before, "a failed load left the raw schema modified"

    def test_a_header_mismatch_rolls_back_the_whole_load(self, engine, data_dir, tmp_path):
        """Same guarantee for the other failure mode: a re-download with a renamed column."""
        for spec in SPECS[:-1]:
            (tmp_path / spec.csv_name).symlink_to((data_dir / spec.csv_name).resolve())
        bad = SPECS[-1]
        (tmp_path / bad.csv_name).write_text("wrong,header\na,b\n", encoding="utf-8")

        with engine.connect() as conn:
            before = conn.execute(text("select count(*) from raw.orders")).scalar_one()

        with pytest.raises(ValueError, match="header does not match"):
            load_all(tmp_path)

        with engine.connect() as conn:
            after = conn.execute(text("select count(*) from raw.orders")).scalar_one()
        assert after == before


class TestReconciliationReport:
    """Unit tests on the report writer."""

    def test_renders_every_table_and_a_total(self, tmp_path):
        results = [(spec, 10, 10, 0.5) for spec in SPECS]
        path = tmp_path / "report.md"
        write_reconciliation_report(results, path)
        body = path.read_text(encoding="utf-8")
        for spec in SPECS:
            assert spec.csv_name in body
            assert spec.qualified in body
        assert f"**{10 * len(SPECS):,}**" in body
        assert "**NO**" not in body

    def test_flags_a_mismatch_loudly(self, tmp_path):
        results = [(SPECS[0], 10, 9, 0.1)]
        path = tmp_path / "report.md"
        write_reconciliation_report(results, path)
        assert "**NO**" in path.read_text(encoding="utf-8")


class TestSpecIntegrity:
    """Unit tests on the spec table itself."""

    def test_no_duplicate_tables_or_files(self):
        assert len({s.table for s in SPECS}) == len(SPECS)
        assert len({s.csv_name for s in SPECS}) == len(SPECS)

    def test_primary_key_columns_are_declared_columns(self):
        for spec in SPECS:
            unknown = set(spec.primary_key) - set(spec.column_names)
            assert not unknown, f"{spec.table}: PK references unknown columns {unknown}"

    def test_primary_key_columns_are_not_null(self):
        """A nullable primary-key column is a contradiction Postgres would silently fix."""
        for spec in SPECS:
            for key_col in spec.primary_key:
                sql_type = dict(spec.columns)[key_col]
                assert "NOT NULL" in sql_type, f"{spec.table}.{key_col} is a PK but nullable"

    def test_reviews_and_geolocation_key_choices_are_deliberate(self):
        """Documents two measured facts as assertions, so a later 'tidy-up' cannot
        reintroduce a key the data does not support."""
        reviews = next(s for s in SPECS if s.table == "order_reviews")
        geo = next(s for s in SPECS if s.table == "geolocation")
        assert reviews.primary_key == ("review_id", "order_id")
        assert geo.primary_key == ()

    def test_copy_statement_uses_csv_format_and_an_explicit_column_list(self):
        for spec in SPECS:
            sql = spec.copy_sql()
            assert "FORMAT csv" in sql
            assert "HEADER true" in sql
            assert ", ".join(spec.column_names) in sql
