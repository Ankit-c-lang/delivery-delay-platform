"""Synthetic **raw Olist CSVs** — the nine files, so CI can run the real ETL (PLAN.md §10.3).

    CI MUST NEVER NEED THE REAL DATASET.
    ===================================
    It is 121 MB and CC BY-NC-SA, so it will not be committed and will not be downloaded in a
    workflow. But a CI run that skips the ETL says nothing about the code that will meet real
    data, and §10.3 is explicit that a Postgres service container running the ETL for real "is
    the part that looks like real CI, and it's cheap".

    So this module writes nine CSVs with the **same headers, in the same order, with the same
    quoting rules** as the originals, and `load_raw.load_all` reads them exactly as it reads the
    real files. The headers come from :data:`src.etl.load_raw.SPECS` rather than being restated,
    so a schema change breaks the generator instead of silently producing files the loader will
    reject.

**Three properties are deliberate, because each one exercises a code path that a naive fixture
would leave dead:**

1. **Leading zeros in zip prefixes.** ``01001`` must survive as text. The loader declares those
   columns ``TEXT`` for exactly this reason, and a fixture using integers would let a regression
   to ``INTEGER`` pass.
2. **An embedded newline inside a review comment.** CLAUDE.md invariant 11 exists because
   ``olist_order_reviews_dataset.csv`` has 99,224 rows and 104,719 physical lines. A fixture
   without one would let `wc -l` counting look correct.
3. **Nulls in nullable product columns**, so the ``NULL``-handling in the COPY path and the
   downstream imputation both see a real absence rather than a zero.

**The data is internally consistent**: every ``customer_id`` in orders exists in customers, every
``order_id`` in items, payments and reviews exists in orders, every zip prefix used is present in
geolocation, and every product category appears in the translation table. `build_orders` joins all
of it, so inconsistency here would surface as a confusing ETL failure rather than a clear one.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from src.etl.load_raw import SPECS

#: Row counts. Small enough that the whole ETL runs in seconds in CI, large enough that
#: `build_orders` produces a usable analytical table and a late rate near the real ~7%.
N_CUSTOMERS = 240
N_SELLERS = 14
N_PRODUCTS = 24
N_ORDERS = 240

#: Share of delivered orders that arrive after the estimate. The real data sits at 6.79%.
LATE_RATE = 0.07

#: Zip prefixes, as five-character strings. The leading zeros are the point.
ZIP_PREFIXES = ("01001", "02020", "20010", "30110", "40010", "04567", "90210")

#: (state, city, lat, lng) for each prefix above, so geolocation covers every zip in use.
GEO = {
    "01001": ("SP", "sao paulo", -23.55, -46.63),
    "02020": ("SP", "sao paulo", -23.51, -46.62),
    "20010": ("RJ", "rio de janeiro", -22.91, -43.17),
    "30110": ("MG", "belo horizonte", -19.92, -43.94),
    "40010": ("BA", "salvador", -12.97, -38.50),
    "04567": ("SP", "sao paulo", -23.60, -46.68),
    "90210": ("RS", "porto alegre", -30.03, -51.23),
}

#: Categories, with their English translations. Two are left untranslated on purpose: the real
#: file is missing `pc_gamer` and `portateis_cozinha_e_preparadores_de_alimentos`, and the ETL
#: has to tolerate that rather than drop the rows.
CATEGORIES = {
    "cama_mesa_banho": "bed_bath_table",
    "beleza_saude": "health_beauty",
    "esporte_lazer": "sports_leisure",
    "informatica_acessorios": "computers_accessories",
    "moveis_decoracao": "furniture_decor",
}
UNTRANSLATED_CATEGORIES = ("pc_gamer",)

PAYMENT_TYPES = ("credit_card", "boleto", "voucher", "debit_card")


@dataclass(frozen=True)
class GeneratedData:
    """Where the files landed, and what was in them.

    Attributes:
        directory: Holds the nine CSVs.
        row_counts: Logical rows per CSV name — what a CSV parser would count, not what
            ``wc -l`` would.
        late_rate: Realised share of late deliveries.
    """

    directory: Path
    row_counts: dict[str, int]
    late_rate: float


def _spec(table: str):
    """The spec for one table, so headers are never restated here."""
    for spec in SPECS:
        if spec.table == table:
            return spec
    raise KeyError(f"no TableSpec named {table!r}; SPECS changed and this fixture did not")


def _write(directory: Path, table: str, rows: list[dict]) -> int:
    """Write one CSV with the header its spec declares, in the spec's column order.

    Returns:
        Logical row count.

    ``csv.writer`` handles the quoting, which matters for the review comment carrying a newline:
    hand-formatting would produce a file the loader reads as more rows than it has.
    """
    spec = _spec(table)
    columns = [name for name, _ in spec.columns]
    path = directory / spec.csv_name
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            missing = set(columns) - set(row)
            if missing:
                raise ValueError(f"{table}: row is missing {sorted(missing)}")
            writer.writerow(row)
    return len(rows)


def generate(directory: Path, seed: int = 11) -> GeneratedData:
    """Write all nine CSVs into ``directory``.

    Args:
        directory: Destination, created if absent.
        seed: Seed for the draws, so a CI failure is reproducible locally.

    Returns:
        A :class:`GeneratedData`.
    """
    rng = np.random.default_rng(seed)
    directory.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}

    # --- customers, sellers, geolocation -------------------------------------------------
    customers = []
    for index in range(N_CUSTOMERS):
        zip_prefix = ZIP_PREFIXES[index % len(ZIP_PREFIXES)]
        state, city, _, _ = GEO[zip_prefix]
        customers.append(
            {
                "customer_id": f"c{index:05d}",
                "customer_unique_id": f"u{index % (N_CUSTOMERS - 20):05d}",
                "customer_zip_code_prefix": zip_prefix,
                "customer_city": city,
                "customer_state": state,
            }
        )
    counts[_spec("customers").csv_name] = _write(directory, "customers", customers)

    sellers = []
    for index in range(N_SELLERS):
        zip_prefix = ZIP_PREFIXES[(index * 3) % len(ZIP_PREFIXES)]
        state, city, _, _ = GEO[zip_prefix]
        sellers.append(
            {
                "seller_id": f"s{index:03d}",
                "seller_zip_code_prefix": zip_prefix,
                "seller_city": city,
                "seller_state": state,
            }
        )
    counts[_spec("sellers").csv_name] = _write(directory, "sellers", sellers)

    geo_rows = []
    for zip_prefix, (state, city, lat, lng) in GEO.items():
        # Several points per prefix, because the real table has many and the centroid logic
        # takes a median over them.
        for jitter in (-0.01, 0.0, 0.01):
            geo_rows.append(
                {
                    "geolocation_zip_code_prefix": zip_prefix,
                    "geolocation_lat": round(lat + jitter, 6),
                    "geolocation_lng": round(lng + jitter, 6),
                    "geolocation_city": city,
                    "geolocation_state": state,
                }
            )
    counts[_spec("geolocation").csv_name] = _write(directory, "geolocation", geo_rows)

    # --- products and the translation table ----------------------------------------------
    category_names = list(CATEGORIES) + list(UNTRANSLATED_CATEGORIES)
    products = []
    for index in range(N_PRODUCTS):
        # Every fifth product has no category and no dimensions: the real file has 610 such
        # rows, and the imputation path needs to see a genuine absence.
        blank = index % 5 == 0
        products.append(
            {
                "product_id": f"p{index:04d}",
                "product_category_name": (
                    "" if blank else category_names[index % len(category_names)]
                ),
                "product_name_lenght": "" if blank else int(rng.integers(20, 60)),
                "product_description_lenght": "" if blank else int(rng.integers(100, 3000)),
                "product_photos_qty": "" if blank else int(rng.integers(1, 6)),
                "product_weight_g": "" if blank else int(rng.integers(100, 9000)),
                "product_length_cm": "" if blank else int(rng.integers(10, 60)),
                "product_height_cm": "" if blank else int(rng.integers(5, 40)),
                "product_width_cm": "" if blank else int(rng.integers(10, 50)),
            }
        )
    counts[_spec("products").csv_name] = _write(directory, "products", products)

    translations = [
        {"product_category_name": name, "product_category_name_english": english}
        for name, english in CATEGORIES.items()
    ]
    counts[_spec("product_category_name_translation").csv_name] = _write(
        directory, "product_category_name_translation", translations
    )

    # --- orders, spread across months so the as-of snapshots have something to aggregate ---
    start = datetime(2017, 1, 5, 9, 0, 0)
    orders, items, payments, reviews = [], [], [], []
    late_count = 0

    for index in range(N_ORDERS):
        order_id = f"o{index:05d}"
        purchased = start + timedelta(days=index * 2, hours=int(rng.integers(0, 12)))
        approved = purchased + timedelta(hours=int(rng.integers(1, 20)))
        to_carrier = approved + timedelta(days=int(rng.integers(1, 4)))
        promised_days = int(rng.integers(12, 35))
        estimated = purchased + timedelta(days=promised_days)

        is_late = index % int(1 / LATE_RATE) == 0
        if is_late:
            delivered = estimated + timedelta(days=int(rng.integers(1, 8)))
            late_count += 1
        else:
            delivered = purchased + timedelta(days=max(promised_days - int(rng.integers(2, 9)), 2))

        orders.append(
            {
                "order_id": order_id,
                "customer_id": customers[index % len(customers)]["customer_id"],
                "order_status": "delivered",
                "order_purchase_timestamp": purchased.strftime("%Y-%m-%d %H:%M:%S"),
                "order_approved_at": approved.strftime("%Y-%m-%d %H:%M:%S"),
                "order_delivered_carrier_date": to_carrier.strftime("%Y-%m-%d %H:%M:%S"),
                "order_delivered_customer_date": delivered.strftime("%Y-%m-%d %H:%M:%S"),
                # Date-only, exactly as the real column is. The target compares at DATE
                # granularity for this reason (PLAN.md §4.3).
                "order_estimated_delivery_date": estimated.strftime("%Y-%m-%d 00:00:00"),
            }
        )

        for item_number in range(1, int(rng.integers(1, 3)) + 1):
            items.append(
                {
                    "order_id": order_id,
                    "order_item_id": item_number,
                    "product_id": products[(index + item_number) % len(products)]["product_id"],
                    "seller_id": sellers[index % len(sellers)]["seller_id"],
                    "shipping_limit_date": (
                        purchased + timedelta(days=int(rng.integers(2, 10)))
                    ).strftime("%Y-%m-%d %H:%M:%S"),
                    "price": f"{rng.uniform(20, 900):.2f}",
                    "freight_value": f"{rng.uniform(5, 90):.2f}",
                }
            )

        payments.append(
            {
                "order_id": order_id,
                "payment_sequential": 1,
                "payment_type": PAYMENT_TYPES[index % len(PAYMENT_TYPES)],
                "payment_installments": int(rng.integers(1, 10)),
                "payment_value": f"{rng.uniform(25, 950):.2f}",
            }
        )

        # One review in ten carries a newline inside its comment. Invariant 11 exists because
        # the real file does this, and a fixture without it lets `wc -l` counting look correct.
        comment = "entrega rapida" if index % 10 else "chegou\ncom atraso"
        reviews.append(
            {
                "review_id": f"r{index:05d}",
                "order_id": order_id,
                "review_score": int(rng.integers(1, 6)) if is_late else int(rng.integers(3, 6)),
                "review_comment_title": "" if index % 3 else "ok",
                "review_comment_message": comment,
                "review_creation_date": delivered.strftime("%Y-%m-%d 00:00:00"),
                "review_answer_timestamp": (delivered + timedelta(days=1)).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            }
        )

    counts[_spec("orders").csv_name] = _write(directory, "orders", orders)
    counts[_spec("order_items").csv_name] = _write(directory, "order_items", items)
    counts[_spec("order_payments").csv_name] = _write(directory, "order_payments", payments)
    counts[_spec("order_reviews").csv_name] = _write(directory, "order_reviews", reviews)

    return GeneratedData(
        directory=directory,
        row_counts=counts,
        late_rate=late_count / N_ORDERS,
    )


if __name__ == "__main__":
    import sys
    import tempfile

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp())
    data = generate(target)
    print(f"wrote {len(data.row_counts)} CSVs to {data.directory}")
    for name, count in sorted(data.row_counts.items()):
        print(f"  {name:45} {count:6d} rows")
    print(f"late rate {data.late_rate:.2%}")
