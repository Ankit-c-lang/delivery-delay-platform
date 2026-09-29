"""As-of monthly aggregate snapshots — the leakage-critical module (PLAN.md §4.5, §18 A1).

**The rule.** A snapshot for calendar month *M* aggregates only orders whose
``order_delivered_customer_date < M``. Not ``order_purchase_timestamp < M``.

**Why the distinction decides whether this project is honest.** Late orders take longer to
resolve — that is what makes them late. So the orders still in flight at any cutoff are far
more likely to end up late than the ones already resolved. Measured here:

===========  =================  =========  ======================  =========  =====
Snapshot M   resolved by M      late rate  still in flight at M    late rate  ratio
===========  =================  =========  ======================  =========  =====
2017-07      12,839             3.24%      1,093                   11.25%     3.5x
2018-01      40,930             4.30%      2,496                   27.68%     6.4x
2018-04      60,201             5.99%      3,852                   39.10%     6.5x
===========  =================  =========  ======================  =========  =====

The purchase-time rule admits exactly that in-flight population, and it can only know their
outcome by reading the future. At 2017-05, the first training month, the leaky global late
rate is 4.66% against a true 2.37% — nearly double. The distortion is worst where history is
thinnest, which is also where each observation carries the most weight, and §4.5 calls this
the most predictive feature family available. That combination is why it is worth this much
care.

**Using post-purchase columns here is not a §4.2 violation.** The denylist forbids a column
of the order *being predicted*. These are columns of *other* orders, resolved before the
prediction moment. ``is_late`` is derived the same way; that is the premise of §4.5. The
columns come from ``features.order_outcomes``, the quarantine table (DECISIONS.md D19), and
this module is its only reader.

**Cold start.** 7.56% of training rows have a seller with no resolved history, and it does
not fade after the warm-up — new sellers keep joining the marketplace, so the rate stays
between 4.6% and 8.8% every month. Values fall back entity -> entity's state -> global and the
row is flagged ``{entity}_is_new``. The fallback supplies *rates*, never *volumes*:
``order_count`` stays the entity's own, which is 0 for a cold entity.

Run with::

    python -m src.features.history
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sqlalchemy.engine import Connection

# sqlalchemy, src.db and src.etl are imported INSIDE the functions that need them. Every one of
# those functions is on the build path; none is reached when serving a prediction. At module scope
# they dragged sqlalchemy, psycopg2 and pandera into the API image, because this module is imported
# by the pyfunc wrapper (DECISIONS.md D38).

logger = logging.getLogger(__name__)

FEATURES_SCHEMA = "features"

#: The four aggregates every snapshot level carries.
METRIC_COLUMNS: tuple[str, ...] = (
    "order_count",
    "late_rate",
    "avg_delivery_days",
    "avg_handling_days",
)

#: Metrics a cold entity may inherit from a coarser level. Rates and averages generalise
#: across a state; a volume does not.
FALLBACK_METRICS: tuple[str, ...] = ("late_rate", "avg_delivery_days", "avg_handling_days")

#: Internal constant group key used to aggregate the global level with the same code path.
_GLOBAL_KEY = "_global"


@dataclass(frozen=True)
class LevelSpec:
    """One snapshot level.

    Attributes:
        key: Column to group by, or ``None`` for the global level.
        table: Table name inside the ``features`` schema.
    """

    key: str | None
    table: str


@dataclass(frozen=True)
class EntitySpec:
    """An entity family and its cold-start fallback chain.

    Attributes:
        name: Feature prefix, e.g. ``seller`` gives ``seller_late_rate_hist``.
        key: Column identifying the entity, in both the resolved and the order frame.
        table: Snapshot table for this entity.
        fallback_key: Coarser column to fall back to, or ``None`` to go straight to global.
        fallback_table: Snapshot table for that coarser level.
    """

    name: str
    key: str
    table: str
    fallback_key: str | None
    fallback_table: str | None


#: The global level. Last resort, and the only level guaranteed to have data once any order
#: anywhere has resolved.
GLOBAL_LEVEL = LevelSpec(key=None, table="global_monthly")

#: The three entity families of §4.5. Route falls back to the *customer* state because
#: delivery duration is dominated by the destination; category has no meaningful coarser
#: level, so it goes straight to global.
ENTITIES: tuple[EntitySpec, ...] = (
    EntitySpec(
        name="seller",
        key="seller_id",
        table="seller_monthly",
        fallback_key="seller_state",
        fallback_table="seller_state_monthly",
    ),
    EntitySpec(
        name="route",
        key="route",
        table="route_monthly",
        fallback_key="customer_state",
        fallback_table="customer_state_monthly",
    ),
    EntitySpec(
        name="category",
        key="dominant_category",
        table="category_monthly",
        fallback_key=None,
        fallback_table=None,
    ),
)

#: Everything the resolved frame must carry. Note what is absent: no purchase timestamp, so
#: the purchase-time rule is not merely avoided, it is unavailable.
RESOLVED_QUERY = f"""
SELECT a.seller_id,
       a.seller_state,
       a.route,
       a.customer_state,
       a.dominant_category,
       o.order_delivered_customer_date,
       o.is_late,
       o.delivery_days,
       o.handling_days
FROM {FEATURES_SCHEMA}.orders_analytical a
JOIN {FEATURES_SCHEMA}.order_outcomes o USING (order_id)
"""

ORDERS_QUERY = f"""
SELECT order_id, purchase_month, seller_id, seller_state, route, customer_state,
       dominant_category
FROM {FEATURES_SCHEMA}.orders_analytical
"""


def snapshot_months(start: pd.Timestamp, end_exclusive: pd.Timestamp) -> list[pd.Timestamp]:
    """Every calendar month start in ``[start, end_exclusive)``."""
    return list(
        pd.date_range(
            pd.Timestamp(start).to_period("M").to_timestamp(),
            pd.Timestamp(end_exclusive),
            freq="MS",
            inclusive="left",
        )
    )


def _aggregate(eligible: pd.DataFrame, key: str | None) -> pd.DataFrame:
    """Aggregate an already-filtered frame by ``key``.

    ``order_count`` uses ``size`` so it counts every eligible order, including those whose
    ``handling_days`` is NULL. The averages use ``mean``, which skips NaN — 184 real orders
    record a carrier handoff before their own purchase and Phase 2 stores those as NULL
    rather than clipping them, so treating NULL as 0.0 would make a seller look instant.

    ``dropna=True`` in the groupby matters for ``dominant_category``, which is NULL for 1,330
    orders: a missing category must not become a group of its own.
    """
    frame = eligible
    group_key = key
    if group_key is None:
        frame = frame.assign(**{_GLOBAL_KEY: 1})
        group_key = _GLOBAL_KEY

    out = (
        frame.groupby(group_key, dropna=True, observed=True)
        .agg(
            order_count=("is_late", "size"),
            late_rate=("is_late", "mean"),
            avg_delivery_days=("delivery_days", "mean"),
            avg_handling_days=("handling_days", "mean"),
        )
        .reset_index()
    )
    if key is None:
        out = out.drop(columns=[_GLOBAL_KEY])
    return out


def build_snapshots(
    resolved: pd.DataFrame, months: list[pd.Timestamp], key: str | None
) -> pd.DataFrame:
    """Build expanding as-of snapshots for one level.

    Args:
        resolved: Resolved orders. Must carry ``order_delivered_customer_date``, ``is_late``,
            ``delivery_days``, ``handling_days`` and ``key``.
        months: Snapshot months, each the first of a month.
        key: Column to group by, or ``None`` for the global level.

    Returns:
        One row per (key, snapshot_month) with :data:`METRIC_COLUMNS`. Months with no
        resolved history are **absent**, not zero-filled: a zero-filled row would read as an
        entity with a 0% late rate rather than one we know nothing about.

    The filter is ``<`` and never ``<=``. An order delivered at 00:00:00 on the first of *M*
    was not known before *M*.
    """
    delivered = resolved["order_delivered_customer_date"]
    columns = ([key] if key else []) + ["snapshot_month", *METRIC_COLUMNS]

    frames: list[pd.DataFrame] = []
    for month in months:
        eligible = resolved.loc[delivered < month]
        if eligible.empty:
            continue
        grouped = _aggregate(eligible, key)
        grouped["snapshot_month"] = month
        frames.append(grouped)

    if not frames:
        return pd.DataFrame({column: pd.Series(dtype="object") for column in columns})
    return pd.concat(frames, ignore_index=True)[columns]


def build_all_snapshots(
    resolved: pd.DataFrame, months: list[pd.Timestamp]
) -> dict[str, pd.DataFrame]:
    """Build every snapshot level, keyed by table name."""
    snapshots: dict[str, pd.DataFrame] = {}
    for spec in ENTITIES:
        snapshots[spec.table] = build_snapshots(resolved, months, spec.key)
        if spec.fallback_table and spec.fallback_table not in snapshots:
            snapshots[spec.fallback_table] = build_snapshots(resolved, months, spec.fallback_key)
    snapshots[GLOBAL_LEVEL.table] = build_snapshots(resolved, months, GLOBAL_LEVEL.key)
    return snapshots


def _lookup(orders: pd.DataFrame, key: str | None, snapshot: pd.DataFrame | None) -> pd.DataFrame:
    """Left-join one snapshot level onto the orders frame, on (key, purchase_month).

    Returns a frame indexed like ``orders`` with exactly :data:`METRIC_COLUMNS`, all NaN
    where the level has nothing for that row.
    """
    empty = pd.DataFrame(
        {
            column: pd.Series([float("nan")] * len(orders), index=orders.index)
            for column in METRIC_COLUMNS
        }
    )
    if snapshot is None or snapshot.empty:
        return empty

    left_on = ["purchase_month"] + ([key] if key else [])
    right_on = ["snapshot_month"] + ([key] if key else [])
    merged = orders.reset_index()[["index", *left_on]].merge(
        snapshot[[*right_on, *METRIC_COLUMNS]],
        how="left",
        left_on=left_on,
        right_on=right_on,
    )
    # A snapshot is keyed (entity, month), so a left join cannot fan out. Assert it anyway:
    # a duplicated key would silently multiply the feature matrix.
    if len(merged) != len(orders):
        raise RuntimeError(
            f"snapshot join fanned out for key={key!r}: {len(orders)} rows became {len(merged)}"
        )
    return merged.set_index("index")[list(METRIC_COLUMNS)].reindex(orders.index)


def attach_history(orders: pd.DataFrame, snapshots: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Attach as-of history features to an orders frame.

    Args:
        orders: One row per order, carrying ``purchase_month`` and every entity key.
        snapshots: Output of :func:`build_all_snapshots`.

    Returns:
        A new frame — the input is never mutated (pandas 3 copy-on-write, CLAUDE.md
        invariant 9) — with row count and order preserved, plus for each entity family
        ``{name}_{metric}_hist`` for all four metrics and a ``{name}_is_new`` flag.

    Each order joins to the snapshot for **its own purchase month**, which by construction
    contains only outcomes resolved before that month.

    ``{name}_is_new`` reflects the *entity's own* history, not whether a value was found. A
    cold entity is flagged even though the state or global level supplied its rates, because
    "we substituted a coarser estimate" is exactly what the model needs to know.
    """
    out = orders.copy()

    for spec in ENTITIES:
        own = _lookup(out, spec.key, snapshots.get(spec.table))
        is_new = own["order_count"].isna()

        values = own.copy()
        chain: list[tuple[str | None, str]] = []
        if spec.fallback_key and spec.fallback_table:
            chain.append((spec.fallback_key, spec.fallback_table))
        chain.append((GLOBAL_LEVEL.key, GLOBAL_LEVEL.table))

        for key, table in chain:
            if not values[list(FALLBACK_METRICS)].isna().to_numpy().any():
                break
            coarser = _lookup(out, key, snapshots.get(table))
            for metric in FALLBACK_METRICS:
                values[metric] = values[metric].fillna(coarser[metric])

        # Volumes are never inherited: a cold entity has seen 0 orders, whatever its state has.
        values["order_count"] = own["order_count"].fillna(0)

        for metric in METRIC_COLUMNS:
            out[f"{spec.name}_{metric}_hist"] = values[metric].to_numpy()
        out[f"{spec.name}_is_new"] = is_new.to_numpy()

    return out


def write_snapshots(conn: Connection, snapshots: dict[str, pd.DataFrame]) -> dict[str, int]:
    """Replace every snapshot table in Postgres.

    Args:
        conn: Open connection inside a transaction; the caller owns it.
        snapshots: Output of :func:`build_all_snapshots`.

    Returns:
        Rows written per table.
    """
    key_by_table = {GLOBAL_LEVEL.table: None}
    for spec in ENTITIES:
        key_by_table[spec.table] = spec.key
        if spec.fallback_table:
            key_by_table[spec.fallback_table] = spec.fallback_key

    from sqlalchemy import text

    written: dict[str, int] = {}
    for table, frame in snapshots.items():
        key = key_by_table[table]
        qualified = f"{FEATURES_SCHEMA}.{table}"
        key_ddl = f"    {key} TEXT NOT NULL,\n" if key else ""
        primary = f"({key}, snapshot_month)" if key else "(snapshot_month)"
        conn.execute(text(f"DROP TABLE IF EXISTS {qualified}"))
        conn.execute(
            text(
                f"CREATE TABLE {qualified} (\n"
                f"{key_ddl}"
                "    snapshot_month TIMESTAMP NOT NULL,\n"
                "    order_count INTEGER NOT NULL,\n"
                "    late_rate DOUBLE PRECISION NOT NULL,\n"
                "    avg_delivery_days DOUBLE PRECISION NOT NULL,\n"
                # NULL when every contributing order lacked a valid carrier handoff.
                "    avg_handling_days DOUBLE PRECISION,\n"
                f"    CONSTRAINT {table}_pkey PRIMARY KEY {primary}\n)"
            )
        )

        columns = ([key] if key else []) + ["snapshot_month", *METRIC_COLUMNS]
        buffer = io.StringIO()
        frame.reindex(columns=columns).to_csv(buffer, index=False, header=False, na_rep="")
        buffer.seek(0)
        cursor = conn.connection.cursor()
        try:
            cursor.copy_expert(
                f"COPY {qualified} ({', '.join(columns)}) FROM STDIN "
                "WITH (FORMAT csv, ENCODING 'UTF8')",
                buffer,
            )
            written[table] = cursor.rowcount
        finally:
            cursor.close()
        conn.execute(text(f"ANALYZE {qualified}"))
        logger.info("%-34s %7d rows", qualified, written[table])
    return written


def build() -> dict[str, object]:
    """Read resolved orders, build every snapshot level, and write them, in one transaction."""
    from sqlalchemy import text

    from src.db import get_engine
    from src.etl.build_orders import analysis_window

    start, end_exclusive = analysis_window()
    months = snapshot_months(pd.Timestamp(start), pd.Timestamp(end_exclusive))
    logger.info("Snapshot months: %d, %s .. %s", len(months), months[0].date(), months[-1].date())

    engine = get_engine()
    with engine.begin() as conn:
        resolved = pd.read_sql(text(RESOLVED_QUERY), conn)
        logger.info("Resolved orders available: %d", len(resolved))

        snapshots = build_all_snapshots(resolved, months)
        written = write_snapshots(conn, snapshots)

        orders = pd.read_sql(text(ORDERS_QUERY), conn)
        attached = attach_history(orders, snapshots)

    for spec in ENTITIES:
        flag = f"{spec.name}_is_new"
        logger.info("%-10s cold-start rate %.4f", spec.name, float(attached[flag].mean()))
    return {"months": len(months), "written": written, "attached": attached}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", type=Path, default=Path("reports/as_of_snapshots.md"))
    parser.add_argument("--no-report", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    started = time.perf_counter()
    summary = build()
    if not args.no_report:
        write_report(summary, args.report)
    logger.info("Done in %.1fs", time.perf_counter() - started)
    return 0


def write_report(summary: dict[str, object], path: Path) -> None:
    """Write a short snapshot report to ``reports/``."""
    attached = summary["attached"]
    train = attached[attached["purchase_month"] >= "2017-05-01"]
    lines = [
        "# As-of monthly snapshots",
        "",
        "Generated by `python -m src.features.history`. The rule (PLAN.md §4.5, §18 A1): a",
        "snapshot for month *M* aggregates only orders with",
        "`order_delivered_customer_date < M`, never `order_purchase_timestamp < M`.",
        "",
        f"Snapshot months: **{summary['months']}**",
        "",
        "| Table | Rows |",
        "|---|---:|",
    ]
    for table, count in summary["written"].items():
        lines.append(f"| `features.{table}` | {count:,} |")
    lines += [
        "",
        "## Cold-start rate on training rows (2017-05 onward)",
        "",
        "| Entity | Cold-start rate |",
        "|---|---:|",
    ]
    for spec in ENTITIES:
        lines.append(f"| {spec.name} | {float(train[f'{spec.name}_is_new'].mean()):.2%} |")
    lines += [
        "",
        "Cold start does not fade after the warm-up: new sellers keep joining the",
        "marketplace, so `seller_is_new` stays between 4.6% and 8.8% every month. Values fall",
        "back entity -> state -> global; `order_count` is never inherited.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Report written to %s", path)


if __name__ == "__main__":
    sys.exit(main())
