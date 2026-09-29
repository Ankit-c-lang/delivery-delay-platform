"""Feature construction — the eight families of PLAN.md §5.

    PREDICTION-TIME CONTRACT
    ========================
    Prediction is made at ``order_purchase_timestamp``. Only information available at the
    moment the customer completes checkout may enter the feature matrix.

    Strictly forbidden as features, because every one is populated after that moment
    (PLAN.md §4.2, enforced by :data:`src.etl.schema.DENYLIST`):

        order_approved_at
        order_delivered_carrier_date
        order_delivered_customer_date   (this defines the target)
        order_status                    (constant after filtering, and a filter-leak trap)
        every column of olist_order_reviews_dataset

    Not leakage, despite looking like it:

        order_estimated_delivery_date   shown to the customer at checkout
        shipping_limit_date             a contractual seller deadline set at order time,
                                        verified against the data in §18 A4

    Historical aggregates of *other, already-resolved* orders are also not leakage. That is
    the premise of §4.5, and the boundary is enforced in :mod:`src.features.history`.

This module is deliberately **stateless**. Every fitted quantity — category level sets,
winsorisation bounds, imputation medians — arrives as an argument, because fitting anything
here would fit it on whatever frame happened to be passed, including the evaluation set.
:class:`src.features.artifact.PreprocessingArtifact` owns the fitting.

Family sizes, totalling 38: F1 promise and timing 5, F2 geography 7, F3 order composition 6,
F4 product physical 6, F5 payment 4, F6 seller history 5, F7 route history 3, F8 category
history 2. F1 lost ``purchase_month`` in DECISIONS.md D29 — it could not have been learned,
because no calibration window shares a calendar month with its fit window.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import text

from src.db import get_engine

logger = logging.getLogger(__name__)

#: Sentinel for a category level not seen during fitting. A frozen level set plus an explicit
#: sentinel is what keeps train and serve encodings identical (PLAN.md §6.2).
UNKNOWN = "__unknown__"

#: Placeholder for a genuinely missing category, kept distinct from UNKNOWN: "this order has
#: no category" (1,330 orders) is different information from "this category is one I have
#: never seen".
MISSING = "__missing__"

#: IBGE macro-regions. All 27 Brazilian federal units, checked to sum to 27.
STATE_TO_REGION: dict[str, str] = {
    # Norte (7)
    "AC": "N",
    "AP": "N",
    "AM": "N",
    "PA": "N",
    "RO": "N",
    "RR": "N",
    "TO": "N",
    # Nordeste (9)
    "AL": "NE",
    "BA": "NE",
    "CE": "NE",
    "MA": "NE",
    "PB": "NE",
    "PE": "NE",
    "PI": "NE",
    "RN": "NE",
    "SE": "NE",
    # Centro-Oeste (4)
    "DF": "CO",
    "GO": "CO",
    "MT": "CO",
    "MS": "CO",
    # Sudeste (4)
    "ES": "SE",
    "MG": "SE",
    "RJ": "SE",
    "SP": "SE",
    # Sul (3)
    "PR": "S",
    "RS": "S",
    "SC": "S",
}

#: The eight families of §5, named rather than sliced by index, so the report and the tests
#: cannot drift from the list when a feature is added or removed.
#:
#: ``purchase_month`` was **removed** after the Phase 5 audit (DECISIONS.md D29). It was the
#: top feature by SHAP and could not have been learned: under the §4.4 forward split no
#: calibration window shares a single calendar month with its fit window, and months 9-12
#: occur in only one year, so a month effect is perfectly confounded with whatever happened
#: operationally that year. The other three timing features stay, because their cycles repeat
#: often enough inside the window to be estimated — 87 weeks for day-of-week against 1.67
#: years for month-of-year.
F1_PROMISE_TIMING: tuple[str, ...] = (
    "promised_days",
    "purchase_hour",
    "purchase_dayofweek",
    "is_weekend",
    "days_to_shipping_limit",
)
F2_GEOGRAPHY: tuple[str, ...] = (
    "customer_state",
    "seller_state",
    "same_state",
    "customer_region",
    "haversine_km",
    "n_distinct_seller_states",
    "customer_zip_prefix_2",
)
F3_ORDER_COMPOSITION: tuple[str, ...] = (
    "n_items",
    "n_distinct_products",
    "n_distinct_sellers",
    "total_price",
    "total_freight",
    "freight_ratio",
)
F4_PRODUCT_PHYSICAL: tuple[str, ...] = (
    "total_weight_g",
    "max_item_weight_g",
    "total_volume_cm3",
    "max_item_volume_cm3",
    "avg_density",
    "product_category",
)
F5_PAYMENT: tuple[str, ...] = (
    "payment_type",
    "max_installments",
    "payment_value",
    "n_payment_methods",
)
F6_SELLER_HISTORY: tuple[str, ...] = (
    "seller_order_count_hist",
    "seller_late_rate_hist",
    "seller_avg_delivery_days_hist",
    "seller_avg_handling_days_hist",
    "seller_is_new",
)
F7_ROUTE_HISTORY: tuple[str, ...] = (
    "route_order_count_hist",
    "route_late_rate_hist",
    "route_avg_delivery_days_hist",
)
F8_CATEGORY_HISTORY: tuple[str, ...] = (
    "category_late_rate_hist",
    "category_avg_delivery_days_hist",
)

#: Family label -> members, in report order.
FEATURE_FAMILIES: dict[str, tuple[str, ...]] = {
    "F1 promise and timing": F1_PROMISE_TIMING,
    "F2 geography": F2_GEOGRAPHY,
    "F3 order composition": F3_ORDER_COMPOSITION,
    "F4 product physical": F4_PRODUCT_PHYSICAL,
    "F5 payment": F5_PAYMENT,
    "F6 seller history (as-of)": F6_SELLER_HISTORY,
    "F7 route history (as-of)": F7_ROUTE_HISTORY,
    "F8 category history (as-of)": F8_CATEGORY_HISTORY,
}

#: The features, in the order the matrix must always present them (§6.6: column order must
#: match the artifact's stored list exactly). 38 after D29.
FEATURE_NAMES: tuple[str, ...] = tuple(
    name for members in FEATURE_FAMILIES.values() for name in members
)

#: Features that must carry pandas ``category`` dtype. §6.1 uses each library's native
#: categorical handling rather than one-hot encoding, which avoids target-encoding leakage
#: entirely: LightGBM reads ``category`` dtype, XGBoost with ``enable_categorical=True``,
#: CatBoost natively.
CATEGORICAL_FEATURES: tuple[str, ...] = (
    "customer_state",
    "seller_state",
    "customer_region",
    "customer_zip_prefix_2",
    "product_category",
    "payment_type",
)

#: Which raw column each categorical feature is built from, so a caller can be told *what* it
#: sent that was unrecognised rather than merely which feature went unknown. Declared here, beside
#: `construct_features`, because that is the only place the mapping is enacted — a copy anywhere
#: else could drift silently. `customer_region` maps through :data:`STATE_TO_REGION` and
#: `customer_zip_prefix_2` is a prefix of the zip, so in both cases the useful thing to report is
#: the raw column that produced them. A test asserts every categorical feature has an entry.
CATEGORICAL_SOURCES: dict[str, str] = {
    "customer_state": "customer_state",
    "seller_state": "seller_state",
    "customer_region": "customer_state",
    "customer_zip_prefix_2": "customer_zip_code_prefix",
    "product_category": "dominant_category",
    "payment_type": "dominant_payment_type",
}

#: Boolean features. Kept as bool rather than category: the tree libraries split on them
#: directly and a 2-level category buys nothing.
BOOLEAN_FEATURES: tuple[str, ...] = ("is_weekend", "same_state", "seller_is_new")

#: Numeric features, derived rather than listed so it cannot drift from FEATURE_NAMES.
NUMERIC_FEATURES: tuple[str, ...] = tuple(
    name
    for name in FEATURE_NAMES
    if name not in CATEGORICAL_FEATURES and name not in BOOLEAN_FEATURES
)

#: Columns :func:`construct_features` needs on its input frame.
REQUIRED_INPUT_COLUMNS: tuple[str, ...] = (
    "order_purchase_timestamp",
    "promised_days",
    "days_to_shipping_limit",
    "customer_state",
    "seller_state",
    "customer_zip_code_prefix",
    "customer_seller_distance_km",
    "n_seller_states",
    "n_items",
    "n_distinct_products",
    "n_distinct_sellers",
    "total_price",
    "total_freight",
    "total_weight_g",
    "max_item_weight_g",
    "total_volume_cm3",
    "max_item_volume_cm3",
    "dominant_category",
    "dominant_payment_type",
    "max_installments",
    "total_payment_value",
    "n_payment_methods",
    "seller_order_count_hist",
    "seller_late_rate_hist",
    "seller_avg_delivery_days_hist",
    "seller_avg_handling_days_hist",
    "seller_is_new",
    "route_order_count_hist",
    "route_late_rate_hist",
    "route_avg_delivery_days_hist",
    "category_late_rate_hist",
    "category_avg_delivery_days_hist",
)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide, yielding NaN rather than inf where the denominator is zero or missing.

    §5 asks for a price-zero guard on ``freight_ratio``. No order in the current population
    has a zero price or a zero volume, but an inf would poison a tree split silently and the
    guard costs nothing.
    """
    denom = denominator.replace(0, np.nan)
    return numerator / denom


def cap_categories(values: pd.Series, levels: tuple[str, ...]) -> pd.Series:
    """Map a raw category column onto a frozen level set.

    Args:
        values: Raw category values, possibly containing NaN.
        levels: The fitted level set, which must already contain
            :data:`UNKNOWN` and :data:`MISSING`.

    Returns:
        A ``category``-dtype series whose categories are exactly ``levels``, in that order.
        A genuinely absent value becomes :data:`MISSING`; a value never seen at fit time
        becomes :data:`UNKNOWN`. Keeping those apart matters: "no category recorded" and
        "a category new since training" are different facts, and collapsing them would hide
        a growing catalogue behind a data-quality problem.
    """
    filled = values.astype("object").where(values.notna(), MISSING)
    known = set(levels)
    mapped = filled.map(lambda value: value if value in known else UNKNOWN)
    return pd.Categorical(mapped, categories=list(levels)).rename_categories(list(levels))


def construct_features(
    orders: pd.DataFrame,
    *,
    category_levels: dict[str, tuple[str, ...]],
    winsor_bounds: dict[str, float],
) -> pd.DataFrame:
    """Build the 38-feature matrix.

    Args:
        orders: One row per order, with history already attached by
            :func:`src.features.history.attach_history`. Must carry
            :data:`REQUIRED_INPUT_COLUMNS`.
        category_levels: Frozen level set per categorical feature, fitted on training rows.
        winsor_bounds: Upper bound per numeric feature to clip, fitted on training rows.

    Returns:
        A new frame with exactly :data:`FEATURE_NAMES`, in that order. The input is never
        mutated.

    Raises:
        ValueError: If an input column is missing, or if a §4.2 denylist column is present.
    """
    from src.etl.schema import DENYLIST

    missing = [column for column in REQUIRED_INPUT_COLUMNS if column not in orders.columns]
    if missing:
        raise ValueError(f"construct_features is missing input columns: {missing}")

    forbidden = DENYLIST & set(orders.columns)
    if forbidden:
        raise ValueError(f"§4.2 denylist columns reached feature construction: {sorted(forbidden)}")

    purchased = orders["order_purchase_timestamp"]
    out = pd.DataFrame(index=orders.index)

    # --- F1 promise and timing ---------------------------------------------------------
    out["promised_days"] = orders["promised_days"].astype("float64")
    out["purchase_hour"] = purchased.dt.hour.astype("int16")
    out["purchase_dayofweek"] = purchased.dt.dayofweek.astype("int16")
    # No calendar-month feature. `orders_analytical` still has a `purchase_month` column — a
    # first-of-month timestamp used to join as-of snapshots — but it is deliberately not a
    # feature (DECISIONS.md D29).
    out["is_weekend"] = (purchased.dt.dayofweek >= 5).astype("bool")
    # Winsorised: the raw column runs to 1,052 days against a train p99 of 19 (§18 A4).
    out["days_to_shipping_limit"] = (
        orders["days_to_shipping_limit"]
        .astype("float64")
        .clip(upper=winsor_bounds.get("days_to_shipping_limit", np.inf))
    )

    # --- F2 geography ------------------------------------------------------------------
    out["customer_state"] = cap_categories(
        orders["customer_state"], category_levels["customer_state"]
    )
    out["seller_state"] = cap_categories(orders["seller_state"], category_levels["seller_state"])
    out["same_state"] = (orders["seller_state"] == orders["customer_state"]).astype("bool")
    out["customer_region"] = cap_categories(
        orders["customer_state"].map(STATE_TO_REGION), category_levels["customer_region"]
    )
    out["haversine_km"] = orders["customer_seller_distance_km"].astype("float64")
    out["n_distinct_seller_states"] = orders["n_seller_states"].astype("int16")
    out["customer_zip_prefix_2"] = cap_categories(
        orders["customer_zip_code_prefix"].str.slice(0, 2),
        category_levels["customer_zip_prefix_2"],
    )

    # --- F3 order composition ----------------------------------------------------------
    out["n_items"] = orders["n_items"].astype("int16")
    out["n_distinct_products"] = orders["n_distinct_products"].astype("int16")
    out["n_distinct_sellers"] = orders["n_distinct_sellers"].astype("int16")
    out["total_price"] = orders["total_price"].astype("float64")
    out["total_freight"] = orders["total_freight"].astype("float64")
    out["freight_ratio"] = _safe_ratio(out["total_freight"], out["total_price"])

    # --- F4 product physical -----------------------------------------------------------
    out["total_weight_g"] = orders["total_weight_g"].astype("float64")
    out["max_item_weight_g"] = orders["max_item_weight_g"].astype("float64")
    out["total_volume_cm3"] = orders["total_volume_cm3"].astype("float64")
    out["max_item_volume_cm3"] = orders["max_item_volume_cm3"].astype("float64")
    out["avg_density"] = _safe_ratio(out["total_weight_g"], out["total_volume_cm3"])
    out["product_category"] = cap_categories(
        orders["dominant_category"], category_levels["product_category"]
    )

    # --- F5 payment --------------------------------------------------------------------
    out["payment_type"] = cap_categories(
        orders["dominant_payment_type"], category_levels["payment_type"]
    )
    out["max_installments"] = orders["max_installments"].astype("int16")
    out["payment_value"] = orders["total_payment_value"].astype("float64")
    out["n_payment_methods"] = orders["n_payment_methods"].astype("int16")

    # --- F6/F7/F8 as-of history --------------------------------------------------------
    # Attached upstream by src.features.history, which owns the leakage boundary. §5 takes
    # only 10 of the 15 columns that module produces; the rest stay available but unused.
    out["seller_order_count_hist"] = orders["seller_order_count_hist"].astype("float64")
    out["seller_late_rate_hist"] = orders["seller_late_rate_hist"].astype("float64")
    out["seller_avg_delivery_days_hist"] = orders["seller_avg_delivery_days_hist"].astype("float64")
    out["seller_avg_handling_days_hist"] = orders["seller_avg_handling_days_hist"].astype("float64")
    out["seller_is_new"] = orders["seller_is_new"].astype("bool")
    out["route_order_count_hist"] = orders["route_order_count_hist"].astype("float64")
    out["route_late_rate_hist"] = orders["route_late_rate_hist"].astype("float64")
    out["route_avg_delivery_days_hist"] = orders["route_avg_delivery_days_hist"].astype("float64")
    out["category_late_rate_hist"] = orders["category_late_rate_hist"].astype("float64")
    out["category_avg_delivery_days_hist"] = orders["category_avg_delivery_days_hist"].astype(
        "float64"
    )

    return out[list(FEATURE_NAMES)]


# ==================================================================================
# Orchestration. Everything above is stateless; everything below talks to Postgres.
# ==================================================================================

FEATURES_SCHEMA = "features"
ARTIFACT_PATH = Path("artifacts/preprocessing.joblib")
FEATURE_REPORT = Path("reports/feature_matrix.md")

#: **The ORDER BY is not cosmetic** (DECISIONS.md D35). Postgres may return an unordered
#: `SELECT` in any order, and it varies in practice — a parallel sequential scan interleaves its
#: workers' output. Downstream, `ensemble.slice_window` sorts with `kind="stable"`, which
#: preserves the input order among **tied** timestamps, and `fit_final_model` then cuts its
#: early-stopping tail at a positional 85%. So an arbitrary scan order silently changed which
#: rows validated the models, which changed their tree counts, which changed the shipped model.
#: `order_id` is unique, so this is a total order and the matrix is a function of the data alone.
ORDERS_QUERY = f"""
SELECT * FROM {FEATURES_SCHEMA}.orders_analytical
ORDER BY order_purchase_timestamp, order_id
"""

LABELS_QUERY = f"""
SELECT order_id, is_late FROM {FEATURES_SCHEMA}.order_outcomes
"""


def window_slice(
    window: Any,
    orders: pd.DataFrame,
    matrix: pd.DataFrame,
    labels: pd.Series,
) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    """Select one window's rows and put them in a **deterministic total order**.

    Args:
        window: A :class:`src.splits.Window`.
        orders: The orders frame, carrying ``order_purchase_timestamp`` and ``order_id``.
        matrix: The feature matrix, row-aligned to ``orders``.
        labels: Labels, row-aligned to ``orders``.

    Returns:
        ``(X, y, timestamps)``, reindexed from 0 and sorted by
        ``(order_purchase_timestamp, order_id)``.

    **Why this exists rather than an inline sort at each call site** (DECISIONS.md D35).
    Both Phase 4 and Phase 5 sorted with ``np.argsort(..., kind="stable")``, which is correct
    about ties only if the incoming order is itself deterministic — and it was not, because
    ``ORDERS_QUERY`` had no ``ORDER BY``. Everything downstream slices **positionally**:
    ``TimeSeriesSplit`` cuts folds by index and ``fit_final_model`` takes the last 15% as its
    early-stopping tail. So tied timestamps landing in a different order silently changed which
    rows validated the models, and the shipped model changed with them. Two call sites meant two
    chances to get it wrong; now there is one, and ``order_id`` makes the order total.
    """
    mask = window.mask(orders["order_purchase_timestamp"]).to_numpy()
    stamps = orders.loc[mask, "order_purchase_timestamp"]
    order = np.lexsort((orders.loc[mask, "order_id"].to_numpy(), stamps.to_numpy()))
    return (
        matrix.loc[mask].iloc[order].reset_index(drop=True),
        labels.loc[mask].to_numpy()[order].astype(int),
        stamps.iloc[order].reset_index(drop=True),
    )


def load_inputs(conn) -> dict[str, object]:
    """Read orders, labels, snapshots and centroids from Postgres."""
    from src.features.history import ENTITIES, GLOBAL_LEVEL

    tables = [spec.table for spec in ENTITIES]
    tables += [spec.fallback_table for spec in ENTITIES if spec.fallback_table]
    tables.append(GLOBAL_LEVEL.table)

    return {
        "orders": pd.read_sql(text(ORDERS_QUERY), conn),
        "labels": pd.read_sql(text(LABELS_QUERY), conn),
        "snapshots": {
            table: pd.read_sql(text(f"SELECT * FROM {FEATURES_SCHEMA}.{table}"), conn)
            for table in tables
        },
        "zip_centroids": pd.read_sql(text(f"SELECT * FROM {FEATURES_SCHEMA}.zip_centroids"), conn),
        "state_centroids": pd.read_sql(
            text(f"SELECT * FROM {FEATURES_SCHEMA}.state_centroids"), conn
        ),
    }


def build_matrix(version: str = "v1") -> dict[str, object]:
    """Fit the artifact on one version's training window and transform every order.

    Args:
        version: Key in ``configs/splits.yaml``.

    Returns:
        ``artifact``, the full ``matrix``, the ``labels`` series and the ``orders`` frame.

    The matrix is deliberately **not** persisted as a table. Phase 4 recomputes it through
    the artifact instead, which exercises the artifact on every run and removes an entire
    class of bug where a stored matrix drifts out of step with the artifact that supposedly
    produced it.
    """
    from src.features.artifact import PreprocessingArtifact
    from src.features.history import attach_history
    from src.splits import assert_temporal_ordering, version_splits

    engine = get_engine()
    with engine.connect() as conn:
        inputs = load_inputs(conn)

    orders = inputs["orders"]
    snapshots = inputs["snapshots"]

    # src/splits.py is the only reader of configs/splits.yaml, so there is exactly one copy
    # of every boundary. The ordering assertion runs here rather than only in a test: it is
    # cheap, and a split that has drifted should stop the build rather than train a model.
    splits = version_splits(version)
    assert_temporal_ordering(orders, version)

    train_orders = attach_history(splits.fit.select(orders), snapshots)
    logger.info("Fitting on %s: %s, %d rows", version, splits.fit, len(train_orders))

    artifact = PreprocessingArtifact.fit(
        train_orders,
        snapshots=snapshots,
        zip_centroids=inputs["zip_centroids"],
        state_centroids=inputs["state_centroids"],
        train_cutoff=train_orders["order_purchase_timestamp"].max(),
        # The full snapshot set is passed above because the training matrix joins per purchase
        # month, which is leakage-safe. This bound governs only which snapshot is bundled for
        # SERVING — without it the artifact bundles the newest month in the database, which is
        # eight months past v1's window and inside the promotion evaluation set (D33).
        aggregates_through=splits.aggregates_through,
    )
    matrix = artifact.transform_with_snapshots(orders, snapshots)
    labels = (
        inputs["labels"]
        .set_index("order_id")
        .loc[orders["order_id"], "is_late"]
        .reset_index(drop=True)
    )
    logger.info("Feature matrix: %d rows x %d features", *matrix.shape)
    return {
        "artifact": artifact,
        "matrix": matrix,
        "labels": labels,
        "orders": orders,
        "snapshots": snapshots,
    }


def write_feature_report(
    artifact, matrix: pd.DataFrame, raw_matrix: pd.DataFrame, path: Path = FEATURE_REPORT
) -> None:
    """Write the feature inventory and pre-imputation null counts to ``reports/``."""
    families = list(FEATURE_FAMILIES.items())
    lines = [
        "# Feature matrix",
        "",
        "Generated by `python -m src.features.build`. Contract: PLAN.md §4.2 — only data",
        "available at `order_purchase_timestamp` may become a feature.",
        "",
        f"**{len(FEATURE_NAMES)} features**, {len(matrix):,} rows. "
        f"{len(CATEGORICAL_FEATURES)} categorical, {len(BOOLEAN_FEATURES)} boolean, "
        f"{len(NUMERIC_FEATURES)} numeric.",
        "",
        "| Family | n | Features |",
        "|---|---:|---|",
    ]
    for name, members in families:
        lines.append(f"| {name} | {len(members)} | {', '.join(f'`{m}`' for m in members)} |")

    lines += [
        "",
        "## Missing values before imputation",
        "",
        "Numeric features are imputed with the **training** median, stored in the artifact.",
        "",
        "| Feature | Nulls | Share |",
        "|---|---:|---:|",
    ]
    nulls = raw_matrix.isna().sum()
    for column in nulls[nulls > 0].sort_values(ascending=False).index:
        lines.append(
            f"| `{column}` | {int(nulls[column]):,} | {nulls[column] / len(raw_matrix):.2%} |"
        )
    if int(nulls.sum()) == 0:
        lines.append("| _none_ | 0 | 0.00% |")

    from src.features.artifact import WINSOR_QUANTILE

    summary = artifact.summary()
    lines += [
        "",
        "## Artifact",
        "",
        f"- Fitted on **{summary['train_rows']:,}** training rows, cutoff "
        f"`{summary['train_cutoff']}`",
        f"- Latest bundled snapshot month: `{summary['latest_snapshot_month']}` "
        "(the §18 A6 serving path)",
        f"- Input schema hash: `{summary['input_schema_hash']}`",
        f"- Winsorisation: {summary['winsor_bounds']} at q={WINSOR_QUANTILE}",
        "",
        "### Category level counts (fitted on training rows only)",
        "",
        "| Column | Levels |",
        "|---|---:|",
    ]
    for column, count in summary["category_levels"].items():
        lines.append(f"| `{column}` | {count} |")
    lines += [
        "",
        "`product_category` is capped to the 30 most frequent training categories, which",
        "cover 95.80% of training rows; the remaining 42 collapse into `__unknown__`.",
        '`__missing__` is kept distinct from `__unknown__`: "no category recorded" (1,330',
        'orders) is different information from "a category new since training".',
        "",
        "## §18 A4, closed",
        "",
        "`days_to_shipping_limit` correlates **0.271** with `promised_days` on the v1 fit",
        "window (0.388 after winsorising at 30 days). Moderate, not near-collinear, so both",
        "features stay. The 1,052-day tail is winsorised at the training q99.5.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Feature report written to %s", path)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: fit the artifact, transform, save, report."""
    parser = argparse.ArgumentParser(description="Build the PLAN.md §5 feature matrix")
    parser.add_argument("--version", default="v1", help="splits.yaml version key to fit on")
    parser.add_argument("--artifact", type=Path, default=ARTIFACT_PATH)
    parser.add_argument("--no-report", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    started = time.perf_counter()
    result = build_matrix(args.version)
    artifact = result["artifact"]
    artifact.save(args.artifact)

    if not args.no_report:
        # Pre-imputation matrix, for the null report only.
        raw_matrix = construct_features(
            attach_history_for_report(result),
            category_levels=artifact.category_levels,
            winsor_bounds=artifact.winsor_bounds,
        )
        write_feature_report(artifact, result["matrix"], raw_matrix)

    logger.info("Done in %.1fs", time.perf_counter() - started)
    return 0


def attach_history_for_report(result: dict[str, object]) -> pd.DataFrame:
    """Re-attach history without imputation, so the report can count real nulls."""
    from src.features.history import attach_history

    return attach_history(result["orders"], result["snapshots"])


if __name__ == "__main__":
    sys.exit(main())
