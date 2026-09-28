"""The preprocessing artifact — PLAN.md §6.2, the structural answer to train/serve skew.

One serializable object holding **everything** that was fitted, logged to MLflow alongside
every model. Inference loads it and applies it; it never re-fits anything. That is why the
class exposes :meth:`PreprocessingArtifact.fit` as a *classmethod* and has no
``fit_transform`` at all: once an instance exists there is no public way to change what it
learned, so a serving path cannot accidentally re-derive a median or a category level set.

Contents, per §6.2:

* numeric imputation medians, fitted on training rows only
* the frozen category level set per categorical column, plus ``__unknown__``
* the latest as-of aggregate snapshots (seller / route / category, and their fallbacks)
* the zip-prefix and state centroid tables
* the exact feature name list, in order
* the winsorisation bounds (§18 A4)
* the training cutoff date and a schema hash
* the calibrator and decision threshold — set in Phase 5, ``None`` until then

**The §18 A6 asymmetry, made explicit in the API.** Training attaches history per purchase
month; serving uses the single latest bundled snapshot, because a live request has no
history table to join to. Rather than hide that behind an optional argument, the two paths
are separate named methods — :meth:`transform_with_snapshots` and :meth:`transform` — so
every call site says which one it means. Parity is scoped to "same raw record + same
artifact -> same probability"; snapshot selection is outside that scope by design.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.features.build import (
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_NAMES,
    MISSING,
    NUMERIC_FEATURES,
    REQUIRED_INPUT_COLUMNS,
    STATE_TO_REGION,
    UNKNOWN,
    construct_features,
)
from src.features.history import ENTITIES, GLOBAL_LEVEL, attach_history

logger = logging.getLogger(__name__)

#: Bumped whenever the artifact's contents or semantics change, so a stale file cannot be
#: silently loaded by newer code.
ARTIFACT_VERSION = "1"

#: Categories kept for ``product_category`` before the rest collapse into ``__unknown__``.
#: Measured: the top 30 cover 95.80% of the v1 fit window's rows, out of 72 present.
CATEGORY_CAP = 30

#: Quantile used to winsorise heavy tails. ``days_to_shipping_limit`` runs to 1,052 days
#: against a train p99 of 19.01 and p99.5 of 21.22 (§18 A4).
WINSOR_QUANTILE = 0.995

#: Columns winsorised at :data:`WINSOR_QUANTILE`.
WINSORISED_COLUMNS: tuple[str, ...] = ("days_to_shipping_limit",)


def _snapshot_tables() -> list[str]:
    """Every snapshot table name, entity levels and fallbacks and global."""
    tables = [spec.table for spec in ENTITIES]
    tables += [spec.fallback_table for spec in ENTITIES if spec.fallback_table]
    tables.append(GLOBAL_LEVEL.table)
    return tables


@dataclass(frozen=True)
class PreprocessingArtifact:
    """Everything fitted, in one serializable object.

    Frozen: an artifact is immutable once built. Phase 5 attaches the calibrator and
    threshold with :meth:`with_decision`, which returns a new artifact rather than mutating
    this one, so an object already logged to MLflow can never drift.
    """

    feature_names: tuple[str, ...]
    categorical_features: tuple[str, ...]
    boolean_features: tuple[str, ...]
    numeric_features: tuple[str, ...]
    numeric_medians: dict[str, float]
    category_levels: dict[str, tuple[str, ...]]
    winsor_bounds: dict[str, float]
    latest_snapshots: dict[str, pd.DataFrame]
    latest_snapshot_month: pd.Timestamp
    zip_centroids: pd.DataFrame
    state_centroids: pd.DataFrame
    train_cutoff: pd.Timestamp
    train_rows: int
    input_schema_hash: str
    version: str = ARTIFACT_VERSION
    calibrator: Any | None = None
    threshold: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------------- fitting ----------
    @classmethod
    def fit(
        cls,
        train_orders: pd.DataFrame,
        *,
        snapshots: dict[str, pd.DataFrame],
        zip_centroids: pd.DataFrame,
        state_centroids: pd.DataFrame,
        train_cutoff: pd.Timestamp,
    ) -> PreprocessingArtifact:
        """Fit every parameter on training rows only.

        Args:
            train_orders: Training rows **with history already attached**. Nothing outside
                the training window may appear here — §6.6 requires imputation values,
                category levels and calibrators to be fitted on training rows only.
            snapshots: All snapshot levels, used both to fit and to select the latest for
                bundling.
            zip_centroids: Zip-prefix centroid table, bundled for serving.
            state_centroids: State centroid table, bundled as the geo fallback.
            train_cutoff: Last purchase timestamp included in training.

        Returns:
            A fitted, immutable artifact.

        Raises:
            ValueError: If ``train_orders`` is empty, or a required column is absent.
        """
        if train_orders.empty:
            raise ValueError("cannot fit on an empty training frame")
        missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in train_orders.columns]
        if missing:
            raise ValueError(f"train_orders is missing columns: {missing}")

        category_levels = cls._fit_category_levels(train_orders)
        winsor_bounds = {
            column: float(train_orders[column].quantile(WINSOR_QUANTILE))
            for column in WINSORISED_COLUMNS
        }

        # Medians come from the constructed matrix, not the raw columns, because derived
        # features (freight_ratio, avg_density) have no raw counterpart to take a median of.
        train_matrix = construct_features(
            train_orders, category_levels=category_levels, winsor_bounds=winsor_bounds
        )
        numeric_medians = {
            column: float(train_matrix[column].median()) for column in NUMERIC_FEATURES
        }
        unfilled = [c for c, v in numeric_medians.items() if not np.isfinite(v)]
        if unfilled:
            raise ValueError(
                f"no finite median could be fitted for {unfilled}; the training window is "
                "probably too narrow to contain any observation of them"
            )

        latest_month = max(
            frame["snapshot_month"].max() for frame in snapshots.values() if not frame.empty
        )
        latest = {
            table: frame.loc[frame["snapshot_month"] == latest_month].reset_index(drop=True)
            for table, frame in snapshots.items()
            if not frame.empty
        }

        return cls(
            feature_names=FEATURE_NAMES,
            categorical_features=CATEGORICAL_FEATURES,
            boolean_features=BOOLEAN_FEATURES,
            numeric_features=NUMERIC_FEATURES,
            numeric_medians=numeric_medians,
            category_levels=category_levels,
            winsor_bounds=winsor_bounds,
            latest_snapshots=latest,
            latest_snapshot_month=pd.Timestamp(latest_month),
            zip_centroids=zip_centroids.reset_index(drop=True),
            state_centroids=state_centroids.reset_index(drop=True),
            train_cutoff=pd.Timestamp(train_cutoff),
            train_rows=len(train_orders),
            input_schema_hash=input_schema_hash(),
            metadata={
                "category_cap": CATEGORY_CAP,
                "winsor_quantile": WINSOR_QUANTILE,
            },
        )

    @staticmethod
    def _fit_category_levels(train_orders: pd.DataFrame) -> dict[str, tuple[str, ...]]:
        """Freeze the level set for each categorical feature, from training rows only.

        ``product_category`` is capped to the :data:`CATEGORY_CAP` most frequent levels; the
        remaining 42 collapse into ``__unknown__`` at transform time. The states, regions and
        zip prefixes are not capped — they are bounded sets and dropping a rare state would
        lose real geography.

        Every level set ends with ``__missing__`` then ``__unknown__``, always present even
        when the training window happened to contain neither, so the encoding cannot change
        shape between fitting and serving.
        """
        tail = (MISSING, UNKNOWN)

        top_categories = (
            train_orders["dominant_category"]
            .dropna()
            .value_counts()
            .head(CATEGORY_CAP)
            .index.tolist()
        )

        def present(series: pd.Series) -> list[str]:
            return sorted(series.dropna().astype(str).unique().tolist())

        return {
            "customer_state": (*present(train_orders["customer_state"]), *tail),
            "seller_state": (*present(train_orders["seller_state"]), *tail),
            # The five IBGE macro-regions are fixed, so the full set is used rather than
            # whatever the training window happened to contain.
            "customer_region": (*sorted(set(STATE_TO_REGION.values())), *tail),
            "customer_zip_prefix_2": (
                *sorted({str(v)[:2] for v in train_orders["customer_zip_code_prefix"].dropna()}),
                *tail,
            ),
            "product_category": (*sorted(top_categories), *tail),
            "payment_type": (*present(train_orders["dominant_payment_type"]), *tail),
        }

    # ------------------------------------------------------------- transforming --------
    def _finish(self, matrix: pd.DataFrame) -> pd.DataFrame:
        """Impute, set dtypes and enforce column order. The last step of every transform."""
        out = matrix.copy()
        for column in self.numeric_features:
            out[column] = out[column].astype("float64").fillna(self.numeric_medians[column])
        for column in self.boolean_features:
            out[column] = out[column].astype("bool")
        for column in self.categorical_features:
            levels = list(self.category_levels[column])
            out[column] = pd.Categorical(out[column].astype("object"), categories=levels)
        # §6.6: column order must match the artifact's stored list exactly.
        return out[list(self.feature_names)]

    def transform_with_snapshots(
        self, orders: pd.DataFrame, snapshots: dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Transform using per-purchase-month snapshots. **The training path.**

        Args:
            orders: Orders **without** history attached; this method attaches it.
            snapshots: All snapshot levels, keyed by table name.

        Returns:
            The feature matrix, columns in :attr:`feature_names` order.

        Each order joins to the snapshot for its own purchase month, which is the
        leakage-safe behaviour §4.5 requires and the only correct choice for training.
        """
        attached = attach_history(orders, snapshots)
        matrix = construct_features(
            attached,
            category_levels=self.category_levels,
            winsor_bounds=self.winsor_bounds,
        )
        return self._finish(matrix)

    def transform(self, orders: pd.DataFrame) -> pd.DataFrame:
        """Transform using the single latest bundled snapshot. **The serving path.**

        Args:
            orders: Orders without history attached.

        Returns:
            The feature matrix, columns in :attr:`feature_names` order.

        A live request has no history table to join to, so every row is scored against
        :attr:`latest_snapshot_month` whatever its own purchase month. That is the §18 A6
        asymmetry, accepted deliberately: parity is defined as "same raw record + same
        artifact -> same probability", and snapshot selection sits outside it. Retraining is
        what refreshes the bundled snapshot, and it is a second independent reason a new
        model version differs from the last.
        """
        forced = orders.copy()
        forced["purchase_month"] = self.latest_snapshot_month
        attached = attach_history(forced, self.latest_snapshots)
        matrix = construct_features(
            attached,
            category_levels=self.category_levels,
            winsor_bounds=self.winsor_bounds,
        )
        return self._finish(matrix)

    # ------------------------------------------------------------------ decision -------
    def with_decision(self, *, calibrator: Any, threshold: float) -> PreprocessingArtifact:
        """Return a copy carrying Phase 5's calibrator and decision threshold.

        Returns a new artifact rather than mutating this one: an artifact already logged to
        MLflow must never change under a run that referenced it.
        """
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"threshold must be in (0, 1), got {threshold}")
        return replace(self, calibrator=calibrator, threshold=threshold)

    # --------------------------------------------------------------- persistence -------
    def save(self, path: Path) -> Path:
        """Serialize to ``path`` with joblib, creating parent directories."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)
        logger.info("Artifact saved to %s (%.1f KB)", path, path.stat().st_size / 1024)
        return path

    @classmethod
    def load(cls, path: Path) -> PreprocessingArtifact:
        """Load from ``path``.

        Raises:
            ValueError: If the file was written by a different artifact version, or if its
                expected input schema no longer matches this code. Both mean the calling code
                and the artifact disagree about the feature contract, which is exactly the
                skew this class exists to prevent.
        """
        artifact = joblib.load(Path(path))
        if artifact.version != ARTIFACT_VERSION:
            raise ValueError(
                f"artifact version {artifact.version!r} but this code expects "
                f"{ARTIFACT_VERSION!r}; refit rather than reusing it"
            )
        if artifact.input_schema_hash != input_schema_hash():
            raise ValueError(
                "artifact input schema hash does not match this code's expected inputs; "
                "the feature contract changed since it was fitted"
            )
        return artifact

    def summary(self) -> dict[str, Any]:
        """A small JSON-serializable description, for logging to MLflow as params."""
        return {
            "version": self.version,
            "n_features": len(self.feature_names),
            "n_categorical": len(self.categorical_features),
            "n_numeric": len(self.numeric_features),
            "train_rows": self.train_rows,
            "train_cutoff": str(self.train_cutoff),
            "latest_snapshot_month": str(self.latest_snapshot_month),
            "zip_centroids": len(self.zip_centroids),
            "category_levels": {k: len(v) for k, v in self.category_levels.items()},
            "winsor_bounds": {k: round(v, 4) for k, v in self.winsor_bounds.items()},
            "has_calibrator": self.calibrator is not None,
            "threshold": self.threshold,
            "input_schema_hash": self.input_schema_hash,
        }


def input_schema_hash() -> str:
    """Stable hash of the feature contract this code expects.

    Covers the required input columns, the feature names and their order, and the artifact
    version. Any change to any of those changes the hash, so a mismatch on load is caught
    before a single wrong prediction is served.
    """
    payload = json.dumps(
        {
            "version": ARTIFACT_VERSION,
            "inputs": sorted(REQUIRED_INPUT_COLUMNS),
            "features": list(FEATURE_NAMES),
            "categorical": list(CATEGORICAL_FEATURES),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
