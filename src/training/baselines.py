"""Baselines (PLAN.md §13 Phase 4, §4.6).

Two of them, and the first matters more than it looks:

**Majority class — predict never late.** ~93% accuracy, **0% recall**. Reported explicitly
because it is the cleanest way to show why accuracy is the wrong metric here, and because it
pre-empts the obvious trap question. As a *ranking* it scores PR-AUC equal to the base rate,
which is the floor every real model must clear by a visible margin.

**Logistic regression.** A linear model on the same features, so "the GBDT is better" is a
measured claim rather than an assumption. Median imputation and standardisation are fitted
inside the pipeline on training folds only, and only the low-cardinality categoricals are
one-hot encoded — a 100-level zip prefix would add 100 sparse columns to a linear model for
very little, and the GBDTs handle it natively anyway (§6.1).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.build import BOOLEAN_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES

logger = logging.getLogger(__name__)

#: Categoricals with at most this many levels are one-hot encoded for the linear baseline.
#: Above it they are dropped: `customer_zip_prefix_2` has 100 levels and `product_category`
#: 32, and neither earns its sparse columns in a linear model. The GBDTs see all of them.
MAX_ONE_HOT_LEVELS = 30


class MajorityClassBaseline:
    """Predict the training base rate for every order.

    A constant score, so it ranks nothing: PR-AUC equals the base rate and ROC-AUC is exactly
    0.5. That is the floor, and it is the number to quote when someone is impressed by 93%
    accuracy.
    """

    def __init__(self) -> None:
        self.base_rate_: float | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> MajorityClassBaseline:
        """Record the training base rate. ``X`` is ignored and accepted for API symmetry."""
        del X
        self.base_rate_ = float(np.asarray(y, dtype=float).mean())
        return self

    def predict_proba_positive(self, X: pd.DataFrame) -> np.ndarray:
        """Return the constant base rate for every row."""
        if self.base_rate_ is None:
            raise RuntimeError("fit the baseline before predicting")
        return np.full(len(X), self.base_rate_, dtype=float)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """The majority class, which at a ~7% positive rate is always 'not late'."""
        return np.zeros(len(X), dtype=int)


def one_hot_columns(X: pd.DataFrame) -> list[str]:
    """Categorical columns small enough to one-hot for the linear baseline."""
    chosen = []
    for column in CATEGORICAL_FEATURES:
        if column not in X.columns:
            continue
        levels = (
            len(X[column].cat.categories)
            if isinstance(X[column].dtype, pd.CategoricalDtype)
            else X[column].nunique()
        )
        if levels <= MAX_ONE_HOT_LEVELS:
            chosen.append(column)
    return chosen


def build_logistic_baseline(X: pd.DataFrame) -> Pipeline:
    """A logistic-regression pipeline sized to the given feature frame.

    Args:
        X: A feature matrix, used only to decide which categoricals are small enough to
            encode. No values are read, so this leaks nothing.

    Returns:
        An unfitted pipeline. Every fitted quantity — medians, means, scales, encoder
        categories — is learned inside ``fit``, so it is fitted per CV fold on that fold's
        training rows only (§6.6).

    ``handle_unknown="infrequent_if_exist"`` means a level absent from a training fold does
    not blow up at scoring time, which matters because the early time-series folds are small.
    """
    numeric = [c for c in NUMERIC_FEATURES if c in X.columns]
    boolean = [c for c in BOOLEAN_FEATURES if c in X.columns]
    categorical = one_hot_columns(X)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            ("boolean", "passthrough", boolean),
            (
                "categorical",
                OneHotEncoder(handle_unknown="infrequent_if_exist", sparse_output=False),
                categorical,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    # debug, not info: this is called once per CV fold and the message is identical each time.
    logger.debug(
        "Logistic baseline: %d numeric, %d boolean, %d one-hot categoricals (%s); "
        "dropped %s as too high-cardinality",
        len(numeric),
        len(boolean),
        len(categorical),
        ", ".join(categorical),
        ", ".join(c for c in CATEGORICAL_FEATURES if c not in categorical) or "none",
    )
    return Pipeline(
        [
            ("preprocess", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    # No class_weight: §6.4 says train on raw probabilities and fix the
                    # operating point with a threshold in Phase 5, not by reweighting here.
                    solver="lbfgs",
                    # No n_jobs: it has had no effect on lbfgs since sklearn 1.8 and is
                    # removed in 1.10. Passing it only emits a FutureWarning.
                ),
            ),
        ]
    )
