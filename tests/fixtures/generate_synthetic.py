"""Synthetic feature matrix with the real schema (PLAN.md §10.3, §13 Phase 4).

**CI cannot use the real dataset.** It is 121 MB and CC BY-NC-SA, so it will never be
committed or downloaded in a workflow. Everything CI verifies about training has to run on
something generated, and that something must carry the *same columns, in the same order, with
the same dtypes* as the real matrix — otherwise a green CI run says nothing about the code
that will meet real data.

The column list and the categorical/boolean/numeric split are imported from
:mod:`src.features.build` rather than restated here, so the fixture cannot drift from the
schema it is supposed to imitate. Adding a feature without updating this file is impossible:
the generator builds from ``FEATURE_NAMES``.

The signal is deliberately weak but real — ``promised_days`` and ``seller_late_rate_hist``
push the label — so a smoke-trained model lands well above the base rate without the task
being trivially separable. A fixture a model scores 1.0 on tests nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.build import (
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_NAMES,
    MISSING,
    NUMERIC_FEATURES,
    UNKNOWN,
)

#: Small level sets, shaped like the real ones so the categorical code paths are exercised.
LEVELS: dict[str, tuple[str, ...]] = {
    "customer_state": ("SP", "RJ", "MG", "BA", MISSING, UNKNOWN),
    "seller_state": ("SP", "RJ", "PR", MISSING, UNKNOWN),
    "customer_region": ("SE", "NE", "S", "N", "CO", MISSING, UNKNOWN),
    "customer_zip_prefix_2": ("01", "20", "30", "40", MISSING, UNKNOWN),
    "product_category": ("cama_mesa_banho", "beleza_saude", "esporte_lazer", MISSING, UNKNOWN),
    "payment_type": ("credit_card", "boleto", "voucher", MISSING, UNKNOWN),
}

#: Positive rate of the generated labels, matching the real ~7%.
POSITIVE_RATE = 0.07


def synthetic_matrix(
    n_rows: int = 600, seed: int = 42
) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    """Generate a feature matrix, labels and purchase timestamps.

    Args:
        n_rows: Rows to generate. 600 is enough for ``TimeSeriesSplit(4)`` to leave every
            validation fold with positives at a 7% rate.
        seed: Seed for reproducibility.

    Returns:
        ``(X, y, timestamps)``. ``X`` has exactly :data:`FEATURE_NAMES` in order, with
        ``category`` dtype on the categoricals and ``bool`` on the booleans, matching what
        :meth:`src.features.artifact.PreprocessingArtifact.transform` produces.
        ``timestamps`` is ascending, so it satisfies
        :func:`src.training.tune.assert_time_sorted` directly.
    """
    rng = np.random.default_rng(seed)
    timestamps = pd.Series(
        pd.Timestamp("2017-05-01") + pd.to_timedelta(np.sort(rng.integers(0, 240, n_rows)), "D")
    )

    frame = pd.DataFrame(index=range(n_rows))
    for column in NUMERIC_FEATURES:
        if column.endswith("_rate_hist"):
            frame[column] = rng.beta(2, 25, n_rows)
        elif column.endswith("_count_hist"):
            frame[column] = rng.integers(0, 300, n_rows).astype(float)
        elif "days" in column:
            frame[column] = rng.gamma(4.0, 4.0, n_rows)
        elif column in {"purchase_hour"}:
            frame[column] = rng.integers(0, 24, n_rows).astype(float)
        elif column in {"purchase_dayofweek"}:
            frame[column] = rng.integers(0, 7, n_rows).astype(float)
        elif column in {"purchase_month"}:
            frame[column] = rng.integers(1, 13, n_rows).astype(float)
        else:
            frame[column] = np.abs(rng.normal(100.0, 60.0, n_rows))

    for column in BOOLEAN_FEATURES:
        frame[column] = rng.random(n_rows) < 0.3

    for column in CATEGORICAL_FEATURES:
        levels = LEVELS[column]
        # Never draw the sentinels: they must exist as levels without appearing as values,
        # exactly as the real matrix behaves when training saw every level.
        drawable = [level for level in levels if level not in (MISSING, UNKNOWN)]
        frame[column] = pd.Categorical(rng.choice(drawable, n_rows), categories=list(levels))

    # A weak but real signal, so a smoke-trained model beats the base rate without the task
    # being separable.
    logit = (
        -3.1
        + 0.035 * (frame["promised_days"] - frame["promised_days"].mean())
        + 6.0 * (frame["seller_late_rate_hist"] - frame["seller_late_rate_hist"].mean())
        + 0.7 * frame["seller_is_new"].astype(float)
    )
    probability = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n_rows) < probability).astype(int)

    # Guarantee a workable positive rate whatever the draw did, so a CI run cannot fail on
    # an unlucky seed: nudge the highest-probability rows positive if there are too few.
    wanted = max(round(n_rows * POSITIVE_RATE), 20)
    if y.sum() < wanted:
        candidates = np.argsort(-probability.to_numpy())
        for index in candidates:
            if y.sum() >= wanted:
                break
            y[index] = 1

    return frame[list(FEATURE_NAMES)], y, timestamps


if __name__ == "__main__":
    X, y, stamps = synthetic_matrix()
    print(f"{len(X)} rows x {X.shape[1]} features, positive rate {y.mean():.4f}")
    print(f"span {stamps.min().date()} .. {stamps.max().date()}")
