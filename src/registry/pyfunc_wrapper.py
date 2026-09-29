"""The pyfunc wrapper — one self-contained object that turns a raw order into a decision.

    WHY THE PREPROCESSING ARTIFACT LIVES INSIDE THE PYFUNC
    =====================================================
    Because the alternative has a failure mode that no test catches and no error reports.

    If the API loaded the model from the registry and the preprocessing artifact from
    somewhere else, then serving correctness would depend on a pairing that nothing
    enforces. Retrain, register model v2, forget to ship the new artifact, and v2 runs
    against v1's imputation medians, v1's frozen category levels and v1's as-of snapshots.
    Every prediction still returns a plausible probability. Nothing raises. The only symptom
    is that the model is quietly worse than its own metrics said, and the metrics were
    computed with the *correct* artifact, so they cannot reveal it.

    Bundling makes the pairing structural instead of procedural. A model version *is* its
    preprocessing: one artifact URI, one atomic thing to promote, one thing to roll back.
    ``@champion`` points at a decision, not at half of one.

    It also makes the §18 A6 parity claim expressible at all: "same raw record + same
    artifact -> same probability" is only a testable statement if "the artifact" is a single
    identifiable object rather than a convention about which files were deployed together.

    The corollary is CLAUDE.md invariant 4: **no preprocessing logic may exist outside this
    wrapper.** The API layer receives raw order records and returns decisions; it does not
    know what a category level set is.

**One expected warning.** MLflow 3 inspects ``predict``'s type hints for its newer typed-model
flow, which assumes a *list of instances*. This is a classic DataFrame-in / DataFrame-out
pyfunc, so the hints are omitted and MLflow notes that it will not derive validation from
them. That is correct: validation comes from the signature and ``input_example`` passed to
``log_model``, which is what §13 asks for. The warning is left visible rather than suppressed —
silencing MLflow's warning channel here would also hide the next, real message it sends.

The wrapper deliberately owns three things beyond the model: the fitted
:class:`~src.features.artifact.PreprocessingArtifact`, the isotonic calibrator, and the
decision threshold. A threshold is part of the model's contract with the business — shipping
the probabilities and leaving the threshold in a config file elsewhere is the same class of
mistake as leaving the preprocessing there.
"""

from __future__ import annotations

import logging
from typing import Any

import mlflow.pyfunc
import numpy as np
import pandas as pd

from src.features.artifact import PreprocessingArtifact
from src.features.build import REQUIRED_INPUT_COLUMNS
from src.features.history import ENTITIES, METRIC_COLUMNS

logger = logging.getLogger(__name__)

#: Columns the wrapper *computes* rather than receives: as-of history is attached from the
#: artifact's bundled snapshots, so a caller must not supply it.
HISTORY_COLUMNS: frozenset[str] = frozenset(
    {f"{spec.name}_{metric}_hist" for spec in ENTITIES for metric in METRIC_COLUMNS}
    | {f"{spec.name}_is_new" for spec in ENTITIES}
)

#: Entity keys the snapshot join needs, plus the month it joins on.
_ENTITY_KEYS: frozenset[str] = frozenset(
    {spec.key for spec in ENTITIES}
    | {spec.fallback_key for spec in ENTITIES if spec.fallback_key}
    | {"purchase_month"}
)

#: What a caller must supply. Derived from the feature contract rather than restated, so it
#: cannot drift when a feature is added or removed.
RAW_INPUT_COLUMNS: tuple[str, ...] = tuple(
    sorted((set(REQUIRED_INPUT_COLUMNS) - HISTORY_COLUMNS) | _ENTITY_KEYS)
)

#: Output column names, fixed so the API and its tests can rely on them.
OUTPUT_COLUMNS: tuple[str, ...] = ("probability", "is_late_predicted", "threshold")


class DelayPredictor(mlflow.pyfunc.PythonModel):
    """Raw order records in, calibrated probabilities and decisions out.

    Attributes:
        artifact: The fitted preprocessing artifact, carrying the calibrator and threshold.
        model: The fitted estimator that ships.
        model_name: Which library it is, needed because CatBoost requires its categoricals as
            strings at predict time while the others take ``category`` dtype.
        feature_names: The artifact's ordered feature list, copied for convenience.
    """

    def __init__(self, artifact: PreprocessingArtifact, model: Any, model_name: str) -> None:
        if artifact.threshold is None:
            raise ValueError(
                "the artifact carries no threshold; call with_decision() before wrapping, "
                "or the wrapper would have to invent an operating point"
            )
        self.artifact = artifact
        self.model = model
        self.model_name = model_name
        self.feature_names = artifact.feature_names

    # ------------------------------------------------------------------ inference -------
    def _scores(self, features: pd.DataFrame) -> np.ndarray:
        """Raw positive-class probabilities from the underlying estimator."""
        frame = features
        if self.model_name == "catboost":
            frame = features.copy()
            for column in self.artifact.categorical_features:
                frame[column] = frame[column].astype(str)
        return np.asarray(self.model.predict_proba(frame)[:, 1], dtype=float)

    # Deliberately unannotated. MLflow inspects this signature and warns that a bare
    # `pd.DataFrame` hint is not supported for its schema validation, asking for either
    # `list[...]` or no hint. Validation comes from the logged signature and input_example
    # instead, which is what §13 actually requires, so the hints are dropped to keep every
    # model load quiet. Types are documented below.
    def predict(self, context=None, model_input=None, params=None):
        """Score raw order records.

        Types: ``context`` is MLflow's ``PythonModelContext``, ``model_input`` a
        :class:`pandas.DataFrame`, ``params`` an optional dict. Returns a
        :class:`pandas.DataFrame`.

        Args:
            context: MLflow's ``PythonModelContext``. Unused — everything this model needs is
                pickled with it rather than fetched from an artifact path at load time.
            model_input: One row per order, carrying :data:`RAW_INPUT_COLUMNS`. History
                features must **not** be supplied; they are attached from the bundled
                snapshots.
            params: Ignored. Accepted because MLflow 3 passes it.

        Returns:
            ``probability`` (calibrated), ``is_late_predicted`` and the ``threshold`` used —
            the threshold is returned rather than assumed, so a caller can log which operating
            point produced a decision without looking it up somewhere else.

        Raises:
            ValueError: If a required column is missing, or the caller supplied history
                columns. Silently ignoring supplied history would let a client override the
                leakage-safe snapshot join with anything it liked.
        """
        del context, params
        if model_input is None:
            raise ValueError("model_input is required")
        frame = pd.DataFrame(model_input)

        missing = [column for column in RAW_INPUT_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"missing required input columns: {missing}")
        supplied_history = sorted(HISTORY_COLUMNS & set(frame.columns))
        if supplied_history:
            raise ValueError(
                f"history columns must not be supplied by the caller: {supplied_history}. "
                "They are attached from the artifact's bundled as-of snapshots so that the "
                "§4.5 leakage boundary cannot be bypassed from outside."
            )

        # transform(), not transform_with_snapshots(): serving uses the single latest bundled
        # snapshot (§18 A6). The two-method split exists so this line states which it means.
        features = self.artifact.transform(frame)
        raw = self._scores(features)
        calibrated = (
            np.asarray(self.artifact.calibrator.predict(raw), dtype=float)
            if self.artifact.calibrator is not None
            else raw
        )
        threshold = float(self.artifact.threshold)
        return pd.DataFrame(
            {
                "probability": calibrated,
                "is_late_predicted": calibrated >= threshold,
                "threshold": np.full(len(calibrated), threshold),
            },
            index=frame.index,
        )

    # -------------------------------------------------------------------- metadata ------
    def describe(self) -> dict[str, Any]:
        """A flat description, for logging as MLflow params."""
        return {
            "shipped_model": self.model_name,
            "n_features": len(self.feature_names),
            "threshold": float(self.artifact.threshold),
            "has_calibrator": self.artifact.calibrator is not None,
            "bundled_snapshot_month": str(self.artifact.latest_snapshot_month),
            "train_cutoff": str(self.artifact.train_cutoff),
            "input_schema_hash": self.artifact.input_schema_hash,
        }


def raw_input_example(orders: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Build an ``input_example`` for ``log_model`` from real order records.

    Args:
        orders: An ``orders_analytical``-shaped frame.
        n: Rows to include.

    Returns:
        Exactly :data:`RAW_INPUT_COLUMNS`, so the logged example matches the signature.

    §13 names omitting the signature as a common mistake: without one, MLflow cannot enforce
    the input schema at load time, and a client that sends columns in a different order gets
    silently wrong answers rather than an error.

    The rows are chosen to include at least one null per nullable column where the data has
    any, which is MLflow's own advice: a signature inferred from a sample that happens to be
    complete declares types the real traffic will violate. Note that the seven integer columns
    here (``n_items``, ``promised_days`` and so on) are ``NOT NULL`` in
    ``features.orders_analytical`` by construction, so MLflow's generic warning about integers
    and missing values does not apply — a null in one of them is a contract violation and
    schema enforcement rejecting it is the behaviour CLAUDE.md asks for.
    """
    missing = [column for column in RAW_INPUT_COLUMNS if column not in orders.columns]
    if missing:
        raise ValueError(f"cannot build an input example; orders lacks {missing}")

    frame = orders.loc[:, list(RAW_INPUT_COLUMNS)]
    chosen: list[int] = []
    for column in frame.columns:
        nulls = frame.index[frame[column].isna()]
        if len(nulls) and not any(idx in set(chosen) for idx in nulls):
            chosen.append(int(nulls[0]))
    for idx in frame.index:
        if len(chosen) >= n:
            break
        if int(idx) not in chosen:
            chosen.append(int(idx))
    return frame.loc[chosen[:n]].reset_index(drop=True)
