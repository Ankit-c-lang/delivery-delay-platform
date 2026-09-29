"""Registry resolution and the in-memory model singleton (PLAN.md §9).

**Loaded once, at startup, in the lifespan handler.** §9 names loading per request as the mistake,
and it is a bigger one than it looks: it would add hundreds of milliseconds to every call *and*
make the service's behaviour depend on the registry being reachable at request time, so a registry
blip would become a serving outage.

**This module contains no preprocessing** (CLAUDE.md invariant 4). It loads a pyfunc, reads its
metadata, and calls it. Everything about features lives inside the artifact the pyfunc carries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

from api.config import REGISTERED_MODEL_NAME, ModelUriKind, Settings, resolve_local_path

logger = logging.getLogger(__name__)


@dataclass
class LoadedModel:
    """A resolved model and everything ``/model/info`` needs to describe it.

    Attributes:
        pyfunc: The loaded ``mlflow.pyfunc`` model.
        predictor: The unwrapped :class:`~src.registry.pyfunc_wrapper.DelayPredictor`.
        version: Registry version served, or ``"local"``.
        description: The predictor's own ``describe()`` output.
    """

    pyfunc: Any
    predictor: Any
    version: str
    description: dict[str, Any]

    def score(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
        """Predictions and the unseen-category report, one transform (see ``DelayPredictor.score``).

        Called on the **unwrapped** predictor rather than through ``pyfunc.predict``, which would
        run the transform twice to recover the warnings §9 requires. The input-schema enforcement
        that ``pyfunc.predict`` would add is replaced by something stronger and earlier: Pydantic
        validates every request, and :func:`assert_signature_matches_contract` checks the loaded
        signature against the contract once at startup, so a mismatched model fails to boot rather
        than failing per request.
        """
        return self.predictor.score(frame)


def _resolve_version(settings: Settings, pyfunc: Any) -> str:
    """The registry version actually being served.

    An alias URI does not say which version it resolved to, and that is the number an operator
    needs when a prediction looks wrong — so it is looked up rather than echoed back.
    """
    if settings.model_uri_kind is ModelUriKind.LOCAL_PATH:
        return "local"
    if settings.model_uri_kind is ModelUriKind.PINNED_VERSION:
        return str(settings.model_version_pin)

    import mlflow

    try:
        client = mlflow.MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
        resolved = client.get_model_version_by_alias(
            REGISTERED_MODEL_NAME, str(settings.model_alias)
        )
        # int from a SQLite store, str over HTTP (DECISIONS.md D32) — normalise, or a caller
        # comparing it behaves differently depending on the backend.
        return str(resolved.version)
    except Exception:
        logger.warning("served the alias but could not read back its version number")
        run_id = getattr(pyfunc.metadata, "run_id", None)
        return f"unknown (run {run_id})" if run_id else "unknown"


def assert_signature_matches_contract(pyfunc: Any) -> None:
    """Fail at startup if the loaded model does not want the columns this API sends.

    Raises:
        RuntimeError: If the signature is absent or disagrees with
            :data:`~src.registry.pyfunc_wrapper.RAW_INPUT_COLUMNS`.

    This is the check that makes scoring through the unwrapped predictor safe. MLflow would
    otherwise enforce the input schema on every ``pyfunc.predict`` call; enforcing it once here is
    both cheaper and better placed — a model whose contract has moved should refuse to start, not
    return 500s under load.
    """
    from src.registry.pyfunc_wrapper import RAW_INPUT_COLUMNS

    signature = getattr(pyfunc.metadata, "signature", None)
    if signature is None or signature.inputs is None:
        raise RuntimeError(
            "the loaded model carries no input signature, so its contract cannot be checked; "
            "log it with signature= (PLAN.md §7 names omitting it as a common mistake)"
        )
    served = tuple(sorted(column.name for column in signature.inputs.inputs))
    expected = tuple(sorted(RAW_INPUT_COLUMNS))
    if served != expected:
        missing = sorted(set(expected) - set(served))
        extra = sorted(set(served) - set(expected))
        raise RuntimeError(
            "the loaded model's signature does not match this API's request contract. "
            f"missing={missing} unexpected={extra}. The model and the API were built from "
            "different versions of src/features; rebuild the image or roll the model back."
        )


def load_model(settings: Settings) -> LoadedModel:
    """Resolve and load the model, or raise.

    Args:
        settings: Where to load from.

    Returns:
        A :class:`LoadedModel`.

    Raises:
        FileNotFoundError: A local ``MODEL_URI`` that does not exist.
        RuntimeError: The model loaded but its signature disagrees with the request contract, or it
            is not a :class:`DelayPredictor`.

    §9's resolution order is a property of the URI, so MLflow is handed the string and does the
    work; what this function adds is the checking MLflow does not do.
    """
    import mlflow

    if settings.model_uri_kind is ModelUriKind.LOCAL_PATH:
        uri = str(resolve_local_path(settings.model_uri))
    else:
        uri = settings.model_uri
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    logger.info("loading %s (%s)", uri, settings.model_uri_kind.value)
    pyfunc = mlflow.pyfunc.load_model(uri)
    assert_signature_matches_contract(pyfunc)

    predictor = pyfunc.unwrap_python_model()
    if not hasattr(predictor, "score"):
        raise RuntimeError(
            f"loaded a {type(predictor).__name__}, which does not expose score(); the API serves "
            "DelayPredictor, and a bare estimator would have no preprocessing, no calibrator and "
            "no threshold"
        )

    description = predictor.describe()
    version = _resolve_version(settings, pyfunc)
    logger.info(
        "serving version %s: %s, %d features, threshold %.5f, snapshot %s",
        version,
        description["shipped_model"],
        description["n_features"],
        description["threshold"],
        description["bundled_snapshot_month"][:10],
    )
    return LoadedModel(pyfunc=pyfunc, predictor=predictor, version=version, description=description)
