"""Model introspection (PLAN.md §9).

Answers "what exactly is serving right now?" — version, alias, training window, feature count,
threshold and bundled snapshot month. Every value is read from the loaded artifact rather than from
config, so the endpoint cannot describe a model other than the one in memory. That is the whole
point: a config file can drift from a deployment, an artifact cannot drift from itself.
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from api.config import REGISTERED_MODEL_NAME, Settings
from api.exceptions import ModelNotLoadedError
from api.schemas import ModelInfo

router = APIRouter(tags=["info"])


@router.get("/model/info", response_model=ModelInfo)
def model_info(request: Request) -> ModelInfo:
    """Describe the loaded model.

    Raises:
        ModelNotLoaded: Rendered as 503, the same as ``/health``, so a client polling either gets
            a consistent answer.
    """
    from src.registry.pyfunc_wrapper import COMPUTED_COLUMNS, RAW_INPUT_COLUMNS

    model = getattr(request.app.state, "model", None)
    if model is None:
        detail = getattr(request.app.state, "load_error", "the model is not loaded")
        raise ModelNotLoadedError(detail)

    settings: Settings = request.app.state.settings
    described = model.description
    return ModelInfo(
        registered_model=REGISTERED_MODEL_NAME,
        model_version=model.version,
        source=settings.describe_source(),
        shipped_model=described["shipped_model"],
        n_features=described["n_features"],
        threshold=described["threshold"],
        has_calibrator=described["has_calibrator"],
        train_cutoff=described["train_cutoff"],
        bundled_snapshot_month=described["bundled_snapshot_month"],
        input_schema_hash=described["input_schema_hash"],
        request_columns=list(RAW_INPUT_COLUMNS),
        rejected_columns=sorted(COMPUTED_COLUMNS),
    )
