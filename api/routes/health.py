"""Liveness (PLAN.md §9).

Returns **503** when the model failed to load. This is the endpoint Compose and CI health checks
hit, so the distinction matters: a service that answers 200 while unable to predict will keep
receiving traffic, and nothing upstream will route around it.
"""

from __future__ import annotations

from fastapi import APIRouter, Request, Response, status

from api.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(request: Request, response: Response) -> HealthResponse:
    """Report whether the service can actually predict.

    Returns:
        ``{status, model_loaded, model_version}``, with HTTP 503 when the model is absent.
    """
    model = getattr(request.app.state, "model", None)
    if model is None:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthResponse(
            status="unavailable",
            model_loaded=False,
            detail=getattr(request.app.state, "load_error", "the model is not loaded"),
        )
    return HealthResponse(status="ok", model_loaded=True, model_version=model.version)
