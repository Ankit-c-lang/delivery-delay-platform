"""Error handling (PLAN.md §9): one JSON envelope, and never a stack trace.

§9's rules, and why each is a rule rather than a preference:

* **Pydantic validation failures stay 422.** FastAPI's default body names the offending field and
  the reason, which is more useful to a client than anything a custom handler would write.
* **Business-rule violations are 400**, with the envelope. A batch of 5,000 records is not a
  malformed request — it is a well-formed request the service refuses.
* **Unhandled exceptions are 500 with a correlation ID, never a traceback.** A traceback in a
  response body leaks file paths, library versions and sometimes data. The ID is logged with the
  full exception, so an operator can find it in seconds and the client still gets something to
  quote.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class BusinessRuleViolationError(Exception):
    """A well-formed request the service declines. Renders as 400.

    Args:
        error: Short machine-readable slug.
        detail: One sentence a client can act on.
    """

    def __init__(self, error: str, detail: str) -> None:
        super().__init__(detail)
        self.error = error
        self.detail = detail


class ModelNotLoadedError(Exception):
    """The model is not available, so prediction cannot be attempted. Renders as 503."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


def _envelope(error: str, detail: str, correlation_id: str) -> dict[str, str]:
    return {"error": error, "detail": detail, "correlation_id": correlation_id}


def register_handlers(app: FastAPI) -> None:
    """Attach the handlers. Called by the app factory."""

    @app.exception_handler(BusinessRuleViolationError)
    async def _business_rule(_: Request, exc: BusinessRuleViolationError) -> JSONResponse:
        correlation_id = str(uuid.uuid4())
        logger.info("400 %s: %s [%s]", exc.error, exc.detail, correlation_id)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=_envelope(exc.error, exc.detail, correlation_id),
        )

    @app.exception_handler(ModelNotLoadedError)
    async def _not_loaded(_: Request, exc: ModelNotLoadedError) -> JSONResponse:
        correlation_id = str(uuid.uuid4())
        logger.error("503 model_not_loaded: %s [%s]", exc.detail, correlation_id)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=_envelope("model_not_loaded", exc.detail, correlation_id),
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        correlation_id = str(uuid.uuid4())
        # exc_info so the traceback reaches the log and only the log.
        logger.error(
            "500 on %s %s [%s]", request.method, request.url.path, correlation_id, exc_info=exc
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_envelope(
                "internal_error",
                "The request could not be completed. Quote the correlation ID when reporting it.",
                correlation_id,
            ),
        )
