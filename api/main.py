"""The FastAPI application (PLAN.md §9).

**One service, not microservices.** The model is a few MB, inference is sub-millisecond once the
features exist, and there is no independent scaling axis — splitting this would add network hops and
deployment surface for nothing.

**The model loads once, in the lifespan handler.** §9 names per-request loading as the mistake. The
handler does not let a load failure kill the process: it records the error and lets ``/health``
answer 503. A container that exits on a registry blip gets restarted in a loop and tells an operator
nothing; one that reports itself unhealthy is diagnosable and gets routed around. The failure is
logged loudly either way.

**Startup ordering on a clean machine.** The API resolves ``@champion`` at startup, and on a fresh
registry there is no champion — so ``make bootstrap`` (Phase 8, §18 A5) must run first. That is a
real ordering constraint, not a nicety, and it is why the startup log says which URI it tried.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.config import Settings, get_settings
from api.exceptions import register_handlers
from api.model_loader import load_model
from api.routes import health, info, predict

logger = logging.getLogger(__name__)

API_TITLE = "Delivery Delay Prediction API"
API_VERSION = "1.0.0"
API_DESCRIPTION = (
    "Predicts at checkout whether a Brazilian e-commerce order will be delivered after its "
    "promised date.\n\n"
    "All preprocessing lives inside the MLflow pyfunc this service loads, so a request carries "
    "**raw order fields only**. The as-of history features and `purchase_month` are computed from "
    "the artifact's bundled snapshots and are **rejected** if supplied — that is what stops a "
    "client bypassing the leakage boundary from outside.\n\n"
    "An unrecognised category is not an error: it is scored against the `__unknown__` level the "
    "model was trained with, and reported in `warnings`."
)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application.

    Args:
        settings: Injected settings. Omitted in production, where they come from the environment;
            a test passes its own so it can serve a locally saved fixture model without mutating
            the process environment.

    Returns:
        The configured app. The model is loaded when the app starts, not here.
    """
    resolved = settings or get_settings()
    logging.basicConfig(
        level=getattr(logging, resolved.log_level, logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = resolved
        app.state.model = None
        app.state.load_error = None
        try:
            app.state.model = load_model(resolved)
        except Exception as exc:
            app.state.load_error = f"{type(exc).__name__}: {exc}"
            logger.error(
                "the model failed to load from %s; /health will answer 503. %s",
                resolved.model_uri,
                app.state.load_error,
            )
        yield
        app.state.model = None

    app = FastAPI(
        title=API_TITLE,
        version=API_VERSION,
        description=API_DESCRIPTION,
        lifespan=lifespan,
    )
    register_handlers(app)
    app.include_router(health.router)
    app.include_router(predict.router)
    app.include_router(info.router)
    return app


app = create_app()
