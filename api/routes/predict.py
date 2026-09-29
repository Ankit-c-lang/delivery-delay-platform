"""Scoring endpoints (PLAN.md §9).

**No preprocessing here** (CLAUDE.md invariant 4). These handlers turn validated Pydantic models
into a DataFrame, hand it to the pyfunc, and turn the result back into JSON. Nothing in this file
knows what a category level set, an imputation median or an as-of snapshot is — and
``tests/test_parity.py`` enforces that by import inspection, not by convention.

**An unseen category is normal traffic, not an error.** §9 is explicit: map it to the
``__unknown__`` sentinel the models were trained with, return 200, and say so in ``warnings``. A
500 would be wrong because nothing failed, and a silent 200 would be worse because a growing
product catalogue is exactly the drift signal an operator wants to see early.
"""

from __future__ import annotations

import logging
import time

import pandas as pd
from fastapi import APIRouter, Request

from api.exceptions import BusinessRuleViolationError, ModelNotLoadedError
from api.schemas import BatchRequest, BatchResponse, OrderRecord, Prediction

logger = logging.getLogger(__name__)

router = APIRouter(tags=["predict"])


def _loaded(request: Request):
    """The loaded model, or a 503. Never loads on demand — §9 forbids per-request loading."""
    model = getattr(request.app.state, "model", None)
    if model is None:
        detail = getattr(request.app.state, "load_error", "the model is not loaded")
        raise ModelNotLoadedError(detail)
    return model


def _to_frame(records: list[OrderRecord]) -> pd.DataFrame:
    """Validated records to the frame the pyfunc expects.

    ``model_dump`` rather than hand-built dicts, so a field added to
    :class:`~api.schemas.OrderRecord` reaches the model without a second edit here. Column order is
    irrelevant — the artifact selects by name — but the frame is built from the declared field order
    anyway, so a diff of a request and a frame reads the same way.
    """
    return pd.DataFrame([record.model_dump() for record in records])


def _warning_lines(report: dict[str, list[str]]) -> list[str]:
    """The wrapper's report as sentences a client can log or surface.

    Built here rather than in the wrapper because phrasing is a presentation concern; the wrapper
    returns the facts.

    **The wording says "not among the levels this model knows", not "never seen in training"**, and
    the difference is not pedantry. Two situations both produce ``__unknown__``:

    1. a value genuinely absent from the fit window — e.g. `seller_state` `'AM'`, `'MA'`, `'PI'`,
       which have too few sellers to appear;
    2. a value that *was* present and was **deliberately capped** — `product_category` keeps the
       top 30 of 72, covering 95.80% of fit-window rows, so the other 42 collapse by design.

    Measured on all 96,203 real orders, the champion reports 3 states and 43 categories. Calling
    those "never seen" would be false for most of them, and a warning that overstates its case is
    one an operator learns to ignore — which would cost the genuine drift signal in case 1.
    """
    lines: list[str] = []
    for feature in sorted(report):
        values = report[feature]
        if values:
            shown = ", ".join(repr(value) for value in values[:5])
            more = f" (+{len(values) - 5} more)" if len(values) > 5 else ""
            lines.append(
                f"{feature}: {shown}{more} is not among the levels this model was trained on and "
                "was scored as '__unknown__'"
            )
        else:
            lines.append(
                f"{feature}: a value outside this model's known levels was scored as '__unknown__'"
            )
    return lines


@router.post("/predict", response_model=Prediction)
def predict(record: OrderRecord, request: Request) -> Prediction:
    """Score one order.

    Args:
        record: One raw order. History columns and ``purchase_month`` are refused by the schema —
            the service computes them (DECISIONS.md D34).

    Returns:
        The calibrated probability, the decision, the threshold used, the version served and any
        warnings.
    """
    model = _loaded(request)
    predictions, report = model.score(_to_frame([record]))
    row = predictions.iloc[0]
    return Prediction(
        probability=float(row["probability"]),
        is_late_predicted=bool(row["is_late_predicted"]),
        threshold=float(row["threshold"]),
        model_version=model.version,
        warnings=_warning_lines(report),
    )


@router.post("/predict/batch", response_model=BatchResponse)
def predict_batch(payload: BatchRequest, request: Request) -> BatchResponse:
    """Score up to ``max_batch_records`` orders in one call.

    Args:
        payload: ``{"records": [...]}``.

    Raises:
        BusinessRuleViolation: Rendered as 400 when the batch exceeds the cap. Deliberately not a
            422: the request is well formed and the service is declining it, and conflating the two
            makes a client retry a request that can never succeed.

    Warnings are reported per batch rather than per row, and the same list is attached to every
    prediction in the response. Per-row attribution would need the row index threaded back out of
    the wrapper for a field whose purpose is to tell an operator "your catalogue has moved" — a
    fact about the request, not about a row.
    """
    settings = request.app.state.settings
    if len(payload.records) > settings.max_batch_records:
        raise BusinessRuleViolationError(
            "batch_too_large",
            f"{len(payload.records)} records exceeds the cap of {settings.max_batch_records}. "
            "Split the batch: an uncapped batch is a tail-latency and availability problem, not a "
            "throughput win.",
        )

    model = _loaded(request)
    started = time.perf_counter()
    predictions, report = model.score(_to_frame(payload.records))
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    warnings = _warning_lines(report)
    if warnings:
        logger.info("batch of %d produced %d warning(s)", len(payload.records), len(warnings))

    rows = [
        Prediction(
            probability=float(row.probability),
            is_late_predicted=bool(row.is_late_predicted),
            threshold=float(row.threshold),
            model_version=model.version,
            warnings=warnings,
        )
        for row in predictions.itertuples(index=False)
    ]
    return BatchResponse(predictions=rows, count=len(rows), elapsed_ms=round(elapsed_ms, 3))
