"""Request and response models (PLAN.md §9, Pydantic v2).

**The 24 request fields are written out explicitly, not generated from
:data:`~src.registry.pyfunc_wrapper.RAW_INPUT_COLUMNS`.** Generating them would guarantee the names
match but would lose the types, and the types are what turn a malformed request into a 422 instead
of a confusing 500 deep inside pandas. The completeness guarantee is recovered by a test asserting
``OrderRecord.model_fields`` equals ``RAW_INPUT_COLUMNS`` exactly — so adding a feature without
touching this file fails loudly, which is the property that mattered.

**What is deliberately not validated here:** anything requiring knowledge of how a feature is
built. Category membership, imputation, winsorisation and the as-of join all belong to the
preprocessing artifact inside the pyfunc (CLAUDE.md invariant 4). An unseen category is *not* an
error — §9 requires 200 with a ``warnings`` field, because a growing product catalogue is normal
traffic, not a client mistake.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class OrderRecord(BaseModel):
    """One order as a client sends it at checkout.

    Exactly :data:`~src.registry.pyfunc_wrapper.RAW_INPUT_COLUMNS`. Notably **absent**:

    * the 15 as-of history columns — attached from the artifact's bundled snapshots, and refused if
      supplied, so a client cannot bypass the §4.5 leakage boundary from outside
    * ``purchase_month`` — serving forces it to the bundled snapshot's month (§18 A6), so accepting
      it would be a lie about what the service does (DECISIONS.md D34)
    """

    model_config = ConfigDict(extra="forbid")

    # --- timing -----------------------------------------------------------------------------
    order_purchase_timestamp: datetime = Field(
        description="When checkout completed. The prediction-time boundary: nothing later exists."
    )
    promised_days: int = Field(
        ge=0, description="Days from purchase to the estimated delivery date shown to the customer."
    )
    days_to_shipping_limit: float = Field(
        description="Days from purchase to the seller's contractual shipping deadline. "
        "Winsorised inside the artifact, so extreme values are accepted here rather than rejected."
    )

    # --- geography --------------------------------------------------------------------------
    customer_state: str = Field(min_length=1, max_length=8)
    seller_state: str = Field(min_length=1, max_length=8)
    customer_zip_code_prefix: str = Field(min_length=1, max_length=16)
    customer_seller_distance_km: float | None = Field(
        default=None,
        ge=0,
        description="Null is acceptable: it falls back to a state centroid inside the artifact.",
    )
    route: str = Field(min_length=1, description="Seller state to customer state, e.g. 'SP->MG'.")
    n_seller_states: int = Field(ge=1)

    # --- composition ------------------------------------------------------------------------
    n_items: int = Field(ge=1)
    n_distinct_products: int = Field(ge=1)
    n_distinct_sellers: int = Field(ge=1)
    seller_id: str = Field(min_length=1)

    # --- money ------------------------------------------------------------------------------
    total_price: float = Field(ge=0)
    total_freight: float = Field(ge=0)
    total_payment_value: float = Field(ge=0)
    max_installments: int = Field(ge=1)
    n_payment_methods: int = Field(ge=1)
    dominant_payment_type: str | None = Field(default=None)

    # --- physical ---------------------------------------------------------------------------
    total_weight_g: float | None = Field(default=None, ge=0)
    max_item_weight_g: float | None = Field(default=None, ge=0)
    total_volume_cm3: float | None = Field(default=None, ge=0)
    max_item_volume_cm3: float | None = Field(default=None, ge=0)

    # --- product ----------------------------------------------------------------------------
    dominant_category: str | None = Field(
        default=None,
        description="Null for real orders whose products carry no category. Distinct from a "
        "category unseen in training: the models were trained to tell those two apart.",
    )


class Prediction(BaseModel):
    """One scored order.

    ``threshold`` is returned rather than assumed so a caller can record which operating point
    produced a decision without looking it up somewhere else — the same reasoning that puts the
    threshold inside the artifact instead of in a config file beside it.
    """

    probability: float = Field(ge=0.0, le=1.0, description="Calibrated probability of being late.")
    is_late_predicted: bool
    threshold: float
    model_version: str = Field(description="Registry version served, or 'local' for a path URI.")
    warnings: list[str] = Field(
        default_factory=list,
        description="Non-fatal notes, e.g. a category not seen during training. A warning never "
        "means the prediction failed.",
    )


class BatchRequest(BaseModel):
    """A batch of orders. Capped, because an uncapped batch is an availability problem."""

    model_config = ConfigDict(extra="forbid")

    records: list[OrderRecord] = Field(min_length=1)


class BatchResponse(BaseModel):
    """Predictions plus timing, so a client can see cost without instrumenting the service."""

    predictions: list[Prediction]
    count: int
    elapsed_ms: float


class HealthResponse(BaseModel):
    """Liveness. Returns 503 when the model failed to load (§9).

    A service that answers 200 while unable to predict is worse than one that is down, because
    nothing upstream will route around it.
    """

    status: str
    model_loaded: bool
    model_version: str | None = None
    detail: str | None = None


class ModelInfo(BaseModel):
    """What is actually loaded (§9): version, alias, window, features, threshold, snapshot."""

    registered_model: str
    model_version: str
    source: dict[str, str | None]
    shipped_model: str
    n_features: int
    threshold: float
    has_calibrator: bool
    train_cutoff: str
    bundled_snapshot_month: str
    input_schema_hash: str
    request_columns: list[str]
    rejected_columns: list[str] = Field(
        description="Columns the wrapper computes and refuses as input: the as-of history features "
        "and purchase_month."
    )


class ErrorEnvelope(BaseModel):
    """One shape for every error, so a client parses failures the same way every time."""

    error: str
    detail: str
    correlation_id: str
