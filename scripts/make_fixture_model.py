"""Save a tiny fixture pyfunc, so CI can smoke-test the image without a registry (PLAN.md §10.3).

    WHY THE MODEL IS GENERATED AND NOT COMMITTED
    ===========================================
    §10.3 asks for "a tiny pre-trained fixture model (a 10-tree LightGBM, a few KB)" committed to
    the repo. This script generates one instead, from committed code, and the reason is a failure
    mode a committed pickle has and generated code does not:

    A pickled model in git is a **binary coupled to the library versions that wrote it**. Nothing in
    the repository records that coupling, so the day LightGBM or MLflow moves, the artifact either
    stops loading — with an error far from its cause — or, worse, loads and behaves subtly
    differently. Neither is visible in a diff.

    Generating it here costs about a second, cannot drift from the code that reads it, and is built
    with **exactly the libraries the image installs**: CI runs this with
    ``requirements-api.txt``, so the fixture and the container share their versions by construction
    rather than by hope.

    The trade is honest: a committed artifact would make the workflow marginally simpler and would
    lie about version-independence. This is the same reasoning that keeps the feature matrix out of
    the database (DECISIONS.md D24) — a stored derivative drifts from the thing that derives it.

Emits the saved model and a reference prediction, so the workflow can assert the container and the
runner agree rather than merely that the container answered — a smoke test that only checks for a
200 proves the web framework works, not the model.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

#: One request, fixed, used both to produce the reference prediction and by the workflow's POST.
#: Plausible rather than minimal: a record of zeros would exercise none of the imputation or
#: category-capping paths inside the artifact.
GOLDEN_RECORD: dict = {
    "order_purchase_timestamp": "2017-06-15T10:00:00",
    "promised_days": 20,
    "days_to_shipping_limit": 5.0,
    "customer_state": "SP",
    "seller_state": "SP",
    "customer_zip_code_prefix": "01001",
    "customer_seller_distance_km": 12.0,
    "route": "SP->SP",
    "n_seller_states": 1,
    "n_items": 1,
    "n_distinct_products": 1,
    "n_distinct_sellers": 1,
    "seller_id": "s001",
    "total_price": 100.0,
    "total_freight": 10.0,
    "total_payment_value": 110.0,
    "max_installments": 1,
    "n_payment_methods": 1,
    "dominant_payment_type": "credit_card",
    "total_weight_g": 500.0,
    "max_item_weight_g": 500.0,
    "total_volume_cm3": 1000.0,
    "max_item_volume_cm3": 1000.0,
    "dominant_category": "cama_mesa_banho",
}


def build(destination: Path) -> dict:
    """Save the fixture pyfunc and compute its reference prediction.

    Args:
        destination: Directory to save the MLflow model into. Must not already exist.

    Returns:
        ``{"path", "probability", "threshold", "is_late_predicted", "shipped_model"}``.
    """
    import mlflow
    import pandas as pd
    from tests.fixtures.raw_orders import fitted_bundle

    from src.registry.pyfunc_wrapper import raw_input_example

    bundle = fitted_bundle()
    predictor = bundle["predictor"]

    mlflow.pyfunc.save_model(
        path=str(destination),
        python_model=predictor,
        input_example=raw_input_example(bundle["orders"], n=3),
    )

    frame = pd.DataFrame(
        [
            {
                **GOLDEN_RECORD,
                "order_purchase_timestamp": pd.Timestamp(GOLDEN_RECORD["order_purchase_timestamp"]),
            }
        ]
    )
    # Through the saved model, not the in-memory object: if saving loses something, the reference
    # must lose it too, or the workflow's comparison would fail for the wrong reason.
    loaded = mlflow.pyfunc.load_model(str(destination))
    prediction = loaded.predict(frame).iloc[0]

    return {
        "path": str(destination),
        "probability": float(prediction["probability"]),
        "threshold": float(prediction["threshold"]),
        "is_late_predicted": bool(prediction["is_late_predicted"]),
        "shipped_model": predictor.model_name,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Save a fixture pyfunc for CI's image smoke test")
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="Write the reference prediction as JSON here.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    result = build(args.destination)

    if args.reference:
        args.reference.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info(
        "saved %s (%s), reference probability %.9f at threshold %.5f",
        result["path"],
        result["shipped_model"],
        result["probability"],
        result["threshold"],
    )
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
