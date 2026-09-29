"""Train/serve parity — the test PLAN.md §13 Phase 7 calls the one that matters most.

    THE CLAIM, STATED PRECISELY (§18 A6, CLAUDE.md invariant 8)
    ==========================================================
    **Same raw record + same artifact -> same probability within 1e-6.**

    Snapshot *selection* is explicitly outside that scope. Training joins each order to its own
    purchase-month snapshot; serving uses the single latest bundled one, because a live request has
    no history table to join against. Feed a historical order to both paths and they legitimately
    disagree — so a parity test that compared ``transform_with_snapshots`` against ``transform``
    would be testing snapshot policy, not skew, and would fail for a correct system. Both sides
    here therefore use the **serving** policy, and the asymmetry is a documented limitation rather
    than something this test can or should police.

**What it would actually catch**, which is the reason it exists rather than a ritual:

* the API reimplementing any preprocessing, and drifting from the artifact (invariant 4)
* a JSON round-trip damaging dtypes — ints arriving as floats, timestamps as strings, a null
  becoming the string ``"None"``
* column order mattering somewhere it should not
* calibration being skipped on one path, which changes probabilities while leaving the *ranking*
  identical, so PR-AUC would look fine and every served probability would be wrong
* the threshold being read from config on one side and from the artifact on the other

**Rows come from ``tests/fixtures/``, never from the 2018-05 -> 08 window** (§18 A3). Phase 9
asserts by import inspection that exactly one module loads that window, and it raises unless
unlocked by the gate, so drawing parity rows from it would make the two tests contradictory. The
fixtures also let this run in CI, which has neither the dataset nor Postgres.

**These tests need no MLflow server.** The fixture model is saved to a local path with
``mlflow.pyfunc.save_model`` and the API resolves it as a filesystem URI — the third of §9's three
``MODEL_URI`` forms, which exists precisely so CI can serve a model without a registry.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from tests.fixtures.raw_orders import fitted_bundle, serving_records

from src.registry.pyfunc_wrapper import OUTPUT_COLUMNS, RAW_INPUT_COLUMNS

#: Rows compared. §13 says 20.
N_PARITY_ROWS = 20

#: The tolerance invariant 8 fixes. Deliberately not `exact`: a float64 probability crossing a
#: process boundary as JSON is decimal text, so requiring bitwise equality would test the JSON
#: library rather than the pipeline. 1e-6 is far tighter than any skew bug would produce — a real
#: one moves probabilities in the second or third decimal.
TOLERANCE = 1e-6


@pytest.fixture(scope="module")
def bundle():
    """The fitted artifact, model and wrapped predictor, built once."""
    return fitted_bundle()


@pytest.fixture(scope="module")
def parity_rows(bundle):
    """Twenty raw records spread across the score range.

    **Not** ``head(20)``. The fixture's rows are time-ordered and the first twenty land on only two
    distinct probabilities, so a comparison over them could pass while the two paths disagreed
    everywhere else. Taking an even stride across the frame spans the distribution instead.
    """
    orders = bundle["orders"]
    stride = max(len(orders) // N_PARITY_ROWS, 1)
    chosen = orders.iloc[::stride].head(N_PARITY_ROWS)
    assert len(chosen) == N_PARITY_ROWS
    return serving_records(chosen).reset_index(drop=True)


@pytest.fixture(scope="module")
def saved_model_path(bundle, tmp_path_factory):
    """The fixture predictor saved to a local path, as §9's third MODEL_URI form."""
    import mlflow

    from src.registry.pyfunc_wrapper import raw_input_example

    path = tmp_path_factory.mktemp("fixture_model") / "model"
    example = raw_input_example(bundle["orders"], n=3)
    mlflow.pyfunc.save_model(
        path=str(path),
        python_model=bundle["predictor"],
        input_example=example,
    )
    return path


@pytest.fixture(scope="module")
def client(saved_model_path):
    """A TestClient whose app resolved the local fixture model at startup.

    Settings are overridden before the app is constructed, because the model loads once in the
    lifespan handler — §9 is explicit that loading per request would be the mistake.
    """
    from fastapi.testclient import TestClient

    from api.config import get_settings
    from api.main import create_app

    get_settings.cache_clear()
    settings = get_settings(_model_uri=str(saved_model_path))
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client
    get_settings.cache_clear()


def training_path_probabilities(bundle, records: pd.DataFrame) -> np.ndarray:
    """Score records by driving the artifact, model and calibrator directly in-process.

    This is §13's "training-time path (PreprocessingArtifact + model directly)": no wrapper, no
    HTTP, no serialization. Written out step by step on purpose — calling the wrapper here would
    make the test compare the wrapper against itself.
    """
    artifact = bundle["artifact"]
    features = artifact.transform(records)
    raw = bundle["model"].predict_proba(features)[:, 1]
    return np.asarray(artifact.calibrator.predict(raw), dtype=float)


class TestTheFixtureCanFail:
    """Guards against a vacuous pass. A parity test over constants proves nothing."""

    def test_the_twenty_rows_span_several_distinct_probabilities(self, bundle, parity_rows):
        probabilities = training_path_probabilities(bundle, parity_rows)
        assert len(np.unique(probabilities)) >= 3, (
            "the parity rows collapsed onto fewer than three distinct probabilities; "
            "a comparison over them could not detect a difference between the two paths"
        )

    def test_the_probabilities_are_not_degenerate(self, bundle, parity_rows):
        probabilities = training_path_probabilities(bundle, parity_rows)
        assert probabilities.min() > 0.0
        assert probabilities.max() < 1.0

    def test_both_decisions_appear_somewhere_in_the_fixture(self, bundle):
        """If every row fell on one side of the threshold, `is_late_predicted` would be untested."""
        everything = serving_records(bundle["orders"])
        out = bundle["predictor"].predict(model_input=everything)
        assert out["is_late_predicted"].any()
        assert not out["is_late_predicted"].all()


class TestParity:
    """§13's required test: 20 rows, both paths, the same artifact, within 1e-6."""

    def test_single_predictions_match_the_training_path(self, bundle, parity_rows, client):
        expected = training_path_probabilities(bundle, parity_rows)
        for position in range(len(parity_rows)):
            payload = _record_payload(parity_rows.iloc[position])
            response = client.post("/predict", json=payload)
            assert response.status_code == 200, response.text
            assert response.json()["probability"] == pytest.approx(
                expected[position], abs=TOLERANCE
            )

    def test_batch_predictions_match_the_training_path(self, bundle, parity_rows, client):
        expected = training_path_probabilities(bundle, parity_rows)
        response = client.post("/predict/batch", json={"records": _batch_payload(parity_rows)})
        assert response.status_code == 200, response.text
        served = np.array([row["probability"] for row in response.json()["predictions"]])
        assert served == pytest.approx(expected, abs=TOLERANCE)

    def test_batching_does_not_change_a_prediction(self, parity_rows, client):
        """A row must score the same alone as in company.

        Not pedantry: anything fitted or normalised across the request — a scaler refitted per
        batch, a groupby, a fillna with the batch mean — would pass every other test here and
        break exactly this one.
        """
        batch = client.post("/predict/batch", json={"records": _batch_payload(parity_rows)}).json()[
            "predictions"
        ]
        for position in range(len(parity_rows)):
            single = client.post(
                "/predict", json=_record_payload(parity_rows.iloc[position])
            ).json()
            assert single["probability"] == pytest.approx(
                batch[position]["probability"], abs=TOLERANCE
            )

    def test_row_order_does_not_change_a_prediction(self, parity_rows, client):
        """The same rows reversed must return the same probabilities, reversed."""
        forward = client.post(
            "/predict/batch", json={"records": _batch_payload(parity_rows)}
        ).json()["predictions"]
        reversed_rows = parity_rows.iloc[::-1].reset_index(drop=True)
        backward = client.post(
            "/predict/batch", json={"records": _batch_payload(reversed_rows)}
        ).json()["predictions"]
        for position in range(len(parity_rows)):
            assert forward[position]["probability"] == pytest.approx(
                backward[-1 - position]["probability"], abs=TOLERANCE
            )

    def test_the_threshold_and_decision_survive_the_boundary(self, bundle, parity_rows, client):
        """The decision must come from the artifact's threshold, not from a config the API reads."""
        expected_threshold = float(bundle["artifact"].threshold)
        response = client.post("/predict/batch", json={"records": _batch_payload(parity_rows)})
        for row in response.json()["predictions"]:
            assert row["threshold"] == pytest.approx(expected_threshold, abs=1e-12)
            assert row["is_late_predicted"] == (row["probability"] >= expected_threshold)

    def test_a_null_category_survives_the_json_round_trip(self, bundle, client):
        """`dominant_category` is null for real orders, and JSON null is where dtypes go wrong.

        A null that arrives as the string "None" maps to `__unknown__` instead of `__missing__` —
        two different facts, and the models were trained to tell them apart.
        """
        records = serving_records(bundle["orders"])
        nulls = records.index[records["dominant_category"].isna()]
        assert len(nulls), "fixture no longer contains a null category; the test would be vacuous"

        row = records.loc[[nulls[0]]]
        expected = training_path_probabilities(bundle, row)[0]
        response = client.post("/predict", json=_record_payload(row.iloc[0]))
        assert response.status_code == 200, response.text
        assert response.json()["probability"] == pytest.approx(expected, abs=TOLERANCE)


class TestParityIsScopedToOneArtifact:
    """§18 A6 made executable: the asymmetry is documented behaviour, not a parity failure."""

    def test_the_two_transforms_disagree_and_that_is_the_documented_asymmetry(self, bundle):
        """Proving the asymmetry exists is what makes A6's wording necessary.

        If these agreed, "parity is scoped to one artifact" would be an empty distinction. They
        disagree because the training path joins each order to its own purchase month while serving
        uses the bundled month — so the scope limit is load-bearing, not defensive.
        """
        artifact, orders, snapshots = bundle["artifact"], bundle["orders"], bundle["snapshots"]
        training = artifact.transform_with_snapshots(orders, snapshots)
        serving = artifact.transform(serving_records(orders))

        history = [column for column in training.columns if column.endswith("_hist")]
        assert not training[history].equals(serving[history])

    def test_serving_forces_the_bundled_snapshot_month(self, bundle):
        """Which is why `purchase_month` is refused as input (DECISIONS.md D34)."""
        artifact = bundle["artifact"]
        records = serving_records(bundle["orders"])
        assert "purchase_month" not in records.columns

        first = artifact.transform(records.head(1))
        last = artifact.transform(records.tail(1))
        assert list(first.columns) == list(last.columns) == list(artifact.feature_names)


class TestTheApiAddsNoPreprocessing:
    """CLAUDE.md invariant 4, checked structurally rather than by reading the code."""

    def test_the_api_package_imports_nothing_from_src_features(self):
        """The API may import the wrapper's *contract*, never the feature machinery.

        Import inspection rather than a naming convention: this is the check that fails when
        someone adds "just one" normalisation to a route.
        """
        import ast
        from pathlib import Path

        offenders: list[str] = []
        for path in sorted(Path("api").rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                modules = []
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    modules = [node.module]
                for module in modules:
                    if module.startswith("src.features") or module.startswith("src.training"):
                        offenders.append(f"{path}:{node.lineno} imports {module}")
        assert not offenders, "the API layer must contain no preprocessing:\n" + "\n".join(
            offenders
        )

    def test_the_request_contract_is_the_wrappers_own_column_list(self):
        """So the API cannot accept a shape the pyfunc would reject, or vice versa."""
        from api.schemas import OrderRecord

        assert tuple(sorted(OrderRecord.model_fields)) == tuple(sorted(RAW_INPUT_COLUMNS))

    def test_the_response_carries_the_wrappers_own_output_columns(self):
        from api.schemas import Prediction

        for column in OUTPUT_COLUMNS:
            assert column in Prediction.model_fields


def _record_payload(row: pd.Series) -> dict:
    """One row as JSON-ready primitives, the way a client would send it."""
    payload: dict = {}
    for column, value in row.items():
        if pd.isna(value):
            payload[str(column)] = None
        elif isinstance(value, pd.Timestamp):
            payload[str(column)] = value.isoformat()
        elif isinstance(value, np.integer):
            payload[str(column)] = int(value)
        elif isinstance(value, np.floating):
            payload[str(column)] = float(value)
        else:
            payload[str(column)] = value
    return payload


def _batch_payload(frame: pd.DataFrame) -> list[dict]:
    return [_record_payload(frame.iloc[i]) for i in range(len(frame))]
