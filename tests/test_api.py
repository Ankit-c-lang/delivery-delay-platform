"""API behaviour (PLAN.md §9, §13 Phase 7).

Covers the phase's required list: health when loaded and when not, single and batch predict, 422 on
malformed input, the batch cap, and an unseen category returning 200 with a warning. Parity lives in
``tests/test_parity.py`` — it is the phase's headline test and deserves its own file.

Everything here runs against a **locally saved fixture model**, so no MLflow server and no Postgres
are needed and CI can run the whole file (§10.3).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from tests.fixtures.raw_orders import fitted_bundle, serving_records

from api.config import ModelUriKind, Settings, get_settings
from api.main import create_app


@pytest.fixture(scope="module")
def bundle():
    return fitted_bundle()


@pytest.fixture(scope="module")
def saved_model_path(bundle, tmp_path_factory):
    import mlflow

    from src.registry.pyfunc_wrapper import raw_input_example

    path = tmp_path_factory.mktemp("api_model") / "model"
    mlflow.pyfunc.save_model(
        path=str(path),
        python_model=bundle["predictor"],
        input_example=raw_input_example(bundle["orders"], n=3),
    )
    return path


@pytest.fixture(scope="module")
def client(saved_model_path):
    get_settings.cache_clear()
    app = create_app(Settings(model_uri=str(saved_model_path)))
    with TestClient(app) as test_client:
        yield test_client
    get_settings.cache_clear()


@pytest.fixture(scope="module")
def one_record(bundle) -> dict:
    """A single valid request body."""
    row = serving_records(bundle["orders"]).iloc[0]
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


class TestHealth:
    def test_health_is_ok_and_names_the_version_when_loaded(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert body["model_version"] == "local"

    def test_health_is_503_when_the_model_failed_to_load(self, tmp_path):
        """The endpoint Compose and CI hit, so this is the case that must not answer 200.

        A service reporting healthy while unable to predict keeps receiving traffic and nothing
        upstream routes around it — strictly worse than being visibly down.
        """
        app = create_app(Settings(model_uri=str(tmp_path / "does-not-exist")))
        with TestClient(app, raise_server_exceptions=False) as broken:
            response = broken.get("/health")
        assert response.status_code == 503
        body = response.json()
        assert body["model_loaded"] is False
        assert "does-not-exist" in body["detail"]

    def test_a_bad_uri_does_not_kill_the_process(self, tmp_path):
        """The app must still start, or a registry blip becomes a restart loop."""
        app = create_app(Settings(model_uri=str(tmp_path / "nope")))
        with TestClient(app, raise_server_exceptions=False) as broken:
            assert broken.get("/health").status_code == 503
            # and prediction reports the same thing rather than a 500
            assert broken.post("/predict", json={}).status_code in (422, 503)


class TestPredict:
    def test_a_single_prediction_returns_the_full_contract(self, client, one_record):
        response = client.post("/predict", json=one_record)
        assert response.status_code == 200, response.text
        body = response.json()
        assert 0.0 <= body["probability"] <= 1.0
        assert body["is_late_predicted"] == (body["probability"] >= body["threshold"])
        assert body["model_version"] == "local"
        assert body["warnings"] == []

    def test_a_batch_returns_one_prediction_per_record_with_timing(self, client, one_record):
        response = client.post("/predict/batch", json={"records": [one_record] * 5})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["count"] == 5
        assert len(body["predictions"]) == 5
        assert body["elapsed_ms"] > 0

    def test_identical_records_score_identically(self, client, one_record):
        body = client.post("/predict/batch", json={"records": [one_record] * 3}).json()
        probabilities = {row["probability"] for row in body["predictions"]}
        assert len(probabilities) == 1


class TestValidation:
    def test_a_missing_field_is_422(self, client, one_record):
        broken = {key: value for key, value in one_record.items() if key != "promised_days"}
        response = client.post("/predict", json=broken)
        assert response.status_code == 422
        assert "promised_days" in response.text

    def test_a_wrongly_typed_field_is_422(self, client, one_record):
        response = client.post("/predict", json={**one_record, "n_items": "several"})
        assert response.status_code == 422

    def test_a_negative_price_is_422(self, client, one_record):
        """§9 wants business-rule violations refused. A price below zero is not a real order."""
        response = client.post("/predict", json={**one_record, "total_price": -5.0})
        assert response.status_code == 422

    def test_an_unknown_field_is_422_rather_than_ignored(self, client, one_record):
        """`extra="forbid"`, so a typo in a field name fails loudly.

        Silently ignoring an unexpected key is how a client ends up believing it sent a value that
        never reached the model.
        """
        response = client.post("/predict", json={**one_record, "promissed_days": 12})
        assert response.status_code == 422

    def test_supplying_a_history_column_is_refused(self, client, one_record):
        """A client must not be able to override the leakage-safe snapshot join from outside."""
        response = client.post("/predict", json={**one_record, "seller_late_rate_hist": 0.99})
        assert response.status_code == 422

    def test_supplying_purchase_month_is_refused(self, client, one_record):
        """DECISIONS.md D34: serving forces it, so accepting it would be a lie."""
        response = client.post(
            "/predict", json={**one_record, "purchase_month": "2017-01-01T00:00:00"}
        )
        assert response.status_code == 422

    def test_an_empty_batch_is_422(self, client):
        assert client.post("/predict/batch", json={"records": []}).status_code == 422


class TestTheBatchCap:
    def test_over_the_cap_is_400_with_the_envelope(self, client, one_record):
        """400, not 422: the request is well formed and the service is declining it.

        Conflating the two makes a client retry something that can never succeed.
        """
        response = client.post("/predict/batch", json={"records": [one_record] * 1001})
        assert response.status_code == 400
        body = response.json()
        assert body["error"] == "batch_too_large"
        assert "1001" in body["detail"]
        assert body["correlation_id"]

    def test_exactly_at_the_cap_is_accepted(self, saved_model_path, one_record):
        """The boundary is inclusive. Tested with a small cap so the assertion is cheap."""
        app = create_app(Settings(model_uri=str(saved_model_path), max_batch_records=4))
        with TestClient(app) as small:
            at_cap = small.post("/predict/batch", json={"records": [one_record] * 4})
            over_cap = small.post("/predict/batch", json={"records": [one_record] * 5})
        assert at_cap.status_code == 200
        assert over_cap.status_code == 400


class TestUnseenCategories:
    """§9: map to `__unknown__`, return 200 with a warning. Never a 500."""

    def test_an_unseen_category_returns_200_with_a_warning(self, client, one_record):
        response = client.post(
            "/predict", json={**one_record, "dominant_category": "quantum_widgets"}
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert 0.0 <= body["probability"] <= 1.0
        assert any("quantum_widgets" in line for line in body["warnings"])
        assert any("__unknown__" in line for line in body["warnings"])

    def test_an_unseen_state_is_also_warned(self, client, one_record):
        response = client.post("/predict", json={**one_record, "customer_state": "ZZ"})
        assert response.status_code == 200, response.text
        assert any("customer_state" in line for line in response.json()["warnings"])

    def test_several_unseen_values_produce_several_warnings(self, client, one_record):
        response = client.post(
            "/predict",
            json={**one_record, "customer_state": "ZZ", "dominant_category": "quantum_widgets"},
        )
        assert response.status_code == 200
        assert len(response.json()["warnings"]) >= 2

    def test_a_null_category_is_not_a_warning(self, client, one_record):
        """Null and unseen are different facts: `__missing__` and `__unknown__`.

        The models were trained to tell them apart, so reporting a null as unseen would be wrong.
        """
        response = client.post("/predict", json={**one_record, "dominant_category": None})
        assert response.status_code == 200
        assert response.json()["warnings"] == []


class TestModelInfo:
    def test_info_describes_the_loaded_artifact(self, client, bundle):
        response = client.get("/model/info")
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["registered_model"] == "delivery_delay_classifier"
        assert body["shipped_model"] == "lightgbm"
        assert body["n_features"] == len(bundle["artifact"].feature_names)
        assert body["threshold"] == pytest.approx(float(bundle["artifact"].threshold))
        assert body["has_calibrator"] is True
        assert body["source"]["kind"] == ModelUriKind.LOCAL_PATH.value

    def test_info_reports_the_request_contract_and_what_is_rejected(self, client):
        from src.registry.pyfunc_wrapper import COMPUTED_COLUMNS, RAW_INPUT_COLUMNS

        body = client.get("/model/info").json()
        assert body["request_columns"] == list(RAW_INPUT_COLUMNS)
        assert set(body["rejected_columns"]) == set(COMPUTED_COLUMNS)
        assert "purchase_month" in body["rejected_columns"]

    def test_info_reports_the_bundled_snapshot_month(self, client, bundle):
        """The D33 number, visible to an operator without opening the artifact."""
        body = client.get("/model/info").json()
        assert body["bundled_snapshot_month"].startswith(
            str(bundle["artifact"].latest_snapshot_month.date())
        )

    def test_info_is_503_when_nothing_is_loaded(self, tmp_path):
        app = create_app(Settings(model_uri=str(tmp_path / "absent")))
        with TestClient(app, raise_server_exceptions=False) as broken:
            response = broken.get("/model/info")
        assert response.status_code == 503
        assert response.json()["error"] == "model_not_loaded"


class TestModelUriResolution:
    """§9's three forms. The third is what lets CI serve a model with no registry."""

    def test_an_alias_uri_is_classified_and_parsed(self):
        settings = Settings(model_uri="models:/delivery_delay_classifier@champion")
        assert settings.model_uri_kind is ModelUriKind.ALIAS
        assert settings.model_alias == "champion"
        assert settings.model_version_pin is None
        assert settings.needs_tracking_server is True

    def test_a_pinned_version_uri_is_the_rollback_path(self):
        settings = Settings(model_uri="models:/delivery_delay_classifier/3")
        assert settings.model_uri_kind is ModelUriKind.PINNED_VERSION
        assert settings.model_version_pin == "3"
        assert settings.model_alias is None

    def test_a_local_path_needs_no_tracking_server(self, saved_model_path):
        settings = Settings(model_uri=str(saved_model_path))
        assert settings.model_uri_kind is ModelUriKind.LOCAL_PATH
        assert settings.needs_tracking_server is False
        assert settings.describe_source()["tracking_uri"] is None

    def test_the_default_uri_is_the_champion_alias(self):
        assert Settings().model_uri == "models:/delivery_delay_classifier@champion"


class TestTheModelLoadsOnce:
    def test_repeated_requests_reuse_one_loaded_object(self, client):
        """§9 names per-request loading as the mistake: hundreds of ms and a hammered registry.

        Identity, not timing: a timing assertion would be flaky on a loaded machine, and identity is
        what the requirement actually is.
        """
        first = client.app.state.model
        for _ in range(3):
            assert client.get("/health").status_code == 200
        assert client.app.state.model is first

    def test_a_signature_mismatch_refuses_to_start_rather_than_serving(self, tmp_path, bundle):
        """A model whose contract has moved must fail at startup, not return 500s under load."""
        import mlflow
        from mlflow.models import ModelSignature
        from mlflow.types import ColSpec, Schema

        path = tmp_path / "wrong_signature"
        mlflow.pyfunc.save_model(
            path=str(path),
            python_model=bundle["predictor"],
            signature=ModelSignature(inputs=Schema([ColSpec("double", "not_a_real_column")])),
        )
        app = create_app(Settings(model_uri=str(path)))
        with TestClient(app, raise_server_exceptions=False) as broken:
            body = broken.get("/health").json()
        assert body["model_loaded"] is False
        assert "signature" in body["detail"]
