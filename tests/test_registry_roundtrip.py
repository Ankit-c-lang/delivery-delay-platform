"""Registry round-trip tests (PLAN.md §13 Phase 6, §7).

The phase's required test is the round-trip: **log a model, load it back by URI in a clean
context, and assert the predictions match the in-memory object to 1e-9.** That is the test that
would catch the failure §13 calls the classic skew bug — a model logged without its
preprocessing artifact loads fine, predicts fine, and is quietly wrong.

These tests run against a **temporary SQLite tracking store**, not the Compose service, so they
need no running server and can execute in CI. A file-backed store cannot host a Model Registry;
SQLite can, which is the same reason §7 chooses it for the real deployment (DECISIONS.md D12).
"""

from __future__ import annotations

import os
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from src.features.artifact import PreprocessingArtifact
from src.features.history import build_all_snapshots
from src.registry.pyfunc_wrapper import (
    HISTORY_COLUMNS,
    OUTPUT_COLUMNS,
    RAW_INPUT_COLUMNS,
    DelayPredictor,
    raw_input_example,
)

N = 400


@pytest.fixture(autouse=True)
def _isolate_tracking_uri():
    """Stop a temp-registry test from redirecting every later test.

    ``mlflow.set_tracking_uri`` writes ``MLFLOW_TRACKING_URI`` into the **process
    environment**, not just an in-memory global. So a test that points MLflow at a throwaway
    SQLite store leaves that pointer behind, and a later test reading the env var gets
    ``sqlite:////tmp/pytest-.../mlflow.db`` instead of the Compose service. That is exactly how
    the fresh-process test came to skip itself with "tracking server not reachable".

    Snapshotting both the env var and MLflow's own global, and restoring them after every test
    in this module, keeps the leak contained. Worth having before Phases 7-9 add more MLflow
    tests.
    """
    import mlflow

    original_env = os.environ.get("MLFLOW_TRACKING_URI")
    original_uri = mlflow.get_tracking_uri()
    try:
        yield
    finally:
        mlflow.set_tracking_uri(original_uri)
        if original_env is None:
            os.environ.pop("MLFLOW_TRACKING_URI", None)
        else:
            os.environ["MLFLOW_TRACKING_URI"] = original_env


def synthetic_orders(n: int = N, seed: int = 7) -> pd.DataFrame:
    """Raw order records carrying every column the wrapper requires."""
    rng = np.random.default_rng(seed)
    purchased = pd.Timestamp("2017-06-01") + pd.to_timedelta(np.arange(n) * 6, unit="h")
    states = np.array(["SP", "RJ", "MG", "BA"])[np.arange(n) % 4]
    frame = pd.DataFrame(
        {
            "order_id": [f"o{i:05d}" for i in range(n)],
            "order_purchase_timestamp": purchased,
            "purchase_month": purchased.to_period("M").to_timestamp(),
            "promised_days": rng.integers(5, 40, n),
            "days_to_shipping_limit": rng.gamma(4.0, 1.5, n),
            "customer_state": states,
            "seller_state": np.where(np.arange(n) % 2 == 0, "SP", "RJ"),
            "customer_zip_code_prefix": np.array(["01001", "20010", "30110", "40010"])[
                np.arange(n) % 4
            ],
            "customer_seller_distance_km": rng.uniform(5, 1800, n),
            "n_seller_states": np.ones(n, dtype=int),
            "n_items": rng.integers(1, 4, n),
            "n_distinct_products": np.ones(n, dtype=int),
            "n_distinct_sellers": np.ones(n, dtype=int),
            "total_price": rng.uniform(20, 900, n),
            "total_freight": rng.uniform(5, 90, n),
            "total_weight_g": rng.uniform(100, 9000, n),
            "max_item_weight_g": rng.uniform(100, 9000, n),
            "total_volume_cm3": rng.uniform(500, 40000, n),
            "max_item_volume_cm3": rng.uniform(500, 40000, n),
            "dominant_category": np.array(["cama_mesa_banho", "beleza_saude", None])[
                np.arange(n) % 3
            ],
            "dominant_payment_type": np.array(["credit_card", "boleto"])[np.arange(n) % 2],
            "max_installments": rng.integers(1, 10, n),
            "total_payment_value": rng.uniform(25, 950, n),
            "n_payment_methods": np.ones(n, dtype=int),
            "seller_id": [f"s{i % 12:03d}" for i in range(n)],
            "route": np.where(np.arange(n) % 2 == 0, "SP->SP", "RJ->RJ"),
        }
    )
    return frame


def synthetic_resolved(orders: pd.DataFrame, seed: int = 8) -> pd.DataFrame:
    """Resolved history for the snapshot builder, all delivered before the orders' months."""
    rng = np.random.default_rng(seed)
    n = len(orders)
    return pd.DataFrame(
        {
            "seller_id": orders["seller_id"],
            "seller_state": orders["seller_state"],
            "route": orders["route"],
            "customer_state": orders["customer_state"],
            "dominant_category": orders["dominant_category"],
            "order_delivered_customer_date": pd.Timestamp("2017-03-01")
            + pd.to_timedelta(rng.integers(0, 40, n), unit="D"),
            "is_late": rng.random(n) < 0.08,
            "delivery_days": rng.uniform(3, 40, n),
            "handling_days": rng.uniform(0.5, 8, n),
        }
    )


@pytest.fixture(scope="module")
def wrapper_and_data():
    """A fitted artifact, a small real model and a wrapped predictor."""
    from lightgbm import LGBMClassifier
    from sklearn.isotonic import IsotonicRegression

    from src.features.history import attach_history

    orders = synthetic_orders()
    resolved = synthetic_resolved(orders)
    months = list(pd.date_range("2017-04-01", "2017-08-01", freq="MS"))
    snapshots = build_all_snapshots(resolved, months)

    artifact = PreprocessingArtifact.fit(
        attach_history(orders, snapshots),
        snapshots=snapshots,
        zip_centroids=pd.DataFrame(
            {
                "zip_code_prefix": ["01001"],
                "lat": [-23.55],
                "lng": [-46.63],
                "state": ["SP"],
                "n_points": [4],
            }
        ),
        state_centroids=pd.DataFrame(
            {"state": ["SP"], "lat": [-23.55], "lng": [-46.63], "n_points": [9]}
        ),
        train_cutoff=orders["order_purchase_timestamp"].max(),
    )

    features = artifact.transform_with_snapshots(orders, snapshots)
    rng = np.random.default_rng(3)
    y = (rng.random(len(orders)) < 0.1).astype(int)
    model = LGBMClassifier(n_estimators=10, num_leaves=7, verbosity=-1).fit(features, y)

    raw_scores = model.predict_proba(features)[:, 1]
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(raw_scores, y)
    decided = artifact.with_decision(calibrator=calibrator, threshold=0.3)
    return DelayPredictor(decided, model, "lightgbm"), orders


class TestTheWrapperContract:
    def test_output_columns_are_fixed(self, wrapper_and_data):
        wrapper, orders = wrapper_and_data
        out = wrapper.predict(None, orders.loc[:, list(RAW_INPUT_COLUMNS)])
        assert list(out.columns) == list(OUTPUT_COLUMNS)
        assert len(out) == len(orders)

    def test_probabilities_are_in_range_and_the_decision_follows_the_threshold(
        self, wrapper_and_data
    ):
        wrapper, orders = wrapper_and_data
        out = wrapper.predict(None, orders.loc[:, list(RAW_INPUT_COLUMNS)])
        assert out["probability"].between(0.0, 1.0).all()
        expected = out["probability"] >= wrapper.artifact.threshold
        assert (out["is_late_predicted"] == expected).all()

    def test_the_threshold_is_returned_rather_than_assumed(self, wrapper_and_data):
        """So a caller can log which operating point produced a decision without looking it
        up somewhere else — the same reasoning that puts the threshold inside the artifact."""
        wrapper, orders = wrapper_and_data
        out = wrapper.predict(None, orders.loc[:, list(RAW_INPUT_COLUMNS)].head(3))
        assert (out["threshold"] == wrapper.artifact.threshold).all()

    def test_a_missing_required_column_is_refused(self, wrapper_and_data):
        wrapper, orders = wrapper_and_data
        with pytest.raises(ValueError, match="missing required input columns"):
            wrapper.predict(None, orders.drop(columns=["promised_days"]))

    def test_caller_supplied_history_is_refused(self, wrapper_and_data):
        """A client must not be able to override the leakage-safe snapshot join from outside.
        Accepting supplied history would make §4.5's boundary a suggestion."""
        wrapper, orders = wrapper_and_data
        poisoned = orders.loc[:, list(RAW_INPUT_COLUMNS)].copy()
        poisoned["seller_late_rate_hist"] = 0.99
        with pytest.raises(ValueError, match="history columns must not be supplied"):
            wrapper.predict(None, poisoned)

    def test_every_history_column_is_covered_by_the_guard(self):
        assert len(HISTORY_COLUMNS) == 15
        assert "seller_is_new" in HISTORY_COLUMNS
        assert "route_late_rate_hist" in HISTORY_COLUMNS

    def test_a_null_in_a_nullable_column_is_accepted(self, wrapper_and_data):
        """`dominant_category` is null for 1,330 real orders, so serving must tolerate it
        rather than reject the request."""
        wrapper, orders = wrapper_and_data
        frame = orders.loc[:, list(RAW_INPUT_COLUMNS)].head(5).copy()
        frame.loc[0, "dominant_category"] = None
        out = wrapper.predict(None, frame)
        assert out["probability"].notna().all()

    def test_wrapping_an_artifact_without_a_threshold_is_refused(self, wrapper_and_data):
        """Otherwise the wrapper would have to invent an operating point."""
        wrapper, _ = wrapper_and_data
        import dataclasses

        naked = dataclasses.replace(wrapper.artifact, threshold=None)
        with pytest.raises(ValueError, match="no threshold"):
            DelayPredictor(naked, wrapper.model, "lightgbm")

    def test_raw_input_columns_exclude_every_history_column(self):
        assert not (set(RAW_INPUT_COLUMNS) & HISTORY_COLUMNS)

    def test_the_input_example_includes_a_null_where_the_data_has_one(self, wrapper_and_data):
        """MLflow's own advice: a signature inferred from an accidentally complete sample
        declares types the real traffic will violate."""
        _, orders = wrapper_and_data
        example = raw_input_example(orders)
        assert len(example) == 5
        assert example["dominant_category"].isna().any()


class TestIntegerSchemaIsSafe:
    """Guards the assumption behind a suppressed MLflow warning.

    MLflow warns that integer columns in a signature cannot represent missing values, so a null
    arriving at inference time becomes a float and fails schema enforcement. That warning is
    filtered in `pyproject.toml`, and this test is the reason it is safe to filter: every
    integer column in the signature is `NOT NULL` in `features.orders_analytical`, so a null in
    one is a contract violation and rejecting it is the behaviour CLAUDE.md asks for.

    If a nullable integer feature is ever added, this test fails and the suppression has to be
    revisited — which is the point. A suppressed warning with no test behind it is just a
    hidden warning.
    """

    pytestmark: ClassVar = [pytest.mark.integration, pytest.mark.slow]

    def test_every_integer_signature_column_is_not_null_in_the_table(self, engine):
        from sqlalchemy import text

        with engine.connect() as conn:
            orders = pd.read_sql(text("select * from features.orders_analytical limit 500"), conn)
            nullability = dict(
                conn.execute(
                    text(
                        "select column_name, is_nullable from information_schema.columns "
                        "where table_schema='features' and table_name='orders_analytical'"
                    )
                ).all()
            )

        example = orders.loc[:, list(RAW_INPUT_COLUMNS)]
        integer_columns = [
            column for column in example.columns if pd.api.types.is_integer_dtype(example[column])
        ]
        assert integer_columns, "expected some integer columns in the signature"
        nullable = [c for c in integer_columns if nullability.get(c) != "NO"]
        assert nullable == [], (
            f"these integer signature columns are nullable: {nullable}. MLflow's integer "
            "warning now applies for real — either declare them float64 or drop the "
            "filterwarnings entry in pyproject.toml."
        )


class TestRoundTrip:
    """The phase's required test."""

    @pytest.fixture
    def logged(self, wrapper_and_data, tmp_path):
        """Log the wrapper to a throwaway SQLite-backed registry and return its URI."""
        import mlflow
        from mlflow.models import infer_signature

        wrapper, orders = wrapper_and_data
        mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
        mlflow.set_experiment("roundtrip")

        example = raw_input_example(orders)
        signature = infer_signature(example, wrapper.predict(None, example))
        with mlflow.start_run():
            info = mlflow.pyfunc.log_model(
                name="model",
                python_model=wrapper,
                signature=signature,
                input_example=example,
                registered_model_name="roundtrip_model",
            )
        return info, wrapper, orders

    def test_loaded_predictions_match_the_in_memory_model_to_1e_9(self, logged):
        import mlflow

        info, wrapper, orders = logged
        frame = orders.loc[:, list(RAW_INPUT_COLUMNS)].head(50)
        in_memory = wrapper.predict(None, frame)
        loaded = mlflow.pyfunc.load_model(info.model_uri).predict(frame)

        np.testing.assert_allclose(
            np.asarray(loaded["probability"], dtype=float),
            in_memory["probability"].to_numpy(),
            atol=1e-9,
            rtol=0,
        )
        assert (
            np.asarray(loaded["is_late_predicted"]).astype(bool)
            == in_memory["is_late_predicted"].to_numpy()
        ).all()

    def test_the_loaded_model_carries_the_same_threshold(self, logged):
        import mlflow

        info, wrapper, orders = logged
        loaded = mlflow.pyfunc.load_model(info.model_uri)
        out = loaded.predict(orders.loc[:, list(RAW_INPUT_COLUMNS)].head(3))
        assert float(out["threshold"].iloc[0]) == pytest.approx(wrapper.artifact.threshold)

    def test_the_signature_matches_the_raw_input_contract(self, logged):
        import mlflow

        info, _, _ = logged
        signature = mlflow.models.get_model_info(info.model_uri).signature
        assert signature is not None, "§13 names a missing signature as a common mistake"
        names = {field.name for field in signature.inputs.inputs}
        assert names == set(RAW_INPUT_COLUMNS)
        outputs = {field.name for field in signature.outputs.inputs}
        assert outputs == set(OUTPUT_COLUMNS)

    def test_the_registry_version_is_loadable_by_alias(self, logged):
        """Aliases, not stages: stages are removed in MLflow 3 (§18 A7)."""
        import mlflow

        _info, wrapper, orders = logged
        client = mlflow.MlflowClient()
        client.set_registered_model_alias("roundtrip_model", "champion", "1")
        loaded = mlflow.pyfunc.load_model("models:/roundtrip_model@champion")
        frame = orders.loc[:, list(RAW_INPUT_COLUMNS)].head(10)
        np.testing.assert_allclose(
            np.asarray(loaded.predict(frame)["probability"], dtype=float),
            wrapper.predict(None, frame)["probability"].to_numpy(),
            atol=1e-9,
        )

    def test_the_preprocessing_artifact_travels_with_the_model(self, logged):
        """§13's classic skew bug, asserted from the other direction: a model loaded in a
        clean process must already know its medians, category levels and snapshots without
        anything else being deployed alongside it."""
        import mlflow

        info, _wrapper, _orders = logged
        loaded = mlflow.pyfunc.load_model(info.model_uri)
        inner = loaded.unwrap_python_model()
        assert inner.artifact.numeric_medians
        assert inner.artifact.category_levels
        assert inner.artifact.latest_snapshots
        assert inner.artifact.calibrator is not None
        assert inner.artifact.threshold is not None

    def test_a_schema_violation_is_rejected_rather_than_coerced(self, logged):
        """The point of logging a signature: a client that omits a column gets an error, not a
        plausible-looking wrong answer."""
        import mlflow

        info, _, orders = logged
        loaded = mlflow.pyfunc.load_model(info.model_uri)
        with pytest.raises(Exception, match=r"(?i)schema|column|missing"):
            loaded.predict(orders.loc[:, list(RAW_INPUT_COLUMNS)].drop(columns=["n_items"]))


class TestAliasAssignment:
    """Who gets `@champion`, and who has to wait for the gate.

    This is the logic that decides whether a retrain silently replaces production, so it is
    tested against a throwaway registry rather than trusted.
    """

    @pytest.fixture
    def registry(self, tmp_path, monkeypatch):
        """A temp SQLite registry with one registered version and no aliases."""
        import mlflow

        from src.training import train as train_module

        uri = f"sqlite:///{tmp_path / 'reg.db'}"
        monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
        mlflow.set_tracking_uri(uri)
        client = mlflow.MlflowClient()
        client.create_registered_model(train_module.REGISTERED_MODEL_NAME)
        return client, train_module

    @staticmethod
    def _make_version(client, module, run_id: str = "abc"):
        return client.create_model_version(
            module.REGISTERED_MODEL_NAME, source=f"file:///tmp/{run_id}", run_id=None
        )

    @staticmethod
    def _stubs():
        from types import SimpleNamespace

        splits = SimpleNamespace(
            fit=SimpleNamespace(start="2017-05-01", end="2017-12-31"),
            calibrate=SimpleNamespace(start="2018-01-01", end="2018-02-28"),
        )
        blend = SimpleNamespace(best_single="xgboost", delta=-0.00156)
        return splits, blend

    def test_the_first_version_becomes_champion_uncontested(self, registry):
        """§8: v1 is auto-promoted because there is nothing to beat."""
        client, module = registry
        self._make_version(client, module)
        splits, blend = self._stubs()
        version, alias = module._assign_alias("v1", splits, blend, 0.2069, 0.2025)
        assert version == "1"
        assert alias == module.CHAMPION_ALIAS
        # str() on MLflow's own return: it gives an int from a SQLite store and a str over
        # HTTP. `_assign_alias` normalises its own return value for exactly this reason, but
        # the raw client objects still carry the backend's type.
        assert (
            str(client.get_model_version_by_alias(module.REGISTERED_MODEL_NAME, "champion").version)
            == "1"
        )

    def test_a_later_version_becomes_challenger_and_does_not_displace_the_champion(self, registry):
        """The property that matters: a retrain must not quietly replace production. Promotion
        is the Phase 9 gate's decision, not a training run's."""
        client, module = registry
        self._make_version(client, module)
        splits, blend = self._stubs()
        module._assign_alias("v1", splits, blend, 0.2069, 0.2025)

        self._make_version(client, module)
        version, alias = module._assign_alias("v2", splits, blend, 0.2200, 0.2100)
        assert version == "2"
        assert alias == module.CHALLENGER_ALIAS
        # The champion must still be v1.
        assert (
            str(client.get_model_version_by_alias(module.REGISTERED_MODEL_NAME, "champion").version)
            == "1"
        )

    def test_the_evaluation_tag_is_left_pending_for_the_gate(self, registry):
        """§7 asks for the holdout PR-AUC here, but the window is locked to the promotion gate
        (§18 A3, D25). A training run cannot compute it, so it says so rather than guessing."""
        client, module = registry
        self._make_version(client, module)
        splits, blend = self._stubs()
        module._assign_alias("v1", splits, blend, 0.2069, 0.2025)
        tags = client.get_model_version(module.REGISTERED_MODEL_NAME, "1").tags
        assert tags["evaluation_pr_auc"].startswith("pending")
        assert tags["calibration_window_pr_auc"] == "0.20250"
        assert tags["training_window"] == "2017-05-01..2017-12-31"
        assert tags["shipped_model"] == "xgboost"

    def test_describe_registry_reports_aliases(self, registry):
        """Guards a real bug: `search_model_versions` returns an empty `aliases` field on every
        version, so reading it there reports "no alias" for a model that is in fact the
        champion. The aliases must come from the registered model object."""
        client, module = registry
        self._make_version(client, module)
        splits, blend = self._stubs()
        module._assign_alias("v1", splits, blend, 0.2069, 0.2025)

        # Confirm the trap still exists in this MLflow version, so the test stays meaningful.
        searched = client.search_model_versions(f"name='{module.REGISTERED_MODEL_NAME}'")
        assert list(searched[0].aliases) == [], (
            "search_model_versions now returns aliases; the workaround in describe_registry "
            "can be simplified"
        )

        rows = module.describe_registry()
        assert rows[0]["aliases"] == ["champion"]
        assert rows[0]["shipped_model"] == "xgboost"


class TestTheRealRegistry:
    """Phase 6's completion criterion, against the live registry."""

    pytestmark: ClassVar = [pytest.mark.integration, pytest.mark.slow]

    def test_champion_is_set_and_loadable_in_a_fresh_python_process(self):
        """ "Loadable in a fresh Python process" taken literally: a subprocess that imports
        nothing of this project and resolves the model by alias alone.

        In-process loading can pass on state the interpreter happens to already hold. A
        subprocess cannot, which is what makes this the real test of a self-contained model.

        The reachability check is done *first*, separately, so that a genuine failure inside the
        subprocess is reported as a failure. An earlier version skipped on any non-zero exit and
        duly reported a typo in this very script as "registry not reachable".
        """
        import json
        import subprocess
        import sys
        import urllib.request

        uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
        try:
            with urllib.request.urlopen(f"{uri}/health", timeout=5) as response:
                if response.status != 200:
                    pytest.skip(f"tracking server at {uri} returned {response.status}")
        except OSError as exc:
            pytest.skip(f"tracking server at {uri} not reachable: {exc}")

        script = (
            "import mlflow, os, json\n"
            "mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])\n"
            "m = mlflow.pyfunc.load_model('models:/delivery_delay_classifier@champion')\n"
            "sig = m.metadata.signature\n"
            "print('RESULT ' + json.dumps({\n"
            "    'inputs': [f.name for f in sig.inputs.inputs],\n"
            "    'outputs': [f.name for f in sig.outputs.inputs],\n"
            "}))\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=300,
            env={**os.environ, "MLFLOW_TRACKING_URI": uri, "MLFLOW_DISABLE_AGENT_HINT": "1"},
        )
        assert (
            result.returncode == 0
        ), f"loading @champion in a fresh process failed:\n{result.stderr[-1500:]}"
        line = next(line for line in result.stdout.splitlines() if line.startswith("RESULT "))
        payload = json.loads(line.removeprefix("RESULT "))
        assert set(payload["inputs"]) == set(RAW_INPUT_COLUMNS)
        assert set(payload["outputs"]) == set(OUTPUT_COLUMNS)
