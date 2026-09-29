"""Container and Compose checks (PLAN.md §10.1, §10.2, §13 Phase 8).

Two kinds of test, split deliberately.

**Static checks** parse the Dockerfile, `.dockerignore`, `docker-compose.yml` and
`requirements-api.txt`. They need no Docker daemon, so they run in CI and they are what catches the
§13 common mistakes — a missing `.dockerignore`, a bare `depends_on`, the training stack installed
into the serving image, a container running as root.

**Live checks** need a built image and a running stack. They are skipped **only** when Docker is
genuinely unavailable, and that condition is tested *first and separately* — the Phase 6 lesson
(DECISIONS.md D32): a broad `except -> skip` turns every bug in a test into a green run. Once Docker
is reachable, a failure here is reported as a failure.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "Dockerfile"
DOCKERIGNORE = ROOT / ".dockerignore"
COMPOSE = ROOT / "docker-compose.yml"
API_REQUIREMENTS = ROOT / "requirements-api.txt"

#: §10.1 names these explicitly: a fat context defeats layer caching.
MUST_IGNORE = (
    "data/",
    "mlartifacts/",
    "mlflow.db",
    ".git/",
    "notebooks/",
    "__pycache__/",
    ".venv/",
)

#: One valid request, shared by the live tests so they cannot drift apart. Every value is plausible
#: rather than minimal: a request built from zeros would exercise none of the imputation or
#: category-capping paths.
_REFERENCE_RECORD: dict = {
    "order_purchase_timestamp": "2018-01-15T10:00:00",
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

#: The training stack. None of it may appear in the serving requirements (§13 Phase 8).
TRAINING_ONLY = ("optuna", "shap", "matplotlib", "pandera", "sqlalchemy", "psycopg2", "xgboost")


def _docker_available() -> tuple[bool, str]:
    """Whether a Docker daemon is reachable, and why not if it is not.

    Checked on its own so that an unavailable daemon is the *only* thing that can produce a skip.
    """
    if shutil.which("docker") is None:
        return False, "the docker CLI is not on PATH"
    try:
        probe = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"docker info did not complete: {type(exc).__name__}"
    if probe.returncode != 0:
        return False, f"docker info failed: {probe.stderr.strip()[:120]}"
    return True, probe.stdout.strip()


@pytest.fixture(scope="module")
def docker_ready() -> str:
    available, detail = _docker_available()
    if not available:
        pytest.skip(f"no Docker daemon: {detail}")
    return detail


class TestTheDockerfile:
    @staticmethod
    def _text() -> str:
        return DOCKERFILE.read_text(encoding="utf-8")

    def test_it_is_multi_stage_on_the_pinned_interpreter(self):
        text = self._text()
        assert text.count("FROM python:3.12-slim") == 2, "§10.1 asks for a builder and a runtime"
        assert "AS builder" in text
        assert "AS runtime" in text

    def test_the_runtime_stage_does_not_install_the_training_stack(self):
        """§13's named mistake. The builder installs requirements-api.txt and nothing else."""
        text = self._text()
        assert "requirements-api.txt" in text
        assert "requirements.txt" not in text.replace("requirements-api.txt", "")

    def test_it_runs_as_a_non_root_user(self):
        text = self._text()
        assert "useradd" in text
        assert "USER appuser" in text
        assert text.index("USER appuser") < text.index("CMD"), "the switch must precede CMD"

    def test_it_has_a_healthcheck_on_slash_health(self):
        text = self._text()
        assert "HEALTHCHECK" in text
        assert "/health" in text
        assert "start-period" in text or "start_period" in text, (
            "the API resolves @champion at startup, so a healthcheck without a start period "
            "marks a healthy container unhealthy"
        )

    def test_no_model_artifact_is_baked_into_the_image(self):
        """§10.1: the image is generic and the model comes from the registry at runtime."""
        text = self._text()
        for path in ("artifacts/", "mlartifacts", "decision.joblib", "preprocessing.joblib"):
            assert path not in text, f"{path} must not be copied into the image"

    def test_it_copies_only_the_service_and_the_modules_the_pyfunc_needs(self):
        text = self._text()
        assert "COPY --chown=1000:1000 api/ ./api/" in text
        assert "COPY --chown=1000:1000 src/ ./src/" in text
        for excluded in ("COPY tests", "COPY notebooks", "COPY data"):
            assert excluded not in text


class TestTheDockerignore:
    def test_every_required_path_is_excluded(self):
        lines = {
            line.strip()
            for line in DOCKERIGNORE.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        }
        for required in MUST_IGNORE:
            assert any(required.rstrip("/") in line for line in lines), f"{required} not ignored"

    def test_the_env_file_is_excluded(self):
        """Secrets must not enter a build context; Compose injects them at runtime instead."""
        text = DOCKERIGNORE.read_text(encoding="utf-8")
        assert ".env" in text


class TestTheServingRequirements:
    def test_the_training_stack_is_absent(self):
        """D38: measured, not assumed. pandera, sqlalchemy and psycopg2 are in this list."""
        listed = {
            line.strip().split("[")[0].lower()
            for line in API_REQUIREMENTS.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        }
        for package in TRAINING_ONLY:
            assert package not in listed, f"{package} is training-only and must not be installed"

    def test_it_uses_mlflow_skinny(self):
        listed = [
            line.strip()
            for line in API_REQUIREMENTS.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]
        assert "mlflow-skinny" in listed
        assert "mlflow" not in listed, "the full mlflow brings a server and a UI the API never runs"

    def test_exactly_one_boosting_library_is_installed(self):
        """Whatever the pyfunc pickles must be installed, and nothing else should be.

        This file deliberately tracks the champion, so it must change when the shipped model does —
        that coupling is the point (DECISIONS.md D41's operational note). LightGBM ships because the
        equivalence band chose it; catboost's 269 MB and xgboost's 85 MB are dead weight in a
        serving image that will never load them.
        """
        listed = {
            line.strip().lower()
            for line in API_REQUIREMENTS.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        }
        boosting = listed & {"lightgbm", "xgboost", "catboost"}
        assert boosting == {"lightgbm"}, (
            f"expected exactly lightgbm, found {sorted(boosting)}. Without the shipped model's "
            "library the artifact raises on unpickle; with the others, the image carries megabytes "
            "it never loads."
        )


class TestComposeConfiguration:
    @staticmethod
    def _config(docker_ready: str) -> dict:
        """The *resolved* config, so defaults and interpolation are tested, not just the text."""
        del docker_ready
        result = subprocess.run(
            ["docker", "compose", "config", "--format", "json"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert result.returncode == 0, f"docker compose config failed:\n{result.stderr[-800:]}"
        return json.loads(result.stdout)

    def test_the_api_waits_for_a_healthy_dependency_not_a_started_one(self, docker_ready):
        """§13's named mistake is a bare depends_on. The race against MLflow is the real one:
        the API resolves @champion at startup, so it needs a server that answers."""
        api = self._config(docker_ready)["services"]["api"]
        assert api["depends_on"]["mlflow"]["condition"] == "service_healthy"

    def test_the_api_does_not_claim_a_postgres_dependency_it_does_not_have(self, docker_ready):
        """DECISIONS.md D39: psycopg2 is not even installed in the image."""
        api = self._config(docker_ready)["services"]["api"]
        assert "postgres" not in api.get("depends_on", {})

    def test_the_api_publishes_on_host_8001(self, docker_ready):
        """8000 belongs to realtime-fraud-detection on this machine (invariant 10)."""
        api = self._config(docker_ready)["services"]["api"]
        published = {(p.get("published"), p.get("target")) for p in api["ports"]}
        assert ("8001", 8000) in published or (8001, 8000) in published
        assert all(p.get("host_ip") == "127.0.0.1" for p in api["ports"])

    def test_the_container_reaches_mlflow_by_service_name_not_loopback(self, docker_ready):
        """Inside the network, 127.0.0.1 is the container itself — the classic Compose bug."""
        api = self._config(docker_ready)["services"]["api"]
        assert api["environment"]["MLFLOW_TRACKING_URI"] == "http://mlflow:5000"

    def test_the_trainer_is_behind_a_profile_so_it_never_starts_by_default(self, docker_ready):
        """Asserted as *behaviour*, not as a declaration.

        `docker compose config` resolves only the services that would actually run, so a
        profile-gated service is **absent** from the default output. That absence is the property
        worth testing: reading back `profiles: ["training"]` from the file would only prove the
        text was written, not that Compose honours it.
        """
        assert "trainer" not in self._config(docker_ready)["services"]

        with_profile = subprocess.run(
            ["docker", "compose", "--profile", "training", "config", "--format", "json"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert with_profile.returncode == 0, with_profile.stderr[-800:]
        trainer = json.loads(with_profile.stdout)["services"]["trainer"]
        assert trainer["profiles"] == ["training"]
        # It genuinely needs the database, unlike the API (DECISIONS.md D39).
        assert trainer["depends_on"]["postgres"]["condition"] == "service_healthy"

    def test_the_named_volume_is_project_scoped(self, docker_ready):
        """An ambiguous volume name is how one project's teardown deletes another's data."""
        config = self._config(docker_ready)
        assert config["volumes"]["pgdata"]["name"] == "delay-prediction-pgdata"

    def test_every_variable_compose_reads_is_documented_in_env_example(self, docker_ready):
        """A variable Compose interpolates but .env.example omits is a broken clean checkout."""
        del docker_ready
        import re

        compose_text = COMPOSE.read_text(encoding="utf-8")
        referenced = set(re.findall(r"\$\{([A-Z_][A-Z0-9_]*)", compose_text))
        documented = {
            line.split("=", 1)[0].strip()
            for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
            if "=" in line and not line.strip().startswith("#")
        }
        assert not (referenced - documented), f"undocumented: {sorted(referenced - documented)}"


class TestTheLiveContainer:
    """§13's completion criterion: healthy stack, working prediction."""

    @staticmethod
    def _inspect(field: str) -> str:
        result = subprocess.run(
            ["docker", "inspect", "-f", field, "delay-api"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else ""

    def test_the_container_reports_healthy(self, docker_ready):
        del docker_ready
        status = self._inspect("{{.State.Health.Status}}")
        if not status:
            pytest.skip("delay-api is not running; `make build && make api-up` first")
        assert status == "healthy", f"delay-api health is {status!r}"

    def test_the_container_runs_as_a_non_root_user(self, docker_ready):
        del docker_ready
        if not self._inspect("{{.State.Health.Status}}"):
            pytest.skip("delay-api is not running; `make build && make api-up` first")
        result = subprocess.run(
            ["docker", "exec", "delay-api", "id", "-u"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "1000", "§13: running as root is a named mistake"

    def test_predict_works_against_the_containerized_service(self, docker_ready):
        """The completion criterion. Uses the real champion through the published port."""
        del docker_ready
        import urllib.error
        import urllib.request

        if not self._inspect("{{.State.Health.Status}}"):
            pytest.skip("delay-api is not running; `make build && make api-up` first")

        body = json.dumps(_REFERENCE_RECORD).encode()
        request = urllib.request.Request(
            "http://127.0.0.1:8001/predict",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            assert response.status == 200
            payload = json.loads(response.read())

        assert 0.0 <= payload["probability"] <= 1.0
        assert payload["is_late_predicted"] == (payload["probability"] >= payload["threshold"])
        assert payload["model_version"], "the response must name the version that served it"

    def test_the_container_and_the_venv_agree_exactly(self, docker_ready):
        """Cross-environment parity, which is stronger than `tests/test_parity.py`.

        That test compares two code paths inside **one** interpreter. This compares two entirely
        different environments: the 110-package development venv against the 65-package serving
        image, where `mlflow` was swapped for `mlflow-skinny` and pandera, sqlalchemy and psycopg2
        were removed (DECISIONS.md D38). Pruning dependencies is exactly the kind of change that can
        move numerics without raising — a different BLAS, a different pyarrow, a subtly different
        scipy — and nothing else in the suite would notice.

        Asserted as **bitwise equality**, not a tolerance: the same artifact and the same model on
        the same input have no licence to differ at all. Measured at 0.0 difference.
        """
        del docker_ready
        import json as _json
        import urllib.request

        import pandas as pd

        if not self._inspect("{{.State.Health.Status}}"):
            pytest.skip("delay-api is not running; `make build && make api-up` first")

        from api.config import Settings
        from api.model_loader import load_model

        record = _REFERENCE_RECORD
        frame = pd.DataFrame(
            [
                {
                    **record,
                    "order_purchase_timestamp": pd.Timestamp(record["order_purchase_timestamp"]),
                }
            ]
        )
        venv_probability = float(load_model(Settings()).score(frame)[0]["probability"].iloc[0])

        request = urllib.request.Request(
            "http://127.0.0.1:8001/predict",
            data=_json.dumps(record).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            container_probability = _json.loads(response.read())["probability"]

        assert container_probability == venv_probability, (
            f"the image and the venv disagree: {container_probability!r} vs {venv_probability!r}. "
            "A pruned dependency changed the numerics."
        )
