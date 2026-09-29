"""API settings (PLAN.md §9).

Configuration comes from the environment through pydantic-settings, never from ``os.environ``
scattered through modules (CLAUDE.md conventions). One object, constructed once, injected into the
app factory — which is also what lets a test point the app at a local fixture model without
touching the process environment.
"""

from __future__ import annotations

from enum import StrEnum
from functools import cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

#: The registered model the API serves. Matches ``configs/base.yaml``.
REGISTERED_MODEL_NAME = "delivery_delay_classifier"

#: §9 caps a batch at roughly 1000 records. A cap is a promise about tail latency: without it one
#: request can hold a worker for minutes, and a health check then fails for reasons unrelated to
#: health.
MAX_BATCH_RECORDS = 1000


class ModelUriKind(StrEnum):
    """How ``MODEL_URI`` was resolved, reported by ``/model/info``.

    §9 asks for three forms, and the third is not a hack: being able to pin a version *is* the
    rollback story, and being able to point at a local path is what lets CI serve a model with no
    registry and no tracking server.
    """

    ALIAS = "registry_alias"
    PINNED_VERSION = "registry_pinned_version"
    LOCAL_PATH = "local_path"


class Settings(BaseSettings):
    """Everything the service reads from its environment.

    Attributes:
        model_uri: What to load. One of ``models:/<name>@<alias>``, ``models:/<name>/<version>``
            or a filesystem path.
        mlflow_tracking_uri: Where the registry lives. Ignored for a local-path ``model_uri``,
            which is exactly why CI can run without a server.
        log_level: Root log level for the service.
        max_batch_records: Hard cap on ``/predict/batch``.
    """

    # pydantic v2 reserves the `model_` prefix for its own attributes and warns on any field using
    # it. The env var is MODEL_URI in §9 and in .env, so the field keeps that name and the
    # protection is switched off here rather than inventing a second name for one thing.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    model_uri: str = Field(default=f"models:/{REGISTERED_MODEL_NAME}@champion")
    mlflow_tracking_uri: str = Field(default="http://127.0.0.1:5000")
    log_level: str = Field(default="INFO")
    max_batch_records: int = Field(default=MAX_BATCH_RECORDS, ge=1)

    @field_validator("log_level")
    @classmethod
    def _upper(cls, value: str) -> str:
        return value.upper()

    @property
    def model_uri_kind(self) -> ModelUriKind:
        """Classify the URI. Reading a string is not preprocessing; it stays in the API layer."""
        if not self.model_uri.startswith("models:/"):
            return ModelUriKind.LOCAL_PATH
        remainder = self.model_uri.removeprefix("models:/")
        return ModelUriKind.ALIAS if "@" in remainder else ModelUriKind.PINNED_VERSION

    @property
    def model_alias(self) -> str | None:
        """The alias being served, when one is."""
        if self.model_uri_kind is not ModelUriKind.ALIAS:
            return None
        return self.model_uri.rsplit("@", 1)[1]

    @property
    def model_version_pin(self) -> str | None:
        """The pinned version, when one is. This is the rollback path §9 asks for."""
        if self.model_uri_kind is not ModelUriKind.PINNED_VERSION:
            return None
        return self.model_uri.rsplit("/", 1)[1]

    @property
    def needs_tracking_server(self) -> bool:
        """False for a local path, which is what makes the CI path serverless."""
        return self.model_uri_kind is not ModelUriKind.LOCAL_PATH

    def describe_source(self) -> dict[str, str | None]:
        """A flat description of where the model came from, for ``/model/info``."""
        return {
            "uri": self.model_uri,
            "kind": self.model_uri_kind.value,
            "alias": self.model_alias,
            "pinned_version": self.model_version_pin,
            "tracking_uri": self.mlflow_tracking_uri if self.needs_tracking_server else None,
        }


@cache
def get_settings(_model_uri: str | None = None) -> Settings:
    """Cached settings.

    Args:
        _model_uri: Override for the model URI. Present so a test can serve a locally saved
            fixture model without mutating the process environment — MLflow's own
            ``set_tracking_uri`` writes into ``os.environ`` and leaks across tests, which cost
            real debugging time in Phase 6 (DECISIONS.md D32).

    Returns:
        The settings object. Call ``get_settings.cache_clear()`` between tests that override.
    """
    if _model_uri is not None:
        return Settings(model_uri=_model_uri)
    return Settings()


def resolve_local_path(uri: str) -> Path:
    """The filesystem path a local ``model_uri`` refers to.

    Raises:
        FileNotFoundError: If it does not exist, with the resolved path in the message. A typo in
            ``MODEL_URI`` should fail at startup naming the path it tried, not later with an
            MLflow stack trace.
    """
    path = Path(uri).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"MODEL_URI points at {path.resolve()}, which does not exist")
    return path
