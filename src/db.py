"""Database settings and SQLAlchemy engine construction.

This is the only module that assembles a Postgres connection URL. Everything that needs
a connection imports :func:`get_engine`; nothing reads ``os.environ`` for credentials of
its own (PLAN.md §12, CLAUDE.md conventions).

Settings come from the environment, falling back to ``.env``. Credentials have no
defaults on purpose: a missing ``.env`` raises at import of the settings object rather
than silently connecting somewhere unintended.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import Engine, create_engine
from sqlalchemy.engine import URL

logger = logging.getLogger(__name__)


class DatabaseSettings(BaseSettings):
    """Postgres connection settings.

    Attributes:
        postgres_user: Role to connect as. Required.
        postgres_password: Password for that role. Required.
        postgres_db: Database name. Required.
        postgres_host: ``127.0.0.1`` from the host, ``postgres`` from inside Compose.
        postgres_port: Port to connect to. The container always listens on 5432
            internally; this is the published host port.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    postgres_user: str
    postgres_password: SecretStr
    postgres_db: str
    postgres_host: str = "127.0.0.1"
    postgres_port: int = 5432

    @property
    def url(self) -> URL:
        """The connection URL.

        Built with :meth:`sqlalchemy.engine.URL.create` rather than string formatting so
        a password containing ``@``, ``/`` or ``:`` cannot corrupt the URL.
        """
        return URL.create(
            drivername="postgresql+psycopg2",
            username=self.postgres_user,
            password=self.postgres_password.get_secret_value(),
            host=self.postgres_host,
            port=self.postgres_port,
            database=self.postgres_db,
        )

    @property
    def safe_url(self) -> str:
        """The connection URL with the password masked, for logs."""
        return self.url.render_as_string(hide_password=True)


@lru_cache(maxsize=1)
def get_settings() -> DatabaseSettings:
    """Return the process-wide database settings, read once."""
    return DatabaseSettings()


def build_engine(
    settings: DatabaseSettings | None = None,
    *,
    echo: bool = False,
    pool_size: int = 5,
) -> Engine:
    """Create a new SQLAlchemy engine.

    Args:
        settings: Settings to use. Defaults to :func:`get_settings`.
        echo: Log every statement. Useful when debugging ETL, noisy otherwise.
        pool_size: Connections kept open in the pool.

    Returns:
        A configured :class:`sqlalchemy.Engine`. No connection is opened yet —
        SQLAlchemy connects lazily on first use.
    """
    settings = settings or get_settings()
    logger.info("Creating engine for %s", settings.safe_url)
    return create_engine(
        settings.url,
        echo=echo,
        pool_size=pool_size,
        # Postgres in a container can vanish between runs; without pre-ping the first
        # query after a restart fails on a stale pooled connection.
        pool_pre_ping=True,
        future=True,
    )


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Return the shared engine, creating it on first call."""
    return build_engine()
