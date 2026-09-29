"""Registry-side code: the pyfunc wrapper and the promotion gate.

:func:`tracking_uri` lives here because three call sites needed it — ``train.py``, ``promote.py``
and ``scripts/holdout_reference.py`` — and a constant copied three times is the shape of bug this
project has hit repeatedly (CLAUDE.md invariant 17). The third copy was written during Phase 11 and
deleted in the same hour.
"""

from __future__ import annotations

import os

#: Where the tracking server listens when nothing says otherwise. Matches ``.env.example`` and the
#: host side of ``docker-compose.yml``; inside Compose the API is given ``http://mlflow:5000``
#: instead, because a container's loopback is its own (DECISIONS.md D40).
DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"


def tracking_uri() -> str:
    """The MLflow tracking URI from the environment.

    Returns:
        ``$MLFLOW_TRACKING_URI``, or :data:`DEFAULT_TRACKING_URI`.

    Clients always speak HTTP to the tracking server; only the server itself touches the SQLite
    backend store (DECISIONS.md D12). A client that opened ``mlflow.db`` directly would corrupt it
    the moment two processes did so at once.
    """
    return os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
