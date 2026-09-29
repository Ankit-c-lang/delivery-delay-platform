# Serving image for the delivery-delay API (PLAN.md §10.1).
#
# The model is NOT baked in. The image is generic; the model arrives from the registry at runtime,
# so rolling out a new model is a config change rather than a rebuild. That separation is the point.
#
# Only the serving subset is installed (requirements-api.txt). The training stack — optuna, shap,
# matplotlib, lightgbm, xgboost — and, since DECISIONS.md D38, pandera, sqlalchemy and psycopg2 are
# all absent, because the import closure was measured rather than assumed.

# ---------- builder -------------------------------------------------------------------------
FROM python:3.12-slim AS builder

# Wheels only where possible; build-essential is present because some serving wheels still
# compile on slim. It stays in this stage and never reaches the runtime image.
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

# --prefix, not --user: a fixed path is copyable into the next stage regardless of which user
# runs there, whereas /root/.local would have to be chowned or run as root.
COPY requirements-api.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements-api.txt

# ---------- runtime -------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

# Unbuffered so container logs appear in `docker logs` immediately rather than on flush;
# no .pyc, because the layer is read-only in practice and they would only add size.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

COPY --from=builder /install /usr/local

# 1000:1000 matches the host user, so a bind mount added later needs no ownership fixing.
RUN useradd --create-home --uid 1000 --shell /usr/sbin/nologin appuser
WORKDIR /app

# api/ is the service; src/ is what the pyfunc unpickles into. Nothing else: no tests, no configs,
# no data. `src/etl` and `src/training` are copied because src/ is a package, but nothing imports
# them at serving time and their dependencies are not installed — an accidental import fails loudly.
COPY --chown=1000:1000 api/ ./api/
COPY --chown=1000:1000 src/ ./src/

USER appuser
EXPOSE 8000

# start-period covers model resolution: the API loads the champion from the registry at startup,
# and a healthcheck that fires before that finishes would mark a healthy container unhealthy.
HEALTHCHECK --interval=15s --timeout=5s --start-period=45s --retries=3 \
  CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status == 200 else 1)"]

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
