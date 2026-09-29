# Phase 0 targets only. ETL, training, registry, serving and `bootstrap` targets are
# added by the phases that create the code behind them (PLAN.md §13) — a target that
# shells into a module that does not exist yet is worse than no target.
#
# Every Python entry point runs through the venv explicitly; no `source activate`.

PY := .venv/bin/python
COMPOSE := docker compose -f docker-compose.yml

.DEFAULT_GOAL := help
.PHONY: help lint format test test-unit test-ci-etl up down stop ps logs psql health check-data load-raw serve bench bootstrap build api-up api-down api-logs promote holdout-reference \
        build-orders etl snapshots features baselines tune decide train registry

help: ## List available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "} {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

lint: ## ruff check + black --check (what CI runs)
	$(PY) -m ruff check .
	$(PY) -m black --check .

format: ## Apply ruff --fix and black
	$(PY) -m ruff check --fix .
	$(PY) -m black .

test: ## pytest with coverage (needs `make up` and the real dataset)
	$(PY) -m pytest

test-unit: ## Only tests needing neither Postgres nor the dataset (CI's first pytest step)
	$(PY) -m pytest -m "not integration and not slow and not ci_etl"

test-ci-etl: ## The real ETL on synthetic CSVs in a throwaway database (CI's second pytest step)
	$(PY) -m pytest -m ci_etl

up: ## Start postgres + mlflow, wait for both to report healthy
	$(COMPOSE) up -d --wait postgres mlflow
	@$(MAKE) --no-print-directory health

down: ## Stop and remove this project's containers. Keeps volumes on purpose.
	$(COMPOSE) down

stop: ## Stop the containers without removing them
	$(COMPOSE) stop

ps: ## Show container status and health
	$(COMPOSE) ps

logs: ## Tail logs from both services
	$(COMPOSE) logs -f --tail=50

psql: ## Open a psql shell inside the postgres container (no host client needed)
	$(COMPOSE) exec postgres psql -U $${POSTGRES_USER:-delay} -d $${POSTGRES_DB:-delay}

health: ## Print the health state of both services
	@$(COMPOSE) ps --format '{{.Service}}\t{{.State}}\t{{.Status}}'

# --- Phase 1 -------------------------------------------------------------------

check-data: ## Verify the 9 Olist CSVs are present and their headers match the loader
	$(PY) -m scripts.download_data

load-raw: ## Load the 9 CSVs into the postgres `raw` schema. Idempotent: truncates first.
	$(PY) -m src.etl.load_raw

# --- Phase 2 -------------------------------------------------------------------

build-orders: ## Build features.orders_analytical + order_outcomes from the raw schema
	$(PY) -m src.etl.build_orders

etl: check-data load-raw build-orders ## The full ETL chain, in order

# --- Phase 3 -------------------------------------------------------------------

snapshots: ## Build the as-of monthly aggregate snapshots (PLAN.md 4.5, 18 A1)
	$(PY) -m src.features.history

features: snapshots ## Fit the preprocessing artifact and build the 38-feature matrix
	$(PY) -m src.features.build

# --- Phase 4 -------------------------------------------------------------------

baselines: ## Baselines only, no tuning (fast: no Optuna)
	$(PY) -m src.training.tune --models

tune: ## Baselines + Optuna for all 3 GBDTs. Budget is 55 min worst case (PLAN.md 6.4).
	$(PY) -m src.training.tune

# --- Phase 5 -------------------------------------------------------------------

decide: ## Blend, calibrate, sweep the threshold, SHAP, and write the ship decision
	$(PY) -m src.training.ensemble

# --- Phase 6 -------------------------------------------------------------------

train: ## Full versioned MLflow run: nested children, pyfunc, registry. V=v1 by default.
	$(PY) -m src.training.train --version $(or $(V),v1)

registry: ## Show the registered versions and their aliases
	$(PY) -m src.training.train --show-registry

# --- Phase 7 -------------------------------------------------------------------

serve: ## Run the API on API_PORT (8001 — 8000 belongs to realtime-fraud-detection)
	@API_PORT=$${API_PORT:-8001}; \
	echo "serving on http://127.0.0.1:$$API_PORT (docs at /docs)"; \
	$(PY) -m uvicorn api.main:app --host 127.0.0.1 --port $$API_PORT

bench: ## p50/p95 for single and batch-100. URL=http://127.0.0.1:8001 to hit a running service.
	$(PY) -m scripts.benchmark_latency $(if $(URL),--url $(URL),) $(if $(N),--iterations $(N),)

# --- Phase 8 -------------------------------------------------------------------

build: ## Build the API image (serving subset only — see requirements-api.txt)
	docker compose build api

api-up: ## Start the containerized API and wait for it to report healthy
	docker compose up -d api
	@echo "waiting for delay-api to report healthy..."
	@until [ "$$(docker inspect -f '{{.State.Health.Status}}' delay-api 2>/dev/null)" = "healthy" ]; do \
		status=$$(docker inspect -f '{{.State.Health.Status}}' delay-api 2>/dev/null || echo missing); \
		if [ "$$status" = "unhealthy" ]; then \
			echo "delay-api is unhealthy. Did you run 'make bootstrap'? Logs:"; \
			docker compose logs --tail 30 api; exit 1; \
		fi; \
		sleep 2; \
	done
	@echo "healthy: http://127.0.0.1:$${API_PORT:-8001}/docs"

api-down: ## Stop the API container only (leaves postgres and mlflow running)
	docker compose stop api

api-logs: ## Tail the API container's logs
	docker compose logs -f --tail 50 api

bootstrap: ## Ordered first run on a clean machine (PLAN.md 18 A5). REQUIRED before compose up.
	@echo "== 1/6 infrastructure =========================================================="
	$(MAKE) up
	@echo "== 2/6 raw load ================================================================"
	$(MAKE) load-raw
	@echo "== 3/6 analytical table ========================================================"
	$(MAKE) build-orders
	@echo "== 4/6 as-of snapshots and the preprocessing artifact =========================="
	$(MAKE) features
	@echo "== 5/6 decision: blend, calibrate, threshold ==================================="
	$(MAKE) decide
	@echo "== 6/6 train and register v1; the first version becomes @champion =============="
	$(MAKE) train V=v1
	@echo
	@echo "Bootstrap complete. The registry now has a champion, so the API can resolve it:"
	@echo "    make build && make api-up"

# --- Phase 9 -------------------------------------------------------------------

holdout-reference: ## Score no-skill and logistic references on the evaluation window (D45)
	$(PY) -m scripts.holdout_reference

promote: ## Run the promotion gate: score @champion and @challenger on the holdout and decide
	$(PY) -m src.registry.promote $(if $(DRY),--dry-run,)
