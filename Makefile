# Phase 0 targets only. ETL, training, registry, serving and `bootstrap` targets are
# added by the phases that create the code behind them (PLAN.md §13) — a target that
# shells into a module that does not exist yet is worse than no target.
#
# Every Python entry point runs through the venv explicitly; no `source activate`.

PY := .venv/bin/python
COMPOSE := docker compose -f docker-compose.yml

.DEFAULT_GOAL := help
.PHONY: help lint format test test-unit up down stop ps logs psql health check-data load-raw \
        build-orders etl snapshots features baselines tune

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

test-unit: ## Only tests that need neither Postgres nor the dataset (what CI runs)
	$(PY) -m pytest -m "not integration and not slow"

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

features: snapshots ## Fit the preprocessing artifact and build the 39-feature matrix
	$(PY) -m src.features.build

# --- Phase 4 -------------------------------------------------------------------

baselines: ## Baselines only, no tuning (fast: no Optuna)
	$(PY) -m src.training.tune --models

tune: ## Baselines + Optuna for all 3 GBDTs. Budget is 55 min worst case (PLAN.md 6.4).
	$(PY) -m src.training.tune
