# Phase 0 targets only. ETL, training, registry, serving and `bootstrap` targets are
# added by the phases that create the code behind them (PLAN.md §13) — a target that
# shells into a module that does not exist yet is worse than no target.
#
# Every Python entry point runs through the venv explicitly; no `source activate`.

PY := .venv/bin/python
COMPOSE := docker compose -f docker-compose.yml

.DEFAULT_GOAL := help
.PHONY: help lint format test up down stop ps logs psql health

help: ## List available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "} {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

lint: ## ruff check + black --check (what CI runs)
	$(PY) -m ruff check .
	$(PY) -m black --check .

format: ## Apply ruff --fix and black
	$(PY) -m ruff check --fix .
	$(PY) -m black .

test: ## pytest with coverage
	$(PY) -m pytest

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
