# CLAUDE.md: delivery-delay-platform

## What this is
Predicts at checkout whether a Brazilian e-commerce (Olist) order will be delivered after
its promised date. **The purpose of this project is the MLOps lifecycle, not the model:**
Postgres feature warehouse -> leakage-safe as-of aggregates -> LightGBM/XGBoost/CatBoost
-> MLflow tracking + registry -> promotion gate -> containerized FastAPI.

The full plan is `PLAN.md`; prompts reference its section numbers.
**`PLAN.md` §18 holds seven binding amendments from a design review. Where §18 and the body
of the plan disagree, §18 wins.** Read §18 before writing code in Phases 3, 7, 8 or 9.

Current status and next task live in `PROGRESS.md`. Decisions live in `DECISIONS.md`.

## Environment (already provisioned — do not re-install)
Verified 2026-09-28:

| Tool | Version | Status |
|---|---|---|
| Ubuntu | 24.04.4 LTS (VMware guest) | ready |
| Python | 3.12.3 (system) | ready |
| venv | `.venv/`, pip 26.2.1 | ready |
| Git | 2.43.0 (+ GitHub SSH key, verified) | ready |
| gh CLI | 2.45.0, authed as `Ankit-c-lang` | ready |
| Docker Engine | 29.8.0 (docker group, no sudo) | ready |
| Docker Compose | v5.5.1 (v2-style CLI) | ready |

- Repo path: `/home/ankit/projects/delay-prediction` — the VM's own ext4 disk, **not** a
  VMware shared folder.
- Remote: `github.com/Ankit-c-lang/delivery-delay-platform` (**private**), SSH, branch `main`.
- `id -u` is **1000** — bind mounts need no `user:` override.
- VM resources: 10 vCPUs, 7.7 GB RAM + 4 GB swap, 48 GB free of 77 GB.
  RAM is below the 8-10 GB PLAN §11 hoped for. Before Phase 4 Optuna tuning,
  `docker compose stop` the `realtime-fraud-detection` stack (see invariant 10).
- CPU only (i5-1235U, integrated GPU). No GPU code; keep tuning budgets to PLAN §6.4.
- **Dataset is already downloaded**: 9 Olist CSVs, 121 MB, in `data/raw/` (gitignored,
  CC BY-NC-SA). Kaggle credentials at `~/.kaggle/kaggle.json`.
- **Postgres and MLflow come from Docker Compose**, added in Phase 0. Do not apt-install
  either. `postgresql-client` on the host is optional —
  `docker compose exec postgres psql` works without it.

Never ask the user to install Ubuntu packages, Git, Docker, or to re-download the dataset —
they exist. Anything else (a new Python dependency, a new container, a new service) needs
explicit approval.

## Commands
`make help` lists what actually exists. Each phase adds its own targets; a target whose
module does not exist yet is not written (DECISIONS.md D15).

**Exist now (Phases 0-2):**
- `make lint | format | test | test-unit` — `test-unit` skips anything needing Postgres or
  the dataset, which is what CI will run from Phase 10
- `make up | down | stop | ps | logs | health | psql`
  `up` waits for both services to report `healthy`. `down` keeps volumes on purpose.
- `make check-data | load-raw | build-orders` and `make etl` for all three in order

**Arrive with their phase:**
- `make features` (3)
- `make train V=v1` (4) · `register` (6) · `promote` (9)
- `make serve | bench` (7-8)
- `make bootstrap` (8) — the ordered first-run sequence (PLAN §18 A5). **Required before
  `docker compose up` on a clean machine**, because the API resolves `@champion` at startup
  and the registry starts empty.

All Python entry points run through `.venv/bin/python -m src.x`.

Installed so far: `requirements-dev.txt` plus `sqlalchemy psycopg2-binary pydantic
pydantic-settings pyyaml pandas pandera` (pandas 3.0.6, pandera 0.33.1 — note the import is
`pandera.pandas`, not top-level `pandera`). The modelling stack installs in Phase 4;
`requirements-api.txt` is unproven until Phase 8. MLflow is reached over HTTP at
`$MLFLOW_TRACKING_URI`, never by opening `mlflow/mlflow.db` directly (D12).

## Invariants (never break these)
1. **PREDICTION-TIME CONTRACT:** only data available at `order_purchase_timestamp` may
   become a feature. Denylist: `order_approved_at`, `order_delivered_carrier_date`,
   `order_delivered_customer_date`, `order_status`, all review columns.
2. **As-of aggregates for snapshot month M use only orders whose DELIVERY OUTCOME was known
   before M** (`order_delivered_customer_date < M`). Filtering on
   `order_purchase_timestamp` leaks future outcomes, and the leak is biased toward late
   orders — they take ~3x longer to resolve (PLAN §18 A1).
3. The 2018-05 -> 2018-08 window is loaded ONLY by `src/evaluation/score_holdout.py`, called
   only by the promotion gate. **The parity test uses `tests/fixtures/`, never that window.**
   It is a PROMOTION EVALUATION set, not an untouched generalization estimate (§18 A3).
4. All preprocessing lives inside the MLflow pyfunc wrapper. The API layer contains none.
5. Time-based CV only. `StratifiedKFold` / `KFold` are forbidden.
6. Split dates come from `configs/splits.yaml`, which is keyed **per version**. Fitting and
   calibration windows must be disjoint for every version (§18 A2).
7. Tests are written in the same phase as the code they cover.
8. **Parity means: same raw record + same artifact -> same probability within 1e-6.**
   Snapshot selection is outside parity's scope; serving uses the latest bundled snapshot
   by design (§18 A6).
9. **pandas 3 copy-on-write:** no chained assignment, never mutate a slice.
10. Host ports come from `.env`. The API publishes on host **8001** — 8000 belongs to the
    `realtime-fraud-detection` project on this machine. **Never `docker compose down -v`;
    never `docker system prune -a` or `docker image prune -a`** — they would destroy that
    project's image and volume.
11. Row counts for the Olist CSVs must come from a CSV parser, never `wc -l`:
    `olist_order_reviews_dataset.csv` is 99,224 rows (not 104,719 — embedded newlines) and
    `product_category_name_translation.csv` is 71 (not 70 — no trailing newline).
12. **`order_delivered_customer_date` lives ONLY in `features.order_outcomes`** (with
    `order_id` and `is_late`). `features.orders_analytical` is feature-safe and its Pandera
    contract is `strict=True`, so a denylist column cannot enter it without being declared
    in `src/etl/schema.py`. Only the §4.5 snapshot builder may read `order_outcomes`
    (DECISIONS.md D19).

## Out of scope (do not add)
Airflow, Prefect, Dagster, Kafka, Kubernetes, Terraform, Spark, a feature store, a vector
DB, cloud deployment, microservices, a UI, real-time monitoring dashboards, A/B traffic
splitting, Streamlit. If a change isn't in `PLAN.md`, ask before writing it.

## Conventions
- Type hints on public functions; Google-style docstrings
- ruff + black, line length 100
- Config via pydantic-settings, never `os.environ` scattered through modules
- Log with `logging`, never `print`
- Fail loudly: raise on contract violations, don't warn and continue
- Stack: Python 3.12 · PostgreSQL 16 (Compose) · pandas 3 · LightGBM/XGBoost/CatBoost ·
  Optuna · SHAP · MLflow 3 (SQLite backend) · FastAPI · Docker Compose ·
  GitHub Actions -> GHCR

## Working agreement
- **One phase per session.** Do not implement future phases.
- Ask for tests in the same step as the implementation, never retrofitted.
- When proposing an addition not in `PLAN.md`, default to no and ask first.
- **Update `PROGRESS.md` as you go** — at the end of every completed task, in the same
  commit as that task. It tracks *where we are*; `PLAN.md` remains the source of truth for
  *what* to build.
- **Update `DECISIONS.md` whenever a non-obvious decision is made** — one entry with the
  date, the decision, the reason, and what was rejected. Amendments to `PLAN.md` go in §18
  and are cross-referenced from here, not duplicated.
- **Commits: never add Claude/AI attribution, and never a `Co-Authored-By` line.**
  Write commit messages in plain imperative mood with no tool credit of any kind.
- Commit or push only when asked.

## Current phase
**Phases 0-2 COMPLETE.** `raw` holds 9 tables / 1,550,922 rows;
`features.orders_analytical` holds 96,203 rows with an `is_late` rate of 6.79%; 80 tests pass.
**Next: Phase 3 — As-of aggregates and feature assembly, the hardest phase. Read `PLAN.md`
§18 A1 before writing any of it.** Do not implement future phases. `PROGRESS.md` carries the
detail, including what is *not* yet verified and the D20 base-rate finding that Phases 5 and 9
must account for.
