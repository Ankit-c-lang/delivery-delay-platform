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

**Exist now (Phases 0-6):**
- `make lint | format | test | test-unit` — `test-unit` skips anything needing Postgres or
  the dataset, which is what CI will run from Phase 10
- `make up | down | stop | ps | logs | health | psql`
  `up` waits for both services to report `healthy`. `down` keeps volumes on purpose.
- `make check-data | load-raw | build-orders` and `make etl` for all three in order
- `make snapshots` (as-of snapshots) · `make features` (fit the artifact, build the matrix)
- `make baselines` (fast, no Optuna) · `make tune` (baselines + all 3 GBDTs, ~10 min measured)
- `make decide` (blend, calibrate, threshold, SHAP, feature audit, ship decision — ~25 s)
- `make train V=v1` (full MLflow run: nested children, pyfunc, registry — ~22 s) ·
  `make registry` (versions and aliases)

**Arrive with their phase:**
- `promote` (9) · `make serve | bench` (7-8)
- `make bootstrap` (8) — the ordered first-run sequence (PLAN §18 A5). **Required before
  `docker compose up` on a clean machine**, because the API resolves `@champion` at startup
  and the registry starts empty.

All Python entry points run through `.venv/bin/python -m src.x`.

**The full `requirements.txt` is installed** (1.9 GB venv): pandas 3.0.6, numpy 2.5.3,
scikit-learn 1.9.1, LightGBM 4.7.0, XGBoost 3.4.1, CatBoost 1.2.10, Optuna 5.0.0, SHAP 0.52.0,
MLflow 3.16.1, pandera 0.33.1. Two API notes that cost time if forgotten: pandera's import is
`pandera.pandas`, not top-level, and LightGBM 4.7 deprecates `eval_set` in favour of
`eval_X`/`eval_y`. `requirements-api.txt` is still unproven until Phase 8. MLflow is reached over HTTP at
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
6. Split dates come from `configs/splits.yaml`, which is keyed **per version**, and
   **`src/splits.py` is its only reader** — no other module may name a boundary. Fitting and
   calibration windows must be disjoint for every version (§18 A2); `VersionSplits` refuses to
   exist otherwise. The 2018-05 -> 08 window is **locked at runtime**: it raises unless
   unlocked by `src.evaluation.score_holdout` or `src.registry.promote` (DECISIONS.md D25).
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
    in `src/etl/schema.py`. **`src/features/history.py` is the only module that may read
    `order_outcomes`** (DECISIONS.md D19). It also carries `order_delivered_carrier_date`,
    `delivery_days` and `handling_days` for the §4.5 aggregates.
13. **The feature matrix comes only from `PreprocessingArtifact`.** Nothing outside
    `src/features/artifact.py` may fit an imputation value, a category level set or a
    winsorisation bound. `fit` is a classmethod, there is no `fit_transform`, and the
    artifact is frozen — Phase 5 adds its calibrator via `with_decision`, which returns a new
    artifact. Training uses `transform_with_snapshots`, serving uses `transform`; the two
    names are the §18 A6 asymmetry made visible (DECISIONS.md D23). The matrix is **not**
    persisted as a table — recompute it (D24).

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
**Phases 0-6 COMPLETE.** 326 tests pass. `delivery_delay_classifier` **v1 is registered with
`@champion`** and loads in a fresh Python process by alias alone. `make train` produces a parent
run plus five nested children and a pyfunc that bundles artifact + model + calibrator +
threshold (D32).

Phase 5 results, unchanged: **Ship XGBoost alone** (D31). The blend is a genuine
0.77/0.23 CatBoost/XGBoost mix and is still **worse** on PR-AUC (delta −0.00156), because weights
are fitted on log-loss while PR-AUC measures ranking. Calibration improves held-out Brier
0.0853 -> 0.0841. Threshold **0.21** under an assumed 5:1 FN:FP ratio: 36.3% recall at a 13.7%
flag rate, 25.6% precision.

`raw` holds 9 tables / 1,550,922 rows; `features.orders_analytical` holds 96,203 rows at a 6.79%
`is_late` rate; six snapshot tables hold 34,427 rows; the **38-feature** matrix and a fitted
`PreprocessingArtifact` build in ~4 s via `make features`; `src/splits.py` owns every date
boundary and locks the promotion evaluation window.

**38, not §5's 39: `purchase_month` was dropped** (D29). It was #1 by SHAP and could not have been
learned — no calibration window shares a calendar month with its fit window, and months 9-12 occur
in only one year, so a month effect is confounded with that year's operations. Dropping it improved
cross-validated PR-AUC for all three GBDTs. The surviving timing features recur often enough to be
estimable: 87 weeks for day-of-week against 1.67 years for month-of-year.

**Next: Phase 7 — FastAPI and the parity test** (~4 h). **Read §18 A3 and A6 first.**
- The parity test uses `tests/fixtures/`, **never** the 2018-05 → 08 window (§18 A3). That
  window raises unless unlocked by the gate, so the guard is enforced, not advisory.
- Parity is scoped to **one fixed artifact**: same raw record + same artifact -> same
  probability within 1e-6. Snapshot selection is outside its scope (§18 A6).
- The API layer contains **no preprocessing** (invariant 4). It resolves
  `models:/delivery_delay_classifier@champion`, passes raw order records to the pyfunc, and
  returns what comes back. `src/registry/pyfunc_wrapper.py::RAW_INPUT_COLUMNS` is the request
  contract — 25 columns, and history columns are **rejected** if a caller supplies them.
- XGBoost ships, so `requirements-api.txt` needs no `catboost` — but `xgboost` pulls the 305 MB
  CUDA library D14 flagged, which makes `xgboost-cpu` directly relevant to Phase 8.

The `realtime-fraud-detection` stack is currently **stopped** to free RAM for tuning. Restart it
with `docker start realtime-fraud-detection-redis-1 realtime-fraud-detection-scorer-1
realtime-fraud-detection-graph-refresh-1` when this project is not training.

Do not implement future phases. `PROGRESS.md` carries the detail. Two findings govern later
phases: **D20** (base-rate shift, Phases 5 and 9) and **D22** (the history features are weak,
and a dominant `seller_late_rate_hist` in Phase 4 is a leakage alarm, not a win).
