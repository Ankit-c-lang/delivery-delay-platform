# CLAUDE.md: delivery-delay-platform

## What this is
Predicts at checkout whether a Brazilian e-commerce (Olist) order will be delivered after
its promised date. **The purpose of this project is the MLOps lifecycle, not the model:**
Postgres feature warehouse -> leakage-safe as-of aggregates -> LightGBM/XGBoost/CatBoost
-> MLflow tracking + registry -> promotion gate -> containerized FastAPI.

The full plan is `PLAN.md`; prompts reference its section numbers.
**`PLAN.md` §18 holds eight binding amendments. Where §18 and the body of the plan disagree,
§18 wins.** A1-A7 came from the pre-build design review; **A8 was added during Phase 8** and changes
model selection, so the body of §6.5 is now out of date on its own. Read §18 before writing code in
Phases 3, 7, 8 or 9.

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
- `make lint | format | test` · `make test-unit` (CI's first pytest step: needs no services) ·
  `make test-ci-etl` (the real ETL on synthetic CSVs in a throwaway database)
- `make up | down | stop | ps | logs | health | psql`
  `up` waits for both services to report `healthy`. `down` keeps volumes on purpose.
- `make check-data | load-raw | build-orders` and `make etl` for all three in order
- `make snapshots` (as-of snapshots) · `make features` (fit the artifact, build the matrix)
- `make baselines` (fast, no Optuna) · `make tune` (baselines + all 3 GBDTs, ~10 min measured)
- `make decide` (blend, calibrate, threshold, SHAP, feature audit, ship decision — ~25 s)
- `make train V=v1` (full MLflow run: nested children, pyfunc, registry — ~22 s) ·
  `make registry` (versions and aliases)

- `make serve` (API on **8001**, docs at `/docs`) · `make bench` (p50/p95; `URL=` to hit a
  running service, `N=` iterations)
- `make build` (API image) · `make api-up | api-down | api-logs` · `make bootstrap`
  **`bootstrap` is required before `docker compose up` on a clean machine** (§18 A5): the API
  resolves `@champion` at startup and a fresh registry has none.

- `make promote` (the gate; `DRY=1` to measure without moving the alias)

**Arrive with their phase:**
- nothing — Phases 10 and 11 add CI and docs, not new targets
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
14. **Reproducibility is checked by VARYING the environment, never by repeating a run.**
    Every model pins its thread count and seed in `configs/models.yaml`; unpinned, all three
    libraries take every core and their FP reductions are partitioned by thread count, which
    silently decided which model shipped (DECISIONS.md D35). A result that is stable across
    consecutive runs on an idle machine can still be irreproducible. Vary `OMP_NUM_THREADS`.
15. **Row order is part of the contract.** `ORDERS_QUERY` sorts by
    `(order_purchase_timestamp, order_id)` and `build.window_slice` is the only place a window's
    rows are ordered. 522 timestamps are tied, and `TimeSeriesSplit` and `fit_final_model` both
    slice **positionally**, so an arbitrary tie order changes the model.
16. **The bundled serving snapshot is bounded by the version's `aggregates_through`**, not by
    "the newest month in the database" (D33). v1 bundles 2018-03-01. `PreprocessingArtifact.fit`
    requires the bound as an argument, so it cannot be forgotten.
17. **One decision, one implementation.** Three bugs have now had the same shape: a config value
    validated but enforced by nothing (D33), a dependency list written from what the code looked
    like it needed (D38), and the shipping choice re-derived inline in `train.py` while
    `build_decision` applied the equivalence band (D41) — which registered a CatBoost model while
    `make decide` had chosen LightGBM. If a rule exists, call it; never restate it.

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
**Phases 0-6 COMPLETE, and Phase 6's output was rebuilt in Phase 7 after two real bugs.**
`delivery_delay_classifier` **v2 is `@champion`** and loads in a fresh Python process by alias
alone. v1 is tagged `withdrawn: do not deploy` and `make registry` prints that status.

**Two corrections that changed the headline results (read D33 and D35 before trusting any older
number in this file or in `reports/`):**

- **D33** — the bundled *serving* snapshot ignored the version's `aggregates_through` and took the
  newest month in the database, so v1's serving path used 2018-08 aggregates against a 2017-12-31
  train cutoff, reaching into the promotion evaluation window. v1 now bundles **2018-03-01**.
  Training was never affected: `transform_with_snapshots` joins per purchase month.
- **D35** — thread counts were unpinned, so floating-point reduction order (and therefore the
  winning model) depended on machine load. **D31's "ship XGBoost" was noise**: its 0.0067 margin
  was smaller than the 0.005 that thread count alone moves CatBoost. Threads and seeds are now
  pinned, hyperparameters were re-tuned, and runs are byte-identical across `OMP_NUM_THREADS`.

**Current results, reproducible.** Ship **CatBoost** alone: cal-window PR-AUC **0.21604**, blend
0.21924 (**delta +0.00320**, under the 0.005 bar). Threshold **0.1792** at an assumed 5:1 FN:FP
ratio: **42.5% recall at a 17.2% flag rate, 24.0% precision**. CV PR-AUC lightgbm 0.15349,
catboost 0.15209, xgboost 0.14666, against a logistic baseline of 0.13545.

**Treat the three GBDTs as indistinguishable.** They span ~0.004 on a 13,624-row calibration
window. Choose between them on inference cost, dependency weight or unseen-category handling —
not on PR-AUC. CatBoost shipping means `requirements-api.txt` carries `catboost`, and that file
notes the coupling: promoting a challenger that ships a different library needs an image rebuild.

`raw` holds 9 tables / 1,550,922 rows; `features.orders_analytical` holds 96,203 rows at a 6.79%
`is_late` rate; six snapshot tables hold 34,427 rows across 2017-02..2018-08; the **38-feature**
matrix and a fitted `PreprocessingArtifact` build in ~5 s via `make features`; `src/splits.py` owns
every date boundary and locks the promotion evaluation window.

**38, not §5's 39: `purchase_month` was dropped** (D29) — no calibration window shares a calendar
month with its fit window, and months 9-12 occur in only one year. **It is also no longer a request
field** (D34): serving forces it to the bundled snapshot month, so the API contract is **24
columns**, not 25, and supplying it is refused.

**Phase 7 COMPLETE.** 378 tests pass, zero skips. `api/` is §9's structure; the model loads once in an
`asynccontextmanager` lifespan; `/health` answers **503** when it did not load and a bad
`MODEL_URI` does not kill the process; `MODEL_URI` resolves as alias, pinned version or local path
(the last is what lets the API tests run in CI with no registry). **Parity is green at 1e-6**, and
`tests/test_parity.py` carries a guard class so it cannot pass vacuously — rows are taken on a
stride, not `head(20)`, and the fixture calibrator is fitted out of sample, because in-sample
scores collapse isotonic into a two-step function. Invariant 4 is enforced by **AST import
inspection**: nothing under `api/` may import `src.features` or `src.training`.

**Measured latency, real HTTP:** single p50 **52.64 ms** / p95 59.98; batch-100 p50 **54.91 ms** /
p95 75.07; **0.549 ms per record in a batch — 96x cheaper**. Nearly the whole bill is fixed
overhead in the as-of snapshot join (1-row transform ~45 ms against ~69 ms for 1,000), not the
model, which is sub-millisecond. Optimise there if it ever matters.

**Operational note for Phase 8:** running the champion over all 96,203 real orders warns on 3
seller states (`AM`, `MA`, `PI`) and 43 product categories. The categories are **not drift** —
`CATEGORY_CAP = 30` collapses the other 42 by design — which is why the warning says "not among
the levels this model was trained on" rather than "never seen".

**Phase 8 COMPLETE.** `docker compose down && docker compose up -d` gives a healthy three-service
stack and a working prediction; `delay-api` runs as **uid 1000**, resolves `@champion` through
`http://mlflow:5000`, and the container's prediction is **bitwise identical** to the venv's.
Image **1.03 GB** from a 65-package serving subset (the full stack is 110 packages / 1.9 GB of
site-packages).

**LightGBM ships, not CatBoost** (D41). Model selection now has an **equivalence band** of
**0.005** — D35's measured reproducibility floor — and breaks ties inside it on serving cost. The
observed gap was **0.00020**, 25× narrower than the floor, and the swap cut the image from 1.62 GB to
**1.03 GB** with latency unchanged (single p50 56.65 ms against CatBoost's 53.41; batch-100 80.45
against 88.21 — opposite directions, so noise). `@champion` is **v4**.

**Three findings worth carrying forward:**
- **D38** — the serving dependency set was *measured*, not assumed, and `pandera`, `sqlalchemy` and
  `psycopg2` fell out. `DENYLIST` now lives in dependency-free `src/contract.py`, and `src.db` /
  `src.etl` imports in `src/features/` are function-local. **Do not restore them to module scope.**
- **D40** — MLflow 3 rejects `Host: mlflow:5000` with a 403 naming DNS rebinding.
  `MLFLOW_SERVER_ALLOWED_HOSTS` fixes it and **replaces** the default allowlist, so the localhost
  entries must stay in the list or host-side tooling breaks.
- **A venv cannot tell you what an image needs.** `pyarrow` was missing from
  `requirements-api.txt` because it reached the venv via *full* `mlflow`, which `mlflow-skinny`
  drops. Only a from-scratch install distinguishes a direct dependency from an inherited one.

**Phase 9 COMPLETE.** 435 tests pass. The gate ran for real and **refused**, which is the phase
working (§8: a no-promote run is a success). `src/evaluation/score_holdout.py` is the only module
that loads 2018-05 → 08, enforced by `src/splits.py` at runtime and by import inspection in tests.

**THE HEADLINE NUMBER IS 1.36x LIFT, NOT 0.216 PR-AUC** (D42). On the calibration window the
champion scores 0.21585 at a ~9.75% base rate, about 2.2x lift. On the genuinely later evaluation
window it manages PR-AUC **0.05983 at a 4.40% base rate — 1.36x lift**, with precision 6.2% at a
16.3% flag rate. Lift already normalises for D20's base-rate shift, so that fall is **real
forward-time degradation**. Phase 11 must lead with it; quoting 0.216 without its window is the one
dishonest thing this project could still do.

**The gate's decision, recorded in `reports/promotions.md`:** challenger v5 (v2 windows, xgboost,
1.30x) against champion v4 (v1 windows, lightgbm, 1.36x) — **not promoted**, PR-AUC delta −0.00282
against a +0.005 bar. Brier, schema and latency all cleared. The shortfall is inside the equivalence
band, so it is a **tie, not a regression**, and a tie leaves the champion alone. v2 saw two months
more data and gained nothing, which is consistent with D22 (weak history features). **Nothing was
retuned** — §13 requires recording the loss.

**Why §8 says "the same rows".** Each version's own calibration-window PR-AUC is 0.20766 (v4) and
0.24084 (v5), which would suggest the challenger is far better — but those are different windows with
different base rates and are not comparable. On the only window both are scored on, the ordering
reverses.

**The gate's known limit, for Phase 10-11 to state plainly:** the challenger ships **XGBoost** while
`requirements-api.txt` installs **LightGBM**, so promoting v5 would have produced a container that
cannot unpickle its own model. With D41's `libgomp1` finding, promotion is **an alias move plus an
image that can load what the alias points at** — and the gate checks only the first half.

**Phase 10 COMPLETE.** `ci.yml` now runs ruff, black, **pytest with a Postgres service container**
and coverage; `build.yml` builds the image, smoke-tests it and publishes to GHCR as `:latest` and
`:{sha}`. `README.md` exists with badges.

**Three things to know before touching CI** (D43):
- **CI never needs the dataset.** `tests/fixtures/synthetic_olist.py` writes all nine raw Olist CSVs
  with headers taken from `load_raw.SPECS`, and the `ci_etl` tests run the **real** loader and
  transform against them in a database they create and drop themselves. Do not point that marker at
  the configured database — `recreate=True` would destroy a developer's loaded data.
- **The `integration` marker means "needs a service ALREADY LOADED with the project's data"**, which
  is why a service container does not let those tests run. The description used to understate this.
- **The fixture model is generated, not committed** (`scripts/make_fixture_model.py`), with
  `requirements-api.txt` so its libraries match the image's. A pickle in git is coupled to the
  versions that wrote it with nothing recording the coupling.

The image smoke test compares the container's probability to the runner's **bitwise**, so a broken
model load or a pruned dependency that shifts numerics fails before anything is published.

**The first CI run failed twice, and neither was findable locally** (D44). `pythonpath = ["."]` is
now in `pyproject.toml` because `python -m pytest` puts the cwd on `sys.path` and the bare `pytest`
console script does not — the Makefile used the former for ten phases, so the suite could not even
*collect* in CI. And `IMAGE` is lowercased in a step, because `${{ github.repository }}` keeps the
owner's capitals and Docker rejects them. **In CI the three live-container tests skip** (no
`delay-api` there), so CI will not show local's zero skips; the job summary says so.

**Phase 11 COMPLETE — all 12 phases done.** `README.md` carries the architecture diagram, the full
results chain, latency, one-command reproduction and a nine-point limitations section;
`reports/interview_notes.md` answers §13's nine questions.

**D45, found while writing the results table, is now the most important fact in the repository:**
a **logistic regression beats the tuned GBDT on the held-out window** — 2.08x lift against 1.36x,
a 0.03181 PR-AUC gap that is **6x the 0.005 reproducibility floor**, so not noise. In-window the
ordering reverses (CV: LightGBM 0.15349, logistic 0.13545). The GBDTs win where they were tuned and
lose where it counts. The champion is still **better calibrated** (Brier 0.049 vs 0.065), because its
probabilities pass through isotonic and the logistic's are raw — ranking and calibration are
different things. Reproduce with `make holdout-reference`.

**Nothing was changed in response.** §13's discipline — record a losing challenger rather than tune
until it wins — applies equally to a winning baseline. Reacting to one window is how an evaluation
set becomes a training set. The principled next step is to **register the logistic as a challenger
and let the gate decide**, which is the first item in the interview notes' final answer.

The `realtime-fraud-detection` stack is currently **stopped** to free RAM. Restart it with
`docker start realtime-fraud-detection-redis-1 realtime-fraud-detection-scorer-1
realtime-fraud-detection-graph-refresh-1` when this project is not training.

Do not implement future phases. `PROGRESS.md` carries the detail. Two findings govern later
phases: **D20** (base-rate shift, Phases 5 and 9) and **D22** (the history features are weak,
and a dominant `seller_late_rate_hist` in Phase 4 is a leakage alarm, not a win).
