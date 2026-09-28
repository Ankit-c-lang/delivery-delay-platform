# PROGRESS — delivery-delay-platform

Living status file. **Update it at the end of every completed task**, in the same commit as
that task. Source of truth for *what* to build is `PLAN.md`; this file only tracks *where we
are*. Decisions and their reasons live in `DECISIONS.md`.

- **Plan version:** v1 (2026-09-28) + amendments §18 · **Pre-flight completed:** 2026-09-28
- **Current phase:** Phase 2 — Transform, validate, analytical table · **✅ COMPLETE** —
  96,203 orders, `is_late` 6.79%, Pandera green, 80 tests green
- **Overall:** 3 / 12 phases · pre-flight and environment complete (not counted as a phase)
- **Estimated remaining:** ~33 h of build work (PLAN §3 budget: 36-42 h)

---

## Currently working on

**Nothing in flight.** Phase 2 is complete and verified.

`features.orders_analytical` holds **96,203 rows, 38 columns, one row per order**, built in
**12.5 s**. The `is_late` rate is **6.79%**, inside §4.3's 5-10% band. Pandera passes with
`strict=True`. 80 tests pass, 89% coverage; 37 of them are unit tests needing neither Postgres
nor the dataset.

**The design question Phase 2 forced, and how it was resolved.** Phase 2's prompt says to drop
every §4.2 denylist column, but §18 A1 rewrote §4.5 to filter snapshots on
`order_delivered_customer_date` — so Phase 3 needs a column Phase 2 was told to destroy. That
is an A1 consequence the amendment never propagated into Phase 2's prompt. Resolved by
quarantining the column in a separate two-column table, `features.order_outcomes`, leaving
`orders_analytical` genuinely denylist-free and `strict=True` truthfully enforceable. Full
reasoning and the two rejected alternatives are in `DECISIONS.md` D19, and it is now
`CLAUDE.md` invariant 12.

**Division of labour:** SQL does the set-based aggregation (a join across 1.55 M raw rows),
pandas computes the great-circle distance and holds the frame for validation. The distance uses
`src.etl.geolocation.haversine_km` rather than a second formula in SQL, so the function the
tests exercise is the function that runs.

---

## A finding that changes Phases 5 and 9

The late rate is **strongly non-stationary**, and both calibration windows sit at roughly
2.2-2.7x the base rate of the window both models are judged on:

| Window | Orders | Late rate |
|---|---:|---:|
| v1 fit 2017-05 → 12 | 36,174 | 5.85% |
| **v1 calibrate** 2018-01 → 02 | 13,624 | **9.75%** |
| v2 fit 2017-05 → 2018-02 | 49,798 | 6.92% |
| **v2 calibrate** 2018-03 → 04 | 13,801 | **11.84%** |
| **promotion evaluation** 2018-05 → 08 | 25,352 | **4.40%** |

Monthly extremes are 18.96% (2018-03) and 1.16% (2018-06) — a 16-fold spread. November 2017
at 12.40% is the Black Friday spike §4.4 predicts.

**Consequence:** isotonic calibration fitted at a ~10-12% prior will over-predict on a 4.4%
evaluation set, and a threshold swept on the calibration window will be too low there. Phase 5
will see a poor Brier score on the evaluation set and the cause will not be the code. The
promotion gate itself is unaffected — champion and challenger are scored on the *same* set, so
the shift cancels — but the absolute Brier tolerance must be set with these numbers in mind.
The split boundaries were **not** moved to flatten this; see `DECISIONS.md` D20.

---

## Data-quality findings, carried from Phase 1

Measured against the real files, before writing any DDL. These matter for §4.3's population
filter and §4.5's snapshot rule:

| Finding | Count | Consequence |
|---|---|---|
| `order_status = 'delivered'` but `order_delivered_customer_date IS NULL` | **8** | §4.3's filter needs **both** conditions, not just the status |
| `order_status = 'canceled'` but delivery date **present** | **6** | filtering on the date alone is also not enough |
| `order_estimated_delivery_date` is exactly midnight | **99,441 / 99,441** | it is date-only; §4.3 must cast both sides to `date` or nearly every same-day delivery reads as late |
| `review_id` duplicates | **814** | reviews are keyed `(review_id, order_id)`; one review can cover several orders |
| `product_category_name` NULL in products | **610** | join to the translation table must tolerate it |
| Order status mix | 97.02% delivered, 1.11% shipped, 0.63% canceled | the funnel §4.3 asks for in the README starts here |
| Purchase span | 2016-09-04 → 2018-10-17 | §4.4 drops everything outside 2017-01-01 → 2018-08-31 |
| Categories in products vs translation | 73 non-null distinct vs 71 rows | **2 categories have no English translation**: `pc_gamer` and `portateis_cozinha_e_preparadores_de_alimentos`. A plain inner join silently drops them — Phase 2 must left-join and decide the fallback |

---

## Next up (in order)

**Phase 3 — As-of aggregates and feature assembly** (PLAN §13 Phase 3, ~5 h, **the hardest
phase**).

**Read `PLAN.md` §18 A1 before writing a line of it.** The original snapshot rule leaked
future outcomes and the leak favours the positive class.

**Carry into Phase 3:**
- Snapshots for month *M* admit only orders whose **delivery outcome** was known before *M*:
  `order_delivered_customer_date < M`, never `order_purchase_timestamp < M`. Assert
  `max(order_delivered_customer_date) < M` in a test (§6.6, §18 A1).
- That column lives **only** in `features.order_outcomes`. Join it to `orders_analytical` on
  `order_id`. Add the import-inspection guard so `order_outcomes` has exactly one reader, the
  same way §18 A3 requires for the evaluation window (D19).
- Entities are seller (`seller_id`), route (`route`, already built as
  `seller_state->customer_state`) and product category (`dominant_category`). All three are in
  `orders_analytical`.
- Join each order to the snapshot for its own `purchase_month`, which is already a
  first-of-month timestamp in the table.
- Cold start: emit a `*_is_new` flag and fall back seller → seller state → global. Never
  impute silently.
- `days_to_shipping_limit` is stored **raw**, range 2.00 to 1052.00 days. Winsorize the tail
  and audit near-collinearity with `promised_days` (§18 A4) — both are Phase 3's job.
- 1,330 orders have no `dominant_category` and 16 no weight or volume. Handle explicitly.
- The warm-up window 2017-01 → 04 (7,252 orders) exists so 2017-05 already has four months of
  *resolved* history. Those rows are aggregate input only, never training rows.

**Start of session:** `make up`, then `make etl` if the volume is fresh (about 17 s end to
end). Credentials are already in `.env`.

---

## Environment (done — not part of the build work)

Completed 2026-09-28. Full table in `CLAUDE.md`.

| Item | Result |
|---|---|
| Disk | **39 GB -> 77 GB** via `lvextend` + `resize2fs`; 48 GB free. VG had 39 GB unallocated |
| `python3-venv` / `pip` / `python3-dev` | installed; venv creation verified working |
| Dependency stack | **117 packages resolve on 3.12, all binary wheels, no source builds** |
| Dataset | 9 Olist CSVs, 121 MB, in `data/raw/`; counts parser-verified; span 2016-09 -> 2018-10 |
| Kaggle credentials | `~/.kaggle/kaggle.json`, mode 600 |
| Git / GitHub | repo on `main`, private remote wired over SSH, push verified |
| Ports | 5432, 5000, 8001 free. **8000 avoided** — owned by `realtime-fraud-detection` |
| Registries | `postgres:16-alpine`, `python:3.12-slim`, `ghcr.io/mlflow/mlflow` reachable |
| Sibling-project safety | no destructive Docker commands; see `CLAUDE.md` invariant 10 |

---

## Completed

### Pre-flight and design review — ✅ COMPLETE (2026-09-28)

- Environment audited end to end; two blockers found and cleared (broken `venv`, 9.9 GB
  free disk).
- `PLAN.md` committed unamended at `a098948` so the review reads as a diff.
- **Design review found six defects plus one version-drift change.** All resolved in
  `PLAN.md` §18 and patched through the body (30 edits) at `1d76389`. Two were provable
  contradictions between required tests; one was a target-correlated label leak measured
  on the real data. See `DECISIONS.md` D5-D11.
- `CLAUDE.md`, `PROGRESS.md`, `DECISIONS.md` created.

### Phase 0 — Skeleton and environment — ✅ COMPLETE (2026-09-28)

All eight items from the Phase 0 prompt:

| # | Item | Verified by |
|---|---|---|
| 1 | §12 directory tree, empty `__init__.py`, `.gitkeep` in the empty leaves | `find` listing; all 7 packages importable paths exist |
| 2 | `pyproject.toml` — ruff (line-length 100), black, pytest (`testpaths`, `--cov=src`) | `ruff check .` and `black --check .` both clean |
| 3 | Full `.gitignore` (replaced the interim one), `.dockerignore`, `.gitattributes` (LF), `.env.example` | `git check-ignore -v` on `data/`, `.env`, `mlflow/`, `mlartifacts/`; negations for `.env.example` and `reports/.gitkeep` confirmed; nothing from `data/` staged |
| 4 | `docker-compose.yml` — `postgres` + `mlflow` only, explicit `name: delay-prediction` | `docker compose config --quiet`; `up -d --wait` reached `healthy` for both in 21 s |
| 5 | `src/db.py` — SQLAlchemy engine factory over pydantic-settings | live query: `server_version` 16.15, `current_user` `delay`, password masked in `safe_url` and `repr` |
| 6 | `configs/base.yaml`, `configs/splits.yaml` keyed **per version** (§18 A2) | both parse; fit/calibrate/eval windows proven pairwise disjoint for v1 and v2 |
| 7 | `.github/workflows/ci.yml` — 3.12, `requirements-dev.txt`, ruff + black, no tests | green on the first push, run `36430031887`, 17 s |
| 8 | `requirements.txt` / `-dev` / `-api`, unpinned | `-dev` installed and working; the other two not yet installed |

Also, in `PLAN.md` scope but outside the numbered prompt (D15): `Makefile` (Phase 0 targets
only) and `.pre-commit-config.yaml` (hook installed; all 10 hooks pass on every tracked file).

**Measured:** MLflow 3.16.1 on `127.0.0.1:5000`, `/health` 200, UI 200, registry API
answering. `--serve-artifacts` confirmed working — the default experiment reports
`artifact_location: mlflow-artifacts:/0`, i.e. proxied through the server rather than a raw
host path. Postgres 16.15 on `127.0.0.1:5432`. Bind-mounted `mlflow/mlflow.db` (856 KB) is
owned by `ankit`, not `root`.

**Sibling project untouched, checked after bringing the stack up:** all three
`realtime-fraud-detection` containers still up (5 h uptime, redis healthy),
`fraud-app:local` 4.65 GB and `realtime-fraud-detection_redis-data` both still present. No
prune of any kind was run; the only new volume is `delay-prediction-pgdata`.

### Phase 1 — Raw load into Postgres — ✅ COMPLETE (2026-09-28)

| # | Item | Verified by |
|---|---|---|
| 1 | `src/etl/load_raw.py` — explicit DDL per table, `COPY FROM STDIN`, idempotent, one transaction | 9/9 tables reconcile; 3 consecutive runs give identical counts |
| 2 | `scripts/download_data.py` — presence + header check, non-zero exit, no scraping | `make check-data` passes; reports 126 MB across 9 files |
| 3 | `tests/test_etl_load_raw.py` + `tests/conftest.py` | 33 tests, 88% coverage |
| 4 | `make load-raw` (plus `check-data`, `test-unit`) | run from a clean start and twice more |

**Row counts, all reconciled three ways** (CSV parser = Postgres = Olist's published figures):
orders 99,441 · customers 99,441 · order_items 112,650 · order_payments 103,886 ·
order_reviews **99,224** · products 32,951 · sellers 3,095 · geolocation 1,000,163 ·
product_category_name_translation **71**. Report: `reports/raw_load_reconciliation.md`
(committed on purpose — a cloner cannot regenerate it without the dataset).

**Verified in the database, not just asserted in code:** all 8 timestamp columns are
`timestamp without time zone`; leading zeros survive (23,995 / 1,027 / 245,733 prefixes start
with `0`); 3,852 review comments keep their embedded newlines; the UTF-8 BOM on the
translation file never reached a value; nulls landed as NULL with **zero** empty strings;
money is `numeric(10,2)`; 8 primary keys exist and `geolocation` correctly has none.

**The `TIMESTAMPTZ` trap was demonstrated, not assumed** — the same row reads as on-time under
`TIMESTAMP` and late under `TIMESTAMPTZ` when the client timezone changes. See `DECISIONS.md`
D17.

### Phase 2 — Transform, validate, analytical table — ✅ COMPLETE (2026-09-28)

| # | Item | Verified by |
|---|---|---|
| 1 | `src/etl/geolocation.py` — median zip centroid, state fallback, `haversine_km` | 19,011 zip + 27 state centroids; haversine checked against 4 known pairs and 1° = 111.195 km |
| 2 | `src/etl/build_orders.py` — filter, aggregates, joins, target, funnel | 96,203 rows, one per order, `is_late` 6.79% |
| 3 | `src/etl/schema.py` — Pandera contract, `strict=True`, raises on failure | passes on the built table; rejects a smuggled denylist column, a 100% rate, duplicate ids and a Portugal coordinate |
| 4 | `tests/test_etl_build_orders.py` | 47 new tests (80 total), 89% coverage |
| 5 | `make build-orders`, and `make etl` for the whole chain | run three times; identical counts and rate each time |

**Funnel** (`reports/etl_funnel.md`, committed): 99,441 → 96,478 delivered (−2,963) → 96,470
with a delivery date (−8) → **96,203** in window (−267). Kept 96.74%.

**Verified in the database, not just asserted:** exactly 96,203 distinct `order_id`s for
96,203 rows; **zero** denylist columns present; the state-centroid fallback fired 265 times for
customers and 213 for sellers with **no** order left unresolved; same-state orders average
153 km against 853 km cross-state; every coordinate inside the Brazil bounds;
`days_to_shipping_limit` never below the 2.00-day floor §18 A4 measured.

**The geolocation reduction earned its design.** Prefix `68275` (Porto Trombetas, Pará) has 9
points: 4 correctly in Pará and 5 geocoded to **Portugal**, having matched "Porto". A mean
lands in the Atlantic; a median over the unfiltered points still returns Portugal (41.15°N).
Dropping the 31 out-of-Brazil points first moves it to −1.7435 — a ~4,800 km correction.
Median then handles the 199 prefixes that remain noisy within Brazil.

**A test of mine caught a real trap in the tests themselves:** the positive-rate band does
**not** detect a fan-out join. Measured on the item-level fan-out the rate is 6.61%, still
inside 5-10%. Only the one-row-per-order test catches it, which is now stated in the test
module's docstring so nobody relies on the wrong guard.

---

## Remaining work

| Phase | Title | Est. | Difficulty | Status |
|---|---|---|---|---|
| 0 | Skeleton and environment | 1.5 h | easy | ✅ **complete** · CI green |
| 1 | Raw load into Postgres | 2.5 h | easy-med | ✅ **complete** · 33 tests |
| 2 | Transform, validate, analytical table | 4 h | medium | ✅ **complete** · 80 tests |
| 3 | As-of aggregates and feature assembly | 5 h | **hardest** | **next** · ⚠️ read §18 A1 first |
| 4 | Baselines and single models | 4 h | medium | full dependency install lands here |
| 5 | Ensemble, calibration, threshold | 4 h | medium | ⚠️ base-rate shift (D20); fixes whether CatBoost must re-enter `requirements-api.txt` (D14) |
| 6 | MLflow, pyfunc wrapper, registry | 4 h | medium | |
| 7 | FastAPI and the parity test | 4 h | medium | ⚠️ read §18 A3, A6 first |
| 8 | Docker and Compose | 3 h | medium | ⚠️ read §18 A5 first · validate `mlflow-skinny` (D14) |
| 9 | Retraining lifecycle and promotion gate | 3.5 h | medium | ⚠️ read §18 A2 first; set the Brier tolerance against D20 |
| 10 | Full CI/CD | 3 h | medium | CI gains pytest + Postgres service container |
| 11 | Documentation and interview prep | 2.5 h | easy | |

**If behind schedule, cut in this order** (PLAN §3): drift report, prediction logging,
API-key auth, CLI wrapper, CatBoost, Optuna (keep `TimeSeriesSplit`), trainer profile.
**Never cut:** the as-of aggregate design, the frozen evaluation set, the parity test, the
registry chain, the promotion gate.

---

## How to update this file

At the end of every completed task, in the same commit:

1. Bump **Phase N progress** and **Overall**.
2. Rewrite **Currently working on** to describe what actually just happened — including
   anything that went wrong and what it cost. Measured numbers, not estimates.
3. Rewrite **Next up** so a cold session can resume from it alone.
4. Move finished phases into **Completed** with the date.
5. Keep a **Not yet verified** list. A thing that was written but never run is not done.
6. Any non-obvious decision made along the way goes in `DECISIONS.md`, not here.
