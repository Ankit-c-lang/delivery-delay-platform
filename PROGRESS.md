# PROGRESS — delivery-delay-platform

Living status file. **Update it at the end of every completed task**, in the same commit as
that task. Source of truth for *what* to build is `PLAN.md`; this file only tracks *where we
are*. Decisions and their reasons live in `DECISIONS.md`.

- **Plan version:** v1 (2026-09-28) + amendments §18 · **Pre-flight completed:** 2026-09-28
- **Current phase:** Phase 4 — Baselines and single models · **✅ COMPLETE**
- **Overall:** **5 / 12 phases** · 267 tests green
- **Estimated remaining:** ~24 h of build work (PLAN §3 budget: 36-42 h)

---

## Currently working on

**Nothing in flight. Phase 4 is complete.**

All three GBDTs beat both baselines, which is the phase's completion criterion, and **no study
timed out** — the whole run took **~10 minutes against a 55-minute worst case**, so §6.4's
budget was conservative for this dataset size.

| Model | CV PR-AUC | Lift | Trials | Wall clock | Budget |
|---|---:|---:|---:|---:|---:|
| _majority class_ | 0.0656 | 1.12x | — | — | — |
| _logistic regression_ | 0.1357 | 2.32x | — | — | — |
| **LightGBM** | **0.1528** | **2.61x** | 40/40 | **105 s** | 900 s |
| CatBoost | 0.1489 | 2.55x | 20/20 | 373 s | 1500 s |
| XGBoost | 0.1468 | 2.51x | 40/40 | 107 s | 900 s |

CatBoost is **3.5x slower than LightGBM for a slightly worse score**, exactly as §6.4 warned.

**The margins are modest, and that is consistent with D22.** The best GBDT beats logistic
regression by +0.017 PR-AUC — a 12.6% relative gain — and the three GBDTs land within 0.006 of
each other. Both follow from the history features being weak: there is limited non-linear
structure to find beyond what a linear model already extracts from `promised_days`.

**`scale_pos_weight` tested, not assumed (§6.4).** At the implied ratio of 16.1, LightGBM scored
**0.1456 against 0.1528 unweighted — worse by 0.0071**. §6.4 predicted precisely that:
reweighting moves the probabilities without improving the ranking, and PR-AUC measures the
ranking. No resampling anywhere.

### The finding that needed digging for: per-fold PR-AUC mostly tracks the base rate

LightGBM's folds scored 0.055, 0.105, 0.147, **0.304** — a 5.5x spread that looks like the model
improving dramatically. It is mostly not:

| Fold | Validation window | Base rate | PR-AUC | Lift |
|---|---|---:|---:|---:|
| 1 | 2017-07-06 → 08-30 | 2.82% | 0.0547 | 1.94x |
| 2 | 2017-08-30 → 10-19 | 4.17% | 0.1052 | 2.52x |
| 3 | 2017-10-19 → 11-26 | 9.59% | 0.1471 | **1.53x** |
| 4 | 2017-11-26 → 12-31 | 9.65% | 0.3040 | 3.15x |

**Correlation between fold base rate and fold PR-AUC: +0.80.** PR-AUC is bounded below by the
base rate, so a high-late-rate fold scores higher for free — the same D20 non-stationarity, now
inside CV.

Two consequences:
1. **The model is weakest exactly when it matters most.** Fold 3 is November 2017, the Black
   Friday spike, with the highest base rate and the **lowest lift (1.53x)**. Delivery performance
   degrades during the peak in ways these 39 features do not capture, so the model is least
   useful in the period operations would most want it.
2. **A mean over unequal-base-rate folds is not a neutral average** — it is dominated by the
   high-rate folds, so tuning prefers hyperparameters that work *late* in the window. Arguably
   the right bias for forward deployment, but a choice rather than an accident.

Both numbers are now in `reports/phase4_models.md`. See `DECISIONS.md` D26 and D27.

### Three defects found and fixed while building

- **A `FutureWarning` I introduced.** `LogisticRegression(n_jobs=1)` has had no effect on lbfgs
  since scikit-learn 1.8 and is removed in 1.10. Dropped.
- **LightGBM 4.7 deprecates `eval_set`** in favour of `eval_X`/`eval_y`. Checked the actual
  signature rather than guessing, and switched. The smoke tests now pass with
  `-W error::FutureWarning`.
- **Ruff rejected `X` as an argument name** (N803). Rather than rename to something worse, the
  exception is scoped to `src/training/*`, `src/evaluation/*` and `tests/*` with the reason
  written in `pyproject.toml`: `X, y` is the convention every ML reader expects.

### A note on the install

`requirements.txt` took ~17 minutes, which looked like a hang. It was not: downloads had
finished and pip was unpacking — the venv grew 923 MB → 1342 MB in 20 seconds while the network
sat at 11 KB/s. Worth knowing before killing it next time. Final venv is **1.9 GB**; 42 GB free.

**The `realtime-fraud-detection` stack is currently stopped** to free RAM. Restart with
`docker start realtime-fraud-detection-redis-1 realtime-fraud-detection-scorer-1
realtime-fraud-detection-graph-refresh-1`. Its image and volume are untouched.

---

## Earlier: pre-push audit of Phases 0-2 (2026-09-28) — two defects found and fixed

**1. A first-run performance bug that would have looked like a hang.** The whole ETL was run
into a freshly created, empty database to test reproducibility. `load_raw` was fine, but the
Phase 2 staging query ran **over 833 s before being cancelled**, against ~12 s in the working
database — same code, same data. Cause: the centroid tables are created *and joined* inside one
transaction, so autovacuum can never analyze them, and with no statistics the planner chose
nested loops against 96,203 rows. The working database only looked fast because earlier
committed runs had left statistics behind. **This is exactly the `make bootstrap` path (§18
A5)**, so the slow run was the one nobody would have measured. Fixed with an explicit `ANALYZE`
after every bulk load; from-scratch build is now **16.1 s**, whole ETL **21.6 s**. See
`DECISIONS.md` D21, including an honest caveat about memory pressure during the measurement.

**2. A `git filter-branch` backup ref was never deleted.** `refs/original/refs/heads/main`
still pointed at the pre-rewrite history, keeping two commits with `Co-Authored-By` trailers
alive **locally**. They were unreachable from `main` and the remote only ever had
`refs/heads/main` — re-verified trailer-free — so nothing wrong was published. Ref deleted;
`main`'s commit and tree hashes are unchanged. Orphaned objects remain in the object database
until a future `gc`; pruning them was declined as irreversible and is not needed, since `git
push` only sends objects reachable from the ref being pushed.

### What it verified as sound

| Check | Result |
|---|---|
| Reproducibility from an empty database | `features.orders_analytical` and `order_outcomes` are **md5-identical** to the working database; all 9 raw tables match too |
| Transaction boundary | the cancelled 833 s run rolled back with **no** `features` table left behind |
| Attribution | **zero** `Co-Authored-By` trailers reachable from any ref, local or remote |
| Secrets and data | nothing under `data/`, no `.env`, no `kaggle.json` tracked; a fresh clone carries no CSVs |
| Fresh clone completeness | all 19 files the phases need are present; `scripts/download_data.py` exits 1 with instructions |
| Doc accuracy | every number in `CLAUDE.md`, `PROGRESS.md` and `DECISIONS.md` matches the live database |
| `requirements.txt` / `-api.txt` | both resolve cleanly (exit 0), all wheels — closes the "never installed" gap for **resolution** |
| Python version | 3.12 everywhere, no strays |
| Leftover markers | no TODO / FIXME / XXX / HACK in source |
| Tests | 80 collected — 37 unit, 43 needing Postgres or the dataset; all pass |
| Sibling project | all 3 `realtime-fraud-detection` containers still up, image and volume intact |

### Two findings it carried forward

- **`xgboost` pulls a 305 MB CUDA library.** A dry-run resolve of `requirements-api.txt` brings
  in `nvidia-nccl-cu13` (measured: 305.1 MB) on a CPU-only machine. `xgboost-cpu` is the slim
  alternative. **Phase 8 must measure the image both ways** before repeating the "skinny MLflow
  roughly halves it" claim — see `DECISIONS.md` D14.
- **`raw.geolocation` contains 261,831 exact duplicate rows** (26% of 1,000,163). Harmless for a
  median centroid, and it explains why an `ORDER BY` over only 3 of its 5 columns gives an
  unstable checksum. Worth knowing before anyone builds a per-prefix count feature.

---

Phase 2 itself:

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

**Phase 5 — Ensemble, calibration, threshold** (PLAN §13 Phase 5, ~4 h, medium). Read §6.5 and
§4.7 first.

**Three findings govern this phase, and all three say "report honestly rather than manufacture a
win":**

1. **D27 — the blend may not pay.** The three GBDTs are within **0.006 PR-AUC** of each other, so
   §6.5's expectation of +0.002 to +0.01 is exactly the range to expect. Measure the delta, and
   be ready to **ship the single model** with the ensemble kept as a logged experiment. §6.5 is
   explicit that stating that decision plainly is the stronger answer.
2. **D20 — calibration will look broken and will not be.** Both calibration windows sit at
   9.75% and 11.84% against the evaluation window's 4.40%. Isotonic calibration fitted at a
   ~10-12% prior **will over-predict** on a 4.4% set, and a threshold swept on the calibration
   window will be too low there. Report calibration on both windows and name the cause.
3. **D26 — the model is weakest during the November spike** (lift 1.53x against 3.15x in the
   adjacent fold). The cost-based threshold in §4.7 is swept on a window whose base rate is not
   the deployment base rate; say so.

**Practical notes:**
- Blend weights, calibration and threshold are fitted on the **calibrate** window, which is
  `version_splits(v)` — never the fit window and never the evaluation window.
- Phase 5 attaches its calibrator and threshold via `artifact.with_decision(...)`, which returns
  a **new** artifact (D23).
- §4.7 wants the review-score damage analysis too: reviews are in `raw.order_reviews`, for
  business impact only, **never as features** (§4.2).
- Frame the cost ratio as an assumption, never as measured.

**Start of session:** `make up`, then `make etl && make features` if the Postgres volume is fresh
(~29 s), then `make tune` (~10 min) if you need the tuned parameters.

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

### Phase 3 part A — As-of monthly snapshots — ✅ COMPLETE (2026-09-28)

| # | Item | Verified by |
|---|---|---|
| 1 | `src/features/history.py` — expanding as-of snapshots for seller / route / category | 6 tables, 34,427 rows, 3.2 s |
| 2 | Cold-start fallback entity → state → global with `{entity}_is_new` | seller 7.56%, route 0.29%, category 1.33% on training rows; the state level rescues 6,711 of 6,729 cold sellers, 18 need global |
| 3 | `tests/test_features_history.py` | 29 tests, 22 of them synthetic and database-free |
| 4 | `make snapshots` | run repeatedly; deterministic |

**Snapshot tables:** `seller_monthly` 26,730 · `route_monthly` 5,557 · `category_monthly` 1,255 ·
`customer_state_monthly` 503 · `seller_state_monthly` 363 · `global_monthly` 19.
`global_monthly` has 19 rows, not 20, because 2017-01 has **zero** resolved orders — months with
no history are absent, never zero-filled, since a zero-filled row reads as "0% late" rather than
"unknown".

**The leakage boundary is proven three ways:** synthetically at the exact boundary (an order
delivered at 00:00:00 on the first of *M* is excluded — `<`, never `<=`); by the plan's
sharp-change seller (clean for months 1-3 then late in month 4 reads **0.0** at month 4, not
0.25); and against the real data by an independent set-based SQL recount of all 26,730
seller-month snapshots, FULL OUTER JOIN, zero mismatches.

**The resolved frame has no purchase-timestamp column at all**, so the leaky rule is not merely
avoided — it is unavailable. A test asserts that, so widening the contract requires making the
argument in the open.

**Design decision worth knowing:** the fallback supplies *rates*, never *volumes*.
`seller_order_count_hist` is 0 for a cold seller, not São Paulo's thousands. Inheriting the
volume would tell the model a brand-new seller has a long track record — the opposite of true,
and it would make `order_count` an anti-signal exactly where `is_new` is trying to help.

### Phase 3 part B — Feature assembly and the preprocessing artifact — ✅ COMPLETE (2026-09-28)

| # | Item | Verified by |
|---|---|---|
| 1 | `src/features/build.py` — the 39 features of §5, stateless, §4.2 contract as its docstring | 39 columns in a fixed order; raises on a denylist column or a missing input |
| 2 | `src/features/artifact.py` — `PreprocessingArtifact` with fit / transform / save / load | frozen dataclass, `fit` a classmethod, **no `fit_transform`**; 454 KB on disk |
| 3 | `tests/test_features_build.py` | 51 tests, 43 needing no database |
| 4 | `make features` | fits on 36,174 v1 rows, transforms 96,203, ~4 s |

**Artifact contents, per §6.2:** 30 numeric medians · 6 frozen category level sets with
`__missing__` and `__unknown__` kept distinct · winsorisation bounds · the six latest snapshot
tables · zip and state centroids · the ordered 39-name feature list · train cutoff
`2017-12-31 23:29:31` · schema hash `03522e4226b5dc31`. Calibrator and threshold are `None` until
Phase 5 attaches them through `with_decision`, which returns a **new** artifact so one already
logged to MLflow cannot drift.

**Category levels, fitted on training rows only:** `product_category` capped to the 30 most
frequent, covering **95.80%** of training rows, with the remaining 42 collapsing to
`__unknown__`. `customer_region` uses all five IBGE macro-regions rather than whatever training
contained, because the set is fixed and a window with no northern customer must still be able to
encode one. `seller_state` has fewer levels than `customer_state` — sellers are concentrated, so
an order from an unusual seller state legitimately encodes as unknown.

**Load-time guards:** `load()` refuses an artifact whose version differs, and refuses one whose
input-schema hash no longer matches the running code. A hash mismatch is exactly the train/serve
skew the class exists to prevent, so it fails rather than serving one wrong prediction.

### Phase 3 part C — Time splits and the evaluation lock — ✅ COMPLETE (2026-09-28)

| # | Item | Verified by |
|---|---|---|
| 1 | `src/splits.py` — sole reader of `configs/splits.yaml`, per-version windows | both versions' windows returned and proven disjoint |
| 2 | The §6.6 assertion `max(fit) < min(calibrate) < evaluation.start` | holds for v1 and v2 on real data; runs on every `make features` |
| 3 | Runtime lock on the 2018-05 → 08 window (§4.4, §18 A3) | locked by default, refuses unauthorised callers, re-locks on exception, refuses nesting |
| 4 | `tests/test_splits.py` | 38 tests, 34 needing no database |

**Measured window sizes:** v1 fit 36,174 / calibrate 13,624 · v2 fit 49,798 / calibrate 13,801 ·
promotion evaluation 25,352. No order appears in both a fit and a calibrate window, and no
training row of either version reaches the evaluation window.

**`VersionSplits` refuses to exist with overlapping windows**, so §18 A2's defect — v2 training
on its own validation window — cannot be reintroduced by editing the config. The config would
load and then immediately raise.

### Phase 4 — Baselines and single models — ✅ COMPLETE (2026-09-28)

| # | Item | Verified by |
|---|---|---|
| 1 | `src/evaluation/metrics.py` — PR-AUC, ROC-AUC, Brier, recall/precision@capacity, lift | 33 tests, every expected value **derived by hand in a comment**, not copied from a run |
| 2 | `src/training/baselines.py` — majority class + logistic regression | majority scores the base rate and ROC-AUC exactly 0.5; logistic 0.1357 |
| 3 | `src/training/tune.py` — Optuna over `TimeSeriesSplit(4)`, spaces from `configs/models.yaml` | 40/40, 40/40, 20/20 trials, no timeouts |
| 4 | `tests/test_training_cv.py` | 13 tests; every fold's max train date < min validation date |
| 5 | `tests/fixtures/generate_synthetic.py` + smoke tests | 17 tests, LightGBM smoke train in **3.9 s** against a 30 s requirement |
| 6 | `configs/models.yaml`, `make baselines`, `make tune` | run repeatedly |

**The CV guard is the part worth re-reading.** `TimeSeriesSplit` splits by *position*, not by
timestamp — it has no idea what the dates are. Hand it unsorted rows and it produces folds that
look perfectly ordered, leak badly, and raise nothing. So `assert_time_sorted` is called before
every study, and one test *demonstrates* the failure: on shuffled rows the splitter still reports
ordered index ranges while the dates inside them interleave.

**Two inspection tests enforce §6.6 by parsing the source with `ast`:** no module in `src/`
imports `KFold`, `StratifiedKFold`, `ShuffleSplit` or `StratifiedShuffleSplit`, and none uses
`train_test_split` — which shuffles by default and is the same mistake wearing a different name.

**The synthetic fixture imports its schema from `src/features/build.py`** rather than restating
it, so it cannot drift from the real 39-column contract. It is what CI will run from Phase 10,
since the real dataset is 121 MB and CC BY-NC-SA and will never be in a workflow.

---

## Remaining work

| Phase | Title | Est. | Difficulty | Status |
|---|---|---|---|---|
| 0 | Skeleton and environment | 1.5 h | easy | ✅ **complete** · CI green |
| 1 | Raw load into Postgres | 2.5 h | easy-med | ✅ **complete** · 33 tests |
| 2 | Transform, validate, analytical table | 4 h | medium | ✅ **complete** · 80 tests |
| 3 | As-of aggregates and feature assembly | 5 h | **hardest** | ✅ **complete** · all 3 parts |
| 4 | Baselines and single models | 4 h | medium | ✅ **complete** · LightGBM leads at 0.1528 |
| 5 | Ensemble, calibration, threshold | 4 h | medium | **next** · ⚠️ D20 base-rate shift, D26 fold lift, D27 blend may not pay · fixes whether CatBoost must re-enter `requirements-api.txt` (D14) |
| 6 | MLflow, pyfunc wrapper, registry | 4 h | medium | |
| 7 | FastAPI and the parity test | 4 h | medium | ⚠️ read §18 A3, A6 first |
| 8 | Docker and Compose | 3 h | medium | ⚠️ read §18 A5 first · validate `mlflow-skinny` **and** the 305 MB CUDA dependency (D14) |
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
