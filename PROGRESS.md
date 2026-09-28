# PROGRESS — delivery-delay-platform

Living status file. **Update it at the end of every completed task**, in the same commit as
that task. Source of truth for *what* to build is `PLAN.md`; this file only tracks *where we
are*. Decisions and their reasons live in `DECISIONS.md`.

- **Plan version:** v1 (2026-09-28) + amendments §18 · **Pre-flight completed:** 2026-09-28
- **Current phase:** Phase 1 — Raw load into Postgres · **✅ COMPLETE** — 9 tables,
  1,550,922 rows, 33 tests green
- **Overall:** 2 / 12 phases · pre-flight and environment complete (not counted as a phase)
- **Estimated remaining:** ~37 h of build work (PLAN §3 budget: 36-42 h)

---

## Currently working on

**Nothing in flight.** Phase 1 is complete and verified.

All nine CSVs load into the `raw` schema in **4.1 s**, 1,550,922 rows, and every table
reconciles exactly against a CSV-parser count. `make load-raw` run three times in a row gives
identical counts. 33 tests pass (11 of them pure unit tests that need neither Postgres nor the
dataset, so they will run in CI from Phase 10). Coverage 88% overall, 85% on `load_raw.py`;
the gap is the CLI entry point and the `--recreate` branch, both exercised by hand.

**One test of mine was wrong and got fixed, not worked around.** The `wc -l` trap test used
`sum(1 for _ in open(path))`, which yields the final newline-less line as a line and so
reported 71 where `wc -l` reports 70. Python's line iterator and `wc -l` genuinely disagree on
files with no trailing newline — and all nine Olist files lack one. Replaced with an explicit
newline-byte count plus a unit test pinning that helper's behaviour, so the two row-count
tests keep comparing like with like.

---

## Data-quality findings, for Phase 2

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

**Phase 2 — Transform, validate, analytical table** (PLAN §13 Phase 2, ~4 h, medium).

Read `PLAN.md` §4.2, §4.3 and §13 Phase 2 first. §4.2 and §4.5 changed materially in the
§18 amendment.

**Carry into Phase 2:** the findings table above, and in particular:
- `is_late` compares at **date** granularity on both sides (§4.3). The positive rate must
  land in 5-10%; outside that band means the granularity bug, not a discovery.
- The population filter needs `order_status = 'delivered'` **and**
  `order_delivered_customer_date IS NOT NULL` — 8 rows satisfy the first but not the second,
  and 6 satisfy the second but not the first.
- Log how many rows each filter drops and keep the funnel; the README wants it.
- `src/etl/schema.py` (Pandera contracts) was listed under Phase 1 in an earlier reading of
  the plan but belongs with the transform, which is where validation has something to assert.
  It is **not** yet written.

**Deferred from Phase 1, deliberately:** `src/etl/geolocation.py` (zip prefix → centroid) is
a Phase 2 item per §12. `raw.geolocation` has no index; a 1 M-row sequential scan is cheap,
so Phase 2 should add one only if it measures a need.

**Start of session:** `make up`, `make check-data`, then `make load-raw` if the volume is
fresh. Credentials are already in `.env`.

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

---

## Remaining work

| Phase | Title | Est. | Difficulty | Status |
|---|---|---|---|---|
| 0 | Skeleton and environment | 1.5 h | easy | ✅ **complete** · CI green |
| 1 | Raw load into Postgres | 2.5 h | easy-med | ✅ **complete** · 33 tests |
| 2 | Transform, validate, analytical table | 4 h | medium | **next** · §4.3 date granularity; Pandera contracts land here |
| 3 | As-of aggregates and feature assembly | 5 h | **hardest** | ⚠️ read §18 A1 first |
| 4 | Baselines and single models | 4 h | medium | full dependency install lands here |
| 5 | Ensemble, calibration, threshold | 4 h | medium | fixes whether CatBoost must re-enter `requirements-api.txt` (D14) |
| 6 | MLflow, pyfunc wrapper, registry | 4 h | medium | |
| 7 | FastAPI and the parity test | 4 h | medium | ⚠️ read §18 A3, A6 first |
| 8 | Docker and Compose | 3 h | medium | ⚠️ read §18 A5 first · validate `mlflow-skinny` (D14) |
| 9 | Retraining lifecycle and promotion gate | 3.5 h | medium | ⚠️ read §18 A2 first |
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
