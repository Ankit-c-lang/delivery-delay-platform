# PROGRESS — delivery-delay-platform

Living status file. **Update it at the end of every completed task**, in the same commit as
that task. Source of truth for *what* to build is `PLAN.md`; this file only tracks *where we
are*. Decisions and their reasons live in `DECISIONS.md`.

- **Plan version:** v1 (2026-09-28) + amendments §18 · **Pre-flight completed:** 2026-09-28
- **Current phase:** Phase 0 — Skeleton and environment (PLAN §13, Phase 0) · **8 / 8 items
  done, local verification green, CI unverified until first push**
- **Overall:** 1 / 12 phases · pre-flight and environment complete (not counted as a phase)
- **Estimated remaining:** ~39.5 h of build work (PLAN §3 budget: 36-42 h)

---

## Currently working on

**Nothing in flight.** Phase 0 is built and verified locally.

All eight prompt items exist, `ruff check` and `black --check` pass on 8 Python files, and
both containers reached `healthy` on the first `up` — 21 seconds from create to healthy, no
retries burnt. `src/db.py` was exercised against the live database, not just linted: it
connects as `delay` to Postgres 16.15 and masks the password in both `repr()` and the log
line. `configs/splits.yaml` was checked programmatically — for both versions the fit,
calibrate and promotion-evaluation windows are pairwise disjoint and `aggregates_through`
matches the calibrate end (§18 A2).

**One deliberate scope addition, flagged for approval:** `Makefile` and
`.pre-commit-config.yaml`. Both are named in `PLAN.md` (§12 tree and the Phase 0 *Tasks*
line respectively) but neither appears in Phase 0's numbered prompt. See `DECISIONS.md` D15.

**Partial dependency install.** Only `requirements-dev.txt` plus five core packages
(`sqlalchemy`, `psycopg2-binary`, `pydantic`, `pydantic-settings`, `pyyaml`) are installed —
enough to lint and to actually run `src/db.py`. The full ~3 GB training stack waits for
Phase 4, and `requirements.lock.txt` waits with it, since a lockfile from a partial install
would be a lie.

### Not yet verified

- **CI has never run.** Phase 0's completion criterion says "green CI on the first push", and
  nothing has been pushed. `.github/workflows/ci.yml` is untested; the local `pre-commit` run
  uses the same ruff 0.16.9 and black 26.5.1 that CI will resolve, so the lint result should
  hold, but the workflow syntax itself is unproven.
- `requirements.txt` and `requirements-api.txt` have never been installed, so neither is
  proven resolvable as written. `requirements-api.txt` in particular names `mlflow-skinny`
  (D14) which Phase 8 must validate against a real pyfunc load.

---

## Next up (in order)

**Immediately:** push `main` and confirm the CI run goes green. That closes Phase 0's last
criterion. If the workflow fails it is a Phase 0 fix, not a Phase 1 problem.

**Then Phase 1 — Raw load into Postgres** (PLAN §13 Phase 1, ~2.5 h, easy-medium).

1. `src/etl/load_raw.py` — the 9 Olist CSVs into the `raw` schema via `COPY FROM STDIN`
2. `src/etl/schema.py` — Pandera contracts for each raw table
3. A row-count reconciliation report, CSV against table
4. `tests/test_etl_load_raw.py`

**Carry into Phase 1:** expected row counts must come from a CSV parser, never `wc -l`.
`olist_order_reviews_dataset.csv` is **99,224** rows (not 104,719 — embedded newlines in
quoted review text) and `product_category_name_translation.csv` is **71** (not 70 — no
trailing newline). `csv.field_size_limit` may need raising for the reviews file. See
`DECISIONS.md` D7. Getting this wrong makes the reconciliation test fail spuriously, and the
tempting "fix" is to break the loader to match a wrong number.

**Start of session:** `make up` (both services), then read `CLAUDE.md` and `PLAN.md` §13
Phase 1. Credentials are already in `.env`.

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

### Phase 0 — Skeleton and environment — ✅ built, CI pending (2026-09-28)

All eight items from the Phase 0 prompt:

| # | Item | Verified by |
|---|---|---|
| 1 | §12 directory tree, empty `__init__.py`, `.gitkeep` in the empty leaves | `find` listing; all 7 packages importable paths exist |
| 2 | `pyproject.toml` — ruff (line-length 100), black, pytest (`testpaths`, `--cov=src`) | `ruff check .` and `black --check .` both clean |
| 3 | Full `.gitignore` (replaced the interim one), `.dockerignore`, `.gitattributes` (LF), `.env.example` | `git check-ignore -v` on `data/`, `.env`, `mlflow/`, `mlartifacts/`; negations for `.env.example` and `reports/.gitkeep` confirmed; nothing from `data/` staged |
| 4 | `docker-compose.yml` — `postgres` + `mlflow` only, explicit `name: delay-prediction` | `docker compose config --quiet`; `up -d --wait` reached `healthy` for both in 21 s |
| 5 | `src/db.py` — SQLAlchemy engine factory over pydantic-settings | live query: `server_version` 16.15, `current_user` `delay`, password masked in `safe_url` and `repr` |
| 6 | `configs/base.yaml`, `configs/splits.yaml` keyed **per version** (§18 A2) | both parse; fit/calibrate/eval windows proven pairwise disjoint for v1 and v2 |
| 7 | `.github/workflows/ci.yml` — 3.12, `requirements-dev.txt`, ruff + black, no tests | YAML parses (`check-yaml`); **workflow itself unrun** |
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

---

## Remaining work

| Phase | Title | Est. | Difficulty | Status |
|---|---|---|---|---|
| 0 | Skeleton and environment | 1.5 h | easy | ✅ built · CI unverified |
| 1 | Raw load into Postgres | 2.5 h | easy-med | **next** · use parser row counts (D7) |
| 2 | Transform, validate, analytical table | 4 h | medium | |
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
