# PROGRESS — delivery-delay-platform

Living status file. **Update it at the end of every completed task**, in the same commit as
that task. Source of truth for *what* to build is `PLAN.md`; this file only tracks *where we
are*. Decisions and their reasons live in `DECISIONS.md`.

- **Plan version:** v1 (2026-09-28) + amendments §18 · **Pre-flight completed:** 2026-09-28
- **Current phase:** Phase 0 — Skeleton and environment (PLAN §13, Phase 0) · **not started**
- **Phase 0 progress:** 0 / 8
- **Overall:** 0 / 12 phases · pre-flight and environment complete (not counted as a phase)
- **Estimated remaining:** ~41 h of build work (PLAN §3 budget: 36-42 h)

---

## Currently working on

**Nothing in flight.** Pre-flight is complete and Phase 0 is cleared to start, pending the
user's go-ahead.

Everything Phase 0 needs exists: a working venv on Python 3.12, the full dependency stack
verified as resolvable (117 packages, binary wheels only, zero source builds), Docker with
the required registries reachable, the dataset on disk, a private GitHub remote wired over
SSH, and three ports confirmed free.

---

## Next up (in order)

**Phase 0 — Skeleton and environment** (PLAN §13 Phase 0, ~1.5 h, easy).
Per the plan's own prompt, explain the Compose healthcheck choices *before* writing
`docker-compose.yml`.

1. The §12 directory tree, with empty `__init__.py` files
2. `pyproject.toml` — ruff (line-length 100), black, pytest (`testpaths=tests`, `--cov=src`)
3. `.gitignore` (full version, replacing the interim one), `.dockerignore`,
   `.gitattributes` (LF), `.env.example`
4. `docker-compose.yml` — **exactly two services**: `postgres` (16-alpine, named volume,
   `pg_isready` healthcheck) and `mlflow` (sqlite backend store, bind-mounted artifacts,
   HTTP healthcheck). **No `api` service yet.** Explicit `name: delay-prediction`.
5. `src/db.py` — SQLAlchemy engine factory reading from pydantic-settings
6. `configs/base.yaml` and `configs/splits.yaml` — **keyed per version** (§18 A2)
7. `.github/workflows/ci.yml` — checkout, Python 3.12, install `requirements-dev.txt`,
   `ruff check`, `black --check`. No tests yet.
8. `requirements.txt` / `requirements-dev.txt` / `requirements-api.txt`, unpinned;
   freeze to `requirements.lock.txt` after first install

**Verify:** `docker compose up postgres mlflow` -> MLflow UI on `localhost:5000`, psql
connects. CI green on the first push.
**Completion:** skeleton committed, green CI, two healthy containers.

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

---

## Remaining work

| Phase | Title | Est. | Difficulty | Status |
|---|---|---|---|---|
| 0 | Skeleton and environment | 1.5 h | easy | **next** |
| 1 | Raw load into Postgres | 2.5 h | easy-med | |
| 2 | Transform, validate, analytical table | 4 h | medium | |
| 3 | As-of aggregates and feature assembly | 5 h | **hardest** | ⚠️ read §18 A1 first |
| 4 | Baselines and single models | 4 h | medium | |
| 5 | Ensemble, calibration, threshold | 4 h | medium | |
| 6 | MLflow, pyfunc wrapper, registry | 4 h | medium | |
| 7 | FastAPI and the parity test | 4 h | medium | ⚠️ read §18 A3, A6 first |
| 8 | Docker and Compose | 3 h | medium | ⚠️ read §18 A5 first |
| 9 | Retraining lifecycle and promotion gate | 3.5 h | medium | ⚠️ read §18 A2 first |
| 10 | Full CI/CD | 3 h | medium | |
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
5. Any non-obvious decision made along the way goes in `DECISIONS.md`, not here.
