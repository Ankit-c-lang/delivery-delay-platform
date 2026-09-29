# delivery-delay-platform

[![CI](https://github.com/Ankit-c-lang/delivery-delay-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/Ankit-c-lang/delivery-delay-platform/actions/workflows/ci.yml)
[![Build](https://github.com/Ankit-c-lang/delivery-delay-platform/actions/workflows/build.yml/badge.svg)](https://github.com/Ankit-c-lang/delivery-delay-platform/actions/workflows/build.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Predicts at checkout whether a Brazilian e-commerce order ([Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce))
will be delivered **after its promised date**.

> **The purpose of this project is the MLOps lifecycle, not the model.** Postgres feature warehouse
> → leakage-safe as-of aggregates → LightGBM/XGBoost/CatBoost → MLflow tracking and registry →
> a promotion gate → a containerized FastAPI service.

## The honest headline

| | calibration window (2018-01 → 02) | **held-out window (2018-05 → 08)** |
|---|---:|---:|
| base rate | ~9.75% | **4.40%** |
| PR-AUC | 0.21585 | **0.05983** |
| **lift over base rate** | ~2.2× | **1.36×** |
| precision at the operating threshold | 24.8% | **6.2%** at a 16.3% flag rate |

**Read the right-hand column.** Lift already normalises for the base-rate difference, so the fall
from 2.2× to 1.36× is real degradation on genuinely later data — not arithmetic. **This model is weak
on future orders**, and quoting 0.216 without naming its window would be the one dishonest number
available here. The pipeline around it is the point; see [`DECISIONS.md`](DECISIONS.md) D42.

## What it does, and what it refuses to do

- **Leakage-safe as-of aggregates.** A snapshot for month *M* uses only orders whose delivery
  outcome was known before *M*. Filtering on purchase time instead inflates a seller's late-rate
  feature by 2.0× (D22), because late orders take ~3× longer to resolve.
- **A runtime lock on the evaluation window.** 2018-05 → 08 raises unless unlocked by the promotion
  gate; the guard protects the *window definition*, because the realistic leak is a stray filter in
  a notebook, not a rogue import.
- **All preprocessing inside the MLflow pyfunc.** One artifact carries the imputation values, frozen
  category levels, as-of snapshots, calibrator and threshold, so a model version *is* its
  preprocessing. Parity is asserted at 1e-6, and the container's prediction is **bitwise identical**
  to the development environment's.
- **A promotion gate that refuses.** It ran for real and declined to promote a retrain
  ([`reports/promotions.md`](reports/promotions.md)). A gate that always passes is decoration.

## Quickstart

```bash
cp .env.example .env          # then edit POSTGRES_PASSWORD
make bootstrap                # REQUIRED first: loads data, trains, registers a champion
make build && make api-up     # the containerized API on http://127.0.0.1:8001/docs
```

`make bootstrap` is not optional on a clean machine: the API resolves
`models:/delivery_delay_classifier@champion` at startup, and a fresh registry has no champion, so it
will correctly report itself unhealthy. `docker compose up` is the steady-state story *after* that.

`make help` lists every target. The dataset is not committed (121 MB, CC BY-NC-SA); `make check-data`
tells you what it expects.

## Measured latency

| | single p50 | batch-100 p50 | per record |
|---|---:|---:|---:|
| host uvicorn | 52.64 ms | 54.91 ms | 0.549 ms |
| container, published port | 53.41 ms | 88.21 ms | 0.882 ms |
| container, from inside | 58.08 ms | 62.81 ms | 0.628 ms |

Nearly all of it is fixed overhead in the as-of snapshot join, not the model, which is
sub-millisecond. The 25 ms gap on batch-100 through the published port is Docker's port forwarding
for a ~60 KB body — measured, not guessed.

## CI is not CD

- **CI** (`ci.yml`): ruff, black, then pytest — including the **real ETL against synthetic CSVs in a
  Postgres service container**, because CI must never need the 121 MB non-commercial dataset.
- **CD** (`build.yml`): build the image, smoke-test it against a fixture model, and publish to GHCR
  tagged `:latest` and `:{sha}`. The smoke test compares the container's probability to a reference
  **bitwise** — a build step that only checks the image exists proves nothing.

**There is no hosted environment.** CD here means *a versioned, smoke-tested artifact in a registry*
plus a one-command Compose deploy on a machine you own. Anything more would be a claim this
repository does not support.

## Documentation

| | |
|---|---|
| [`PLAN.md`](PLAN.md) | the full design. **§18 holds eight binding amendments**; where §18 and the body disagree, §18 wins |
| [`DECISIONS.md`](DECISIONS.md) | 42 decisions with their reasons, measurements, and what was rejected |
| [`PROGRESS.md`](PROGRESS.md) | what is built, what it cost, and what went wrong |
| [`CLAUDE.md`](CLAUDE.md) | the invariants, which are enforced by tests rather than trusted |

Three of those decisions are worth reading on their own, because each was a bug that passed its own
tests for weeks: **D33** (a config value validated but enforced by nothing), **D35** (unpinned thread
counts, which meant CPU availability chose which model shipped), and **D38** (a dependency list
written from what the code looked like it needed rather than from a measurement).

## Licence

Code under this repository's licence; the Olist dataset is CC BY-NC-SA 4.0 and is not redistributed
here.
