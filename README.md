# delivery-delay-platform

[![CI](https://github.com/Ankit-c-lang/delivery-delay-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/Ankit-c-lang/delivery-delay-platform/actions/workflows/ci.yml)
[![Build](https://github.com/Ankit-c-lang/delivery-delay-platform/actions/workflows/build.yml/badge.svg)](https://github.com/Ankit-c-lang/delivery-delay-platform/actions/workflows/build.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Predicts at checkout whether a Brazilian e-commerce order ([Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce))
will be delivered **after its promised date**.

> **The purpose of this project is the MLOps lifecycle, not the model.** Postgres feature warehouse →
> leakage-safe as-of aggregates → LightGBM/XGBoost/CatBoost → MLflow tracking and registry → a
> promotion gate that can refuse → a containerized FastAPI service.

---

## Start here: the model is weak, and the pipeline is the point

| | calibration window (2018-01 → 02) | **held-out window (2018-05 → 08)** |
|---|---:|---:|
| base rate | ~9.75% | **4.40%** |
| PR-AUC | 0.21585 | **0.05983** |
| **lift over base rate** | ~2.2× | **1.36×** |
| precision at the operating threshold | 24.8% | **6.2%** at a 16.3% flag rate |

Lift already normalises for the base-rate difference, so the fall from 2.2× to 1.36× is **real
degradation on genuinely later data**. Quoting 0.216 without naming its window would be the one
dishonest number available here.

**And it gets worse — a logistic regression beats it on that window** ([D45](DECISIONS.md)):

| model (holdout, 25,352 rows, identical features) | PR-AUC | lift | ROC-AUC | Brier |
|---|---:|---:|---:|---:|
| majority class (no skill) | 0.04398 | 1.00× | 0.500 | 0.042256 |
| **logistic regression** | **0.09163** | **2.08×** | **0.668** | 0.064591 |
| champion — tuned LightGBM | 0.05983 | 1.36× | 0.612 | **0.049307** |

The gap is 0.03181, **six times** the 0.005 reproducibility floor, so it is not noise. In-window the
ordering reverses (CV: LightGBM 0.15349 against logistic 0.13545) — the GBDTs win where they were
tuned and lose where it counts. One nuance travels with it: the champion is **better calibrated**
(Brier 0.049 vs 0.065), because its probabilities pass through isotonic calibration and the
logistic's are raw. Ranking and calibration are different things and the two models are better at
different ones.

Reproduce: `make holdout-reference`.

---

## Architecture

```
                    ┌──────────────────────────────────────────────┐
  9 Olist CSVs ───► │ raw.*            1,550,922 rows              │
  (never committed) │   COPY FROM STDIN, header verified first     │
                    └───────────────────┬──────────────────────────┘
                                        ▼
                    ┌──────────────────────────────────────────────┐
                    │ features.orders_analytical   96,203 rows     │  Pandera strict=True
                    │ features.order_outcomes      (quarantined)   │  ◄── the ONLY place
                    └───────────────────┬──────────────────────────┘      the outcome date lives
                                        ▼
                    ┌──────────────────────────────────────────────┐
                    │ as-of monthly snapshots      34,427 rows     │  delivered < M, never <=
                    │ seller · route · category  + fallbacks       │  (§18 A1)
                    └───────────────────┬──────────────────────────┘
                                        ▼
                    ┌──────────────────────────────────────────────┐
                    │ PreprocessingArtifact  ── 38 features        │  medians, frozen levels,
                    │   fit() is a classmethod; no fit_transform   │  winsor bounds, snapshots,
                    └───────────────────┬──────────────────────────┘  calibrator, threshold
                                        ▼
    ┌───────────────────────────────────┴───────────────────────────────┐
    ▼                                                                   ▼
┌─────────────────────────────┐                        ┌────────────────────────────────┐
│ train.py                    │                        │ promote.py  ── the gate        │
│  4 nested MLflow runs       │                        │  scores @champion vs           │
│  + equivalence-band select  │──► registry ──────────►│  @challenger on IDENTICAL rows │
│  pyfunc = artifact + model  │    @champion           │  PR-AUC · Brier · p95 · schema │
│         + calibrator + thr  │    @challenger         │  → reports/promotions.md       │
└─────────────────────────────┘                        └────────────────────────────────┘
                                        │
                                        ▼
                    ┌──────────────────────────────────────────────┐
                    │ FastAPI  ── resolves models:/…@champion      │  NO preprocessing here
                    │ 1.03 GB image, 65 packages, uid 1000         │  (enforced by AST test)
                    └──────────────────────────────────────────────┘
```

**The 2018-05 → 08 window is locked at runtime.** It raises unless unlocked by the promotion gate,
and the guard protects the *window definition*, not just the loading code — because the realistic
leak is a stray `orders[orders.purchase >= "2018-05-01"]` in a notebook, not a rogue import.

---

## Results chain

**Cross-validated on v1's fit window** (36,174 rows, `TimeSeriesSplit(4)`, PR-AUC):

| | PR-AUC | lift over base rate |
|---|---:|---:|
| majority class | 0.06559 | 1.00× |
| logistic regression | 0.13545 | 2.07× |
| XGBoost | 0.14666 | 2.24× |
| CatBoost | 0.15209 | 2.32× |
| **LightGBM** | **0.15349** | **2.34×** |

The lift column divides by the **majority-class PR-AUC (0.06559)**, not by the window's overall
5.85% positive rate. A no-skill classifier's PR-AUC equals the base rate *of the folds it is scored
on*, and `TimeSeriesSplit` folds here run 2.8%, 4.2%, 9.6%, 9.7% (D26), averaging 0.0656 — which
is the majority-class figure. So the majority-class row is
the correct 1.00× reference and the window rate is not.

**On the calibration window**, the blend scored 0.21924 against the best single model's 0.21604 —
a gain of **+0.00320**, which does **not** clear the 0.005 bar in `configs/base.yaml`. So a single
model ships. Three GBDTs on identical tabular features produce highly correlated predictions; a
small gain was the expected outcome, not a failure, and the bar exists so that conclusion is a
threshold rather than a mood.

**Which single model** is decided by an equivalence band ([D41](DECISIONS.md)): highest PR-AUC, then
cheapest to serve among models within **0.005** of it — the floor D35 measured for this pipeline's
own reproducibility. LightGBM ships at a cost of **0.00020 PR-AUC** and a saving of **259 MB**
against CatBoost.

**The promotion gate refused the retrain** ([`reports/promotions.md`](reports/promotions.md)). v2,
trained on two months more data, scored 1.30× against the champion's 1.36× — inside the band, so a
**tie rather than a regression**, and a tie leaves the incumbent in place. Nothing was retuned.

---

## Latency

| | single p50 | batch-100 p50 | per record |
|---|---:|---:|---:|
| host uvicorn | 52.64 ms | 54.91 ms | 0.549 ms |
| container, published port | 53.41 ms | 88.21 ms | 0.882 ms |
| container, from inside | 58.08 ms | 62.81 ms | 0.628 ms |

Almost all of it is **fixed overhead in the as-of snapshot join**, not the model, which is
sub-millisecond: a 1-row `transform` costs ~45 ms against ~69 ms for 1,000 rows. The 25 ms gap on
batch-100 through the published port is Docker's port forwarding for a ~60 KB body — measured by
benchmarking from inside the container, not guessed.

---

## Reproduce it

```bash
cp .env.example .env          # then edit POSTGRES_PASSWORD
make bootstrap                # REQUIRED first: load, train, register a champion
make build && make api-up     # containerized API on http://127.0.0.1:8001/docs
```

`make bootstrap` is not optional on a clean machine: the API resolves
`models:/delivery_delay_classifier@champion` at startup, and a fresh registry has no champion, so it
correctly reports itself unhealthy. `docker compose up` is the steady-state story *after* that.

`make help` lists every target. The dataset is not committed (121 MB, CC BY-NC-SA); `make check-data`
says what it expects.

---

## Limitations

Written plainly, because a limitations section is the clearest signal that the author understands
their own system.

1. **The model is weak on future data, and a logistic regression beats it there** (D45, above). The
   single most important fact in this repository, and it is above the fold rather than here.
2. **The 2018-05 → 08 window is a promotion evaluation set, not a test set** (§18 A3). Two gate runs
   have read it, so it has influenced selection — weakly, but it has. With two evaluations the bias
   is negligible; calling it a "test set" would still be wrong. A third period was considered and
   rejected: it would cost evaluation-set size to remove a bias two looks cannot induce.
3. **Training and serving use different as-of snapshots by design** (§18 A6). Training joins each
   order to its own purchase-month snapshot; serving uses the single latest bundled one, because a
   live request has no history table to join to. Parity is therefore scoped to *one fixed artifact*
   — same raw record + same artifact → same probability within 1e-6. Feed a historical order and the
   two paths legitimately disagree. That is a genuine distribution shift, not an implementation
   detail papered over.
4. **The cost ratio is an assumption, never a measurement.** The 5:1 FN:FP ratio behind the 0.177
   threshold came from reasoning about support contacts and review damage, not from Olist's books.
   Change the premise and the threshold moves.
5. **The base rate is non-stationary** (D20): ~9.75% where the threshold was chosen, 4.40% where it
   is judged. A calibrator fitted on the former over-predicts on the latter. The gate's Brier
   tolerance is deliberately generous for this reason, and both models meet the same shift so the
   *comparison* stays fair.
6. **The history features are weak** (D22). `seller_late_rate_hist` is worth +0.0197 PR-AUC measured
   correctly, against +0.0398 measured leakily — a 2.0× inflation. If it ever dominates a SHAP
   ranking, that is a leakage alarm rather than a win.
7. **Promotion is not purely a registry operation.** A challenger shipping a different library needs
   the image rebuilt — and possibly *edited*: swapping CatBoost for LightGBM required `libgomp1`, an
   apt package no requirements file could predict (D41). The gate checks the alias half and not the
   image half.
8. **There is no hosted environment.** CD here means a versioned, smoke-tested artifact in GHCR plus
   a one-command Compose deploy on a machine you own. Nothing in this repository deploys to a cloud,
   and a CV built on it should not say otherwise.
9. **One dataset, one 20-month span, one country.** Every number here is conditional on that. The
   thing that would settle D45 — more evaluation windows — is precisely what this data cannot supply.

---

## Documentation

| | |
|---|---|
| [`PLAN.md`](PLAN.md) | the full design. **§18 holds eight binding amendments**; where §18 and the body disagree, §18 wins |
| [`DECISIONS.md`](DECISIONS.md) | 45 decisions with their reasons, measurements, and what was rejected |
| [`PROGRESS.md`](PROGRESS.md) | what is built, what it cost, and what went wrong |
| [`reports/interview_notes.md`](reports/interview_notes.md) | the nine questions, answered |
| [`CLAUDE.md`](CLAUDE.md) | the 17 invariants, enforced by tests rather than trusted |

Five decisions are worth reading on their own, because each was a bug that passed its own tests for
weeks: **D33** (a config value validated but enforced by nothing), **D35** (unpinned thread counts —
CPU availability was choosing which model shipped), **D38** (a dependency list written from what the
code looked like it needed), **D41** (one decision expressed in two places), and **D44** (a test
suite that had only ever been run through one entry point).

## Licence

Code under this repository's licence; the Olist dataset is CC BY-NC-SA 4.0 and is not redistributed
here.
