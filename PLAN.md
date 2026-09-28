# Delivery Delay Prediction Platform — Build Plan (v1)

**One-line description:** An end-to-end MLOps pipeline that predicts, at checkout time, whether a Brazilian e-commerce order will be delivered after its promised date — with a Postgres-backed feature warehouse, MLflow experiment tracking and model registry, a time-based retraining lifecycle with an automated promotion gate, and a containerized FastAPI serving layer published by CI.

**Project name:** `delivery-delay-platform`
(Replacing "AI Enterprise Suite" — see §0.1.)

---

## 0. Framing decisions

### 0.1 Why the rename

"AI Enterprise Suite" tells a reader nothing, implies multiple applications when you're building one service, and reads as a buzzword title on a CV that is otherwise trying to signal engineering. "Delivery Delay Prediction Platform" states the problem, the output, and the scope. Use the long form on GitHub, and on the CV lead with the capability rather than the name.

### 0.2 What this project owns, and what it must not duplicate

Your three-project set has to cover distinct ground. Enforce this boundary throughout:

| | This project | Fraud Detection project |
|---|---|---|
| **Owns** | MLflow tracking + registry, reproducibility, retraining lifecycle, CI/CD, containerized serving, train/serve parity | Streaming ingestion, graph features, unsupervised anomaly detection, real-time latency |
| **Domain** | Logistics / e-commerce fulfilment | Fintech / payments |
| **Store** | PostgreSQL | Redis Streams + DuckDB |
| **Interview topic** | "How does a model get to production and get replaced?" | "How do you design a real-time ML system?" |

Both use gradient boosting, SHAP, FastAPI and Docker. That overlap is fine and unavoidable — those are baseline tools. The *narrative* must not overlap. When you talk about this project, the story is the **lifecycle**, not the model.

### 0.3 The spine

The one chain that has to be real, because everything else is scaffolding around it:

```
training run
  → model + fitted preprocessing artifact + feature schema logged together to MLflow
  → registered as a new version of `delivery_delay_classifier`
  → promotion gate compares challenger vs champion on a frozen holdout
  → winner gets the `champion` alias
  → FastAPI resolves models:/delivery_delay_classifier@champion at startup
  → a parity test proves the API's predictions match the training pipeline's to 1e-6
    for the same raw record and the same artifact (§18 A6)
```

If any link in that chain is faked — most student versions of this project unpickle a `model.pkl` committed to the repo — the project collapses under one question. Build this chain first in skeleton form, then make each link good.

---

## 1. Scope (Step 2)

### MUST HAVE

1. Raw Olist CSVs loaded into PostgreSQL (running under Docker Compose) as a `raw` schema.
2. ETL producing a single order-level analytical table in a `features` schema, with Pandera validation that can fail the run.
3. As-of monthly aggregate tables for seller / route / category history (§4.5) — this is the leakage-safe design and the main data-engineering content.
4. ~38 engineered features across 8 named families (§5).
5. Strict time-based splits with a warm-up window, plus `TimeSeriesSplit` cross-validation inside the training window.
6. Two baselines: the trivial majority-class rule, and logistic regression.
7. LightGBM, XGBoost, CatBoost — each tuned with Optuna under a wall-clock budget.
8. A probability-blend ensemble with an **honestly reported** delta over the best single model.
9. Probability calibration, and a threshold chosen from an explicit cost model, not `0.5`.
10. PR-AUC as the primary metric (positives are ~6–8% of orders), ROC-AUC secondary, plus recall at a fixed operational review capacity.
11. SHAP analysis logged to MLflow as artifacts, computed offline.
12. MLflow with a SQLite backend store and local filesystem artifacts: params, metrics, plots, model signature, input example, and the preprocessing artifact.
13. Model Registry with `champion` / `challenger` aliases.
14. **Retraining lifecycle**: v1 trained on 2017 data, v2 on 2017 + early-2018, both scored on the same frozen 2018-05→08 promotion evaluation set, with a scripted promotion gate.
15. FastAPI service: `/health`, `/predict`, `/predict/batch`, `/model/info`. Model loaded once at startup, resolved from the registry.
16. **Train/serve parity test** — the highest-value test in the repo.
17. Multi-stage Dockerfile, plus Docker Compose with `postgres`, `mlflow`, `api`.
18. GitHub Actions: lint, tests against a Postgres service container, container build, smoke test, publish to GHCR.
19. Tests written *within* each phase, not batched at the end.
20. README with architecture diagram, one-command reproduction, and an honest limitations section.

### NICE TO HAVE

- PSI / KS drift report comparing the v1 training window to the holdout window, logged to MLflow.
- A `trainer` Compose service behind a `profiles: ["training"]` flag so training is containerized but doesn't start by default.
- Static API-key header auth (one middleware, ~15 lines) purely to show you know it belongs there.
- Prediction request/response logging to a Postgres `predictions` table.
- A CLI (`typer` or `argparse`) wrapping the pipeline stages.
- Business-impact quantification using Olist review scores (§4.7).

### DO NOT BUILD

Airflow, Prefect, Dagster, Kafka, Kubernetes, Terraform, Spark, a feature store, a vector DB, cloud deployment, microservices, a UI, real-time monitoring dashboards, model A/B traffic splitting, or a Streamlit app. Each of these either has a bad effort-to-value ratio here or actively muddies the story. Your other two projects already carry Streamlit; this one being headless is a feature, not a gap.

**Deliberately excluded, with a defence you should memorize:**
- *"Why no Airflow?"* — There is one linear DAG that runs on a weekly-to-monthly cadence. A Makefile plus a scheduled GitHub Actions workflow expresses that. Airflow's value is dynamic scheduling, backfills and inter-team dependency management, none of which exist here. Adding it would be resume-driven architecture.
- *"Why no cloud deployment?"* — Free tiers expire and cold-start badly; I preferred a deployment that reproduces identically on any machine with one command. The image is versioned and published to GHCR by CI, so the deploy step is real, just not hosted.

---

## 2. Architecture (Step 3)

```
Kaggle Olist CSVs (9 files, downloaded once, gitignored)
   │
   ├─ scripts/download_data.py ─────────────────► data/raw/*.csv
   │
   ▼
[ EXTRACT ]  src/etl/load_raw.py
   │  bulk COPY into PostgreSQL schema `raw` (1:1 with CSVs, no transformation)
   ▼
PostgreSQL `raw`  ◄── docker compose service: postgres
   │
   ▼
[ TRANSFORM ]  src/etl/build_orders.py
   │  filter to delivered orders, join 6 tables, derive target,
   │  aggregate geolocation to zip centroids, dedupe, type-cast
   ▼
[ VALIDATE ]  src/etl/schema.py (Pandera)  ── fails the run on contract violation
   │
   ▼
PostgreSQL `features.orders_analytical`  (one row per order, ~96k rows)
   │
   ▼
[ AS-OF AGGREGATES ]  src/features/history.py
   │  monthly expanding stats per seller / route / category,
   │  each month using only orders DELIVERED strictly before it (§18 A1)
   ▼
PostgreSQL `features.seller_monthly`, `features.route_monthly`, `features.category_monthly`
   │
   ▼
[ FEATURE ASSEMBLY ]  src/features/build.py  → ~38 features, 8 families
   │
   ▼
[ TIME SPLIT ]  warm-up | train | val | frozen holdout        (§4.4)
   │
   ├──► baselines (majority class, logistic regression)
   │
   ├──► LightGBM ─┐
   ├──► XGBoost  ─┼─ Optuna, TimeSeriesSplit CV, wall-clock capped
   ├──► CatBoost ─┘
   │
   ▼
[ ENSEMBLE ]  weighted probability blend, weights fit on validation
   │
   ▼
[ CALIBRATE + THRESHOLD ]  isotonic calibration, cost-optimal threshold
   │
   ▼
[ EVALUATE ]  PR-AUC, ROC-AUC, recall@capacity, SHAP, PR curve, confusion matrix
   │
   ▼
[ MLflow ]  sqlite:///mlflow.db  +  ./mlartifacts
   │  run params/metrics/artifacts  →  log_model(signature, input_example)
   │  →  register as delivery_delay_classifier vN
   ▼
[ PROMOTION GATE ]  src/registry/promote.py
   │  score challenger + champion on the frozen holdout
   │  promote only if PR-AUC improves by > min_delta and latency is within budget
   ▼
Registry alias @champion
   │
   ▼
[ SERVE ]  FastAPI  ── loads models:/delivery_delay_classifier@champion at startup
   │  /health  /predict  /predict/batch  /model/info
   ▼
[ CONTAINER ]  multi-stage Dockerfile → docker compose up (postgres + mlflow + api)
   │
   ▼
[ CI/CD ]  GitHub Actions → ruff/black → pytest (+ postgres service) → build → smoke → GHCR
```

### Component justification

| Component | Tech | Why it's here | What you must learn | Complexity | Required? |
|---|---|---|---|---|---|
| Raw load | psycopg2 `COPY` | Makes "ETL" literal rather than `read_csv` | `COPY FROM STDIN`, schemas | Low | **Yes** |
| Warehouse | PostgreSQL 16 in Compose | Real backend signal; also the thing that justifies Compose existing | SQL DDL, SQLAlchemy engines, connection strings in containers | Low–Med | **Yes** |
| Validation | Pandera | Turns "data validation" from a claim into a failing test | Schema objects, `Check`, coerce vs strict | Low | **Yes** |
| As-of aggregates | pandas + SQL | The leakage-safe historical features; your strongest data reasoning | `merge_asof`, expanding windows, cold-start fallbacks | **Med–High** | **Yes** |
| Models | LGBM / XGB / CatBoost | Standard; their differing categorical handling is a real talking point | Native categorical support in each, early stopping | Med | **Yes** |
| Tuning | Optuna | Beats grid search and is a fair interview topic | TPE sampler, pruners, `timeout` | Low–Med | **Yes** |
| Calibration | sklearn isotonic | Required before any cost-based threshold is meaningful | Reliability curves, Brier score | Low | **Yes** |
| Explainability | SHAP TreeExplainer | Expected; blend-SHAP is a differentiator (§6.7) | Additivity, why blend-SHAP is valid | Med | **Yes** |
| Tracking + registry | MLflow (SQLite) | The spine | Tracking URI, `log_model`, signatures, aliases | Med | **Yes** |
| Serving | FastAPI + Pydantic v2 | Standard; lifespan model loading is the interesting part | `lifespan`, `model_validate`, 422 semantics | Med | **Yes** |
| Containers | Docker + Compose | Reproducibility; Compose justified by 3 real services | Multi-stage builds, healthchecks, `depends_on: condition` | Med | **Yes** |
| CI/CD | GitHub Actions | Automates the above; service containers are the good bit | Jobs, matrices, service containers, GHCR auth | Med | **Yes** |
| Retraining | Custom script | Makes the registry non-decorative | Promotion gates, holdout discipline | Med | **Yes** |
| Drift | PSI / KS | Cheap, closes the loop | PSI formula and its limits | Low | *Nice* |

---

## 3. Time budget — read this before starting

Honest estimate for the full MUST HAVE scope at a typical student's pace: **36–42 hours.** Your stated budget is 2–3 weeks at 8–10 hrs/week, so **16–30 hours**. You finished the Steel Agent's six-week plan in two days, so your effective throughput with Claude Code is roughly 1.5–2x, which puts this at maybe 22–28 hours. That fits the top of your range but leaves no slack.

**If you fall behind, cut in exactly this order** (each cut is individually defensible; the first four cost you almost nothing):

1. Drift report (nice-to-have anyway)
2. Prediction logging to Postgres
3. API-key auth
4. CLI wrapper — use plain `python -m src.x` entry points
5. Drop CatBoost, keep LightGBM + XGBoost — say so on the CV, don't claim three
6. Drop Optuna, use a small fixed hyperparameter grid — but keep `TimeSeriesSplit`
7. Drop the `trainer` Compose profile

**Never cut, in any circumstance:** the as-of aggregate design, the frozen holdout, the parity test, the registry chain, or the promotion gate. Those are the project.

---

## 4. Dataset and problem definition (Step 4)

### 4.1 Why Olist wins

Candidates considered:

| Dataset | Verdict |
|---|---|
| **Olist Brazilian E-Commerce** (Kaggle, ~100k orders, 9 relational tables, Sep 2016 – Oct 2018) | **Selected.** Genuinely relational, so joins and ETL are real work. Has a natural, unambiguous target. Spans two years, which is what makes time-based splits and a retraining story possible. Small enough for your i5 / 16 GB. Category, geography, seller and product dimensions give ~38 features without padding. |
| Brazilian delivery / last-mile logistics sets | Usually single flat tables. No ETL story, no time span, weaker feature richness. |
| Synthetic supply-chain data | Zero credibility. An interviewer will ask where the data came from, and "I generated it" ends the conversation. |
| Telco / bank churn | Flat single table, no temporal structure, and generic enough to be forgettable. |

The decisive factor is the **two-year time span with a relational schema**. Without a time axis you cannot honestly demonstrate a retraining lifecycle, and that lifecycle is what this project is for.

**License:** CC BY-NC-SA 4.0. Non-commercial portfolio use is fine. Attribute Olist in the README. **Do not commit the CSVs** — gitignore `data/` and ship `scripts/download_data.py` plus setup instructions. CI uses a synthetic fixture instead (§10.3).

### 4.2 The prediction-time contract — the most important decision in this project

> **Prediction is made at `order_purchase_timestamp`.** Only information available at the moment the customer completes checkout may enter the feature matrix.

This is the business-useful framing (flag at-risk orders at checkout so ops can intervene or set expectations) and the only framing that's clean. Write this contract at the top of `src/features/build.py` as a docstring and make a test enforce it.

**Columns that are strictly forbidden as features** — every one is populated after the prediction moment:

- `order_approved_at`
- `order_delivered_carrier_date`
- `order_delivered_customer_date` *(this defines the target)*
- `order_status` *(the analytical table is filtered to `delivered`, so it's a constant and also a filter-leak trap)*
- Everything in `olist_order_reviews_dataset` — reviews are written after delivery. **Use review data for business-impact quantification only (§4.7), never as a feature.** Being able to articulate that distinction is worth more than the features would have been.

**Not leakage, despite looking like it:**
- `order_estimated_delivery_date` — shown to the customer at checkout. `promised_days` is your single strongest feature.
- `shipping_limit_date` (order_items) — a contractual seller deadline set at order time. **Verified against the data, not assumed (§18 A4):** across 110,189 order items none has a `shipping_limit_date` earlier than its `order_purchase_timestamp`, and the minimum offset is a hard 2.00 days, consistent with an SLA assigned at checkout. Two caveats — winsorize the tail (one item sits 1,052 days out), and audit `days_to_shipping_limit` for near-collinearity with `promised_days` in the §5 correlation pass.

### 4.3 Target definition

```python
is_late = order_delivered_customer_date.date() > order_estimated_delivery_date.date()
```

Compare at **date granularity**, not timestamp — `order_estimated_delivery_date` is date-only (midnight), so a timestamp comparison would silently mark almost every delivery on the promised day as late. This is a real bug that bites people on this dataset. Assert the positive rate lands in the 5–10% band after your ETL; if it doesn't, you have this bug.

Population filter: `order_status == 'delivered'` AND `order_delivered_customer_date IS NOT NULL` AND `order_purchase_timestamp` within the analysis window. Log how many rows each filter drops and put the funnel in the README — it's exactly the kind of thing interviewers probe.

### 4.4 The timeline

```
2017-01-01 ──────► 2017-04-30   WARM-UP   aggregates only, zero training rows

v1   TRAIN  2017-05-01 ──► 2017-12-31
     VAL    2018-01-01 ──► 2018-02-28   blend weights, calibration, threshold

v2   TRAIN  2017-05-01 ──► 2018-02-28   (v1's train window + v1's val window)
     VAL    2018-03-01 ──► 2018-04-30   blend weights, calibration, threshold

     2018-05-01 ──────► 2018-08-31   PROMOTION EVALUATION SET
                                     shared by both versions, scored twice, at the very end
```

Rules:
- Orders before 2017-01-01 are dropped (only a few hundred, and messy).
- Orders purchased after 2018-08-31 are dropped — delivery records tail off and the target is unreliable.
- Windows are **per version**, and `configs/splits.yaml` is keyed by version. **v1** fits on 2017-05→2017-12 and calibrates on 2018-01→2018-02. **v2** fits on 2017-05→2018-02 and calibrates on 2018-03→2018-04, with aggregates refreshed through 2018-04. Fitting and calibration windows must be disjoint for every version — the first draft had v2 training on its own validation window, which would have made its blend weights, calibration and cost-optimal threshold all in-sample (§18 A2). A sliding validation window is also what happens in production.
- The 2018-05→08 window is loaded by exactly one script, `src/evaluation/score_holdout.py`, called only by the promotion gate. Do not import it anywhere else — **the parity test uses `tests/fixtures/`, never this window** (§18 A3). Consider a runtime guard that raises if it's loaded during training. Because it decides which model gets promoted, describe it as a **promotion evaluation set**, not an untouched generalization estimate.
- November 2017 contains a Black Friday order spike with visibly degraded delivery performance. That's free narrative for both the drift report and the "why does v2 differ from v1" answer.

### 4.5 As-of monthly aggregates — the leakage-safe design

Historical performance features (seller late rate, route late rate) are the most predictive features available and the easiest to leak. Naively computing "seller X's late rate" over the whole dataset leaks the future into every row.

**Design: monthly expanding snapshots.**

For every entity *E* (seller_id, route = seller_state→customer_state, product category) and every calendar month *M* in the analysis window, compute statistics over **all orders whose delivery outcome was already known before *M*, i.e. `order_delivered_customer_date < first day of M`**. Store as `features.{entity}_monthly` keyed on `(entity_id, snapshot_month)`.

At feature-assembly time, each order joins to the snapshot for **its own purchase month**. Since that snapshot contains only orders already *delivered* before *M*, no row can see itself, its future, or an outcome that had not yet happened.

Filtering on `order_purchase_timestamp` instead — as the first draft of this plan did — is a real leak, and a target-correlated one. Late orders take a median 31 days to deliver against 9 for on-time orders, so the orders still in flight at any cutoff are **3-7x more likely to be late** than the resolved ones. Admitting them injects future late-outcomes into exactly the feature family this section calls the most predictive available. Measured on the real data, see §18 A1.

The warm-up window exists so that 2017-05 rows already have four months of *resolved* history behind them.

At **serving** time the API uses the single latest snapshot, bundled inside the preprocessing artifact. Retraining refreshes it — which is a second, independent reason v2 differs from v1, and a nice thing to say out loud.

**Cold start** (seller's first order, unseen route): emit a `*_is_new` boolean flag and fall back through a hierarchy — seller → seller's state → global. Never impute silently; the flag is a feature and lets the model learn that new sellers behave differently.

Implement with `pandas.merge_asof` on a time-sorted snapshot table, or as a SQL window function if you prefer. Either is fine; `merge_asof` is easier to test.

*Alternative you should be able to discuss:* per-row trailing-90-day windows are more rigorous but much heavier, and no production batch system recomputes features per row anyway — they refresh on a cadence. Monthly snapshots mirror how this actually works. Say that if asked.

### 4.6 Metrics

- **Primary: PR-AUC (average precision).** With ~7% positives, ROC-AUC is optimistic and flattering. Lead with PR-AUC and be ready to explain why.
- Secondary: ROC-AUC, Brier score (calibration quality), recall at a fixed review capacity (e.g. "if ops can review the top 10% of orders daily, what fraction of late orders do we catch?").
- Report the trivial baseline (predict never-late: accuracy ~93%, recall 0%) explicitly. It's the cleanest way to show why accuracy is the wrong metric here, and it pre-empts the obvious trap question.

### 4.7 Business impact

Two analyses, both cheap:

1. **Review-score damage.** Compute mean review score for on-time vs late orders in the Olist review table. The gap is large. This converts your model's recall into a concrete "reviews protected" number.
2. **Cost-based threshold.** State an explicit cost ratio — false negative (late delivery: support contact, review damage, churn risk) versus false positive (proactive notification / expedite). Sweep the threshold on the validation set, pick the cost-minimizing point, report it. Frame it as assumption-driven: *"under a 5:1 FN:FP cost assumption, the optimal threshold is 0.23, catching 68% of late orders at a 19% flag rate."* Never present the cost figures as if they were measured.

---

## 5. Feature specification (~38 features, 8 families)

Count features **pre-encoding**. If asked to "walk me through your features," name the eight families, not thirty-eight columns.

**F1 — Promise & timing (6)**
`promised_days` (estimated − purchase, days) · `purchase_hour` · `purchase_dayofweek` · `purchase_month` · `is_weekend` · `days_to_shipping_limit`

**F2 — Geography (7)**
`customer_state` · `seller_state` · `same_state` · `customer_region` (N/NE/CO/SE/S macro-region) · `haversine_km` (zip-prefix centroids) · `n_distinct_seller_states` · `customer_zip_prefix_2` (first two digits, coarse region)

**F3 — Order composition (6)**
`n_items` · `n_distinct_products` · `n_distinct_sellers` · `total_price` · `total_freight` · `freight_ratio` (freight ÷ price, guard price = 0)

**F4 — Product physical (6)**
`total_weight_g` · `max_item_weight_g` · `total_volume_cm3` · `max_item_volume_cm3` · `avg_density` · `product_category` (top-30 + `other`)

**F5 — Payment (4)**
`payment_type` (dominant method) · `max_installments` · `payment_value` · `n_payment_methods`

**F6 — Seller history, as-of (5)**
`seller_order_count_hist` · `seller_late_rate_hist` · `seller_avg_delivery_days_hist` · `seller_avg_handling_days_hist` · `seller_is_new`

**F7 — Route history, as-of (3)**
`route_late_rate_hist` · `route_avg_delivery_days_hist` · `route_order_count_hist`

**F8 — Category history, as-of (2)**
`category_late_rate_hist` · `category_avg_delivery_days_hist`

**Total: 39.** Run a correlation and permutation-importance audit after the first model and drop 1–3 dead or redundant features. Report the surviving number honestly on the CV — if it lands at 37, write 37.

**Notes.** Geolocation needs aggregating to one centroid per `zip_code_prefix` (median lat/lng) before joining; there are multiple rows per prefix. Some prefixes are missing on one side — fall back to the state centroid and flag it. Haversine is worth a dedicated unit test with a known-distance pair.

---

## 6. ML pipeline design (Step 5)

### 6.1 Preprocessing strategy

Use each library's **native categorical handling** rather than one-hot encoding: CatBoost natively, LightGBM via `category` dtype, XGBoost via `enable_categorical=True` with pandas `category` dtype. This keeps the pipeline simple, avoids target-encoding leakage entirely, and gives you a genuine talking point about how the three libraries differ.

Preprocessing then reduces to: numeric imputation (median, fitted on train only), category level fixing, feature construction, and column ordering.

### 6.2 The preprocessing artifact — the train/serve skew answer

One serializable object, logged to MLflow **alongside every model**, containing:

- fitted numeric imputation values
- the frozen category level set per categorical column (plus the `__unknown__` sentinel)
- the latest as-of aggregate snapshots (seller / route / category lookups)
- the zip-prefix → centroid table
- the exact feature name list, in order
- the calibrator
- the chosen decision threshold
- the training data cutoff date and a schema hash

Inference **loads and applies** this. It never re-fits anything. Encode that as an invariant in the class API: give it `transform()` and no public `fit_transform()` after serialization.

### 6.3 Cross-validation

`TimeSeriesSplit(n_splits=4)` on time-sorted training data. Do not use `KFold` or `StratifiedKFold` — with as-of features and temporal drift, random folds let the future leak into the past. Be able to say that in one sentence.

Early stopping uses each fold's validation slice, not the held-out val period.

### 6.4 Models and tuning budget

Your hardware is an i5-1235U (low-power mobile, 12 threads, no usable GPU). Budget deliberately or CatBoost will eat an evening:

| Model | Optuna trials | Wall-clock cap | Notes |
|---|---|---|---|
| LightGBM | 40 | 15 min | Fastest; tune `num_leaves`, `min_child_samples`, `learning_rate`, `feature_fraction`, `bagging_fraction`, `lambda_l2` |
| XGBoost | 40 | 15 min | `hist` tree method, `enable_categorical=True` |
| CatBoost | 20 | 25 min | Slowest by far. Cap `iterations`, use `od_type='Iter'`, set `thread_count=-1` |

Use Optuna's `timeout` parameter as a hard stop and log the number of trials actually completed. Set `n_jobs=1` on the Optuna study and let the boosting library use the threads — nested parallelism will thrash this CPU.

**Class imbalance:** do not resample. Train on raw probabilities, then calibrate, then tune the threshold. Run `scale_pos_weight` as one logged experiment so you can say you tested it, and report what happened (usually: better recall at default threshold, worse calibration, no gain once the threshold is tuned properly).

### 6.5 Ensemble

Weighted average of predicted probabilities, weights fitted on the validation period by minimizing log-loss (constrained to sum to 1, non-negative). Simpler and more defensible than stacking, which would need its own nested CV.

**Report the delta honestly.** Three GBDTs on identical tabular features produce highly correlated predictions; expect roughly +0.002 to +0.01 PR-AUC. If the gain is negligible relative to 3x inference cost, **ship the single best model and keep the ensemble as a documented, logged experiment.** That decision, stated plainly, is a stronger interview answer than a manufactured win:

> *"The blend gained 0.004 PR-AUC over tuned CatBoost for roughly 3x inference latency, so I shipped the single model and kept the ensemble run in MLflow. If the operational cost of a missed late delivery justified it later, promoting the blend is one registry alias change."*

Log both. Let the promotion gate decide.

### 6.6 Leakage checklist — turn each line into a test

- [ ] No forbidden column (§4.2) appears in the feature matrix. Assert against an explicit denylist.
- [ ] Aggregate snapshots for month *M* contain no order whose outcome was unresolved at *M*. Assert `max(order_delivered_customer_date) < M`, **not** `max(order_purchase_timestamp) < M` (§18 A1).
- [ ] Imputation values, category levels and calibrators are fitted on training rows only.
- [ ] `TimeSeriesSplit`, never `KFold`.
- [ ] Holdout is loaded by exactly one module.
- [ ] Max `order_purchase_timestamp` in train < min in val < min in holdout.
- [ ] Feature matrix column order matches the artifact's stored list exactly.

### 6.7 SHAP

`TreeExplainer` on the test sample (2–5k rows is plenty; full SHAP on 20k rows will be slow on your CPU). Log the beeswarm summary, a bar plot of mean |SHAP|, and 2–3 dependence plots for the top features.

If you ship the ensemble: SHAP values are additive, and a weighted average of models is a linear combination, so **the blend's SHAP values are the same weighted average of the per-model SHAP values.** That's mathematically valid and a genuinely good thing to be able to explain. Make sure you can derive it in one line before you claim it.

Expect `promised_days`, `seller_late_rate_hist` and `haversine_km` to dominate. If a feature you thought was harmless shows up implausibly high, treat it as a leakage alarm, not a win.

---

## 7. MLflow design (Step 6)

**Configuration.** Tracking URI `sqlite:///mlflow.db` (a file backend store is required for the Model Registry — plain `./mlruns` filesystem tracking does not support it). Artifacts to `./mlartifacts`. Both are gitignored, both are bind-mounted into the `mlflow` Compose service.

**Run structure.** One parent run per training session, named `{version}_{timestamp}`, with nested child runs:

```
delivery_delay_v1_20260215
├── baseline_logreg
├── lightgbm      (nested: optuna trials logged as a study, not 40 runs)
├── xgboost
├── catboost
└── ensemble      ← the model registered from here
```

Log Optuna trials to the study, not as 40 MLflow runs — otherwise the UI is unusable.

**Per run, log:**
- *Params:* model hyperparameters, split dates, feature count, feature-family list, aggregate snapshot date, git SHA, random seed
- *Metrics:* PR-AUC, ROC-AUC, Brier, recall@10% capacity, best threshold, expected cost at threshold, fit seconds
- *Artifacts:* PR curve, ROC curve, calibration curve, confusion matrix at chosen threshold, SHAP beeswarm, SHAP bar, feature list JSON, the Optuna study, the preprocessing artifact
- *Model:* `mlflow.pyfunc.log_model` wrapping (preprocessor + model + calibrator + threshold) with a `signature` and `input_example`

Wrapping everything in a single **`pyfunc`** is the key call. It means the API receives raw order records and gets back a probability and a decision, with zero preprocessing logic duplicated in the serving layer. It's the structural fix for train/serve skew, not just a convenience.

**Registry.** Model name `delivery_delay_classifier`. Use **aliases** (`@champion`, `@challenger`), not the stage API — stages were deprecated in MLflow 2.9 and **removed in MLflow 3**, which is the version that resolves here (3.16.1), so aliases are the only mechanism available. Note `log_model(name=...)` replaces 2.x's `artifact_path=` (§18 A7). Set tags on each version: training window, holdout PR-AUC, git SHA, promotion decision and reason.

---

## 8. Retraining lifecycle and promotion gate

This is what makes the registry mean something. Without it you have one model version and "model versioning" is decoration.

**`src/registry/promote.py`:**

1. Load the current `@champion` (skip on first run) and the new `@challenger`.
2. Score both on the **frozen holdout** — same rows, same code path, no re-fitting.
3. Apply the gate:
   - challenger PR-AUC > champion PR-AUC + `min_delta` (config, e.g. 0.005)
   - challenger Brier score not worse by more than a tolerance (guards against buying discrimination with calibration)
   - challenger p95 single-prediction latency under budget
   - challenger feature schema matches the serving contract
4. On pass: move the `champion` alias, tag the old version `archived`, write the decision to `reports/promotions.md`.
5. On fail: leave `champion` alone, log why. **A "no-promote" run is a success, not a failure** — say that in the interview.

Run it twice for real: v1 (auto-promoted, nothing to beat) and v2 (contested). Keep both decisions in the repo. If v2 loses on the gate, keep that too — it's a more interesting and more honest story than a manufactured improvement, and it demonstrates the gate actually does something.

---

## 9. FastAPI service (Step 7)

### Structure

```
api/
├── main.py            # app, lifespan, router registration
├── config.py          # pydantic-settings: MODEL_URI, MLFLOW_TRACKING_URI, LOG_LEVEL
├── schemas.py         # Pydantic v2 request/response models
├── model_loader.py    # registry resolution + in-memory singleton
├── routes/
│   ├── health.py
│   ├── predict.py
│   └── info.py
└── exceptions.py      # handlers → consistent JSON error envelope
```

### Endpoints

| Route | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness. Returns 200 with `{status, model_loaded, model_version}`. Returns 503 if the model failed to load — this is what Compose and CI healthchecks hit. |
| `/predict` | POST | One order record → `{probability, is_late_predicted, threshold, model_version}` |
| `/predict/batch` | POST | Array of records (cap at ~1000) → array of predictions plus timing |
| `/model/info` | GET | Version, registry alias, training window, feature count, threshold, snapshot date |

### Design decisions

**Model loading:** once at startup via an `asynccontextmanager` lifespan handler, held in module state. Never per-request — that would add hundreds of milliseconds and hammer the registry. Fail loudly on startup if resolution fails, and have `/health` report 503 rather than the app silently serving a stale model.

**`MODEL_URI` resolution order** — this matters for CI, where no MLflow server exists:
1. `models:/delivery_delay_classifier@champion` (default, production path)
2. `models:/delivery_delay_classifier/{version}` (pinned rollback)
3. A local filesystem path (CI and tests, using a tiny fixture model)

Supporting all three isn't a hack; being able to pin a version *is* your rollback story.

**Unseen categories:** map to the `__unknown__` sentinel the models were trained with. Return 200 with a prediction plus a `warnings` field. Do not 500, and do not silently pretend it didn't happen.

**Errors:** Pydantic validation failures → 422 automatically. Business-rule violations (batch too large, negative price) → 400 with a structured envelope. Unhandled → 500 with a correlation ID, never a stack trace.

**Latency:** measure p50/p95 for single and batch-100 with a small script. One measurement, recorded in the README. Interviewers consistently reward "I measured it" over any specific number.

**No microservices.** One service. If asked why: the model is a few MB, inference is sub-millisecond, and there is no independent scaling axis. Splitting it would add network hops and deployment surface for nothing.

---

## 10. Docker and CI/CD (Steps 8–9)

### 10.1 Dockerfile

Multi-stage on `python:3.12-slim`:
- **Stage 1 (builder):** install build deps, `pip install --user -r requirements.txt`
- **Stage 2 (runtime):** copy site-packages and app code only, create a non-root user, `HEALTHCHECK` hitting `/health`, `CMD uvicorn`

Keep it under ~40 lines and be able to explain every one. `.dockerignore` must exclude `data/`, `mlartifacts/`, `mlflow.db`, `.git`, `notebooks/`, `__pycache__` — otherwise your build context balloons and layer caching stops working.

Model artifacts are **not** baked into the image. The image is generic; the model comes from the registry at runtime. That separation is the point: rolling out a new model is a config change, not a rebuild.

### 10.2 Compose

```yaml
services:
  postgres:   # 16-alpine, named volume, healthcheck: pg_isready
  mlflow:     # mlflow server, sqlite backend, bind-mounted artifacts, healthcheck
  api:        # built from Dockerfile, depends_on both with condition: service_healthy
  # trainer:  # profiles: ["training"] — optional, does not start by default
```

Use `depends_on: condition: service_healthy`, not bare `depends_on` — otherwise the API races Postgres on cold start and you'll spend an hour debugging it. Credentials come from `.env`; ship `.env.example`. **Host ports come from `.env` as well — publish the API on host 8001, since 8000 belongs to another project on this machine.** `docker compose up` is the entire deployment story *once a model is registered*: on a clean machine the registry is empty, the API cannot resolve `@champion`, and it will correctly refuse to start. The ordered bootstrap in §18 A5 (`make bootstrap`) runs first, and the README's one-command claim must be scoped to say so.

### 10.3 GitHub Actions

**`ci.yml`** on push/PR:
1. `ruff check` + `black --check`
2. `pytest` with a **Postgres service container** (this is the part that looks like real CI, and it's cheap)
3. ETL smoke test on a synthetic fixture
4. Coverage report

**`build.yml`** on push to `main`:
5. `docker build`
6. `docker run` + curl `/health` and `/predict` with a golden fixture — a build that isn't smoke-tested proves nothing
7. `docker push` to **GHCR**, tagged `:latest` and `:{sha}` (GHCR is free and needs no new account — `GITHUB_TOKEN` is enough)

**Critical constraint:** CI cannot train models and must not need the real dataset — it's large and the license is non-commercial. Write `tests/fixtures/generate_synthetic.py` producing a few hundred rows with the same schema and plausible distributions, and commit a tiny pre-trained fixture model (a 10-tree LightGBM, a few KB) for API tests. Solving this properly is itself a good interview anecdote.

**CI vs CD, stated honestly:** CI is lint + tests + build + smoke. CD here is *publish a versioned, smoke-tested artifact to a registry*, plus a one-command Compose deploy. There is no hosted environment, and your CV must not imply there is. See §13.

---

## 11. Environment setup (Step 10)

### Required software

| Software | Version | Notes |
|---|---|---|
| Linux environment | Ubuntu 22.04 / 24.04 | **VM, WSL2, or native — all fine.** Develop on Linux, not bare Windows: path handling, line endings and Docker volume mounts all behave. |
| Python | **3.12.x** | **Decided: 3.12**, the Ubuntu 24.04 system interpreter. Verified — the full 117-package stack resolves on 3.12 with binary wheels only, zero source builds. `python:3.12-slim`, `ci.yml` and `CLAUDE.md` are all 3.12; keep them identical, because a `pip freeze` lockfile is only valid for the interpreter that produced it (§18 A7). |
| Git | latest | Add a `.gitattributes` pinning LF |
| Docker Engine + Compose plugin | latest | Installed **inside** the Linux environment. Docker Desktop is only needed if the host is Windows/macOS. Add your user to the `docker` group. |
| VS Code | latest | Remote-SSH (VM) or Remote-WSL, plus Python and Docker extensions |
| Claude Code | latest | Run it from the Linux terminal inside VS Code |
| PostgreSQL | 16 | **Container only** — do not install natively |
| MLflow | **3.16+** | pip, not a separate install. Stages are *removed* in 3.x, so §7's alias-based registry design becomes the only option rather than merely the current one. `log_model` takes `name=` where 2.x took `artifact_path=` (§18 A7). |

**Repo location.** Keep the repo on the Linux filesystem (`~/projects/...`) — never a VMware shared folder or `/mnt/c/...`. Cross-filesystem I/O is dramatically slower and makes every pandas operation feel inexplicably broken.

**If you're on a VM,** allocate as many cores and as much RAM as you can spare (6+ cores, 8–10 GB). The only CPU-hungry step is Optuna tuning in Phase 4. With fewer cores, raise the timeouts in `configs/models.yaml` rather than letting the trial budget silently truncate — and log the trials actually completed so the numbers stay honest.

### Python packages

**Core** — `pandas` `numpy` `scikit-learn` `scipy` `lightgbm` `xgboost` `catboost` `optuna` `shap` `mlflow` `fastapi` `uvicorn[standard]` `pydantic` `pydantic-settings` `sqlalchemy` `psycopg2-binary` `pandera` `pyyaml` `python-dotenv` `joblib` `matplotlib`

**Dev** — `pytest` `pytest-cov` `httpx` `ruff` `black` `pre-commit`

**Optional** — `jupyterlab` `ipykernel` `typer` `seaborn`

**Pinning:** pin **every direct dependency to an exact version**, let transitives float. Install once, run `pip freeze > requirements.lock.txt`, and use the lockfile in Docker and CI. Do not pin versions from this document — install fresh, then freeze what actually resolves. Reproducibility is a claim you'll be asked to defend; an unpinned `requirements.txt` makes that claim false.

Separate `requirements.txt` (core), `requirements-dev.txt` (dev), and `requirements-api.txt` (a minimal serving subset — no shap, optuna, catboost, or matplotlib). The API image should not carry the training stack; it roughly halves the image and is a good thing to point at.

---

## 12. Repository structure (Step 11)

```
delivery-delay-platform/
├── CLAUDE.md                       # project context for Claude Code — see §14.1
├── PLAN.md                         # this file
├── README.md                       # architecture, results, one-command repro, limitations
├── Makefile                        # make etl / train / promote / serve / test
├── .env.example  .gitignore  .dockerignore  .gitattributes
├── pyproject.toml                  # ruff + black + pytest config
├── requirements.txt  requirements-dev.txt  requirements-api.txt
├── docker-compose.yml
├── Dockerfile
│
├── configs/
│   ├── base.yaml                   # paths, DB, MLflow URI, seed
│   ├── features.yaml               # feature families, denylist, category caps
│   ├── splits.yaml                 # the §4.4 date boundaries — single source of truth
│   ├── models.yaml                 # search spaces, trial counts, timeouts
│   └── promotion.yaml              # min_delta, latency budget, tolerances
│
├── src/
│   ├── etl/
│   │   ├── load_raw.py             # CSVs → postgres raw schema (COPY)
│   │   ├── build_orders.py         # joins, target, filters → features.orders_analytical
│   │   ├── geolocation.py          # zip prefix → centroid
│   │   └── schema.py               # Pandera contracts
│   ├── features/
│   │   ├── history.py              # as-of monthly aggregate snapshots  ← core logic
│   │   ├── build.py                # the 8 families; prediction-time contract enforced here
│   │   └── artifact.py             # PreprocessingArtifact (fit / transform / save / load)
│   ├── splits.py                   # time splits; the ONLY module that may load holdout
│   ├── training/
│   │   ├── baselines.py
│   │   ├── tune.py                 # Optuna, TimeSeriesSplit
│   │   ├── train.py                # orchestrates a full versioned training run
│   │   └── ensemble.py             # blend weights + calibration + threshold
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── plots.py
│   │   ├── explain.py              # SHAP
│   │   ├── drift.py                # PSI / KS (nice-to-have)
│   │   └── score_holdout.py        # ← the only holdout consumer
│   ├── registry/
│   │   ├── pyfunc_wrapper.py       # preprocessor + model + calibrator + threshold
│   │   └── promote.py              # the gate
│   └── db.py                       # SQLAlchemy engine factory
│
├── api/                            # §9
├── tests/
│   ├── fixtures/generate_synthetic.py
│   ├── test_etl_*.py  test_features_*.py  test_leakage.py
│   ├── test_parity.py              # ← the important one
│   ├── test_api_*.py  test_registry_*.py
│   └── conftest.py
├── scripts/
│   ├── download_data.py
│   ├── run_pipeline.py             # etl → features → train → register
│   └── benchmark_latency.py
├── notebooks/                      # EDA only. Nothing here is imported by src/.
├── reports/                        # promotions.md, drift reports, metrics tables
└── .github/workflows/{ci.yml,build.yml}
```

`configs/splits.yaml` as the single source of truth for dates is deliberate — split boundaries duplicated across files is how holdout contamination happens.

---

## 13. Phase plan and Claude Code prompts (Steps 12–13)

Tests are written **inside** each phase, not batched. Every phase ends green.

### How to work with Claude Code on this

1. **Write `CLAUDE.md` first** (§14.1) and keep it current. It's the difference between Claude Code understanding your architecture and re-deriving it every session.
2. **One phase per session.** Start fresh between phases; long contexts drift.
3. **Never accept code you can't explain.** After each phase, close the editor and explain the component out loud. If you can't, re-read it. That habit is what makes the project defensible.
4. **Ask for tests in the same prompt as the implementation.** Retrofitted tests test what the code does, not what it should do.
5. **When Claude Code proposes an addition not in this plan, default to no.** Scope creep is the main failure mode here.

---

### Phase 0 — Skeleton and environment (1.5 h, easy)

**Objective:** repo, tooling, and Compose skeleton running.

**Tasks:** working Linux environment with Docker Engine + Compose plugin; venv on Python 3.12; `pyproject.toml` with ruff/black/pytest; `.gitignore` (must include `data/`, `mlartifacts/`, `mlflow.db`, `.env`); pre-commit; `docker-compose.yml` with `postgres` + `mlflow`; a stub `ci.yml` running lint on push.

**Verify:** `docker compose up postgres mlflow` → MLflow UI at `localhost:5000`, `psql` connects. CI green on the first push.

**Completion:** empty repo, green CI, two healthy containers.

```
Read CLAUDE.md.

Set up the project skeleton for Phase 0 only. Do not write any ETL,
feature, or model code.

Create:
1. The directory structure in PLAN.md §12, with empty __init__.py files
2. pyproject.toml configuring ruff (line-length 100), black, and pytest
   (testpaths=tests, --cov=src)
3. .gitignore, .dockerignore, .gitattributes (LF), .env.example
4. docker-compose.yml with exactly two services: postgres (16-alpine,
   named volume, pg_isready healthcheck) and mlflow (sqlite backend store
   at /mlflow/mlflow.db, artifact root /mlartifacts, both bind-mounted,
   HTTP healthcheck). No api service yet.
5. src/db.py — SQLAlchemy engine factory reading from pydantic-settings
6. configs/base.yaml and configs/splits.yaml with the §4.4 dates
7. .github/workflows/ci.yml — checkout, Python 3.12, install
   requirements-dev.txt, ruff check, black --check. No tests yet.
8. requirements.txt / requirements-dev.txt / requirements-api.txt,
   unpinned for now — I'll freeze after first install.

Explain your docker-compose healthcheck choices before writing the file.
```

---

### Phase 1 — Raw load into Postgres (2.5 h, easy–medium)

**Objective:** all nine Olist CSVs in a `raw` schema, verified.

**Learn beforehand:** Postgres `COPY FROM STDIN`, schemas vs databases, SQLAlchemy 2.x engine/connection semantics.

**Tasks:** `scripts/download_data.py` (manual Kaggle download is fine — document it); `src/etl/load_raw.py` with explicit DDL per table and `COPY`; idempotent re-runs; a row-count reconciliation report.

**Common mistakes:** letting pandas infer types on load (it will mangle zip prefixes with leading zeros — read them as strings); non-idempotent loads that duplicate on re-run; forgetting that `olist_geolocation_dataset` is large and needs `COPY`, not `to_sql`.

**Tests:** row counts match the CSVs; every expected table exists; re-running is idempotent.

**Completion:** `make load-raw` twice in a row produces identical counts.

```
Read CLAUDE.md and PLAN.md §4.1.

Phase 1: load the 9 Olist CSVs from data/raw/ into the postgres `raw`
schema. Only this.

1. src/etl/load_raw.py:
   - Explicit CREATE TABLE DDL per file — no pandas type inference.
     zip_code_prefix columns are TEXT (leading zeros matter).
     Timestamps are TIMESTAMP, nullable where Olist has nulls.
   - Load with psycopg2 copy_expert / COPY FROM STDIN, not to_sql
   - Idempotent: TRUNCATE before load, all inside one transaction
   - Log rows read vs rows inserted per table
2. scripts/download_data.py: check for the 9 files, print Kaggle
   instructions and exit non-zero if missing. Do not scrape Kaggle.
3. tests/test_etl_load_raw.py: expected tables exist; counts match
   the CSVs; a second run leaves counts unchanged.
4. A `load-raw` Makefile target.

Tell me the exact column types you chose for olist_orders_dataset
and why, before writing the DDL.
```

---

### Phase 2 — Transform, validate, analytical table (4 h, medium)

**Objective:** `features.orders_analytical`, one row per order, target derived, contract-validated.

**Learn beforehand:** Pandera schemas, the §4.2 prediction-time contract, why `order_status` is a leakage trap.

**Tasks:** filter to delivered; aggregate `order_items` to order level (items, sellers, price, freight, weight, volume, category); aggregate payments; join customers and sellers; geolocation → zip centroids; derive `is_late` at **date granularity**; drop forbidden columns; Pandera validation; write to `features.orders_analytical`; log the filter funnel.

**Common mistakes:** the timestamp-vs-date target bug (§4.3) — check your positive rate lands at 5–10%; forgetting geolocation has many rows per prefix; letting forbidden columns survive the join.

**Tests:** exactly one row per order; positive rate in band; no forbidden column present; funnel counts sum correctly; haversine correct against a known pair.

**Completion:** table built, Pandera passes, positive rate 5–10%, funnel in `reports/`.

```
Read CLAUDE.md and PLAN.md §4.2, §4.3.

Phase 2: build features.orders_analytical from the raw schema.

1. src/etl/geolocation.py — median lat/lng per zip_code_prefix;
   state-level centroid fallback; a haversine_km function.
2. src/etl/build_orders.py:
   - Filter: order_status='delivered' AND delivered_customer_date NOT NULL
     AND purchase_timestamp between the configs/splits.yaml bounds
   - Aggregate order_items to order level (n_items, n_distinct_products,
     n_distinct_sellers, total/max price, freight, weight, volume,
     dominant category)
   - Aggregate payments (dominant type, max installments, total value,
     n methods)
   - Join customers and sellers for state/zip
   - Target: is_late = delivered_customer_date.date() >
     estimated_delivery_date.date()  ← DATE granularity, this matters
   - Explicitly drop the §4.2 denylist columns before writing
   - Log a filter funnel to reports/etl_funnel.md
3. src/etl/schema.py — Pandera schema for the output. Raise on failure.
4. tests/test_etl_build_orders.py — one row per order_id; positive rate
   in [0.05, 0.10]; no denylist column present; funnel arithmetic;
   haversine against a known city pair.

Before writing code: tell me the positive rate you expect and what it
would mean if you got 45% instead.
```

---

### Phase 3 — As-of aggregates and feature assembly (5 h, **hardest phase**)

**Objective:** the ~39 features, with provably leakage-free history features.

**Learn beforehand:** `pandas.merge_asof`; expanding vs rolling windows; cold-start handling. Read §4.5 twice before starting.

**Tasks:** `src/features/history.py` building monthly snapshots for seller / route / category; `src/features/build.py` assembling all 8 families; `src/features/artifact.py` (`PreprocessingArtifact` with fit/transform/save/load); `src/splits.py` reading `configs/splits.yaml`.

**Common mistakes:** including month *M*'s own orders in the *M* snapshot (the whole point is to exclude them); silently imputing cold-start entities instead of flagging; column order drifting between fit and transform.

**Tests (this is where the leakage tests live):** a snapshot for month *M* contains no order *delivered* on or after *M* (outcome availability, not purchase date — §18 A1); a synthetic seller whose late rate changes sharply mid-period gets the *prior* rate, not the blended one; cold-start rows get `is_new=True` and the fallback value; `transform()` on the same input twice is identical; feature count and order match the artifact; `max(train.purchase) < min(val.purchase) < min(holdout.purchase)`.

**Completion:** all leakage tests green. Do not proceed until they are.

```
Read CLAUDE.md and PLAN.md §4.5 and §5 carefully. This is the most
important phase — leakage here invalidates everything downstream.

Phase 3, part A only (do not write build.py yet):

src/features/history.py — as-of monthly aggregate snapshots.

For each entity type (seller_id; route = seller_state||'_'||customer_state;
product_category) and each calendar month M in the analysis window,
compute over ALL orders DELIVERED strictly before the first day of M
(order_delivered_customer_date < M). Filtering on purchase_timestamp
leaks future outcomes and the leak favours the positive class — see
PLAN.md §18 A1:
  - order_count, late_rate, avg_delivery_days, avg_handling_days
Write to features.{entity}_monthly keyed (entity_id, snapshot_month).

Then a join function attaching each order to the snapshot for its own
purchase month, with cold-start fallback: entity → entity's state →
global, plus an is_new boolean flag.

Tests in tests/test_features_history.py:
  1. For a snapshot month M, no contributing order has
     order_delivered_customer_date >= M (outcome availability)
  2. Synthetic seller with late_rate 0.0 in months 1-3 then 1.0 in
     month 4 → the month-4 snapshot shows 0.0, not 0.25
  3. An entity's first-ever order gets is_new=True and the fallback
  4. Global fallback used when the state fallback is also empty

Explain the leakage argument for test 1 before writing the code, then
show me the tests before the implementation.
```

Then a second prompt for `build.py` + `artifact.py`, and a third for `splits.py`. Do not combine them.

---

### Phase 4 — Baselines and single models (4 h, medium)

**Objective:** three tuned GBDTs plus baselines, evaluated with time-series CV.

**Tasks:** majority-class and logistic-regression baselines; `TimeSeriesSplit(4)`; Optuna per model within the §6.4 budget; native categorical handling per library; metrics module (PR-AUC, ROC-AUC, Brier, recall@capacity).

**Common mistakes:** reaching for `StratifiedKFold`; nested parallelism (Optuna `n_jobs>1` and library threads together) thrashing your CPU; no `timeout` and a 3-hour CatBoost run.

**Tests:** the metrics functions against hand-computed cases; the CV splitter never puts a later date in a training fold than in its validation fold; a smoke train on the synthetic fixture finishes in under 30 s.

**Completion:** all three beat both baselines on PR-AUC; runtime within budget.

```
Read CLAUDE.md and PLAN.md §6.3, §6.4.

Phase 4: baselines and single-model training. No ensemble, no MLflow yet.

1. src/evaluation/metrics.py — pr_auc, roc_auc, brier, recall_at_capacity(k),
   and a summary dict. Unit-test each against hand-computed small cases.
2. src/training/baselines.py — majority-class and logistic regression
   (median impute + standardize numerics, one-hot the low-cardinality
   categoricals only).
3. src/training/tune.py — Optuna over TimeSeriesSplit(n_splits=4) on
   time-sorted training data. StratifiedKFold is forbidden; add a comment
   saying why. Search spaces from configs/models.yaml.
   Per-model trial counts and timeouts exactly as PLAN.md §6.4.
   Optuna n_jobs=1; let the boosting libraries use the threads.
   Native categoricals: CatBoost cat_features, LightGBM category dtype,
   XGBoost enable_categorical=True.
4. tests/test_training_cv.py — assert every CV fold's max training date
   is < its min validation date.

Report the actual wall-clock time per model when you run it.
```

---

### Phase 5 — Ensemble, calibration, threshold (4 h, medium)

**Objective:** blend, calibrated probabilities, cost-derived threshold, honest delta.

**Tasks:** blend weights fitted on validation by log-loss; isotonic calibration; threshold sweep under the §4.7 cost model; the review-score impact analysis; SHAP.

**Common mistakes:** fitting blend weights on the training set (they'll collapse onto the most overfit model); calibrating on data used to fit the blend; reporting a 0.001 PR-AUC gain as meaningful.

**Tests:** blend weights are non-negative and sum to 1; the calibrator improves Brier on held-out data; the threshold sweep returns the actual argmin.

**Completion:** the honest ensemble-vs-best-single delta is written into `reports/`, along with your ship-or-don't decision and its reasoning.

```
Read CLAUDE.md and PLAN.md §6.5, §6.7, §4.7.

Phase 5: ensemble, calibration, thresholding, SHAP.

1. src/training/ensemble.py — fit non-negative weights summing to 1 by
   minimizing log-loss on the VALIDATION period (not train). Report the
   PR-AUC delta vs the best single model to 4 decimal places.
2. Isotonic calibration fitted on validation. Report Brier before/after.
3. Threshold sweep under a configurable FN:FP cost ratio (default 5:1);
   return the cost-minimizing threshold plus the resulting precision,
   recall and flag rate.
4. Business impact: mean review score for on-time vs late orders,
   using the reviews table for ANALYSIS ONLY — assert it never enters
   the feature matrix.
5. src/evaluation/explain.py — SHAP TreeExplainer on a 3000-row sample.
   Beeswarm, mean-|SHAP| bar, top-3 dependence plots, saved to reports/.
6. Write reports/model_selection.md: the ensemble delta, and an explicit
   ship / don't-ship recommendation with reasoning.

Do not inflate the ensemble result. If the gain is under 0.005 PR-AUC,
say plainly that the single model should ship.
```

---

### Phase 6 — MLflow, pyfunc wrapper, registry (4 h, medium)

**Objective:** the spine. A training run produces a registered, self-contained model version.

**Tasks:** tracking config; parent/nested run structure; log everything in §7; `PyfuncWrapper` bundling preprocessor + model + calibrator + threshold; `log_model` with signature and input example; register and alias.

**Common mistakes:** logging 40 Optuna trials as MLflow runs; omitting the signature (which breaks schema enforcement at load); logging the model but not the preprocessing artifact — the classic skew bug.

**Tests:** a run creates the expected params/metrics/artifacts; the pyfunc round-trips (log → load → predict) with identical output; the signature matches the feature list.

**Completion:** `delivery_delay_classifier` v1 in the registry, `@champion` set, loadable in a fresh Python process.

```
Read CLAUDE.md and PLAN.md §7.

Phase 6: MLflow integration and the registry.

1. src/registry/pyfunc_wrapper.py — an mlflow.pyfunc.PythonModel that
   bundles PreprocessingArtifact + model(s) + calibrator + threshold.
   predict() takes a RAW order-record DataFrame and returns
   probability, is_late_predicted, and the threshold used.
   No preprocessing logic may exist outside this wrapper.
2. src/training/train.py — orchestrate a full versioned run:
   parent run named {version}_{timestamp}, nested children for
   baseline / lightgbm / xgboost / catboost / ensemble.
   Log everything listed in PLAN.md §7. Optuna trials go to the study,
   NOT as separate MLflow runs.
   log_model with signature and input_example, register as
   delivery_delay_classifier.
3. Aliases, not stages (stages are deprecated). Tag each version with
   training window, holdout PR-AUC, git SHA.
4. tests/test_registry_roundtrip.py — log a tiny model, load it by URI
   in a clean context, assert predictions match the in-memory model
   to 1e-9.

Explain why the preprocessing artifact must live inside the pyfunc
rather than being loaded separately by the API.
```

---

### Phase 7 — FastAPI and the parity test (4 h, medium)

**Objective:** a working service that provably matches the training pipeline.

**Tasks:** the §9 structure; lifespan loading; the four endpoints; three-way `MODEL_URI` resolution; unseen-category handling; **the parity test**; latency benchmark.

**Common mistakes:** loading the model per request; returning 500 for unseen categories; letting any preprocessing logic leak into the API layer (it all belongs in the pyfunc).

**Tests:** health when loaded and when not; single and batch predict; 422 on malformed input; batch size cap; unseen category returns 200 with a warning; **parity — 20 rows from `tests/fixtures/` through the training pipeline and through the API TestClient, the same artifact on both sides, predictions equal within 1e-6** (§18 A3, A6).

**Completion:** parity green; p50/p95 recorded.

```
Read CLAUDE.md and PLAN.md §9.

Phase 7: the FastAPI service.

Structure exactly as §9. Requirements:
- Model loaded ONCE via an asynccontextmanager lifespan handler
- MODEL_URI resolution: registry alias → pinned version → local path
  (the local path is what CI uses)
- /health returns 503 if the model failed to load
- Unseen categorical values map to the __unknown__ sentinel and return
  200 with a `warnings` field — never a 500
- Batch capped at 1000 records → 400 if exceeded
- Zero preprocessing logic in the API layer; the pyfunc does all of it

Then the test that matters most, tests/test_parity.py:
  Take 20 rows from tests/fixtures/ — NEVER the promotion evaluation
  set, which only score_holdout.py may load (§18 A3). Run them through
  the training-time path (PreprocessingArtifact + model directly). Run
  the same 20 raw records through the API via TestClient using the same
  locally-saved fixture model and the SAME artifact. Assert
  probabilities match within 1e-6.
  Parity is scoped to ONE fixed artifact. Snapshot selection is outside
  its scope: training joins each order to its own purchase-month
  snapshot, serving uses the latest bundled one (§18 A6).

Finally scripts/benchmark_latency.py — p50/p95 for single and batch-100.

Write test_parity.py BEFORE the API implementation.
```

---

### Phase 8 — Docker and Compose (3 h, medium)

**Objective:** `docker compose up` gives a working stack from clean, once `make bootstrap` has registered a first model (§18 A5).

**Common mistakes:** no `.dockerignore` (huge build context); bare `depends_on` racing Postgres; installing the full training stack into the API image; running as root.

**Tests:** image builds; container passes healthcheck; `/predict` works against the containerized service.

**Completion:** `docker compose down && docker compose up` → healthy stack, working prediction. **Never pass `-v`** — it destroys named volumes, and from the wrong directory it destroys another project's (§18 A5).

```
Read CLAUDE.md and PLAN.md §10.1, §10.2.

Phase 8: containerize the API and complete Compose.

1. Multi-stage Dockerfile on python:3.12-slim. Builder installs
   requirements-api.txt (NOT the training stack). Runtime copies
   site-packages + api/ + the src modules the pyfunc needs.
   Non-root user. HEALTHCHECK on /health. Under 40 lines.
2. requirements-api.txt — the minimal serving subset. No shap, optuna,
   catboost or matplotlib unless the pyfunc genuinely needs them at
   load time. Tell me the image size before and after this split.
3. .dockerignore excluding data/, mlartifacts/, mlflow.db, .git,
   notebooks/, __pycache__, .venv
4. Add the api service to docker-compose.yml with
   depends_on: {postgres: service_healthy, mlflow: service_healthy}
5. Optional trainer service under profiles: ["training"]
6. .env.example with every variable Compose reads

Verify: docker compose down && docker compose up -d, wait for
healthy, curl /health and /predict. Report the results.
Never pass -v to down; it deletes named volumes (§18 A5).
```

---

### Phase 9 — Retraining lifecycle and promotion gate (3.5 h, medium)

**Objective:** two real model versions and a gate that actually decides.

**Tasks:** `score_holdout.py`; `promote.py` with the §8 gate; train v2 on the extended window with refreshed snapshots; run the gate; write `reports/promotions.md`; optional drift report.

**Common mistakes:** re-fitting anything on the holdout; changing the holdout window between v1 and v2 (then the comparison is meaningless); making the gate always pass.

**Tests:** the gate rejects a deliberately worse challenger; it accepts a clearly better one; it rejects on a schema mismatch; the holdout is loaded by exactly one module (assert via import inspection).

**Completion:** v1 and v2 both registered, both scored on the identical holdout, a decision recorded with its reason.

```
Read CLAUDE.md and PLAN.md §8, §4.4.

Phase 9: retraining lifecycle and promotion gate.

1. src/evaluation/score_holdout.py — the ONLY module permitted to load
   the holdout window. Add a module-level comment saying so.
2. src/registry/promote.py implementing the §8 gate:
   - score @champion and @challenger on the identical holdout
   - promote only if PR-AUC > champion + min_delta (configs/promotion.yaml)
     AND Brier not worse beyond tolerance AND p95 latency within budget
     AND feature schema matches
   - on pass: move the champion alias, tag the old version archived
   - on fail: leave champion alone, log the reason
   - append the decision to reports/promotions.md either way
3. Train v2 on 2017-05 → 2018-04 with refreshed aggregate snapshots.
   Run the gate for real.
4. tests/test_promotion_gate.py — a deliberately worse challenger is
   rejected; a clearly better one is accepted; a schema mismatch is
   rejected regardless of metrics.

If v2 loses the gate, DO NOT tune until it wins. Record the loss.
A gate that rejects is a working gate.
```

---

### Phase 10 — Full CI/CD (3 h, medium)

**Objective:** the pipeline in §10.3, green.

**Common mistakes:** CI that needs the real dataset; building without smoke-testing; leaking secrets into the image.

**Completion:** green badge; a `:{sha}`-tagged image in GHCR; the smoke test would catch a broken model load.

```
Read CLAUDE.md and PLAN.md §10.3.

Phase 10: complete CI/CD.

1. tests/fixtures/generate_synthetic.py — a few hundred rows matching
   the orders_analytical schema with plausible distributions and a
   ~7% positive rate. Committed. CI never needs the real Olist data.
2. Commit a tiny fixture model (10-tree LightGBM in the pyfunc wrapper,
   a few KB) for API tests.
3. Extend ci.yml: ruff, black --check, then pytest with a postgres:16
   SERVICE CONTAINER (so ETL integration tests run for real), coverage
   summary in the job output.
4. New build.yml on push to main: build the image, docker run it,
   wait for healthy, curl /health and POST /predict with a golden
   fixture payload asserting a sensible probability, then push to GHCR
   tagged :latest and :{sha} using GITHUB_TOKEN.
5. README badges.

Make the smoke test strict enough that a broken model load fails the
build. A build step that only checks the image exists proves nothing.
```

---

### Phase 11 — Documentation and interview prep (2.5 h, easy)

**Tasks:** README with the architecture diagram, results table (baselines → single models → ensemble, on the holdout), the honest ensemble delta, latency numbers, the promotion decision, one-command reproduction, and a **limitations** section. Then write `reports/interview_notes.md` answering, in your own words:

- Why PR-AUC over ROC-AUC here?
- Walk me through the leakage risks and what you did about each.
- Why monthly as-of snapshots rather than per-row trailing windows?
- How does a new model reach production? How would you roll back?
- What prevents train/serve skew?
- Did the ensemble help? Was it worth it?
- Why no Airflow / Kubernetes / cloud?
- What breaks first at 100x traffic?
- What would you do differently with another month?

A limitations section is not a weakness. It's the clearest signal that you understand your own system, and interviewers read it that way.

---

## 14. Working notes

### 14.1 `CLAUDE.md` template

```markdown
# Delivery Delay Prediction Platform

## What this is
Predicts at checkout whether an Olist order will be delivered after its
promised date. The project's purpose is the MLOps lifecycle, not the model.

## Non-negotiable invariants
1. PREDICTION-TIME CONTRACT: only data available at order_purchase_timestamp
   may become a feature. Denylist: order_approved_at,
   order_delivered_carrier_date, order_delivered_customer_date,
   order_status, all review columns.
2. As-of aggregates for snapshot month M use only orders whose DELIVERY
   OUTCOME was known before M (order_delivered_customer_date < M).
   Filtering on order_purchase_timestamp leaks future outcomes, and the
   leak is biased toward late orders — they take ~3x longer to resolve.
3. The 2018-05 → 2018-08 window is loaded ONLY by
   src/evaluation/score_holdout.py, called only by the promotion gate.
   The parity test uses tests/fixtures/, never that window. It is a
   PROMOTION EVALUATION set, not an untouched generalization estimate.
4. All preprocessing lives inside the MLflow pyfunc wrapper. The API layer
   contains none.
5. Time-based CV only. StratifiedKFold/KFold are forbidden.
6. Split dates come from configs/splits.yaml, which is keyed PER VERSION.
   Fitting and calibration windows must be disjoint for every version.
7. Tests are written in the same phase as the code they cover.
8. Parity means: same raw record + same artifact -> same probability within
   1e-6. Snapshot selection is outside parity's scope; serving uses the
   latest bundled snapshot by design.
9. pandas 3 copy-on-write: no chained assignment, never mutate a slice.
10. Host ports come from .env. The API publishes on host 8001 — 8000
    belongs to the realtime-fraud-detection project on this machine.
    Never `docker compose down -v`; never `docker system/image prune -a`.

## Stack
Python 3.12 · PostgreSQL 16 (Compose) · pandas 3 · LightGBM/XGBoost/CatBoost ·
Optuna · SHAP · MLflow 3 (SQLite backend) · FastAPI · Docker Compose ·
GitHub Actions → GHCR

## Style
- Type hints on public functions; Google-style docstrings
- ruff + black, line length 100
- Config via pydantic-settings, never os.environ scattered through modules
- Log with `logging`, never print
- Fail loudly: raise on contract violations, don't warn and continue

## Scope discipline
Do not add Airflow, Kafka, Kubernetes, cloud services, a feature store,
a UI, or microservices. If a change isn't in PLAN.md, ask before writing it.

## Current phase
Phase N — <objective>. Do not implement future phases.
```

### 14.2 Daily rhythm

Start each session by re-reading `CLAUDE.md` and the phase section. End each session with a green test suite and a commit. Never leave a phase half-finished overnight — you'll lose the context and Claude Code will lose it too.

---

## 15. Difficulty (Step 16)

**Recommended version: 6 / 10.**

The genuinely hard parts are the as-of aggregate design (Phase 3), the pyfunc wrapper plus parity guarantee (Phases 6–7), and the promotion gate (Phase 9). Everything else is careful assembly. The intellectual difficulty sits in the leakage reasoning, not the tooling.

**Version to avoid: 8.5 / 10.** Airflow orchestration, Kubernetes deployment, a Feast feature store, Evidently monitoring with alerting, cloud infrastructure via Terraform, and a Streamlit dashboard. That version takes 3–4x the time, and in an interview you'd be defending eleven technologies at surface depth instead of six at real depth. Depth beats breadth every time — a single question at depth exposes breadth-only knowledge immediately.

**Floor version: 4 / 10** — flat CSV, random CV, one model, `model.pkl` in the repo, a Dockerfile. Indistinguishable from thousands of other student projects.

---

## 16. Final CV bullets (Step 17)

Write these **after** the build, with your real numbers. Bracketed values are placeholders — replace them, never guess them.

**Delivery Delay Prediction Platform** — *Python, PostgreSQL, MLflow, LightGBM/XGBoost/CatBoost, FastAPI, Docker, GitHub Actions*

- Built a reproducible ETL pipeline loading a 9-table e-commerce dataset (~100K orders) into PostgreSQL, producing an order-level analytical table with Pandera schema validation and leakage-safe as-of monthly aggregate features computed strictly from prior-period data.
- Engineered [37] features across 8 families and trained LightGBM, XGBoost and CatBoost models with Optuna tuning under time-series cross-validation, reaching [0.XX] PR-AUC against a [0.XX] logistic-regression baseline on a frozen out-of-time holdout.
- Designed a full model lifecycle in MLflow: experiment tracking, `pyfunc`-packaged models bundling preprocessing, calibration and decision threshold, registry versioning with champion/challenger aliases, and an automated promotion gate comparing retrained models on a fixed holdout.
- Served predictions through a containerized FastAPI service resolving the champion model from the registry at startup, with single and batch endpoints, [X] ms p95 latency, and an automated train/serve parity test guaranteeing inference matches the training pipeline to 1e-6.
- Automated CI/CD with GitHub Actions — linting, [XX] tests against a PostgreSQL service container, container build with a live smoke test, and publication of versioned images to GHCR.

### What changed from your original draft, and why

| Original | Issue | Replacement |
|---|---|---|
| "AI Enterprise Suite" | Says nothing; implies multiple apps | Named for what it does |
| "36 features" | Invites "name them"; only safe if grouped | Real count, defended by 8 families |
| "ensemble of CatBoost, LightGBM, XGBoost" | Implies the ensemble shipped | State what actually shipped after the honest delta |
| "SHAP for interpretability" | Vague | Folded into the lifecycle bullet; details go in the README |
| "deployment of the model-serving service" | Implies a live hosted service | "publication of versioned images to GHCR" — true, and still a real CD step |
| — | Missing entirely | Parity test, promotion gate, leakage-safe features — the three most distinctive things you'll build |

The last row is the important one. Parity testing and promotion gates are things most candidates have never implemented, and both are one sentence to explain and impossible to fake.

---

## 17. Before you start

1. Download the Olist dataset from Kaggle and confirm all 9 CSVs are present.
2. In your Linux environment, install Docker Engine + the Compose plugin, add yourself to the `docker` group, and confirm `docker run hello-world` works without sudo.
3. Create the repo, write `CLAUDE.md` from §14.1 by hand — don't generate it.
4. Re-read §4.2 and §4.5. Those two sections are the project.
5. Read §18. A design review found six defects in the first draft; the amendments are binding, and the body of this document has been patched to match them.
6. Start Phase 0.

---

## 18. Amendments (design review, 2026-09-28)

A review of v1 of this plan found six defects, plus one change forced by library
versions. All are resolved below and the body of the document has been patched to
match. **Where the body and this section disagree, this section wins.** The original
unamended plan is preserved in git history at commit `a098948` — the diff is the record
of what changed and why, and it is worth reading before an interview, because "I found
a target-correlated leak in my own design and measured it" is a stronger answer than a
plan that was simply right the first time.

### A1 — As-of aggregates leaked future outcomes *(severity: high)*

**Defect.** §4.5 built snapshot *M* from orders with `order_purchase_timestamp < M`. An
order purchased 30 January and delivered 10 February has no known outcome on 1 February,
yet it contributed its late/on-time label to the 1 February snapshot.

**Evidence** (measured on `data/raw/`, 96,470 delivered orders):

| snapshot cutoff | orders admitted | unresolved at cutoff | late% resolved | late% **unresolved** |
|---|---|---|---|---|
| 2017-07-01 | 14,199 | 7.7% | 3.20% | 11.25% |
| 2017-10-01 | 26,414 | 6.0% | 3.07% | 12.17% |
| 2018-01-01 | 43,693 | 5.7% | 4.28% | 27.68% |
| 2018-04-01 | 64,320 | 6.0% | 5.97% | 39.10% |

Median delivery lag is **31 days for late orders against 9 for on-time** ones, so orders
still in flight at a cutoff are mechanically 3-7x more likely to be late. The leak is
therefore *biased toward the positive class* and inflates precisely the feature families
(F6/F7/F8) that §4.5 calls the most predictive available.

**Resolution.** Snapshot *M* admits an order only if
`order_delivered_customer_date < first day of M`. The Phase 3 test asserts on delivery
date, not purchase date. Snapshots thin slightly and the warm-up window matters more;
with a 10-day median lag, four months remains ample.

### A2 — v2 trained on its own validation window *(severity: high)*

**Defect.** §4.4 reserved 2018-01→02 for blend weights, calibration and threshold, then
had v2 train on 2017-05→2018-04, which contains it. Every quantity §6.5 fits on the
validation period would have been in-sample for v2.

**Resolution.** Windows are per version, and `configs/splits.yaml` is keyed by version:

| | fit | calibrate / select | promotion evaluation |
|---|---|---|---|
| **v1** | 2017-05 → 2017-12 | 2018-01 → 2018-02 | 2018-05 → 2018-08 |
| **v2** | 2017-05 → 2018-02 | 2018-03 → 2018-04 | 2018-05 → 2018-08 |

Fitting and calibration stay disjoint, the evaluation window is untouched by either, v2
still sees more data than v1 (10 months against 8), and a sliding validation window is
what a real retraining cadence does anyway.

### A3 — Two required tests contradicted each other *(severity: high)*

**Defect.** Phase 7's prompt said *"take 20 rows from the holdout"* for the parity test.
Phase 9's test asserts *"the holdout is loaded by exactly one module, via import
inspection."* Both cannot pass.

**Resolution.** Parity uses `tests/fixtures/`, which §10.3 already mandates because CI
cannot use the real dataset. No extra work — strictly less.

Separately: the 2018-05→08 window decides promotions, so it is a **promotion evaluation
set**, not a pristine generalization estimate. With two gate evaluations the selection
bias is negligible, but the README and CV must name it accurately rather than imply it
was never used for selection. Carving out a third period was considered and rejected: it
costs evaluation-set size for a bias that two evaluations cannot meaningfully induce.
State that trade-off in the limitations section instead of hiding it.

### A4 — `shipping_limit_date` availability was asserted, not shown *(severity: low)*

**Defect.** §4.2 declared it available at checkout with no evidence.

**Evidence.** Across 110,189 order items: **zero** rows have `shipping_limit_date`
earlier than `order_purchase_timestamp`; the offset has a hard floor at 2.00 days,
median 6.01, p99 18.96. The distribution sits just above whole-day values, consistent
with `purchase_ts + N days` assigned at order time.

**Resolution.** Keep the feature — the evidence supports the claim. Two caveats:
`max = 1,052 days` is a data-quality outlier needing winsorizing, and only 3.3% of
offsets are exact whole days across 55 distinct values, so it is not one clean rule.
Audit `days_to_shipping_limit` for near-collinearity with `promised_days` in the §5
correlation pass and drop it if it adds nothing.

### A5 — No bootstrap path on a clean machine *(severity: medium)*

**Defect.** §9 has the API resolve `@champion` at startup and fail loudly otherwise.
§10.2 claimed `docker compose up` was the entire deployment story. On a fresh clone the
registry is empty, so both cannot hold.

**Resolution.** An explicit ordered bootstrap, exposed as `make bootstrap`:

```
postgres + mlflow up  →  load-raw  →  build-orders  →  features
                      →  train v1  →  register  →  promote (uncontested)
                      →  api up
```

The README's one-command claim is scoped to "after bootstrap." `docker compose up`
remains the steady-state story. Also: **never `docker compose down -v`** as a routine
check — it deletes named volumes, and run from the wrong directory it deletes another
project's. Likewise never `docker system prune -a` or `docker image prune -a` on this
machine.

### A6 — Training and serving snapshot policies differ *(severity: medium)*

**Defect.** Training joins each order to its *own* purchase-month snapshot; serving uses
the *single latest* snapshot bundled in the artifact. "The API matches the training
pipeline" was therefore undefined — feed a historical order and the two paths legitimately
disagree.

**Resolution.** Parity is a property of an artifact, not of history:

> **same raw record + same artifact → same probability within 1e-6**

Snapshot *selection* is explicitly outside parity's scope. The training-versus-serving
asymmetry is intentional and is how batch feature systems behave — §4.5 already argues
this — but it is a genuine distribution shift and belongs in the limitations section, not
papered over. If historical scoring is ever needed, the pyfunc takes an optional as-of
date and defaults to latest; that is out of scope for v1.

### A7 — Library versions moved a major release *(severity: low, but pervasive)*

Resolved fresh on 2026-09-28 per §11's own instruction not to pin versions from this
document. All 117 packages install as binary wheels on Python 3.12 — no source builds.

| | plan assumed | resolves to | consequence |
|---|---|---|---|
| Python | 3.11 | **3.12.3** | one interpreter everywhere; a `pip freeze` lockfile is only valid for the interpreter that made it |
| MLflow | 2.16+ | **3.16.1** | stages *removed*, so §7's alias design is the only option; `log_model(name=)` replaces `artifact_path=` |
| pandas | 2.x | **3.0.6** | copy-on-write mandatory, PyArrow-backed strings default — write CoW-correct code from the start, retrofitting is miserable |
| Optuna | 3.x | 5.0.0 | API stable for `timeout`, TPE, pruners |
| numpy / pytest | — | 2.5.3 / 9.1.1 | no action |

MLflow 3 strengthens the narrative rather than weakening it: stages being gone makes the
champion/challenger alias design mandatory rather than merely current.

### Not changed

The architecture, scope boundaries, §3 cut-order, model selection, MLflow run structure,
CI/CD design and §12 repository layout all survived review unaltered. None of these
findings challenged the design; they are corrections within it.
