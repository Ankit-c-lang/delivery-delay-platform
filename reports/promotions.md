# Promotion decisions

Every run of `make promote` appends here, pass or fail. **The refusals are the interesting entries**: a gate that always passes is decoration (PLAN.md §8), and a challenger that loses is recorded rather than tuned until it wins (§13).

Read every PR-AUC on this page as **lift over the base rate**. The evaluation window sits at a 4.40% positive rate against the calibration window's ~9.75% (`DECISIONS.md` D20), so absolute PR-AUC is much lower here by arithmetic, before any question of model quality. The gate compares both models on the same rows, so the shift cancels in the comparison.

This window is a **promotion evaluation set, not a test set** (§18 A3): the decisions below are what it was used for.

---

## 2026-09-29 12:47 UTC — challenger v5 vs v4 — **NOT PROMOTED**

not promoted — pr_auc: PR-AUC 0.05700 against the champion's 0.05983 (delta -0.00282, bar +0.00500); in lift terms 1.30x against 1.36x at a 4.40% base rate. The two models are within the 0.005 equivalence band, so this is a tie rather than a regression: neither is demonstrably better on these rows. A tie leaves the champion in place, because promotion costs an image build, a deployment and risk, and inertia is the right default for a model that works

| Check | Result | Detail |
|---|:--:|---|
| `pr_auc` | **FAIL** | PR-AUC 0.05700 against the champion's 0.05983 (delta -0.00282, bar +0.00500); in lift terms 1.30x against 1.36x at a 4.40% base rate |
| `brier` | pass | Brier 0.053789 against the champion's 0.049307 (regression +0.004482, tolerance 0.010000) |
| `feature_schema` | pass | signature matches the serving contract |
| `p95_latency` | pass | p95 69.1 ms against a 250 ms budget |

| Metric | Challenger | Champion |
|---|---:|---:|
| rows scored | 25352 | 25352 |
| base rate | 0.04398 | 0.04398 |
| PR-AUC | 0.057 | 0.05983 |
| **lift over base rate** | 1.296 | 1.36 |
| ROC-AUC | 0.57219 | 0.61247 |
| Brier | 0.053789 | 0.049307 |
| threshold | 0.17117 | 0.17742 |
| recall | 0.23587 | 0.23049 |
| precision | 0.05824 | 0.0622 |
| flag rate | 0.17813 | 0.16299 |
| p95 latency (ms) | 69.07 | 83.94 |

Both models scored the **same rows** through the **same code path**, with nothing refit (§8). Each carries its own bundled as-of snapshot, which is deliberate: part of what a retrain buys is fresher aggregates, and neither snapshot reaches into this window (`DECISIONS.md` D33).

---
