# DECISIONS — delivery-delay-platform

Decision log. **Add an entry whenever a non-obvious decision is made**, in the same commit as
the change. One entry = what was decided, why, and what was rejected.

Design-review amendments live in `PLAN.md` §18 and are **cross-referenced here, not
duplicated** — §18 is the authority, and two copies would drift apart.

| ID | Decision | Date |
|---|---|---|
| [D1](#d1) | Python 3.12, not 3.11 | 2026-09-28 |
| [D2](#d2) | API on host port 8001 | 2026-09-28 |
| [D3](#d3) | Take current library majors (MLflow 3, pandas 3) | 2026-09-28 |
| [D4](#d4) | GitHub repo `delivery-delay-platform`, private | 2026-09-28 |
| [D5](#d5) | Amend `PLAN.md` in place after committing the original | 2026-09-28 |
| [D6](#d6) | Six design-review amendments (§18 A1-A6) | 2026-09-28 |
| [D7](#d7) | Olist row counts from a CSV parser, never `wc -l` | 2026-09-28 |
| [D8](#d8) | Interim `.gitignore` before Phase 0 | 2026-09-28 |
| [D9](#d9) | Grow the LV rather than resize the VMware disk | 2026-09-28 |
| [D10](#d10) | Three living docs; generate `CLAUDE.md` | 2026-09-28 |
| [D11](#d11) | No AI attribution in commit messages | 2026-09-28 |
| [D12](#d12) | MLflow: the server owns the SQLite store; clients go over HTTP | 2026-09-28 |
| [D13](#d13) | Compose healthchecks force TCP, avoid curl, and run MLflow as uid 1000 | 2026-09-28 |
| [D14](#d14) | `requirements-dev.txt` standalone; `mlflow-skinny` for the API subset | 2026-09-28 |
| [D15](#d15) | Added `Makefile` and pre-commit in Phase 0, with Phase 0 targets only | 2026-09-28 |
| [D16](#d16) | Pin the CI runner OS and the action majors | 2026-09-28 |
| [D17](#d17) | Raw timestamps are `TIMESTAMP`, never `TIMESTAMPTZ` | 2026-09-28 |
| [D18](#d18) | The `raw` schema is a faithful landing zone | 2026-09-28 |
| [D19](#d19) | Quarantine `order_delivered_customer_date` in `features.order_outcomes` | 2026-09-28 |
| [D20](#d20) | The late rate is strongly non-stationary across the §4.4 windows | 2026-09-28 |
| [D21](#d21) | `ANALYZE` explicitly after every bulk load | 2026-09-28 |

---

## D1
### Python 3.12, not the plan's 3.11
**Decision.** Use the Ubuntu 24.04 system interpreter, 3.12.3, and set it identically in
`python:3.12-slim`, `ci.yml` and `CLAUDE.md`. `PLAN.md` §11 and §18 A7 updated.

**Why.** Verified before committing to it: all 117 packages resolve on 3.12 as binary wheels
with zero source builds. A `pip freeze` lockfile is only valid for the interpreter that
produced it, so local/Docker/CI must match exactly or the reproducibility claim is false.

**Rejected.** Adding the deadsnakes PPA for 3.11 to match the plan literally — a second
interpreter to manage forever for no benefit.

## D2
### API publishes on host port 8001
**Decision.** Host 8001, container-internal 8000. Host ports come from `.env`
(`API_PORT`, `POSTGRES_PORT`, `MLFLOW_PORT`), bound to `127.0.0.1`.

**Why.** The `realtime-fraud-detection` project on this machine publishes 6379, **8000** and
8501. Its api service is currently stopped, so the clash would only appear the first time
both stacks ran together. Keeping the container port at 8000 means the Dockerfile
`HEALTHCHECK`, CI's `docker run`, and PLAN §9/§10.2 need no change.

**Rejected.** Taking 8000 and stopping the other project on demand — a trap for later.

## D3
### Take current library majors rather than pinning to plan-era versions
**Decision.** MLflow 3.16.1, pandas 3.0.6, Optuna 5.0.0, numpy 2.5.3. See §18 A7.

**Why.** §11 instructs "do not pin versions from this document — install fresh, then freeze
what actually resolves." MLflow 3 *removes* stages, which makes §7's champion/challenger
alias design mandatory rather than merely current — the narrative gets stronger. Shipping a
2026 portfolio project on MLflow 2.x invites an awkward question.

**Cost accepted.** pandas 3 makes copy-on-write mandatory and PyArrow-backed strings the
default, so code must be CoW-correct from the start; retrofitting is miserable. Hence
`CLAUDE.md` invariant 9. MLflow 3's `log_model` takes `name=` where 2.x took
`artifact_path=`.

**Rejected.** Pinning MLflow and pandas to the plan-era majors — defensible, but older and
with no upside.

## D4
### GitHub repo is `delivery-delay-platform`, private
**Decision.** Remote `git@github.com:Ankit-c-lang/delivery-delay-platform.git`, private,
while the local folder stays `delay-prediction`.

**Why.** PLAN §0.1 says use the long form on GitHub. The name mismatch is harmless because
Compose pins `name: delay-prediction` explicitly rather than deriving it from the directory.
Private now, flippable to public whenever the work justifies it.

## D5
### Amend `PLAN.md` in place, after committing the original unmodified
**Decision.** Commit `PLAN.md` as drafted (`a098948`), then patch 30 spots in the body and
append §18 (`1d76389`). Where §18 and the body disagree, §18 wins.

**Why.** Two reasons. First, PLAN §13 runs one phase per fresh session, each opening with
"read PLAN.md §x" — a stale phase prompt would actively produce the defect. Phase 7's prompt
said "take 20 rows from the holdout" while Phase 9's test asserts exactly one module may load
it; following both verbatim writes a test that breaks another test. Second, committing the
original first means the diff *is* the record, so nothing is lost by editing — and "I found a
target-correlated leak in my own design and measured it" is a stronger interview answer than
a plan that was simply right first time.

**Rejected.** An amendments-only appendix leaving the body stale (the damaging instructions
survive in the phase prompts); relying on `CLAUDE.md` and memory alone (the phase prompts are
what actually gets handed over).

## D6
### Six design-review amendments
**Decision.** All six findings accepted and resolved. Full detail, evidence and reasoning in
`PLAN.md` §18 — summarised here only as an index:

| | Defect | Resolution |
|---|---|---|
| **A1** | As-of snapshots admitted orders with no known outcome at the cutoff | filter on `order_delivered_customer_date < M` |
| **A2** | v2's training window contained its own validation window | splits keyed per version, fit/calibration disjoint |
| **A3** | Parity test read the holdout, contradicting Phase 9's one-module assertion | parity uses `tests/fixtures/`; window renamed a promotion evaluation set |
| **A4** | `shipping_limit_date` availability asserted, not shown | evidenced (0 of 110,189 precede purchase); kept, winsorize the tail |
| **A5** | `docker compose up` cannot work on a clean machine | ordered `make bootstrap`; `down -v` dropped from checks |
| **A6** | Training and serving snapshot policies differed, leaving parity undefined | parity scoped to one fixed artifact |

**A1 is the one to remember.** It was measured, not theorised: late orders take a median 31
days to deliver against 9 for on-time ones, so orders still in flight at a snapshot cutoff
are **3-7x more likely to be late**. The leak was therefore biased toward the positive class
and inflated the most predictive feature families in the project.

**Rejected** (A3): carving out a third untouched period for an independent final estimate. It
costs evaluation-set size to correct a bias that two gate evaluations cannot meaningfully
induce. The trade-off is stated in the limitations section instead of hidden.

## D7
### Olist row counts come from a CSV parser, never `wc -l`
**Decision.** Expected counts in the Phase 1 reconciliation test are produced with
`csv.reader`/pandas. `CLAUDE.md` invariant 11.

**Why.** `wc -l` is wrong for 2 of the 9 files: `olist_order_reviews_dataset.csv` is 99,224
rows not 104,719 (review comments contain newlines inside quoted fields) and
`product_category_name_translation.csv` is 71 not 70 (no trailing newline). Postgres
`COPY ... FORMAT csv` parses quoted newlines correctly, so a `wc -l` expectation fails
spuriously — and the tempting "fix" is to break the loader to match a wrong number. 99,224 is
also the canonical published count, which confirms the download is intact.

## D8
### Interim `.gitignore` written before Phase 0
**Decision.** A four-rule `.gitignore` (`data/`, `.venv/`, `.env`, `__pycache__/`, plus
`kaggle.json`) created at `git init` time, to be replaced by the full §12 version in Phase 0.

**Why.** The repo existed and the dataset was about to be downloaded, so there was a window
in which `git add .` would stage 121 MB of CC BY-NC-SA data. Verified by test: a throwaway
`data/raw/fake.csv` was correctly ignored, and `git check-ignore -v` named the rule.

## D9
### Grow the logical volume instead of resizing the VMware disk
**Decision.** `lvextend -l +100%FREE` then `resize2fs`, taking root from 39 GB to 77 GB and
free space from 9.9 GB to 48 GB.

**Why.** The Ubuntu installer had allocated only half of the 78 GB LVM partition — `vgs`
showed 39 GB unallocated already inside the volume group. No VMware change, no reboot, ext4
grows online. The project needs ~8-9 GB and 9.9 GB free would have hit `ENOSPC` mid-Phase-8.

**Noted risk.** `sda` is an 80 GB thin-provisioned vmdk, so the backing file grows on the
Windows host as space is used, and a guest `df` cannot see host capacity. Host has 120 GB
(C:) / 112 GB (D:) free, so the full 80 GB ceiling is absorbable. `vmware-toolbox-cmd disk
shrink /` reclaims host space later (`open-vm-tools` 13.0.10 present).

## D10
### Three living docs, and `CLAUDE.md` is generated
**Decision.** `CLAUDE.md` (invariants and working agreement), `PROGRESS.md` (where we are,
updated every task), `DECISIONS.md` (this file, updated on every non-obvious call).

**Why.** PLAN §13 runs one phase per fresh session, so anything not written down is lost
between phases. `PROGRESS.md` follows the convention already working in
`realtime-fraud-detection`. A separate decision log keeps `PROGRESS.md` about status rather
than accumulating rationale.

**Deviation from the plan.** PLAN §17 step 3 says write `CLAUDE.md` by hand and don't
generate it — the reasoning being that hand-typing the invariants is how you come to own
them. The user chose to have it generated instead. The invariants still need reading closely,
especially 2, 3 and 8, because they are the ones a design review had to correct.

## D11
### No AI attribution in commit messages
**Decision.** Commit messages carry no `Co-Authored-By` line and no tool credit of any kind.
Plain imperative mood. Recorded in `CLAUDE.md` under Working agreement.

**Why.** User instruction, 2026-09-28. Applies to every commit.

**Applied retroactively.** The three commits made before the instruction carried trailers, so
their messages were rewritten with `git filter-branch --msg-filter` and force-pushed on
2026-09-28. Safe because the repository is private with no forks, no other branches and no
open pull requests, so nobody else's history was disturbed. Verified: every commit tree is
identical before and after — only messages changed. The old SHAs are dead; the mapping is
`d63c6c0` -> `a098948`, `8bce10c` -> `1d76389`, `17ec901` -> `80b9e21`.

## D12
### MLflow: the tracking server owns the SQLite store; clients talk HTTP
**Decision.** The backend store `sqlite:////mlflow/mlflow.db` exists only inside the `mlflow`
container. Everything else — training, promotion, the API — reaches MLflow through
`MLFLOW_TRACKING_URI=http://127.0.0.1:5000`, and the server is started with
`--serve-artifacts --artifacts-destination /mlartifacts` so artifact reads and writes are
proxied through it too. The tracking URI lives in `.env`, deliberately **not** in
`configs/base.yaml`, because it differs between the host (`127.0.0.1:5000`) and a container
(`mlflow:5000`); `base.yaml` keeps only the experiment name, model name and aliases.

**Why.** `PLAN.md` §7 says "tracking URI `sqlite:///mlflow.db`", which reads as if client code
should open the SQLite file directly. That works exactly until something runs in a container:
SQLite over a shared mount has no useful locking story, and the artifact URI recorded in a run
would be a filesystem path that resolves on one side of the mount and not the other. Proxying
proves it works — the default experiment reports
`artifact_location: mlflow-artifacts:/0`, not `/mlartifacts/0`.

**Rejected.** Pointing host code at the SQLite file (breaks as soon as Phase 8 containerizes
anything, and it is the standard MLflow footgun). Postgres as the backend store (a second
database for no gain; the registry needs *a* file/DB backend, not a good one).

## D13
### Compose healthchecks: force TCP for Postgres, avoid `curl` for MLflow, run MLflow as uid 1000
**Decision.**

```yaml
postgres: test: ["CMD-SHELL", "pg_isready -h 127.0.0.1 -U $$POSTGRES_USER -d $$POSTGRES_DB"]
          interval 5s · timeout 5s · retries 10 · start_period 30s
mlflow:   test: ["CMD", "python", "-c", "...urllib.request.urlopen('http://localhost:5000/health')..."]
          interval 10s · timeout 5s · retries 5 · start_period 20s
mlflow:   user: "1000:1000"
```

**Why, per line.**
- `-h 127.0.0.1` is load-bearing. During `initdb` the official Postgres entrypoint runs a
  *temporary* server with `--listen_addresses=''`, i.e. Unix socket only. A bare `pg_isready`
  reaches that socket and reports healthy while the real server is still seconds away, so
  anything with `condition: service_healthy` starts and gets `connection refused`. Forcing TCP
  means the check can only pass once the real server is listening.
- `$$` escapes the variable so Compose passes it through and the *container's* shell expands
  it. A single `$` would interpolate on the host, at parse time, to an empty string.
- `start_period: 30s` covers first-boot `initdb`; failures inside it do not count toward
  `retries`. 5s × 10 retries after that is a generous ceiling for a restart.
- Python, not `curl -f`, for MLflow: `ghcr.io/mlflow/mlflow` does not reliably ship `curl` or
  `wget`, and a healthcheck whose binary is missing fails permanently while looking like a
  service fault. `python` is guaranteed present in that image.
- `user: "1000:1000"` keeps files written into the `./mlflow` and `./mlartifacts` bind mounts
  owned by the host user. Without it `mlflow.db` and every artifact land root-owned and need
  `sudo` to clean up. `id -u` is 1000 here. Postgres keeps its default user — it needs root to
  drop privileges during `initdb`, and its data lives in a named volume anyway.

**Rejected.** Bare `depends_on` (the race §10.2 explicitly warns about). `pg_isready` without
`-h`. `curl -f http://localhost:5000/health`. Named volumes for MLflow — the run database and
the artifacts are the project's record and should be readable from the host.

## D14
### `requirements-dev.txt` is standalone; the API subset uses `mlflow-skinny`
**Decision.** `requirements-dev.txt` does **not** `-r requirements.txt`. `requirements-api.txt`
lists `mlflow-skinny` instead of `mlflow`, and omits optuna, shap, matplotlib, pandera and
catboost.

**Why.** Phase 0 CI only lints. Making it install catboost, shap and matplotlib to run
`ruff check` would add minutes to every push for nothing; from Phase 10 the job that runs
pytest installs both files. And the API loads a registered pyfunc and resolves an alias — it
never runs a tracking server or the UI, which is precisely what skinny drops. §11 already
states the intent ("the API image should not carry the training stack; it roughly halves the
image"); skinny is the mechanism.

**Risk, recorded on purpose.** If the Phase 5 blend keeps CatBoost, `catboost` must go back
into `requirements-api.txt` or the artifact will not load at serve time. Phase 8 must also
prove a pyfunc loads under `mlflow-skinny` and measure the image-size difference — that
number is the whole justification. Both caveats are written into
`requirements-api.txt` itself, where whoever edits it next will see them.

**Rejected.** `mlflow` in all three files (simpler, but then the API image carries the
training stack and the §11 claim is false). One requirements file with extras.

**Measured later, and it undercuts the "halves the image" claim:** a dry-run resolve of
`requirements-api.txt` pulls `nvidia-nccl-cu13`, a **305 MB** CUDA collective-communications
library, as a dependency of `xgboost`. This machine is CPU-only (i5-1235U, integrated
graphics), so it is 305 MB of code that can never execute. `xgboost-cpu` is the published
slim wheel. **Phase 8 must measure the image with and without it** before repeating any
size claim — skinny MLflow saves far less than one unusable CUDA library costs.

## D15
### `Makefile` and pre-commit added in Phase 0, with Phase 0 targets only
**Decision.** Created `Makefile` and `.pre-commit-config.yaml`, and installed the git hook.
The `Makefile` carries only targets whose code exists: `lint format test up down stop ps logs
psql health`. No `load-raw`, `train`, `promote`, `serve` or `bootstrap` yet. Whitespace hooks
exclude `PLAN.md`.

**Why.** Both files are in `PLAN.md` — the `Makefile` in the §12 tree, pre-commit in Phase 0's
*Tasks* paragraph — but neither appears in Phase 0's numbered prompt, so this is a deliberate
addition and is flagged for the user rather than assumed. A target that shells into a module
that does not exist yet is worse than no target: it fails at the point where someone trusts
it. `CLAUDE.md`'s Commands section is explicitly aspirational until the phases fill it in, and
each phase adds its own targets. `check-added-large-files` (512 KB) and `detect-private-key`
matter more here than the formatting hooks: the Olist data is CC BY-NC-SA and a committed
`kaggle.json` would be permanent history. `PLAN.md` is excluded from the whitespace hooks
because it is the frozen design document and a cosmetic rewrite would muddy any diff against
the original plan.

**Verified.** `pre-commit autoupdate` resolved the hook revisions to exactly the versions
installed locally (ruff 0.16.9, black 26.5.1), so local and hook results cannot disagree. All
10 hooks pass on every tracked file with zero modifications.

**Rejected.** Deferring both to a later phase (pre-commit's value is catching things *before*
the first push). Writing the full aspirational target list now with placeholder bodies.

## D16
### Pin the CI runner OS; track the action majors
**Decision.** `runs-on: ubuntu-24.04`, not `ubuntu-latest`. `actions/checkout@v7` and
`actions/setup-python@v7`, not v4/v5.

**Why.** The first green run emitted two annotations. `ubuntu-latest` migrates to Ubuntu 26 on
2026-10-19, which is inside this project's build window — a silent OS change mid-project is
exactly the kind of thing that costs an hour in Phase 10, when CI gains a Postgres service
container and the OS starts to matter. 24.04 is also what the dev VM runs, so CI, the VM and
`python:3.12-slim` now agree on interpreter *and* base OS; that is the same argument §11 makes
for pinning Python. Separately, checkout v4 and setup-python v5 still declare Node 20, which
GitHub force-runs on Node 24 while warning on every run; v7 of both targets Node 24 natively.

**Rejected.** Leaving `ubuntu-latest` (GitHub's recommended default, but it trades
reproducibility for freshness, and reproducibility is a claim this project has to defend).
Pinning the actions to full SHAs — right for a public repo handling secrets, overkill for a
private repo running a linter, and it makes upgrades invisible.

**Cost.** The pin needs revisiting when 24.04 is retired. Noted here so it is not a mystery.

## D17
### Raw timestamps are `TIMESTAMP`, never `TIMESTAMPTZ`
**Decision.** All eight timestamp columns in the `raw` schema are `TIMESTAMP` (without time
zone). A test asserts that no `timestamptz` column exists anywhere in the schema.

**Why.** The Olist CSVs carry naive local times with no offset — there is nothing to convert
*from*. `TIMESTAMPTZ` would make Postgres interpret each string in the session `TimeZone`,
store a UTC instant, and shift it back on read according to whatever the *client's* zone is.
Because `order_estimated_delivery_date` is always exactly midnight (verified: 0 of 99,441
values are non-midnight), any shift crosses a date boundary, and §4.3 compares at date
granularity. Demonstrated in the live database rather than argued:

```
-- one row: delivered 2017-10-18 20:30, estimated 2017-10-18 (midnight)
-- stored once as timestamp and once as timestamptz, read back in Asia/Kolkata:
 is_late_naive | is_late_tz
---------------+------------
 f             | t
```

The same row is on-time under `TIMESTAMP` and late under `TIMESTAMPTZ`. The target would then
depend on the reader's timezone — different in a container, in CI, and on a laptop — and
nothing would error.

**Rejected.** `TIMESTAMPTZ` "because it is generally the better type". It is, for data that
records an instant with a known offset. This data records neither. Also rejected: typing
`order_estimated_delivery_date` as `DATE` in raw. It *is* date-only, but raw must mirror the
file; Phase 2 casts both sides of the comparison instead.

## D18
### The `raw` schema is a faithful landing zone
**Decision.** `raw` mirrors the CSVs and nothing more. Specifically:

| Choice | Rule | Evidence |
|---|---|---|
| Table names | `olist_` prefix and `_dataset` suffix stripped: `raw.orders`, `raw.order_items` | cosmetic, and the file name is recorded in the spec |
| Column names | **exactly** as shipped, typos included (`product_name_lenght`) | renaming breaks reconciliation against the file |
| Strings | `TEXT` everywhere, no `VARCHAR(n)` | Postgres gains nothing from a length cap; a cap only rejects a future refresh |
| `*_zip_code_prefix` | `TEXT` | 24.13% of customer, 33.18% of seller, 24.57% of geolocation prefixes start with `0` |
| Money | `NUMERIC(10, 2)` | `price`, `freight_value`, `payment_value` are 2dp throughout, max 5 integer digits |
| Coordinates | `DOUBLE PRECISION` | measurements, not money; source carries up to 20 spurious decimals |
| Nullability | measured per column, never assumed | e.g. `order_approved_at` 160 nulls, `product_photos_qty` 610, `review_comment_title` 87,656 |
| Primary keys | only where verified unique and non-null | `order_reviews` is keyed `(review_id, order_id)` because `review_id` alone has **814 duplicates**; `geolocation` has **no** key — 19,015 prefixes over 1,000,163 rows |
| Foreign keys | none | a landing zone must accept the file as it is |
| Header guard | the CSV header must equal the spec, in order, before any data moves | `COPY` matches by **position**, so a re-download with two columns swapped would load `freight_value` into `price` silently |

Loads run as `TRUNCATE` + `COPY FROM STDIN` inside **one** transaction for all nine files, so
a failure on the ninth rolls back the first eight. Two tests prove this by symlinking eight
files into a temp directory and omitting or corrupting the ninth.

`reports/raw_load_reconciliation.md` is committed (a `.gitignore` exception): it is evidence
the load is complete, a cloner cannot regenerate it without the non-committable dataset, and
it contains counts rather than data.

**Rejected.** `pandas.read_csv` + `to_sql` — type inference mangles zip prefixes, and
`to_sql` on 1,000,163 geolocation rows is orders of magnitude slower than `COPY` (the whole
load is 4.1 s). Cleaning in `raw` — then there is nowhere to look when a value is
questioned. Literal hand-written DDL strings — generating from a spec keeps the
`CREATE TABLE` column order, the `COPY` column list and the header check in sync, which is
the actual failure mode worth defending against.

**Data-quality findings for Phase 2**, recorded in `PROGRESS.md` rather than here: 8 rows are
`delivered` with no delivery date, 6 `canceled` rows have one, and `order_estimated_delivery_date`
is midnight in all 99,441 rows.

## D19
### Quarantine `order_delivered_customer_date` in `features.order_outcomes`
**Decision.** `features.orders_analytical` contains **no** §4.2 denylist column, enforced by a
Pandera schema with `strict=True`. `order_delivered_customer_date` lives in a separate
two-column table, `features.order_outcomes` (`order_id`, `order_delivered_customer_date`,
`is_late`), which only the Phase 3 as-of snapshot builder may read.

**Why.** Phase 2's prompt says to "explicitly drop the §4.2 denylist columns before writing".
§18 A1 then rewrote the §4.5 snapshot rule to filter on **delivery outcome time**
(`order_delivered_customer_date < M`) instead of purchase time. Those two instructions
collide: Phase 3 cannot implement the amended rule if the column exists nowhere downstream of
raw. This is a consequence of A1 that the amendment did not propagate into Phase 2's prompt —
the same class of inconsistency §18 exists to catch.

Three ways out were considered:

1. Keep the column in `orders_analytical`, clearly labelled. Rejected: it puts a denylist
   column one careless `SELECT *` away from the feature matrix, and it makes `strict=True`
   impossible to state as "no denylist column, ever".
2. Have Phase 3 read `raw.orders` directly. Rejected: Phase 3 would have to re-derive the
   §4.3 population filter, and a second copy of that filter is how the snapshot population
   drifts from the training population.
3. **Chosen:** a separate outcome table. The population filter is applied once, the label and
   the resolution timestamp are stored together, and the boundary is a table name rather than
   a convention — so "who reads the delivery date" is answerable with `grep`.

**Consequence.** `strict=True` on the contract now means a column cannot enter the
feature-safe table without being declared in `src/etl/schema.py` in the open. A test asserts
the declared column list and the DDL are the same set, so the two cannot drift.

**Follow-up for Phase 3.** The snapshot builder joins `order_outcomes` to
`orders_analytical` on `order_id` and filters on `order_delivered_customer_date < M`. Add the
import-inspection guard §18 A3 already requires for the promotion evaluation window to this
table too, so `order_outcomes` has exactly one reader.

## D20
### The late rate is strongly non-stationary across the §4.4 windows
**Decision.** Record the measurement now and carry it into Phases 5 and 9. **Do not** move
the split boundaries to flatten it.

**Measured** on the 96,203-order population:

| Window | Orders | Late rate |
|---|---:|---:|
| warm-up 2017-01 → 04 | 7,252 | 4.66% |
| **v1 fit** 2017-05 → 12 | 36,174 | 5.85% |
| **v1 calibrate** 2018-01 → 02 | 13,624 | **9.75%** |
| **v2 fit** 2017-05 → 2018-02 | 49,798 | 6.92% |
| **v2 calibrate** 2018-03 → 04 | 13,801 | **11.84%** |
| **promotion evaluation** 2018-05 → 08 | 25,352 | **4.40%** |

Monthly extremes: 2018-03 at 18.96%, 2018-06 at 1.16% — a 16-fold spread. November 2017
(12.40% against a ~3% autumn baseline) is the Black Friday spike §4.4 predicts.

**Why it matters.** Both calibration windows carry 2.2-2.7x the base rate of the window both
models are judged on. Isotonic calibration fitted at a ~10-12% prior will **over-predict** on
a 4.4% evaluation set, and the cost-optimal threshold chosen on the calibration window will
be too low there. Phase 5 will see a poor Brier score on the evaluation set and the cause
will not be the code.

**Why not move the boundaries.** Choosing split dates to make the base rate look stable is
fitting the experiment design to the answer. The instability is a real property of this
dataset and of delivery operations, the windows follow §4.4 and §18 A2, and a production
retraining cadence faces exactly this.

**How to apply.**
- *Phase 5:* report calibration on the calibration window **and** on the evaluation set, and
  say plainly that the gap is a base-rate shift. Consider reporting a threshold swept on each.
- *Phase 9:* the promotion gate is unaffected in its comparison, because champion and
  challenger are scored on the **same** evaluation set — the shift cancels. But the absolute
  Brier tolerance must be set with these numbers in mind, not against an assumption of
  stationarity.
- *Phase 11:* this is the honest answer to "how would you know your model had drifted?" —
  the drift is already in the training data.

## D21
### `ANALYZE` explicitly after every bulk load, inside the transaction
**Decision.** `load_raw` runs `ANALYZE` on all nine raw tables after the COPYs;
`build_centroid_tables` analyzes each centroid table right after creating it; `build_orders`
analyzes the staging table. All inside the existing transaction.

**Why — measured, not theorised.** A pre-push audit ran the whole ETL into a freshly created,
genuinely empty database. `load_raw` was fine at 4.5 s, but the Phase 2 staging query ran for
**over 833 seconds before being cancelled**, against ~12 s in the working database. Same code,
same data, same machine.

The cause is statistics timing. `features.zip_centroids` and `features.state_centroids` are
created *and joined* inside one transaction, so autovacuum can never analyze them — it runs
outside the transaction and cannot see uncommitted tables. With no column statistics the
planner assumes they are tiny and chooses nested loops against the 96,203-row order side. The
working database only looked fast because earlier committed runs had left usable statistics
behind; the very first run on a clean machine was the slow one, and it is the run nobody would
have measured.

**After the fix:** build_orders on an empty database takes **16.1 s**, and the whole ETL
(`load_raw` + `build_orders`) takes 21.6 s from nothing.

**Why it matters more than a speed number.** This *is* the `make bootstrap` path §18 A5
defines for a clean machine. A first run that appears to hang for a quarter of an hour looks
like a broken pipeline, and the natural reaction — killing it and rewriting the query — would
have chased the wrong problem entirely.

**Honest caveat.** The 833 s measurement was taken while a 305 MB wheel download and a full
dependency dry-run were competing for page cache on a 7.7 GB machine, so memory pressure
inflated it to an unknown degree. The fix is correct regardless: an explicit `ANALYZE` after a
bulk load is standard practice, it removes the dependence on autovacuum timing entirely, and
the post-fix 16.1 s was measured on an equally empty database.

**Rejected.** Committing between the load and the join so autovacuum can run (gives up the
all-or-nothing transaction, which is the property two tests exist to protect). `SET
enable_nestloop = off` (treats the symptom and distorts every other plan in the statement).
