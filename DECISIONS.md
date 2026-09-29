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
| [D22](#d22) | The as-of history features are weak, and that is the correct answer | 2026-09-28 |
| [D23](#d23) | The §18 A6 asymmetry is two named methods, not an optional argument | 2026-09-28 |
| [D24](#d24) | The feature matrix is not persisted as a table | 2026-09-28 |
| [D25](#d25) | The promotion evaluation window is locked at runtime, not by convention | 2026-09-28 |
| [D26](#d26) | Per-fold PR-AUC tracks the base rate, so lift is the comparable number | 2026-09-28 |
| [D27](#d27) | Ship LightGBM as the lead model; `scale_pos_weight` made it worse | 2026-09-28 |
| [D28](#d28) | Split the calibration window interleaved, not temporally | 2026-09-29 |
| [D29](#d29) | `purchase_month` cannot generalise forward — **dropped**, 38 features | 2026-09-29 |
| [D30](#d30) | Ship the single model, not the blend (superseded by D31 on the model) | 2026-09-29 |
| [D31](#d31) | After dropping `purchase_month`: ship XGBoost | 2026-09-29 |
| [D32](#d32) | The pyfunc bundles everything; the holdout tag waits for the gate | 2026-09-29 |

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

## D22
### The as-of history features are weak, and that is the correct answer
**Finding, not a decision to change anything.** §4.5 calls the historical performance features
"the most predictive features available". Measured on 88,951 training rows with the §18 A1 rule
applied, they are not:

| Feature | corr with `is_late` (A1 rule) | corr under the **leaky** rule | inflation |
|---|---:|---:|---:|
| `route_late_rate_hist` | **+0.0785** | +0.1037 | 1.3x |
| `seller_avg_handling_days_hist` | +0.0428 | +0.0460 | 1.1x |
| `seller_avg_delivery_days_hist` | +0.0392 | +0.0536 | 1.4x |
| `seller_late_rate_hist` | **+0.0197** | +0.0398 | **2.0x** |
| `category_late_rate_hist` | −0.0062 | +0.0070 | — |

The right-hand columns were produced by rebuilding the identical snapshots with the filter
column swapped to `order_purchase_timestamp`. **`seller_late_rate_hist` looks twice as
predictive under the leaky rule**, and `category_late_rate_hist` even changes sign. That is the
measured cost of correctness, and it is the strongest evidence that A1 was a real defect rather
than a pedantic one: half the apparent signal in the headline history feature was the future
leaking in.

**Ruled out: a time confound.** Correlations computed *within* each purchase month are
essentially unchanged (`seller_late_rate_hist` +0.0278 against +0.0197 raw), and month explains
only 3.6% of the variance in `is_late`. The weakness is not D20's base-rate shift hiding the
signal.

**Ruled out: a bug.** `order_count`, `late_rate` and `avg_delivery_days` were recomputed in SQL,
set-based, for all 26,730 seller-month snapshots and compared with a FULL OUTER JOIN: zero
mismatches. The features compute what they claim.

**So the honest reading:** on Olist, *when* and *where* an order ships predicts lateness far
better than *who* ships it. `route_late_rate_hist` is the strongest of the family at +0.079,
which fits — a route encodes distance and lane quality. Seller identity carries little once you
stop reading the future.

**`seller_is_new` carries no lift either:** 7.01% late for cold-start sellers against 6.96% for
established ones. It stays in the matrix regardless, because its job is to tell the model that
the accompanying rate was *imputed from a coarser level* rather than measured — that is a
statement about the feature vector, not a prediction.

**How to apply.**
- *Phase 4:* expect `promised_days` to dominate and F6/F7/F8 to contribute little. Do not treat
  that as a bug or go looking for a fix.
- *Phase 4 audit (§5 says drop 1-3 dead features):* `category_late_rate_hist` at −0.006 is the
  first candidate. Decide on permutation importance, not on this correlation alone.
- *§6.7 gives a leakage alarm for a feature that is implausibly **high**. This finding adds the
  mirror: if `seller_late_rate_hist` turns out to dominate the model, that is also an alarm*,
  because with the correct rule it correlates at +0.02. A dominant `seller_late_rate_hist` would
  mean the snapshot join regressed.
- *Phase 11:* "I removed a leak and watched my best feature family lose half its apparent power"
  is a better interview answer than any metric in this project.

## D23
### The §18 A6 asymmetry is two named methods, not an optional argument
**Decision.** `PreprocessingArtifact` exposes two transform methods:

| Method | Snapshot used | Path |
|---|---|---|
| `transform_with_snapshots(orders, snapshots)` | each order's **own purchase month** | training |
| `transform(orders)` | the **single latest bundled** snapshot | serving |

**Why not one method with a flag.** §18 A6 accepts that training and serving select snapshots
differently, because a live request has no history table to join to. An optional
`snapshots=None` argument would make the safe default invisible at the call site and the unsafe
one a typo away. Two names mean every call site states which semantics it wants, and a reviewer
can `grep` for `transform(` to find every place the bundled snapshot is used.

**Two tests hold the claim honest.** One forces every order's purchase month to the artifact's
bundled snapshot month and asserts the two paths then produce *byte-identical* frames — that is
what "snapshot selection is the only difference" means, and if it failed the asymmetry would be
wider than A6 admits. The mirror test asserts they *do* differ when the months differ, so the
first test cannot pass vacuously.

**Related artifact decisions, same commit.**
- **Frozen dataclass.** `fit` is a *classmethod* and there is no `fit_transform` anywhere, so no
  instance can refit itself — §6.2 asks for exactly this. Phase 5 attaches its calibrator and
  threshold through `with_decision`, which returns a **new** artifact, so one already logged to
  MLflow cannot drift under a run that referenced it.
- **`__missing__` is kept distinct from `__unknown__`.** "No category recorded" (1,330 orders)
  and "a category new since training" are different facts; collapsing them would hide a growing
  catalogue behind a data-quality problem.
- **`customer_region` uses all five IBGE macro-regions**, not whatever the training window
  contained, because the set is fixed and known. A training window with no northern customer
  must still be able to *encode* one.
- **A schema hash on load.** `load()` refuses an artifact whose expected input columns, feature
  names or order differ from the running code. A mismatch is precisely the train/serve skew this
  class exists to prevent, so it must fail loudly rather than serve one wrong prediction.

**§18 A4 is now closed with a number.** A4 asked for two things: winsorise the
`days_to_shipping_limit` tail, and audit its collinearity with `promised_days`. Measured on the
v1 fit window the correlation is **0.271** raw and **0.388** after winsorising at 30 days —
moderate, not near-collinear, so **both features stay**. The tail is winsorised at the training
q99.5, which is **21.22 days** against a maximum of 1,052.

## D24
### The feature matrix is not persisted as a table
**Decision.** `make features` fits and saves the artifact; it does **not** write a
`features.feature_matrix` table. Phase 4 recomputes the matrix through the artifact.

**Why.** A stored matrix and the artifact that produced it can drift apart, and nothing would
notice: refit the artifact, forget to rebuild the table, and training silently uses features
from a previous fitting. Recomputing costs about 1 second for 96,203 rows and exercises the
artifact on every single run, which means the Phase 7 parity path is being tested continuously
rather than only in Phase 7.

**Rejected.** Persisting the matrix for speed. There is no speed problem to solve — the
transform is ~1 s — and the staleness class of bug is the expensive kind, because it produces
plausible numbers.

**Consequence for Phase 4.** Call `build_matrix(version)` or
`artifact.transform_with_snapshots(...)`; do not look for a matrix table. The labels come from
`features.order_outcomes`, already aligned by `order_id`.

## D25
### The promotion evaluation window is locked at runtime, not by convention
**Decision.** `src/splits.py` is the only module that reads `configs/splits.yaml`, and
`promotion_evaluation_window()` raises `PromotionEvaluationLockedError` unless the caller has
entered `unlock_promotion_evaluation(caller)` and named a module in `AUTHORISED_UNLOCKERS`
(`src.evaluation.score_holdout` and `src.registry.promote`, nothing else).

**Why a runtime guard and not a comment.** §4.4 suggests "a runtime guard that raises if it's
loaded during training", and §18 A3 requires the window to have exactly one reader. A comment
saying "do not use this" is obeyed until the afternoon someone is debugging a metric. The
failure this prevents is silent: contaminating the evaluation set produces *better* numbers,
so nothing looks wrong until the model reaches production and does not perform.

**Design details that matter more than they look.**
- **The guard protects the window definition, not just the loading code.** A stray
  `orders[orders.purchase >= "2018-05-01"]` in a training notebook is the realistic mistake,
  so the *dates* are what is locked.
- **`promotion_evaluation_starts_on()` is readable without unlocking.** Knowing where the
  window begins is not the same as reading its rows, and the §6.6 ordering assertion needs
  the boundary. Without this split, that assertion would have to defeat the guard to do its
  job, and a guard routinely defeated by its own codebase is decoration.
- **It re-locks in a `finally`.** Otherwise one failed promotion run leaves the window open
  for the rest of the process.
- **Nesting is refused.** An unlock that can be layered has an extent that is hard to reason
  about, and the entire value here is that the extent is one block in one module.
- **The error message names `tests/fixtures/`.** A guard that blocks without saying what to do
  instead gets worked around rather than obeyed.

**Two inspection tests back it up**, which is §18 A3's "by import inspection" done now rather
than deferred to Phase 9: no module in `src/` outside the authorised set mentions
`promotion_evaluation_window`, and no module hardcodes a date inside the window. The second is
parsed with `ast` so that prose in a docstring — including this module's own description of the
mistake to avoid — is not mistaken for code. A line-based first version flagged exactly that and
was wrong to.

**Consequence.** `src/features/build.py::fit_window`, the inline stand-in that read
`splits.yaml` during part B, is deleted. `build_matrix` now calls `version_splits` and runs
`assert_temporal_ordering` on every build: it is cheap, and a split that has drifted should stop
the build rather than quietly train a model.

## D26
### Per-fold PR-AUC tracks the base rate, so lift is the fold-comparable number
**Finding.** LightGBM's four time-series folds scored PR-AUC 0.0547, 0.1052, 0.1471, 0.3040 — a
**5.5x spread** that looks like the model improving dramatically across the window. It is mostly
not.

| Fold | Validation window | Base rate | PR-AUC | Lift |
|---|---|---:|---:|---:|
| 1 | 2017-07-06 → 08-30 | 2.82% | 0.0547 | 1.94x |
| 2 | 2017-08-30 → 10-19 | 4.17% | 0.1052 | 2.52x |
| 3 | 2017-10-19 → 11-26 | 9.59% | 0.1471 | **1.53x** |
| 4 | 2017-11-26 → 12-31 | 9.65% | 0.3040 | 3.15x |

**Correlation between fold base rate and fold PR-AUC: +0.80.** PR-AUC is bounded below by the
base rate, so a fold validating on a high-late-rate period scores higher for free. The base
rate moves 3.4x across these folds — the same D20 non-stationarity, now showing up inside CV.

**Two consequences that are easy to miss.**

1. **The model is weakest exactly when it matters most.** Fold 3 covers November 2017, the Black
   Friday spike §4.4 predicts. It has the highest base rate and the **lowest lift, 1.53x**.
   Delivery performance degrades during the peak in ways these 39 features do not capture, so
   the model is least useful in the period an operations team would most want it. That is a
   finding to report in Phase 11, not to average away.
2. **A mean over folds with unequal base rates is not a neutral average.** It is dominated by
   the high-rate folds, so tuning on it prefers hyperparameters that do well *late* in the
   window. For a model about to be deployed forward in time that is arguably the right bias —
   but it is a choice, and it should be stated rather than inherited by accident.

**Not changed.** The mean CV PR-AUC stays the selection metric: it is what §6.4 specifies, it is
comparable *between models* on identical folds, and switching to mean lift would make this
project's numbers incomparable to the plan's. Both are now reported side by side in
`reports/phase4_models.md`.

## D27
### Ship LightGBM as the lead model; `scale_pos_weight` made it worse
**Measured** on 36,174 v1 fit rows, `TimeSeriesSplit(4)`, base rate 5.85%:

| Model | CV PR-AUC | Lift | Trials | Wall clock | Budget |
|---|---:|---:|---:|---:|---:|
| majority class | 0.0656 | 1.12x | — | — | — |
| logistic regression | 0.1357 | 2.32x | — | — | — |
| **LightGBM** | **0.1528** | **2.61x** | 40/40 | **105 s** | 900 s |
| CatBoost | 0.1489 | 2.55x | 20/20 | 373 s | 1500 s |
| XGBoost | 0.1468 | 2.51x | 40/40 | 107 s | 900 s |

**All three beat both baselines**, which is Phase 4's completion criterion, and **no study timed
out** — the whole run took ~10 minutes against a 55-minute worst case, so the §6.4 budget was
conservative for this dataset size. CatBoost is 3.5x slower than LightGBM for a slightly worse
score, exactly as §6.4 warned.

**The margins are modest and that is consistent.** The best GBDT beats logistic regression by
+0.017 PR-AUC, a 12.6% relative gain. Three GBDTs land within 0.006 of each other. Both facts
follow from D22: the history features are weak, so there is limited non-linear structure for a
tree to find beyond what a linear model already gets from `promised_days`.

**`scale_pos_weight` tested, not assumed (§6.4).** At the ratio the data implies, 16.1, LightGBM
scored **0.1456 against 0.1528 unweighted — worse by 0.0071**. §6.4 predicted exactly this:
reweighting the loss moves the probabilities without improving the ranking, and PR-AUC measures
the ranking. No resampling is used anywhere. Train on raw probabilities, calibrate in Phase 5,
then tune the threshold.

**How to apply in Phase 5.** LightGBM leads, but the three are close enough that §6.5's warning
applies in advance: a blend of three highly correlated models may gain very little for 3x
inference cost. Measure the delta and be prepared to ship the single model with the ensemble
kept as a logged experiment — §6.5 says that decision, stated plainly, is the stronger answer.

## D28
### Split the calibration window interleaved, not temporally
**Decision.** The calibration window is divided in two: blend weights are fitted on one half,
the isotonic calibrator on the other. The split is **interleaved by time order**, not temporal.
Configurable as `decision.calibration_split` in `configs/base.yaml`.

**Why split at all.** §13 Phase 5 lists "calibrating on data used to fit the blend" as a common
mistake, while asking for both the blend weights *and* the calibrator to be fitted on
validation. Those instructions collide. Splitting the window resolves it.

**Why interleaved.** A temporal split is the instinctive choice — it mirrors production, where
you calibrate on the past and apply forward. Measured on v1's window it is a bad trade:

| Split | First half | Second half |
|---|---|---|
| temporal | 6,812 rows, **5.74%** late (Jan 1-30) | 6,812 rows, **13.77%** late (Jan 30 - Feb 28) |
| interleaved | 6,812 rows, **9.84%** late | 6,812 rows, **9.67%** late |

A temporal split would fit the calibrator at 2.4x the base rate of the data the weights were
fitted on, and 3.1x the evaluation window's 4.40%. That is D20 reappearing *inside* a two-month
window. Interleaving keeps both halves at the window's own base rate.

**Why this does not violate invariant 5.** That invariant forbids random cross-validation folds
for *model* fitting, where random folds let future outcomes into as-of features. Nothing is
recomputed here: the features are already fixed by Phase 3, both halves sit inside the same two
months, and what is being fitted is a set of blend weights and a one-dimensional monotone
function. `temporal` remains available so the comparison stays measurable rather than asserted.

**Rejected.** A three-way split (weights / calibrator / threshold). It leaves 210, 331 and 788
positives respectively — isotonic regression is non-parametric and gets unstable on a few
hundred positives. The threshold is instead swept on the calibrator's own half and the resulting
in-sample optimism is stated in the report.

## D29
### `purchase_month` cannot generalise forward — recommend dropping it
**Finding, and the one thing Phase 5 leaves open.** SHAP ranked `purchase_month` the **#1
feature** at mean |SHAP| 0.198, 33% above the second. It should not be there, and it cannot be
doing what it appears to be doing.

`purchase_month` is the calendar month as an integer 1-12. Under the §4.4 forward split:

- **fit window values: {5, 6, 7, 8, 9, 10, 11, 12}**
- **calibration window values: {1, 2}**
- **overlap: none**

The top feature takes values at scoring time that it never saw in training. A tree cannot have
learned anything about month 1; it routes an unseen value to whichever branch the last split
left open, so any apparent benefit is an accident of geometry, not a learned seasonal effect.

**A leave-one-out refit confirms it** — each row is a genuine refit on the fit window, scored on
the whole calibration window:

| Model | PR-AUC with | without | delta | Brier with | without |
|---|---:|---:|---:|---:|---:|
| CatBoost | 0.21498 | 0.20927 | **−0.00571** | 0.08528 | **0.08374** |
| LightGBM | 0.18755 | 0.20251 | **+0.01496** | 0.08910 | **0.08637** |
| XGBoost | 0.18443 | 0.21498 | **+0.03055** | 0.08916 | **0.08528** |

Dropping it **improves Brier for all three models** and improves PR-AUC for two of three —
XGBoost by +0.031, which is twice the entire GBDT-over-logistic margin Phase 4 measured. Only
CatBoost prefers keeping it, and CatBoost is precisely the model whose handling of the unseen
values happens to land favourably. The sign of the effect is not stable across libraries, which
is what an accident looks like.

**Recommendation: drop it, leaving 38 features.** §5 sanctions exactly this — "run a correlation
and permutation-importance audit after the first model and drop 1-3 dead or redundant features.
Report the surviving number honestly on the CV — if it lands at 37, write 37."

**Why it is not already done.** The change is larger than a Phase 5 step: `FEATURE_NAMES` goes
39 -> 38, which changes the artifact's input schema hash, invalidates Phase 4's tuned
hyperparameters and its committed report, and needs `make tune` and `make decide` re-run
(~12 minutes). It also plausibly changes which model leads — XGBoost without the feature scores
0.21498 on the calibration window against CatBoost's 0.20927. That is a decision worth making
deliberately rather than inside a phase whose subject is calibration.

**The alternative worth considering.** Cyclical encoding — `sin(2πm/12)`, `cos(2πm/12)` — would
give month 1 a representation adjacent to month 12, which *is* in training, so the extrapolation
problem disappears rather than being removed along with the seasonality. It is two features
instead of one and is not in §5, so it needs a decision too.

**Either way, this should be settled before Phase 6 registers a model.** Registering one whose
top feature is structurally unable to generalise forward would put a known defect behind an
`@champion` alias.

### Resolved: dropped, leaving 38 features

Cyclical encoding was rejected. `sin`/`cos` would place month 1 adjacent to month 12 and soften
the *extrapolation*, but it cannot create a second November. Months 9-12 occur in exactly one
year each, so a month effect stays perfectly confounded with that year's operations whatever the
encoding. It treats the readout, not the cause — and it adds a feature §5 does not list.

**The line now drawn, and it is not ad hoc:** a cyclical time feature is kept when its cycle
repeats often enough inside the window to be separated from the trend.

| Feature | Cycles observed in the data | Kept? |
|---|---:|---|
| `purchase_hour` | ~600 days | yes |
| `purchase_dayofweek`, `is_weekend` | **87 weeks** | yes |
| `purchase_month` | **1.67 years** | **no** |

**What actually happened, measured.** Cross-validated PR-AUC on the fit window improved for
**every** GBDT:

| Model | 39 features | 38 features | change |
|---|---:|---:|---:|
| LightGBM | 0.15278 | **0.15331** | +0.00053 |
| CatBoost | 0.14893 | **0.14997** | +0.00104 |
| XGBoost | 0.14675 | **0.14694** | +0.00019 |
| _logistic regression_ | 0.13567 | 0.13545 | −0.00022 |

**One number went down, and it should have.** Best single-model PR-AUC on the calibration window
fell from 0.22455 (CatBoost, 39 features) to 0.21520 (XGBoost, 38). That is the point rather than
a regression: CatBoost's old score was inflated by routing out-of-range months to a branch that
happened to help. Remove the accident and the inflated number goes with it, while the honest
metric — cross-validation, where every fold sees months in range — improves for all three.

**The replacement top feature is structurally sound**, which is the cleanest confirmation the
pathology was specific. `customer_zip_prefix_2` now ranks first at mean |SHAP| 0.319, and its
overlap is **98 of 98** calibration-time values seen in training, against `purchase_month`'s
**0 of 2**. Its own leave-one-out audit is genuinely mixed (+0.009 CatBoost, +0.003 LightGBM,
−0.007 XGBoost), so there is no case to drop it — and the model that ships is the one that wants
it. `promised_days` also rose from 6th to 5th.

The feature audit is a reproducible function that audits **whatever SHAP ranks first**, so this
check survives the change rather than being a one-off about one feature.

## D30
### Ship the single model: the blend collapsed onto one model
**Decision.** Ship **CatBoost** alone. The ensemble is kept as a logged experiment, exactly as
§6.5 anticipates.

**Measured** on the weight-fitting half of v1's calibration window:

| | PR-AUC | Blend weight |
|---|---:|---:|
| **CatBoost** | **0.22455** | **1.0000** |
| LightGBM | 0.21107 | 0.0000 |
| XGBoost | 0.20321 | 0.0000 |
| blend | 0.22455 | — |
| **delta vs best single** | **+0.00000** | |

The log-loss optimum is a **corner solution**: the optimiser put the entire weight on CatBoost,
so the blend *is* CatBoost and the delta is exactly zero. Against the 0.005 bar from
`configs/base.yaml`, the single model ships. §13 could not be more direct — "if the gain is under
0.005 PR-AUC, say plainly that the single model should ship" — and here the gain is not small,
it is nil.

**Worth noting: the winner changed between phases.** Phase 4 ranked LightGBM first on
cross-validated PR-AUC over the *fit* window (0.1528 against CatBoost's 0.1489). On the
*calibration* window CatBoost leads (0.22455 against 0.21107). Model selection is unstable at
this margin, which is consistent with D26's fold-to-fold variance, and is a reason to treat the
promotion gate in Phase 9 as the real arbiter rather than either of these numbers.

**Calibration worked**, on held-out rows rather than in-sample: Brier **0.085847 -> 0.083776**
on the half the calibrator never saw. The in-sample figure (0.084706 -> 0.082229) is reported
beside it only to make the difference visible.

**Threshold, under a stated assumption.** Under an assumed 5:1 FN:FP cost ratio the optimal
threshold is **0.17**, catching **48.1%** of late orders at a **21.6%** flag rate with 21.5%
precision. The cost ratio is a premise, not a measurement (§4.7). A second caveat: the sweep runs
on a window whose base rate is roughly twice the evaluation window's, so this threshold is lower
than one chosen on deployment-era data would be.

**Business impact (§4.7).** On-time orders average **4.291** out of 5; late orders **2.272** — a
gap of **2.019 review-score points**. 100% of orders in the population carry a review. Reviews
are used for this analysis only; `review_impact()` asserts no review column has reached the
feature matrix before returning a number.

## D31
### After dropping `purchase_month`: ship XGBoost
**Decision.** Ship **XGBoost** alone. This supersedes D30's choice of model; D30's *reasoning* —
ship the single model, keep the ensemble as a logged experiment — stands unchanged.

**Measured** on the weight-fitting half of v1's calibration window, 38 features:

| | PR-AUC | Blend weight |
|---|---:|---:|
| **XGBoost** | **0.21520** | 0.2328 |
| CatBoost | 0.20850 | 0.7672 |
| LightGBM | 0.20219 | 0.0000 |
| blend | 0.21364 | — |
| **delta vs best single** | **−0.00156** | |

**The blend is now a genuine two-model blend** — weights 0.77 CatBoost / 0.23 XGBoost, not D30's
corner solution — **and it is still worse than the best single model on PR-AUC.** That is the
caveat this module documents from the start: weights are fitted by minimising log-loss because it
is smooth under the simplex constraint, and a log-loss-optimal blend can rank marginally worse.
CatBoost carries most of the weight because its probabilities are better *scaled*; XGBoost wins
because it *ranks* better. PR-AUC measures ranking.

So the ensemble loses on both counts, and the single model ships with a negative delta rather than
a small positive one.

**Calibration**, on held-out rows: Brier **0.085314 -> 0.084070**. Slightly less improvement than
the 39-feature run (−0.00124 against −0.00207), which is consistent — there is less
miscalibration left to fix once the out-of-range feature is gone.

**Threshold moved, and in the direction that makes sense.** Under the same assumed 5:1 FN:FP cost
ratio: **0.21**, catching **36.3%** of late orders at a **13.7%** flag rate with **25.6%**
precision — against 0.17 / 48.1% / 21.6% / 21.5% before. The model is better calibrated and
better ranked, so the cost-minimising point tolerates flagging fewer orders. Precision up 4
points, flag rate down 8. Whether that trade is right depends on the cost ratio, which remains an
assumption (§4.7).

**Three model rankings, three different winners.** LightGBM on fit-window CV (0.15331), XGBoost on
the calibration window (0.21520), CatBoost by blend weight (0.77). At margins this small the
ranking is not stable, which is the argument for letting Phase 9's promotion gate decide on the
evaluation set rather than trusting any single number here.

**Carry into Phase 6:** the shipped model is XGBoost, so `requirements-api.txt` does **not** need
`catboost` after all — but it does need `xgboost`, which is what pulls the 305 MB CUDA library
D14 flagged. `xgboost-cpu` is now directly relevant to the Phase 8 image size.

## D32
### The pyfunc bundles everything, and the holdout tag waits for the gate
**Decision.** `DelayPredictor` bundles the fitted `PreprocessingArtifact`, the model, the
isotonic calibrator and the decision threshold into one logged object. Nothing is loaded
separately at serving time. `@champion`/`@challenger` aliases, never stages. The first
registered version becomes `@champion` uncontested; every later version becomes `@challenger`
and waits for the Phase 9 gate.

**Why bundling, rather than the API loading the artifact itself.** Because the alternative has
a failure mode that no test catches and no error reports. If the model came from the registry
and the artifact from somewhere else, serving correctness would depend on a pairing nothing
enforces: retrain, register v2, forget to ship the new artifact, and v2 runs against v1's
imputation medians, v1's frozen category levels and v1's as-of snapshots. Every prediction
still returns a plausible probability. Nothing raises. The only symptom is that the model is
quietly worse than its own metrics said — and those metrics were computed with the *correct*
artifact, so they cannot reveal it.

Bundling makes the pairing structural instead of procedural. A model version *is* its
preprocessing: one URI, one atomic thing to promote, one thing to roll back. It also makes the
§18 A6 parity claim expressible at all — "same raw record + same artifact -> same probability"
is only testable if "the artifact" is a single identifiable object rather than a convention
about which files were deployed together. The corollary is CLAUDE.md invariant 4: no
preprocessing logic outside the wrapper.

The threshold is bundled for the same reason. Shipping probabilities and leaving the operating
point in a config file elsewhere is the same class of mistake in a smaller coat.

**The §7 instruction that cannot be followed, and why that is right.** §7 asks each registered
version to be tagged with its **holdout PR-AUC**. The 2018-05 → 08 window is locked at runtime
and readable only by the promotion gate (§18 A3, D25), so a training run has no legitimate way
to compute it. The tag is written as `pending: set by the promotion gate (§18 A3)` and Phase 9
fills it in. Registering a model is not the same event as evaluating it, and the lock makes
that ordering explicit rather than optional. A test asserts the tag stays pending.

**Why the first version is champion but later ones are not.** §8 auto-promotes v1 because there
is nothing to beat. Any later version becomes `@challenger`: promotion is the gate's decision,
not a training run's, and encoding that here is what stops a retrain from quietly replacing
production. Two tests cover both branches, including that the champion is *not* displaced.

**Optuna trials are not MLflow runs.** §13 names that as a common mistake and it is: 100 trials
across three studies would bury the five runs that mean something. Tuning is a separate step
whose output is a JSON file; `train.py` loads it and logs the provenance — trials completed
against requested, and seconds spent — as params on each child run.

**Three MLflow behaviours worth knowing, each of which caused a real bug here.**

1. **`search_model_versions` returns an empty `aliases` field.** Only `get_model_version` and
   the registered-model object populate it. Reading aliases from the search result reported
   "no alias" for a model that was in fact the champion. Aliases now come from
   `get_registered_model(...).aliases`, inverted — one call rather than N. A test asserts the
   trap still exists, so the workaround can be removed when MLflow fixes it.
2. **`ModelVersion.version` is an int from a SQLite store and a str over HTTP.** A caller
   comparing it behaves differently depending on the backend, so `_assign_alias` normalises its
   return to `str`.
3. **`mlflow.set_tracking_uri` writes into the process environment**, not just an in-memory
   global. A test pointing MLflow at a throwaway SQLite store therefore redirected every later
   test — which is how the fresh-process test came to skip itself with "tracking server not
   reachable". An autouse fixture now snapshots and restores both the env var and MLflow's
   global around every test in the module.

**Rejected.** Logging the model with `mlflow.xgboost.log_model` and the artifact as a separate
file (the skew bug above). Stages instead of aliases — removed in MLflow 3, so not available
even if wanted (§18 A7). Re-tuning inside `train.py` (ten minutes per run, for a dictionary
that already exists).

---

## D33
### The bundled serving snapshot is a per-version contract, not "the newest month"
**Date.** 2026-09-29 (found in Phase 7, fixing Phase 3)

**Decision.** `PreprocessingArtifact.fit` takes a required `aggregates_through` and bundles the
newest snapshot month `M` satisfying `M <= aggregates_through + 1 day`, raising if none exists.
`build_matrix` passes `splits.aggregates_through`. The cutoff is recorded in
`artifact.metadata` so an artifact on disk can be checked against its version's contract
without `configs/splits.yaml`.

**The bug.** `fit` took `max(snapshot_month)` over whatever snapshots it was handed, and
`build_matrix` handed it every snapshot in the database — 2017-02 through 2018-08. So the v1
artifact, whose `train_cutoff` is 2017-12-31, bundled the **2018-08** snapshot. Every served
prediction used aggregates from eight months after the model was supposedly built. A model
built as of end-February 2018 cannot have seen August 2018; the serving path was an
anachronism.

**Why it mattered, beyond tidiness.** The 2018-08 snapshot aggregates outcomes delivered before
2018-08-01, which includes May, June and July 2018 deliveries — the outcomes of orders inside
the promotion evaluation window (2018-05 → 08). Scoring the champion through the serving path on
that window would have read the gate's own answer key. Worse, v1 and v2 would both have bundled
the *same* 2018-08 snapshot, so the Phase 9 contest would not even have been between two
distinct models' histories.

**`aggregates_through` was a dangling invariant.** It was declared per version in
`configs/splits.yaml`, and `assert_windows_ordered` even asserted it equals `calibrate.end` —
and then nothing read it. Declared, validated, unused. The lesson is that validating a config
value is not the same as enforcing it: the assertion proved the number was *consistent*, never
that anything *obeyed* it.

**The off-by-one is load-bearing.** `build_snapshots` filters on `delivered < M`, so snapshot
month `M` contains only outcomes resolved strictly before `M`. `M` is therefore permissible iff
every outcome in it resolved on or before `aggregates_through`, i.e. iff
`M <= aggregates_through + 1 day`. v1 (`aggregates_through` 2018-02-28) bundles **2018-03-01**;
v2 (2018-04-30) bundles **2018-05-01**. Both contain only outcomes predating the evaluation
window, because eval-window orders are *purchased* from 2018-05-01 and delivered later still.
Being one month tighter would have discarded a month of legitimate history; one looser is the
leak.

**Measured.** Same 3,000 orders, bundled 2018-08 versus the correct 2018-03: **100% of rows**
change, `route_order_count_hist` by a mean of **4,224.76** orders, `seller_order_count_hist` by
**71.76**, `route_avg_delivery_days_hist` by **0.67** days. Not cosmetic.

**What was *not* affected.** The training matrix. `transform_with_snapshots` joins each order to
its own purchase-month snapshot, so fit-window rows only ever saw snapshots ≤ 2017-12 and §18 A1
held throughout. Every Phase 4 and Phase 5 number — tuned params, blend weights, the isotonic
calibrator, threshold 0.20690, cal PR-AUC 0.20250 — is computed from that matrix and is
unchanged. A test asserts this directly: two artifacts differing only in `aggregates_through`
produce byte-identical `transform_with_snapshots` output. The bug lived entirely in the serving
path, which is why six phases of green tests never touched it.

**A test encoded the bug.** `test_the_bundled_snapshot_is_the_latest_month` asserted
`latest_snapshot_month == 2018-08-01` and passed for two phases. It pinned the *mechanism* ("the
newest month is bundled") rather than the *contract* ("the newest month this version may see").
A test written from the implementation cannot catch the implementation being wrong. It is now
`test_the_bundled_snapshot_is_the_newest_the_version_is_allowed`, derived from `splits.yaml`,
plus a config-level guard in `test_splits.py` asserting no version's bundleable month can reach
the evaluation window — so the bug cannot return through either the code or the config.

**Registry consequence: v1 was withdrawn, not deleted.** The registered v1 carried the
anachronistic artifact, so it could not remain `@champion`: the Phase 9 gate would have scored a
contaminated champion against a correct challenger and the champion would have looked unfairly
good. The corrected model was registered as **v2**, `@champion` moved to it, and v1 tagged
`status=withdrawn: do not deploy` with the reason. `print_registry` now prints that status,
because a withdrawn version keeps its metrics and would otherwise read as a plausible candidate.

Deleting v1 outright was the first intent and was **rejected** — partly because destroying
registry history is the opposite of what a registry is for, and partly because it is irreversible
for a cosmetic gain (version numbers restarting at 1). Superseding is also the more honest
artefact: "a bug was found in the champion's serving path, a corrected version was registered,
the alias moved, and the bad version was tagged so nothing deploys it" is what actually happens
in production, and the registry now shows it. Phase 9's gate therefore compares **v2 against v3**
rather than v1 against v2, which needs one sentence of explanation and loses nothing.

**Moving the alias by hand here is a deliberate exception, not a precedent.** `_assign_alias`
refused to promote v2 — correctly, because promotion is the gate's decision, and it logged
exactly that. Withdrawing a version built by broken code is a *correction*, not a promotion on
merit, so it was done explicitly and outside `src/registry/promote.py`. That module stays the only
thing that promotes on evidence.

---

## D34
### `purchase_month` is computed, not requested — the serving contract is 24 columns
**Date.** 2026-09-29 (Phase 7)

**Decision.** `purchase_month` moved out of `RAW_INPUT_COLUMNS` into a new `COMPUTED_COLUMNS`
set, alongside the 15 history columns. A caller supplying it is **refused**. The request contract
is **24 columns**, not 25.

**Why.** `PreprocessingArtifact.transform` forces `purchase_month` to the bundled snapshot's
month (§18 A6), so a supplied value is discarded before it can reach a feature. Requiring a field
and then ignoring it is worse than not requiring it: it invites a client to believe the API scores
historical orders by sending an old month, and it does not. §18 A6 reserves historical scoring for
an explicit as-of argument that v1 does not implement, so refusing the column now keeps that door
usable later instead of silently pre-empting it.

**Found by writing the API contract, which is the point of doing that first.** The column had been
in the contract since Phase 6 purely because `RAW_INPUT_COLUMNS` is *derived* — `REQUIRED_INPUT_COLUMNS`
plus the entity keys the snapshot join needs. Deriving the list stopped it drifting from the
feature set, which was the right call, but it also meant nobody ever read the list and asked
whether a client could sensibly supply every item on it.

**Rejected.** Accepting and ignoring it (a lie in the signature). Having the API derive it from
`order_purchase_timestamp` — that is preprocessing, and CLAUDE.md invariant 4 puts all of it inside
the pyfunc.

---

## D35
### Thread counts and seeds are pinned, because model selection was measuring the CPU
**Date.** 2026-09-29 (Phase 7)

**Decision.** Every model pins its thread count and its random seed in `configs/models.yaml`:
`num_threads: 4` / `seed: 42` for LightGBM (plus `deterministic: true` and `force_row_wise: true`,
which LightGBM's docs require together), `nthread: 4` / `random_state: 42` for XGBoost,
`thread_count: 4` / `random_seed: 42` for CatBoost — replacing CatBoost's explicit
`thread_count: -1`.

**The bug.** Re-running Phase 5 flipped the shipped model from XGBoost to CatBoost with **no
source change**. The investigation, in order, because each step ruled something out:

1. Stashed every Phase 7 edit and ran the **committed** code — still CatBoost. Not my changes.
2. `artifacts/tuned_params.json` matched the committed `reports/phase4_models.md` to four
   decimals. Same hyperparameters.
3. `reports/as_of_snapshots.md` regenerated byte-identical, and `reports/feature_matrix.md`
   differed by exactly one line (the bundled snapshot month of D33). Same data, same matrix.
4. `OMP_NUM_THREADS=4` and `=2` reproduced the committed numbers **exactly**: `xgboost 0.21520`,
   blend delta `-0.00156`. The default, 10 threads, gives `catboost 0.21344`.

All three libraries took every core, and their floating-point reductions are partitioned by thread
count, so the model depended on how busy the machine happened to be. Phase 5 evidently ran while
something else held cores — the summary notes a 305 MB download around then.

**What it invalidates.** D31 said "ship XGBoost" on a margin of 0.0067 (0.21520 against CatBoost's
0.20850). Thread count alone moves CatBoost by 0.005. **The margin was inside the noise, so D31
was never a finding.** Under the pinned config with re-tuned hyperparameters the three models
land at CatBoost 0.21604, LightGBM ~0.216 and XGBoost ~0.212 — a spread of about 0.004 on a
calibration window of 13,624 rows. The honest statement is that the three are **indistinguishable
on this data**, and any choice between them should be argued on grounds other than PR-AUC:
inference cost, dependency weight, or how a library handles unseen categories.

**Pinning the thread count, not deriving it from core count,** is what makes a run reproducible on
a *different* machine. A 2-core CI runner oversubscribes 4 threads and is slower, but partitioning
is by thread count, so it computes the same model. Verified: `OMP_NUM_THREADS=2` and `=10` now
produce byte-identical output, because the library parameter overrides the environment.

**Hyperparameters were re-tuned** under the pinned config, since the previous ones were selected
by an experiment whose fold scores carried this noise. Re-tuning took 6 minutes — *faster* than the
unpinned run (69s/78s/209s against 101s/124s/309s), so the reproducibility cost here is negative.

**A latent second copy of the same hazard, fixed at the same time.** `ORDERS_QUERY` had no
`ORDER BY`, and both Phase 4 and Phase 5 sorted with `np.argsort(kind="stable")` — which preserves
the *input* order among ties. There are **522 tied `order_purchase_timestamp` values** in 96,203
rows, and everything downstream slices positionally: `TimeSeriesSplit` cuts folds by index, and
`fit_final_model` takes the last 15% as its early-stopping tail. Measured that Postgres's order is
in fact stable today, so this caused **nothing** — but it would have, the first time anything
rewrote that table. Now `ORDER BY order_purchase_timestamp, order_id` gives a total order, and one
shared `build.window_slice` replaces two hand-rolled sorts, so there is one place to get it wrong
instead of two.

**The general lesson, which is the reusable part.** A result that is *stable across consecutive
runs* can still be *irreproducible*, because the thing that varies is the environment, not a seed.
Every determinism check in this project until now re-ran a command back-to-back on an idle machine,
which is exactly the condition under which this bug is invisible. Checking reproducibility means
varying the environment on purpose — thread count, core count, machine — not repeating the run.

**Consequence for Phase 8.** CatBoost ships, so `requirements-api.txt` needs `catboost` after all,
which is precisely the caveat that file already carried. Noted there, along with the operational
coupling it implies: if the Phase 9 gate promotes a challenger shipping a different library, the
API image must be rebuilt before the promotion can be served, so promotion is not purely a
registry operation.

---

## D36
### The API scores through the unwrapped predictor, and checks the signature once at startup
**Date.** 2026-09-29 (Phase 7)

**Decision.** `api/model_loader.py` loads the pyfunc, then calls
`pyfunc.unwrap_python_model().score(frame)` rather than `pyfunc.predict(frame)`. To compensate for
the schema enforcement that gives up, `assert_signature_matches_contract` compares the loaded
signature against `RAW_INPUT_COLUMNS` **once, at startup**, and a mismatch stops the app booting.

**Why.** §9 requires an unseen category to return 200 with a `warnings` field, so something must
notice which values fell through to `__unknown__`. Preprocessing may not live in the API layer
(invariant 4), and the obvious alternative — let the API call `transform` a second time to inspect
the result — would double the dominant serving cost. Measured: a **1-row** `transform` takes ~45 ms
and 1,000 rows take ~69 ms, because the cost is fixed overhead in the as-of snapshot join rather
than per-row work. So the wrapper grew a `score()` returning `(predictions, warnings)` from one
transform, and `predict()` delegates to it.

**What was given up, and why the replacement is better placed.** `pyfunc.predict` enforces the
logged input schema on every call. Enforcing it once at startup is cheaper and catches the same
class of error earlier: a model whose contract has moved should refuse to boot, not return 500s
under load. Request-level validation is not lost — Pydantic validates every field with types and
bounds, which is strictly stronger than MLflow's schema check for malformed input, and the wrapper
still rejects missing or computed columns itself. A test saves a model with a deliberately wrong
signature and asserts `/health` reports 503 with "signature" in the detail.

**Rejected.** Adding warnings to the pyfunc's *output* schema: it would change `OUTPUT_COLUMNS` and
make every already-registered version's logged signature a lie, for a field that is a fact about a
request rather than about a row.

---

## D37
### The unseen-category warning says "not among the levels this model knows", not "never seen"
**Date.** 2026-09-29 (Phase 7)

**Decision.** The warning text names the model's *known levels* rather than claiming a value was
absent from training.

**Why — measured on all 96,203 real orders.** The champion reports unseen values for **3 seller
states** (`AM`, `MA`, `PI`) and **43 product categories**. Those two numbers have different causes:

- the states are genuinely absent from the fit window — too few sellers to appear;
- the categories were mostly **present and deliberately capped**. `CATEGORY_CAP = 30` keeps the top
  30 of 72, covering 95.80% of fit-window rows, so the other 42 collapse to `__unknown__` by design.

Both routes end at the same sentinel, so the detection cannot separate them — but "was not seen
during training" is *false* for most of the 43, and shipping a warning that overstates its case
teaches an operator to ignore the channel. The cost of that would be the genuine drift signal in the
seller-state case. Being accurate about what is known is achievable without extra machinery; being
accurate about *why* would need the artifact to record which values it capped, which is not worth an
artifact schema change for a message.

**Consequence worth stating in the README:** `product_category` warns routinely and that is normal.
A monitoring rule built on "any warning is drift" would fire on every batch.
