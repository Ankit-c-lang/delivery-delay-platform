"""The prediction-time contract (PLAN.md §4.2) — deliberately dependency-free.

**Why this is its own module rather than living in ``src/etl/schema.py``.**
:data:`DENYLIST` is enforced inside :func:`src.features.build.construct_features`, which runs on
**every served prediction**. Importing it from ``src.etl.schema`` therefore dragged that module's
``pandera`` dependency into the serving image — a data-validation framework the API never invokes,
pulled in only to read a frozenset of ten strings. ``requirements-api.txt`` said pandera was
"excluded on purpose" while the pyfunc could not load without it (DECISIONS.md D38).

So the contract lives here, where it can be imported by anything: ETL, feature construction and the
serving path all read the same definition, and none of them pay for the others' dependencies.

Nothing may be added to this module that imports a third-party package.
"""

from __future__ import annotations

#: Columns that may never reach a feature matrix, because every one is populated *after* the
#: moment a prediction is made (PLAN.md §4.2).
#:
#: ``order_status`` is included even though the population is filtered to ``'delivered'`` — it is
#: then a constant, and a constant that encodes the filter is exactly the shape of a leak that
#: survives review.
#:
#: ``order_delivered_customer_date`` is genuinely needed by the §4.5 as-of snapshot rule, which
#: §18 A1 rewrote to filter on delivery *outcome* time. It lives only in
#: ``features.order_outcomes``, read by exactly one module (CLAUDE.md invariant 12).
DENYLIST: frozenset[str] = frozenset(
    {
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_status",
        "review_id",
        "review_score",
        "review_comment_title",
        "review_comment_message",
        "review_creation_date",
        "review_answer_timestamp",
    }
)
