"""Score reference models on the promotion evaluation window (PLAN.md §13 Phase 11).

The README's results table makes an uncomfortable claim — that a logistic regression ranks better
than the tuned GBDT on held-out data — so the claim is reproducible rather than quoted. Run:

    make holdout-reference

**These are references, not candidates.** Nothing here is registered, promoted, or eligible to be.
That matters for §18 A3's argument about selection bias: bias comes from *choosing* on a window, and
a model that cannot be chosen cannot bias the choice. The gate still reads this window exactly
twice, once per contested run.

The window is loaded through :mod:`src.evaluation.score_holdout`, which is the only module permitted
to touch it — this script asks that module rather than reaching past it.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.evaluation.metrics import brier, pr_auc, roc_auc

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Reference:
    """One model's holdout scores.

    Attributes:
        name: How the README names it.
        pr_auc: Average precision.
        lift: ``pr_auc / base_rate`` — the figure that is comparable across windows.
        roc_auc: Secondary, and flattering at a 4.4% positive rate.
        brier: Calibration. Raw probabilities are penalised here; the champion's are isotonic.
    """

    name: str
    pr_auc: float
    lift: float
    roc_auc: float
    brier: float


def _score(name: str, labels: np.ndarray, probabilities: np.ndarray, base_rate: float) -> Reference:
    score = pr_auc(labels, probabilities)
    return Reference(
        name=name,
        pr_auc=score,
        lift=score / base_rate,
        roc_auc=roc_auc(labels, probabilities),
        brier=brier(labels, probabilities),
    )


def build_table(version: str = "v1") -> tuple[list[Reference], float, int]:
    """Score the no-skill baseline, a logistic regression and the champion on the same rows.

    Args:
        version: Whose fit window the reference models train on, and whose artifact builds the
            holdout features. ``v1`` matches the champion.

    Returns:
        ``(references, base_rate, n_rows)``.

    Every model sees **identical features**: the holdout records are transformed by the champion's
    own artifact through the serving path, so a difference in scores is a difference in models
    rather than in preprocessing.
    """
    import mlflow

    from src.registry import tracking_uri

    mlflow.set_tracking_uri(tracking_uri())

    from src.evaluation.score_holdout import load_holdout
    from src.features.build import build_matrix, window_slice
    from src.splits import version_splits
    from src.training.baselines import MajorityClassBaseline, build_logistic_baseline

    records, labels = load_holdout()
    base_rate = float(labels.mean())

    built = build_matrix(version)
    splits = version_splits(version)
    X_fit, y_fit, _ = window_slice(splits.fit, built["orders"], built["matrix"], built["labels"])
    X_holdout = built["artifact"].transform(records)

    references = []

    majority = MajorityClassBaseline().fit(X_fit, y_fit)
    references.append(
        _score(
            "majority class (no skill)",
            labels,
            majority.predict_proba_positive(X_holdout),
            base_rate,
        )
    )

    logistic = build_logistic_baseline(X_fit)
    logistic.fit(X_fit, y_fit)
    references.append(
        _score(
            "logistic regression",
            labels,
            logistic.predict_proba(X_holdout)[:, 1],
            base_rate,
        )
    )

    champion = mlflow.pyfunc.load_model("models:/delivery_delay_classifier@champion")
    references.append(
        _score(
            "champion (registry @champion)",
            labels,
            champion.predict(records)["probability"].to_numpy(),
            base_rate,
        )
    )

    return references, base_rate, len(labels)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Reference models on the evaluation window")
    parser.add_argument("--version", default="v1")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING)
    references, base_rate, n_rows = build_table(args.version)

    print()
    print(f"Promotion evaluation window: {n_rows:,} rows, base rate {base_rate:.2%}")
    print()
    print(f"{'model':32} {'PR-AUC':>8} {'lift':>7} {'ROC-AUC':>8} {'Brier':>9}")
    print("-" * 68)
    for reference in references:
        print(
            f"{reference.name:32} {reference.pr_auc:8.5f} {reference.lift:6.2f}x "
            f"{reference.roc_auc:8.5f} {reference.brier:9.6f}"
        )
    print()

    frame = pd.DataFrame([vars(reference) for reference in references])
    best = frame.loc[frame["pr_auc"].idxmax(), "name"]
    champion_row = frame.loc[frame["name"].str.startswith("champion")].iloc[0]
    if best != champion_row["name"]:
        gap = frame["pr_auc"].max() - champion_row["pr_auc"]
        print(
            f"NOTE: {best!r} ranks ABOVE the champion by {gap:.5f} PR-AUC on this window.\n"
            "      That is 6x the 0.005 reproducibility floor (DECISIONS.md D35), so it is a real\n"
            "      difference rather than noise. See DECISIONS.md D45."
        )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
