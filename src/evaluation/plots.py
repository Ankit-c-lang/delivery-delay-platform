"""Evaluation plots logged with every run (PLAN.md §7).

PR curve, ROC curve, calibration curve and a confusion matrix at the chosen threshold. Each
returns the path it wrote so a caller can hand it straight to ``mlflow.log_artifact``.

The PR curve carries a base-rate reference line and the ROC curve a diagonal, because both
metrics are meaningless without the thing they should be read against — at a 6% positive rate
a PR curve that looks poor in isolation may be well above the only floor that matters.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


def _save(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close("all")
    return path


def pr_curve(y_true: ArrayLike, y_score: ArrayLike, path: Path, label: str = "model") -> Path:
    """Precision-recall curve with the base rate drawn as the floor."""
    truth = np.asarray(y_true, dtype=int)
    precision, recall, _ = precision_recall_curve(truth, y_score)
    base = float(truth.mean())
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"{label} (AP={average_precision_score(truth, y_score):.4f})")
    plt.axhline(base, linestyle="--", color="grey", label=f"base rate {base:.4f} (random ranking)")
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.title("Precision-recall")
    plt.legend(loc="upper right", fontsize=8)
    plt.ylim(0, 1)
    return _save(path)


def roc_curve_plot(y_true: ArrayLike, y_score: ArrayLike, path: Path, label: str = "model") -> Path:
    """ROC curve. Secondary to PR-AUC here and labelled as such (§4.6)."""
    truth = np.asarray(y_true, dtype=int)
    fpr, tpr, _ = roc_curve(truth, y_score)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"{label} (AUC={roc_auc_score(truth, y_score):.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey", label="random")
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.title("ROC — secondary to PR-AUC at a ~6% positive rate")
    plt.legend(loc="lower right", fontsize=8)
    return _save(path)


def calibration_plot(y_true: ArrayLike, y_score: ArrayLike, path: Path, n_bins: int = 10) -> Path:
    """Reliability diagram against the diagonal.

    ``strategy="quantile"`` so every bin holds the same number of orders: with a 6% positive
    rate, uniform bins leave the upper ones nearly empty and the curve becomes noise that
    looks like miscalibration.
    """
    truth = np.asarray(y_true, dtype=int)
    observed, predicted = calibration_curve(truth, y_score, n_bins=n_bins, strategy="quantile")
    plt.figure(figsize=(6, 5))
    plt.plot(predicted, observed, marker="o", label="model")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey", label="perfect calibration")
    upper = max(float(np.max(predicted)), float(np.max(observed))) * 1.15
    plt.xlim(0, upper)
    plt.ylim(0, upper)
    plt.xlabel("mean predicted probability")
    plt.ylabel("observed frequency")
    plt.title(f"Calibration ({n_bins} equal-count bins)")
    plt.legend(loc="upper left", fontsize=8)
    return _save(path)


def confusion_plot(y_true: ArrayLike, y_score: ArrayLike, threshold: float, path: Path) -> Path:
    """Confusion matrix at the chosen operating point, annotated with counts and rates."""
    truth = np.asarray(y_true, dtype=int)
    predicted = (np.asarray(y_score, dtype=float) >= threshold).astype(int)
    matrix = confusion_matrix(truth, predicted, labels=[0, 1])

    plt.figure(figsize=(5.2, 4.6))
    plt.imshow(matrix, cmap="Blues")
    plt.xticks([0, 1], ["predicted on-time", "predicted late"])
    plt.yticks([0, 1], ["actually on-time", "actually late"])
    total = matrix.sum()
    for i in range(2):
        for j in range(2):
            plt.text(
                j,
                i,
                f"{matrix[i, j]:,}\n{matrix[i, j] / total:.1%}",
                ha="center",
                va="center",
                color="white" if matrix[i, j] > matrix.max() / 2 else "black",
                fontsize=10,
            )
    plt.title(f"Confusion matrix at threshold {threshold:.3f}")
    plt.colorbar(shrink=0.8)
    return _save(path)


def all_plots(
    y_true: ArrayLike,
    y_score: ArrayLike,
    threshold: float,
    directory: Path,
    label: str = "model",
) -> dict[str, Path]:
    """Write all four plots into ``directory``, keyed by kind."""
    return {
        "pr_curve": pr_curve(y_true, y_score, directory / "pr_curve.png", label),
        "roc_curve": roc_curve_plot(y_true, y_score, directory / "roc_curve.png", label),
        "calibration": calibration_plot(y_true, y_score, directory / "calibration.png"),
        "confusion": confusion_plot(y_true, y_score, threshold, directory / "confusion_matrix.png"),
    }
