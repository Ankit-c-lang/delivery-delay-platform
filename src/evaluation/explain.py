"""SHAP explanations (PLAN.md §6.7, §13 Phase 5 item 5).

``TreeExplainer`` on a 3,000-row sample. Full SHAP over 96,203 rows would be slow on this CPU
for no extra insight — the beeswarm saturates long before then.

**The blend's SHAP values are the weighted average of the per-model SHAP values.** SHAP is
additive and a weighted average of models is a linear combination, so by linearity of the
Shapley attribution the blend's attribution is the same weighted average. §6.7 asks for this to
be derivable in one line before it is claimed, so :func:`blended_shap_values` computes it that
way and :func:`check_additivity` verifies it numerically against the blended model output
rather than taking it on trust.

**Read the ranking as a leakage detector, not a trophy.** §6.7 warns that a feature you thought
was harmless showing up implausibly high is an alarm. DECISIONS.md D22 adds the mirror for this
project specifically: with the §18 A1 snapshot rule applied, `seller_late_rate_hist` correlates
just **+0.0197** with the target. If it dominates the model, the snapshot join has regressed —
that is an alarm, not a win. :func:`leakage_alarms` encodes both directions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

# Agg before pyplot: this runs headless in a VM and in CI, with no display to attach to.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

SHAP_SAMPLE_ROWS = 3000
PLOTS_DIR = Path("reports/shap")

#: Features §6.7 expects to dominate. Their absence is as informative as their presence.
EXPECTED_TOP = ("promised_days", "seller_late_rate_hist", "haversine_km")

#: Features whose dominance would be an alarm, with the reason. `seller_late_rate_hist` is here
#: *and* in EXPECTED_TOP on purpose: §6.7 expected it to lead, and D22 measured that it cannot
#: legitimately, so it sits on both lists and the report has to say which happened.
ALARM_IF_DOMINANT: dict[str, str] = {
    "seller_late_rate_hist": (
        "correlates only +0.0197 with the target under the §18 A1 rule (D22); dominance "
        "suggests the as-of snapshot join regressed to the leaky purchase-time filter"
    ),
    "route_late_rate_hist": (
        "correlates +0.0785 (D22); dominance well above that is worth checking against the "
        "snapshot boundary"
    ),
}


@dataclass(frozen=True)
class ExplainResult:
    """SHAP output and the plots written for it.

    Attributes:
        model_name: Which model, or ``blend``.
        mean_abs_shap: Mean absolute SHAP per feature, descending.
        plots: Paths written, keyed by kind.
        n_rows: Rows explained.
        additivity_error: Max absolute difference between summed SHAP plus the base value and
            the model's own output. Near zero is the point.
        alarms: Human-readable leakage warnings, empty when none fired.
    """

    model_name: str
    mean_abs_shap: pd.Series
    plots: dict[str, Path] = field(default_factory=dict)
    n_rows: int = 0
    additivity_error: float = float("nan")
    alarms: list[str] = field(default_factory=list)

    def top(self, n: int = 10) -> list[tuple[str, float]]:
        """The ``n`` most important features."""
        return [(str(k), float(v)) for k, v in self.mean_abs_shap.head(n).items()]

    def summary(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "n_rows": self.n_rows,
            "additivity_error": self.additivity_error,
            "top_10": [(name, round(value, 5)) for name, value in self.top(10)],
            "alarms": self.alarms,
        }


def sample_rows(X: pd.DataFrame, n: int = SHAP_SAMPLE_ROWS, seed: int = 42) -> pd.DataFrame:
    """A reproducible random sample for explanation."""
    if len(X) <= n:
        return X
    return X.sample(n=n, random_state=seed).sort_index()


def _as_positive_class(values: Any) -> NDArray[np.float64]:
    """Normalise the several shapes SHAP returns for a binary classifier.

    Depending on the library and version this is a single ``(n, features)`` array, a list of two
    such arrays, or an ``(n, features, 2)`` stack. Getting this wrong silently explains the
    negative class, which inverts every sign in the plot.
    """
    if isinstance(values, list):
        return np.asarray(values[-1], dtype=float)
    array = np.asarray(values, dtype=float)
    if array.ndim == 3:
        return array[:, :, -1]
    return array


def shap_values_for_model(model: Any, X: pd.DataFrame) -> tuple[NDArray[np.float64], float]:
    """SHAP values for the positive class, plus the explainer's base value.

    Returns:
        ``(values, base_value)`` where ``values`` has shape ``(len(X), n_features)``.
    """
    import shap

    explainer = shap.TreeExplainer(model)
    values = _as_positive_class(explainer.shap_values(X))
    expected = explainer.expected_value
    if isinstance(expected, list | np.ndarray):
        base = float(np.asarray(expected).ravel()[-1])
    else:
        base = float(expected)
    return values, base


def blended_shap_values(
    per_model: dict[str, NDArray[np.float64]], weights: dict[str, float]
) -> NDArray[np.float64]:
    """The blend's SHAP values: the same weighted average as the blend itself.

    One line, because that is all it is — SHAP attributions are additive and a weighted average
    of models is a linear combination, so the attribution of the average is the average of the
    attributions (§6.7).

    Raises:
        KeyError: If models and weights disagree.
    """
    if set(per_model) != set(weights):
        raise KeyError(f"models {sorted(per_model)} do not match weights {sorted(weights)}")
    return sum(weights[name] * per_model[name] for name in sorted(per_model))


def check_additivity(
    values: NDArray[np.float64], base_value: float, model_output: NDArray[np.float64]
) -> float:
    """Max absolute gap between ``base + sum(shap)`` and the model's own output.

    SHAP's guarantee is exact for tree models in margin space, so a large gap means the wrong
    output space is being compared — typically log-odds SHAP against probability predictions.
    Returned rather than asserted so the caller can report the number.
    """
    reconstructed = base_value + values.sum(axis=1)
    return float(np.max(np.abs(reconstructed - np.asarray(model_output, dtype=float))))


def mean_abs_shap(values: NDArray[np.float64], feature_names: list[str]) -> pd.Series:
    """Mean absolute SHAP per feature, sorted descending."""
    return pd.Series(np.abs(values).mean(axis=0), index=feature_names).sort_values(ascending=False)


def leakage_alarms(ranking: pd.Series, top_n: int = 3) -> list[str]:
    """Flag a ranking that contradicts what the features were measured to be worth.

    Args:
        ranking: Output of :func:`mean_abs_shap`.
        top_n: How many leading features count as "dominant".

    Returns:
        One message per alarm, empty when the ranking looks as expected. §6.7 asks for a
        feature that is implausibly high to be treated as an alarm rather than a win; this
        makes that reflex automatic instead of relying on someone remembering.
    """
    leaders = [str(name) for name in ranking.head(top_n).index]
    alarms = [
        f"{name} is in the top {top_n} by mean |SHAP|: {reason}"
        for name, reason in ALARM_IF_DOMINANT.items()
        if name in leaders
    ]
    missing = [name for name in EXPECTED_TOP if name not in leaders]
    if len(missing) == len(EXPECTED_TOP):
        alarms.append(
            f"none of the features §6.7 expected to dominate {EXPECTED_TOP} is in the top "
            f"{top_n}; worth confirming the feature matrix is the one that was trained on"
        )
    return alarms


def plot_beeswarm(values: NDArray[np.float64], X: pd.DataFrame, path: Path, title: str) -> Path:
    """Beeswarm summary plot."""
    import shap

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    shap.summary_plot(values, X, show=False, max_display=20)
    plt.title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close("all")
    return path


def plot_mean_abs_bar(ranking: pd.Series, path: Path, title: str, top_n: int = 20) -> Path:
    """Horizontal bar chart of mean |SHAP|."""
    path.parent.mkdir(parents=True, exist_ok=True)
    top = ranking.head(top_n).iloc[::-1]
    plt.figure(figsize=(8, max(4, 0.32 * len(top))))
    plt.barh(top.index.astype(str), top.to_numpy(), color="#4C78A8")
    plt.xlabel("mean |SHAP|")
    plt.title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close("all")
    return path


def plot_dependence(values: NDArray[np.float64], X: pd.DataFrame, feature: str, path: Path) -> Path:
    """Dependence plot for one feature.

    Categorical columns are cast to their integer codes first: SHAP's dependence plot needs a
    numeric x-axis, and passing a ``category`` dtype raises.
    """
    import shap

    path.parent.mkdir(parents=True, exist_ok=True)
    frame = X.copy()
    for column in frame.columns:
        if isinstance(frame[column].dtype, pd.CategoricalDtype):
            frame[column] = frame[column].cat.codes
    plt.figure()
    shap.dependence_plot(feature, values, frame, show=False, interaction_index=None)
    plt.tight_layout()
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close("all")
    return path


def explain_model(
    model: Any,
    X: pd.DataFrame,
    model_name: str,
    model_output: NDArray[np.float64] | None = None,
    plots_dir: Path = PLOTS_DIR,
    n_dependence: int = 3,
) -> ExplainResult:
    """SHAP a single model and write its plots.

    Args:
        model: A fitted tree model.
        X: Rows to explain, already sampled.
        model_name: Used in filenames and titles.
        model_output: Margin-space output, for the additivity check. Skipped when absent.
        plots_dir: Where to write.
        n_dependence: Dependence plots, for the top features.

    Returns:
        The ranking, the plot paths and any leakage alarms.
    """
    values, base = shap_values_for_model(model, X)
    ranking = mean_abs_shap(values, list(X.columns))

    plots = {
        "beeswarm": plot_beeswarm(
            values, X, plots_dir / f"{model_name}_beeswarm.png", f"SHAP beeswarm — {model_name}"
        ),
        "bar": plot_mean_abs_bar(
            ranking, plots_dir / f"{model_name}_mean_abs.png", f"Mean |SHAP| — {model_name}"
        ),
    }
    for rank, feature in enumerate(ranking.head(n_dependence).index, start=1):
        plots[f"dependence_{rank}_{feature}"] = plot_dependence(
            values, X, str(feature), plots_dir / f"{model_name}_dependence_{rank}_{feature}.png"
        )

    error = (
        check_additivity(values, base, model_output) if model_output is not None else float("nan")
    )
    alarms = leakage_alarms(ranking)
    for alarm in alarms:
        logger.warning("SHAP leakage alarm: %s", alarm)

    logger.info(
        "%s: top features %s",
        model_name,
        [n for n, _ in [(str(k), v) for k, v in ranking.head(5).items()]],
    )
    return ExplainResult(
        model_name=model_name,
        mean_abs_shap=ranking,
        plots=plots,
        n_rows=len(X),
        additivity_error=error,
        alarms=alarms,
    )
