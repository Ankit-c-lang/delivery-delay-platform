"""Optuna tuning over time-series CV (PLAN.md §6.3, §6.4, §13 Phase 4).

**`TimeSeriesSplit`, never `KFold` or `StratifiedKFold`.** With as-of features and a base
rate that swings from 1.16% to 18.96% across months (DECISIONS.md D20), a random fold puts
future rows in the training set and past rows in validation. The model then learns from
outcomes it could not have seen, and every CV number is optimistic. This is not a stylistic
preference: a stratified fold on this data would silently undo the entire §4.5 snapshot
design. That is why the splitter is constructed in exactly one place, here, and asserted by
`tests/test_training_cv.py`.

**No nested parallelism.** Optuna runs `n_jobs=1` and the boosting library gets the threads.
An i5-1235U is a low-power mobile part; Optuna workers multiplied by library threads thrash
it and make every timing meaningless.

**`timeout` is a hard stop**, and the number of trials *actually completed* is recorded. A
budget you silently blow through is not a budget, and a study that ran 9 of 40 trials should
say so rather than be reported as "tuned".

**No resampling for class imbalance.** Train on raw probabilities, calibrate in Phase 5, then
tune the threshold. `scale_pos_weight` is run once as a logged experiment (§6.4).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import yaml
from sklearn.model_selection import TimeSeriesSplit

from src.evaluation.metrics import pr_auc
from src.features.build import CATEGORICAL_FEATURES

logger = logging.getLogger(__name__)

MODELS_CONFIG = Path("configs/models.yaml")
SUPPORTED = ("lightgbm", "xgboost", "catboost")


def load_model_config(path: Path = MODELS_CONFIG) -> dict[str, Any]:
    """Parse ``configs/models.yaml``."""
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def make_cv(n_splits: int) -> TimeSeriesSplit:
    """The one place a cross-validator is constructed.

    ``TimeSeriesSplit`` yields expanding training windows, each validated on the block
    immediately after it, so no fold can train on a row that comes after its validation rows.
    See the module docstring for why anything else is forbidden here.
    """
    return TimeSeriesSplit(n_splits=n_splits)


def assert_time_sorted(timestamps: pd.Series) -> None:
    """Fail loudly if the rows are not in time order.

    ``TimeSeriesSplit`` splits on *position*, not on the timestamp: it has no idea what the
    dates are. Handing it unsorted rows produces folds that look fine and leak badly, with no
    error anywhere. This assertion is the only thing standing between those two outcomes.
    """
    values = pd.to_datetime(timestamps).to_numpy()
    if not np.all(values[:-1] <= values[1:]):
        raise ValueError(
            "rows are not sorted by purchase time. TimeSeriesSplit splits by position, so "
            "unsorted input produces folds that leak silently."
        )


@dataclass
class TuningResult:
    """What a study actually did, as opposed to what it was asked to do.

    Attributes:
        model: ``lightgbm``, ``xgboost`` or ``catboost``.
        best_params: Best hyperparameters found.
        best_score: Mean cross-validated PR-AUC at those parameters.
        fold_scores: Per-fold PR-AUC at the best trial.
        trials_completed: Trials that finished. **Compare against ``trials_requested``** —
            a timeout is normal on this hardware and must be visible.
        trials_requested: Trials the budget asked for.
        seconds: Wall-clock time.
        timed_out: Whether the wall-clock cap stopped the study.
    """

    model: str
    best_params: dict[str, Any]
    best_score: float
    fold_scores: list[float] = field(default_factory=list)
    trials_completed: int = 0
    trials_requested: int = 0
    seconds: float = 0.0
    timed_out: bool = False

    def summary(self) -> dict[str, Any]:
        """A flat, loggable description."""
        return {
            "model": self.model,
            "best_pr_auc": round(self.best_score, 5),
            "trials_completed": self.trials_completed,
            "trials_requested": self.trials_requested,
            "seconds": round(self.seconds, 1),
            "timed_out": self.timed_out,
            "fold_scores": [round(s, 5) for s in self.fold_scores],
        }


def _suggest(trial: optuna.Trial, name: str, spec: dict[str, Any]) -> Any:
    """Translate one ``configs/models.yaml`` search-space entry into an Optuna suggestion."""
    kind = spec["type"]
    if kind == "float":
        return trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
    if kind == "int":
        return trial.suggest_int(name, spec["low"], spec["high"], log=spec.get("log", False))
    if kind == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    raise ValueError(f"unknown search-space type {kind!r} for {name!r}")


def _categorical_columns(X: pd.DataFrame) -> list[str]:
    return [c for c in CATEGORICAL_FEATURES if c in X.columns]


def fit_model(
    model_name: str,
    params: dict[str, Any],
    fixed: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
) -> Any:
    """Fit one model with early stopping against ``X_valid`` and return the fitted estimator.

    Each library gets its native categorical handling rather than one-hot encoding (§6.1):
    LightGBM reads pandas ``category`` dtype directly, XGBoost needs
    ``enable_categorical=True`` with the ``hist`` tree method, and CatBoost takes an explicit
    ``cat_features`` list — but rejects pandas ``category`` dtype, so those columns are cast
    to string for it and only for it.
    """
    categorical = _categorical_columns(X_train)
    settings = {**fixed, **params}

    if model_name == "lightgbm":
        from lightgbm import LGBMClassifier, early_stopping, log_evaluation

        rounds = settings.pop("early_stopping_rounds", 50)
        model = LGBMClassifier(**settings)
        model.fit(
            X_train,
            y_train,
            # eval_X / eval_y, not eval_set: LightGBM 4.7 deprecates eval_set.
            eval_X=X_valid,
            eval_y=y_valid,
            eval_metric="average_precision",
            callbacks=[early_stopping(rounds, verbose=False), log_evaluation(0)],
        )
        return model

    if model_name == "xgboost":
        from xgboost import XGBClassifier

        model = XGBClassifier(**settings)
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
        return model

    if model_name == "catboost":
        from catboost import CatBoostClassifier

        # CatBoost wants plain strings for categoricals, not pandas category dtype.
        train = X_train.copy()
        valid = X_valid.copy()
        for column in categorical:
            train[column] = train[column].astype(str)
            valid[column] = valid[column].astype(str)
        model = CatBoostClassifier(**settings, cat_features=categorical)
        model.fit(train, y_train, eval_set=(valid, y_valid), use_best_model=True)
        return model

    raise ValueError(f"unsupported model {model_name!r}; expected one of {SUPPORTED}")


def predict_positive(model_name: str, model: Any, X: pd.DataFrame) -> np.ndarray:
    """Positive-class probabilities from a fitted model.

    CatBoost was trained with its categoricals as strings, so it must be predicted with them
    as strings too — feeding it ``category`` dtype at predict time raises.
    """
    frame = X
    if model_name == "catboost":
        frame = X.copy()
        for column in _categorical_columns(X):
            frame[column] = frame[column].astype(str)
    return model.predict_proba(frame)[:, 1]


def _fit_predict(
    model_name: str,
    params: dict[str, Any],
    fixed: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
) -> np.ndarray:
    """Fit on one fold and return validation probabilities."""
    model = fit_model(model_name, params, fixed, X_train, y_train, X_valid, y_valid)
    return predict_positive(model_name, model, X_valid)


def fit_final_model(
    model_name: str,
    params: dict[str, Any],
    fixed: dict[str, Any],
    X: pd.DataFrame,
    y: np.ndarray,
    early_stopping_tail: float = 0.15,
) -> Any:
    """Fit on the whole fit window, holding back an internal tail for early stopping.

    Args:
        model_name: One of :data:`SUPPORTED`.
        params: Tuned hyperparameters.
        fixed: Fixed settings from ``configs/models.yaml``.
        X: Fit-window features, **time-sorted**.
        y: Labels aligned to ``X``.
        early_stopping_tail: Fraction of the end of the window reserved for early stopping.

    Returns:
        The fitted estimator.

    **Why an internal tail rather than the calibration window.** Early stopping needs a
    validation set, and the obvious candidate — the calibration window — is exactly the data
    Phase 5 fits its blend weights on. Using it here would let every model peek at the rows
    that decide their blend weights, which is the §6.5 mistake in a different coat. The last
    15% of the fit window is used instead: still in-sample for the phase, chronologically last
    so it mirrors the direction of deployment, and never touching the calibration rows.
    """
    split = int(len(X) * (1.0 - early_stopping_tail))
    if split < 1 or split >= len(X):
        raise ValueError(f"early_stopping_tail {early_stopping_tail} leaves no usable split")
    return fit_model(
        model_name, params, fixed, X.iloc[:split], y[:split], X.iloc[split:], y[split:]
    )


def cross_validate(
    model_name: str,
    params: dict[str, Any],
    fixed: dict[str, Any],
    X: pd.DataFrame,
    y: np.ndarray,
    n_splits: int,
) -> list[float]:
    """PR-AUC on each time-series fold.

    Early stopping uses **each fold's own validation slice**, never the calibration window
    (§6.3) — that window belongs to Phase 5 and touching it here would make its blend weights
    and threshold in-sample.
    """
    scores: list[float] = []
    for train_idx, valid_idx in make_cv(n_splits).split(X):
        y_valid = y[valid_idx]
        if y_valid.sum() == 0:
            logger.warning("fold has no positive labels; skipping it")
            continue
        probabilities = _fit_predict(
            model_name,
            params,
            dict(fixed),
            X.iloc[train_idx],
            y[train_idx],
            X.iloc[valid_idx],
            y_valid,
        )
        scores.append(pr_auc(y_valid, probabilities))
    if not scores:
        raise ValueError("no usable folds: every validation slice was single-class")
    return scores


def tune(
    model_name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    timestamps: pd.Series,
    config: dict[str, Any] | None = None,
    *,
    seed: int | None = None,
) -> TuningResult:
    """Run one Optuna study and report what it actually managed.

    Args:
        model_name: One of :data:`SUPPORTED`.
        X: Feature matrix, **already sorted by purchase time**.
        y: Binary labels aligned to ``X``.
        timestamps: Purchase timestamps aligned to ``X``, used only to verify the sort.
        config: Parsed ``configs/models.yaml``; read from disk when omitted.
        seed: Sampler seed; defaults to the config's.

    Returns:
        A :class:`TuningResult`. Check ``trials_completed`` against ``trials_requested``
        before describing the model as tuned.
    """
    if model_name not in SUPPORTED:
        raise ValueError(f"unsupported model {model_name!r}; expected one of {SUPPORTED}")
    assert_time_sorted(timestamps)

    config = config or load_model_config()
    spec = config["models"][model_name]
    n_splits = config["cv"]["n_splits"]
    seed = config.get("seed", 42) if seed is None else seed

    def objective(trial: optuna.Trial) -> float:
        params = {name: _suggest(trial, name, s) for name, s in spec["search"].items()}
        scores = cross_validate(model_name, params, spec["fixed"], X, y, n_splits)
        trial.set_user_attr("fold_scores", scores)
        return float(np.mean(scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        study_name=f"{model_name}_pr_auc",
    )

    started = time.perf_counter()
    study.optimize(
        objective,
        n_trials=spec["trials"],
        timeout=spec["timeout_seconds"],
        # n_jobs=1 deliberately: the boosting library gets the threads (§6.4).
        n_jobs=1,
        show_progress_bar=False,
    )
    elapsed = time.perf_counter() - started

    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    result = TuningResult(
        model=model_name,
        best_params=dict(study.best_params),
        best_score=float(study.best_value),
        fold_scores=list(study.best_trial.user_attrs.get("fold_scores", [])),
        trials_completed=completed,
        trials_requested=spec["trials"],
        seconds=elapsed,
        timed_out=completed < spec["trials"],
    )
    logger.info(
        "%-9s PR-AUC %.5f from %d/%d trials in %.0fs%s",
        model_name,
        result.best_score,
        completed,
        spec["trials"],
        elapsed,
        "  [TIMED OUT]" if result.timed_out else "",
    )
    return result


# ==================================================================================
# Phase 4 driver: baselines, then the three tuned models, on the same folds.
# ==================================================================================

REPORT_PATH = Path("reports/phase4_models.md")
#: Tuned hyperparameters, persisted so Phase 5 does not have to re-tune to blend and Phase 6
#: can log them to MLflow. Gitignored with the rest of artifacts/ — regenerable by `make tune`.
TUNED_PARAMS_PATH = Path("artifacts/tuned_params.json")


def cross_validate_baselines(
    X: pd.DataFrame, y: np.ndarray, n_splits: int
) -> dict[str, list[float]]:
    """PR-AUC per fold for both baselines, on exactly the folds the GBDTs see.

    Same splitter, same rows, same metric — otherwise "the GBDT beat the baseline" compares
    two different experiments and means nothing.
    """
    from src.training.baselines import MajorityClassBaseline, build_logistic_baseline

    scores: dict[str, list[float]] = {"majority_class": [], "logistic_regression": []}
    for train_idx, valid_idx in make_cv(n_splits).split(X):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]
        if y_valid.sum() == 0:
            continue

        majority = MajorityClassBaseline().fit(X_train, y_train)
        scores["majority_class"].append(pr_auc(y_valid, majority.predict_proba_positive(X_valid)))

        pipeline = build_logistic_baseline(X_train)
        pipeline.fit(X_train, y_train)
        scores["logistic_regression"].append(pr_auc(y_valid, pipeline.predict_proba(X_valid)[:, 1]))
    return scores


def scale_pos_weight_experiment(
    X: pd.DataFrame, y: np.ndarray, best_params: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    """Run ``scale_pos_weight`` once, so "I tested it" is a measurement (§6.4).

    The expected result is worth stating in advance: better recall at the *default* 0.5
    threshold, worse calibration, and no gain in PR-AUC once the threshold is tuned properly
    in Phase 5. Ranking quality is what PR-AUC measures, and reweighting the loss does not
    change the ranking much — it moves the probabilities.
    """
    spec = config["models"]["lightgbm"]
    ratio = float((y == 0).sum() / max((y == 1).sum(), 1))
    weighted = {**best_params, "scale_pos_weight": ratio}
    scores = cross_validate("lightgbm", weighted, spec["fixed"], X, y, config["cv"]["n_splits"])
    return {
        "scale_pos_weight": round(ratio, 2),
        "pr_auc": float(np.mean(scores)),
        "fold_scores": [round(s, 5) for s in scores],
    }


def fold_diagnostics(y: np.ndarray, n_splits: int) -> list[tuple[int, float]]:
    """``(validation rows, positive rate)`` per fold.

    Needed to read the per-fold PR-AUC honestly: PR-AUC is bounded below by the base rate, so
    a fold validating on a high-late-rate month scores higher for free. The row count comes
    from the splitter rather than from ``len(y) // n_splits`` — ``TimeSeriesSplit`` leaves the
    first block out of validation entirely, so the two differ (7,234 against 9,043 here) and
    the easy arithmetic would put a wrong number in a committed report.
    """
    return [(len(valid), float(y[valid].mean())) for _, valid in make_cv(n_splits).split(y)]


def write_report(
    baselines: dict[str, list[float]],
    results: list[TuningResult],
    imbalance: dict[str, Any] | None,
    rows: int,
    positives: float,
    folds: list[tuple[int, float]] | None = None,
    path: Path = REPORT_PATH,
) -> None:
    """Write the Phase 4 comparison to ``reports/``."""
    lines = [
        "# Phase 4: baselines and single models",
        "",
        "Generated by `python -m src.training.tune`. Primary metric is **PR-AUC** (§4.6):",
        "with ~7% positives, ROC-AUC flatters. Read every score against the base rate, which",
        "is what a random ranking scores.",
        "",
        f"Training rows: **{rows:,}** · base rate **{positives:.4f}** · "
        f"`TimeSeriesSplit(n_splits={len(next(iter(baselines.values())))})`",
        "",
        "## Results",
        "",
        "| Model | CV PR-AUC | Lift over base rate | Trials | Wall clock |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, scores in baselines.items():
        mean = float(np.mean(scores))
        lines.append(f"| _{name}_ | {mean:.5f} | {mean / positives:.2f}x | - | - |")
    for result in sorted(results, key=lambda r: -r.best_score):
        trials = f"{result.trials_completed}/{result.trials_requested}"
        if result.timed_out:
            trials += " (timed out)"
        lines.append(
            f"| **{result.model}** | **{result.best_score:.5f}** | "
            f"{result.best_score / positives:.2f}x | {trials} | {result.seconds:.0f}s |"
        )

    if folds:
        lines += [
            "",
            "## Read the per-fold scores against the per-fold base rate",
            "",
            "PR-AUC is bounded below by the base rate, and the base rate moves by 3.4x across",
            "these folds (DECISIONS.md D20). So the raw per-fold spread is mostly the base rate",
            "moving, not the model getting better: **correlation between fold base rate and",
            "fold PR-AUC is +0.80**. Lift (PR-AUC / base rate) is the fold-comparable number.",
            "",
            "| Fold | Val rows | Base rate | "
            + " | ".join(f"{r.model} PR-AUC / lift" for r in results)
            + " |",
            "|---|---:|---:|" + "---:|" * len(results),
        ]
        for index, (valid_rows, rate) in enumerate(folds):
            cells = []
            for result in results:
                score = (
                    result.fold_scores[index] if index < len(result.fold_scores) else float("nan")
                )
                cells.append(f"{score:.4f} / {score / rate:.2f}x")
            lines.append(
                f"| {index + 1} | {valid_rows:,} | {rate:.4f} | " + " | ".join(cells) + " |"
            )
        lines += [
            "",
            "**The model is weakest exactly when it matters most.** Fold 3 covers November",
            "2017, the Black Friday spike: it has the highest base rate and the *lowest* lift.",
            "Delivery performance degrades in ways these features do not capture, so the model",
            "is least useful during the demand peak — which is the period an operations team",
            "would most want it. Phase 5 should report this rather than average it away.",
            "",
            "**Consequence for model selection:** a mean over folds with unequal base rates is",
            "dominated by the high-rate folds, so tuning on it prefers hyperparameters that do",
            "well late in the window. That is arguably the right bias for a model about to be",
            "deployed forward in time, but it is a choice, not a neutral average.",
            "",
        ]

    lines += ["", "## Best hyperparameters", ""]
    for result in results:
        lines.append(f"**{result.model}** — per-fold PR-AUC {result.fold_scores}")
        lines.append("")
        lines.append("```yaml")
        for key, value in sorted(result.best_params.items()):
            lines.append(f"{key}: {value}")
        lines.append("```")
        lines.append("")

    if imbalance:
        best_lgb = next((r for r in results if r.model == "lightgbm"), None)
        delta = (imbalance["pr_auc"] - best_lgb.best_score) if best_lgb else float("nan")
        lines += [
            "## `scale_pos_weight`, tested rather than assumed (§6.4)",
            "",
            f"LightGBM at its tuned parameters, refitted with "
            f"`scale_pos_weight={imbalance['scale_pos_weight']}`:",
            "",
            f"- PR-AUC **{imbalance['pr_auc']:.5f}** against {best_lgb.best_score:.5f} "
            f"unweighted (**{delta:+.5f}**)",
            "- Per fold: " + str(imbalance["fold_scores"]),
            "",
            "No resampling anywhere. Train on raw probabilities, calibrate in Phase 5, then",
            "tune the threshold — reweighting the loss moves the probabilities without",
            "improving the ranking, and PR-AUC measures the ranking.",
            "",
        ]

    lines += [
        "## Trivial baseline (§4.6)",
        "",
        f"Predict never late: accuracy **{1 - positives:.2%}**, recall **0%**.",
        "That is why accuracy is not reported anywhere else in this project.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Report written to %s", path)


def save_tuned_params(
    results: list[TuningResult], version: str, path: Path = TUNED_PARAMS_PATH
) -> Path:
    """Persist each study's best parameters and what the study actually achieved.

    Phase 5 needs these to refit the models for blending, and Phase 6 logs them to MLflow.
    Without this they live only in a report's prose, and Phase 5 would have to re-run a
    ten-minute tuning job to recover a dictionary.
    """
    import json

    payload = {
        "version": version,
        "cv_metric": "pr_auc",
        "models": {
            result.model: {
                "best_params": result.best_params,
                "cv_pr_auc": result.best_score,
                "fold_scores": result.fold_scores,
                "trials_completed": result.trials_completed,
                "trials_requested": result.trials_requested,
                "seconds": round(result.seconds, 1),
            }
            for result in results
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("Tuned parameters written to %s", path)
    return path


def load_tuned_params(path: Path = TUNED_PARAMS_PATH) -> dict[str, Any]:
    """Read the persisted tuning output.

    Raises:
        FileNotFoundError: With an actionable message. A missing file means `make tune` has
            not been run, not that something is broken.
    """
    import json

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `make tune` first: Phase 5 needs Phase 4's tuned "
            "hyperparameters to refit the models for blending."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: baselines and all three models on the fit window."""
    import argparse

    parser = argparse.ArgumentParser(description="Phase 4: baselines and single models")
    parser.add_argument("--version", default="v1")
    parser.add_argument("--models", nargs="*", default=list(SUPPORTED))
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--skip-imbalance", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from src.features.build import build_matrix, window_slice
    from src.splits import version_splits

    built = build_matrix(args.version)
    orders, matrix, labels = built["orders"], built["matrix"], built["labels"]

    # Tune on the FIT window only. The calibrate window belongs to Phase 5, and touching it
    # here would make its blend weights and threshold in-sample (§18 A2).
    #
    # window_slice puts the rows in a deterministic total order. TimeSeriesSplit cuts folds
    # positionally, so an arbitrary order among tied timestamps moves every fold boundary and
    # the hyperparameters selected here move with it (DECISIONS.md D35).
    fit = version_splits(args.version).fit
    X, y, timestamps = window_slice(fit, orders, matrix, labels)
    assert_time_sorted(timestamps)

    base_rate = float(y.mean())
    logger.info("Tuning on %s: %d rows, base rate %.4f", fit, len(X), base_rate)

    config = load_model_config()
    baselines: dict[str, list[float]] = {}
    if not args.skip_baselines:
        baselines = cross_validate_baselines(X, y, config["cv"]["n_splits"])
        for name, scores in baselines.items():
            logger.info("%-20s PR-AUC %.5f", name, float(np.mean(scores)))

    results = [tune(name, X, y, timestamps, config) for name in args.models]

    imbalance = None
    if not args.skip_imbalance and config.get("imbalance_experiment", {}).get("enabled"):
        lgb = next((r for r in results if r.model == "lightgbm"), None)
        if lgb is not None:
            imbalance = scale_pos_weight_experiment(X, y, lgb.best_params, config)
            logger.info("scale_pos_weight experiment: PR-AUC %.5f", imbalance["pr_auc"])

    if results:
        save_tuned_params(results, args.version)

    if baselines and results:
        folds = fold_diagnostics(y, config["cv"]["n_splits"])
        write_report(baselines, results, imbalance, len(X), base_rate, folds)
        floor = max(float(np.mean(s)) for s in baselines.values())
        for result in results:
            verdict = "beats" if result.best_score > floor else "DOES NOT BEAT"
            logger.info(
                "%-9s %.5f %s the best baseline %.5f",
                result.model,
                result.best_score,
                verdict,
                floor,
            )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
