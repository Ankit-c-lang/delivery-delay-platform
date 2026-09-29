"""A full versioned training run, logged to MLflow and registered (PLAN.md §7, §13 Phase 6).

**This module is the spine.** One invocation produces a registered, self-contained model
version: a parent run named ``{version}_{timestamp}`` with nested children for the baseline
and each model, then an ``ensemble`` child that fits the blend, calibrates, chooses the
threshold and logs the pyfunc.

**Optuna trials are not MLflow runs.** §13 names that as a common mistake, and it is: 100
trials across three studies would bury the five runs that mean something. Tuning is a separate
step (`make tune`) whose output is a JSON file; this module loads it and logs the provenance —
trials completed against trials requested, and the seconds spent — as params on each child.

**One §7 instruction cannot be followed here, and that is correct.** §7 asks each registered
version to be tagged with its *holdout* PR-AUC. The 2018-05 → 08 window is locked at runtime
and may be read only by the promotion gate (§18 A3, DECISIONS.md D25), so Phase 6 has no
legitimate way to compute it. The tag is written as ``pending`` and Phase 9's gate fills it in
when it scores the version. Registering a model is not the same event as evaluating it, and the
lock makes that ordering explicit rather than optional.

**Aliases, never stages.** Stages were deprecated in MLflow 2.9 and **removed** in MLflow 3,
which is the version installed here (3.16.1), so `@champion`/`@challenger` is the only
mechanism available rather than merely the recommended one (§18 A7).

Run with::

    python -m src.training.train --version v1
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import yaml
from mlflow.models import infer_signature

from src.evaluation.metrics import summarize, trivial_baseline
from src.evaluation.plots import all_plots
from src.features.build import FEATURE_FAMILIES, FEATURE_NAMES
from src.registry.pyfunc_wrapper import DelayPredictor, raw_input_example
from src.training.ensemble import (
    blend_probabilities,
    evaluate_blend,
    fit_blend_weights,
    fit_calibrator,
    select_single_model,
    split_calibration_window,
    sweep_threshold,
)
from src.training.tune import (
    fit_final_model,
    load_model_config,
    load_tuned_params,
    predict_positive,
)

logger = logging.getLogger(__name__)

REGISTERED_MODEL_NAME = "delivery_delay_classifier"
CHAMPION_ALIAS = "champion"
CHALLENGER_ALIAS = "challenger"
BASE_CONFIG = Path("configs/base.yaml")


def git_sha() -> str:
    """Current commit, or ``unknown`` outside a repository.

    Logged as a param on every run: a metric without the code that produced it is not
    reproducible, and "which commit was this?" is the first question asked of any result.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=10
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def git_is_dirty() -> bool:
    """Whether the working tree has uncommitted changes.

    Logged alongside the SHA, because a SHA plus a dirty tree does not identify the code that
    ran and it is better to know that than to assume otherwise.
    """
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return bool(result.stdout.strip())
    except (subprocess.SubprocessError, OSError):
        return False


@dataclass
class TrainingResult:
    """What a run produced."""

    version: str
    parent_run_id: str
    model_version: str | None
    alias: str | None
    shipped_model: str
    threshold: float
    metrics: dict[str, float]
    blend_delta: float


def _log_common_params(version: str, splits: Any, artifact: Any, seed: int) -> None:
    """Params every run in this session shares (§7)."""
    mlflow.log_params(
        {
            "version": version,
            "git_sha": git_sha(),
            "git_dirty": git_is_dirty(),
            "seed": seed,
            "n_features": len(FEATURE_NAMES),
            "feature_families": json.dumps(
                {name: len(members) for name, members in FEATURE_FAMILIES.items()}
            ),
            "fit_start": str(splits.fit.start),
            "fit_end": str(splits.fit.end),
            "calibrate_start": str(splits.calibrate.start),
            "calibrate_end": str(splits.calibrate.end),
            "aggregates_through": str(splits.aggregates_through),
            "snapshot_month": str(artifact.latest_snapshot_month),
            "train_cutoff": str(artifact.train_cutoff),
            "input_schema_hash": artifact.input_schema_hash,
        }
    )


def _log_feature_list(directory: Path) -> None:
    """Log the ordered feature list as an artifact (§7).

    The order matters as much as the membership: §6.6 requires the matrix column order to
    match the artifact's stored list, so the list is logged as an ordered array rather than a
    set.
    """
    path = directory / "feature_list.json"
    path.write_text(
        json.dumps(
            {
                "n_features": len(FEATURE_NAMES),
                "ordered": list(FEATURE_NAMES),
                "families": {name: list(members) for name, members in FEATURE_FAMILIES.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    mlflow.log_artifact(str(path))


def train(version: str = "v1", register: bool = True) -> TrainingResult:
    """Run the full versioned training session.

    Args:
        version: Key in ``configs/splits.yaml``.
        register: Whether to register the resulting pyfunc in the Model Registry.

    Returns:
        A :class:`TrainingResult` naming the run, the registered version and the alias set.
    """
    from src.features.build import build_matrix
    from src.splits import version_splits

    base = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    decision_cfg, mlflow_cfg = base["decision"], base["mlflow"]
    capacity = base["metrics"]["review_capacity"]
    model_cfg = load_model_config()
    tuned = load_tuned_params()
    seed = model_cfg.get("seed", 42)

    mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri_env", None) or _tracking_uri())
    mlflow.set_experiment(mlflow_cfg["experiment"])

    built = build_matrix(version)
    orders, matrix, labels, artifact = (
        built["orders"],
        built["matrix"],
        built["labels"],
        built["artifact"],
    )
    splits = version_splits(version)

    def slice_window(window):
        mask = window.mask(orders["order_purchase_timestamp"]).to_numpy()
        stamps = orders.loc[mask, "order_purchase_timestamp"]
        order = np.argsort(stamps.to_numpy(), kind="stable")
        return (
            matrix.loc[mask].iloc[order].reset_index(drop=True),
            labels.loc[mask].to_numpy()[order].astype(int),
            stamps.iloc[order].reset_index(drop=True),
            orders.loc[mask].iloc[order].reset_index(drop=True),
        )

    X_fit, y_fit, _, _ = slice_window(splits.fit)
    X_cal, y_cal, stamps_cal, orders_cal = slice_window(splits.calibrate)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    parent_name = f"delivery_delay_{version}_{timestamp}"
    probabilities: dict[str, np.ndarray] = {}
    models: dict[str, Any] = {}

    with tempfile.TemporaryDirectory() as tmp, mlflow.start_run(run_name=parent_name) as parent:
        scratch = Path(tmp)
        _log_common_params(version, splits, artifact, seed)
        mlflow.set_tags(
            {
                "phase": "6",
                "plan_section": "7",
                "shipped_decision": "pending",
                "note": "Optuna trials live in the study, not as runs (§13)",
            }
        )
        _log_feature_list(scratch)
        mlflow.log_params(
            {"fit_rows": len(X_fit), "calibrate_rows": len(X_cal), "base_rate": float(y_fit.mean())}
        )

        # --- child: trivial + logistic baselines --------------------------------------
        with mlflow.start_run(run_name="baseline_logreg", nested=True):
            from src.training.baselines import build_logistic_baseline

            started = time.perf_counter()
            pipeline = build_logistic_baseline(X_fit)
            pipeline.fit(X_fit, y_fit)
            scores = pipeline.predict_proba(X_cal)[:, 1]
            mlflow.log_param("model", "logistic_regression")
            mlflow.log_metric("fit_seconds", time.perf_counter() - started)
            mlflow.log_metrics(summarize(y_cal, scores, capacity))
            trivial = trivial_baseline(y_cal)
            mlflow.log_metrics(
                {"trivial_accuracy": trivial["accuracy"], "trivial_recall": trivial["recall"]}
            )
            logger.info("baseline_logreg PR-AUC %.5f", summarize(y_cal, scores)["pr_auc"])

        # --- children: one per model ---------------------------------------------------
        for name in sorted(tuned["models"]):
            entry = tuned["models"][name]
            with mlflow.start_run(run_name=name, nested=True):
                started = time.perf_counter()
                models[name] = fit_final_model(
                    name, entry["best_params"], model_cfg["models"][name]["fixed"], X_fit, y_fit
                )
                probabilities[name] = predict_positive(name, models[name], X_cal)
                mlflow.log_param("model", name)
                mlflow.log_params({f"hp_{k}": v for k, v in entry["best_params"].items()})
                # Tuning provenance, not tuning runs (§13).
                mlflow.log_params(
                    {
                        "tuning_trials_completed": entry["trials_completed"],
                        "tuning_trials_requested": entry["trials_requested"],
                        "tuning_seconds": entry["seconds"],
                        "tuning_cv_pr_auc": entry["cv_pr_auc"],
                    }
                )
                mlflow.log_metric("fit_seconds", time.perf_counter() - started)
                mlflow.log_metrics(summarize(y_cal, probabilities[name], capacity))
                logger.info(
                    "%-9s PR-AUC %.5f on the calibration window",
                    name,
                    summarize(y_cal, probabilities[name])["pr_auc"],
                )

        # --- child: blend, calibration, threshold, and the registered model ------------
        with mlflow.start_run(run_name="ensemble", nested=True):
            weight_half, calib_half = split_calibration_window(
                stamps_cal, decision_cfg["calibration_split"]
            )
            weights = fit_blend_weights(
                {n: p[weight_half] for n, p in probabilities.items()}, y_cal[weight_half]
            )
            blend = evaluate_blend(
                {n: p[weight_half] for n, p in probabilities.items()},
                y_cal[weight_half],
                weights,
                decision_cfg["min_ensemble_gain_pr_auc"],
            )
            # select_single_model, NOT `blend.best_single`. This line used to re-derive the
            # shipping decision inline, so when ensemble.build_decision gained the equivalence
            # band (DECISIONS.md D41) the two disagreed and v3 was registered as catboost while
            # `make decide` had chosen lightgbm. The rule now has one implementation and this is a
            # caller of it — the same shape of bug as D33 and D38, and the same fix.
            if blend.ship_ensemble:
                shipped = "blend"
                selection = None
                scores_cal = blend_probabilities(probabilities, weights)
            else:
                selection = select_single_model(
                    blend.single_pr_auc,
                    decision_cfg["selection_band_pr_auc"],
                    decision_cfg["serving_cost_mb"],
                )
                shipped = selection.chosen
                logger.info("selection: %s", selection.sentence())
                scores_cal = probabilities[shipped]

            calibration = fit_calibrator(scores_cal[calib_half], y_cal[calib_half])
            calibrated_all = calibration.calibrator.predict(scores_cal)
            threshold = sweep_threshold(
                calibration.calibrator.predict(scores_cal[calib_half]),
                y_cal[calib_half],
                decision_cfg["fn_fp_cost_ratio"],
            )

            mlflow.log_params(
                {
                    "shipped_model": shipped,
                    "calibration_split": decision_cfg["calibration_split"],
                    "min_ensemble_gain_pr_auc": decision_cfg["min_ensemble_gain_pr_auc"],
                    "fn_fp_cost_ratio": decision_cfg["fn_fp_cost_ratio"],
                    **{f"blend_weight_{n}": round(w, 6) for n, w in weights.items()},
                }
            )
            mlflow.log_metrics(
                {
                    "blend_pr_auc": blend.blend_pr_auc,
                    "best_single_pr_auc": blend.single_pr_auc[blend.best_single],
                    "blend_delta_pr_auc": blend.delta,
                    "brier_before_calibration": calibration.brier_before,
                    "brier_after_calibration": calibration.brier_after,
                    "best_threshold": threshold.threshold,
                    "expected_cost_at_threshold": threshold.expected_cost_per_order,
                    "threshold_precision": threshold.precision,
                    "threshold_recall": threshold.recall,
                    "threshold_flag_rate": threshold.flag_rate,
                }
            )
            mlflow.log_metrics(summarize(y_cal, calibrated_all, capacity, prefix="calibrated_"))

            plots = all_plots(
                y_cal, calibrated_all, threshold.threshold, scratch / "plots", shipped
            )
            for path in plots.values():
                mlflow.log_artifact(str(path), artifact_path="plots")
            for shap_png in sorted(Path("reports/shap").glob("*.png")):
                mlflow.log_artifact(str(shap_png), artifact_path="shap")

            # The tuning provenance file stands in for "the Optuna study" (§7): the study
            # itself lives in `make tune`, which is a separate step by design.
            tuned_path = scratch / "tuned_params.json"
            tuned_path.write_text(json.dumps(tuned, indent=2, sort_keys=True), encoding="utf-8")
            mlflow.log_artifact(str(tuned_path))

            decided = artifact.with_decision(
                calibrator=calibration.calibrator, threshold=threshold.threshold
            )
            # Logged explicitly as well as bundled: §13 names "logging the model but not the
            # preprocessing artifact" as the classic skew bug, and having it separately
            # downloadable makes an audit possible without unpickling the pyfunc.
            artifact_path = scratch / "preprocessing_artifact.joblib"
            decided.save(artifact_path)
            mlflow.log_artifact(str(artifact_path))

            wrapper = DelayPredictor(decided, models.get(shipped), shipped)
            example = raw_input_example(orders_cal)
            signature = infer_signature(example, wrapper.predict(None, example))
            mlflow.log_params(wrapper.describe())

            info = mlflow.pyfunc.log_model(
                name="model",
                python_model=wrapper,
                signature=signature,
                input_example=example,
                registered_model_name=REGISTERED_MODEL_NAME if register else None,
            )
            logger.info("pyfunc logged at %s", info.model_uri)

        mlflow.set_tag("shipped_decision", shipped)

    model_version, alias = (None, None)
    if register:
        model_version, alias = _assign_alias(
            version_key=version,
            splits=splits,
            blend=blend,
            threshold=threshold.threshold,
            calibration_pr_auc=summarize(y_cal, calibrated_all)["pr_auc"],
            # Passed in rather than re-derived from `blend`: deriving it here is exactly how the
            # tag came to disagree with the pyfunc's own model (D41).
            shipped_model=shipped,
        )

    return TrainingResult(
        version=version,
        parent_run_id=parent.info.run_id,
        model_version=model_version,
        alias=alias,
        shipped_model=shipped,
        threshold=threshold.threshold,
        metrics=summarize(y_cal, calibrated_all, capacity),
        blend_delta=blend.delta,
    )


def _tracking_uri() -> str:
    """Tracking URI from the environment, defaulting to the Compose service.

    Clients always speak HTTP to the tracking server; only the server itself touches the
    SQLite backend store (DECISIONS.md D12).
    """
    import os

    return os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")


def _assign_alias(
    version_key: str,
    splits: Any,
    blend: Any,
    threshold: float,
    calibration_pr_auc: float,
    shipped_model: str,
) -> tuple[str, str]:
    """Tag the new registry version and give it an alias.

    Returns:
        ``(model_version, alias)``.

    The first version becomes ``@champion`` uncontested — §8 says v1 is auto-promoted because
    there is nothing to beat. Any later version becomes ``@challenger`` and waits for the
    Phase 9 gate. Promotion is the gate's decision, not a training run's, and encoding that
    here keeps a retrain from quietly replacing production.
    """
    client = mlflow.MlflowClient()
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    latest = max(versions, key=lambda v: int(v.version))
    # Normalised to str: MLflow returns `version` as an int from a SQLite store and as a str
    # over HTTP, so a caller comparing it would behave differently depending on the backend.
    latest_version = str(latest.version)

    client.set_model_version_tag(
        REGISTERED_MODEL_NAME,
        latest_version,
        "training_window",
        f"{splits.fit.start}..{splits.fit.end}",
    )
    client.set_model_version_tag(
        REGISTERED_MODEL_NAME,
        latest_version,
        "calibration_window",
        f"{splits.calibrate.start}..{splits.calibrate.end}",
    )
    client.set_model_version_tag(REGISTERED_MODEL_NAME, latest_version, "git_sha", git_sha())
    client.set_model_version_tag(
        REGISTERED_MODEL_NAME, latest_version, "split_version", version_key
    )
    client.set_model_version_tag(
        REGISTERED_MODEL_NAME, latest_version, "shipped_model", shipped_model
    )
    client.set_model_version_tag(
        REGISTERED_MODEL_NAME, latest_version, "blend_delta_pr_auc", f"{blend.delta:.5f}"
    )
    client.set_model_version_tag(
        REGISTERED_MODEL_NAME, latest_version, "threshold", f"{threshold:.5f}"
    )
    client.set_model_version_tag(
        REGISTERED_MODEL_NAME,
        latest_version,
        "calibration_window_pr_auc",
        f"{calibration_pr_auc:.5f}",
    )
    # §7 asks for the holdout PR-AUC here. The 2018-05 -> 08 window is locked to the promotion
    # gate (§18 A3, D25), so a training run cannot compute it. The gate fills this in.
    client.set_model_version_tag(
        REGISTERED_MODEL_NAME,
        latest_version,
        "evaluation_pr_auc",
        "pending: set by the promotion gate (§18 A3)",
    )

    existing_champion = None
    try:
        existing_champion = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS)
    except Exception:
        existing_champion = None

    if existing_champion is None:
        alias = CHAMPION_ALIAS
        reason = "first version; nothing to beat (§8)"
    else:
        alias = CHALLENGER_ALIAS
        reason = f"champion v{existing_champion.version} exists; promotion is the gate's call"
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, alias, latest_version)
    client.set_model_version_tag(REGISTERED_MODEL_NAME, latest_version, "alias_reason", reason)
    logger.info("registered v%s and set @%s (%s)", latest_version, alias, reason)
    return latest_version, alias


def describe_registry() -> list[dict[str, Any]]:
    """Every registered version with its aliases and tags, oldest first.

    Lives here rather than in a shell one-liner because the Makefile cannot carry nested
    quoting reliably, and because the code that writes these tags is the right place to read
    them back.
    """
    mlflow.set_tracking_uri(_tracking_uri())
    client = mlflow.MlflowClient()
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")

    # Aliases come from the registered model, NOT from search_model_versions: that endpoint
    # returns an empty `aliases` field on every version, so reading it there reports "no alias"
    # for a model that is in fact the champion. One call, inverted alias -> version into
    # version -> aliases.
    alias_map: dict[str, list[str]] = {}
    for alias, version_number in client.get_registered_model(REGISTERED_MODEL_NAME).aliases.items():
        alias_map.setdefault(str(version_number), []).append(alias)

    rows = []
    for version in sorted(versions, key=lambda v: int(v.version)):
        rows.append(
            {
                "version": str(version.version),
                "aliases": sorted(alias_map.get(str(version.version), [])),
                "shipped_model": version.tags.get("shipped_model", "?"),
                "threshold": version.tags.get("threshold", "?"),
                "training_window": version.tags.get("training_window", "?"),
                "calibration_window_pr_auc": version.tags.get("calibration_window_pr_auc", "?"),
                "evaluation_pr_auc": version.tags.get("evaluation_pr_auc", "?"),
                "git_sha": (version.tags.get("git_sha") or "?")[:8],
                "alias_reason": version.tags.get("alias_reason", "?"),
                # A withdrawn version keeps its metrics, so without this the listing shows a
                # plausible-looking candidate that must never be deployed (DECISIONS.md D33).
                "status": version.tags.get("status", "active"),
                "withdrawn_reason": version.tags.get("withdrawn_reason", ""),
            }
        )
    return rows


def print_registry() -> None:
    """Print :func:`describe_registry` as a table."""
    rows = describe_registry()
    if not rows:
        print(f"no versions registered under {REGISTERED_MODEL_NAME!r}; run `make train`")
        return
    print(f"{REGISTERED_MODEL_NAME}:")
    for row in rows:
        aliases = ("@" + ", @".join(row["aliases"])) if row["aliases"] else "(no alias)"
        print(
            f"  v{row['version']:>3s}  {aliases:<24s} {row['shipped_model']:<10s} "
            f"thr={row['threshold']:<9s} cal_pr_auc={row['calibration_window_pr_auc']:<9s} "
            f"git={row['git_sha']}"
        )
        print(f"        train={row['training_window']}  eval_pr_auc={row['evaluation_pr_auc']}")
        print(f"        alias: {row['alias_reason']}")
        if row["status"] != "active":
            print(f"        STATUS: {row['status'].upper()} — {row['withdrawn_reason']}")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Phase 6: a full versioned training run")
    parser.add_argument("--version", default="v1")
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--show-registry", action="store_true", help="print the registry and exit")
    args = parser.parse_args(argv)

    if args.show_registry:
        print_registry()
        return 0

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    result = train(args.version, register=not args.no_register)
    logger.info(
        "run %s: shipped %s, threshold %.4f, registry v%s @%s",
        result.parent_run_id,
        result.shipped_model,
        result.threshold,
        result.model_version,
        result.alias,
    )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
