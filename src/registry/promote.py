"""The promotion gate (PLAN.md §8) — what makes the registry mean something.

Without a gate you have one model version and "model versioning" is decoration. With one, a
retrain cannot quietly replace production: it has to clear a bar that was written down before the
numbers were known.

    A "NO-PROMOTE" RUN IS A SUCCESS
    ==============================
    §8 says it and it is worth repeating where the code lives: a gate that always passes is
    decoration too. The refusals are the entries in ``reports/promotions.md`` that demonstrate the
    thing works, and §13 is explicit that a challenger losing must be recorded rather than tuned
    until it wins.

**Four checks, and what each one is for** (§8):

1. **PR-AUC** must beat the champion by more than ``min_delta_pr_auc``. That threshold is the same
   reproducibility floor as ``decision.selection_band_pr_auc`` (DECISIONS.md D35, D41), and
   :func:`load_gate_config` asserts the gate is never *looser* than selection — promoting on a
   difference selection itself refused to believe would be incoherent.
2. **Brier** must not worsen beyond ``max_brier_regression``. PR-AUC only measures ranking, so
   without this a challenger can buy discrimination with calibration and every downstream decision
   that reads a probability rather than a rank degrades silently.
3. **p95 single-prediction latency** must be under budget. A model that ranks marginally better and
   serves ten times slower is not an improvement at checkout.
4. **The feature schema** must match the serving contract, checked by the same function the API
   uses to refuse to start (CLAUDE.md invariant 17).

**Read PR-AUC here as lift over the base rate, always.** This window sits at a 4.40% positive rate
against the calibration window's ~9.75% (DECISIONS.md D20), so absolute PR-AUC is far lower here by
arithmetic before any model quality enters. The gate compares champion against challenger on the
same rows, so the shift cancels in the *comparison* — but a number quoted without its base rate
will be misread, and this module never quotes one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from src.evaluation.metrics import COMPARISON_EPSILON
from src.evaluation.score_holdout import HoldoutScores, describe, load_holdout, score_model
from src.registry import tracking_uri
from src.registry.pyfunc_wrapper import (
    SignatureMismatchError,
    assert_signature_matches_contract,
)

logger = logging.getLogger(__name__)

REGISTERED_MODEL_NAME = "delivery_delay_classifier"
GATE_CONFIG_PATH = Path("configs/promotion.yaml")
BASE_CONFIG_PATH = Path("configs/base.yaml")

#: This module is authorised to unlock the evaluation window (``src.splits.AUTHORISED_UNLOCKERS``),
#: but it does not need to: it reaches the window only through ``score_holdout``, which owns the
#: unlock. The entry exists so the gate *could* be the caller, and so removing score_holdout would
#: not silently strand the gate.
_THIS_MODULE = "src.registry.promote"


class GateConfigError(ValueError):
    """The gate's configuration is internally inconsistent."""


@dataclass(frozen=True)
class GateCheck:
    """One check's outcome.

    Attributes:
        name: Short identifier, e.g. ``pr_auc``.
        passed: Whether it cleared.
        detail: One sentence naming the numbers, written to be readable months later.
    """

    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class PromotionDecision:
    """The full record of one gate run.

    Attributes:
        promoted: Whether the alias moved.
        challenger_version: Version evaluated.
        champion_version: Version it was measured against, or ``None`` on the first run.
        checks: Every check, in order, pass or fail.
        challenger_scores: The challenger's holdout metrics.
        champion_scores: The champion's, or ``None``.
        reason: Why the decision went the way it did.
        decided_at: UTC timestamp.
    """

    promoted: bool
    challenger_version: str
    champion_version: str | None
    checks: tuple[GateCheck, ...]
    challenger_scores: HoldoutScores | None
    champion_scores: HoldoutScores | None
    reason: str
    decided_at: str

    @property
    def failed_checks(self) -> tuple[str, ...]:
        """Names of the checks that did not clear."""
        return tuple(check.name for check in self.checks if not check.passed)


def load_gate_config(
    gate_path: Path = GATE_CONFIG_PATH, base_path: Path = BASE_CONFIG_PATH
) -> dict[str, Any]:
    """Read the gate's thresholds, and refuse an incoherent set.

    Args:
        gate_path: ``configs/promotion.yaml``.
        base_path: ``configs/base.yaml``, for the selection band.

    Returns:
        The parsed gate config.

    Raises:
        GateConfigError: If ``min_delta_pr_auc`` is below ``decision.selection_band_pr_auc``, or a
            tolerance is negative.

    **Why the cross-file assertion.** ``selection_band_pr_auc`` and ``min_delta_pr_auc`` express the
    same belief — that a PR-AUC difference under this pipeline's reproducibility floor is not a
    difference (D35). Selection uses it to stop choosing between equivalent models on noise; the
    gate uses it to stop promoting on noise. The gate may be *stricter*, because promotion costs an
    image build, a deployment and risk. It must never be looser: that would promote a challenger on
    a margin that selection would have called a tie. Checking it here means the two numbers cannot
    drift apart unnoticed, which is exactly how D33, D38 and D41 each happened.
    """
    gate = yaml.safe_load(gate_path.read_text(encoding="utf-8"))
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    selection_band = float(base["decision"]["selection_band_pr_auc"])
    min_delta = float(gate["min_delta_pr_auc"])

    if min_delta < selection_band:
        raise GateConfigError(
            f"min_delta_pr_auc ({min_delta}) is below decision.selection_band_pr_auc "
            f"({selection_band}). The gate would promote on a margin that model selection treats "
            "as a tie, which is incoherent: the same reproducibility floor cannot be binding in "
            "one place and ignored in the other (DECISIONS.md D35, D41)."
        )
    if float(gate["max_brier_regression"]) < 0:
        raise GateConfigError("max_brier_regression must be non-negative")
    if float(gate["max_p95_latency_ms"]) <= 0:
        raise GateConfigError("max_p95_latency_ms must be positive")
    return gate


def _resolve(client: Any, alias: str) -> tuple[str, Any] | None:
    """The version behind an alias, and its loaded pyfunc, or ``None`` if the alias is unset."""
    import mlflow

    try:
        version = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, alias)
    except Exception:
        logger.info("no @%s alias is set", alias)
        return None
    # str(): int from a SQLite store, str over HTTP (DECISIONS.md D32).
    number = str(version.version)
    model = mlflow.pyfunc.load_model(f"models:/{REGISTERED_MODEL_NAME}@{alias}")
    return number, model


def evaluate_gate(
    challenger: HoldoutScores,
    champion: HoldoutScores | None,
    config: dict[str, Any],
    *,
    schema_ok: bool,
    schema_detail: str,
) -> tuple[bool, tuple[GateCheck, ...], str]:
    """Apply the four checks.

    Args:
        challenger: The challenger's holdout scores.
        champion: The champion's, or ``None`` on the first run.
        config: From :func:`load_gate_config`.
        schema_ok: Whether the challenger's signature matched the serving contract.
        schema_detail: What the schema check found, for the record.

    Returns:
        ``(promoted, checks, reason)``.

    Pure: it takes measurements and returns a verdict, touching no registry and no clock. That is
    what lets the tests drive it with synthetic scores and check the boundaries directly, rather
    than having to manufacture a registry to test a comparison.
    """
    checks: list[GateCheck] = [
        GateCheck("feature_schema", schema_ok, schema_detail),
        GateCheck(
            "p95_latency",
            challenger.p95_latency_ms <= float(config["max_p95_latency_ms"]),
            f"p95 {challenger.p95_latency_ms:.1f} ms against a "
            f"{float(config['max_p95_latency_ms']):.0f} ms budget",
        ),
    ]

    if champion is None:
        # §8: the first version is promoted uncontested, because there is nothing to beat. The
        # schema and latency checks still apply — "nothing to beat" is not "anything goes".
        checks.insert(
            0,
            GateCheck(
                "pr_auc",
                True,
                f"no champion to beat; PR-AUC {challenger.pr_auc:.5f} "
                f"({challenger.lift:.2f}x the {challenger.base_rate:.2%} base rate)",
            ),
        )
        promoted = all(check.passed for check in checks)
        reason = (
            "promoted uncontested: no @champion was set, so there was nothing to beat (§8)"
            if promoted
            else "refused even uncontested: " + "; ".join(c.detail for c in checks if not c.passed)
        )
        return promoted, tuple(checks), reason

    min_delta = float(config["min_delta_pr_auc"])
    delta = challenger.pr_auc - champion.pr_auc
    brier_regression = challenger.brier - champion.brier
    max_regression = float(config["max_brier_regression"])

    checks.insert(
        0,
        GateCheck(
            "pr_auc",
            # > min_delta + COMPARISON_EPSILON, not a bare >. §8 says strictly greater, and
            # `0.065 - 0.060` is 0.005000000000000002 in float64, so a challenger sitting exactly
            # on the bar would promote on representation error alone — the mirror image of the
            # boundary bug D41 hit on the selection band's inclusive side.
            delta > min_delta + COMPARISON_EPSILON,
            f"PR-AUC {challenger.pr_auc:.5f} against the champion's {champion.pr_auc:.5f} "
            f"(delta {delta:+.5f}, bar {min_delta:+.5f}); in lift terms "
            f"{challenger.lift:.2f}x against {champion.lift:.2f}x "
            f"at a {challenger.base_rate:.2%} base rate",
        ),
    )
    checks.insert(
        1,
        GateCheck(
            "brier",
            brier_regression <= max_regression,
            f"Brier {challenger.brier:.6f} against the champion's {champion.brier:.6f} "
            f"(regression {brier_regression:+.6f}, tolerance {max_regression:.6f})",
        ),
    )

    promoted = all(check.passed for check in checks)
    if promoted:
        reason = (
            f"promoted: every check cleared, PR-AUC improved by {delta:+.5f} against a "
            f"{min_delta:+.5f} bar"
        )
    else:
        failed = [check for check in checks if not check.passed]
        reason = "not promoted — " + "; ".join(f"{c.name}: {c.detail}" for c in failed)
        if len(failed) == 1 and failed[0].name == "pr_auc" and abs(delta) <= min_delta:
            # Worth naming explicitly, because it is the interesting case and the one a reader is
            # most likely to mistake for a bug: the two models are statistically indistinguishable,
            # so the refusal is inertia rather than a measured regression.
            #
            # Deliberately says nothing about serving cost. An earlier version of this message
            # asserted the challenger was "cheaper to serve", which was simply untrue in the first
            # real run — v5 shipped XGBoost at 85 MB against the champion's LightGBM at 10 MB. A
            # gate's own explanation is the last place that should contain a claim it has not
            # measured.
            reason += (
                f". The two models are within the {min_delta} equivalence band, so this is a tie "
                "rather than a regression: neither is demonstrably better on these rows. A tie "
                "leaves the champion in place, because promotion costs an image build, a "
                "deployment and risk, and inertia is the right default for a model that works"
            )
    return promoted, tuple(checks), reason


def render_decision(decision: PromotionDecision) -> str:
    """The decision as a Markdown section, appended to ``reports/promotions.md``."""
    verdict = "PROMOTED" if decision.promoted else "NOT PROMOTED"
    champion = f"v{decision.champion_version}" if decision.champion_version else "none"
    lines = [
        f"## {decision.decided_at} — challenger v{decision.challenger_version} vs {champion}"
        f" — **{verdict}**",
        "",
        decision.reason,
        "",
        "| Check | Result | Detail |",
        "|---|:--:|---|",
    ]
    for check in decision.checks:
        lines.append(
            f"| `{check.name}` | {'pass' if check.passed else '**FAIL**'} | {check.detail} |"
        )

    if decision.challenger_scores is not None:
        lines += [
            "",
            "| Metric | Challenger | Champion |",
            "|---|---:|---:|",
        ]
        champion_scores = decision.champion_scores
        for label, key in (
            ("rows scored", "n_rows"),
            ("base rate", "base_rate"),
            ("PR-AUC", "pr_auc"),
            ("**lift over base rate**", "lift"),
            ("ROC-AUC", "roc_auc"),
            ("Brier", "brier"),
            ("threshold", "threshold"),
            ("recall", "recall"),
            ("precision", "precision"),
            ("flag rate", "flag_rate"),
            ("p95 latency (ms)", "p95_latency_ms"),
        ):
            challenger_value = decision.challenger_scores.as_dict()[key]
            champion_value = champion_scores.as_dict()[key] if champion_scores else "—"
            lines.append(f"| {label} | {challenger_value} | {champion_value} |")

    lines += [
        "",
        "Both models scored the **same rows** through the **same code path**, with nothing refit "
        "(§8). Each carries its own bundled as-of snapshot, which is deliberate: part of what a "
        "retrain buys is fresher aggregates, and neither snapshot reaches into this window "
        "(`DECISIONS.md` D33).",
        "",
        "---",
        "",
    ]
    return "\n".join(lines)


def append_report(decision: PromotionDecision, path: Path) -> Path:
    """Append the decision, creating the file with a header on first use."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(
            "# Promotion decisions\n\n"
            "Every run of `make promote` appends here, pass or fail. **The refusals are the "
            "interesting entries**: a gate that always passes is decoration (PLAN.md §8), and a "
            "challenger that loses is recorded rather than tuned until it wins (§13).\n\n"
            "Read every PR-AUC on this page as **lift over the base rate**. The evaluation window "
            "sits at a 4.40% positive rate against the calibration window's ~9.75% "
            "(`DECISIONS.md` D20), so absolute PR-AUC is much lower here by arithmetic, before any "
            "question of model quality. The gate compares both models on the same rows, so the "
            "shift cancels in the comparison.\n\n"
            "This window is a **promotion evaluation set, not a test set** (§18 A3): the decisions "
            "below are what it was used for.\n\n---\n\n",
            encoding="utf-8",
        )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(render_decision(decision))
    logger.info("decision appended to %s", path)
    return path


def run_gate(*, apply: bool = True, config: dict[str, Any] | None = None) -> PromotionDecision:
    """Score both aliases on the holdout, apply the gate, and record the decision.

    Args:
        apply: When ``False``, everything is measured and recorded but no alias moves. Useful for
            seeing what the gate would say without changing production.
        config: Gate thresholds; read from disk when omitted.

    Returns:
        The :class:`PromotionDecision`, which is also appended to the report.

    Raises:
        RuntimeError: If no ``@challenger`` alias is set — there is nothing to evaluate.
    """
    import mlflow

    settings = config or load_gate_config()
    mlflow.set_tracking_uri(tracking_uri())
    client = mlflow.MlflowClient()

    challenger = _resolve(client, settings["challenger_alias"])
    if challenger is None:
        raise RuntimeError(
            f"no @{settings['challenger_alias']} alias is set, so there is nothing to evaluate. "
            "Train a new version first: `make train V=v2`."
        )
    challenger_version, challenger_model = challenger
    champion = _resolve(client, settings["champion_alias"])

    schema_ok, schema_detail = True, "signature matches the serving contract"
    try:
        assert_signature_matches_contract(
            challenger_model, what=f"challenger v{challenger_version}"
        )
    except SignatureMismatchError as exc:
        schema_ok, schema_detail = False, str(exc)
        logger.error("challenger v%s failed the schema check: %s", challenger_version, exc)

    records, labels = load_holdout()
    kwargs = {
        "latency_samples": int(settings["latency_samples"]),
        "latency_warmup": int(settings["latency_warmup"]),
    }
    challenger_scores = score_model(challenger_model, challenger_version, records, labels, **kwargs)
    logger.info("challenger %s", describe(challenger_scores))

    champion_scores = None
    if champion is not None:
        champion_version, champion_model = champion
        champion_scores = score_model(champion_model, champion_version, records, labels, **kwargs)
        logger.info("champion   %s", describe(champion_scores))

    promoted, checks, reason = evaluate_gate(
        challenger_scores,
        champion_scores,
        settings,
        schema_ok=schema_ok,
        schema_detail=schema_detail,
    )

    decision = PromotionDecision(
        promoted=promoted,
        challenger_version=challenger_version,
        champion_version=champion[0] if champion else None,
        checks=checks,
        challenger_scores=challenger_scores,
        champion_scores=champion_scores,
        reason=reason,
        decided_at=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
    )

    if promoted and apply:
        client.set_registered_model_alias(
            REGISTERED_MODEL_NAME, settings["champion_alias"], challenger_version
        )
        if champion is not None:
            client.set_model_version_tag(
                REGISTERED_MODEL_NAME, champion[0], "status", settings["archived_tag_value"]
            )
            client.set_model_version_tag(
                REGISTERED_MODEL_NAME,
                champion[0],
                "withdrawn_reason",
                f"archived: superseded by v{challenger_version} through the promotion gate",
            )
        client.delete_registered_model_alias(REGISTERED_MODEL_NAME, settings["challenger_alias"])
        logger.info("@champion now points at v%s", challenger_version)
    elif promoted:
        logger.info("would promote v%s, but apply=False", challenger_version)
    else:
        logger.info("champion unchanged. %s", reason)

    # Both branches record. A refusal that leaves no trace is indistinguishable from a gate that
    # was never run.
    append_report(decision, Path(settings["report_path"]))
    for check in decision.checks:
        logger.info("  %-15s %s — %s", check.name, "pass" if check.passed else "FAIL", check.detail)
    return decision


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for Phase 9."""
    import argparse

    parser = argparse.ArgumentParser(description="Phase 9: run the promotion gate")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="measure and record, but do not move the alias",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    decision = run_gate(apply=not args.dry_run)
    verdict = "PROMOTED" if decision.promoted else "NOT PROMOTED"
    logger.info("DECISION: %s — %s", verdict, decision.reason)
    # Exit 0 either way: a refusal is a successful run of the gate, not a failure of it (§8).
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
