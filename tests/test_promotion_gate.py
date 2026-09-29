"""The promotion gate (PLAN.md §8, §13 Phase 9).

§13's required tests: a deliberately worse challenger is rejected, a clearly better one is
accepted, a schema mismatch is rejected regardless of metrics, and the holdout window is loaded by
exactly one module — asserted by import inspection.

**Most of these drive :func:`~src.registry.promote.evaluate_gate` directly with synthetic scores.**
That function is pure: measurements in, verdict out, touching no registry and no clock. So the
boundaries can be tested exactly, rather than by manufacturing a registry and hoping a real model
lands near a threshold. The branches a real run *cannot* reach — the uncontested first promotion,
a clearly better challenger — are only testable this way, because the live registry already has a
champion and the real challenger lost.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from src.evaluation.score_holdout import HoldoutScores
from src.registry.promote import (
    GateConfigError,
    evaluate_gate,
    load_gate_config,
    render_decision,
)

ROOT = Path(__file__).resolve().parents[1]

#: The gate's thresholds as configured, so the tests move with the config rather than hardcoding it.
CONFIG = load_gate_config()


def scores(
    *,
    version: str = "9",
    pr_auc: float = 0.060,
    brier: float = 0.0490,
    p95: float = 70.0,
    base_rate: float = 0.044,
) -> HoldoutScores:
    """A holdout result with the real window's shape, overridable per test."""
    return HoldoutScores(
        model_version=version,
        n_rows=25_352,
        base_rate=base_rate,
        pr_auc=pr_auc,
        lift=pr_auc / base_rate,
        roc_auc=0.61,
        brier=brier,
        threshold=0.177,
        recall=0.23,
        precision=0.062,
        flag_rate=0.163,
        p95_latency_ms=p95,
    )


def run(challenger: HoldoutScores, champion: HoldoutScores | None, **overrides):
    """Apply the gate with the real config, optionally overriding a threshold."""
    config = {**CONFIG, **overrides}
    return evaluate_gate(
        challenger, champion, config, schema_ok=True, schema_detail="signature matches"
    )


class TestTheGateRefuses:
    """§13: a deliberately worse challenger is rejected. The refusals are the point (§8)."""

    def test_a_clearly_worse_challenger_is_rejected(self):
        promoted, checks, reason = run(scores(pr_auc=0.030), scores(version="1", pr_auc=0.060))
        assert promoted is False
        assert "pr_auc" in [c.name for c in checks if not c.passed]
        assert "not promoted" in reason

    def test_an_equal_challenger_is_rejected(self):
        """Identical metrics must not promote. A gate that ties goes to the incumbent."""
        promoted, _, reason = run(scores(pr_auc=0.060), scores(version="1", pr_auc=0.060))
        assert promoted is False
        assert "equivalence band" in reason

    def test_a_challenger_better_by_less_than_min_delta_is_rejected(self):
        """The real case from the first live run: better, but not by enough to believe."""
        champion = scores(version="1", pr_auc=0.060)
        promoted, _, reason = run(scores(pr_auc=0.060 + CONFIG["min_delta_pr_auc"] / 2), champion)
        assert promoted is False
        assert "equivalence band" in reason

    def test_the_bar_is_strictly_greater_than_min_delta(self):
        """Exactly at the bar does not clear it. §8 says "> champion + min_delta"."""
        champion = scores(version="1", pr_auc=0.060)
        at_bar = run(scores(pr_auc=0.060 + CONFIG["min_delta_pr_auc"]), champion)
        assert at_bar[0] is False
        just_over = run(scores(pr_auc=0.060 + CONFIG["min_delta_pr_auc"] * 1.01), champion)
        assert just_over[0] is True

    def test_the_tie_message_makes_no_claim_about_serving_cost(self):
        """It once asserted the challenger was "cheaper to serve", which was false in the first
        real run — v5 shipped XGBoost at 85 MB against the champion's LightGBM at 10 MB. A gate's
        own explanation must not contain a claim it has not measured."""
        _, _, reason = run(scores(pr_auc=0.060), scores(version="1", pr_auc=0.060))
        assert "cheaper" not in reason.lower()
        assert "mb" not in reason.lower()


class TestTheGateAccepts:
    """§13: a clearly better challenger is accepted.

    Only reachable synthetically: the real challenger lost.
    """

    def test_a_clearly_better_challenger_is_promoted(self):
        promoted, checks, reason = run(scores(pr_auc=0.090), scores(version="1", pr_auc=0.060))
        assert promoted is True
        assert all(check.passed for check in checks)
        assert "promoted" in reason

    def test_the_first_version_is_promoted_uncontested(self):
        """§8: nothing to beat. The live registry has had a champion since Phase 6, so this branch
        cannot be exercised for real without destroying state — which is why it is tested here."""
        promoted, checks, reason = run(scores(), None)
        assert promoted is True
        assert "uncontested" in reason
        assert [c.name for c in checks] == ["pr_auc", "feature_schema", "p95_latency"]

    def test_uncontested_does_not_mean_unchecked(self):
        """A first version with a broken schema or terrible latency is still refused."""
        config = {**CONFIG}
        promoted, _, reason = evaluate_gate(
            scores(), None, config, schema_ok=False, schema_detail="missing 3 columns"
        )
        assert promoted is False
        assert "refused even uncontested" in reason

        slow = evaluate_gate(
            scores(p95=CONFIG["max_p95_latency_ms"] * 2),
            None,
            config,
            schema_ok=True,
            schema_detail="ok",
        )
        assert slow[0] is False


class TestTheCalibrationGuard:
    """Brier stops a challenger buying ranking with calibration."""

    def test_a_better_ranker_with_much_worse_calibration_is_rejected(self):
        """PR-AUC only measures order. Without this check a model can win on it while its
        probabilities drift, and every downstream decision that reads a probability degrades."""
        champion = scores(version="1", pr_auc=0.060, brier=0.0490)
        challenger = scores(pr_auc=0.090, brier=0.0490 + CONFIG["max_brier_regression"] * 2)
        promoted, checks, _ = run(challenger, champion)
        assert promoted is False
        assert [c.name for c in checks if not c.passed] == ["brier"]

    def test_brier_regression_inside_the_tolerance_is_allowed(self):
        """D20: this window's base rate is 4.40% against ~9.75% at calibration, so a calibrator
        fitted there over-predicts here. A tight tolerance would reject every honest challenger."""
        champion = scores(version="1", pr_auc=0.060, brier=0.0490)
        challenger = scores(pr_auc=0.090, brier=0.0490 + CONFIG["max_brier_regression"] * 0.9)
        assert run(challenger, champion)[0] is True

    def test_improving_brier_is_never_penalised(self):
        champion = scores(version="1", pr_auc=0.060, brier=0.0600)
        assert run(scores(pr_auc=0.090, brier=0.0400), champion)[0] is True


class TestTheSchemaCheck:
    """§13: a schema mismatch is rejected regardless of metrics."""

    def test_a_schema_mismatch_rejects_an_otherwise_perfect_challenger(self):
        promoted, checks, _ = evaluate_gate(
            scores(pr_auc=0.500, brier=0.001, p95=1.0),
            scores(version="1", pr_auc=0.060),
            CONFIG,
            schema_ok=False,
            schema_detail="missing=['promised_days']",
        )
        assert promoted is False
        assert [c.name for c in checks if not c.passed] == ["feature_schema"]

    def test_the_check_shares_one_implementation_with_the_api(self):
        """CLAUDE.md invariant 17. The API refuses to start on a mismatch and the gate refuses to
        promote; both call the same function, which lives beside the contract it checks."""
        from src.registry import pyfunc_wrapper

        assert hasattr(pyfunc_wrapper, "assert_signature_matches_contract")
        api_source = (ROOT / "api" / "model_loader.py").read_text(encoding="utf-8")
        assert "from src.registry.pyfunc_wrapper import assert_signature_matches_contract" in (
            api_source
        )
        assert "def assert_signature_matches_contract" not in api_source


class TestTheLatencyBudget:
    def test_a_slow_challenger_is_rejected_however_good_it_ranks(self):
        """A model that ranks marginally better and serves ten times slower is not an improvement
        at checkout."""
        promoted, checks, _ = run(
            scores(pr_auc=0.200, p95=CONFIG["max_p95_latency_ms"] + 1),
            scores(version="1", pr_auc=0.060),
        )
        assert promoted is False
        assert "p95_latency" in [c.name for c in checks if not c.passed]

    def test_the_budget_boundary_is_inclusive(self):
        champion = scores(version="1", pr_auc=0.060)
        at_budget = run(scores(pr_auc=0.090, p95=CONFIG["max_p95_latency_ms"]), champion)
        assert at_budget[0] is True


class TestTheGateConfigCannotDriftFromSelection:
    """The cross-file invariant, which is the D33/D38/D41 lesson applied before the fact."""

    def test_a_min_delta_below_the_selection_band_is_refused(self, tmp_path):
        import yaml

        gate = yaml.safe_load(Path("configs/promotion.yaml").read_text(encoding="utf-8"))
        gate["min_delta_pr_auc"] = 0.0001
        loose = tmp_path / "promotion.yaml"
        loose.write_text(yaml.safe_dump(gate), encoding="utf-8")

        with pytest.raises(GateConfigError, match=r"below decision\.selection_band_pr_auc"):
            load_gate_config(loose, Path("configs/base.yaml"))

    def test_the_configured_gate_is_at_least_as_strict_as_selection(self):
        import yaml

        base = yaml.safe_load(Path("configs/base.yaml").read_text(encoding="utf-8"))
        assert CONFIG["min_delta_pr_auc"] >= base["decision"]["selection_band_pr_auc"]

    def test_a_stricter_gate_is_allowed(self, tmp_path):
        """The gate may demand more than selection, because promotion costs a deployment."""
        import yaml

        gate = yaml.safe_load(Path("configs/promotion.yaml").read_text(encoding="utf-8"))
        gate["min_delta_pr_auc"] = 0.05
        strict = tmp_path / "promotion.yaml"
        strict.write_text(yaml.safe_dump(gate), encoding="utf-8")
        assert load_gate_config(strict, Path("configs/base.yaml"))["min_delta_pr_auc"] == 0.05


class TestOnlyOneModuleLoadsTheHoldout:
    """§13: asserted by import inspection, because the lock protects the window definition.

    The realistic leak is not a rogue import. It is a stray
    ``orders[orders.purchase >= "2018-05-01"]`` in a notebook, written by someone who only wanted
    "recent data" — which is why `src/splits.py` guards the definition and a test looks for the
    literal as well as the import.
    """

    HOLDOUT_START = "2018-05-01"

    def test_score_holdout_is_the_only_authorised_unlocker_that_reads_the_window(self):
        from src.splits import AUTHORISED_UNLOCKERS

        assert {"src.evaluation.score_holdout", "src.registry.promote"} == AUTHORISED_UNLOCKERS

    def test_no_other_module_calls_promotion_evaluation_window(self):
        callers: list[str] = []
        for path in sorted((ROOT / "src").rglob("*.py")):
            module = "src." + str(path.relative_to(ROOT / "src")).removesuffix(".py").replace(
                "/", "."
            )
            module = module.removesuffix(".__init__")
            if module in {"src.splits", "src.evaluation.score_holdout", "src.registry.promote"}:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and node.id in {
                    "promotion_evaluation_window",
                    "unlock_promotion_evaluation",
                }:
                    callers.append(f"{path}:{node.lineno}")
        assert not callers, "only score_holdout and promote may touch the window: " + ", ".join(
            callers
        )

    def test_the_holdout_start_date_appears_in_no_module_but_splits(self):
        """A hardcoded boundary anywhere else is the leak the lock cannot catch.

        Inspects string literals via `ast`, not raw text, so prose in a docstring explaining the
        window does not trip it — the same approach `tests/test_splits.py` uses.
        """
        offenders: list[str] = []
        for path in sorted((ROOT / "src").rglob("*.py")):
            if path.name == "splits.py":
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and self.HOLDOUT_START in node.value
                    and node.col_offset != 0  # a module docstring starts at column 0
                ):
                    offenders.append(f"{path}:{node.lineno}")
        assert not offenders, (
            "the window's boundary must come from configs/splits.yaml via src/splits.py, "
            "never a literal: " + ", ".join(offenders)
        )

    def test_the_window_raises_for_an_unauthorised_caller(self):
        from src.splits import PromotionEvaluationLockedError, promotion_evaluation_window

        with pytest.raises(PromotionEvaluationLockedError):
            promotion_evaluation_window()


class TestTheDecisionRecord:
    """A refusal that leaves no trace is indistinguishable from a gate that never ran."""

    def test_a_refusal_renders_with_its_failing_check_marked(self):
        from src.registry.promote import PromotionDecision

        challenger = scores(pr_auc=0.057)
        champion = scores(version="4", pr_auc=0.060)
        promoted, checks, reason = run(challenger, champion)
        decision = PromotionDecision(
            promoted=promoted,
            challenger_version="5",
            champion_version="4",
            checks=checks,
            challenger_scores=challenger,
            champion_scores=champion,
            reason=reason,
            decided_at="2026-09-29 12:47 UTC",
        )
        rendered = render_decision(decision)
        assert "NOT PROMOTED" in rendered
        assert "**FAIL**" in rendered
        assert "lift over base rate" in rendered, "PR-AUC must never appear without its base rate"
        assert decision.failed_checks == ("pr_auc",)

    def test_the_report_states_the_window_is_not_a_test_set(self):
        """§18 A3: two gate runs have read it, so calling it a test set would be a lie."""
        report = (ROOT / "reports" / "promotions.md").read_text(encoding="utf-8")
        assert "promotion evaluation set, not a test set" in report
