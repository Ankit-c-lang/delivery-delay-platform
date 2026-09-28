"""Tests for the time splits and the promotion-evaluation lock (PLAN.md §13 Phase 3 part C).

What each group defends:

1. **The lock.** §4.4 asks for "a runtime guard that raises if [the 2018-05 → 08 window] is
   loaded during training". A stray ``orders[orders.purchase >= "2018-05-01"]`` in a training
   notebook is the mistake that would otherwise surface only as promotion numbers that looked
   too good.
2. **Per-version disjointness.** The first draft had v2 training on its own validation window
   (§18 A2). ``VersionSplits`` refuses to exist in that state.
3. **The §6.6 ordering line**, against real data, with an empty window treated as a failure
   rather than a vacuous pass.
4. **One reader.** §18 A3 wants the window loaded by exactly one module, checked by
   inspection.
"""

from __future__ import annotations

import ast
import re
from datetime import date
from pathlib import Path
from typing import ClassVar

import pandas as pd
import pytest

from src.splits import (
    AUTHORISED_UNLOCKERS,
    PromotionEvaluationLockedError,
    VersionSplits,
    Window,
    analysis_window,
    assert_temporal_ordering,
    assert_windows_ordered,
    available_versions,
    promotion_evaluation_starts_on,
    promotion_evaluation_window,
    unlock_promotion_evaluation,
    version_splits,
    warmup_window,
)

GATE = "src.evaluation.score_holdout"


class TestWindow:
    def test_end_is_inclusive_and_end_exclusive_is_the_next_day(self):
        window = Window("w", date(2018, 1, 1), date(2018, 2, 28))
        assert window.end_exclusive == date(2018, 3, 1)

    def test_mask_includes_the_last_day_right_up_to_midnight(self):
        """The bug this prevents: comparing a timestamp against an inclusive *date* bound
        silently drops everything after midnight on the final day."""
        window = Window("w", date(2018, 1, 1), date(2018, 1, 31))
        stamps = pd.to_datetime(
            [
                "2017-12-31 23:59:59",
                "2018-01-01 00:00:00",
                "2018-01-31 23:59:59",
                "2018-02-01 00:00:00",
            ]
        )
        assert window.mask(pd.Series(stamps)).tolist() == [False, True, True, False]

    def test_select_filters_a_frame(self):
        window = Window("w", date(2018, 1, 1), date(2018, 1, 31))
        frame = pd.DataFrame(
            {
                "order_id": ["a", "b", "c"],
                "order_purchase_timestamp": pd.to_datetime(
                    ["2017-12-15", "2018-01-15", "2018-02-15"]
                ),
            }
        )
        assert window.select(frame)["order_id"].tolist() == ["b"]

    def test_select_raises_on_a_missing_column_rather_than_returning_everything(self):
        window = Window("w", date(2018, 1, 1), date(2018, 1, 31))
        with pytest.raises(KeyError, match="order_purchase_timestamp"):
            window.select(pd.DataFrame({"x": [1]}))

    def test_a_backwards_window_is_refused(self):
        with pytest.raises(ValueError, match="after end"):
            Window("w", date(2018, 2, 1), date(2018, 1, 1))

    def test_overlaps_is_inclusive_at_the_touching_day(self):
        a = Window("a", date(2018, 1, 1), date(2018, 1, 31))
        assert a.overlaps(Window("b", date(2018, 1, 31), date(2018, 2, 5)))
        assert not a.overlaps(Window("c", date(2018, 2, 1), date(2018, 2, 5)))

    def test_single_day_window_is_valid(self):
        window = Window("w", date(2018, 1, 1), date(2018, 1, 1))
        assert window.end_exclusive == date(2018, 1, 2)


class TestVersionSplits:
    """§18 A2: fit and calibrate must be disjoint for every version."""

    def test_overlapping_fit_and_calibrate_is_refused(self):
        with pytest.raises(ValueError, match="fit and calibrate overlap"):
            VersionSplits(
                version="bad",
                fit=Window("f", date(2017, 5, 1), date(2018, 4, 30)),
                calibrate=Window("c", date(2018, 3, 1), date(2018, 4, 30)),
                aggregates_through=date(2018, 4, 30),
            )

    def test_calibrate_before_fit_is_refused(self):
        with pytest.raises(ValueError, match=r"fit and calibrate overlap|must start after"):
            VersionSplits(
                version="bad",
                fit=Window("f", date(2018, 1, 1), date(2018, 2, 28)),
                calibrate=Window("c", date(2017, 5, 1), date(2017, 12, 31)),
                aggregates_through=date(2017, 12, 31),
            )

    def test_a_valid_pair_is_accepted(self):
        splits = VersionSplits(
            version="ok",
            fit=Window("f", date(2017, 5, 1), date(2017, 12, 31)),
            calibrate=Window("c", date(2018, 1, 1), date(2018, 2, 28)),
            aggregates_through=date(2018, 2, 28),
        )
        assert splits.version == "ok"


class TestConfiguredWindows:
    """The values actually in configs/splits.yaml (§4.4 as amended by §18 A2)."""

    def test_both_versions_are_configured(self):
        assert available_versions() == ("v1", "v2")

    def test_v1_windows(self):
        splits = version_splits("v1")
        assert (splits.fit.start, splits.fit.end) == (date(2017, 5, 1), date(2017, 12, 31))
        assert (splits.calibrate.start, splits.calibrate.end) == (
            date(2018, 1, 1),
            date(2018, 2, 28),
        )
        assert splits.aggregates_through == date(2018, 2, 28)

    def test_v2_trains_on_v1s_fit_plus_v1s_calibrate(self):
        """§18 A2's resolution: v2 absorbs v1's validation window into its fit window and
        calibrates on a later, disjoint one."""
        v1, v2 = version_splits("v1"), version_splits("v2")
        assert v2.fit.start == v1.fit.start
        assert v2.fit.end == v1.calibrate.end
        assert not v2.fit.overlaps(v2.calibrate)

    def test_v2_sees_more_data_than_v1(self):
        v1, v2 = version_splits("v1"), version_splits("v2")
        assert (v2.fit.end - v2.fit.start) > (v1.fit.end - v1.fit.start)

    def test_both_versions_pass_the_boundary_check(self):
        for version in available_versions():
            assert_windows_ordered(version)

    def test_an_unknown_version_raises_and_names_the_options(self):
        with pytest.raises(KeyError, match="v1"):
            version_splits("v99")

    def test_warmup_precedes_every_fit_window(self):
        warmup = warmup_window()
        for version in available_versions():
            assert warmup.end < version_splits(version).fit.start

    def test_analysis_window_contains_everything(self):
        analysis = analysis_window()
        assert analysis.start <= warmup_window().start
        assert analysis.end >= promotion_evaluation_starts_on()
        for version in available_versions():
            splits = version_splits(version)
            assert analysis.start <= splits.fit.start
            assert splits.calibrate.end <= analysis.end


class TestThePromotionEvaluationLock:
    """§4.4's runtime guard, and §18 A3."""

    def test_locked_by_default(self):
        with pytest.raises(PromotionEvaluationLockedError, match="locked"):
            promotion_evaluation_window()

    def test_the_error_points_at_the_fixtures_instead(self):
        """A guard that blocks without saying what to do instead gets worked around."""
        with pytest.raises(PromotionEvaluationLockedError, match="tests/fixtures"):
            promotion_evaluation_window()

    def test_an_authorised_caller_can_unlock(self):
        with unlock_promotion_evaluation(GATE):
            window = promotion_evaluation_window()
        assert (window.start, window.end) == (date(2018, 5, 1), date(2018, 8, 31))

    def test_it_relocks_when_the_block_ends(self):
        with unlock_promotion_evaluation(GATE):
            promotion_evaluation_window()
        with pytest.raises(PromotionEvaluationLockedError):
            promotion_evaluation_window()

    def test_it_relocks_even_when_the_block_raises(self):
        """Otherwise one failed promotion run leaves the window open for the rest of the
        process, and the next training run in the same process would silently be allowed."""
        with pytest.raises(RuntimeError, match="boom"), unlock_promotion_evaluation(GATE):
            raise RuntimeError("boom")
        with pytest.raises(PromotionEvaluationLockedError):
            promotion_evaluation_window()

    @pytest.mark.parametrize(
        "caller",
        ["src.training.train", "src.features.build", "tests.test_parity", "__main__", ""],
    )
    def test_unauthorised_callers_are_refused(self, caller):
        with (
            pytest.raises(PromotionEvaluationLockedError, match="may not unlock"),
            unlock_promotion_evaluation(caller),
        ):
            pass

    def test_nested_unlocking_is_refused(self):
        with (
            unlock_promotion_evaluation(GATE),
            pytest.raises(PromotionEvaluationLockedError, match="nested"),
            unlock_promotion_evaluation("src.registry.promote"),
        ):
            pass

    def test_a_failed_unlock_does_not_leave_the_window_open(self):
        with (
            pytest.raises(PromotionEvaluationLockedError),
            unlock_promotion_evaluation("src.training.train"),
        ):
            pass
        with pytest.raises(PromotionEvaluationLockedError):
            promotion_evaluation_window()

    def test_only_the_gate_and_its_caller_are_authorised(self):
        assert {
            "src.evaluation.score_holdout",
            "src.registry.promote",
        } == AUTHORISED_UNLOCKERS

    def test_the_start_date_is_readable_without_unlocking(self):
        """Knowing where the window begins is not the same as reading its rows, and the §6.6
        ordering assertion needs the boundary. If this required unlocking, the assertion
        would have to defeat the guard to do its job."""
        assert promotion_evaluation_starts_on() == date(2018, 5, 1)


class TestOnlyOneModuleReadsTheWindow:
    """§18 A3 by inspection: the window has exactly one reader in ``src/``."""

    def test_no_unauthorised_module_calls_promotion_evaluation_window(self):
        offenders: list[str] = []
        for path in sorted(Path("src").rglob("*.py")):
            module = ".".join(path.with_suffix("").parts)
            if module in AUTHORISED_UNLOCKERS or module == "src.splits":
                continue
            source = path.read_text(encoding="utf-8")
            if re.search(r"\bpromotion_evaluation_window\b", source):
                offenders.append(module)
        assert offenders == [], f"modules reaching for the evaluation window: {offenders}"

    def test_no_module_hardcodes_the_evaluation_boundary(self):
        """Every date lives in configs/splits.yaml. A literal ``2018-05-01`` in *code* is a
        second copy of a boundary, which is how contamination starts.

        Parsed with :mod:`ast` rather than scanned line by line, so prose in a docstring that
        happens to quote a date — including this module's own description of the mistake to
        avoid — is not mistaken for code. A line-based version flagged exactly that and was
        wrong to.
        """
        pattern = re.compile(r"2018-0[5-8]-\d\d")
        offenders: list[str] = []
        for path in sorted(Path("src").rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            docstrings = {
                id(node.body[0].value)
                for node in ast.walk(tree)
                if isinstance(
                    node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
                )
                and node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            }
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and id(node) not in docstrings
                    and pattern.search(node.value)
                ):
                    offenders.append(f"{path}:{node.lineno} {node.value[:60]!r}")
        assert offenders == [], f"hardcoded evaluation dates in code: {offenders}"


class TestTemporalOrdering:
    """The §6.6 checklist line."""

    def test_an_empty_fit_window_is_a_failure_not_a_pass(self):
        """A vacuous ordering check is worse than none, because it reports success."""
        empty = pd.DataFrame({"order_purchase_timestamp": pd.to_datetime(["2016-01-01"])})
        with pytest.raises(ValueError, match="no rows in the fit window"):
            assert_temporal_ordering(empty, "v1")

    def test_an_empty_calibrate_window_is_a_failure(self):
        fit_only = pd.DataFrame(
            {"order_purchase_timestamp": pd.to_datetime(["2017-06-01", "2017-07-01"])}
        )
        with pytest.raises(ValueError, match="no rows in the calibrate window"):
            assert_temporal_ordering(fit_only, "v1")

    def test_ordering_holds_on_a_synthetic_frame(self):
        frame = pd.DataFrame(
            {
                # Consistent full timestamps: pandas 3 infers one format from the first
                # element, so mixing "2017-06-01" with "2017-12-31 23:00" raises.
                "order_purchase_timestamp": pd.to_datetime(
                    [
                        "2017-06-01 10:00:00",
                        "2017-12-31 23:00:00",
                        "2018-01-02 08:00:00",
                        "2018-02-28 23:00:00",
                    ]
                )
            }
        )
        observed = assert_temporal_ordering(frame, "v1")
        assert observed["fit_max"] < observed["calibrate_min"]
        assert observed["calibrate_max"] < observed["evaluation_start"]


class TestAgainstTheRealData:
    pytestmark: ClassVar = [pytest.mark.integration, pytest.mark.slow]

    @pytest.fixture(scope="class")
    @staticmethod
    def purchases(engine) -> pd.DataFrame:
        from sqlalchemy import text

        with engine.connect() as conn:
            return pd.read_sql(
                text(
                    "select order_id, order_purchase_timestamp " "from features.orders_analytical"
                ),
                conn,
            )

    def test_ordering_holds_for_every_version(self, purchases):
        for version in available_versions():
            observed = assert_temporal_ordering(purchases, version)
            assert observed["fit_max"] < observed["calibrate_min"]
            assert observed["calibrate_max"] < observed["evaluation_start"]

    def test_window_row_counts_are_the_measured_ones(self, purchases):
        v1, v2 = version_splits("v1"), version_splits("v2")
        assert len(v1.fit.select(purchases)) == 36_174
        assert len(v1.calibrate.select(purchases)) == 13_624
        assert len(v2.fit.select(purchases)) == 49_798
        assert len(v2.calibrate.select(purchases)) == 13_801

    def test_no_order_appears_in_both_fit_and_calibrate(self, purchases):
        for version in available_versions():
            splits = version_splits(version)
            fit_ids = set(splits.fit.select(purchases)["order_id"])
            calibrate_ids = set(splits.calibrate.select(purchases)["order_id"])
            assert not (fit_ids & calibrate_ids), version

    def test_the_evaluation_window_holds_the_measured_rows(self, purchases):
        """Reads the window, so it unlocks as the gate would — the only test here that does."""
        with unlock_promotion_evaluation("src.registry.promote"):
            window = promotion_evaluation_window()
        assert len(window.select(purchases)) == 25_352

    def test_no_training_row_reaches_the_evaluation_window(self, purchases):
        evaluation_start = pd.Timestamp(promotion_evaluation_starts_on())
        for version in available_versions():
            splits = version_splits(version)
            for window in (splits.fit, splits.calibrate):
                assert window.select(purchases)["order_purchase_timestamp"].max() < evaluation_start
