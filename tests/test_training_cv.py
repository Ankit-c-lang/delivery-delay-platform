"""Cross-validation ordering tests (PLAN.md §13 Phase 4, §6.3, §6.6).

The §6.6 checklist line is "`TimeSeriesSplit`, never `KFold`", and the phase's required test
is that **every fold's maximum training date is before its minimum validation date**.

Why this is worth its own file: `TimeSeriesSplit` splits by *position*, not by timestamp. It
has no idea what the dates are. Hand it rows in the wrong order and it produces folds that
look perfectly ordered, leak badly, and raise nothing. The ordering property everyone assumes
is really a property of *sorted input plus* the splitter, so both halves are tested here.
"""

from __future__ import annotations

import ast
import itertools
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.training.tune import assert_time_sorted, make_cv


@pytest.fixture
def ordered_frame() -> pd.DataFrame:
    """200 rows spanning 200 days, strictly ascending."""
    return pd.DataFrame(
        {
            "order_purchase_timestamp": pd.Timestamp("2017-05-01")
            + pd.to_timedelta(np.arange(200), unit="D"),
            "feature": np.arange(200, dtype=float),
        }
    )


class TestFoldOrdering:
    """The phase's required assertion."""

    def test_every_folds_max_training_date_precedes_its_min_validation_date(self, ordered_frame):
        stamps = ordered_frame["order_purchase_timestamp"]
        folds = list(make_cv(4).split(ordered_frame))
        assert len(folds) == 4
        for index, (train_idx, valid_idx) in enumerate(folds):
            train_max = stamps.iloc[train_idx].max()
            valid_min = stamps.iloc[valid_idx].min()
            assert train_max < valid_min, f"fold {index}: {train_max} !< {valid_min}"

    def test_training_and_validation_indices_never_overlap(self, ordered_frame):
        for train_idx, valid_idx in make_cv(4).split(ordered_frame):
            assert not (set(train_idx) & set(valid_idx))

    def test_training_windows_expand(self, ordered_frame):
        """Expanding, not sliding: each fold trains on everything before its validation
        block, which is what a retraining cadence actually does."""
        sizes = [len(train_idx) for train_idx, _ in make_cv(4).split(ordered_frame)]
        assert sizes == sorted(sizes)
        assert len(set(sizes)) == len(sizes)

    def test_validation_blocks_are_consecutive_and_disjoint(self, ordered_frame):
        blocks = [valid_idx for _, valid_idx in make_cv(4).split(ordered_frame)]
        for earlier, later in itertools.pairwise(blocks):
            assert earlier.max() < later.min()

    def test_no_fold_trains_on_a_row_that_comes_after_its_validation_block(self, ordered_frame):
        """The leakage statement in its strongest form: not just the boundary, every row."""
        for train_idx, valid_idx in make_cv(4).split(ordered_frame):
            assert train_idx.max() < valid_idx.min()

    def test_the_split_count_is_honoured(self, ordered_frame):
        for n_splits in (2, 3, 5):
            assert len(list(make_cv(n_splits).split(ordered_frame))) == n_splits


class TestTimeSortGuard:
    """Sorted input is half the guarantee, so it is asserted rather than assumed."""

    def test_sorted_input_passes(self, ordered_frame):
        assert_time_sorted(ordered_frame["order_purchase_timestamp"])

    def test_shuffled_input_is_rejected(self, ordered_frame):
        shuffled = ordered_frame.sample(frac=1.0, random_state=0)
        with pytest.raises(ValueError, match="not sorted by purchase time"):
            assert_time_sorted(shuffled["order_purchase_timestamp"])

    def test_a_single_row_out_of_place_is_rejected(self, ordered_frame):
        broken = ordered_frame.copy()
        broken.loc[50, "order_purchase_timestamp"] = pd.Timestamp("2099-01-01")
        with pytest.raises(ValueError, match="not sorted"):
            assert_time_sorted(broken["order_purchase_timestamp"])

    def test_ties_are_allowed(self):
        """Two orders can share a timestamp; only a decrease is a problem."""
        stamps = pd.Series(pd.to_datetime(["2017-05-01", "2017-05-01", "2017-05-02"]))
        assert_time_sorted(stamps)

    def test_the_guard_would_have_caught_silent_leakage(self, ordered_frame):
        """Demonstrates *why* the guard exists: on shuffled rows the splitter still reports
        perfectly ordered index ranges while the dates inside them are interleaved."""
        shuffled = ordered_frame.sample(frac=1.0, random_state=0).reset_index(drop=True)
        stamps = shuffled["order_purchase_timestamp"]
        violations = 0
        for train_idx, valid_idx in make_cv(4).split(shuffled):
            assert train_idx.max() < valid_idx.min()  # positions still look fine
            if stamps.iloc[train_idx].max() >= stamps.iloc[valid_idx].min():
                violations += 1
        assert violations > 0, "shuffling should have broken the date ordering"


class TestNoRandomFoldsAnywhere:
    """§6.6 by inspection: `KFold` and `StratifiedKFold` must not appear in ``src/``."""

    def test_no_module_imports_a_random_splitter(self):
        forbidden = {"KFold", "StratifiedKFold", "ShuffleSplit", "StratifiedShuffleSplit"}
        offenders: list[str] = []
        for path in sorted(Path("src").rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        if alias.name in forbidden:
                            offenders.append(f"{path}:{node.lineno} imports {alias.name}")
                elif isinstance(node, ast.Name) and node.id in forbidden:
                    offenders.append(f"{path}:{node.lineno} references {node.id}")
        assert offenders == [], f"random cross-validation found: {offenders}"

    def test_train_test_split_is_not_used_either(self):
        """`train_test_split` shuffles by default, which is the same mistake wearing a
        different name."""
        offenders = [
            str(path)
            for path in sorted(Path("src").rglob("*.py"))
            if re.search(r"\btrain_test_split\b", path.read_text(encoding="utf-8"))
        ]
        assert offenders == []
