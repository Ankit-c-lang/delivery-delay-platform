"""Time splits — the single reader of ``configs/splits.yaml`` (PLAN.md §4.4, §18 A2, A3).

No date boundary is written anywhere else in this project. Split boundaries duplicated across
modules is how evaluation-set contamination happens, and it happens quietly.

**Windows are per version.** The first draft of the plan had v2 training on its own validation
window, which would have made its blend weights, calibration and cost-optimal threshold all
in-sample (§18 A2). :func:`version_splits` therefore returns a fit window and a calibrate
window that are proven disjoint before they are handed out.

**The 2018-05 → 08 window is locked.** §4.4 asks for "a runtime guard that raises if it's
loaded during training", and this module provides one: :func:`promotion_evaluation_window`
raises :class:`PromotionEvaluationLockedError` unless the caller has entered
:func:`unlock_promotion_evaluation` and named itself. Only the promotion gate may do that
(via ``src/evaluation/score_holdout.py`` in Phase 9). The parity test uses
``tests/fixtures/``, never this window (§18 A3).

The guard deliberately protects the *window definition*, not merely the loading code, because
a stray ``orders[orders.purchase >= "2018-05-01"]`` in a training notebook is exactly the
mistake that would otherwise go unnoticed until the promotion numbers looked too good.

Because the window decides which model gets promoted, it is a **promotion evaluation set**,
not an untouched generalization estimate. Describe it that way (§18 A3).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

SPLITS_PATH = Path("configs/splits.yaml")

#: Modules permitted to unlock the promotion evaluation window. §4.4 names exactly one
#: consumer; the promotion gate itself is listed because it is what calls that consumer.
AUTHORISED_UNLOCKERS: frozenset[str] = frozenset(
    {
        "src.evaluation.score_holdout",
        "src.registry.promote",
    }
)


class PromotionEvaluationLockedError(RuntimeError):
    """Raised when the 2018-05 → 08 window is requested outside the promotion gate."""


_unlocked_by: str | None = None


@dataclass(frozen=True)
class Window:
    """A closed date interval, inclusive at both ends.

    Attributes:
        name: Human-readable label, used in error messages and logs.
        start: First date in the window.
        end: Last date in the window, **inclusive** — as ``configs/splits.yaml`` writes it.
    """

    name: str
    start: date
    end: date

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError(f"{self.name}: start {self.start} is after end {self.end}")

    @property
    def end_exclusive(self) -> date:
        """One day past :attr:`end`.

        Timestamps are compared against this, never against ``end``: an inclusive date bound
        compared to a timestamp silently drops everything after midnight on the last day.
        """
        return self.end + timedelta(days=1)

    def mask(self, timestamps: pd.Series) -> pd.Series:
        """Boolean mask of the timestamps falling inside this window."""
        values = pd.to_datetime(timestamps)
        return (values >= pd.Timestamp(self.start)) & (values < pd.Timestamp(self.end_exclusive))

    def select(self, frame: pd.DataFrame, column: str = "order_purchase_timestamp") -> pd.DataFrame:
        """Rows of ``frame`` whose ``column`` falls inside this window."""
        if column not in frame.columns:
            raise KeyError(f"{column!r} not in frame; cannot apply window {self.name}")
        return frame.loc[self.mask(frame[column])]

    def overlaps(self, other: Window) -> bool:
        """Whether two windows share any day."""
        return self.start <= other.end and other.start <= self.end

    def __str__(self) -> str:
        return f"{self.name} [{self.start} .. {self.end}]"


@dataclass(frozen=True)
class VersionSplits:
    """One model version's windows.

    Attributes:
        version: Key in ``configs/splits.yaml``, e.g. ``v1``.
        fit: Rows the model is fitted on.
        calibrate: Rows used for blend weights, calibration and the decision threshold.
            Disjoint from :attr:`fit`, checked on construction (§18 A2).
        aggregates_through: Last month whose resolved outcomes may enter this version's
            as-of snapshots.
    """

    version: str
    fit: Window
    calibrate: Window
    aggregates_through: date

    def __post_init__(self) -> None:
        if self.fit.overlaps(self.calibrate):
            raise ValueError(
                f"{self.version}: fit and calibrate overlap ({self.fit} vs {self.calibrate}). "
                "Every quantity fitted on the calibration window would be in-sample (§18 A2)."
            )
        if self.calibrate.start <= self.fit.end:
            raise ValueError(
                f"{self.version}: calibrate must start after fit ends, "
                f"got {self.calibrate.start} <= {self.fit.end}"
            )


@lru_cache(maxsize=1)
def _config(path: str = str(SPLITS_PATH)) -> dict:
    """Parse ``configs/splits.yaml`` once."""
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _window(name: str, spec: dict) -> Window:
    return Window(
        name=name,
        start=date.fromisoformat(str(spec["start"])),
        end=date.fromisoformat(str(spec["end"])),
    )


def available_versions() -> tuple[str, ...]:
    """Version keys present in the config, in file order."""
    return tuple(_config()["versions"])


def analysis_window() -> Window:
    """The whole modelling period. Orders outside it are dropped (§4.4)."""
    return _window("analysis", _config()["analysis_window"])


def warmup_window() -> Window:
    """Aggregates only, zero training rows.

    Exists so the earliest training month already has four months of *resolved* delivery
    history behind it (§4.5).
    """
    return _window("warmup", _config()["warmup"])


def version_splits(version: str) -> VersionSplits:
    """Return one version's fit and calibrate windows.

    Args:
        version: Key in ``configs/splits.yaml``.

    Raises:
        KeyError: If the version is not configured.
        ValueError: If its windows overlap, which §18 A2 forbids.
    """
    versions = _config()["versions"]
    if version not in versions:
        raise KeyError(f"unknown version {version!r}; configured: {sorted(versions)}")
    spec = versions[version]
    return VersionSplits(
        version=version,
        fit=_window(f"{version}.fit", spec["fit"]),
        calibrate=_window(f"{version}.calibrate", spec["calibrate"]),
        aggregates_through=date.fromisoformat(str(spec["aggregates_through"])),
    )


@contextmanager
def unlock_promotion_evaluation(caller: str) -> Iterator[None]:
    """Permit :func:`promotion_evaluation_window` for the duration of the block.

    Args:
        caller: The importable module name doing the unlocking. Must be in
            :data:`AUTHORISED_UNLOCKERS`.

    Raises:
        PromotionEvaluationLockedError: If ``caller`` is not authorised.

    Nesting is not supported on purpose: an unlock that can be layered is an unlock whose
    extent is hard to reason about, and the whole value here is that the extent is one block
    in one module.
    """
    global _unlocked_by
    if caller not in AUTHORISED_UNLOCKERS:
        raise PromotionEvaluationLockedError(
            f"{caller!r} may not unlock the promotion evaluation window. "
            f"Authorised: {sorted(AUTHORISED_UNLOCKERS)}. "
            "The parity test uses tests/fixtures/, never this window (§18 A3)."
        )
    if _unlocked_by is not None:
        raise PromotionEvaluationLockedError(
            f"already unlocked by {_unlocked_by!r}; nested unlocking is not supported"
        )
    _unlocked_by = caller
    logger.warning("Promotion evaluation window unlocked by %s", caller)
    try:
        yield
    finally:
        _unlocked_by = None


def promotion_evaluation_window() -> Window:
    """The 2018-05 → 08 window, shared by every version and scored at the very end.

    Raises:
        PromotionEvaluationLockedError: Unless called inside
            :func:`unlock_promotion_evaluation`.

    This is a **promotion evaluation set**, not an untouched generalization estimate: it
    decides which model is promoted, so it has been looked at by the time any number from it
    is reported (§18 A3). Say so when reporting it.
    """
    if _unlocked_by is None:
        raise PromotionEvaluationLockedError(
            "the promotion evaluation window (2018-05 -> 08) is locked. It may be loaded only "
            "by the promotion gate, inside unlock_promotion_evaluation(). If you are writing "
            "a parity or smoke test, use tests/fixtures/ instead (§4.4, §18 A3)."
        )
    return _window("promotion_evaluation", _config()["promotion_evaluation"])


def promotion_evaluation_starts_on() -> date:
    """First date of the promotion evaluation window, **without** unlocking it.

    Knowing where the window begins is not the same as being allowed to read its rows, and
    the §6.6 ordering assertion needs the boundary. Exposed separately so that assertion does
    not have to defeat the guard to do its job.
    """
    return date.fromisoformat(str(_config()["promotion_evaluation"]["start"]))


def assert_windows_ordered(version: str) -> None:
    """Check one version's configured boundaries, without touching any data.

    Raises:
        ValueError: If fit, calibrate and the promotion evaluation window are not strictly
            ordered and disjoint.
    """
    splits = version_splits(version)
    evaluation_start = promotion_evaluation_starts_on()
    if splits.calibrate.end >= evaluation_start:
        raise ValueError(
            f"{version}: calibrate ends {splits.calibrate.end}, on or after the promotion "
            f"evaluation window starts {evaluation_start}"
        )
    if splits.fit.end >= evaluation_start:
        raise ValueError(f"{version}: fit window reaches into the promotion evaluation window")
    if splits.aggregates_through != splits.calibrate.end:
        raise ValueError(
            f"{version}: aggregates_through {splits.aggregates_through} should equal the "
            f"calibrate end {splits.calibrate.end}"
        )


def assert_temporal_ordering(
    orders: pd.DataFrame,
    version: str,
    column: str = "order_purchase_timestamp",
) -> dict[str, pd.Timestamp]:
    """The §6.6 checklist line, asserted against real data.

    Checks ``max(fit.purchase) < min(calibrate.purchase) < promotion_evaluation.start``.

    Args:
        orders: Any frame carrying ``column``.
        version: Version whose windows to check.
        column: Purchase timestamp column.

    Returns:
        The observed boundary timestamps, for logging.

    Raises:
        ValueError: If either window is empty in ``orders``, or the ordering does not hold.
            An empty window is treated as a failure rather than a pass: a vacuous ordering
            check is worse than none, because it reports success.
    """
    assert_windows_ordered(version)
    splits = version_splits(version)

    fit_rows = splits.fit.select(orders, column)
    calibrate_rows = splits.calibrate.select(orders, column)
    if fit_rows.empty:
        raise ValueError(f"{version}: no rows in the fit window {splits.fit}")
    if calibrate_rows.empty:
        raise ValueError(f"{version}: no rows in the calibrate window {splits.calibrate}")

    fit_max = pd.Timestamp(fit_rows[column].max())
    calibrate_min = pd.Timestamp(calibrate_rows[column].min())
    calibrate_max = pd.Timestamp(calibrate_rows[column].max())
    evaluation_start = pd.Timestamp(promotion_evaluation_starts_on())

    if not fit_max < calibrate_min:
        raise ValueError(
            f"{version}: max fit purchase {fit_max} is not before min calibrate purchase "
            f"{calibrate_min}"
        )
    if not calibrate_max < evaluation_start:
        raise ValueError(
            f"{version}: max calibrate purchase {calibrate_max} reaches the promotion "
            f"evaluation window, which starts {evaluation_start}"
        )

    logger.info(
        "%s ordering OK: fit<=%s < calibrate>=%s..%s < evaluation>=%s",
        version,
        fit_max,
        calibrate_min,
        calibrate_max,
        evaluation_start,
    )
    return {
        "fit_max": fit_max,
        "calibrate_min": calibrate_min,
        "calibrate_max": calibrate_max,
        "evaluation_start": evaluation_start,
    }
