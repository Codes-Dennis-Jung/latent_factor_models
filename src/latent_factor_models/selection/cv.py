# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dennis Jung
"""Model-agnostic cross-validation and grid search for time-series panels.

Splitters yield ``(train_rows, test_rows)`` index arrays over the time
dimension.  Data containers (:class:`FactorData`,
:class:`CharacteristicPanel`) are sliced with ``take``, so targets that
depend on the data (alpha target, macro target) are rebuilt on each
training window and there is no look-ahead.

Every (parameter combination, fold) pair is an independent task executed
through :class:`~latent_factor_models.engine.parallel.Executor`.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.base import LatentFactorModel
from ..engine.parallel import Backend, Executor, as_executor

Split = tuple[np.ndarray, np.ndarray]
Scorer = Callable[[LatentFactorModel, Any], float]


# ---------------------------------------------------------------- splitters
class KFold:
    """K folds over time.

    Parameters
    ----------
    n_splits : int
        Number of folds.
    shuffle : bool
        Random folds (Giglio, Xiu and Zhang, Appendix E) instead of
        contiguous blocks (Lettau-Pelger, target PCA).
    n_repeats : int
        Repeat with independent shuffles (only with ``shuffle=True``).
    seed : int, optional
        Seed for shuffling.
    """

    def __init__(
        self, n_splits: int = 5, shuffle: bool = False, n_repeats: int = 1, seed: int | None = None
    ) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        self.n_splits, self.shuffle, self.n_repeats, self.seed = n_splits, shuffle, n_repeats, seed

    def split(self, T: int) -> Iterator[Split]:
        """Yield train/test row indices."""
        rng = np.random.default_rng(self.seed)
        reps = self.n_repeats if self.shuffle else 1
        for _ in range(reps):
            order = rng.permutation(T) if self.shuffle else np.arange(T)
            for fold in np.array_split(order, self.n_splits):
                test = np.sort(fold)
                train = np.setdiff1d(np.arange(T), test)
                yield train, test


class TimeSeriesSplit:
    """Expanding or rolling train window followed by a test block.

    Parameters
    ----------
    n_splits : int
        Number of test blocks (equally spaced at the end of the sample).
    test_size : int, optional
        Length of each test block (default ``T // (n_splits + 1)``).
    min_train : int, optional
        Minimum training length (default: first block).
    gap : int
        Periods dropped between train and test (purging).
    rolling : int, optional
        Use a rolling window of this length instead of an expanding one.
    """

    def __init__(
        self,
        n_splits: int = 5,
        test_size: int | None = None,
        min_train: int | None = None,
        gap: int = 0,
        rolling: int | None = None,
    ) -> None:
        self.n_splits, self.test_size, self.min_train = n_splits, test_size, min_train
        self.gap, self.rolling = gap, rolling

    def split(self, T: int) -> Iterator[Split]:
        """Yield train/test row indices."""
        test_size = self.test_size or T // (self.n_splits + 1)
        first_test = T - self.n_splits * test_size
        min_train = self.min_train or first_test - self.gap
        if min_train < 2 or first_test - self.gap < min_train:
            raise ValueError("not enough observations for the requested splits")
        for k in range(self.n_splits):
            start = first_test + k * test_size
            end = min(start + test_size, T)
            train_end = start - self.gap
            train_start = 0 if self.rolling is None else max(0, train_end - self.rolling)
            yield np.arange(train_start, train_end), np.arange(start, end)


# ------------------------------------------------------------------ scorers
def _sharpe(model: Any, test: Any) -> float:
    return float(model.evaluate(test.returns)["sharpe_ratio"])


def _neg_rms_alpha(model: Any, test: Any) -> float:
    return -float(model.evaluate(test.returns)["rms_alpha"])


def _neg_unexplained(model: Any, test: Any) -> float:
    return -float(model.evaluate(test.returns)["unexplained_variation"])


def _total_r2(model: Any, test: Any) -> float:
    if hasattr(test, "characteristics"):
        return float(model.evaluate(test)["total_r2"])
    return float(model.evaluate(test.returns)["total_r2"])


def _hedging_r2(model: Any, test: Any) -> float:
    return float(np.mean(model.hedging_r2(test.returns, test.factors)))


SCORERS: dict[str, Scorer] = {
    "sharpe": _sharpe,
    "neg_rms_alpha": _neg_rms_alpha,
    "neg_unexplained_variation": _neg_unexplained,
    "total_r2": _total_r2,
    "hedging_r2": _hedging_r2,
}


def get_scorer(scoring: str | Scorer) -> Scorer:
    """Resolve a scorer name (see ``SCORERS``) or pass a callable through."""
    if callable(scoring):
        return scoring
    try:
        return SCORERS[scoring]
    except KeyError:
        raise KeyError(f"unknown scoring {scoring!r}; available: {sorted(SCORERS)}") from None


# ---------------------------------------------------------------- execution
def _fit_and_score(task: tuple[LatentFactorModel, dict[str, Any], Any, Split, Scorer]) -> float:
    model, params, data, (train, test), scorer = task
    est = model.clone(**params)
    try:
        est.fit(data.take(train))
        return float(scorer(est, data.take(test)))
    except (np.linalg.LinAlgError, ValueError):
        return float("nan")


def _splits(cv: Any, T: int) -> list[Split]:
    if isinstance(cv, int):
        cv = KFold(cv)
    if hasattr(cv, "split"):
        return list(cv.split(T))
    return list(cv)


def cross_validate(
    model: LatentFactorModel,
    data: Any,
    *,
    cv: Any = 5,
    scoring: str | Scorer = "sharpe",
    executor: Executor | Backend | None = None,
    n_jobs: int | None = None,
) -> np.ndarray:
    """Out-of-fold scores of ``model`` on ``data`` (one per split)."""
    data = model._coerce(data)
    scorer = get_scorer(scoring)
    tasks: list[tuple[LatentFactorModel, dict[str, Any], Any, Split, Scorer]] = [
        (model, {}, data, s, scorer) for s in _splits(cv, data.T)
    ]
    return np.asarray(as_executor(executor, n_jobs).map(_fit_and_score, tasks))


@dataclass
class GridSearchResult:
    """All evaluated parameter combinations with their fold scores."""

    params: list[dict[str, Any]]
    scores: np.ndarray  # (n_combinations, n_splits)

    @property
    def mean_scores(self) -> np.ndarray:
        """Average score per combination (NaN folds ignored)."""
        with np.errstate(all="ignore"):
            return np.asarray(np.nanmean(self.scores, axis=1))

    def to_records(self) -> list[dict[str, Any]]:
        """Rows ``{**params, mean_score, std_score}`` (e.g. for ``pandas.DataFrame``)."""
        return [
            {**p, "mean_score": float(np.nanmean(s)), "std_score": float(np.nanstd(s))}
            for p, s in zip(self.params, self.scores, strict=True)
        ]


class GridSearchCV:
    """Exhaustive search over a parameter grid with time-series cross-validation.

    Parameters
    ----------
    model : LatentFactorModel
        Template estimator (cloned for every task).
    param_grid : mapping of str to sequence
        Candidate values, e.g. ``{"gamma": [0, 5, 10, 20], "n_factors": [3, 5]}``.
    cv : int, splitter or list of splits
        Integer means contiguous :class:`KFold`.
    scoring : str or callable
        Higher is better; see :data:`SCORERS`.
    refit : bool
        Refit the best combination on the full data.
    executor, n_jobs :
        Parallel backend (see :class:`~latent_factor_models.engine.parallel.Executor`).

    Examples
    --------
    >>> from latent_factor_models import RPPCA, simulate_factor_returns
    >>> sim = simulate_factor_returns(T=240, N=40, seed=0)
    >>> gs = GridSearchCV(RPPCA(3), {"gamma": [-1, 10]}, cv=3).fit(sim.returns)
    >>> sorted(gs.best_params_)
    ['gamma']
    """

    def __init__(
        self,
        model: LatentFactorModel,
        param_grid: Mapping[str, Sequence[Any]],
        *,
        cv: Any = 5,
        scoring: str | Scorer = "sharpe",
        refit: bool = True,
        executor: Executor | Backend | None = None,
        n_jobs: int | None = None,
    ) -> None:
        self.model, self.param_grid, self.cv, self.scoring = model, dict(param_grid), cv, scoring
        self.refit, self.executor, self.n_jobs = refit, executor, n_jobs

    def fit(self, data: Any, **kwargs: Any) -> GridSearchCV:
        """Run the search; ``kwargs`` are forwarded to the data container."""
        data = self.model._coerce(data, **kwargs)
        keys = list(self.param_grid)
        combos = [
            dict(zip(keys, vals, strict=True))
            for vals in itertools.product(*self.param_grid.values())
        ]
        splits = _splits(self.cv, data.T)
        scorer = get_scorer(self.scoring)
        tasks = [(self.model, c, data, s, scorer) for c in combos for s in splits]
        flat = as_executor(self.executor, self.n_jobs).map(_fit_and_score, tasks)
        scores = np.asarray(flat, dtype=float).reshape(len(combos), len(splits))
        self.result_ = GridSearchResult(combos, scores)
        means = self.result_.mean_scores
        if np.all(np.isnan(means)):
            raise RuntimeError("all grid points failed")
        best = int(np.nanargmax(means))
        self.best_params_ = combos[best]
        self.best_score_ = float(means[best])
        if self.refit:
            self.best_model_ = self.model.clone(**self.best_params_).fit(data)
        return self
