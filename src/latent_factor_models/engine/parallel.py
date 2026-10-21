# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dennis Jung
"""Backend-agnostic parallel map with reproducible random streams.

The :class:`Executor` is the single place where the package decides how work
is distributed.  Everything that is embarrassingly parallel (cross-validation
grids, bootstrap draws, rolling windows, Monte-Carlo replications, multi-model
runs) goes through :meth:`Executor.map`, so switching from serial execution to
threads or processes is a one-argument change.

Notes
-----
* ``"process"`` uses :class:`concurrent.futures.ProcessPoolExecutor`.  On
  Windows and macOS (``spawn`` start method) the mapped function and its
  arguments must be picklable, i.e. defined at module level, and scripts must
  be protected by ``if __name__ == "__main__":``.
* NumPy's BLAS is already multi-threaded.  When running many small linear
  algebra problems in processes, limit BLAS threads (e.g. set the environment
  variable ``OMP_NUM_THREADS=1`` before starting Python) to avoid
  oversubscription.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import Executor as _FuturesExecutor
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Literal, TypeVar

import numpy as np

T = TypeVar("T")
R = TypeVar("R")

Backend = Literal["serial", "thread", "process"]
_BACKENDS: tuple[str, ...] = ("serial", "thread", "process")


def resolve_n_jobs(n_jobs: int | None) -> int:
    """Translate ``n_jobs`` (``None``, positive, or negative à la joblib) to a worker count."""
    cpus = os.cpu_count() or 1
    if n_jobs is None or n_jobs == 0:
        return 1
    if n_jobs < 0:
        return max(1, cpus + 1 + n_jobs)
    return int(n_jobs)


class Executor:
    """Ordered parallel ``map`` over a configurable backend.

    Parameters
    ----------
    backend : {"serial", "thread", "process"}
        Execution backend.  ``"serial"`` is used automatically when the
        resolved number of workers is one.
    n_jobs : int, optional
        Number of workers; ``-1`` uses all cores, ``-2`` all but one, etc.

    Examples
    --------
    >>> ex = Executor("thread", n_jobs=2)
    >>> ex.map(abs, [-1, -2, 3])
    [1, 2, 3]
    """

    def __init__(self, backend: Backend = "serial", n_jobs: int | None = None) -> None:
        if backend not in _BACKENDS:
            raise ValueError(f"backend must be one of {_BACKENDS}, got {backend!r}")
        self.backend: Backend = backend
        self.n_jobs = resolve_n_jobs(n_jobs if backend != "serial" else 1)

    def __repr__(self) -> str:
        return f"Executor(backend={self.backend!r}, n_jobs={self.n_jobs})"

    def _pool(self) -> _FuturesExecutor:
        if self.backend == "thread":
            return ThreadPoolExecutor(max_workers=self.n_jobs)
        return ProcessPoolExecutor(max_workers=self.n_jobs)

    def map(self, fn: Callable[[T], R], items: Iterable[T]) -> list[R]:
        """Apply ``fn`` to every item and return the results in input order."""
        seq: Sequence[T] = list(items)
        if self.backend == "serial" or self.n_jobs == 1 or len(seq) <= 1:
            return [fn(x) for x in seq]
        chunksize = max(1, len(seq) // (4 * self.n_jobs)) if self.backend == "process" else 1
        with self._pool() as pool:
            return list(pool.map(fn, seq, chunksize=chunksize))

    def starmap(self, fn: Callable[..., R], items: Iterable[tuple[object, ...]]) -> list[R]:
        """Like :meth:`map` but unpacks each item as positional arguments."""
        star: Callable[[tuple[object, ...]], R] = _Star(fn)  # type: ignore[assignment]
        return self.map(star, items)


class _Star:
    """Picklable helper turning ``fn(*args)`` into a one-argument callable."""

    def __init__(self, fn: Callable[..., object]) -> None:
        self.fn = fn

    def __call__(self, args: tuple[object, ...]) -> object:
        return self.fn(*args)


def as_executor(executor: Executor | Backend | None = None, n_jobs: int | None = None) -> Executor:
    """Coerce ``None``, a backend name or an :class:`Executor` into an :class:`Executor`."""
    if isinstance(executor, Executor):
        return executor
    if executor is None:
        return Executor("serial") if n_jobs in (None, 1) else Executor("process", n_jobs)
    return Executor(executor, n_jobs)


def spawn_seeds(seed: int | np.random.SeedSequence | None, n: int) -> list[np.random.SeedSequence]:
    """Spawn ``n`` statistically independent child seeds from ``seed``.

    Each parallel task should build its own generator via
    ``np.random.default_rng(child)``; results are then reproducible and
    independent of the backend and of the number of workers.
    """
    root = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    return root.spawn(n)
