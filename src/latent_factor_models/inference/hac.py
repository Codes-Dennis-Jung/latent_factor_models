# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Dennis Jung
"""Heteroskedasticity and autocorrelation consistent (HAC) covariance estimation."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy import stats

from ..core.linalg import FloatArray, as_float_array

LagRule = Literal["newey-west", "stock-watson"]


def newey_west_lag(T: int, rule: LagRule = "newey-west") -> int:
    """Rule-of-thumb truncation lag for Bartlett-kernel HAC estimators.

    Parameters
    ----------
    T : int
        Sample size.
    rule : {"newey-west", "stock-watson"}
        * ``"newey-west"``: ``floor(4 (T/100)^{2/9})`` (Newey and West, 1994).
        * ``"stock-watson"``: ``round(0.75 T^{1/3})`` (Stock and Watson, 2007).

    Returns
    -------
    int
        Lag in ``[1, T - 1]``.

    Examples
    --------
    >>> newey_west_lag(100), newey_west_lag(600)
    (4, 5)
    """
    if not isinstance(T, (int, np.integer)) or T < 2:
        raise ValueError(f"T must be an integer >= 2, got {T!r}")
    if rule == "newey-west":
        lag = int(np.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))
    elif rule == "stock-watson":
        lag = round(0.75 * T ** (1.0 / 3.0))
    else:
        raise ValueError(f"unknown lag rule {rule!r}")
    return max(1, min(lag, T - 1))


def long_run_covariance(A: FloatArray, lags: int, demean: bool = False) -> FloatArray:
    r"""Bartlett-kernel (Newey-West) long-run covariance of a vector series.

    .. math::
        \hat\Omega = \hat\Gamma_0 + \sum_{s=1}^{q} \Big(1 - \frac{s}{q+1}\Big)
        (\hat\Gamma_s + \hat\Gamma_s'), \qquad
        \hat\Gamma_s = \frac1T \sum_{t=s+1}^{T} a_t a_{t-s}'.

    Parameters
    ----------
    A : ndarray of shape (T, m)
        Observations ``a_t`` in rows.  They are *not* demeaned unless
        ``demean`` is True (moment conditions are mean zero by construction).
    lags : int
        Truncation lag ``q >= 0``.
    demean : bool
        Subtract the sample mean first.

    Returns
    -------
    ndarray of shape (m, m)
    """
    A = as_float_array(A, 2, "A")
    if demean:
        A = A - A.mean(axis=0)
    T = A.shape[0]
    omega = A.T @ A / T
    for s in range(1, min(lags, T - 1) + 1):
        gamma_s = A[s:].T @ A[:-s] / T
        omega += (1.0 - s / (lags + 1.0)) * (gamma_s + gamma_s.T)
    return 0.5 * (omega + omega.T)


def two_sided_pvalue(tstat: FloatArray, df: int | None = None) -> FloatArray:
    """Two-sided p-values from t-statistics (normal if ``df`` is None)."""
    t = np.abs(np.asarray(tstat, dtype=float))
    if df is None:
        return np.asarray(2.0 * stats.norm.sf(t))
    return np.asarray(2.0 * stats.t.sf(t, df))
