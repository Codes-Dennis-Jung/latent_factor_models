# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Dennis Jung
r"""Construction of economic targets for target PCA (Bryzgalova et al., 2026).

Time-series targets ``G`` (``T x J``) enter through the projector ``P_G``;
cross-sectional targets ``Lambda`` (``N x L``) through ``P_Lambda``.  Both
projectors are invariant to invertible right-rotations and to rescaling of
the target, so only the spanned space matters.

To avoid look-ahead bias, targets that depend on data (alpha target, macro
target) are rebuilt inside every estimation window by the models; the
functions here are the building blocks.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from ..core.exceptions import DataError
from ..core.linalg import FloatArray, as_float_array

TSKind = Literal["mean", "alpha", "comove"]


def mean_target(T: int) -> FloatArray:
    """Mean target ``G = 1_T``; TS-target PCA with it equals RP-PCA."""
    return np.ones((T, 1))


def alpha_target(reference: FloatArray, publication_lag: int = 0) -> FloatArray:
    r"""Alpha target of eq. (10): :math:`G = (I_T - \tilde F(\tilde F'\tilde F)^{-1}F'^{ref})1_T`.

    :math:`\tilde F` is the demeaned and :math:`F^{ref}` the raw reference
    factor matrix, so that :math:`R'G/T` equals the intercepts of the
    time-series regressions of the assets on the reference factors.

    Parameters
    ----------
    reference : ndarray of shape (T, K_ref)
        Reference factors (e.g. the Fama-French five factors).
    publication_lag : int
        Use only the first ``T - lag`` periods (the reference factors are
        published with a delay); the remaining rows of ``G`` are zero.
    """
    F = as_float_array(reference, 2, "reference")
    T = F.shape[0]
    Ts = T - int(publication_lag)
    if Ts <= F.shape[1] + 1:
        raise DataError("not enough periods to build the alpha target")
    F_ref = F[:Ts]
    F_dm = F_ref - F_ref.mean(axis=0)
    # (F~'F~)^{-1} F^ref' 1_T, where F^ref' 1_T is the column sum of the raw factors
    coef = np.linalg.lstsq(F_dm.T @ F_dm, F_ref.sum(axis=0), rcond=None)[0]
    g = np.ones(Ts) - F_dm @ coef
    G = np.zeros((T, 1))
    G[:Ts, 0] = g * (T / Ts)
    return G


def comove_target(reference: FloatArray, publication_lag: int = 0) -> FloatArray:
    """Co-movement target ``G = F_ref - mean(F_ref)`` (package extension, not in the paper).

    ``P_G`` then up-weights the part of returns that co-moves with the
    reference series.
    """
    F = as_float_array(reference, 2, "reference")
    T = F.shape[0]
    Ts = T - int(publication_lag)
    G = np.zeros_like(F)
    G[:Ts] = F[:Ts] - F[:Ts].mean(axis=0)
    return G


def ts_target(
    kind: TSKind, T: int, reference: FloatArray | None = None, publication_lag: int = 0
) -> FloatArray:
    """Dispatch helper building a time-series target of the given ``kind``."""
    if kind == "mean":
        return mean_target(T)
    if reference is None:
        raise DataError(f"the {kind!r} target needs reference factors (pass factors=...)")
    if kind == "alpha":
        return alpha_target(reference, publication_lag)
    if kind == "comove":
        return comove_target(reference, publication_lag)
    raise ValueError(f"unknown time-series target kind {kind!r}")


def macro_target(returns: FloatArray, macro: FloatArray) -> FloatArray:
    r"""Macro-mimicking loadings :math:`\Lambda^{XS} = R'Y(Y'Y)^{-1}` (eq. 7), shape (N, L)."""
    R = as_float_array(returns, 2, "returns")
    Y = as_float_array(macro, 2, "macro")
    if Y.shape[0] != R.shape[0]:
        raise DataError("macro and returns must have the same number of periods")
    return np.asarray(np.linalg.lstsq(Y, R, rcond=None)[0].T)


def shape_target(groups: FloatArray, n_groups: int = 10) -> FloatArray:
    r"""Characteristic "shape" target (Bryzgalova et al., 2026, footnote 12).

    Parameters
    ----------
    groups : array-like of shape (N, L)
        ``groups[n, l] = i`` if asset ``n`` is the ``i``-th portfolio
        (``1..n_groups``) of a sort on characteristic ``l``; ``0`` or NaN if
        asset ``n`` does not belong to characteristic ``l``.
    n_groups : int
        Number of portfolios per sort (10 for deciles).

    Returns
    -------
    ndarray of shape (N, L)
        ``-1 + 2 (i - 1) / (n_groups - 1)`` for members, ``0`` otherwise; for
        deciles this is ``-1 + (2/9)(i - 1)``.
    """
    g = np.nan_to_num(as_float_array(groups, 2, "groups"), nan=0.0)
    if np.any((g < 0) | (g > n_groups)):
        raise DataError(f"group ids must lie in 0..{n_groups}")
    return np.where(g > 0, -1.0 + 2.0 * (g - 1.0) / (n_groups - 1.0), 0.0)


def var_residuals(Y: FloatArray, lags: int = 1) -> FloatArray:
    """Residuals of a VAR(``lags``) with intercept estimated by OLS, shape (T - lags, k)."""
    Y = as_float_array(Y, 2, "Y")
    T = Y.shape[0]
    if lags * Y.shape[1] + 1 >= T:
        raise DataError("not enough observations for the VAR")
    X = np.column_stack([np.ones(T - lags)] + [Y[lags - j - 1 : T - j - 1] for j in range(lags)])
    target = Y[lags:]
    coef = np.linalg.lstsq(X, target, rcond=None)[0]
    return np.asarray(target - X @ coef)


def macro_innovations(
    panel: FloatArray,
    n_components: int | None = 7,
    var_lags: int = 1,
    standardize: bool = True,
) -> FloatArray:
    """Macro innovations as in Bryzgalova et al. (2026, Section 3.5).

    The (transformed) macro panel is standardised, reduced to its first
    ``n_components`` principal components and passed through a VAR whose
    residuals are the innovations.  The first ``var_lags`` periods are lost:
    align returns with ``returns[var_lags:]``.

    Parameters
    ----------
    panel : ndarray of shape (T, M)
        Stationary macro series (e.g. FRED-MD after McCracken-Ng transforms).
    n_components : int or None
        Number of principal components; ``None`` skips the PCA step.
    var_lags : int
        VAR order.
    standardize : bool
        Standardise series before PCA.
    """
    Y = as_float_array(panel, 2, "panel")
    if standardize:
        sd = Y.std(axis=0, ddof=1)
        sd[sd == 0] = 1.0
        Y = (Y - Y.mean(axis=0)) / sd
    if n_components is not None:
        Yc = Y - Y.mean(axis=0)
        U, s, _ = np.linalg.svd(Yc, full_matrices=False)
        Y = U[:, :n_components] * s[:n_components]
    return var_residuals(Y, var_lags)
