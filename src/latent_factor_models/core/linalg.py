# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dennis Jung
"""Numerically careful linear-algebra building blocks shared by all estimators.

All routines avoid explicit matrix inverses and never form ``T x T`` matrices.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as sla
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

#: Relative tolerance used to decide the numerical rank of a matrix.
RANK_RTOL = 1e-10


def as_float_array(x: ArrayLike, ndim: int | None = None, name: str = "array") -> FloatArray:
    """Convert ``x`` to a C-contiguous float64 array, optionally checking ``ndim``.

    One-dimensional input is promoted to a column when ``ndim == 2``.
    """
    arr = np.asarray(x, dtype=np.float64)
    if ndim == 2 and arr.ndim == 1:
        arr = arr[:, None]
    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional, got shape {arr.shape}")
    return np.ascontiguousarray(arr)


def symmetrize(M: FloatArray) -> FloatArray:
    """Return ``(M + M') / 2`` to remove floating-point asymmetry."""
    return 0.5 * (M + M.T)


def orthonormal_basis(A: FloatArray, rtol: float = RANK_RTOL) -> FloatArray:
    """Orthonormal basis ``Q`` of the column space of ``A`` (rank-revealing).

    The projector onto ``col(A)`` is ``Q @ Q.T``; it equals ``A (A'A)^+ A'``
    but is computed via an SVD, so collinear columns are handled gracefully.
    """
    U, s, _ = np.linalg.svd(A, full_matrices=False)
    if s.size == 0 or s[0] == 0.0:
        return np.zeros((A.shape[0], 0))
    rank = int(np.sum(s > rtol * s[0]))
    return U[:, :rank]


def project(Q: FloatArray, X: FloatArray) -> FloatArray:
    """Apply the projector ``Q Q'`` (``Q`` orthonormal) to ``X`` along axis 0."""
    return Q @ (Q.T @ X)


def top_eigh(M: FloatArray, k: int) -> tuple[FloatArray, FloatArray]:
    """Largest ``k`` eigenpairs of a symmetric matrix, sorted in descending order.

    Returns
    -------
    eigenvalues : ndarray of shape (k,)
    eigenvectors : ndarray of shape (n, k), orthonormal columns
    """
    n = M.shape[0]
    if not 0 < k <= n:
        raise ValueError(f"k must lie in [1, {n}], got {k}")
    vals, vecs = sla.eigh(symmetrize(M), subset_by_index=[n - k, n - 1])
    return vals[::-1].copy(), vecs[:, ::-1].copy()


def all_eigvalsh(M: FloatArray) -> FloatArray:
    """All eigenvalues of a symmetric matrix in descending order."""
    return np.linalg.eigvalsh(symmetrize(M))[::-1].copy()


def sqrtm_psd(M: FloatArray) -> FloatArray:
    """Symmetric square root of a symmetric positive semi-definite matrix."""
    vals, vecs = np.linalg.eigh(symmetrize(M))
    return (vecs * np.sqrt(np.clip(vals, 0.0, None))) @ vecs.T


def solve_psd(A: FloatArray, B: FloatArray) -> FloatArray:
    """Solve ``A X = B`` for symmetric positive (semi-)definite ``A``.

    Uses a Cholesky factorisation and falls back to a least-squares solution
    when ``A`` is singular.
    """
    try:
        c, low = sla.cho_factor(A, check_finite=False)
        return np.asarray(sla.cho_solve((c, low), B, check_finite=False))
    except np.linalg.LinAlgError:
        return np.asarray(np.linalg.lstsq(A, B, rcond=None)[0])


def ols(X: FloatArray, Y: FloatArray, intercept: bool = False) -> tuple[FloatArray, FloatArray]:
    """Least-squares regression of every column of ``Y`` on ``X``.

    Parameters
    ----------
    X : ndarray of shape (n, p)
    Y : ndarray of shape (n, m)
    intercept : bool
        Add a constant regressor.

    Returns
    -------
    coef : ndarray of shape (m, p)
        Slope coefficients, one row per column of ``Y``.
    const : ndarray of shape (m,)
        Intercepts (zeros when ``intercept`` is False).
    """
    n = X.shape[0]
    design = np.column_stack([np.ones(n), X]) if intercept else X
    theta = np.linalg.lstsq(design, Y, rcond=None)[0]
    if intercept:
        return theta[1:].T.copy(), theta[0].copy()
    return theta.T.copy(), np.zeros(Y.shape[1])


def sign_by_mean(F: FloatArray) -> FloatArray:
    """Signs (+1/-1) that make each column of ``F`` have a non-negative mean."""
    s = np.sign(F.mean(axis=0))
    s[s == 0] = 1.0
    return s
