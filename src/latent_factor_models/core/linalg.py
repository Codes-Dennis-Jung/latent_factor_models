# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Dennis Jung
"""Numerically careful, fast linear-algebra building blocks shared by all estimators.

All routines avoid explicit matrix inverses.  Leading eigenpairs of Gram
matrices are computed either with an implicitly restarted Lanczos method
(ARPACK) on the implicit operator ``v -> X'(X v)`` - which never forms a Gram
matrix and costs ``O(T N)`` per iteration - or with LAPACK's subset
eigensolver, whichever is cheaper.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import scipy.linalg as sla
import scipy.sparse.linalg as spla
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

#: Relative tolerance used to decide the numerical rank of a matrix.
RANK_RTOL = 1e-10

#: Use Lanczos when ``k <= min(T, N) / LANCZOS_RATIO`` and the problem is not tiny.
LANCZOS_RATIO = 5
LANCZOS_MIN_DIM = 60

EigenMethod = Literal["auto", "dense", "lanczos"]


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
    # LAPACK's subset driver reads one triangle only, so no symmetrising copy is needed.
    vals, vecs = sla.eigh(M, subset_by_index=[n - k, n - 1], driver="evr", check_finite=False)
    return vals[::-1].copy(), vecs[:, ::-1].copy()


def _lanczos_gram(
    X: FloatArray, k: int, side: Literal["right", "left"]
) -> tuple[FloatArray, FloatArray]:
    """Top-``k`` eigenpairs of ``X'X`` (side="right") or ``X X'`` (side="left") by ARPACK."""
    n = X.shape[1] if side == "right" else X.shape[0]
    if side == "right":
        op = spla.LinearOperator((n, n), matvec=lambda v: X.T @ (X @ v), dtype=np.float64)
    else:
        op = spla.LinearOperator((n, n), matvec=lambda v: X @ (X.T @ v), dtype=np.float64)
    v0 = np.random.default_rng(0).standard_normal(n)  # fixed start -> reproducible output
    vals, vecs = spla.eigsh(op, k=k, which="LA", tol=0.0, v0=v0)
    order = np.argsort(vals)[::-1]
    return vals[order], vecs[:, order]


def top_gram_eigh(
    X: FloatArray, k: int, method: EigenMethod = "auto"
) -> tuple[FloatArray, FloatArray]:
    """Largest ``k`` eigenpairs of ``X'X`` without forming ``X'X`` when avoidable.

    Works on the smaller side of ``X`` (shape ``(T, N)``): if ``N > T`` the
    eigenvectors of ``X X'`` are mapped back with ``X' u / sqrt(lambda)``.

    Parameters
    ----------
    X : ndarray of shape (T, N)
    k : int
        Number of eigenpairs.
    method : {"auto", "dense", "lanczos"}
        ``"lanczos"``: ARPACK on the implicit operator (``O(T N)`` per
        iteration, machine-precision tolerance); ``"dense"``: LAPACK subset
        solver on the explicit Gram matrix (``O(T N min(T, N))``); ``"auto"``
        uses Lanczos when ``k`` is small relative to ``min(T, N)`` and falls
        back to the dense solver if ARPACK does not converge.

    Returns
    -------
    eigenvalues : ndarray of shape (k,)
        Descending eigenvalues of ``X'X``.
    eigenvectors : ndarray of shape (N, k)
        Orthonormal eigenvectors of ``X'X``.
    """
    T, N = X.shape
    small = min(T, N)
    if not 0 < k <= small:
        raise ValueError(f"k must lie in [1, {small}], got {k}")
    if method not in ("auto", "dense", "lanczos"):
        raise ValueError(f"unknown eigen method {method!r}")
    side: Literal["right", "left"] = "right" if N <= T else "left"
    use_lanczos = method == "lanczos" or (
        method == "auto" and small >= LANCZOS_MIN_DIM and k <= small // LANCZOS_RATIO
    )
    vals: FloatArray
    vecs: FloatArray
    if use_lanczos and k < small:
        try:
            vals, vecs = _lanczos_gram(X, k, side)
        except spla.ArpackNoConvergence:
            use_lanczos = False
    else:
        use_lanczos = False
    if not use_lanczos:
        vals, vecs = top_eigh(X.T @ X if side == "right" else X @ X.T, k)
    if side == "right":
        return vals, vecs
    norms = np.sqrt(np.clip(vals, np.finfo(float).tiny, None))
    return vals, (X.T @ vecs) / norms


def top_singular_triplet(A: FloatArray) -> tuple[FloatArray, float, FloatArray]:
    """Leading singular triplet ``(u, s, v)`` of ``A`` with ``A v = s u`` (Lanczos/LAPACK)."""
    vals, V = top_gram_eigh(A, 1)
    v = V[:, 0]
    s = float(np.sqrt(max(vals[0], 0.0)))
    u = A @ v / s if s > 0 else np.zeros(A.shape[0])
    return u, s, v


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


def batched_matvec(A: FloatArray, x: FloatArray) -> FloatArray:
    """``A[t] @ x[t]`` for all ``t``; shapes ``(T, m, n)`` and ``(T, n)``, BLAS ``matmul``."""
    return np.asarray(np.matmul(A, x[..., None])[..., 0])


def sign_by_mean(F: FloatArray) -> FloatArray:
    """Signs (+1/-1) that make each column of ``F`` have a non-negative mean."""
    s = np.sign(F.mean(axis=0))
    s[s == 0] = 1.0
    return s
