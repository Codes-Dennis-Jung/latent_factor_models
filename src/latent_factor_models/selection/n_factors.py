# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Dennis Jung
"""Estimators of the number of factors from an eigenvalue spectrum."""

from __future__ import annotations

import numpy as np

from ..core.linalg import FloatArray, as_float_array


def covariance_eigenvalues(returns: FloatArray, scale: str = "NT") -> FloatArray:
    """Descending eigenvalues of the sample covariance of ``returns`` (T x N).

    ``scale="NT"`` returns eigenvalues of ``Rbar'Rbar/(NT)`` (Giglio-Xiu),
    ``scale="T"`` those of ``Rbar'Rbar/T``.
    """
    R = as_float_array(returns, 2, "returns")
    R = R - R.mean(axis=0)
    T, N = R.shape
    s = np.linalg.svd(R, compute_uv=False)
    denom = N * T if scale == "NT" else T
    return np.asarray(s**2 / denom)


def onatski(eigenvalues: FloatArray, r_max: int = 10, max_iter: int = 100) -> int:
    r"""Onatski (2010) edge-distribution estimator of the number of factors.

    Iterates: with ``j = r_hat + 1`` (initially ``r_max + 1``) regress
    :math:`\lambda_j,\dots,\lambda_{j+4}` on a constant and
    :math:`(j-1)^{2/3},\dots,(j+3)^{2/3}`; set :math:`\delta = 2|\hat b|` and
    :math:`\hat r = \max\{i \le r_{max}: \lambda_i - \lambda_{i+1} \ge \delta\}`
    (0 if empty) until :math:`\hat r` converges.

    Parameters
    ----------
    eigenvalues : array-like
        Eigenvalues in descending order (any common scale); at least
        ``r_max + 5`` are needed.
    r_max : int
        Upper bound on the number of factors.  Must exceed the true number;
        it is *not* the number of factors of a particular model.
    """
    lam = np.sort(np.asarray(eigenvalues, dtype=float))[::-1]
    if len(lam) < r_max + 5:
        raise ValueError(f"need at least r_max + 5 = {r_max + 5} eigenvalues, got {len(lam)}")
    diffs = lam[:r_max] - lam[1 : r_max + 1]
    j = r_max + 1
    r_hat = -1
    for _ in range(max_iter):
        y = lam[j - 1 : j + 4]
        x = np.arange(j - 1, j + 4, dtype=float) ** (2.0 / 3.0)
        slope = np.polyfit(x, y, 1)[0]
        delta = 2.0 * abs(slope)
        above = np.flatnonzero(diffs >= delta)
        new = int(np.max(above)) + 1 if above.size else 0
        if new == r_hat:
            break
        r_hat = new
        j = r_hat + 1
    return r_hat


def giglio_xiu(eigenvalues: FloatArray, N: int, T: int, p_max: int = 20, c: float = 0.5) -> int:
    r"""Giglio and Xiu (2021, Appendix I.1) penalised eigenvalue estimator.

    .. math::
        \hat p = \arg\min_{1\le j\le p_{max}} \big[\lambda_j + j\,\phi(N,T)\big] - 1,
        \quad \phi = c\,(\log N + \log T)(N^{-1/2} + T^{-1/2})\,
        \mathrm{median}(\lambda_1,\dots,\lambda_{p_{max}}).
    """
    lam = np.sort(np.asarray(eigenvalues, dtype=float))[::-1]
    p_max = min(p_max, len(lam))
    lam = lam[:p_max]
    phi = c * (np.log(N) + np.log(T)) * (N**-0.5 + T**-0.5) * np.median(lam)
    return int(np.argmin(lam + np.arange(1, p_max + 1) * phi))


def bai_ng(returns: FloatArray, k_max: int = 10, criterion: str = "IC2") -> int:
    r"""Bai and Ng (2002) information criteria ``IC1``-``IC3`` (``log V(k) + k g(N,T)``)."""
    R = as_float_array(returns, 2, "returns")
    R = R - R.mean(axis=0)
    T, N = R.shape
    s2 = np.linalg.svd(R, compute_uv=False) ** 2
    total = s2.sum()
    k = np.arange(0, k_max + 1)
    V = np.array([(total - s2[:kk].sum()) / (N * T) for kk in k])
    NT, NpT = N * T, N + T
    C2 = min(N, T)
    pen = {
        "IC1": NpT / NT * np.log(NT / NpT),
        "IC2": NpT / NT * np.log(C2),
        "IC3": np.log(C2) / C2,
    }[criterion]
    return int(np.argmin(np.log(V) + k * pen))


def eigenvalue_ratio(eigenvalues: FloatArray, k_max: int = 10) -> int:
    """Ahn and Horenstein (2013) eigenvalue-ratio estimator ``argmax lambda_k/lambda_{k+1}``."""
    lam = np.sort(np.asarray(eigenvalues, dtype=float))[::-1]
    k_max = min(k_max, len(lam) - 1)
    ratios = lam[:k_max] / lam[1 : k_max + 1]
    return int(np.argmax(ratios) + 1)
