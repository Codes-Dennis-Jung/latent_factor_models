# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dennis Jung
r"""Asset-pricing evaluation metrics.

The definitions follow Lettau and Pelger (2020, RFS) and Bryzgalova,
DeMiguel, Li and Pelger (2026):

* **Maximum Sharpe ratio** of the factors' mean-variance (tangency)
  portfolio, :math:`b = \Sigma_F^{-1}\mu_F` estimated on a training sample.
  Out of sample it is the realised Sharpe ratio of :math:`F_{t} b`.
* **Pricing errors** :math:`\alpha_i = \overline{e}_i` with
  :math:`e_t = R_t - \beta F_t` and :math:`\beta` from the training sample;
  :math:`\text{RMS}_\alpha = \sqrt{\alpha'\alpha/N}`.  In sample, with
  :math:`\beta` from a time-series OLS with intercept, this is exactly the
  OLS intercept.
* **Unexplained variation**
  :math:`\bar\sigma^2_e = \frac1N\sum_i \sigma^2(e_i)/\sigma^2(R_i)`.
* **XS-metric** :math:`100\,\lVert P_\Lambda \omega\rVert/\lVert\omega\rVert`
  and **TS-metric**
  :math:`100\,\lVert P_G R P_\omega\rVert_F/\lVert P_G R\rVert_F`.

All functions take ``T x N`` returns and ``T x K`` factors.
"""

from __future__ import annotations

import numpy as np

from ..core.linalg import FloatArray, orthonormal_basis, solve_psd


def tangency_weights(F: FloatArray) -> FloatArray:
    r"""Mean-variance weights :math:`b = \Sigma_F^{-1}\mu_F` (sample covariance, ddof=1)."""
    F = np.atleast_2d(F.T).T
    mu = F.mean(axis=0)
    sigma = np.atleast_2d(np.cov(F, rowvar=False))
    return np.asarray(solve_psd(sigma, mu))


def sharpe_ratio(x: FloatArray) -> float:
    """Per-period Sharpe ratio ``mean / std`` (ddof=1) of a return series."""
    x = np.asarray(x, dtype=float)
    sd = x.std(ddof=1)
    return float(x.mean() / sd) if sd > 0 else float("nan")


def max_sharpe_ratio(F: FloatArray) -> float:
    r"""In-sample maximum Sharpe ratio :math:`\sqrt{\mu'\Sigma^{-1}\mu}` of the factors."""
    b = tangency_weights(F)
    return sharpe_ratio(np.atleast_2d(F.T).T @ b)


def pricing_errors(
    R: FloatArray, F: FloatArray, loadings: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Residuals ``e = R - F loadings'`` and pricing errors ``alpha = mean(e)``."""
    e = R - F @ loadings.T
    return e, e.mean(axis=0)


def rms_alpha(alpha: FloatArray) -> float:
    """Root-mean-squared pricing error."""
    return float(np.sqrt(np.mean(np.square(alpha))))


def unexplained_variation(R: FloatArray, residuals: FloatArray) -> float:
    r"""Average idiosyncratic variance share :math:`\frac1N\sum_i\sigma^2(e_i)/\sigma^2(R_i)`."""
    var_r = R.var(axis=0, ddof=1)
    var_e = residuals.var(axis=0, ddof=1)
    ok = var_r > 0
    return float(np.mean(var_e[ok] / var_r[ok]))


def total_r2(R: FloatArray, fitted: FloatArray) -> float:
    """Uncentred panel R² ``1 - sum (R - fitted)^2 / sum R^2`` (Kelly, Pruitt and Su, 2019)."""
    return float(1.0 - np.sum((R - fitted) ** 2) / np.sum(R**2))


def xs_metric(weights: FloatArray, xs_target: FloatArray) -> FloatArray:
    r"""XS-metric per column of ``weights``.

    :math:`100\lVert P_\Lambda\omega\rVert/\lVert\omega\rVert`.
    """
    W = np.atleast_2d(weights.T).T
    Q = orthonormal_basis(xs_target)
    num = np.linalg.norm(Q.T @ W, axis=0)
    den = np.linalg.norm(W, axis=0)
    return np.asarray(100.0 * np.divide(num, den, out=np.zeros_like(den), where=den > 0))


def ts_metric(R: FloatArray, weights: FloatArray, ts_target: FloatArray) -> float:
    r"""TS-metric, computed without ``T x T`` matrices.

    :math:`100\lVert P_G R P_\omega\rVert_F/\lVert P_G R\rVert_F`.
    """
    QG = orthonormal_basis(ts_target)
    Qw = orthonormal_basis(np.atleast_2d(weights.T).T)
    A = QG.T @ R  # (J, N); ||P_G R||_F = ||Q_G' R||_F
    den = np.linalg.norm(A)
    if den == 0:
        return 0.0
    return float(100.0 * np.linalg.norm((A @ Qw) @ Qw.T) / den)


def generalized_correlation(F_hat: FloatArray, F_true: FloatArray) -> FloatArray:
    r"""Canonical correlations between estimated and true factors (Lettau-Pelger, fn. 15).

    Square roots of the eigenvalues of
    :math:`\Sigma_{\hat F}^{-1}\Sigma_{\hat F F}\Sigma_F^{-1}\Sigma_{F\hat F}`
    in descending order; invariant to rotations of either factor set.
    """
    A = np.atleast_2d(F_hat.T).T
    B = np.atleast_2d(F_true.T).T
    Qa = np.linalg.qr(A - A.mean(axis=0))[0]
    Qb = np.linalg.qr(B - B.mean(axis=0))[0]
    return np.asarray(np.clip(np.linalg.svd(Qa.T @ Qb, compute_uv=False), 0.0, 1.0))


def evaluate_factors(
    R: FloatArray,
    F: FloatArray,
    loadings: FloatArray,
    sdf_weights: FloatArray,
) -> dict[str, float]:
    """Core metrics given realised factors, loadings and SDF (tangency) weights.

    Returns
    -------
    dict
        ``sharpe_ratio`` (per period), ``rms_alpha``, ``unexplained_variation``
        and ``total_r2`` (of ``F loadings'``).
    """
    e, alpha = pricing_errors(R, F, loadings)
    return {
        "sharpe_ratio": sharpe_ratio(F @ sdf_weights),
        "rms_alpha": rms_alpha(alpha),
        "unexplained_variation": unexplained_variation(R, e),
        "total_r2": total_r2(R, F @ loadings.T),
    }


def evaluate_portfolio_factors(
    R: FloatArray,
    weights: FloatArray,
    *,
    R_train: FloatArray,
    loadings: FloatArray,
) -> dict[str, float]:
    """Evaluate portfolio factors ``F = R @ weights`` with parameters fixed on ``R_train``.

    The tangency weights are estimated from the training factors
    ``R_train @ weights``, loadings are taken as given; passing
    ``R = R_train`` yields in-sample metrics.
    """
    b = tangency_weights(R_train @ weights)
    return evaluate_factors(R, R @ weights, loadings, b)
