# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dennis Jung
r"""Risk premia of observable factors with omitted factors (Giglio and Xiu, 2021).

Notation (time in rows)
-----------------------
``V`` (``T x p``) demeaned latent factors, ``Gbar`` (``T x d``) demeaned
observable factors, ``eta`` (``d x p``) from the time-series regression
``g_t = eta v_t + z_t``, ``gamma`` (``p``) latent risk premia,
``gamma_g = eta gamma``.

The Newey-West matrices :math:`\Pi_{11},\Pi_{12},\Pi_{22},\Pi_{13},\Pi_{33}`
are blocks of one long-run covariance of the stacked moment vector
:math:`[\operatorname{vec}(z_t v_t'),\, v_t,\, z_t]`, computed in a single
vectorised pass.

.. math::
    \Phi = K\Pi_{11}K' + K\Pi_{12}\eta' + \eta\Pi_{12}'K' + \eta\Pi_{22}\eta',
    \qquad K = \gamma'\Sigma_v^{-1}\otimes I_d,

    \operatorname{Avar}(\hat\gamma_g) = \Phi/T,\qquad
    \operatorname{Avar}(\hat\alpha) = (K\Pi_{11}K' - K\Pi_{13} - \Pi_{13}'K' + \Pi_{33})/T.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats

from ..core.base import PortfolioFactorModel
from ..core.data import FactorData
from ..core.exceptions import DataError
from ..core.linalg import FloatArray, ols, solve_psd
from .hac import LagRule, long_run_covariance, newey_west_lag, two_sided_pvalue


@dataclass
class PiMatrices:
    """Newey-West blocks of the stacked moments ``[vec(z v'), v, z]``."""

    pi11: FloatArray
    pi12: FloatArray
    pi22: FloatArray
    pi13: FloatArray
    pi33: FloatArray


def pi_matrices(V: FloatArray, Z: FloatArray, lags: int) -> PiMatrices:
    """Compute the Newey-West Pi matrices.

    Parameters
    ----------
    V : ndarray of shape (T, p)
        Demeaned latent factors.
    Z : ndarray of shape (T, d)
        Residuals of the observable factors on ``V``.
    lags : int
        Bartlett truncation lag.
    """
    T, p = V.shape
    d = Z.shape[1]
    # vec(z_t v_t') in column-major order = kron(v_t, z_t)
    A = (V[:, :, None] * Z[:, None, :]).reshape(T, p * d)
    omega = long_run_covariance(np.hstack([A, V, Z]), lags)
    dp = d * p
    return PiMatrices(
        pi11=omega[:dp, :dp],
        pi12=omega[:dp, dp : dp + p],
        pi22=omega[dp : dp + p, dp : dp + p],
        pi13=omega[:dp, dp + p :],
        pi33=omega[dp + p :, dp + p :],
    )


def _k_matrix(gamma: FloatArray, sigma_v: FloatArray, d: int) -> FloatArray:
    row = solve_psd(sigma_v, gamma)  # Sigma_v^{-1} gamma  (p,)
    return np.asarray(np.kron(row[None, :], np.eye(d)), dtype=float)


def gamma_g_avar(
    V: FloatArray,
    Z: FloatArray,
    eta: FloatArray,
    gamma: FloatArray,
    lags: int,
    pis: PiMatrices | None = None,
) -> FloatArray:
    """Asymptotic covariance (divided by ``T``) of ``gamma_g = eta gamma``, shape (d, d)."""
    T = V.shape[0]
    pis = pis or pi_matrices(V, Z, lags)
    Kmat = _k_matrix(gamma, V.T @ V / T, Z.shape[1])
    cross = Kmat @ pis.pi12 @ eta.T
    phi = Kmat @ pis.pi11 @ Kmat.T + cross + cross.T + eta @ pis.pi22 @ eta.T
    return np.asarray(0.5 * (phi + phi.T) / T)


def alpha_avar(
    V: FloatArray, Z: FloatArray, gamma: FloatArray, lags: int, pis: PiMatrices | None = None
) -> FloatArray:
    """Asymptotic covariance of ``alpha = mean(g) - eta gamma`` for tradable ``g``, shape (d, d)."""
    T = V.shape[0]
    pis = pis or pi_matrices(V, Z, lags)
    Kmat = _k_matrix(gamma, V.T @ V / T, Z.shape[1])
    cross = Kmat @ pis.pi13
    phi = Kmat @ pis.pi11 @ Kmat.T - cross - cross.T + pis.pi33
    return np.asarray(0.5 * (phi + phi.T) / T)


def weak_factor_wald(
    V: FloatArray, eta: FloatArray, pis: PiMatrices
) -> tuple[FloatArray, FloatArray]:
    r"""Wald test of ``H0: eta_l = 0`` (observable factor ``l`` is weak/useless), GX Theorem 4.

    .. math:: W_l = T\,\eta_l(\Sigma_v^{-1}\Pi_{11}^{(l)}\Sigma_v^{-1})^{-1}\eta_l' \sim \chi^2_p

    Returns
    -------
    stat, pvalue : ndarray of shape (d,)
    """
    T, p = V.shape
    d = eta.shape[0]
    sigma_inv = np.linalg.inv(V.T @ V / T)
    stat = np.empty(d)
    for ell in range(d):
        idx = ell + d * np.arange(p)
        middle = sigma_inv @ pis.pi11[np.ix_(idx, idx)] @ sigma_inv
        stat[ell] = T * eta[ell] @ solve_psd(middle, eta[ell])
    return stat, np.asarray(stats.chi2.sf(stat, p))


@dataclass
class RiskPremiaResult:
    """Output of :class:`ThreePassRiskPremia` and of SPCA inference."""

    gamma_g: FloatArray
    se: FloatArray
    tstat: FloatArray
    pvalue: FloatArray
    eta: FloatArray
    gamma: FloatArray
    r2_g: FloatArray
    wald_stat: FloatArray
    wald_pvalue: FloatArray
    lags: int
    gamma_0: float | None = None
    gamma_0_se: float | None = None
    r2_cs: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def summary(self, names: list[str] | None = None) -> str:
        """Plain-text table of the risk-premium estimates."""
        names = names or [f"g{i + 1}" for i in range(len(self.gamma_g))]
        lines = [f"{'factor':<12}{'gamma_g':>12}{'se':>12}{'t':>9}{'p':>9}{'R2_g':>8}{'Wald p':>9}"]
        for i, n in enumerate(names):
            lines.append(
                f"{n!s:<12}{self.gamma_g[i]:>12.5f}{self.se[i]:>12.5f}"
                f"{self.tstat[i]:>9.2f}{self.pvalue[i]:>9.3f}{self.r2_g[i]:>8.3f}"
                f"{self.wald_pvalue[i]:>9.3f}"
            )
        return "\n".join(lines)


def risk_premia_inference(
    V: FloatArray, Gbar: FloatArray, eta: FloatArray, gamma: FloatArray, lags: int
) -> tuple[RiskPremiaResult, PiMatrices, FloatArray]:
    """Shared inference given latent factors, ``eta`` and ``gamma`` (used by 3-pass and SPCA)."""
    Z = Gbar - V @ eta.T
    pis = pi_matrices(V, Z, lags)
    gamma_g = eta @ gamma
    avar = gamma_g_avar(V, Z, eta, gamma, lags, pis)
    se = np.sqrt(np.clip(np.diag(avar), 0.0, None))
    tstat = np.divide(gamma_g, se, out=np.full_like(gamma_g, np.nan), where=se > 0)
    wald, wald_p = weak_factor_wald(V, eta, pis)
    fitted = V @ eta.T
    r2_g = np.sum(fitted**2, axis=0) / np.sum(Gbar**2, axis=0)
    res = RiskPremiaResult(
        gamma_g=gamma_g,
        se=se,
        tstat=tstat,
        pvalue=two_sided_pvalue(tstat),
        eta=eta,
        gamma=gamma,
        r2_g=r2_g,
        wald_stat=wald,
        wald_pvalue=wald_p,
        lags=lags,
    )
    return res, pis, Z


class ThreePassRiskPremia:
    """Giglio-Xiu (2021) three-pass estimator with any latent factor model as first pass.

    1. Estimate ``p`` latent factors ``v_t`` from the test-asset returns (PCA
       in GX; any :class:`PortfolioFactorModel`, e.g. RP-PCA, here) and the
       time-series betas ``beta`` of the returns on them.
    2. Cross-sectional regression of average returns on ``beta`` (optionally
       with a zero-beta rate) gives latent risk premia ``gamma``.
    3. Time-series regression of the observable factors on ``v_t`` gives
       ``eta``; the risk premia are ``gamma_g = eta gamma``.

    The estimates are invariant to rotations of the latent factors, so the
    normalisation of the first-pass model does not matter.

    Parameters
    ----------
    model : PortfolioFactorModel
        First-pass latent factor model (its ``n_factors`` is ``p``).
    zero_beta : bool
        Include an intercept (zero-beta rate) in the cross-sectional step.
    lags : int, optional
        Newey-West lag; default from ``lag_rule``.
    lag_rule : {"newey-west", "stock-watson"}
        Rule-of-thumb lag when ``lags`` is None.
    """

    def __init__(
        self,
        model: PortfolioFactorModel,
        zero_beta: bool = False,
        lags: int | None = None,
        lag_rule: LagRule = "newey-west",
    ) -> None:
        self.model = model
        self.zero_beta = zero_beta
        self.lags = lags
        self.lag_rule: LagRule = lag_rule

    def fit(self, returns: Any, factors: Any) -> RiskPremiaResult:
        """Estimate risk premia of ``factors`` (``T x d``) using test assets ``returns``."""
        data = FactorData.coerce(returns, factors=factors)
        if data.factors is None:
            raise DataError("observable factors are required")
        R, G = data.returns, data.factors
        T, N = R.shape
        self.model.fit(data)
        F = self.model.factors_
        V = F - F.mean(axis=0)
        beta = ols(V, R, intercept=True)[0]  # (N, p) time-series betas
        rbar = R.mean(axis=0)
        lags = self.lags if self.lags is not None else newey_west_lag(T, self.lag_rule)

        if self.zero_beta:
            coef, gamma0 = ols(beta, rbar[:, None], intercept=True)
            gamma = coef[0]
            alpha_cs = rbar - gamma0[0] - beta @ gamma
        else:
            gamma = np.linalg.lstsq(beta, rbar, rcond=None)[0]
            alpha_cs = rbar - beta @ gamma
        Gbar = G - G.mean(axis=0)
        eta = np.linalg.lstsq(V, Gbar, rcond=None)[0].T  # (d, p)
        res, _pis, _Z = risk_premia_inference(V, Gbar, eta, gamma, lags)

        rbar_c = rbar - rbar.mean()
        beta_c = beta - beta.mean(axis=0)
        fit_cs = beta_c @ np.linalg.lstsq(beta_c, rbar_c, rcond=None)[0]
        res.r2_cs = float(fit_cs @ fit_cs / (rbar_c @ rbar_c))
        res.extra = {"beta": beta, "alpha_cs": alpha_cs, "latent_factors": V}

        if self.zero_beta:
            sigma2_alpha = float(np.mean(alpha_cs**2))
            b0 = beta.mean(axis=0)
            sigma_b = beta.T @ beta / N
            res.gamma_0 = float(gamma0[0])
            res.gamma_0_se = float(np.sqrt(sigma2_alpha / N / (1.0 - b0 @ solve_psd(sigma_b, b0))))
            extra = sigma2_alpha * eta @ solve_psd(sigma_b - np.outer(b0, b0), eta.T) / N
            var = res.se**2 + np.diag(extra)
            res.se = np.sqrt(var)
            res.tstat = res.gamma_g / res.se
            res.pvalue = two_sided_pvalue(res.tstat)
        self.result_ = res
        return res
