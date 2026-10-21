# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dennis Jung
r"""Supervised PCA for risk premia of weak factors (Giglio, Xiu and Zhang, 2025, JF).

Algorithm (with ``Rbar``, ``rbar``, ``Gbar`` the demeaned returns, mean
returns and demeaned observable factors; for ``k = 1..p``):

1. **Screen**: ``s_i = max_j |corr(R^{(k)}_i, G^{(k)}_j)|``; keep the
   ``N0`` assets with the largest ``s_i`` (set ``I_k``).
2. **PCA** on the selected residual returns: leading singular triplet
   ``(varsigma, sigma, xi)`` of ``R^{(k)}_{I_k}``; latent factor
   ``v^{(k)} = sqrt(T) xi``,
   ``gamma^{(k)} = sqrt(T) varsigma' r^{(k)}_{I_k} / sigma``,
   ``eta^{(k)} = G^{(k)'} xi / sqrt(T)``.
3. **Deflate**: ``R^{(k+1)} = (I - xi xi') R^{(k)}``,
   ``r^{(k+1)} = r^{(k)} - beta^{(k)} gamma^{(k)}`` with
   ``beta^{(k)} = R^{(k)'} xi / sqrt(T)`` and
   ``G^{(k+1)} = (I - xi xi') G^{(k)}``.

Then ``gamma_g = eta gamma`` and, for tradable ``g``,
``alpha = mean(g) - eta gamma``.  The factor-mimicking (hedging) portfolio
is ``W = Gbar' Fbar (Fbar' Fbar)^{-1} B'`` with ``Fbar = Rbar B`` and the
sparse weight matrix ``B`` (column ``k`` equals ``varsigma`` on ``I_k``);
SDF weights are ``b = B (B' Rbar' Rbar B / T)^{-1} B' rbar``.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from ..core.base import PortfolioFactorModel
from ..core.data import FactorData
from ..core.exceptions import DataError
from ..core.linalg import FloatArray, ols, solve_psd
from ..core.registry import register
from ..inference.hac import LagRule, newey_west_lag, two_sided_pvalue
from ..inference.three_pass import RiskPremiaResult, alpha_avar, risk_premia_inference

Screening = Literal["correlation", "covariance"]


def screening_scores(R: FloatArray, G: FloatArray, method: Screening = "correlation") -> FloatArray:
    """``max_j |corr(R_i, G_j)|`` (or absolute covariance) for every asset, shape (N,).

    Assets or factors with zero variance get a score of zero.
    """
    T = R.shape[0]
    cross = np.abs(R.T @ G) / T  # demeaned inputs -> covariances (N, d)
    if method == "correlation":
        sr = np.sqrt(np.mean(R**2, axis=0))
        sg = np.sqrt(np.mean(G**2, axis=0))
        denom = np.outer(sr, sg)
        cross = np.divide(cross, denom, out=np.zeros_like(cross), where=denom > 0)
    elif method != "covariance":
        raise ValueError("screening must be 'correlation' or 'covariance'")
    return np.asarray(cross.max(axis=1))


@register("supervised-pca", "supervised_pca")
class SPCA(PortfolioFactorModel):
    """Supervised PCA (Giglio, Xiu and Zhang).

    Parameters
    ----------
    n_factors : int
        Number of SPCA steps ``p`` (latent factors).
    n_select : int or float
        Assets kept in each screening step: an integer ``N0`` or a share
        ``q`` in (0, 1] (``N0 = floor(q N)``).
    screening : {"correlation", "covariance"}
        Screening statistic; the authors' code uses correlations.
    zero_beta : bool
        Cross-sectionally demean returns first (Section 2.2.6), removing a
        common zero-beta rate.
    lags : int, optional
        Newey-West lag for inference; default from ``lag_rule``.
    lag_rule : {"newey-west", "stock-watson"}
        Rule-of-thumb lag when ``lags`` is None.

    Fitted attributes
    -----------------
    latent_factors_ : ndarray (T, p)
        ``V`` with ``V'V/T = I``.
    eta_ : ndarray (d, p)
    gamma_ : ndarray (p,)
        Risk premia of the latent factors.
    gamma_g_ : ndarray (d,)
        Risk premia of the observable factors.
    alpha_g_ : ndarray (d,)
        Pricing errors of the observable factors (meaningful if tradable).
    selected_ : list of ndarray
        Indices of the assets selected in each step.
    mimicking_weights_ : ndarray (d, N)
        Hedging portfolio weights ``W``.
    sdf_weights_ : ndarray (N,)
        SDF weights ``b``.
    weights_ : ndarray (N, p)
        Sparse factor-portfolio weights (scaled so that step one reproduces ``v``).
    """

    name = "spca"
    requires = ("factors",)

    def __init__(
        self,
        n_factors: int = 1,
        n_select: int | float = 0.5,
        screening: Screening = "correlation",
        zero_beta: bool = False,
        lags: int | None = None,
        lag_rule: LagRule = "newey-west",
    ) -> None:
        self.n_factors = n_factors
        self.n_select = n_select
        self.screening = screening
        self.zero_beta = zero_beta
        self.lags = lags
        self.lag_rule: LagRule = lag_rule

    def _validate_params(self) -> None:
        super()._validate_params()
        if isinstance(self.n_select, (float, np.floating)) and not 0 < self.n_select <= 1:
            raise ValueError("a fractional n_select must lie in (0, 1]")
        if isinstance(self.n_select, (int, np.integer)) and self.n_select < 1:
            raise ValueError("n_select must be positive")
        if self.screening not in ("correlation", "covariance"):
            raise ValueError("screening must be 'correlation' or 'covariance'")

    def _n0(self, N: int) -> int:
        if isinstance(self.n_select, (float, np.floating)):
            return max(1, int(np.floor(self.n_select * N)))
        return min(int(self.n_select), N)

    def _fit(self, data: FactorData) -> None:
        assert data.factors is not None
        R = data.returns
        if self.zero_beta:
            R = R - R.mean(axis=1, keepdims=True)
        G = data.factors
        T, N = R.shape
        p = self.n_factors
        if p >= T:
            raise DataError("n_factors must be smaller than T")
        n0 = self._n0(N)

        rbar = R.mean(axis=0)
        Rbar = R - rbar
        Gbar = G - G.mean(axis=0)
        Rk, rk, Gk = Rbar.copy(), rbar.copy(), Gbar.copy()
        sqT = np.sqrt(T)
        V = np.zeros((T, p))
        eta = np.zeros((G.shape[1], p))
        gamma = np.zeros(p)
        B = np.zeros((N, p))
        selected: list[np.ndarray] = []
        step_eigenvalues = np.zeros(p)

        for k in range(p):
            scores = screening_scores(Rk, Gk, self.screening)
            idx = np.sort(np.argsort(-scores, kind="stable")[:n0])
            U, s, Vt = np.linalg.svd(Rk[:, idx], full_matrices=False)  # (T, |I|) = xi s varsigma'
            xi, sigma, vs = U[:, 0], s[0], Vt[0]
            if sigma <= 0:
                raise DataError(f"SPCA step {k + 1}: selected returns are identically zero")
            g_k = sqT * (vs @ rk[idx]) / sigma
            if g_k < 0:  # sign convention: positive latent risk premium
                xi, vs, g_k = -xi, -vs, -g_k
            beta_k = Rk.T @ xi / sqT
            V[:, k] = sqT * xi
            gamma[k] = g_k
            eta[:, k] = Gk.T @ xi / sqT
            B[idx, k] = vs * sqT / sigma
            selected.append(idx)
            step_eigenvalues[k] = sigma**2 / T
            rk = rk - beta_k * g_k
            Rk = Rk - np.outer(xi, xi @ Rk)
            Gk = Gk - np.outer(xi, xi @ Gk)

        Fbar = Rbar @ B
        mimic = Gbar.T @ Fbar @ np.linalg.solve(Fbar.T @ Fbar, B.T)
        sdf = B @ solve_psd(Fbar.T @ Fbar / T, B.T @ rbar)

        # In zero-beta mode all portfolio quantities live in the space of
        # cross-sectionally demeaned returns; keep training data consistent.
        self.data_ = FactorData(R, factors=G) if self.zero_beta else data
        self.latent_factors_ = V
        self.eta_ = eta
        self.gamma_ = gamma
        self.gamma_g_ = eta @ gamma
        self.alpha_g_ = G.mean(axis=0) - self.gamma_g_
        self.selected_ = selected
        self.mimicking_weights_ = mimic
        self.sdf_weights_ = sdf
        self.eigenvalues_ = step_eigenvalues
        self.weights_ = B
        self.factors_ = R @ B
        self.loadings_, self.intercepts_ = ols(self.factors_, R, intercept=True)
        resid = Gbar - Rbar @ mimic.T
        self.hedging_r2_ = 1.0 - np.sum(resid**2, axis=0) / np.sum(Gbar**2, axis=0)

    def _as_returns(self, returns: object) -> FloatArray:
        R = super()._as_returns(returns)
        return R - R.mean(axis=1, keepdims=True) if self.zero_beta else R

    # ------------------------------------------------------------------
    def risk_premia(self, n_factors: int | None = None) -> RiskPremiaResult:
        """Risk premia, standard errors, alphas and weak-factor Wald tests.

        Parameters
        ----------
        n_factors : int, optional
            Use only the first ``n_factors`` SPCA steps (nested models need no
            refit); defaults to all.

        Returns
        -------
        RiskPremiaResult
            ``extra`` contains ``alpha``, ``alpha_se``, ``alpha_tstat`` and
            ``alpha_pvalue`` for tradable observable factors.
        """
        self._check_fitted()
        p = self.n_factors if n_factors is None else int(n_factors)
        if not 1 <= p <= self.n_factors:
            raise ValueError(f"n_factors must lie in [1, {self.n_factors}]")
        assert self.data_.factors is not None
        T = self.data_.T
        lags = self.lags if self.lags is not None else newey_west_lag(T, self.lag_rule)
        V, eta, gamma = self.latent_factors_[:, :p], self.eta_[:, :p], self.gamma_[:p]
        Gbar = self.data_.factors - self.data_.factors.mean(axis=0)
        res, pis, Z = risk_premia_inference(V, Gbar, eta, gamma, lags)
        alpha = self.data_.factors.mean(axis=0) - eta @ gamma
        a_se = np.sqrt(np.clip(np.diag(alpha_avar(V, Z, gamma, lags, pis)), 0.0, None))
        a_t = np.divide(alpha, a_se, out=np.full_like(alpha, np.nan), where=a_se > 0)
        res.extra = {
            "alpha": alpha,
            "alpha_se": a_se,
            "alpha_tstat": a_t,
            "alpha_pvalue": two_sided_pvalue(a_t),
        }
        return res

    def hedging_r2(self, returns: FloatArray, factors: FloatArray) -> FloatArray:
        """Out-of-sample R² of the hedging portfolios, per observable factor.

        Both inputs are demeaned within the evaluation sample (Proposition C.5).
        """
        self._check_fitted()
        R = np.asarray(returns, dtype=float)
        if self.zero_beta:
            R = R - R.mean(axis=1, keepdims=True)
        G = np.atleast_2d(np.asarray(factors, dtype=float).T).T
        Rb, Gb = R - R.mean(axis=0), G - G.mean(axis=0)
        resid = Gb - Rb @ self.mimicking_weights_.T
        return np.asarray(1.0 - np.sum(resid**2, axis=0) / np.sum(Gb**2, axis=0))
