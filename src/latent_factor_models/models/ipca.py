# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Dennis Jung
r"""Instrumented PCA (Kelly, Pruitt and Su, 2019).

Model
-----
.. math::
    r_{i,t} = z_{i,t}'\Gamma_\alpha + z_{i,t}'\Gamma_\beta f_t
              + z_{i,t}'\Gamma_\delta g_t + \varepsilon_{i,t},

with ``L`` instruments :math:`z_{i,t}` (lagged characteristics), ``K``
latent factors :math:`f_t` and ``M`` optional observed factors :math:`g_t`.
Writing :math:`\tilde\Gamma = [\Gamma_\beta, \Gamma_\delta, \Gamma_\alpha]`
and :math:`\tilde f_t = [f_t; g_t; 1]`, the alternating least squares (ALS)
steps on the managed portfolios :math:`X_t = Z_t'r_t/N_t`,
:math:`W_t = Z_t'Z_t/N_t` are

.. math::
    f_t = (\Gamma_\beta' W_t \Gamma_\beta)^{-1}\Gamma_\beta'
          (X_t - W_t[\Gamma_\delta, \Gamma_\alpha][g_t; 1]),

    \operatorname{vec}(\tilde\Gamma') = \Big[\sum_t N_t\, W_t\otimes
    \tilde f_t\tilde f_t'\Big]^{-1} \sum_t N_t\, X_t\otimes\tilde f_t.

Both steps are fully vectorised over time.  After every iteration the
solution is normalised: :math:`\Gamma_\beta'\Gamma_\beta = I_K`,
:math:`\Gamma_\beta'[\Gamma_\delta,\Gamma_\alpha] = 0`, the factors'
second-moment matrix is diagonal with descending entries and factor means
are positive.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ..core.base import LatentFactorModel
from ..core.data import CharacteristicPanel
from ..core.exceptions import ConvergenceWarning, DataError
from ..core.linalg import FloatArray, as_float_array, sign_by_mean
from ..core.registry import register
from ..engine.parallel import Backend, Executor, as_executor, spawn_seeds

Weighting = Literal["observations", "periods"]


# ============================================================================
# ALS core (module-level functions so they are picklable for parallel bootstraps)
# ============================================================================
@dataclass
class ALSResult:
    """Output of :func:`als`."""

    gamma: FloatArray  # (L, K + M') columns [beta | delta | alpha]
    factors: FloatArray  # (T, K)
    n_iter: int
    converged: bool
    tol_reached: float


def _factor_step(
    gamma: FloatArray, X: FloatArray, W: FloatArray, G: FloatArray, K: int
) -> FloatArray:
    gb, gd = gamma[:, :K], gamma[:, K:]
    A = np.einsum("lk,tlm,mj->tkj", gb, W, gb)
    resid = X if gd.shape[1] == 0 else X - np.einsum("tlm,mj,tj->tl", W, gd, G)
    rhs = resid @ gb
    return np.asarray(np.linalg.solve(A, rhs[..., None])[..., 0])


def _gamma_step(
    F: FloatArray, X: FloatArray, W: FloatArray, G: FloatArray, weights: FloatArray
) -> FloatArray:
    Ft = np.column_stack([F, G])
    L, Kt = X.shape[1], Ft.shape[1]
    denom = np.einsum("t,tlm,tk,tj->lkmj", weights, W, Ft, Ft).reshape(L * Kt, L * Kt)
    numer = np.einsum("t,tl,tk->lk", weights, X, Ft).reshape(L * Kt)
    sol = np.linalg.solve(denom, numer)
    return np.asarray(sol.reshape(L, Kt))


def _normalize(
    gamma: FloatArray, F: FloatArray, G: FloatArray, K: int
) -> tuple[FloatArray, FloatArray]:
    gamma, F = gamma.copy(), F.copy()
    gb = gamma[:, :K]
    if gamma.shape[1] > K:
        b = np.linalg.lstsq(gb, gamma[:, K:], rcond=None)[0]  # (K, M')
        gamma[:, K:] -= gb @ b
        F += G @ b.T
    R1 = np.linalg.cholesky(gb.T @ gb).T  # upper triangular, gb'gb = R1'R1
    U = np.linalg.svd(R1 @ F.T @ F @ R1.T)[0]
    gb = np.linalg.solve(R1.T, gb.T).T @ U  # gb R1^{-1} U
    F = F @ R1.T @ U
    s = sign_by_mean(F)
    gamma[:, :K] = gb * s
    return gamma, F * s


def als(
    X: FloatArray,
    W: FloatArray,
    n_obs: FloatArray,
    G: FloatArray,
    K: int,
    *,
    max_iter: int = 10_000,
    tol: float = 1e-6,
    weighting: Weighting = "observations",
    init: tuple[FloatArray, FloatArray] | None = None,
) -> ALSResult:
    """Alternating least squares for IPCA on managed portfolios.

    Parameters
    ----------
    X : ndarray of shape (T, L)
        Managed portfolio returns.
    W : ndarray of shape (T, L, L)
        Instrument second moments.
    n_obs : ndarray of shape (T,)
        Cross-section sizes ``N_t``.
    G : ndarray of shape (T, M')
        Observed factors followed by a column of ones for the intercept
        (``M' = 0`` for the restricted model without either).
    K : int
        Number of latent factors (``>= 0``).
    max_iter, tol : int, float
        Stop when the largest absolute change in ``Gamma`` and ``F`` is below
        ``tol`` or after ``max_iter`` iterations.
    weighting : {"observations", "periods"}
        Weight every stock-period equally (KPS eq. 5) or every period equally.
    init : (gamma, factors), optional
        Starting values (e.g. from a previous fit); default is the SVD of ``X``.
    """
    T, L = X.shape
    Mp = G.shape[1]
    if K + Mp > L:
        raise DataError(f"K + observed factors (+ intercept) = {K + Mp} exceeds L = {L}")
    weights = n_obs if weighting == "observations" else np.ones(T)

    if K == 0:
        gamma = _gamma_step(np.zeros((T, 0)), X, W, G, weights)
        return ALSResult(gamma, np.zeros((T, 0)), 1, True, 0.0)

    if init is None:
        U, s, Vt = np.linalg.svd(X.T, full_matrices=False)
        gamma = np.zeros((L, K + Mp))
        gamma[:, :K] = U[:, :K]
        F = (s[:K, None] * Vt[:K]).T
    else:
        gamma, F = init[0].copy(), init[1].copy()

    change = np.inf
    n_iter = 0
    while n_iter < max_iter:
        F_new = _factor_step(gamma, X, W, G, K)
        gamma_new = _gamma_step(F_new, X, W, G, weights)
        gamma_new, F_new = _normalize(gamma_new, F_new, G, K)
        change = max(np.max(np.abs(gamma_new - gamma)), np.max(np.abs(F_new - F)))
        gamma, F = gamma_new, F_new
        n_iter += 1
        if change < tol:
            return ALSResult(gamma, F, n_iter, True, float(change))
    return ALSResult(gamma, F, n_iter, False, float(change))


# ============================================================================
# Bootstrap workers
# ============================================================================
@dataclass
class _BootstrapProblem:
    X_h0: FloatArray  # systematic part under H0, (T, L)
    resid: FloatArray  # residual managed portfolios d_t, (T, L)
    W: FloatArray
    n_obs: FloatArray
    G: FloatArray
    K: int
    stat_columns: tuple[slice, slice]  # (rows, cols) of Gamma entering the statistic
    max_iter: int
    tol: float
    weighting: Weighting
    block_size: int
    init: tuple[FloatArray, FloatArray]


def _wild_residuals(rng: np.random.Generator, resid: FloatArray, block_size: int) -> FloatArray:
    """Block-wild bootstrap residuals with unit-variance Student-t(5) multipliers."""
    T = resid.shape[0]
    n_blocks = int(np.ceil(T / block_size))
    max_start = T - block_size
    out = np.empty_like(resid)
    for b in range(n_blocks):
        start = int(rng.integers(0, max_start + 1))
        q = rng.standard_t(5) * np.sqrt(3.0 / 5.0)
        lo, hi = b * block_size, min((b + 1) * block_size, T)
        out[lo:hi] = q * resid[start : start + hi - lo]
    return out


def _bootstrap_draw(args: tuple[_BootstrapProblem, np.random.SeedSequence]) -> float:
    prob, seed = args
    rng = np.random.default_rng(seed)
    rows, cols = prob.stat_columns
    for _ in range(10):  # redraw on (rare) singular systems instead of looping forever
        X_b = prob.X_h0 + _wild_residuals(rng, prob.resid, prob.block_size)
        try:
            res = als(
                X_b,
                prob.W,
                prob.n_obs,
                prob.G,
                prob.K,
                max_iter=prob.max_iter,
                tol=prob.tol,
                weighting=prob.weighting,
                init=prob.init,
            )
        except np.linalg.LinAlgError:
            continue
        return float(np.sum(res.gamma[rows, cols] ** 2))
    return float("nan")


@dataclass
class BootstrapResult:
    """Result of an IPCA bootstrap test."""

    statistic: float
    pvalue: float
    draws: FloatArray
    hypothesis: str

    def __repr__(self) -> str:
        return (
            f"BootstrapResult({self.hypothesis}: W={self.statistic:.6g}, "
            f"p={self.pvalue:.4f}, draws={np.isfinite(self.draws).sum()})"
        )


# ============================================================================
# Estimator
# ============================================================================
@register("instrumented-pca")
class IPCA(LatentFactorModel):
    """Instrumented principal components analysis.

    Parameters
    ----------
    n_factors : int
        Number of latent factors ``K``.
    intercept : bool
        Estimate ``Gamma_alpha`` (unrestricted model).
    max_iter : int
        Maximum number of ALS iterations.
    tol : float
        Convergence tolerance on parameter changes (KPS use ``1e-6``).
    weighting : {"observations", "periods"}
        Equal weight per stock-period (default, KPS eq. 5) or per period.

    Fitted attributes
    -----------------
    gamma_beta_ : ndarray (L, K)
    gamma_delta_ : ndarray (L, M)
        Loadings of observed factors (``M = 0`` without observed factors).
    gamma_alpha_ : ndarray (L,)
        Zeros when ``intercept=False``.
    factors_ : ndarray (T, K)
    observed_factors_ : ndarray (T, M)
    n_iter_, converged_ : int, bool

    Notes
    -----
    Fit with a :class:`~latent_factor_models.core.data.CharacteristicPanel`,
    or with ``fit(returns, characteristics=Z, factors=G)``, or with the
    legacy stacked array ``[entity, time, y, z_1..z_L]``.  The instruments
    for ``returns[t]`` are ``characteristics[t]`` - lag them beforehand.
    """

    name = "ipca"

    gamma_beta_: FloatArray
    gamma_delta_: FloatArray
    gamma_alpha_: FloatArray
    factors_: FloatArray
    observed_factors_: FloatArray
    n_iter_: int
    converged_: bool
    panel_: CharacteristicPanel

    def __init__(
        self,
        n_factors: int = 1,
        intercept: bool = False,
        max_iter: int = 10_000,
        tol: float = 1e-6,
        weighting: Weighting = "observations",
    ) -> None:
        self.n_factors = n_factors
        self.intercept = intercept
        self.max_iter = max_iter
        self.tol = tol
        self.weighting = weighting

    def _validate_params(self) -> None:
        if not isinstance(self.n_factors, (int, np.integer)) or self.n_factors < 0:
            raise ValueError("n_factors must be a non-negative integer")
        if not isinstance(self.intercept, bool):
            raise ValueError("intercept must be a bool")
        if not 0 < self.tol < 1:
            raise ValueError("tol must lie in (0, 1)")
        if self.weighting not in ("observations", "periods"):
            raise ValueError("weighting must be 'observations' or 'periods'")

    # ----------------------------------------------------------------- data
    def _coerce(self, data: Any, **kwargs: Any) -> CharacteristicPanel:
        chars = kwargs.pop("characteristics", None)
        factors = kwargs.pop("factors", None)
        if kwargs:
            raise TypeError(f"unexpected arguments {sorted(kwargs)}")
        if isinstance(data, CharacteristicPanel):
            panel = data
        elif chars is not None:
            panel = CharacteristicPanel(as_float_array(data, 2), chars)
        else:
            panel = CharacteristicPanel.from_stacked(data)
        if factors is not None:
            panel = CharacteristicPanel(
                panel.returns,
                panel.characteristics,
                factors,
                panel.dates,
                panel.assets,
                panel.characteristic_names,
            )
        return panel

    def _design(self, panel: CharacteristicPanel) -> FloatArray:
        cols = [] if panel.factors is None else [panel.factors]
        if self.intercept:
            cols.append(np.ones((panel.T, 1)))
        return np.column_stack(cols) if cols else np.zeros((panel.T, 0))

    # ------------------------------------------------------------------ fit
    def _fit(self, data: CharacteristicPanel) -> None:
        if self.n_factors == 0 and not self.intercept and data.factors is None:
            raise ValueError("nothing to estimate: n_factors=0 without intercept or factors")
        X, W, n = data.managed_portfolios()
        G = self._design(data)
        res = als(
            X,
            W,
            n,
            G,
            self.n_factors,
            max_iter=self.max_iter,
            tol=self.tol,
            weighting=self.weighting,
        )
        if not res.converged:
            warnings.warn(
                f"IPCA did not converge in {res.n_iter} iterations "
                f"(last change {res.tol_reached:.2e})",
                ConvergenceWarning,
                stacklevel=3,
            )
        K, M = self.n_factors, 0 if data.factors is None else data.factors.shape[1]
        self.panel_ = data
        self._X, self._W, self._n, self._G = X, W, n, G
        self.gamma_ = res.gamma
        self.gamma_beta_ = res.gamma[:, :K]
        self.gamma_delta_ = res.gamma[:, K : K + M]
        self.gamma_alpha_ = res.gamma[:, -1] if self.intercept else np.zeros(data.L)
        self.factors_ = res.factors
        self.observed_factors_ = np.zeros((data.T, 0)) if data.factors is None else data.factors
        self.n_iter_, self.converged_ = res.n_iter, res.converged

    # ------------------------------------------------------- inference API
    @property
    def factor_means_(self) -> FloatArray:
        """Time-series mean of the latent factors (``lambda`` for predictive fits)."""
        self._check_fitted()
        return np.asarray(self.factors_.mean(axis=0))

    def transform(self, panel: Any = None, **kwargs: Any) -> FloatArray:
        """Latent factor realisations ``f_t`` given the estimated ``Gamma``.

        For a new panel this is the out-of-sample factor estimate of KPS
        Section 4.4: a cross-sectional regression of returns on
        ``Z_t Gamma_beta`` after removing the alpha and observed-factor parts.
        """
        self._check_fitted()
        if panel is None:
            return self.factors_
        p = self._coerce(panel, **kwargs)
        X, W, _ = p.managed_portfolios()
        return _factor_step(self.gamma_, X, W, self._design(p), self.n_factors)

    def _systematic(self, panel: CharacteristicPanel, F: FloatArray) -> FloatArray:
        """Expected returns ``Z_t (Gamma_alpha + Gamma_beta f_t + Gamma_delta g_t)``, (T, N)."""
        G = self._design(panel)
        coef = self.gamma_[:, : self.n_factors] @ F.T + self.gamma_[:, self.n_factors :] @ G.T
        _, z = panel.clean()
        out = np.einsum("tnl,lt->tn", z, coef)
        return np.where(panel.mask, out, np.nan)

    def predict(
        self, panel: Any = None, factors: Literal["realized", "mean"] = "realized", **kwargs: Any
    ) -> FloatArray:
        """Fitted (``"realized"``) or predicted (``"mean"``) returns, shape (T, N).

        Missing observations are NaN.
        """
        self._check_fitted()
        p = self.panel_ if panel is None else self._coerce(panel, **kwargs)
        if factors == "realized":
            F = self.transform(None if panel is None else p)
        elif factors == "mean":
            F = np.tile(self.factor_means_, (p.T, 1))
        else:
            raise ValueError("factors must be 'realized' or 'mean'")
        return self._systematic(p, F)

    def loadings(self, panel: Any = None, **kwargs: Any) -> FloatArray:
        """Time-varying betas ``Z_t Gamma_beta``, shape (T, N, K)."""
        self._check_fitted()
        p = self.panel_ if panel is None else self._coerce(panel, **kwargs)
        _, z = p.clean()
        return np.asarray(np.einsum("tnl,lk->tnk", z, self.gamma_beta_))

    def evaluate(self, panel: Any = None, **kwargs: Any) -> dict[str, float]:
        """Total and predictive R² at the asset and managed-portfolio level (uncentred).

        Out of sample (``panel`` given), factors are the cross-sectional
        estimates from :meth:`transform` and the predictive R² uses the
        in-sample factor means.
        """
        self._check_fitted()
        p = self.panel_ if panel is None else self._coerce(panel, **kwargs)
        r, _ = p.clean()
        X, W, _ = p.managed_portfolios()
        G = self._design(p)
        out: dict[str, float] = {}
        for label, F in (
            ("total", self.transform(None if panel is None else p)),
            ("pred", np.tile(self.factor_means_, (p.T, 1))),
        ):
            fit = np.nan_to_num(self._systematic(p, F))
            out[f"{label}_r2"] = float(1 - np.sum((r - fit) ** 2) / np.sum(r**2))
            coef = self.gamma_[:, : self.n_factors] @ F.T + self.gamma_[:, self.n_factors :] @ G.T
            fit_x = np.einsum("tlm,mt->tl", W, coef)
            out[f"{label}_r2_x"] = float(1 - np.sum((X - fit_x) ** 2) / np.sum(X**2))
        return out

    # ----------------------------------------------------------- bootstrap
    def _residual_portfolios(self) -> FloatArray:
        coef = (
            self.gamma_[:, : self.n_factors] @ self.factors_.T
            + self.gamma_[:, self.n_factors :] @ self._G.T
        )
        return np.asarray(self._X - np.einsum("tlm,mt->tl", self._W, coef))

    def _run_bootstrap(
        self,
        gamma_h0: FloatArray,
        stat_rows: Any,
        stat_cols: Any,
        hypothesis: str,
        n_draws: int,
        block_size: int,
        seed: int | None,
        executor: Executor | Backend | None,
        n_jobs: int | None,
    ) -> BootstrapResult:
        if block_size < 1 or block_size > self.panel_.T:
            raise ValueError(f"block_size must lie in [1, T={self.panel_.T}]")
        coef = (
            gamma_h0[:, : self.n_factors] @ self.factors_.T
            + gamma_h0[:, self.n_factors :] @ self._G.T
        )
        prob = _BootstrapProblem(
            X_h0=np.einsum("tlm,mt->tl", self._W, coef),
            resid=self._residual_portfolios(),
            W=self._W,
            n_obs=self._n,
            G=self._G,
            K=self.n_factors,
            stat_columns=(stat_rows, stat_cols),
            max_iter=self.max_iter,
            tol=self.tol,
            weighting=self.weighting,
            block_size=block_size,
            init=(self.gamma_, self.factors_),
        )
        stat = float(np.sum(self.gamma_[stat_rows, stat_cols] ** 2))
        ex = as_executor(executor, n_jobs)
        draws = np.array(ex.map(_bootstrap_draw, [(prob, s) for s in spawn_seeds(seed, n_draws)]))
        ok = np.isfinite(draws)
        if not ok.all():
            warnings.warn(f"{(~ok).sum()} bootstrap draws failed and were dropped", stacklevel=3)
        pvalue = float(np.mean(draws[ok] > stat)) if ok.any() else float("nan")
        return BootstrapResult(stat, pvalue, draws, hypothesis)

    def test_alpha(
        self,
        n_draws: int = 1000,
        *,
        block_size: int = 1,
        seed: int | None = None,
        executor: Executor | Backend | None = None,
        n_jobs: int | None = None,
    ) -> BootstrapResult:
        """Wild residual bootstrap test of ``H0: Gamma_alpha = 0`` (KPS Section 3.1).

        Requires a model fitted with ``intercept=True``.  The statistic is
        ``W_alpha = Gamma_alpha' Gamma_alpha``.
        """
        self._check_fitted()
        if not self.intercept:
            raise ValueError("test_alpha requires a model fitted with intercept=True")
        g0 = self.gamma_.copy()
        g0[:, -1] = 0.0
        return self._run_bootstrap(
            g0, slice(None), -1, "Gamma_alpha = 0", n_draws, block_size, seed, executor, n_jobs
        )

    def test_instrument(
        self,
        instruments: int | list[int],
        n_draws: int = 1000,
        *,
        block_size: int = 1,
        seed: int | None = None,
        executor: Executor | Backend | None = None,
        n_jobs: int | None = None,
    ) -> BootstrapResult:
        """Bootstrap test that instrument(s) ``l`` do not affect betas (KPS Section 3.2).

        ``H0``: the whole row(s) ``l`` of ``Gamma_beta`` are zero.  The
        statistic ``sum(Gamma_beta[l, :]**2)`` is rotation invariant.  Passing a
        list tests the rows jointly.  Requires ``intercept=False``.
        """
        self._check_fitted()
        if self.intercept:
            raise ValueError("test_instrument requires a model fitted with intercept=False")
        if self.n_factors == 0:
            raise ValueError("test_instrument requires latent factors")
        rows = np.atleast_1d(np.asarray(instruments, dtype=int))
        g0 = self.gamma_.copy()
        g0[rows, : self.n_factors] = 0.0
        return self._run_bootstrap(
            g0,
            rows[:, None],
            np.arange(self.n_factors)[None, :],
            f"Gamma_beta[{rows.tolist()}, :] = 0",
            n_draws,
            block_size,
            seed,
            executor,
            n_jobs,
        )

    def test_observed_factors(
        self,
        n_draws: int = 1000,
        *,
        block_size: int = 1,
        seed: int | None = None,
        executor: Executor | Backend | None = None,
        n_jobs: int | None = None,
    ) -> BootstrapResult:
        """Bootstrap test of ``H0: Gamma_delta = 0`` for all observed factors (KPS Section 3.3)."""
        self._check_fitted()
        M = self.gamma_delta_.shape[1]
        if M == 0 or self.intercept:
            raise ValueError("requires observed factors and intercept=False")
        cols = np.arange(self.n_factors, self.n_factors + M)
        g0 = self.gamma_.copy()
        g0[:, cols] = 0.0
        return self._run_bootstrap(
            g0, slice(None), cols, "Gamma_delta = 0", n_draws, block_size, seed, executor, n_jobs
        )
