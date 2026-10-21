# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Dennis Jung
r"""The spectral engine shared by PCA, RP-PCA and target PCA.

All five estimators are the top-``K`` eigenvectors of one ``N x N`` operator

.. math::
    M = \frac{1}{NT}\, C\, R' \,(I_T + \gamma_{TS} P_G)\, R\, C,
    \qquad C = I_N + \tilde\gamma\, P_\Lambda,
    \qquad \tilde\gamma = \sqrt{\gamma_{XS} N / L + 1} - 1,

with uncentred returns ``R`` (``T x N``), a time-series target ``G``
(``T x J``) and a cross-sectional target ``Lambda`` (``N x L``).  Since
``P_Lambda`` is a projector, ``C^2 = I + gamma_XS (N/L) P_Lambda = W_XS``,
so ``C`` is the closed-form square root of the XS weighting matrix.

================  ===========================  ===================
Model             time-series part             cross-sectional
================  ===========================  ===================
PCA (centred)     ``G = 1_T, gamma_TS = -1``   ``C = I``
RP-PCA            ``G = 1_T, gamma_TS = gamma``  ``C = I``
TS-target PCA     ``G = G_TS``                 ``C = I``
XS-target PCA     none                         ``Lambda = Lambda_XS``
XS-TS-target PCA  ``G = G_TS``                 ``Lambda = Lambda_XS``
================  ===========================  ===================

With eigenvectors ``Q_K`` of ``M`` the factor portfolio weights are
``omega = C Q_K / (1 + gamma~)``, factors ``F = R omega`` and loadings are
time-series OLS estimates (with intercept) of ``R`` on ``F`` (eq. 13).  The
``T x T`` matrix ``W_TS`` is never formed:
``R' W_TS R = R'R + gamma_TS (Q_G'R)'(Q_G'R)`` with an orthonormal basis
``Q_G`` of ``col(G)``.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass

import numpy as np

from ..core.base import PortfolioFactorModel
from ..core.data import FactorData
from ..core.linalg import FloatArray, ols, orthonormal_basis, sign_by_mean, top_gram_eigh
from ..evaluation import metrics


@dataclass(frozen=True)
class SpectralSpec:
    """Ingredients of the spectral operator for one estimation sample."""

    ts_target: FloatArray | None = None
    gamma_ts: float = 0.0
    xs_target: FloatArray | None = None
    gamma_xs: float = 0.0


class SpectralFactorModel(PortfolioFactorModel):
    """Abstract weighted-PCA estimator; subclasses only define :meth:`_spec`.

    Additional fitted attributes
    ----------------------------
    eigenvectors_ : ndarray of shape (N, K)
        Unit-norm eigenvectors ``Q_K`` of the operator ``M`` (sign-aligned with
        the factors).
    operator_ : ndarray of shape (N, N)
        The operator ``M`` (computed on access; only the thin ``T x N``
        weighted panel is stored).  ``eigenvalues_ / operator_trace_`` are
        absorption ratios comparable across models.
    operator_trace_ : float
        ``trace(M)``.
    gamma_tilde_ : float
        Square-root XS weight ``gamma~``.
    spec_ : SpectralSpec
        Targets and weights actually used.
    scale_ : ndarray of shape (N,)
        Asset volatilities used for standardisation (ones if not standardised).
    """

    eigenvectors_: FloatArray
    operator_trace_: float
    _weighted: FloatArray
    gamma_tilde_: float
    spec_: SpectralSpec
    scale_: FloatArray

    #: Subclasses with a ``standardize`` parameter set it as an attribute.
    standardize: bool = False

    @abstractmethod
    def _spec(self, data: FactorData) -> SpectralSpec:
        """Build targets and weights from the (training) data."""

    # ------------------------------------------------------------------
    def _fit(self, data: FactorData) -> None:
        R = data.returns
        T, N = R.shape
        K = self.n_factors
        if K > N:
            raise ValueError(f"n_factors={K} exceeds the number of assets N={N}")
        spec = self._spec(data)
        # Optional cross-sectional weighting Q = diag(1/sigma_i) (Lettau-Pelger,
        # empirical application): estimate on standardised returns, map back.
        scale = R.std(axis=0) if self.standardize else np.ones(N)
        if np.any(scale <= 0):
            raise ValueError("cannot standardize assets with zero variance")
        # Thin weighted panel X = W_TS^{1/2} R C, so that M = X'X / (NT).  Both
        # square roots are closed form because P_G and P_Lambda are projectors:
        # (I + g P)^2 = I + (2g + g^2) P, i.e. g = sqrt(1 + gamma) - 1.
        X = R / scale if self.standardize else R
        if spec.ts_target is not None and spec.gamma_ts != 0.0:
            if spec.gamma_ts < -1.0:
                raise ValueError("gamma_ts must be >= -1 (W_TS positive semi-definite)")
            QG = orthonormal_basis(spec.ts_target)
            g_ts = float(np.sqrt(1.0 + spec.gamma_ts) - 1.0)
            X = X + g_ts * (QG @ (QG.T @ X))

        gamma_tilde = 0.0
        QL: FloatArray | None = None
        if spec.xs_target is not None and spec.gamma_xs != 0.0:
            QL = orthonormal_basis(spec.xs_target)
            L = QL.shape[1]
            if spec.gamma_xs * N / L + 1.0 < 0.0:
                raise ValueError("gamma_xs too negative: W_XS must be positive semi-definite")
            gamma_tilde = float(np.sqrt(spec.gamma_xs * N / L + 1.0) - 1.0)
            X = X + gamma_tilde * ((X @ QL) @ QL.T)

        # Top-K eigenpairs of X'X through the smaller of the N x N / T x T Gram matrices.
        gram_vals, Q = top_gram_eigh(X, K)
        eigenvalues = gram_vals / (N * T)
        omega = (Q if QL is None else Q + gamma_tilde * (QL @ (QL.T @ Q))) / (1.0 + gamma_tilde)
        omega = omega / scale[:, None]
        F = R @ omega
        signs = sign_by_mean(F)
        omega, Q, F = omega * signs, Q * signs, F * signs
        loadings, intercepts = ols(F, R, intercept=True)

        self.data_ = data
        self.spec_ = spec
        self.weights_ = omega
        self.factors_ = F
        self.loadings_ = loadings
        self.intercepts_ = intercepts
        self.eigenvalues_ = eigenvalues
        self.eigenvectors_ = Q
        self._weighted = X
        self.operator_trace_ = float(np.einsum("ij,ij->", X, X)) / (N * T)
        self.gamma_tilde_ = gamma_tilde
        self.scale_ = scale

    # ------------------------------------------------------------------
    @property
    def operator_(self) -> FloatArray:
        """The ``N x N`` operator ``M = X'X/(NT)``, built on demand (not stored)."""
        self._check_fitted()
        X = self._weighted
        return np.asarray(X.T @ X) / X.size

    def spectrum(self) -> FloatArray:
        """All ``N`` eigenvalues of the operator in descending order (zeros beyond rank)."""
        self._check_fitted()
        X = self._weighted
        s2 = np.linalg.svd(X, compute_uv=False) ** 2 / X.size
        return np.concatenate([s2, np.zeros(X.shape[1] - s2.size)])

    def absorption_ratio(self) -> FloatArray:
        """Share ``lambda_k / trace(M)`` of the operator captured by each factor."""
        self._check_fitted()
        return self.eigenvalues_ / self.operator_trace_

    def variance_signal(self, n_residual_factors: int = 6) -> FloatArray:
        r"""Signal-to-noise of the factors (Lettau and Pelger, Table 2).

        Eigenvalues of :math:`\frac1T R'(I_T + \gamma P_G)R` divided by the
        average idiosyncratic variance :math:`\hat\sigma^2_e` left after
        ``n_residual_factors`` PCA factors.  Unlike :meth:`absorption_ratio`
        this is comparable across models and values of ``gamma``.
        """
        self._check_fitted()
        R = self.data_.returns / self.scale_
        Rc = R - R.mean(axis=0)
        k = min(n_residual_factors, min(Rc.shape) - 1)
        U, s, Vt = np.linalg.svd(Rc, full_matrices=False)
        resid = Rc - (U[:, :k] * s[:k]) @ Vt[:k]
        sigma2_e = float(np.mean(resid.var(axis=0)))
        return np.asarray(self.eigenvalues_ * R.shape[1] / sigma2_e)

    def target_metrics(self) -> dict[str, FloatArray | float]:
        """In-sample XS- and TS-metrics of the factor weights (where targets exist)."""
        self._check_fitted()
        out: dict[str, FloatArray | float] = {}
        if self.spec_.xs_target is not None:
            out["xs_metric"] = metrics.xs_metric(self.weights_, self.spec_.xs_target)
            b = metrics.tangency_weights(self.factors_)
            sdf_weights = self.weights_ @ b
            out["xs_metric_sdf"] = float(metrics.xs_metric(sdf_weights, self.spec_.xs_target)[0])
        if self.spec_.ts_target is not None:
            out["ts_metric"] = metrics.ts_metric(
                self.data_.returns, self.weights_, self.spec_.ts_target
            )
        return out
