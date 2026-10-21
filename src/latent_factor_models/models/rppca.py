# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dennis Jung
"""Risk-premium PCA (Lettau and Pelger, 2020)."""

from __future__ import annotations

from ..core.data import FactorData
from ..core.registry import register
from .spectral import SpectralFactorModel, SpectralSpec
from .targets import mean_target


@register("rp-pca", "rp_pca")
class RPPCA(SpectralFactorModel):
    r"""Risk-premium PCA.

    Eigenvectors of :math:`\frac1T R'R + \gamma\,\bar R\bar R' =
    \hat\Sigma_R + (1+\gamma)\bar R\bar R'`: PCA with a penalty on pricing
    errors.  ``gamma = -1`` is covariance PCA, ``gamma = 0`` uncentred PCA.
    Lettau and Pelger use ``gamma = 10`` empirically; any ``gamma > -1``
    dominates PCA asymptotically, larger values help to detect weak factors
    but can overstate the in-sample Sharpe ratio.  Equivalent to TS-target
    PCA with the mean target ``G = 1_T``.

    Parameters
    ----------
    n_factors : int
        Number of factors ``K``.
    gamma : float
        Risk-premium weight (``>= -1``).
    standardize : bool
        Apply the cross-sectional weighting ``Q = diag(1/sigma_i)`` of Lettau
        and Pelger's empirical application (volatilities from the training
        sample).
    """

    name = "rppca"

    def __init__(self, n_factors: int = 1, gamma: float = 10.0, standardize: bool = False) -> None:
        self.n_factors = n_factors
        self.gamma = gamma
        self.standardize = standardize

    def _validate_params(self) -> None:
        super()._validate_params()
        if self.gamma < -1:
            raise ValueError(f"gamma must be >= -1, got {self.gamma}")

    def _spec(self, data: FactorData) -> SpectralSpec:
        return SpectralSpec(ts_target=mean_target(data.T), gamma_ts=float(self.gamma))
