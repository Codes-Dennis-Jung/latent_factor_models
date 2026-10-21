# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Dennis Jung
"""Principal component analysis of excess returns."""

from __future__ import annotations

from ..core.data import FactorData
from ..core.registry import register
from .spectral import SpectralFactorModel, SpectralSpec
from .targets import mean_target


@register()
class PCA(SpectralFactorModel):
    r"""Statistical factors from the second-moment matrix of returns.

    ``centered=True`` diagonalises the covariance matrix
    :math:`\hat\Sigma_R` (eq. 2 of Bryzgalova et al., 2026);
    ``centered=False`` the uncentred second moment :math:`R'R/T`
    (their footnote 4; the special case ``gamma = 0`` of all target models).

    Factors are portfolios of the *raw* returns, so they carry risk premia.

    Parameters
    ----------
    n_factors : int
        Number of factors ``K``.
    centered : bool
        Use the covariance (True) or the uncentred second moment (False).
    standardize : bool
        Divide each asset by its (training-sample) volatility before the
        eigen-decomposition (correlation PCA when ``centered``).

    Examples
    --------
    >>> import numpy as np
    >>> R = np.random.default_rng(0).standard_normal((200, 20))
    >>> PCA(n_factors=2).fit(R).factors_.shape
    (200, 2)
    """

    name = "pca"

    def __init__(
        self, n_factors: int = 1, centered: bool = True, standardize: bool = False
    ) -> None:
        self.n_factors = n_factors
        self.centered = centered
        self.standardize = standardize

    def _spec(self, data: FactorData) -> SpectralSpec:
        if not self.centered:
            return SpectralSpec()
        return SpectralSpec(ts_target=mean_target(data.T), gamma_ts=-1.0)
