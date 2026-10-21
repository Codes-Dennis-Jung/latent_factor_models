# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dennis Jung
"""Synthetic data generating processes with known ground truth.

``simulate_factor_returns``
    Approximate factor model ``R = alpha + F B' + e`` in the spirit of the
    Lettau-Pelger simulations: a strong "market" factor, a strong factor
    with a small risk premium and a *weak but highly priced* factor (the
    case where RP-PCA and target PCA beat PCA).  It also produces
    observable factors (one strong tradable proxy, one useless factor),
    reference factors for the alpha target, macro innovations and an
    XS target.

``simulate_ipca_panel``
    Unbalanced panel with characteristics-driven loadings
    ``r_{i,t} = z_{i,t}'(Gamma_alpha + Gamma_beta f_t) + e_{i,t}``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.data import CharacteristicPanel, FactorData
from ..core.linalg import FloatArray


@dataclass
class SimulatedFactorData:
    """Synthetic return panel and its ground truth.

    Attributes
    ----------
    returns : (T, N)
    factors : (T, K) true latent factors (with means = risk premia)
    loadings : (N, K) true loadings
    alpha : (N,) true pricing errors
    observable : (T, 2) ``[noisy proxy of factor 1, useless factor]``
    reference : (T, 1) reference factor (proxy of factor 1) for the alpha target
    macro : (T, 2) macro innovations correlated with the weak factor
    xs_target : (N, 1) noisy version of the weak factor's loadings
    risk_premia : (K,) true factor means
    """

    returns: FloatArray
    factors: FloatArray
    loadings: FloatArray
    alpha: FloatArray
    observable: FloatArray
    reference: FloatArray
    macro: FloatArray
    xs_target: FloatArray
    risk_premia: FloatArray

    def to_data(self, factors: str = "observable") -> FactorData:
        """Bundle into a :class:`FactorData`.

        ``factors`` selects the ``observable`` or the ``reference`` factors.
        """
        g = self.observable if factors == "observable" else self.reference
        return FactorData(self.returns, factors=g, xs_target=self.xs_target, macro=self.macro)


def simulate_factor_returns(
    T: int = 600,
    N: int = 100,
    *,
    factor_vol: tuple[float, ...] = (4.0, 2.0, 0.6),
    sharpe: tuple[float, ...] = (0.12, 0.05, 0.35),
    idio_vol: float = 3.0,
    alpha_vol: float = 0.0,
    seed: int | None = None,
) -> SimulatedFactorData:
    """Simulate a static factor model (monthly returns in percent).

    Parameters
    ----------
    T, N : int
        Periods and assets.
    factor_vol : tuple of float
        Factor volatilities (one entry per factor).  The default third factor
        is weak (explains little variance) but has a high Sharpe ratio.
    sharpe : tuple of float
        Per-period Sharpe ratios of the factors (risk premium / volatility).
    idio_vol : float
        Idiosyncratic volatility.
    alpha_vol : float
        Cross-sectional dispersion of true pricing errors.
    seed : int, optional
        Random seed.
    """
    rng = np.random.default_rng(seed)
    vol = np.asarray(factor_vol, dtype=float)
    K = vol.size
    mu = np.asarray(sharpe, dtype=float) * vol
    F = mu + vol * rng.standard_normal((T, K))
    B = rng.standard_normal((N, K))
    B[:, 0] = 1.0 + 0.3 * rng.standard_normal(N)  # market-like factor
    alpha = alpha_vol * rng.standard_normal(N)
    R = alpha + F @ B.T + idio_vol * rng.standard_normal((T, N))

    proxy = F[:, 0] + 0.1 * vol[0] * rng.standard_normal(T)  # tradable, noisy proxy
    useless = rng.standard_normal(T) * vol[0]
    observable = np.column_stack([proxy, useless])
    reference = proxy[:, None].copy()
    macro = np.column_stack(
        [
            (F[:, -1] - mu[-1]) / vol[-1] + 0.5 * rng.standard_normal(T),
            rng.standard_normal(T),
        ]
    )
    xs_target = B[:, [-1]] + 0.3 * rng.standard_normal((N, 1))
    return SimulatedFactorData(R, F, B, alpha, observable, reference, macro, xs_target, mu)


@dataclass
class SimulatedIPCAData:
    """Synthetic characteristic panel and its ground truth."""

    panel: CharacteristicPanel
    gamma_beta: FloatArray
    gamma_alpha: FloatArray
    factors: FloatArray


def simulate_ipca_panel(
    T: int = 200,
    N: int = 300,
    L: int = 6,
    K: int = 2,
    *,
    alpha_scale: float = 0.0,
    factor_mean: float = 0.5,
    idio_vol: float = 1.0,
    missing: float = 0.1,
    irrelevant: int = 0,
    seed: int | None = None,
) -> SimulatedIPCAData:
    """Simulate ``r_{i,t} = z_{i,t}'(Gamma_alpha + Gamma_beta f_t) + e_{i,t}``.

    Characteristics are persistent AR(1) processes, cross-sectionally
    rank-transformed to ``[-0.5, 0.5]`` as in Kelly, Pruitt and Su; the last
    instrument is a constant.  A share ``missing`` of observations is
    randomly removed (unbalanced panel).  The first ``irrelevant`` instruments
    have zero rows in ``Gamma_beta`` (useful to illustrate instrument tests).
    """
    rng = np.random.default_rng(seed)
    raw = np.zeros((T, N, L - 1))
    raw[0] = rng.standard_normal((N, L - 1))
    for t in range(1, T):
        raw[t] = 0.9 * raw[t - 1] + np.sqrt(1 - 0.81) * rng.standard_normal((N, L - 1))
    ranks = raw.argsort(axis=1).argsort(axis=1)
    Z = np.concatenate([(ranks + 1) / (N + 1) - 0.5, np.ones((T, N, 1))], axis=2)

    gamma_beta = np.linalg.qr(rng.standard_normal((L, K)))[0]
    gamma_beta[:irrelevant] = 0.0
    gamma_alpha = alpha_scale * rng.standard_normal(L)
    F = factor_mean + rng.standard_normal((T, K))
    signal = np.einsum("tnl,l->tn", Z, gamma_alpha) + np.einsum("tnl,lk,tk->tn", Z, gamma_beta, F)
    R = signal + idio_vol * rng.standard_normal((T, N))
    if missing > 0:
        R[rng.random((T, N)) < missing] = np.nan
    return SimulatedIPCAData(CharacteristicPanel(R, Z), gamma_beta, gamma_alpha, F)
