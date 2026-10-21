# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dennis Jung
"""Target PCA: factors with economic targets (Bryzgalova, DeMiguel, Li and Pelger, 2026)."""

from __future__ import annotations

from typing import Literal

from ..core.data import FactorData
from ..core.exceptions import DataError
from ..core.linalg import FloatArray
from ..core.registry import register
from .spectral import SpectralFactorModel, SpectralSpec
from .targets import macro_target, ts_target

TSTargetKind = Literal["given", "mean", "alpha", "comove"]
XSTargetKind = Literal["given", "macro"]
_TS_KINDS = ("given", "mean", "alpha", "comove")
_XS_KINDS = ("given", "macro")


def _build_ts_target(kind: str, data: FactorData, publication_lag: int) -> FloatArray:
    if kind == "given":
        if data.ts_target is None:
            raise DataError("ts_target='given' requires ts_target=... in fit()")
        return data.ts_target
    return ts_target(kind, data.T, data.factors, publication_lag)  # type: ignore[arg-type]


def _build_xs_target(kind: str, data: FactorData) -> FloatArray:
    if kind == "given":
        if data.xs_target is None:
            raise DataError("xs_target='given' requires xs_target=... in fit()")
        return data.xs_target
    if data.macro is None:
        raise DataError("xs_target='macro' requires macro=... (T x L innovations) in fit()")
    return macro_target(data.returns, data.macro)


def _check_kind(value: str, allowed: tuple[str, ...], name: str) -> None:
    if value not in allowed:
        raise ValueError(f"{name} must be one of {allowed}, got {value!r}")


@register("xs-target", "xs_target", "xs-pca", "xspca")
class XSTargetPCA(SpectralFactorModel):
    r"""Cross-sectional target PCA (Proposition 1).

    Loadings are pushed towards the span of a target :math:`\Lambda^{XS}`
    through :math:`W_{XS} = I_N + \gamma_{XS}\frac NL P_\Lambda`.

    Parameters
    ----------
    n_factors : int
        Number of factors.
    gamma_xs : float
        Target weight (``>= 0``).
    xs_target : {"given", "macro"}
        ``"given"`` uses ``xs_target=`` passed to :meth:`fit`; ``"macro"``
        builds :math:`\Lambda = R'Y(Y'Y)^{-1}` from ``macro=`` inside each
        estimation window (no look-ahead).
    """

    name = "xs_target_pca"

    def __init__(
        self, n_factors: int = 1, gamma_xs: float = 1.0, xs_target: XSTargetKind = "given"
    ) -> None:
        self.n_factors = n_factors
        self.gamma_xs = gamma_xs
        self.xs_target = xs_target

    def _validate_params(self) -> None:
        super()._validate_params()
        _check_kind(self.xs_target, _XS_KINDS, "xs_target")
        if self.gamma_xs < 0:
            raise ValueError("gamma_xs must be non-negative")

    def _spec(self, data: FactorData) -> SpectralSpec:
        return SpectralSpec(
            xs_target=_build_xs_target(self.xs_target, data), gamma_xs=float(self.gamma_xs)
        )


@register("ts-target", "ts_target", "ts-pca", "tspca")
class TSTargetPCA(SpectralFactorModel):
    r"""Time-series target PCA (Proposition 2).

    Up-weights the time-series projection of returns on a target ``G``
    through :math:`W_{TS} = I_T + \gamma_{TS} P_G`.

    Parameters
    ----------
    n_factors : int
        Number of factors.
    gamma_ts : float
        Target weight (``>= -1``); the paper's benchmark is 15.
    ts_target : {"alpha", "mean", "comove", "given"}
        ``"alpha"``: alpha target (eq. 10) from reference factors passed as
        ``factors=`` (e.g. FF5), rebuilt in each window; ``"mean"``:
        ``G = 1_T`` (RP-PCA); ``"comove"``: demeaned reference factors
        (package extension); ``"given"``: ``ts_target=`` from :meth:`fit`.
    publication_lag : int
        Ignore the last ``publication_lag`` periods of the reference factors.
    """

    name = "ts_target_pca"

    def __init__(
        self,
        n_factors: int = 1,
        gamma_ts: float = 15.0,
        ts_target: TSTargetKind = "alpha",
        publication_lag: int = 0,
    ) -> None:
        self.n_factors = n_factors
        self.gamma_ts = gamma_ts
        self.ts_target = ts_target
        self.publication_lag = publication_lag

    def _validate_params(self) -> None:
        super()._validate_params()
        _check_kind(self.ts_target, _TS_KINDS, "ts_target")
        if self.gamma_ts < -1:
            raise ValueError("gamma_ts must be >= -1")

    def _spec(self, data: FactorData) -> SpectralSpec:
        return SpectralSpec(
            ts_target=_build_ts_target(self.ts_target, data, self.publication_lag),
            gamma_ts=float(self.gamma_ts),
        )


@register("xsts-target", "xsts_target", "xs-ts-pca", "xsts-pca", "xstspca")
class XSTSTargetPCA(SpectralFactorModel):
    r"""Combined cross-sectional and time-series target PCA (Proposition 3).

    Eigenvectors of :math:`\frac1{NT} C R' W_{TS} R C` with
    :math:`C = I + \tilde\gamma P_\Lambda`, weights
    :math:`\hat\omega = C Q_K/(1+\tilde\gamma)`.

    Parameters
    ----------
    n_factors : int
        Number of factors.
    gamma_xs, gamma_ts : float
        Target weights.
    xs_target : {"given", "macro"}
        See :class:`XSTargetPCA`.
    ts_target : {"alpha", "mean", "comove", "given"}
        See :class:`TSTargetPCA`.
    publication_lag : int
        See :class:`TSTargetPCA`.
    """

    name = "xsts_target_pca"

    def __init__(
        self,
        n_factors: int = 1,
        gamma_xs: float = 1.0,
        gamma_ts: float = 15.0,
        xs_target: XSTargetKind = "given",
        ts_target: TSTargetKind = "alpha",
        publication_lag: int = 0,
    ) -> None:
        self.n_factors = n_factors
        self.gamma_xs = gamma_xs
        self.gamma_ts = gamma_ts
        self.xs_target = xs_target
        self.ts_target = ts_target
        self.publication_lag = publication_lag

    def _validate_params(self) -> None:
        super()._validate_params()
        _check_kind(self.xs_target, _XS_KINDS, "xs_target")
        _check_kind(self.ts_target, _TS_KINDS, "ts_target")
        if self.gamma_xs < 0 or self.gamma_ts < -1:
            raise ValueError("require gamma_xs >= 0 and gamma_ts >= -1")

    def _spec(self, data: FactorData) -> SpectralSpec:
        return SpectralSpec(
            ts_target=_build_ts_target(self.ts_target, data, self.publication_lag),
            gamma_ts=float(self.gamma_ts),
            xs_target=_build_xs_target(self.xs_target, data),
            gamma_xs=float(self.gamma_xs),
        )
