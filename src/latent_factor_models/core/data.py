# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dennis Jung
"""Validated data containers.

Conventions used throughout the package
---------------------------------------
* Returns are excess returns stored as a ``T x N`` matrix (time in rows,
  assets in columns), matching a pandas ``DataFrame`` indexed by date.
* Observable factors ``G`` are ``T x d``.
* A cross-sectional target ``Lambda_XS`` is ``N x L``; a time-series target
  ``G_TS`` is ``T x J``.
* Characteristic panels (IPCA) are ``T x N x L`` arrays aligned with the
  return matrix: ``Z[t]`` are the instruments for ``R[t]``.  Lag the
  characteristics *before* building the panel if they must be predetermined.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .exceptions import DataError
from .linalg import FloatArray, as_float_array


def _labels(obj: Any, axis: int) -> list[Any] | None:
    """Extract pandas index (axis 0) or columns (axis 1) labels if present."""
    attr = "index" if axis == 0 else "columns"
    labels = getattr(obj, attr, None)
    return None if labels is None else list(labels)


def _values(obj: Any) -> Any:
    return obj.to_numpy() if hasattr(obj, "to_numpy") else obj


def _check_finite(arr: FloatArray, name: str) -> None:
    if not np.all(np.isfinite(arr)):
        raise DataError(
            f"{name} contains NaN or inf. Balanced-panel estimators need complete data; "
            "impute or drop assets/periods first (see FactorData.dropna)."
        )


@dataclass
class FactorData:
    """A balanced panel of returns plus optional side information.

    Parameters
    ----------
    returns : array-like of shape (T, N)
        Excess returns.  A pandas ``DataFrame`` keeps its index/columns as
        labels.
    factors : array-like of shape (T, d), optional
        Observable factors ``g_t`` (e.g. for risk-premium inference).
    xs_target : array-like of shape (N, L), optional
        Cross-sectional target (loadings directions) for XS-target PCA.
    ts_target : array-like of shape (T, J), optional
        Time-series target for TS-target PCA.
    macro : array-like of shape (T, L), optional
        Macro innovations used to build a cross-sectional macro target
        ``Lambda_XS = R' Y (Y'Y)^{-1}`` inside each estimation window.
    index, assets, factor_names : sequence, optional
        Labels for time periods, assets and observable factors.
    """

    returns: FloatArray
    factors: FloatArray | None = None
    xs_target: FloatArray | None = None
    ts_target: FloatArray | None = None
    macro: FloatArray | None = None
    index: list[Any] | None = None
    assets: list[Any] | None = None
    factor_names: list[Any] | None = None
    allow_nan: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        if self.index is None:
            self.index = _labels(self.returns, 0)
        if self.assets is None:
            self.assets = _labels(self.returns, 1)
        if self.factors is not None and self.factor_names is None:
            self.factor_names = _labels(self.factors, 1)
        self.returns = as_float_array(_values(self.returns), 2, "returns")
        T, N = self.returns.shape
        if T < 2 or N < 1:
            raise DataError(f"returns must have at least 2 periods and 1 asset, got {(T, N)}")
        if not self.allow_nan:
            _check_finite(self.returns, "returns")
        if self.factors is not None:
            self.factors = as_float_array(_values(self.factors), 2, "factors")
            if self.factors.shape[0] != T:
                raise DataError(f"factors has {self.factors.shape[0]} rows, returns has {T}")
            _check_finite(self.factors, "factors")
        if self.macro is not None:
            self.macro = as_float_array(_values(self.macro), 2, "macro")
            if self.macro.shape[0] != T:
                raise DataError(f"macro has {self.macro.shape[0]} rows, returns has {T}")
            _check_finite(self.macro, "macro")
        if self.xs_target is not None:
            self.xs_target = as_float_array(_values(self.xs_target), 2, "xs_target")
            if self.xs_target.shape[0] != N:
                raise DataError(f"xs_target must have N={N} rows, got {self.xs_target.shape[0]}")
            _check_finite(self.xs_target, "xs_target")
        if self.ts_target is not None:
            self.ts_target = as_float_array(_values(self.ts_target), 2, "ts_target")
            if self.ts_target.shape[0] != T:
                raise DataError(f"ts_target must have T={T} rows, got {self.ts_target.shape[0]}")
            _check_finite(self.ts_target, "ts_target")
        for name, labels, size in (("index", self.index, T), ("assets", self.assets, N)):
            if labels is not None and len(labels) != size:
                raise DataError(f"{name} has {len(labels)} labels, expected {size}")

    # ------------------------------------------------------------------ API
    @property
    def T(self) -> int:
        """Number of time periods."""
        return int(self.returns.shape[0])

    @property
    def N(self) -> int:
        """Number of assets."""
        return int(self.returns.shape[1])

    @classmethod
    def coerce(cls, data: Any, **kwargs: Any) -> FactorData:
        """Build a :class:`FactorData` from an array/DataFrame, or update an existing one."""
        if isinstance(data, FactorData):
            updates = {k: v for k, v in kwargs.items() if v is not None}
            return replace(data, **updates) if updates else data
        return cls(returns=data, **{k: v for k, v in kwargs.items() if v is not None})

    def take(self, rows: ArrayLike) -> FactorData:
        """Sub-sample time periods (rows); cross-sectional targets are kept."""
        idx = np.asarray(rows)
        return FactorData(
            returns=self.returns[idx],
            factors=None if self.factors is None else self.factors[idx],
            xs_target=self.xs_target,
            ts_target=None if self.ts_target is None else self.ts_target[idx],
            macro=None if self.macro is None else self.macro[idx],
            index=None if self.index is None else [self.index[i] for i in np.arange(self.T)[idx]],
            assets=self.assets,
            factor_names=self.factor_names,
        )

    def take_assets(self, cols: ArrayLike) -> FactorData:
        """Sub-sample assets (columns)."""
        idx = np.asarray(cols)
        return FactorData(
            returns=self.returns[:, idx],
            factors=self.factors,
            xs_target=None if self.xs_target is None else self.xs_target[idx],
            ts_target=self.ts_target,
            macro=self.macro,
            index=self.index,
            assets=(
                None if self.assets is None else [self.assets[i] for i in np.arange(self.N)[idx]]
            ),
            factor_names=self.factor_names,
        )

    @staticmethod
    def dropna(returns: Any, max_missing: float = 0.0) -> Any:
        """Drop assets with more than ``max_missing`` share of NaNs, then incomplete periods.

        Works on arrays and DataFrames and returns the same type.
        """
        values = np.asarray(_values(returns), dtype=float)
        keep_cols = np.isnan(values).mean(axis=0) <= max_missing
        values = values[:, keep_cols]
        keep_rows = ~np.isnan(values).any(axis=1)
        if hasattr(returns, "iloc"):
            return returns.iloc[np.flatnonzero(keep_rows), np.flatnonzero(keep_cols)]
        return values[keep_rows]


@dataclass
class CharacteristicPanel:
    """An (unbalanced) panel of returns and instruments for IPCA.

    Parameters
    ----------
    returns : array-like of shape (T, N)
        Returns; ``NaN`` marks a missing observation.
    characteristics : array-like of shape (T, N, L)
        Instruments ``z_{i,t}`` aligned with ``returns[t, i]``.  Any NaN in
        an observation's characteristics drops that observation.
    factors : array-like of shape (T, M), optional
        Observed (pre-specified) factors ``g_t``.
    dates, assets, characteristic_names : sequence, optional
        Labels.
    """

    returns: FloatArray
    characteristics: FloatArray
    factors: FloatArray | None = None
    dates: list[Any] | None = None
    assets: list[Any] | None = None
    characteristic_names: list[Any] | None = None
    mask: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.returns = as_float_array(self.returns, 2, "returns")
        self.characteristics = as_float_array(self.characteristics, 3, "characteristics")
        T, N = self.returns.shape
        if self.characteristics.shape[:2] != (T, N):
            raise DataError(
                f"characteristics shape {self.characteristics.shape[:2]} does not match "
                f"returns shape {(T, N)}"
            )
        if self.factors is not None:
            self.factors = as_float_array(self.factors, 2, "factors")
            if self.factors.shape[0] != T:
                raise DataError(f"factors has {self.factors.shape[0]} rows, expected T={T}")
            _check_finite(self.factors, "factors")
        valid = np.isfinite(self.returns) & np.all(np.isfinite(self.characteristics), axis=2)
        self.mask = valid
        if not valid.any(axis=1).all():
            empty = np.flatnonzero(~valid.any(axis=1))
            raise DataError(f"periods {empty.tolist()} have no valid observation")

    @property
    def T(self) -> int:
        """Number of time periods."""
        return int(self.returns.shape[0])

    @property
    def N(self) -> int:
        """Number of assets (maximum cross-section)."""
        return int(self.returns.shape[1])

    @property
    def L(self) -> int:
        """Number of instruments."""
        return int(self.characteristics.shape[2])

    @property
    def n_obs(self) -> FloatArray:
        """Number of valid observations per period, shape (T,)."""
        return np.asarray(self.mask.sum(axis=1), dtype=float)

    def clean(self) -> tuple[FloatArray, FloatArray]:
        """Returns and characteristics with invalid observations set to zero.

        Zeroed rows drop out of all cross-sectional sums, which is how the
        unbalanced panel is handled without loops.
        """
        r = np.where(self.mask, self.returns, 0.0)
        z = np.where(self.mask[:, :, None], self.characteristics, 0.0)
        return r, z

    def managed_portfolios(self) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Characteristic-managed portfolios.

        Returns
        -------
        X : ndarray of shape (T, L)
            ``X_t = Z_t' r_t / N_t``.
        W : ndarray of shape (T, L, L)
            ``W_t = Z_t' Z_t / N_t``.
        n_obs : ndarray of shape (T,)
            ``N_t``.
        """
        r, z = self.clean()
        n = self.n_obs
        X = np.einsum("tnl,tn->tl", z, r) / n[:, None]
        W = np.einsum("tnl,tnm->tlm", z, z) / n[:, None, None]
        return X, W, n

    def take(self, rows: ArrayLike) -> CharacteristicPanel:
        """Sub-sample time periods."""
        idx = np.asarray(rows)
        return CharacteristicPanel(
            returns=self.returns[idx],
            characteristics=self.characteristics[idx],
            factors=None if self.factors is None else self.factors[idx],
            dates=None if self.dates is None else [self.dates[i] for i in np.arange(self.T)[idx]],
            assets=self.assets,
            characteristic_names=self.characteristic_names,
        )

    # ----------------------------------------------------------- builders
    @classmethod
    def from_stacked(cls, panel: ArrayLike) -> CharacteristicPanel:
        """Build from the legacy stacked layout ``[entity, time, y, z_1, ..., z_L]``."""
        P = as_float_array(panel, 2, "panel")
        if P.shape[1] < 4:
            raise DataError("stacked panel needs columns [entity, time, y, z_1, ...]")
        ids, i_idx = np.unique(P[:, 0], return_inverse=True)
        dates, t_idx = np.unique(P[:, 1], return_inverse=True)
        if len(np.unique(i_idx * len(dates) + t_idx)) != len(P):
            raise DataError("duplicate (entity, time) pairs in stacked panel")
        L = P.shape[1] - 3
        R = np.full((len(dates), len(ids)), np.nan)
        Z = np.full((len(dates), len(ids), L), np.nan)
        R[t_idx, i_idx] = P[:, 2]
        Z[t_idx, i_idx] = P[:, 3:]
        return cls(R, Z, dates=dates.tolist(), assets=ids.tolist())

    @classmethod
    def from_long(
        cls,
        frame: Any,
        entity: str,
        time: str,
        target: str,
        characteristics: Sequence[str],
    ) -> CharacteristicPanel:
        """Build from a long-format pandas ``DataFrame`` (one row per entity-date)."""
        ids, i_idx = np.unique(frame[entity].to_numpy(), return_inverse=True)
        dates, t_idx = np.unique(frame[time].to_numpy(), return_inverse=True)
        if len(np.unique(i_idx * len(dates) + t_idx)) != len(frame):
            raise DataError("duplicate (entity, time) pairs in frame")
        chars = list(characteristics)
        R = np.full((len(dates), len(ids)), np.nan)
        Z = np.full((len(dates), len(ids), len(chars)), np.nan)
        R[t_idx, i_idx] = frame[target].to_numpy(dtype=float)
        Z[t_idx, i_idx] = frame[chars].to_numpy(dtype=float)
        return cls(R, Z, dates=list(dates), assets=list(ids), characteristic_names=chars)
