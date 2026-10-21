# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dennis Jung
"""Abstract interfaces shared by every latent factor model.

Class hierarchy
---------------
``LatentFactorModel``
    Parameter handling (``get_params``/``set_params``/``clone``), input
    coercion, requirement checks and the ``fit`` template method.

``PortfolioFactorModel(LatentFactorModel)``
    Models whose factors are *portfolios* of the test assets,
    ``F = R @ weights_`` (PCA, RP-PCA, target PCA, SPCA).  Implements
    ``transform``, ``fitted_values``, ``residuals`` and ``evaluate`` once
    for all of them.

IPCA has time-varying loadings and characteristics-based factors and
therefore implements its own ``transform``/``evaluate`` on top of
``LatentFactorModel``.
"""

from __future__ import annotations

import copy
import inspect
from abc import ABC, abstractmethod
from typing import Any, ClassVar, TypeVar

import numpy as np

from .data import FactorData
from .exceptions import DataError, NotFittedError
from .linalg import FloatArray, as_float_array

M = TypeVar("M", bound="LatentFactorModel")


class LatentFactorModel(ABC):
    """Base class for all estimators.

    Subclasses implement :meth:`_fit` and list their constructor arguments
    explicitly (no ``*args``/``**kwargs``) so that :meth:`get_params` and
    :meth:`clone` work automatically.
    """

    #: Registry name, e.g. ``"rppca"``.
    name: ClassVar[str] = "base"
    #: Data fields that must be present in :class:`FactorData` for ``fit``.
    requires: ClassVar[tuple[str, ...]] = ()

    n_factors: int

    # ----------------------------------------------------------- parameters
    @classmethod
    def _param_names(cls) -> list[str]:
        sig = inspect.signature(cls.__init__)
        return [
            p.name
            for p in sig.parameters.values()
            if p.name != "self" and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        ]

    def get_params(self) -> dict[str, Any]:
        """Constructor parameters of this estimator."""
        return {k: getattr(self, k) for k in self._param_names()}

    def set_params(self: M, **params: Any) -> M:
        """Set constructor parameters (invalidates a previous fit)."""
        valid = set(self._param_names())
        for key, value in params.items():
            if key not in valid:
                raise ValueError(f"invalid parameter {key!r} for {type(self).__name__}")
            setattr(self, key, value)
        self._validate_params()
        return self

    def clone(self: M, **overrides: Any) -> M:
        """Return an unfitted copy, optionally with modified parameters."""
        params = copy.deepcopy(self.get_params())
        params.update(overrides)
        return type(self)(**params)

    def _validate_params(self) -> None:
        """Check parameter values; subclasses extend this."""
        if not isinstance(self.n_factors, (int, np.integer)) or self.n_factors < 1:
            raise ValueError(f"n_factors must be a positive integer, got {self.n_factors!r}")

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{type(self).__name__}({args})"

    # -------------------------------------------------------------- fitting
    def _coerce(self, data: Any, **kwargs: Any) -> Any:
        """Turn user input into the model's data container."""
        return FactorData.coerce(data, **kwargs)

    def _check_requirements(self, data: Any) -> None:
        for field_name in self.requires:
            if getattr(data, field_name, None) is None:
                raise DataError(f"{type(self).__name__} requires `{field_name}` to fit")

    def fit(self: M, data: Any, **kwargs: Any) -> M:
        """Estimate the model.

        Parameters
        ----------
        data : array-like, DataFrame, FactorData or CharacteristicPanel
            Training data.  For return-based models a ``T x N`` matrix of
            excess returns.
        **kwargs
            Extra fields of the data container, e.g. ``factors=``,
            ``xs_target=``, ``ts_target=``.

        Returns
        -------
        self
        """
        self._validate_params()
        container = self._coerce(data, **kwargs)
        self._check_requirements(container)
        self._fit(container)
        self.is_fitted_ = True
        return self

    @abstractmethod
    def _fit(self, data: Any) -> None:
        """Estimate parameters from a validated data container."""

    def _check_fitted(self) -> None:
        if not getattr(self, "is_fitted_", False):
            raise NotFittedError(f"{type(self).__name__} is not fitted yet; call fit() first")


class PortfolioFactorModel(LatentFactorModel):
    """Latent factor model whose factors are portfolios ``F = R @ weights_``.

    Fitted attributes (set by subclasses in :meth:`_fit`)
    ------------------------------------------------------
    weights_ : ndarray of shape (N, K)
        Factor portfolio weights.
    factors_ : ndarray of shape (T, K)
        In-sample factors ``R @ weights_``.
    loadings_ : ndarray of shape (N, K)
        Factor loadings (exposures) ``beta``.
    intercepts_ : ndarray of shape (N,)
        Time-series intercepts of the loading regressions (zeros if the
        model is estimated without intercept).
    eigenvalues_ : ndarray of shape (K,)
        Leading eigenvalues of the model's operator (model specific scale).
    data_ : FactorData
        The training data.
    """

    weights_: FloatArray
    factors_: FloatArray
    loadings_: FloatArray
    intercepts_: FloatArray
    eigenvalues_: FloatArray
    data_: FactorData

    def _as_returns(self, returns: Any) -> FloatArray:
        if isinstance(returns, FactorData):
            R = returns.returns
        else:
            R = as_float_array(returns.to_numpy() if hasattr(returns, "to_numpy") else returns, 2)
        if R.shape[1] != self.weights_.shape[0]:
            raise DataError(
                f"returns have {R.shape[1]} assets, model was fitted on {self.weights_.shape[0]}"
            )
        return R

    def transform(self, returns: Any = None) -> FloatArray:
        """Factor realisations ``R @ weights_`` (in-sample if ``returns`` is None)."""
        self._check_fitted()
        if returns is None:
            return self.factors_
        return self._as_returns(returns) @ self.weights_

    def fitted_values(self, returns: Any = None) -> FloatArray:
        """Systematic component ``F @ loadings_.T``.

        The intercepts are pricing errors and deliberately *not* part of the
        systematic component, so ``residuals(R).mean(axis=0)`` are the alphas.
        """
        F = self.transform(returns)
        return np.asarray(F @ self.loadings_.T)

    def residuals(self, returns: Any = None) -> FloatArray:
        """Idiosyncratic component ``R - F @ loadings_.T`` (mean = pricing errors)."""
        R = self.data_.returns if returns is None else self._as_returns(returns)
        return R - self.fitted_values(R)

    def projection_loadings(self) -> FloatArray:
        """Loadings ``L = W (W'W)^{-1}`` for which ``F L'`` projects returns on the factor space.

        For RP-PCA these are (up to scale) the eigenvector loadings that
        Lettau and Pelger use to price assets out of sample.
        """
        self._check_fitted()
        W = self.weights_
        return np.asarray(W @ np.linalg.inv(W.T @ W))

    def oos_loadings(self, kind: str = "ols") -> FloatArray:
        """Loadings pricing new returns: ``"ols"`` (time-series betas) or ``"projection"``."""
        if kind == "ols":
            return self.loadings_
        if kind == "projection":
            return self.projection_loadings()
        raise ValueError("loadings must be 'ols' or 'projection'")

    def evaluate(
        self, returns: Any = None, *, train_returns: Any = None, loadings: str = "ols"
    ) -> dict[str, float]:
        """Lettau-Pelger style asset-pricing metrics.

        Parameters
        ----------
        returns : array-like of shape (T', N), optional
            Evaluation sample (in-sample when omitted).
        train_returns : array-like, optional
            Training sample used to estimate the tangency weights of the
            factors for the out-of-sample Sharpe ratio (defaults to the fit
            sample).
        loadings : {"ols", "projection"}
            Loadings that price the evaluation sample: time-series OLS betas
            from the fit (Bryzgalova et al., eq. 13; default) or the
            projection on the factor-weight space (Lettau and Pelger's
            out-of-sample "cross-sectional pricing regression").  In sample
            with ``"ols"`` the pricing errors are the OLS intercepts.

        Returns
        -------
        dict
            See :func:`latent_factor_models.evaluation.metrics.evaluate_portfolio_factors`.
        """
        from ..evaluation.metrics import evaluate_portfolio_factors

        self._check_fitted()
        R = self.data_.returns if returns is None else self._as_returns(returns)
        R_train = self.data_.returns if train_returns is None else self._as_returns(train_returns)
        return evaluate_portfolio_factors(
            R, self.weights_, R_train=R_train, loadings=self.oos_loadings(loadings)
        )
