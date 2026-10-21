# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dennis Jung
"""Experiment engine: fit and compare many models in and out of sample.

Out-of-sample protocol (Lettau and Pelger, 2020; Bryzgalova et al., 2026)
-------------------------------------------------------------------------
For every period ``t`` after the first window, each model is estimated on
the window ending at ``t - 1``.  With weights ``omega``, loadings ``beta``
and tangency weights ``b`` from that window, the period-``t`` factors are
``F_t = R_t omega``, residuals ``e_t = R_t - beta F_t`` and the SDF return
``b'F_t``.  ``beta`` are the window's time-series OLS betas
(``oos_loadings="ols"``, Bryzgalova et al., eq. 13) or the projection
loadings ``omega (omega'omega)^{-1}`` (``oos_loadings="projection"``, Lettau
and Pelger's cross-sectional pricing with out-of-sample loadings).  Pooled
over all test periods this gives the out-of-sample
Sharpe ratio, pricing errors ``alpha = mean(e)``, ``RMS_alpha`` and the
unexplained variation.  For IPCA the pooled out-of-sample total and
predictive R² of Kelly, Pruitt and Su (2019, Section 4.4) are reported.

Windows are independent tasks executed by the :class:`Executor`; re-estimation
can be thinned with ``refit_every``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.base import LatentFactorModel, PortfolioFactorModel
from ..core.data import CharacteristicPanel, FactorData
from ..core.registry import create_model
from ..evaluation import metrics
from .parallel import Backend, Executor, as_executor


@dataclass
class ModelSpec:
    """A named model configuration, e.g. ``ModelSpec("rppca", {"n_factors": 3, "gamma": 10})``."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)
    label: str | None = None

    def build(self) -> LatentFactorModel:
        """Instantiate the model."""
        return create_model(self.name, **self.params)

    @property
    def display(self) -> str:
        """Label used in result tables."""
        if self.label:
            return self.label
        args = ",".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.name}({args})"


ModelLike = ModelSpec | LatentFactorModel | dict[str, Any]


def _short_label(model: LatentFactorModel) -> str:
    """Class name plus parameters that differ from their defaults."""
    import inspect

    sig = inspect.signature(type(model).__init__)
    changed = [
        f"{k}={v}"
        for k, v in model.get_params().items()
        if k == "n_factors" or sig.parameters[k].default != v
    ]
    return f"{type(model).__name__}({','.join(changed)})"


def _as_spec(m: ModelLike) -> tuple[str, LatentFactorModel]:
    if isinstance(m, LatentFactorModel):
        return _short_label(m), m
    if isinstance(m, dict):
        m = ModelSpec(m["name"], dict(m.get("params", {})), m.get("label"))
    return m.display, m.build()


@dataclass
class EngineResult:
    """Tidy result records (one dict per model and sample)."""

    records: list[dict[str, Any]]

    def to_frame(self) -> Any:
        """Return a ``pandas.DataFrame`` (requires pandas)."""
        import pandas as pd

        return pd.DataFrame.from_records(self.records)

    def __str__(self) -> str:
        if not self.records:
            return "EngineResult(empty)"
        keys = list(self.records[0])
        widths = [max(len(str(k)), 12) for k in keys]
        lines = ["  ".join(f"{k:>{w}}" for k, w in zip(keys, widths, strict=True))]
        for rec in self.records:
            cells = []
            for k, w in zip(keys, widths, strict=True):
                v = rec.get(k, "")
                cells.append(f"{v:>{w}.4f}" if isinstance(v, float) else f"{v!s:>{w}}")
            lines.append("  ".join(cells))
        return "\n".join(lines)


# --------------------------------------------------------------------- tasks
def _in_sample_task(args: tuple[str, LatentFactorModel, Any]) -> dict[str, Any]:
    label, model, data = args
    model.fit(data)
    rec: dict[str, Any] = {"model": label, "sample": "in-sample"}
    rec.update(model.evaluate())  # type: ignore[attr-defined]
    return rec


def _window_task(
    args: tuple[LatentFactorModel, Any, np.ndarray, np.ndarray, str],
) -> dict[str, Any]:
    """Fit on ``train`` rows and return what is needed to evaluate the ``test`` rows."""
    model, data, train, test, oos_loadings = args
    est = model.clone().fit(data.take(train))
    if isinstance(est, PortfolioFactorModel):
        R = est._as_returns(data.returns[test])
        F = R @ est.weights_
        b = metrics.tangency_weights(est.factors_)
        beta = est.oos_loadings(oos_loadings)
        return {"R": R, "F": F, "resid": R - F @ beta.T, "sdf": F @ b}
    test_panel = data.take(test)
    r, _ = test_panel.clean()
    total = np.nan_to_num(est.predict(test_panel, factors="realized"))  # type: ignore[attr-defined]
    pred = np.nan_to_num(est.predict(test_panel, factors="mean"))  # type: ignore[attr-defined]
    return {"r": r, "total": total, "pred": pred}


def _aggregate(label: str, parts: list[dict[str, Any]]) -> dict[str, Any]:
    rec: dict[str, Any] = {"model": label, "sample": "out-of-sample"}
    if "R" in parts[0]:
        R = np.vstack([p["R"] for p in parts])
        e = np.vstack([p["resid"] for p in parts])
        sdf = np.concatenate([p["sdf"] for p in parts])
        rec["sharpe_ratio"] = metrics.sharpe_ratio(sdf)
        rec["rms_alpha"] = metrics.rms_alpha(e.mean(axis=0))
        rec["unexplained_variation"] = metrics.unexplained_variation(R, e)
        rec["total_r2"] = float(1 - np.sum(e**2) / np.sum(R**2))
    else:
        r = np.vstack([p["r"] for p in parts])
        denom = np.sum(r**2)
        total = np.vstack([p["total"] for p in parts])
        pred = np.vstack([p["pred"] for p in parts])
        rec["total_r2"] = float(1 - np.sum((r - total) ** 2) / denom)
        rec["pred_r2"] = float(1 - np.sum((r - pred) ** 2) / denom)
    rec["n_windows"] = len(parts)
    return rec


class Engine:
    """Run a set of models on one data set.

    Parameters
    ----------
    models : sequence of ModelSpec, dict or model instance
        Models to compare.  Dicts look like
        ``{"name": "rppca", "params": {"n_factors": 3, "gamma": 10}}``.
    executor : Executor or {"serial", "thread", "process"}, optional
        Parallel backend for models and windows.
    n_jobs : int, optional
        Number of workers.

    Examples
    --------
    >>> from latent_factor_models import Engine, ModelSpec, simulate_factor_returns
    >>> sim = simulate_factor_returns(T=300, N=30, seed=1)
    >>> eng = Engine([ModelSpec("pca", {"n_factors": 3}), ModelSpec("rppca", {"n_factors": 3})])
    >>> res = eng.run(sim.returns, window=200, refit_every=20)
    >>> len(res.records)
    4
    """

    def __init__(
        self,
        models: Sequence[ModelLike],
        executor: Executor | Backend | None = None,
        n_jobs: int | None = None,
    ) -> None:
        self.models = [_as_spec(m) for m in models]
        self.executor = as_executor(executor, n_jobs)

    def _coerce(self, data: Any, **kwargs: Any) -> Any:
        if isinstance(data, CharacteristicPanel):
            return data
        return FactorData.coerce(data, **kwargs)

    def in_sample(self, data: Any, **kwargs: Any) -> EngineResult:
        """Fit every model on the full sample and report in-sample metrics."""
        data = self._coerce(data, **kwargs)
        tasks = [(label, model.clone(), data) for label, model in self.models]
        return EngineResult(self.executor.map(_in_sample_task, tasks))

    @staticmethod
    def windows(
        T: int, window: int, refit_every: int = 1, expanding: bool = False
    ) -> Iterable[tuple[np.ndarray, np.ndarray]]:
        """Train/test index pairs of the rolling (or expanding) out-of-sample design."""
        if not 2 <= window < T:
            raise ValueError(f"window must lie in [2, T={T})")
        for start in range(window, T, refit_every):
            train = np.arange(0 if expanding else start - window, start)
            yield train, np.arange(start, min(start + refit_every, T))

    def out_of_sample(
        self,
        data: Any,
        window: int,
        *,
        refit_every: int = 1,
        expanding: bool = False,
        oos_loadings: str = "ols",
        **kwargs: Any,
    ) -> EngineResult:
        """Rolling/expanding out-of-sample evaluation of every model.

        ``oos_loadings`` selects how test returns are priced (see module docs).
        """
        if oos_loadings not in ("ols", "projection"):
            raise ValueError("oos_loadings must be 'ols' or 'projection'")
        data = self._coerce(data, **kwargs)
        splits = list(self.windows(data.T, window, refit_every, expanding))
        tasks = [
            (model, data, tr, te, oos_loadings) for _, model in self.models for tr, te in splits
        ]
        parts = self.executor.map(_window_task, tasks)
        n = len(splits)
        return EngineResult(
            [
                _aggregate(label, parts[i * n : (i + 1) * n])
                for i, (label, _) in enumerate(self.models)
            ]
        )

    def run(
        self,
        data: Any,
        *,
        window: int | None = None,
        refit_every: int = 1,
        expanding: bool = False,
        oos_loadings: str = "ols",
        **kwargs: Any,
    ) -> EngineResult:
        """In-sample results, plus out-of-sample results when ``window`` is given."""
        res = self.in_sample(data, **kwargs)
        if window is not None:
            oos = self.out_of_sample(
                data,
                window,
                refit_every=refit_every,
                expanding=expanding,
                oos_loadings=oos_loadings,
                **kwargs,
            )
            res.records.extend(oos.records)
        return res
