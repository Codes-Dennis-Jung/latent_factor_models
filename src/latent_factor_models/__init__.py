# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Dennis Jung
"""latent_factor_models - latent asset-pricing factor models in one framework.

Estimators
----------
PCA, RPPCA                      statistical / risk-premium PCA
XSTargetPCA, TSTargetPCA,
XSTSTargetPCA                   target PCA (Bryzgalova, DeMiguel, Li, Pelger)
SPCA                            supervised PCA (Giglio, Xiu, Zhang)
IPCA                            instrumented PCA (Kelly, Pruitt, Su)

Infrastructure
--------------
ThreePassRiskPremia             Giglio-Xiu risk premia with any first-pass model
GridSearchCV, KFold, ...        cross-validation (parallel)
Engine, ModelSpec, Executor     multi-model in/out-of-sample runs (parallel)
simulate_*                      synthetic data with known ground truth
"""

from ._version import __version__
from .core import (
    CharacteristicPanel,
    DataError,
    FactorData,
    LatentFactorModel,
    NotFittedError,
    PortfolioFactorModel,
    available_models,
    create_model,
)
from .engine import Engine, EngineResult, Executor, ModelSpec
from .inference import RiskPremiaResult, ThreePassRiskPremia
from .models import (
    IPCA,
    PCA,
    RPPCA,
    SPCA,
    TSTargetPCA,
    XSTargetPCA,
    XSTSTargetPCA,
)
from .selection import GridSearchCV, KFold, TimeSeriesSplit, cross_validate
from .simulation import simulate_factor_returns, simulate_ipca_panel

__all__ = [
    "IPCA",
    "PCA",
    "RPPCA",
    "SPCA",
    "CharacteristicPanel",
    "DataError",
    "Engine",
    "EngineResult",
    "Executor",
    "FactorData",
    "GridSearchCV",
    "KFold",
    "LatentFactorModel",
    "ModelSpec",
    "NotFittedError",
    "PortfolioFactorModel",
    "RiskPremiaResult",
    "TSTargetPCA",
    "ThreePassRiskPremia",
    "TimeSeriesSplit",
    "XSTSTargetPCA",
    "XSTargetPCA",
    "__version__",
    "available_models",
    "create_model",
    "cross_validate",
    "simulate_factor_returns",
    "simulate_ipca_panel",
]
