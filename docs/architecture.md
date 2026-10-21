# Architecture

This document explains how the package is organised, which commonalities of
the original research code it factors out, and how to extend it.

## 1. The key observation: one spectral operator

Five of the seven estimators are the *same* estimator with different
weighting matrices. Each one takes the top-$K$ eigenvectors of

$$
M \;=\; \frac{1}{NT}\; C\, R' \,\bigl(I_T + \gamma_{TS} P_G\bigr)\, R\, C,
\qquad C = I_N + \tilde\gamma\, P_\Lambda,\qquad
\tilde\gamma = \sqrt{\gamma_{XS} N/L + 1} - 1 .
$$

Here $R$ is the $T\times N$ matrix of *uncentred* excess returns, $P_G$ is the
projector onto a time-series target $G$ ($T\times J$), and $P_\Lambda$ is the
projector onto a cross-sectional target $\Lambda$ ($N\times L$). Because
$P_\Lambda$ is idempotent, $C^2 = I_N + \gamma_{XS}\tfrac{N}{L}P_\Lambda = W_{XS}$:
$C$ is the closed-form square root of the cross-sectional weighting matrix.

| Model | $G$ | $\gamma_{TS}$ | $\Lambda$ | Class |
|---|---|---|---|---|
| PCA (covariance) | $1_T$ | $-1$ | – | `PCA` |
| PCA (second moment) | – | 0 | – | `PCA(centered=False)` |
| RP-PCA | $1_T$ | $\gamma$ | – | `RPPCA` |
| TS-target PCA | $G_{TS}$ (alpha, mean, …) | $\gamma_{TS}$ | – | `TSTargetPCA` |
| XS-target PCA | – | 0 | $\Lambda_{XS}$ | `XSTargetPCA` |
| XS-TS-target PCA | $G_{TS}$ | $\gamma_{TS}$ | $\Lambda_{XS}$ | `XSTSTargetPCA` |

`SpectralFactorModel` (in `models/spectral.py`) implements the operator once:

* $R'W_{TS}R = R'R + \gamma_{TS}(Q_G'R)'(Q_G'R)$, with $Q_G$ an orthonormal
  basis of $\operatorname{col}(G)$. **No $T\times T$ matrix is ever formed.**
* $C$ is applied as $X + \tilde\gamma\, Q_\Lambda(Q_\Lambda'X)$, with
  rank-revealing bases. There are no explicit inverses of $G'G$ or $\Lambda'\Lambda$.
* Weights are $\hat\omega = C Q_K/(1+\tilde\gamma)$, factors are $F = R\hat\omega$
  (so they are portfolios and keep their means), and loadings are
  time-series OLS with intercept (Bryzgalova et al., eq. 13).
* Signs are fixed **once**, at fit time, by making factor means positive.
  Weights, eigenvectors and factors are flipped together, so out-of-sample
  factors use exactly the training weights.

A concrete model only implements `_spec(data) -> SpectralSpec`, which says
which targets and weights to use. Data-dependent targets (the alpha target,
the macro target) are built **inside** `_spec` from the training data. Every
cross-validation fold and rolling window therefore rebuilds them without
look-ahead.

SPCA (iterative screening and deflation) and IPCA (alternating least squares
on characteristic-managed portfolios) do not fit this mould. They share
everything else.

## 2. Layers

```
latent_factor_models/
├── core/            data containers, base classes, registry, linear algebra
│   ├── data.py          FactorData (T x N returns + optional targets/factors/macro)
│   │                    CharacteristicPanel (T x N x L, unbalanced, for IPCA)
│   ├── base.py          LatentFactorModel  -> params, clone, fit template
│   │                    PortfolioFactorModel -> transform/residuals/evaluate
│   ├── registry.py      name -> class registry (used by the engine and CLI)
│   ├── linalg.py        top_eigh, orthonormal_basis, ols, solve_psd, ...
│   └── exceptions.py
├── models/          estimators
│   ├── spectral.py      SpectralFactorModel (the operator above)
│   ├── pca.py, rppca.py, target_pca.py   thin specifications
│   ├── targets.py       mean/alpha/comove TS targets, macro/shape XS targets, VAR
│   ├── spca.py          supervised PCA
│   └── ipca.py          instrumented PCA + bootstrap tests
├── inference/       HAC (Newey-West) and Giglio-Xiu risk-premium inference
├── evaluation/      Sharpe ratio, RMS alpha, unexplained variation, XS/TS metrics
├── selection/       number of factors, KFold/TimeSeriesSplit, GridSearchCV
├── engine/          Executor (parallel backends) and Engine (experiments)
├── simulation/      synthetic data with known ground truth
└── cli.py           `lfm` command line
```

Dependencies only point downwards: `core` depends on NumPy and SciPy only;
models depend on `core`; `inference`, `evaluation` and `selection` are shared
services; `engine` and `cli` sit on top. The only runtime dependencies are
`numpy` and `scipy`. `pandas` is optional (DataFrames are accepted and their
labels kept).

## 3. Class hierarchy

```
LatentFactorModel (ABC)            get_params / set_params / clone, fit() template,
│                                  _coerce (data container), requires (fields)
├── PortfolioFactorModel           factors are portfolios F = R @ weights_
│   │                              transform, fitted_values, residuals, evaluate,
│   │                              projection_loadings, oos_loadings
│   ├── SpectralFactorModel (ABC)  the unified operator; spectrum, absorption_ratio,
│   │   │                          variance_signal, target_metrics
│   │   ├── PCA
│   │   ├── RPPCA
│   │   ├── XSTargetPCA
│   │   ├── TSTargetPCA
│   │   └── XSTSTargetPCA
│   └── SPCA                       + risk_premia(), hedging_r2()
└── IPCA                           time-varying betas; transform/predict/evaluate
                                   on CharacteristicPanel; bootstrap tests
```

`fit()` is a template method. It validates parameters, coerces the input into
the model's data container, checks required fields (`requires`), calls the
model's `_fit`, and marks the model fitted. Every model therefore accepts NumPy
arrays, pandas DataFrames or a prepared container, and fails early with a
`DataError` if data are missing or malformed.

Fitted attributes end in an underscore (`factors_`, `weights_`, `loadings_`,
`intercepts_`, `eigenvalues_`, …). Constructor arguments are plain
attributes, so `clone()` creates a fresh copy for every CV fold, window or
bootstrap draw.

## 4. Shared services

| Concern | Implementation | Used by |
|---|---|---|
| Newey-West long-run covariance | `inference.hac.long_run_covariance` (vectorised over lags) | three-pass, SPCA |
| $\Pi_{11},\Pi_{12},\Pi_{22},\Pi_{13},\Pi_{33}$ | blocks of one HAC matrix of $[\mathrm{vec}(z_tv_t'), v_t, z_t]$ | three-pass, SPCA |
| Avar of $\hat\gamma_g$, $\hat\alpha$, Wald test | `inference.three_pass` | three-pass, SPCA |
| Sharpe ratio, RMS $\alpha$, $\bar\sigma^2_e$, XS/TS metrics | `evaluation.metrics` | all return-based models, engine |
| Number of factors | `selection.n_factors` | any spectrum |
| Cross-validation | `selection.cv` (splitters, scorers, grid search) | any model |
| Parallelism, seeding | `engine.parallel.Executor`, `spawn_seeds` | CV, engine, IPCA bootstrap |

The legacy code had two near-identical $O(T q\, d^2p^2)$ Python loops for the
$\Pi$ matrices, in the RP-PCA and SPCA files. They are now a single
vectorised call.

## 5. Parallel execution

All embarrassingly parallel work goes through `Executor.map`:

* **grid search**: (parameter combination × fold) tasks;
* **engine**: (model × window) tasks for rolling out-of-sample studies, and
  model tasks for in-sample runs;
* **IPCA bootstrap**: one task per draw.

Backends are `"serial"`, `"thread"` and `"process"`, and `n_jobs=-1` uses all
cores. Randomness is reproducible and independent of the backend: each task
receives a child `SeedSequence` from `spawn_seeds(seed, n)`. Tests check that
thread and process runs give identical numbers to serial runs.

Practical notes:

* Use `"process"` for many medium-sized tasks, such as rolling windows or grids.
  Use `"thread"` when each task is dominated by large BLAS calls, which release
  the GIL. On Windows/macOS, protect scripts with `if __name__ == "__main__":`.
* NumPy's BLAS is itself multi-threaded. With many processes, set
  `OMP_NUM_THREADS=1` (or `MKL_NUM_THREADS=1`) before starting Python to avoid
  oversubscription.

## 6. Extending the package

**A new spectral variant** needs about 15 lines:

```python
from latent_factor_models.core.registry import register
from latent_factor_models.models.spectral import SpectralFactorModel, SpectralSpec

@register("my-model")
class MyTargetPCA(SpectralFactorModel):
    name = "my_target_pca"

    def __init__(self, n_factors: int = 1, gamma: float = 5.0) -> None:
        self.n_factors = n_factors
        self.gamma = gamma

    def _spec(self, data):
        G = build_my_target(data)            # from the training data only
        return SpectralSpec(ts_target=G, gamma_ts=self.gamma)
```

It immediately works with `evaluate`, `GridSearchCV`, `Engine`, the CLI
(`lfm fit my-model ...`) and the parallel backends.

**A non-spectral model** subclasses `PortfolioFactorModel` (if its factors are
portfolios of the test assets) or `LatentFactorModel`, and implements `_fit`.
`_fit` must set the fitted attributes listed in the base-class docstring.
