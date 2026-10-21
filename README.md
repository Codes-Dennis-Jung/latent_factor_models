# latent-factor-models

A modular, tested Python framework for latent asset-pricing factor models:

| Model | Class | Reference |
|---|---|---|
| PCA | `PCA` | – |
| Risk-premium PCA | `RPPCA` | Lettau & Pelger (2020, JEcon; RFS) |
| Supervised PCA | `SPCA` | Giglio, Xiu & Zhang (JF) |
| Instrumented PCA | `IPCA` | Kelly, Pruitt & Su (2019, JFE) |
| XS-target PCA | `XSTargetPCA` | Bryzgalova, DeMiguel, Li & Pelger (2026) |
| TS-target PCA | `TSTargetPCA` | Bryzgalova, DeMiguel, Li & Pelger (2026) |
| XS-TS-target PCA | `XSTSTargetPCA` | Bryzgalova, DeMiguel, Li & Pelger (2026) |

The package also includes Giglio-Xiu (2021) three-pass risk-premium
inference, Newey-West HAC, number-of-factor estimators, time-series
cross-validation, a parallel experiment engine, synthetic data generators and
a command-line interface. The only runtime dependencies are NumPy and SciPy.

## Installation

```bash
pip install .            # or: pip install -e ".[dev]" for development
```

Supported: Python 3.10-3.14 with NumPy >= 1.23 and SciPy >= 1.9 (tested on all five
Python versions and on the minimum NumPy/SciPy versions; CI runs Linux, Windows and macOS).

## Quick start

```python
import latent_factor_models as lfm

sim = lfm.simulate_factor_returns(T=600, N=100, seed=0)      # known ground truth

model = lfm.RPPCA(n_factors=3, gamma=10).fit(sim.returns)    # arrays or DataFrames
model.factors_, model.weights_, model.loadings_
model.evaluate()                       # Sharpe ratio, RMS alpha, unexplained variation
model.evaluate(sim.returns[400:])       # evaluate other data with fixed weights

# all estimators share one interface
lfm.TSTargetPCA(3, gamma_ts=15).fit(sim.returns, factors=sim.reference)
lfm.XSTargetPCA(3, gamma_xs=0.01).fit(sim.returns, xs_target=sim.xs_target)
lfm.SPCA(3, n_select=50).fit(sim.returns, factors=sim.observable).risk_premia()
lfm.IPCA(2, intercept=True).fit(lfm.simulate_ipca_panel(seed=0).panel).test_alpha(500)

# tuning and comparison, in parallel
lfm.GridSearchCV(lfm.RPPCA(3), {"gamma": [-1, 0, 10, 20]}, cv=5,
                 executor="process", n_jobs=-1).fit(sim.returns)
lfm.Engine([lfm.PCA(3), lfm.RPPCA(3)]).run(sim.returns, window=240)
```

Command line:

```bash
lfm models -v
lfm simulate -o synthetic_data
lfm fit rppca --returns synthetic_data/returns.csv -p n_factors=3 -p gamma=10
lfm run examples/config_example.json --backend process
```

## Outputs

Every estimator stores its results as attributes ending in `_` and offers
methods for new data. `pandas` labels (dates, assets) are kept when DataFrames
are passed.

**Return-based models** (`PCA`, `RPPCA`, `XSTargetPCA`, `TSTargetPCA`,
`XSTSTargetPCA`, `SPCA`)

| Output | Shape | Content |
|---|---|---|
| `weights_` | N × K | factor-portfolio weights ω̂ (factors are portfolios of the assets) |
| `factors_` | T × K | in-sample factor returns F = R ω̂ (with means = risk premia) |
| `loadings_` | N × K | time-series OLS betas |
| `intercepts_` | N | in-sample pricing errors (alphas) |
| `eigenvalues_` | K | leading eigenvalues of the model's operator |
| `transform(R)`, `fitted_values(R)`, `residuals(R)` | | factors / systematic / idiosyncratic parts for any returns |
| `evaluate(R)` | dict | `sharpe_ratio`, `rms_alpha`, `unexplained_variation`, `total_r2` (in or out of sample) |
| `projection_loadings()`, `oos_loadings(kind)` | N × K | pricing loadings (Lettau-Pelger / OLS) |

Spectral models add `eigenvectors_`, `operator_`, `operator_trace_`,
`spectrum()`, `absorption_ratio()`, `variance_signal()` and — for target
models — `target_metrics()` (XS-metric per factor and for the SDF, TS-metric).

**SPCA** adds `risk_premia(p)` (risk premia, standard errors, t-stats,
p-values, alphas and their standard errors, weak-factor Wald tests),
`latent_factors_`, `eta_`, `gamma_`, `gamma_g_`, `alpha_g_`, `selected_`
(assets per step), `mimicking_weights_`, `sdf_weights_`, `hedging_r2_` and
`hedging_r2(R, G)` out of sample.

**IPCA**

| Output | Shape | Content |
|---|---|---|
| `gamma_beta_`, `gamma_alpha_`, `gamma_delta_` | L × K, L, L × M | characteristics → betas / alpha / observed-factor loadings |
| `factors_` | T × K | latent factor returns |
| `loadings(panel)` | T × N × K | time-varying betas |
| `predict(panel, factors="realized" \| "mean")` | T × N | fitted / predicted returns |
| `transform(panel)` | T' × K | out-of-sample factors |
| `evaluate(panel)` | dict | total and predictive R², assets and managed portfolios |
| `test_alpha()`, `test_instrument(l)`, `test_observed_factors()` | `BootstrapResult` | statistic, p-value, bootstrap draws |
| `n_iter_`, `converged_` | | ALS diagnostics |

**Inference, selection and engine**

| Object | Output |
|---|---|
| `ThreePassRiskPremia(...).fit(R, G)` | `RiskPremiaResult`: `gamma_g`, `se`, `tstat`, `pvalue`, `eta`, `gamma`, `r2_g`, `wald_stat`, `wald_pvalue`, `gamma_0`, `gamma_0_se`, `r2_cs`, `summary()` |
| `GridSearchCV(...).fit(...)` | `best_params_`, `best_score_`, `best_model_`, `result_.to_records()` (all grid points × folds) |
| `Engine(...).run(...)` | `EngineResult` with one record per model and sample (`to_frame()` → DataFrame) |
| `selection.onatski`, `giglio_xiu`, `bai_ng`, `eigenvalue_ratio` | estimated number of factors |
| `lfm fit … -o DIR` | `summary.json`, `factors.csv`, `weights.csv`, `loadings.csv` |
| `lfm run config.json` | results table (stdout) and CSV |

## Design in one paragraph

PCA, RP-PCA and the three target-PCA estimators are the same estimator: the
top-$K$ eigenvectors of
$\frac1{NT}C R'(I_T+\gamma_{TS}P_G)RC$ with $C=I+\tilde\gamma P_\Lambda$. They
are implemented once (`SpectralFactorModel`), never form $T\times T$
matrices, and differ only in a 3-line specification of their targets. SPCA
and IPCA have their own estimation loops. All models share the base classes,
data containers, metrics, Newey-West inference, cross-validation and the
parallel `Executor`. See [docs/architecture.md](docs/architecture.md).

## Validation

* Compared with the **original** code: Giglio-Xiu `PCAFM.m`, SPCA
  `SPCA_cv.m`, RP-PCA `rpPCA_cv.m` (Giglio-Xiu-Zhang) and the original
  `ipca` package (bkelly-lab), with identical results. The few places where
  the authors' code departs from their paper follow the paper and are
  documented in [docs/validation.md](docs/validation.md).
* RP-PCA also matches the independent `rppca` package.
* Target PCA (no public code) is verified against brute-force constructions
  of the paper's operators, the equivalent $T\times T$ formulation and the
  weighted objective.
* 86 tests (incl. executing all notebooks), Monte-Carlo coverage of confidence intervals, `ruff`
  (incl. numpy docstrings) and `mypy` clean.

[docs/validation.md](docs/validation.md) records every check, the documented
deviations, and the errors found in (and fixed relative to) the original
research scripts.

## Documentation

| Document | Content |
|---|---|
| [TECHNICAL_DESCRIPTION.md](TECHNICAL_DESCRIPTION.md) | the mathematics: every estimator, identity, test statistic and algorithm |
| [METHODOLOGY.md](METHODOLOGY.md) | the process: from data to results, choosing a model, tuning, inference, validation |
| [docs/](docs/index.md) | one document per model (technical description, methodology, example), inference, evaluation/engine, CLI |
| [docs/validation.md](docs/validation.md) | comparison with the authors' original code and the papers, documented deviations |

## Example notebooks

Ten executed Jupyter notebooks in [`examples/`](examples/) explain each model
with its specification, parameters, options, outputs, plots and pitfalls, on
synthetic data with known ground truth:

| Notebook | Content |
|---|---|
| [`00_getting_started`](examples/00_getting_started.ipynb) | installation, data formats, common API, all outputs, parallelism, CLI |
| [`01_pca`](examples/01_pca.ipynb) | PCA, number of factors, factor recovery, variants |
| [`02_rppca`](examples/02_rppca.ipynb) | RP-PCA: effect of γ, weak-factor detection, CV, rolling OOS |
| [`03_spca`](examples/03_spca.ipynb) | SPCA: screening CV, risk premia, alphas, Wald tests, hedging |
| [`04_ipca`](examples/04_ipca.ipynb) | IPCA: panels, Γ, bootstrap tests, observed factors, OOS R² |
| [`05_xs_target_pca`](examples/05_xs_target_pca.ipynb) | XS-target PCA: shape and macro targets, XS-metric |
| [`06_ts_target_pca`](examples/06_ts_target_pca.ipynb) | TS-target PCA: alpha/mean/custom targets, TS-metric |
| [`07_xsts_target_pca`](examples/07_xsts_target_pca.ipynb) | XS-TS-target PCA: 2-D parallel cross-validation |
| [`08_engine_parallel`](examples/08_engine_parallel.ipynb) | engine, backends, config files, CLI, custom models |
| [`09_risk_premia_inference`](examples/09_risk_premia_inference.ipynb) | Giglio-Xiu three-pass inference, coverage check |

```bash
pip install -e ".[notebooks]"
jupyter lab examples/
```

## Development

```bash
pip install -e ".[dev]"
pytest                    # tests + doctests
ruff check src tests examples && ruff format --check src tests examples
mypy
python -m build           # sdist + wheel
```

## License

Licensed under either of

* MIT License ([LICENSE-MIT](LICENSE-MIT)), or
* Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE)),

at your option. Unless you explicitly state otherwise, any contribution
intentionally submitted for inclusion in this work shall be dual licensed as
above, without any additional terms or conditions.

Method credits and full references of the underlying papers are in
[NOTICE](NOTICE).
