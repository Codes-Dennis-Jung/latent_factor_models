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

* Giglio-Xiu three-pass and SPCA reproduce the authors' MATLAB code
  (`PCAFM.m`, `SPCA_cv.m`) exactly.
* IPCA reproduces the reference ALS of `bkelly-lab/ipca`.
* RP-PCA matches the independent `rppca` package.
* Target PCA (no public code) is verified against brute-force constructions
  of the paper's operators, the equivalent $T\times T$ formulation and the
  weighted objective.
* 69 tests, including Monte-Carlo coverage of confidence intervals, `ruff`
  (incl. numpy docstrings) and `mypy` clean.

[docs/validation.md](docs/validation.md) records every check, the documented
deviations, and the errors found in (and fixed relative to) the original
research scripts.

## Documentation

Start at [docs/index.md](docs/index.md): one document per model (technical
description, methodology, implementation, example), plus inference,
evaluation/engine and CLI. Runnable examples: `examples/01_pca.py` …
`examples/08_engine_parallel.py`.

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
