# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-10-21

### Added
- Unified spectral engine for PCA, RP-PCA and XS-, TS- and XS-TS-target PCA
  (one operator, never forms `T x T` matrices).
- `SPCA` (supervised PCA) with Newey-West inference for risk premia and alphas.
- `IPCA` with a vectorised ALS, observed factors, intercept and parallel
  wild/block bootstrap tests (alpha, instrument, observed factors).
- `ThreePassRiskPremia` (Giglio-Xiu) with any first-pass latent factor model,
  optional zero-beta rate and weak-factor Wald tests.
- Asset-pricing metrics (Sharpe ratio, RMS alpha, unexplained variation,
  XS-/TS-metrics), number-of-factor estimators (Onatski, Giglio-Xiu,
  Bai-Ng, Ahn-Horenstein).
- Cross-validation (`KFold`, `TimeSeriesSplit`, `GridSearchCV`), the
  experiment `Engine` (in-sample and rolling out-of-sample) and the
  `Executor` (serial / thread / process backends, reproducible seeding).
- Synthetic data generators, the `lfm` command-line interface, documentation
  and runnable examples for every model.
- GitHub Actions CI: lint, type check, tests on Python 3.10-3.14 x
  Linux/Windows/macOS, minimum NumPy/SciPy, wheel smoke test.

### Changed
- Legacy research scripts moved to `archive/legacy`; see
  `archive/README.md` for the errors found and fixed.
