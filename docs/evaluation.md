# Evaluation, model selection and the engine

## Metrics (`latent_factor_models.evaluation`)

All return-based models share one definition of each metric, following
Lettau and Pelger (2020) and Bryzgalova et al. (2026):

| metric | definition | key |
|---|---|---|
| Sharpe ratio | realised $\operatorname{mean}/\operatorname{sd}$ of the SDF portfolio $F b$, $b=\Sigma_F^{-1}\mu_F$ from the training sample | `sharpe_ratio` |
| RMS pricing error | $\sqrt{\frac1N\sum_i\alpha_i^2}$ with $\alpha_i=\overline{e_i}$, $e_t=R_t-\beta F_t$ | `rms_alpha` |
| Unexplained variation | $\frac1N\sum_i\sigma^2(e_i)/\sigma^2(R_i)$ | `unexplained_variation` |
| Total $R^2$ | $1-\sum(R-F\beta')^2/\sum R^2$ (uncentred) | `total_r2` |
| XS-metric | $100\lVert P_\Lambda\omega\rVert/\lVert\omega\rVert$ | `target_metrics()` |
| TS-metric | $100\lVert P_GRP_\omega\rVert_F/\lVert P_GR\rVert_F$ | `target_metrics()` |
| Generalised correlation | canonical correlations of estimated and true factors | `generalized_correlation` |

**In sample**, $\beta$ are time-series OLS betas with intercept, so the
$\alpha_i$ are exactly the OLS intercepts.

**Out of sample**, weights, betas and tangency weights are fixed from the
training sample. `evaluate(R_test)` and the engine support two conventions
for pricing new returns:

* `loadings="ols"` (default): the training window's OLS betas (Bryzgalova et
  al., eq. 13);
* `loadings="projection"`: the projection loadings $\omega(\omega'\omega)^{-1}$,
  i.e. Lettau and Pelger's "cross-sectional pricing regression with
  out-of-sample loadings".

For covariance PCA the two coincide.

IPCA reports total and predictive $R^2$ for individual assets and managed
portfolios instead (Kelly, Pruitt and Su, 2019).

## Number of factors (`latent_factor_models.selection`)

| function | method |
|---|---|
| `onatski(eigs, r_max)` | Onatski (2010) edge distribution. $r_{\max}$ is an upper bound, not the model's $K$ (a legacy bug). |
| `giglio_xiu(eigs, N, T, c=0.5)` | Giglio-Xiu (2021) penalised eigenvalues |
| `bai_ng(R, k_max, "IC2")` | Bai-Ng (2002) information criteria |
| `eigenvalue_ratio(eigs)` | Ahn-Horenstein (2013) |
| `covariance_eigenvalues(R)` | spectrum of $\bar R'\bar R/(NT)$ |

Each estimator is tested to recover $k\in\{1,3,5\}$ on simulated spiked spectra.

## Cross-validation

* `KFold(n_splits, shuffle=False)`: contiguous blocks (Lettau-Pelger, target PCA).
  `shuffle=True, n_repeats=m` gives repeated random folds (SPCA, Appendix E).
* `TimeSeriesSplit(n_splits, test_size, gap, rolling)`: expanding or rolling
  training windows with an optional purge gap.
* `GridSearchCV(model, grid, cv, scoring)`: scorers are `"sharpe"`,
  `"neg_rms_alpha"`, `"neg_unexplained_variation"`, `"total_r2"` (also IPCA),
  `"hedging_r2"` (SPCA), or any callable `score(fitted_model, test_data)`.
  Higher is better.

Training windows are sliced from the data container. Data-dependent targets
(alpha target, macro target) are therefore rebuilt from training data only,
and there is no look-ahead. A failing fold returns NaN and does not count as
the best result, unlike in the legacy CV helpers.

## Engine

```python
eng = lfm.Engine(
    [lfm.ModelSpec("pca", {"n_factors": 5}),
     lfm.ModelSpec("rppca", {"n_factors": 5, "gamma": 10}),
     {"name": "ts_target_pca", "params": {"n_factors": 5, "gamma_ts": 15}}],
    executor="process", n_jobs=-1)
res = eng.run(data, window=240, refit_every=1, oos_loadings="ols")
res.to_frame()          # pandas DataFrame (if pandas is installed)
```

* `in_sample(data)`: fits every model on the full sample.
* `out_of_sample(data, window, refit_every, expanding)`: for every test
  period, the model is estimated on the preceding window. Results are pooled
  over all test periods. Each (model, window) pair is a parallel task.
  `refit_every=k` re-estimates every `k` periods; in between, weights are held
  fixed, which keeps the evaluation out of sample.
* The same engine handles IPCA (`CharacteristicPanel` input), reporting pooled
  out-of-sample total and predictive $R^2$.

A test checks the one-step-ahead engine output against a hand-written rolling
loop.
