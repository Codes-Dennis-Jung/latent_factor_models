# Validation

This document records how the implementation was checked against the
literature and against the authors' code, the deliberate deviations, and the
errors found in the previous research scripts that this package replaces.

The papers themselves are not distributed with the package (copyright); see
`NOTICE` for full references.

## Numerical validation against the original sources

Every estimator was compared with the **original** code published by the
authors or, where no code exists, with the equations of the paper. Matches
are to machine precision (relative tolerance $10^{-8}$) unless noted.

| Model | Reference | What was compared | Result |
|---|---|---|---|
| Three-pass (Giglio-Xiu) | `PCAFM.m` from `Codes_RP.zip` (D. Xiu's website), literal line-by-line translation | $\hat\gamma_0$, $\hat\gamma_g$ with and without zero-beta rate, all asymptotic variances, $R^2_g$, cross-sectional $R^2$, Wald-test p-values, eigenvalues, $\hat p$ ($c=0.25, 0.5, 0.75$), for $p=1,2,3$ and $d=2$ | identical (variances: see deviations 1-2) |
| SPCA | `SPCA_cv.m` from `WF_Simu.zip`, literal translation | $\hat\gamma_g$, Avar($\hat\gamma_g$), $\hat\alpha$, Avar($\hat\alpha$), mimicking weights, SDF weights, selected asset sets, for $p=1,2,3$ and $d=2$ | identical (selection size: see deviation 3) |
| RP-PCA | `rpPCA_cv.m` from `WF_Simu.zip` (Giglio-Xiu-Zhang) | factor space, risk premia $\hat\eta\bar v$, SDF weights, for $\gamma\in\{-1,0,10\}$, $K\in\{1,3\}$ | identical |
| RP-PCA | `rppca` 0.1.1 (PyPI, independent implementation) | eigenvalues, factor space, max. Sharpe ratio, RMS $\alpha$ | identical |
| IPCA | original `ipca` 0.6.7 (bkelly-lab, PyPI) | restricted model: $\Gamma_\beta$, factors, fitted values, total $R^2$; with intercept: $\Gamma_\alpha$, $\Gamma_\beta$, factors, fitted values, $W_\alpha$; with an observed factor: $\Gamma_\delta$, span($\Gamma_\beta$), fitted values; out-of-sample factors vs. KPS §4.4 | identical |
| Target PCA (XS, TS, XS-TS) | Bryzgalova et al. (2026): eqs. (5)-(13), (16), Propositions 1-3 (no public code) | operators (6), (9), (12) against brute-force constructions with explicit $T\times T$ matrices and `scipy.linalg.sqrtm`; Prop. 1 $T\times T$ form; Prop. 3 weights $\hat\omega=(\frac{1}{\tilde\gamma+1}I+\frac{\tilde\gamma}{\tilde\gamma+1}P_\Lambda)Q_K$; optimality for objective (11); alpha target (10) gives $R'G/T=\alpha^{ref}$; XS-metric (not squared) and TS-metric (16) | identical |

Further checks in the test suite (`tests/`): Newey-West $\Pi$ matrices equal
the authors' loop, rotation invariance of the three-pass estimator,
Monte-Carlo coverage of 95% confidence intervals, recovery of the number of
factors, IPCA ALS steps against per-period loops, and equivalences between
nested models (RP-PCA with $\gamma=-1$ = covariance PCA, TS-target PCA with
$G=1_T$ = RP-PCA).

## Documented deviations and extensions

Where the original code and the paper differ, the package follows the paper.
None of these changes an estimator.

1. **Zero-beta variance** (Giglio-Xiu): $\hat\sigma^2_\alpha$ uses the divisor
   $N$ as in the paper; `PCAFM.m` uses $N-1$ (`var`). The results differ
   exactly by the factor $(N-1)/N$.
2. **Variance without zero-beta rate** (Giglio-Xiu): Theorem 1 evaluates $\Phi$
   at the step-2 estimate $\hat\gamma=(\hat\beta'\hat\beta)^{-1}\hat\beta'\bar r$
   (no intercept), and the package does exactly that. `PCAFM.m` reports that
   estimate but plugs the latent premia from the regression *with* intercept
   into `avarhat_nozero`. Both are asymptotically equivalent when the true
   zero-beta rate is zero. In a Monte Carlo with a correctly specified number
   of factors (300 samples), the standard errors agree to within 0.1% and the
   coverage is identical. Differences only appear when too few latent factors
   are used.
3. **SPCA selection size**: the paper selects the $N_0=\lfloor qN\rfloor$ most
   correlated assets. `SPCA_cv.m` uses `L >= bb(n-N0)` on 1-based sorted
   scores, which keeps $N_0+1$ assets. The package selects exactly $N_0$, so
   `SPCA(n_select=N0 + 1)` reproduces the authors' code exactly. The
   authors' code also defaults to a Newey-West lag $q=1$; set `lags=1` to
   reproduce it.
4. **Out-of-sample pricing loadings**: Bryzgalova et al. (eq. 13) price with the
   window's OLS betas, Lettau and Pelger with the eigenvector (projection)
   loadings. Both are available (`oos_loadings="ols"` / `"projection"`).
5. **IPCA $R^2$**: the package reports the uncentred $R^2$ of Kelly, Pruitt and
   Su (2019); the original package's `score()` returns scikit-learn's centred
   $R^2$. The fitted values are identical.
6. **IPCA normalisation order**: the package projects
   ($\Gamma_\beta'[\Gamma_\delta,\Gamma_\alpha]=0$) before rotating, so all
   identification constraints hold exactly. The original package rotates
   first. On the tested panels the estimates are identical; in general they
   can differ only by the identifying rotation, never in fitted values.
7. **SPCA tuning**: the paper (Appendix E) takes the median of the CV-optimal
   $N_0$ over 100 repetitions of random 3-fold CV. `GridSearchCV` with
   `KFold(3, shuffle=True, n_repeats=...)` maximises the mean score over all
   repetitions instead.
8. **SPCA Wald test**: `SPCA_cv.m` has no weak-factor test. The package adds the
   Giglio-Xiu (2021, Theorem 4) Wald test, which applies unchanged to the
   SPCA latent factors.
9. **`comove` time-series target**: a package extension, not in the paper.
10. **PCA centring**: Bryzgalova et al. formulate all estimators with uncentred
    returns (their footnote 4), so their PCA benchmark is `PCA(centered=False)`.
    Lettau and Pelger's PCA (and eq. 2) is covariance PCA, `PCA()` (default).

## Errors found in the previous research scripts

The original scripts (`TPCA/factor_models.py`, `RPPCA/mod_rppca.py`,
`SPCA/mod_spca.py`, `IPCA/ipca.py`, a fork of `bkelly-lab/ipca`) were
replaced by this package and are not published. The list below documents the
errors that were found in them and fixed.

### Target PCA script (`TPCA/factor_models.py`)
1. `predict()` (all target classes) renormalised out-of-sample factors to unit
   norm on the test sample, so their scale depended on the test length and did
   not match the training loadings.
2. Factor signs were flipped in `fit` without flipping the stored weights and
   were re-chosen from the *test* sample in `predict` (also in `PCA.predict`).
3. `PCA._compute_results` recomputed eigenvalues/eigenvectors from the test data
   in `predict`.
4. Loadings were estimated without intercept (TS, XS-TS) or as `R'F` (XS);
   the paper's eq. (13) uses time-series OLS with intercept.
5. Evaluation used factor-mimicking portfolios instead of the actual factors
   `R omega`, re-estimated alphas on test data (not out of sample) and used the
   test sample's own tangency weights for the Sharpe ratio.
6. Cross-validation sliced a full-sample time-series target (look-ahead) and
   ignored the `primary_metric` argument; `gamma = 0` was mapped to `None`.
7. No mean target (`G = 1_T`, i.e. RP-PCA); `kind="comove"` was attributed to
   the wrong equation.
8. Numerical: explicit `T x T` projectors and `W_TS`, `np.linalg.inv` of
   `G'G`, unnecessary eigen-decomposition for `W_XS^{1/2}` (closed form
   `I + gamma~ P_Lambda`).
9. `r_squared` divided a mean squared residual by the pooled panel variance.

### RP-PCA / three-pass script (`RPPCA/mod_rppca.py`)
1. `np.linalg.eig` on a symmetric matrix and only eigenvectors re-ordered:
   eigenvalues stayed unsorted.
2. Number-of-factors estimator: unsorted eigenvalues, a `(20,) + (20,1)`
   broadcast producing a 20x20 matrix, and an off-by-one `argmin - 1`
   (returned 18 instead of 3 in tests).
3. Zero-beta variance: `mean(beta).T @ mean(beta)` is a scalar, not the outer
   product `beta0 beta0'`.
4. `sigma_e` used `np.cov` along the wrong axis and a ratio of sums instead of
   the mean of per-asset variance ratios.
5. p-values used a t distribution with `df = n - 2 - indc` (degrees of freedom
   subtracted twice); the theory is asymptotically normal.
6. Wald test only for `d = 1`; Onatski test called with `r_max = K` (the
   current model size), which makes it meaningless.
7. Sign flips applied in place to a view of the returned factors; NumPy 2.x
   crashes on `float()` of size-1 arrays; NaNs zero-filled silently.

### SPCA script (`SPCA/mod_spca.py`)
1. Crashed for more than one observable factor (`COR[II]` flatten) and on
   NumPy 2.x (`s` shape `(1,)`).
2. Hedging R² pooled over factors instead of per factor.
3. RMS divided by the number of assets instead of the number of factors.
4. Out-of-sample Sharpe ratio always ~0 (returns were demeaned twice).
5. p-values with wrong distribution/degrees of freedom; t-stats divided by zero
   when variances were not requested.
6. Wald test skipped for `d >= 2`; `vhat_all` returned the wrong object.
7. CV helpers: stale index in exception handler, `ll - 1` indexing reading the
   last element, failures scored as best results, global RNG reseeding, an
   ad-hoc "combined score" that is order dependent and not in the paper.

### IPCA fork (`IPCA/ipca.py`)
1. Intercept together with pre-specified factors crashed (ones row appended twice).
2. Refit kept a stale `PSF`; bootstraps then used the wrong design.
3. `predictOOS` estimated the intercept/observed factors as free factors.
4. Block bootstrap left the last block un-resampled when `T % blocksize == 0`.
5. Bootstrap retry loops `while Gamma is None` could run forever.
6. `BS_Wbeta` tested a single entry (not rotation invariant) instead of the
   whole row of `Gamma_beta`; `BS_Wdelta` only the last observed factor.
7. Iteration count off by one and "convergence" reported when not converged;
   `iter_tol=10e-6` is `1e-5`.
8. `cond_mean_factor` used a window including the current factor (look-ahead).
9. Dead elastic-net panel fit (`np.hstack(generator)` fails on NumPy >= 1.24)
   silently ignored `alpha`/`l1_ratio`; unneeded numba/progressbar dependencies.
