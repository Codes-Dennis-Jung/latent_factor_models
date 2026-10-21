# Validation

This document records how the implementation was checked against the
literature and against the authors' code, the deliberate deviations, and the
errors found in the previous research scripts that this package replaces.

The papers themselves are not distributed with the package (copyright); see
`NOTICE` for full references.

## Numerical validation against reference implementations

The new estimators were checked numerically against the authors' own code
where it exists:

* **Giglio-Xiu three-pass**: matches the authors' MATLAB `PCAFM.m`
  (from `Codes_RP.zip`, Dacheng Xiu's website) for risk premia, the zero-beta
  rate, all asymptotic variances (the paper's `sigma_alpha^2` uses `1/n`, the
  MATLAB code `1/(n-1)`; the package follows the paper) and the estimated
  number of factors.
* **SPCA**: matches a port of the authors' MATLAB `SPCA_cv.m` (`WF_Simu.zip`)
  for risk premia, their variance, alphas and their variance, mimicking
  portfolios, SDF weights and Wald statistics.
* **IPCA**: matches the reference ALS (`bkelly-lab/ipca`) for `Gamma`, factors
  and R² with and without intercept on an unbalanced panel.
* **RP-PCA**: matches the independent `rppca` package (PyPI) for eigenvalues,
  factor space, maximum Sharpe ratio and RMS alpha.
* **Target PCA** (no public code): verified against brute-force constructions
  of the paper's operators (explicit `T x T` matrices, matrix square roots),
  the equivalent `T x T` formulation of Proposition 1 and the weighted
  low-rank objective (11). See `tests/test_math_spectral.py`.

## Documented deviations and extensions

These are deliberate choices; none changes a paper's estimator.

* **Zero-beta variance** (Giglio-Xiu): $\hat\sigma^2_\alpha$ uses the divisor
  $N$ as in the paper; the authors' MATLAB code uses $N-1$.
* **Out-of-sample pricing loadings**: Bryzgalova et al. (eq. 13) price with the
  window's OLS betas, Lettau and Pelger with the eigenvector (projection)
  loadings. Both are available (`oos_loadings="ols"` / `"projection"`).
* **`comove` time-series target**: a package extension, not in the paper.
* **SPCA tuning**: the paper (Appendix E) takes the median of the CV-optimal
  $N_0$ over 100 repetitions of random 3-fold CV; `GridSearchCV` with
  `KFold(3, shuffle=True, n_repeats=...)` maximises the mean score over all
  repetitions instead. The SPCA estimator and its inference are exact.
* **IPCA normalisation order**: the projection
  $\Gamma_\beta'[\Gamma_\delta,\Gamma_\alpha]=0$ is applied before the
  rotation, so all identification constraints hold exactly; the public
  reference code rotates first. Results coincide without observed factors.

## Errors found in the previous research scripts

The original scripts (`TPCA/factor_models.py`, `RPPCA/mod_rppca.py`,
`SPCA/mod_spca.py`, `IPCA/ipca.py`, a fork of `bkelly-lab/ipca`) were
removed from the working tree; they remain available in the git history
(commit `fc9a6bc`, folder `archive/legacy`).

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
