# IPCA — instrumented principal components analysis

`latent_factor_models.IPCA` · registry names `ipca`, `instrumented-pca` · notebook [`examples/04_ipca.ipynb`](../../examples/04_ipca.ipynb)

## Technical description

IPCA (Kelly, Pruitt and Su, 2019) lets factor loadings vary over time and
across assets as linear functions of observable characteristics $z_{i,t}$,
such as size, value or momentum. Characteristics thus act as *instruments* for
covariances: an asset's beta is $\beta_{i,t}=z_{i,t}'\Gamma_\beta$.
The model works on large, unbalanced panels of individual assets. It
estimates latent factors and the mapping $\Gamma_\beta$ jointly, and it tests
whether characteristics carry alpha beyond their role as betas
($\Gamma_\alpha=0$).

## Methodology

$$
r_{i,t} = z_{i,t}'\Gamma_\alpha + z_{i,t}'\Gamma_\beta f_t + z_{i,t}'\Gamma_\delta g_t + \varepsilon_{i,t},
$$

The model has $L$ instruments, $K$ latent factors $f_t$, and optional observed
factors $g_t$ (columns of $\Gamma_\delta$). The intercept is treated as an
observed factor equal to one.
With $\tilde\Gamma=[\Gamma_\beta,\Gamma_\delta,\Gamma_\alpha]$,
$\tilde f_t=[f_t; g_t; 1]$ and the **managed portfolios**
$X_t = Z_t'r_t/N_t$, $W_t = Z_t'Z_t/N_t$, alternating least squares iterates

$$
f_t = (\Gamma_\beta'W_t\Gamma_\beta)^{-1}\Gamma_\beta'\bigl(X_t - W_t[\Gamma_\delta,\Gamma_\alpha][g_t;1]\bigr),
\qquad
\operatorname{vec}(\tilde\Gamma') = \Bigl[\sum_t N_t\,W_t\otimes\tilde f_t\tilde f_t'\Bigr]^{-1}\sum_t N_t\,X_t\otimes\tilde f_t .
$$

The $N_t$ weights give every stock-period equal weight (KPS eq. 5).
`weighting="periods"` gives every period equal weight instead (footnote 16).

**Identification**, enforced after every iteration (projection first, then
rotation):

* $\Gamma_\beta'[\Gamma_\delta,\Gamma_\alpha]=0$;
* $\Gamma_\beta'\Gamma_\beta=I_K$;
* $\sum_t f_tf_t'$ diagonal with descending entries;
* positive factor means.

**Initialisation**: SVD of the $L\times T$ matrix of managed portfolios.
**Convergence**: maximum absolute change of $\Gamma$ and $F$ below `tol`
(KPS use $10^{-6}$). Non-convergence raises a `ConvergenceWarning` and sets
`converged_=False`.

**Fit measures** (uncentred), both for individual assets and for managed
portfolios:

* Total $R^2 = 1-\sum(r - z'(\Gamma_\alpha+\Gamma_\beta f_t))^2/\sum r^2$.
* Predictive $R^2$: replaces $f_t$ by its time-series mean $\lambda$.

**Out of sample** (KPS §4.4): with $\Gamma$ estimated through $t$, the factor
at $t+1$ comes from the cross-sectional regression
$f_{t+1}=(\Gamma_\beta'W_{t+1}\Gamma_\beta)^{-1}\Gamma_\beta'(X_{t+1}-\dots)$.
The intercept and observed factors are held fixed, not re-estimated.

**Bootstrap tests** (KPS §3, with the corrigendum's unit-variance multipliers):

| method | $H_0$ | statistic | model |
|---|---|---|---|
| `test_alpha` | $\Gamma_\alpha=0$ | $\Gamma_\alpha'\Gamma_\alpha$ | `intercept=True` |
| `test_instrument(l)` | row $l$ of $\Gamma_\beta$ is zero (all $K$ columns; a list tests rows jointly) | $\sum_k\Gamma_{\beta,lk}^2$ | `intercept=False` |
| `test_observed_factors` | $\Gamma_\delta=0$ (all observed factors) | $\lVert\Gamma_\delta\rVert^2$ | observed factors, no intercept |

Each draw resamples the managed-portfolio residuals
$\hat d_t = X_t - W_t\tilde\Gamma\tilde f_t$ with a wild (block) bootstrap
using Student-$t_5$ multipliers scaled to unit variance. It generates data
under $H_0$, re-estimates the unrestricted model, and records the statistic.
The p-value is the share of draws exceeding the sample statistic. Draws are
independent tasks of the `Executor`, so they run in parallel and are
reproducible given `seed`.

## Implementation notes

* Data: a `CharacteristicPanel` ($T\times N$ returns with NaN for missing,
  $T\times N\times L$ instruments). It can be built from arrays,
  `CharacteristicPanel.from_long(df, entity, time, target, chars)` or the
  legacy stacked array `[id, t, y, z...]`. **The instruments in row $t$ must
  already be lagged** relative to the returns in row $t$.
* Both ALS steps are vectorised over time (`einsum`, batched `solve`). The
  legacy code looped over periods and depended on numba, joblib, sklearn and
  progressbar, none of which are needed now.
* Verified against the original `ipca` package 0.6.7 (bkelly-lab, MIT):
  identical $\Gamma_\beta$, $\Gamma_\alpha$, $\Gamma_\delta$, factors and fitted
  values for the restricted model, with intercept and with an observed
  factor. The original's `score()` reports a centred $R^2$; the package
  reports the paper's uncentred $R^2$.
* Fixed relative to the legacy fork: intercept + observed factors crashed,
  `predictOOS` re-estimated the intercept as a free factor, the block bootstrap
  skipped the last block, retry loops could run forever, `BS_Wbeta` tested a
  single (rotation-dependent) entry, and a refit left stale state.

## Example

```python
import latent_factor_models as lfm

sim = lfm.simulate_ipca_panel(T=200, N=300, L=6, K=2, missing=0.1, irrelevant=1, seed=0)
model = lfm.IPCA(n_factors=2).fit(sim.panel)
model.evaluate()                         # total / predictive R², assets and portfolios
model.loadings()                         # (T, N, K) time-varying betas

unrestricted = lfm.IPCA(2, intercept=True).fit(sim.panel)
unrestricted.test_alpha(n_draws=1000, seed=1, executor="process", n_jobs=-1)
model.test_instrument(0, n_draws=1000, seed=2)       # irrelevant instrument -> large p

lfm.Engine([lfm.IPCA(2)]).out_of_sample(sim.panel, window=120, expanding=True)
```

## Parameters

| name | default | meaning |
|---|---|---|
| `n_factors` | 1 | latent factors $K$ (0 allowed with intercept/observed factors) |
| `intercept` | `False` | estimate $\Gamma_\alpha$ |
| `max_iter`, `tol` | 10000, 1e-6 | ALS stopping rule |
| `weighting` | `"observations"` | equal weight per stock-period or per period |

## References

* Kelly, B., Pruitt, S. and Su, Y. (2019). Characteristics are covariances: a unified model of risk and return. *JFE* 134, 501–524 (NBER WP 24540), and corrigendum (2024).
* Kelly, B., Pruitt, S. and Su, Y. (2020). Instrumented principal component analysis. Working paper.
