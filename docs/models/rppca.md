# RP-PCA — risk-premium PCA

`latent_factor_models.RPPCA` · registry names `rppca`, `rp-pca` · example `examples/02_rppca.py`

## Technical description

Risk-premium PCA (Lettau and Pelger, 2020a,b) estimates latent factors that
explain both the **time series** (co-movement) and the **cross-section**
(average returns) of asset returns. It adds a penalty on pricing errors to the
PCA objective. As a result it finds *weak but priced* factors that PCA misses,
and it delivers factors with higher Sharpe ratios and smaller pricing errors,
both in and out of sample.

## Methodology

RP-PCA minimises a combination of time-series and cross-sectional errors:

$$
\min_{\Lambda,F}\ \frac{1}{NT}\sum_{t,i}(X_{ti}-F_t\Lambda_i')^2
\;+\;\gamma\,\frac1N\sum_i(\bar X_i - \bar F\Lambda_i')^2 .
$$

The solution is given by the top-$K$ eigenvectors of

$$
\hat\Sigma_{RP} \;=\; \frac1T X'X + \gamma\,\bar X\bar X'
\;=\; \hat\Sigma_X + (1+\gamma)\,\bar X\bar X',
\qquad X \text{ uncentred}.
$$

* $\gamma=-1$: covariance PCA. $\gamma=0$: second-moment PCA.
* Lettau and Pelger use $\gamma=10$ in their empirical work. Any $\gamma>-1$
  dominates PCA asymptotically for strong factors. Larger values help to
  detect weak factors (their Theorem 2) but can overstate in-sample Sharpe
  ratios when factor Sharpe ratios are low. Choose $\gamma$ by
  cross-validation.
* **Weighted version.** The empirical application uses $Q=\operatorname{diag}(1/\sigma_i)$,
  i.e. PCA of $QX'(I+\tfrac{\gamma}{T}11')XQ$, with weights mapped back by $Q$.
  Use `standardize=True`.
* The loading normalisation $\hat\Lambda'\hat\Lambda/N = I$ and factors
  $\hat F = X\hat\Lambda(\hat\Lambda'\hat\Lambda)^{-1}$ differ from the
  package's unit-norm weights only by the scalar $\sqrt N$.

**Equivalence with target PCA.** With $G=1_T$, $P_G = 11'/T$ and
$\frac{1}{NT}R'(I+\gamma P_G)R = \frac1N(\frac1T R'R+\gamma\bar R\bar R')$.
RP-PCA is therefore TS-target PCA with the mean target. The package
implements it exactly that way, and a test checks the equivalence.

**Evaluation (Lettau-Pelger protocol).**

* *Maximum Sharpe ratio* of the tangency portfolio of the factors,
  $b=\Sigma_F^{-1}\mu_F$.
* *Pricing errors* $\alpha_i$: the time-series intercepts.
  $\text{RMS}_\alpha = \sqrt{\alpha'\alpha/N}$.
* *Unexplained variation*
  $\bar\sigma^2_e = \frac1N\sum_i\sigma^2(e_i)/\sigma^2(X_i)$.
* *Out of sample*: a rolling 240-month window. Weights, loadings and tangency
  weights come from the window ending at $t$, and returns at $t+1$ are priced.
  Lettau and Pelger price with the eigenvector loadings
  (`oos_loadings="projection"` in `Engine.out_of_sample` and
  `evaluate(..., loadings="projection")`). Bryzgalova et al. use OLS betas
  from the window (`"ols"`, the default).

## Implementation notes

* Implemented as `SpectralSpec(ts_target=1_T, gamma_ts=gamma)`. The mean term
  enters as $\gamma(Q_G'R)'(Q_G'R)$ with $Q_G = 1/\sqrt T$.
* Verified against the authors' `rpPCA_cv.m` (Giglio-Xiu-Zhang code;
  identical factor space, risk premia and SDF weights) and the independent
  `rppca` package (identical eigenvalues, factor space, Sharpe ratio and
  RMS α for $\gamma\in\{-1,0,10,20\}$), and against the brute-force operator.
* Fixed relative to the legacy code: eigenvalues were unsorted (`np.linalg.eig`),
  the Onatski test was called with $r_{\max}=K$, $\sigma_e$ used `np.cov`
  along the wrong axis, and the factor-number estimator returned a flattened
  20×20 index.
* For risk-premium inference on observable factors with RP-PCA as the first
  pass, use `ThreePassRiskPremia(RPPCA(...))` (see `docs/inference.md`).

## Example

```python
import latent_factor_models as lfm

sim = lfm.simulate_factor_returns(T=600, N=100, seed=0)   # weak 3rd factor, high SR
gs = lfm.GridSearchCV(lfm.RPPCA(3), {"gamma": [-1, 0, 5, 10, 20, 50]},
                      cv=5, scoring="sharpe").fit(sim.returns)
best = gs.best_model_
best.variance_signal()          # signal-to-noise per factor (LP Table 2)

lfm.Engine([lfm.PCA(3), lfm.RPPCA(3, gamma=10)]).out_of_sample(
    sim.returns, window=240, refit_every=1, oos_loadings="projection")
```

## Parameters

| name | default | meaning |
|---|---|---|
| `n_factors` | 1 | number of factors |
| `gamma` | 10.0 | risk-premium weight, $\ge -1$ |
| `standardize` | `False` | use $Q=\operatorname{diag}(1/\sigma_i)$ (training-sample volatilities) |

## References

* Lettau, M. and Pelger, M. (2020a). Estimating latent asset-pricing factors. *Journal of Econometrics* 218, 1–31 (NBER WP 24618).
* Lettau, M. and Pelger, M. (2020b). Factors that fit the time series and cross-section of stock returns. *Review of Financial Studies* 33, 2274–2325.
