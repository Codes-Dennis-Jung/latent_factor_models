# PCA — principal component analysis

`latent_factor_models.PCA` · registry name `pca` · example `examples/01_pca.py`

## Technical description

PCA extracts the $K$ statistical factors that explain the most time-series
variation in a panel of $N$ excess returns over $T$ periods. It is the
benchmark that every other model in the package generalises. It is the
special case $\gamma=-1$ of RP-PCA and of the TS-target model with the mean
target, and $\gamma_{XS}=\gamma_{TS}=0$ of target PCA when uncentred.

PCA is consistent for *strong* factors, i.e. factors whose loadings are
pervasive and that explain a non-vanishing share of variance. It
systematically misses *weak* factors that explain little variance, even when
they carry large risk premia. That is the motivation for RP-PCA and target PCA.

## Methodology

Let $R\in\mathbb R^{T\times N}$ be excess returns and
$\hat\Sigma_R = \frac1T (R-1\bar R')'(R-1\bar R')$ the sample covariance.
PCA solves

$$
\min_{\Lambda,F}\ \frac{1}{NT}\sum_{t,i}\bigl(\tilde R_{ti} - F_t'\Lambda_i\bigr)^2
\quad\Longrightarrow\quad
\hat\Lambda \propto \text{top-}K\text{ eigenvectors of } \hat\Sigma_R .
$$

* `centered=True` (default): covariance PCA, eq. (2) of Bryzgalova et al.
  (2026).
* `centered=False`: eigenvectors of the uncentred second moment $R'R/T$, as in
  footnote 4 of that paper.
* `standardize=True`: correlation PCA. Each asset is divided by its
  training-sample volatility before the decomposition, and the weights are
  mapped back to raw returns.

**Factors as portfolios.** With unit-norm eigenvectors $Q_K$ the factor weights
are $\hat\omega = Q_K$ and the factors are $F = R\hat\omega$, computed from *raw*
returns so they keep their means (risk premia). The paper's normalisation
$\hat\Lambda = \sqrt N Q_K$, $F = R\hat\Lambda/N$ differs only by the scalar
$\sqrt N$, which leaves all fits and Sharpe ratios unchanged.

**Loadings** are the time-series OLS slopes of each asset on $F$, with an
intercept. For covariance PCA they equal $Q_K$. The intercepts are the
in-sample pricing errors $\alpha_i$.

**Number of factors.** The package provides Onatski's (2010) edge-distribution
estimator, the Giglio-Xiu penalised estimator, Bai-Ng information criteria and
the Ahn-Horenstein eigenvalue ratio (`latent_factor_models.selection`).

## Implementation notes

* The model is the spectral engine with $G=1_T$ and $\gamma_{TS}=-1$:
  $R'R - T\bar R\bar R' = T\hat\Sigma_R$. No demeaned copy of $R$ is needed and
  no $T\times T$ matrix is formed.
* Only the top $K$ eigenpairs are computed (`scipy.linalg.eigh` with
  `subset_by_index`). `spectrum()` returns all eigenvalues on demand.
* `absorption_ratio()` returns $\lambda_k/\operatorname{tr}(M)$, and
  `variance_signal()` the signal-to-noise ratios of Lettau-Pelger, Table 2.
* Fixed relative to the legacy code: out-of-sample factors are no longer
  re-signed on the test sample, and eigenvalues are no longer recomputed from
  test data.

## Example

```python
import latent_factor_models as lfm
from latent_factor_models.selection import covariance_eigenvalues, onatski

sim = lfm.simulate_factor_returns(T=600, N=100, seed=0)   # 3 true factors
k = onatski(covariance_eigenvalues(sim.returns), r_max=10)   # -> 3

model = lfm.PCA(n_factors=k).fit(sim.returns)
model.evaluate()                          # in-sample SR, RMS alpha, sigma_e, R²
lfm.PCA(k).fit(sim.returns[:400]).evaluate(sim.returns[400:])  # out of sample
```

From the command line:

```bash
lfm simulate -o data
lfm fit pca --returns data/returns.csv -p n_factors=3 -o pca_out
```

## Parameters

| name | default | meaning |
|---|---|---|
| `n_factors` | 1 | number of factors $K$ |
| `centered` | `True` | covariance (`True`) or uncentred second moment |
| `standardize` | `False` | correlation PCA (volatilities from the training sample) |

## References

* Connor, G. and Korajczyk, R. (1986). Performance measurement with the arbitrage pricing theory. *JFE*.
* Bai, J. and Ng, S. (2002). Determining the number of factors in approximate factor models. *Econometrica*.
* Onatski, A. (2010). Determining the number of factors from empirical distribution of eigenvalues. *REStat*.
