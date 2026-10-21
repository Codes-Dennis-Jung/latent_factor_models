# XS-TS-target PCA — cross-sectional and time-series targets combined

`latent_factor_models.XSTSTargetPCA` · registry names `xsts_target_pca`, `xs-ts-pca` · notebook [`examples/07_xsts_target_pca.ipynb`](../../examples/07_xsts_target_pca.ipynb)

## Technical description

The combined estimator of Bryzgalova et al. (2026, Proposition 3) imposes
both targets at once. Loadings are tilted towards a cross-sectional space
$\Lambda^{XS}$ (e.g. characteristic shapes or macro exposures). The factors are
tilted towards explaining a time-series target $G^{TS}$ (e.g. the alphas of a
reference model). It nests every other spectral model in the package.

## Methodology

The objective (eq. 11) is

$$
\min_{F,\beta}\ \frac{1}{NT}\operatorname{tr}\bigl(W_{TS}\,E\,W_{XS}\,E'\bigr),\qquad E=R-F\beta',
$$

with $W_{TS}=I_T+\gamma_{TS}P_G$ and $W_{XS}=I_N+\gamma_{XS}\frac NLP_\Lambda$.
Writing $W_{XS}=C^2$ with $C=I+\tilde\gamma P_\Lambda$, the best rank-$K$ fit
is $F\beta' = RCQ_KQ_K'C^{-1}$. Here $Q_K$ holds the top-$K$ eigenvectors of

$$
M=\frac1{NT}\,C\,R'W_{TS}R\,C .
$$

Hence

$$
\hat\omega=\frac{1}{1+\tilde\gamma}\,CQ_K=\Bigl(\tfrac{1}{1+\tilde\gamma}I_N+\tfrac{\tilde\gamma}{1+\tilde\gamma}P_\Lambda\Bigr)Q_K,
\qquad \hat F=R\hat\omega,
$$

and $\hat\beta$ comes from time-series OLS with intercept (eq. 13). Both XS-
and TS-metrics are reported.

Special cases: $\gamma_{XS}=0$ gives TS-target PCA, and $\gamma_{TS}=0$ gives
XS-target PCA.

## Implementation notes

* One call of the spectral engine with both ingredients. $C$ is applied to the
  left and right of $R'W_{TS}R$ via low-rank updates.
* Verified against the brute-force operator and the weighted objective. A test
  perturbs the optimal eigenvectors and checks that the objective never
  improves.
* Cross-validating the two weights is a natural use of the parallel
  `GridSearchCV`, where each (γ_XS, γ_TS, fold) triple is a separate task.

## Example

```python
import latent_factor_models as lfm

sim = lfm.simulate_factor_returns(T=600, N=100, seed=0)
data = sim.to_data("reference")             # reference factors + xs_target + macro
L, N = sim.xs_target.shape[1], 100

grid = {"gamma_xs": [0.0, L / N, 4 * L / N], "gamma_ts": [0.0, 5.0, 15.0]}
gs = lfm.GridSearchCV(lfm.XSTSTargetPCA(3), grid, cv=5, scoring="sharpe",
                      executor="process", n_jobs=-1).fit(data)
gs.best_params_, gs.best_model_.target_metrics()
```

## Parameters

| name | default | meaning |
|---|---|---|
| `n_factors` | 1 | number of factors |
| `gamma_xs` | 1.0 | XS weight $\ge 0$ |
| `gamma_ts` | 15.0 | TS weight $\ge -1$ |
| `xs_target` | `"given"` | `"given"` or `"macro"` |
| `ts_target` | `"alpha"` | `"alpha"`, `"mean"`, `"comove"`, `"given"` |
| `publication_lag` | 0 | see TS-target PCA |

## References

* Bryzgalova, S., DeMiguel, V., Li, S. and Pelger, M. (2026). Asset-pricing factors with economic targets. Working paper.
