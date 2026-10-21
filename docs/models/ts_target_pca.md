# TS-target PCA — factors with a time-series economic target

`latent_factor_models.TSTargetPCA` · registry names `ts_target_pca`, `ts-target` · notebook [`examples/06_ts_target_pca.ipynb`](../../examples/06_ts_target_pca.ipynb)

## Technical description

The **time-series (TS) target** of target PCA (Bryzgalova et al., 2026) is a
set of $J$ time series $G$ ($T\times J$). The estimator up-weights the part of
returns that projects on $G$. With the *alpha target*, $G$ is built from a
reference factor model (e.g. Fama-French 5). The resulting latent factors then
focus on what the reference model **fails to price**, so they complement it
and do not reproduce it. With the *mean target* $G=1_T$ the estimator is
exactly RP-PCA.

## Methodology

With $W_{TS}=I_T+\gamma_{TS}P_G$, the objective (eq. 8) is
$\min\frac1{NT}\operatorname{tr}\bigl((R-F\Lambda')'W_{TS}(R-F\Lambda')\bigr)$.

**Proposition 2.** The factor weights $\hat\omega$ are the top-$K$
eigenvectors of

$$
M=\frac{1}{NT}R'W_{TS}R=\frac1{NT}\bigl(R'R+\gamma_{TS}\,(Q_G'R)'(Q_G'R)\bigr),
$$

with $\hat F=R\hat\omega$ and loadings $\hat\beta$ from time-series OLS of $R$
on $\hat F$ (with intercept, eq. 13). Here $Q_G$ is an orthonormal basis of
$\operatorname{col}(G)$, so the $T\times T$ matrix $W_{TS}$ is never needed.

**Targets** (`ts_target=`):

| kind | $G$ | notes |
|---|---|---|
| `"alpha"` | $(I_T-\tilde F(\tilde F'\tilde F)^{-1}F^{ref\prime})1_T$ (eq. 10) | $R'G/T$ equals the reference-model alphas. Pass the reference factors as `factors=`. Rebuilt in every window. |
| `"mean"` | $1_T$ | RP-PCA with $\gamma=\gamma_{TS}$ |
| `"comove"` | $F^{ref}-\bar F^{ref}$ | package extension (not in the paper) |
| `"given"` | `ts_target=` from `fit` | any user target |

`publication_lag=h` ignores the last $h$ periods of the reference factors
(their rows of $G$ are zero). Since $P_G$ is scale invariant, the $T/T_s$
rescaling only matters for the interpretation of $R'G/T$.

**TS-metric (eq. 16).**
$100\,\lVert P_GRP_{\hat\omega}\rVert_F/\lVert P_GR\rVert_F$: the share of the
target-projected returns spanned by the factor weights. With $J=1$ it reduces
to $\lVert P_{\hat\omega}\alpha\rVert/\lVert\alpha\rVert$.

**Tuning.** The paper's benchmark is $\gamma_{TS}=15$. It chooses $\gamma$ by
5-fold CV maximising the Sharpe ratio, and does not cross-validate $K$.

## Implementation notes

* `TSTargetPCA(K, gamma, ts_target="mean")` equals `RPPCA(K, gamma)` exactly
  (tested). The engine operator equals the brute-force
  $\frac1{NT}R'(I_T+\gamma P_G)R$ to machine precision (tested).
* Fixed relative to the legacy code: CV sliced a full-sample target, which
  used the full-sample mean of $F^{ref}$ (look-ahead). Out-of-sample factors
  were re-signed on test data. Loadings omitted the intercept. Explicit
  $T\times T$ matrices were formed.

## Example

```python
import latent_factor_models as lfm

sim = lfm.simulate_factor_returns(T=600, N=100, seed=0)
data = sim.to_data("reference")             # factors = reference model (market proxy)

m = lfm.TSTargetPCA(3, gamma_ts=15, ts_target="alpha").fit(data)
m.target_metrics()["ts_metric"]             # how much of the alpha space is spanned

lfm.Engine([lfm.PCA(3), lfm.TSTargetPCA(3, gamma_ts=15)]).out_of_sample(
    data, window=240, refit_every=1)        # alpha target rebuilt in every window
```

## Parameters

| name | default | meaning |
|---|---|---|
| `n_factors` | 1 | number of factors |
| `gamma_ts` | 15.0 | target weight, $\ge -1$ |
| `ts_target` | `"alpha"` | `"alpha"`, `"mean"`, `"comove"`, `"given"` |
| `publication_lag` | 0 | ignore the last periods of the reference factors |

## References

* Bryzgalova, S., DeMiguel, V., Li, S. and Pelger, M. (2026). Asset-pricing factors with economic targets. Working paper.
* Lettau, M. and Pelger, M. (2020). Estimating latent asset-pricing factors. *Journal of Econometrics*.
