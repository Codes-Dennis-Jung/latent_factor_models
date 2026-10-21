# XS-target PCA — factors with a cross-sectional economic target

`latent_factor_models.XSTargetPCA` · registry names `xs_target_pca`, `xs-target` · notebook [`examples/05_xs_target_pca.ipynb`](../../examples/05_xs_target_pca.ipynb)

## Technical description

Target PCA (Bryzgalova, DeMiguel, Li and Pelger, 2026) builds latent factors
that are statistically optimal *and* economically interpretable. The
**cross-sectional (XS) target** is a space of loadings $\Lambda^{XS}$
($N\times L$). The estimator tilts the factors' portfolio weights and
loadings towards that space.

Examples of targets:

* characteristic "shape" targets, where loadings are monotone in the decile
  rank of a characteristic;
* macro targets, i.e. the assets' exposures to macroeconomic innovations.

The weight $\gamma_{XS}$ trades statistical fit against economic content.
$\gamma_{XS}=0$ gives uncentred PCA.

## Methodology

The objective (eq. 5) is a weighted low-rank approximation with
$W_{XS} = I_N + \gamma_{XS}\frac{N}{L}P_\Lambda$:

$$
\min_{F,\Lambda}\ \frac1{NT}\operatorname{tr}\bigl((R-F\Lambda')W_{XS}(R-F\Lambda')'\bigr).
$$

**Proposition 1.** The factors are the top-$K$ unit-norm eigenvectors of the
$T\times T$ matrix $\frac1{NT}RW_{XS}R'$, and $\hat\Lambda=R'\hat F$.
Equivalently, in $N\times N$ form, take the top-$K$ eigenvectors $Q_K$ of

$$
M=\frac1{NT}\,C R'R\,C,\qquad C=W_{XS}^{1/2}=I_N+\tilde\gamma P_\Lambda,\qquad
\tilde\gamma=\sqrt{\gamma_{XS}N/L+1}-1 ,
$$

and set $\hat\omega = CQ_K/(1+\tilde\gamma)$ and $\hat F=R\hat\omega$.
$C$ is the *closed-form* square root of $W_{XS}$, because $P_\Lambda$ is a
projector.

**Targets** (`models/targets.py`):

* `xs_target="given"`: any $N\times L$ matrix passed as `xs_target=`.
* `xs_target="macro"`: $\Lambda = R'Y(Y'Y)^{-1}$ (eq. 7) from macro
  innovations $Y$ passed as `macro=`. It is rebuilt in **each estimation
  window**, avoiding the look-ahead bias of a full-sample target.
  `macro_innovations(panel, n_components=7, var_lags=1)` reproduces the
  paper's pipeline: standardise, PCA to 7 factors, VAR(1) residuals.
* `shape_target(groups, n_groups=10)`: $\Lambda_{nl}=-1+\frac{2}{9}(i-1)$ if
  asset $n$ is decile $i$ of characteristic $l$ (footnote 12).

**Metrics.**

* XS-metric of a weight vector:
  $100\lVert P_\Lambda\hat\omega\rVert/\lVert\hat\omega\rVert$. It is reported
  per factor and for the SDF weights $\hat\omega b$.
* Sharpe ratio, $\text{RMS}_\alpha$ and unexplained variation, as for all
  models. Loadings are OLS with intercept (eq. 13).

**Tuning.** The paper uses 5-fold CV on the first part of the sample,
maximising the Sharpe ratio (or minimising the relevant pricing error), with a
benchmark of $\gamma_{XS}^{shape}=1$; its simulations use
$\gamma_{XS}\in[0,16/N]$. Because $W_{XS}$ scales $\gamma_{XS}$ by $N/L$, grids
expressed in multiples of $L/N$ (as in the example) are a convenient choice.

## Implementation notes

* Implemented as `SpectralSpec(xs_target=Λ, gamma_xs=γ)`. $P_\Lambda$ uses a
  rank-revealing orthonormal basis and is never inverted. Rescaling or
  rotating $\Lambda$ does not change the estimator (tested).
* Verified against brute-force constructions of both the $N\times N$ form (with
  `scipy.linalg.sqrtm`) and the $T\times T$ form of Proposition 1.
* Fixed relative to the legacy code: out-of-sample factors were renormalised
  to unit norm on the test sample (so their scale depended on the test
  length), signs were re-chosen on test data, and loadings were $R'F$ instead
  of OLS.

## Example

```python
import latent_factor_models as lfm

sim = lfm.simulate_factor_returns(T=600, N=100, seed=0)
N, L = 100, sim.xs_target.shape[1]

m = lfm.XSTargetPCA(3, gamma_xs=L / N).fit(sim.returns, xs_target=sim.xs_target)
m.target_metrics()        # {'xs_metric': per factor, 'xs_metric_sdf': SDF weights}
m.evaluate()

# macro target rebuilt in every window, tuned by CV
gs = lfm.GridSearchCV(lfm.XSTargetPCA(3, xs_target="macro"),
                      {"gamma_xs": [0, 0.25 * L / N, L / N, 4 * L / N]}, cv=5)
gs.fit(sim.returns, macro=sim.macro)
```

## Parameters

| name | default | meaning |
|---|---|---|
| `n_factors` | 1 | number of factors |
| `gamma_xs` | 1.0 | target weight $\ge 0$ |
| `xs_target` | `"given"` | `"given"` (pass `xs_target=`) or `"macro"` (pass `macro=`) |

## References

* Bryzgalova, S., DeMiguel, V., Li, S. and Pelger, M. (2026). Asset-pricing factors with economic targets. Working paper.
