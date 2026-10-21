# Inference: risk premia of observable factors

`latent_factor_models.ThreePassRiskPremia`, `SPCA.risk_premia()`,
`latent_factor_models.inference`

## The problem

Two-pass Fama-MacBeth regressions on an observable factor $g_t$ are biased
when the factor model is misspecified, i.e. when other priced factors are
omitted or $g_t$ is measured with noise. Giglio and Xiu (2021) show that
latent factors estimated from many test assets solve this. The risk premium of
$g_t$ is the premium of its projection on the latent factors:
$\gamma_g=\eta\gamma$, where $\eta$ maps latent factors into $g_t$ and
$\gamma$ are the latent premia.

## Three-pass estimator (Giglio and Xiu, 2021)

1. **Latent factors.** Fit any portfolio factor model (PCA in the paper; RP-PCA
   or target PCA are possible here) and take demeaned factors $\hat V$
   ($T\times p$). Time-series betas $\hat\beta$ come from OLS of returns on
   $\hat V$.
2. **Cross-section.** Regress average returns on betas,
   $\hat\gamma=(\hat\beta'\hat\beta)^{-1}\hat\beta'\bar r$. With
   `zero_beta=True`, include an intercept $\gamma_0$ (zero-beta rate).
3. **Time series.** $\hat\eta=\bar G'\hat V(\hat V'\hat V)^{-1}$ gives
   $\hat\gamma_g=\hat\eta\hat\gamma$.

The estimates are invariant to rotations of $\hat V$, so the first-pass
normalisation is irrelevant (tested).

**Asymptotic variance (Theorem 1).**

$$
\operatorname{Avar}(\hat\gamma_g)=\frac1T\Bigl[K\Pi_{11}K'+K\Pi_{12}\hat\eta'+\hat\eta\Pi_{12}'K'+\hat\eta\Pi_{22}\hat\eta'\Bigr],
\qquad K=\hat\gamma'\hat\Sigma_v^{-1}\otimes I_d .
$$

With a zero-beta rate (Theorem I.3), add
$\hat\sigma^2_\alpha\,\hat\eta(\hat\Sigma_\beta-\hat\beta_0\hat\beta_0')^{-1}\hat\eta'/N$.
The variance of $\hat\gamma_0$ is
$(1-\hat\beta_0'\hat\Sigma_\beta^{-1}\hat\beta_0)^{-1}\hat\sigma^2_\alpha/N$.
The package uses the paper's $\hat\sigma^2_\alpha$ with divisor $N$; the
authors' MATLAB code uses $N-1$.

**Newey-West matrices.** $\Pi_{11},\Pi_{12},\Pi_{22}$ (and $\Pi_{13},\Pi_{33}$
for SPCA alphas) are blocks of the long-run covariance of the stacked moment
vector $[\operatorname{vec}(\hat z_t\hat v_t'),\hat v_t,\hat z_t]$, with
$\hat z_t=\bar g_t-\hat\eta\hat v_t$ and Bartlett weights $1-s/(q+1)$. They are
computed in one vectorised pass. The default lag is
$q=\lfloor4(T/100)^{2/9}\rfloor$ (`lag_rule="newey-west"`), with
`"stock-watson"` ($0.75T^{1/3}$) or an explicit `lags=` as alternatives.

**Weak-factor test (Theorem 4).** For each observable factor,
$W_\ell=T\hat\eta_\ell(\hat\Sigma_v^{-1}\Pi_{11}^{(\ell)}\hat\Sigma_v^{-1})^{-1}\hat\eta_\ell'\sim\chi^2_p$.
Rejection means $g_\ell$ is spanned by the latent factors; non-rejection
flags a weak or useless factor whose premium is not identified. The legacy
code computed this only for $d=1$. It is now available for any $d$, using the
correct sub-block of $\Pi_{11}$ under column-major vectorisation.

**Other outputs.** $R^2_g$ (share of $g$ explained by $\hat V$), the
cross-sectional $R^2$, two-sided normal p-values, and a text `summary()`.

**Number of latent factors.** `selection.giglio_xiu(eigs, N, T)` implements
$\hat p=\arg\min_j[\lambda_j+j\,c(\log N+\log T)(N^{-1/2}+T^{-1/2})\operatorname{median}(\lambda_{1:20})]-1$.

## SPCA inference

`SPCA.risk_premia(p)` uses the same functions with the supervised latent
factors. It adds the alpha of tradable factors and its variance
$\frac1T(K\Pi_{11}K'-K\Pi_{13}-\Pi_{13}'K'+\Pi_{33})$. See `docs/models/spca.md`.

## Validation

* Three-pass: identical to a literal translation of the authors' `PCAFM.m`
  for $\hat\gamma_g$, $\hat\gamma_0$, the zero-beta variances (up to the
  documented $N$ vs $N-1$), $R^2_g$, cross-sectional $R^2$, Wald p-values and
  $\hat p$. Without a zero-beta rate the package evaluates Theorem 1 at the
  no-intercept $\hat\gamma$, while `PCAFM.m` plugs in the intercept-regression
  premia; both are asymptotically equivalent (see `docs/validation.md`).
* SPCA: identical to the authors' `SPCA_cv.m` for all inference outputs
  (with `n_select=N0 + 1`, see `docs/models/spca.md`).
* Monte Carlo: 95% intervals cover the true premium in about 95% of samples
  (`tests/test_math_metrics_inference.py`).

## Example

```python
import latent_factor_models as lfm

sim = lfm.simulate_factor_returns(T=600, N=100, seed=0)
res = lfm.ThreePassRiskPremia(lfm.RPPCA(3, gamma=10), zero_beta=True).fit(
    sim.returns, sim.observable)
print(res.summary(["proxy", "useless"]))
res.gamma_0, res.gamma_0_se, res.r2_cs
```
