# SPCA — supervised PCA for risk premia of weak factors

`latent_factor_models.SPCA` · registry names `spca`, `supervised-pca` · notebook [`examples/03_spca.ipynb`](../../examples/03_spca.ipynb)

## Technical description

Supervised PCA (Giglio, Xiu and Zhang, *Test Assets and Weak Factors*, JF)
estimates the risk premium of an observable factor $g_t$ (for example a
macro factor or a candidate anomaly) when the test assets load on it only
**weakly**. In that setting standard PCA-based two-pass or three-pass
estimators are biased, because the latent factors that drive $g_t$ are not
recovered.

SPCA *supervises* the PCA step. At each step it keeps only the assets most
correlated with the (residual) factor, extracts one latent factor from them,
and then removes that factor from returns, mean returns and $g_t$. The output
is a risk premium with valid asymptotic standard errors, the alpha of $g_t$
if it is tradable, a hedging (mimicking) portfolio, SDF weights, and a Wald
test of whether $g_t$ is spanned at all.

## Methodology

Inputs: demeaned returns $\bar R$ ($N\times T$ in the paper), mean returns
$\bar r$, demeaned factors $\bar G$ ($d\times T$), and the number of selected
assets $N_0=\lfloor qN\rfloor$. Set $R^{(1)}=\bar R$, $r^{(1)}=\bar r$,
$G^{(1)}=\bar G$, and for $k=1,\dots,p$:

1. **Screening.** $s_i = \max_j |\operatorname{corr}(R^{(k)}_i, G^{(k)}_j)|$.
   $\hat I_k$ is the set of the $N_0$ largest $s_i$.
2. **PCA on the selection.** Take the leading singular triplet
   $(\varsigma,\sigma,\xi)$ of $R^{(k)}_{[\hat I_k]}$ and set
   $$\hat v^{(k)}=\sqrt T\,\xi',\quad
   \hat\gamma^{(k)} = \sqrt T\,\varsigma' r^{(k)}_{[\hat I_k]}/\sigma,\quad
   \hat\eta^{(k)} = G^{(k)}\xi/\sqrt T .$$
3. **Deflation.**
   $R^{(k+1)} = R^{(k)}(I-\xi\xi')$, $\;r^{(k+1)} = r^{(k)} - \hat\beta^{(k)}\hat\gamma^{(k)}$
   with $\hat\beta^{(k)} = R^{(k)}\xi/\sqrt T$, and $G^{(k+1)} = G^{(k)}(I-\xi\xi')$.

**Outputs.**

* Risk premium: $\hat\gamma_g=\hat\eta\hat\gamma$.
* Alpha of a tradable $g$: $\hat\alpha = \bar g - \hat\eta\hat\gamma$.
* Mimicking portfolio: $W=\bar G\bar F'(\bar F\bar F')^{-1}B'$, with
  $\bar F = B'\bar R$ and $B$ the sparse $N\times p$ weight matrix. Note that
  $W\bar r = \hat\gamma_g$ exactly.
* SDF weights: $b=B(T^{-1}B'\bar R\bar R'B)^{-1}B'\bar r$.

**Inference (Theorem 2).**

$$
\Phi = K\Pi_{11}K' + K\Pi_{12}\hat\eta' + \hat\eta\Pi_{12}'K' + \hat\eta\Pi_{22}\hat\eta',
\quad K = \hat\gamma'\hat\Sigma_v^{-1}\otimes I_d,
\quad \operatorname{Avar}(\hat\gamma_g)=\Phi/T,
$$

$$
\operatorname{Avar}(\hat\alpha) = \bigl(K\Pi_{11}K' - K\Pi_{13} - \Pi_{13}'K' + \Pi_{33}\bigr)/T,
$$

Here $\Pi_{\cdot\cdot}$ are Newey-West long-run covariances of
$[\operatorname{vec}(\hat z_t\hat v_t'),\hat v_t,\hat z_t]$ with
$\hat z_t = \bar g_t - \hat\eta\hat v_t$. Inference is asymptotically normal.

**Weak-factor Wald test** (Giglio-Xiu, Theorem 4), for each factor $\ell$:
$W_\ell = T\hat\eta_\ell(\hat\Sigma_v^{-1}\Pi_{11}^{(\ell)}\hat\Sigma_v^{-1})^{-1}\hat\eta_\ell' \sim\chi^2_p$
under $H_0:\eta_\ell=0$.

**Tuning.** The paper's Appendix E uses 100 repetitions of random 3-fold CV,
maximising the average out-of-fold hedging $R^2$
$1-\lVert\bar G_{oos}-W\bar R_{oos}\rVert^2/\lVert\bar G_{oos}\rVert^2$
(computed per factor). Use `GridSearchCV` with
`KFold(3, shuffle=True, n_repeats=...)` and `scoring="hedging_r2"`.

## Implementation notes

* `n_select` accepts an integer $N_0$ or a share $q\in(0,1]$.
* `zero_beta=True` cross-sectionally demeans returns first (Section 2.2.6).
* `risk_premia(p)` reuses the nested deflation, so inference for any
  $p\le K$ needs no refit.
* Verified against a literal translation of the authors' `SPCA_cv.m`:
  identical $\hat\gamma_g$, Avar, $\hat\alpha$, Avar($\hat\alpha$), $W$, $b$ and
  selected sets. The authors' code keeps $N_0+1$ assets per step (a 1-based
  indexing artefact); the package keeps exactly $N_0$ as in the paper, so
  `n_select=N0 + 1` (and `lags=1`, their default) reproduces their output.
* Fixed relative to the legacy code: crash for $d\ge 2$ and on NumPy 2,
  hedging $R^2$ pooled across factors, RMS divided by $N$ instead of $d$,
  out-of-sample Sharpe ratio always $\approx 0$, wrong p-value distribution,
  and several indexing errors in the CV helpers.

## Example

```python
import latent_factor_models as lfm

sim = lfm.simulate_factor_returns(T=600, N=100, seed=0)
R, G = sim.returns, sim.observable          # [proxy of a true factor, useless factor]

cv = lfm.KFold(3, shuffle=True, n_repeats=20, seed=0)
gs = lfm.GridSearchCV(lfm.SPCA(3), {"n_select": [10, 25, 50, 100]}, cv=cv,
                      scoring="hedging_r2", executor="process", n_jobs=-1)
gs.fit(R, factors=G)

rp = gs.best_model_.risk_premia()
print(rp.summary(["proxy", "useless"]))   # gamma_g, se, t, p, R²_g, Wald p
rp.extra["alpha"], rp.extra["alpha_se"]   # alphas of tradable factors
```

## Parameters

| name | default | meaning |
|---|---|---|
| `n_factors` | 1 | number of SPCA steps $p$ |
| `n_select` | 0.5 | $N_0$ (int) or share $q$ (float) of assets kept per step |
| `screening` | `"correlation"` | `"correlation"` (authors' code) or `"covariance"` (eq. 9) |
| `zero_beta` | `False` | cross-sectionally demean returns |
| `lags`, `lag_rule` | `None`, `"newey-west"` | Newey-West lag (rule $\lfloor 4(T/100)^{2/9}\rfloor$ by default) |

## References

* Giglio, S., Xiu, D. and Zhang, D. Test assets and weak factors. *Journal of Finance*.
* Giglio, S. and Xiu, D. (2021). Asset pricing with omitted factors. *JPE* 129, 1947–1990.
