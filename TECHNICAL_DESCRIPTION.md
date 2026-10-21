# Technical description

This document gives the mathematical definition of every estimator,
statistic and algorithm implemented in `latent-factor-models`, together with
the identities the implementation relies on. Equation numbers in brackets refer
to the original papers (see [References](#references)). For the workflow and
the choice between models see [METHODOLOGY.md](METHODOLOGY.md); for the
validation record see [docs/validation.md](docs/validation.md).

---

## 1. Notation

| Symbol | Dimension | Meaning |
|---|---|---|
| $R=(r_1,\dots,r_T)'$ | $T\times N$ | excess returns, time in rows (uncentred unless stated) |
| $\bar R = T^{-1}R'1_T$ | $N$ | average returns; $\tilde R = R - 1_T\bar R'$ demeaned returns |
| $F$ | $T\times K$ | latent factors; $\omega$ ($N\times K$) factor-portfolio weights with $F=R\omega$ |
| $\beta$ | $N\times K$ | loadings; $\alpha$ ($N$) pricing errors |
| $G$ | $T\times d$ | observable factors (inference) or time-series target $G^{TS}$ ($T\times J$) |
| $\Lambda$ | $N\times L$ | cross-sectional target $\Lambda^{XS}$ |
| $P_A=A(A'A)^{-1}A'$ | | orthogonal projector onto $\operatorname{col}(A)$; computed as $Q_AQ_A'$ with an orthonormal basis $Q_A$ from a rank-revealing SVD |
| $Z_t$ | $N_t\times L$ | instruments (lagged characteristics) of the $N_t$ assets observed at $t$ |

Projectors are idempotent and symmetric, $P^2=P=P'$. Hence for any scalar $a\ge-1$

$$
(I+gP)^2 = I+(2g+g^2)P = I + aP \quad\text{for}\quad g=\sqrt{1+a}-1 , \tag{1.1}
$$

which gives closed-form square roots of all weighting matrices below.

---

## 2. The unified spectral estimator

### 2.1 Operator

PCA, RP-PCA and the three target-PCA estimators are the top-$K$ eigenvectors of

$$
M=\frac{1}{NT}\,C\,R'W_{TS}R\,C,\qquad
W_{TS}=I_T+\gamma_{TS}P_G,\qquad
C=I_N+\tilde\gamma P_\Lambda,\qquad
\tilde\gamma=\sqrt{\gamma_{XS}N/L+1}-1 . \tag{2.1}
$$

By (1.1), $C^2 = W_{XS}:=I_N+\gamma_{XS}\tfrac NL P_\Lambda$. The estimator
solves the weighted low-rank problem [Bryzgalova et al., eq. 11]

$$
\min_{F,\beta}\ \frac{1}{NT}\operatorname{tr}\!\big(W_{TS}\,(R-F\beta')\,W_{XS}\,(R-F\beta')'\big). \tag{2.2}
$$

*Proof sketch.* Write $W_{TS}=S^2$ with $S=I+g_{TS}P_G$, $g_{TS}=\sqrt{1+\gamma_{TS}}-1$.
Then (2.2) equals $\frac1{NT}\lVert S(R-F\beta')C\rVert_F^2=\frac1{NT}\lVert SRC-(SF)(C\beta)'\rVert_F^2$.
By Eckart-Young the best rank-$K$ approximation of $X:=SRC$ is $XQ_KQ_K'$ with
$Q_K$ the top right singular vectors of $X$, i.e. the top eigenvectors of
$X'X/(NT)=M$. Undoing the transformation gives $F\beta' = RCQ_KQ_K'C^{-1}$, so
$F\propto RCQ_K$. $\square$

| Model | $G$ | $\gamma_{TS}$ | $\Lambda$ | Paper |
|---|---|---|---|---|
| PCA (covariance) | $1_T$ | $-1$ | – | eq. (2) |
| PCA (second moment) | – | 0 | – | fn. 4 |
| RP-PCA | $1_T$ | $\gamma$ | – | Lettau-Pelger |
| TS-target PCA | $G^{TS}$ | $\gamma_{TS}$ | – | Prop. 2, eq. (9) |
| XS-target PCA | – | 0 | $\Lambda^{XS}$ | Prop. 1, eq. (6) |
| XS-TS-target PCA | $G^{TS}$ | $\gamma_{TS}$ | $\Lambda^{XS}$ | Prop. 3, eq. (12) |

With $G=1_T$, $P_G=1_T1_T'/T$ and
$\frac1T R'(I+\gamma P_G)R=\frac1TR'R+\gamma\bar R\bar R'=\hat\Sigma_R+(1+\gamma)\bar R\bar R'$,
which is RP-PCA; $\gamma=-1$ gives $\hat\Sigma_R$ (covariance PCA).

### 2.2 Estimates

With $Q_K$ the unit-norm top-$K$ eigenvectors of $M$ (eigenvalues $\lambda_1\ge\dots\ge\lambda_K$):

$$
\hat\omega=\frac{1}{1+\tilde\gamma}\,C\,Q_K
=\Big(\tfrac{1}{1+\tilde\gamma}I_N+\tfrac{\tilde\gamma}{1+\tilde\gamma}P_\Lambda\Big)Q_K,
\qquad \hat F=R\hat\omega, \tag{2.3}
$$

$$
r_{n,t}=\alpha_n+\hat F_t'\beta_n+e_{n,t}\quad\text{(time-series OLS with intercept, eq. 13).} \tag{2.4}
$$

Factors are portfolios of **raw** returns and keep their means. Signs are
fixed once by $\operatorname{sign}(\bar{\hat F}_k)>0$, applied jointly to
$\hat\omega$, $Q_K$ and $\hat F$. Scale normalisations of the papers
($\hat\Lambda=\sqrt NQ_K$, $\hat F=R\hat\Lambda/N$, unit-norm $T$-space
factors) differ from (2.3) by a diagonal rescaling, which leaves fitted values,
pricing errors and Sharpe ratios unchanged.

**Optional standardisation** (Lettau-Pelger's empirical $Q=\operatorname{diag}(1/\hat\sigma_i)$):
the operator is built from $R\,Q$ and the weights are mapped back,
$\hat\omega\leftarrow Q\hat\omega$, with $\hat\sigma_i$ from the training sample.

### 2.3 Computation

Let $X=S\,R\,C$ ($T\times N$), computed with low-rank updates
$X\leftarrow X+g_{TS}Q_G(Q_G'X)$ and $X\leftarrow X+\tilde\gamma(XQ_\Lambda)Q_\Lambda'$;
no $T\times T$ or $N\times N$ weighting matrix is formed. Then $M=X'X/(NT)$ and

* if $K$ is small relative to $m=\min(T,N)$ ($K\le m/5$, $m\ge60$): implicitly
  restarted Lanczos (ARPACK) on the implicit operator $v\mapsto X'(Xv)$ (or
  $u\mapsto X(X'u)$ if $N>T$), cost $O(TN)$ per iteration, tolerance at machine
  precision, fixed start vector (reproducible), dense fallback;
* otherwise LAPACK's subset eigensolver (`?syevr`) on the smaller Gram matrix
  ($X'X$ if $N\le T$, else $XX'$), cost $O(TN\,m+m^2K)$.

For $N>T$ the eigenvectors are mapped back by $Q_K=X'U_K\Lambda_K^{-1/2}$.
$\operatorname{tr}M=\lVert X\rVert_F^2/(NT)$ and the full spectrum
$\{s_i^2/(NT)\}$ come from $X$ without forming $M$; `operator_` builds $M$ on
request only.

### 2.4 Diagnostics

* absorption ratio $\lambda_k/\operatorname{tr}M$;
* variance signal (Lettau-Pelger, Table 2): $N\lambda_k/\hat\sigma_e^2$, with
  $\hat\sigma_e^2=\frac1N\sum_i\widehat{\operatorname{var}}(\hat e_i)$ after six PCA factors;
* XS-metric $100\,\lVert P_\Lambda\hat\omega_k\rVert_2/\lVert\hat\omega_k\rVert_2$
  (per factor, and for the SDF weights $\hat\omega\hat b$);
* TS-metric [eq. 16] $100\,\lVert P_GRP_{\hat\omega}\rVert_F/\lVert P_GR\rVert_F$,
  computed as $\lVert Q_G'RQ_\omega Q_\omega'\rVert_F/\lVert Q_G'R\rVert_F$.

---

## 3. Targets

| Target | Definition | Property |
|---|---|---|
| mean | $G=1_T$ | TS-target PCA = RP-PCA |
| alpha [eq. 10] | $G=\big(I_T-\tilde F(\tilde F'\tilde F)^{-1}F^{ref\prime}\big)1_T$ | $R'G/T=\hat\alpha^{ref}$, the intercepts of $R$ on $F^{ref}$ |
| co-movement (extension) | $G=\tilde F^{ref}$ | $R'G/T$ = covariances with the reference factors |
| macro [eq. 7] | $\Lambda=R'Y(Y'Y)^{-1}$ | macro-mimicking loadings, rebuilt in each window |
| shape [fn. 12] | $\Lambda_{nl}=-1+\tfrac{2}{n_g-1}(i-1)$ if asset $n$ is group $i$ of sort $l$, else 0 | monotone characteristic patterns |

$\tilde F$ is the demeaned reference matrix. *Alpha target identity:* with
$c=(\tilde F'\tilde F)^{-1}F^{ref\prime}1_T$, $R'G/T=\bar R-\frac1TR'\tilde F\,c=\bar R-\hat\beta^{ref}\bar F^{ref}=\hat\alpha^{ref}$,
since $\hat\beta^{ref}=R'\tilde F(\tilde F'\tilde F)^{-1}$ and $F^{ref\prime}1_T=T\bar F^{ref}$.
With a publication lag $h$ only the first $T-h$ rows are used and the remaining
rows of $G$ are zero. Macro innovations: standardise the macro panel, keep the
first principal components, and take VAR($p$) residuals. All targets are
invariant to invertible right-rotations ($P_{AH}=P_A$).

---

## 4. Supervised PCA (Giglio, Xiu & Zhang)

Inputs: $\tilde R$, $\bar r$, $\tilde G$ ($T\times d$), $N_0=\lfloor qN\rfloor$.
Initialise $R^{(1)}=\tilde R$, $r^{(1)}=\bar r$, $G^{(1)}=\tilde G$. For $k=1,\dots,p$:

1. Screening: $s_i=\max_j|\operatorname{corr}(R^{(k)}_{\cdot i},G^{(k)}_{\cdot j})|$, $\hat I_k$ = indices of the $N_0$ largest $s_i$.
2. Leading singular triplet $(\xi,\sigma,\varsigma)$ of $R^{(k)}_{\cdot\hat I_k}$ ($R^{(k)}_{\cdot\hat I_k}\varsigma=\sigma\xi$):
   $$\hat v^{(k)}=\sqrt T\xi,\quad \hat\gamma^{(k)}=\sqrt T\,\varsigma'r^{(k)}_{\hat I_k}/\sigma,\quad \hat\eta^{(k)}=G^{(k)\prime}\xi/\sqrt T .$$
3. Deflation with $\hat\beta^{(k)}=R^{(k)\prime}\xi/\sqrt T$:
   $R^{(k+1)}=(I-\xi\xi')R^{(k)}$, $r^{(k+1)}=r^{(k)}-\hat\beta^{(k)}\hat\gamma^{(k)}$, $G^{(k+1)}=(I-\xi\xi')G^{(k)}$.

The $\xi^{(k)}$ are orthonormal, so $\hat V'\hat V/T=I_p$ and $\hat\eta=\tilde G'\hat V/T$.
Outputs: $\hat\gamma_g=\hat\eta\hat\gamma$; $\hat\alpha=\bar g-\hat\eta\hat\gamma$;
sparse weights $B$ (column $k$ equal to $\sqrt T\varsigma/\sigma$ on $\hat I_k$);
mimicking portfolio $W=\tilde G'\bar F(\bar F'\bar F)^{-1}B'$ with $\bar F=\tilde RB$
(identity $W\bar r=\hat\gamma_g$); SDF weights $b=B(B'\tilde R'\tilde RB/T)^{-1}B'\bar r$;
hedging $R^2_\ell=1-\lVert\tilde G_\ell-\tilde RW_\ell'\rVert^2/\lVert\tilde G_\ell\rVert^2$.
The leading triplet is computed with the Lanczos/LAPACK routine of §2.3 ($k=1$).

---

## 5. Instrumented PCA (Kelly, Pruitt & Su)

$$
r_{i,t}=z_{i,t}'\Gamma_\alpha+z_{i,t}'\Gamma_\beta f_t+z_{i,t}'\Gamma_\delta g_t+\varepsilon_{i,t}. \tag{5.1}
$$

With $\tilde\Gamma=[\Gamma_\beta,\Gamma_\delta,\Gamma_\alpha]$ ($L\times\tilde K$),
$\tilde f_t=[f_t;g_t;1]$ and managed portfolios $X_t=Z_t'r_t/N_t$,
$W_t=Z_t'Z_t/N_t$ (computed with batched matrix products), the least-squares
objective $\sum_t\sum_{i}(r_{i,t}-z_{i,t}'\tilde\Gamma\tilde f_t)^2$ is minimised by
alternating least squares:

$$
f_t=(\Gamma_\beta'W_t\Gamma_\beta)^{-1}\Gamma_\beta'\big(X_t-W_t[\Gamma_\delta,\Gamma_\alpha][g_t;1]\big), \tag{5.2}
$$

$$
\operatorname{vec}(\tilde\Gamma')=\Big[\sum_t N_t\,W_t\otimes\tilde f_t\tilde f_t'\Big]^{-1}\sum_t N_t\,X_t\otimes\tilde f_t . \tag{5.3}
$$

(5.2) is a batched $K\times K$ solve for all $t$. The normal matrix in (5.3)
is computed as one GEMM, $[\,\operatorname{vec}W_t\,]_t'\,[\,N_t\operatorname{vec}(\tilde f_t\tilde f_t')\,]_t$,
reshaped to $(L\tilde K)\times(L\tilde K)$ and solved by Cholesky
(`weighting="periods"` drops the $N_t$ weights).

**Identification**, after each iteration: (i) $b=(\Gamma_\beta'\Gamma_\beta)^{-1}\Gamma_\beta'[\Gamma_\delta,\Gamma_\alpha]$,
$[\Gamma_\delta,\Gamma_\alpha]\leftarrow[\Gamma_\delta,\Gamma_\alpha]-\Gamma_\beta b$, $f_t\leftarrow f_t+b[g_t;1]$;
(ii) $R_1=\operatorname{chol}(\Gamma_\beta'\Gamma_\beta)'$, $U=$ left singular vectors of $R_1\big(\sum_tf_tf_t'\big)R_1'$,
$\Gamma_\beta\leftarrow\Gamma_\beta R_1^{-1}U$, $f_t\leftarrow U'R_1f_t$; (iii) positive factor means.
Result: $\Gamma_\beta'\Gamma_\beta=I_K$, $\sum_tf_tf_t'$ diagonal and descending,
$\Gamma_\beta'[\Gamma_\delta,\Gamma_\alpha]=0$. Initialisation: SVD of $[X_1,\dots,X_T]$.
Stop when $\max(|\Delta\tilde\Gamma|,|\Delta F|)<$ `tol` ($10^{-6}$).

**Fit** (uncentred): total $R^2=1-\sum(r-z'(\Gamma_\alpha+\Gamma_\beta f_t+\Gamma_\delta g_t))^2/\sum r^2$;
predictive $R^2$ replaces $f_t$ by $\bar f$; the same on managed portfolios with $W_t$.
**Out of sample** (KPS §4.4): $f_{t}=(\Gamma_\beta'W_t\Gamma_\beta)^{-1}\Gamma_\beta'(X_t-\dots)$ with $\tilde\Gamma$ from the training window.

**Bootstrap tests.** Residuals $\hat d_t=X_t-W_t\hat{\tilde\Gamma}\hat{\tilde f}_t$. Draw
$b$: $\tilde X_t=W_t\tilde\Gamma^{H_0}\hat{\tilde f}_t+q_t\hat d_{\tau_t}$, with $\tau_t$
uniform on $\{1,\dots,T\}$ (or block starts) and $q_t\sim t_5\sqrt{3/5}$ (unit variance);
re-estimate the unrestricted model; $p=B^{-1}\sum_b\mathbb 1\{W^{(b)}>W\}$.

| Test | $H_0$ | Statistic |
|---|---|---|
| `test_alpha` | $\Gamma_\alpha=0$ | $\Gamma_\alpha'\Gamma_\alpha$ |
| `test_instrument(l)` | row $l$ of $\Gamma_\beta$ zero (all columns; lists = joint) | $\sum_k\Gamma_{\beta,lk}^2$ (rotation invariant) |
| `test_observed_factors` | $\Gamma_\delta=0$ | $\lVert\Gamma_\delta\rVert_F^2$ |

---

## 6. Risk-premium inference (Giglio & Xiu)

**Three passes.** (1) latent factors from any portfolio factor model,
$\hat V=\hat F-1_T\bar{\hat F}'$ ($T\times p$), time-series betas
$\hat\beta=\tilde R'\hat V(\hat V'\hat V)^{-1}$; (2) $\hat\gamma=(\hat\beta'\hat\beta)^{-1}\hat\beta'\bar r$
[eq. 7-8] or, with a zero-beta rate, OLS of $\bar r$ on $[1_N,\hat\beta]$;
(3) $\hat\eta=\tilde G'\hat V(\hat V'\hat V)^{-1}$, $\hat\gamma_g=\hat\eta\hat\gamma$.
All quantities are invariant to $\hat V\to\hat VH$, $\hat\beta\to\hat\beta H^{-1\prime}$.

**Asymptotic variance** [Theorem 1]. With $\hat z_t=\tilde g_t-\hat\eta\hat v_t$,
$\hat\Sigma_v=\hat V'\hat V/T$, $K=\hat\gamma'\hat\Sigma_v^{-1}\otimes I_d$:

$$
\widehat{\operatorname{Avar}}(\hat\gamma_g)=\frac1T\Big[K\Pi_{11}K'+K\Pi_{12}\hat\eta'+\hat\eta\Pi_{12}'K'+\hat\eta\Pi_{22}\hat\eta'\Big], \tag{6.1}
$$

$$
\widehat{\operatorname{Avar}}(\hat\alpha)=\frac1T\Big[K\Pi_{11}K'-K\Pi_{13}-\Pi_{13}'K'+\Pi_{33}\Big]\quad\text{(SPCA, tradable } g). \tag{6.2}
$$

The $\Pi$ matrices are the blocks of one Bartlett long-run covariance of the
stacked moments $m_t=[\operatorname{vec}(\hat z_t\hat v_t');\hat v_t;\hat z_t]$
($\operatorname{vec}(\hat z_t\hat v_t')=\hat v_t\otimes\hat z_t$):

$$
\hat\Omega=\hat\Gamma_0+\sum_{s=1}^q\Big(1-\frac{s}{q+1}\Big)(\hat\Gamma_s+\hat\Gamma_s'),\qquad \hat\Gamma_s=\frac1T\sum_{t>s}m_tm_{t-s}', \tag{6.3}
$$

with $q=\lfloor4(T/100)^{2/9}\rfloor$ by default.
**Zero-beta rate** [Theorem I.3]: with $\hat\beta_0=\hat\beta'1_N/N$,
$\hat\Sigma_\beta=\hat\beta'\hat\beta/N$ and $\hat\sigma^2_\alpha$ the mean squared
cross-sectional residual, $\operatorname{Avar}(\hat\gamma_0)=(1-\hat\beta_0'\hat\Sigma_\beta^{-1}\hat\beta_0)^{-1}\hat\sigma^2_\alpha/N$
and (6.1) gains $\hat\sigma^2_\alpha\hat\eta(\hat\Sigma_\beta-\hat\beta_0\hat\beta_0')^{-1}\hat\eta'/N$.
**Weak-factor test** [Theorem 4]: $W_\ell=T\hat\eta_\ell(\hat\Sigma_v^{-1}\Pi_{11}^{(\ell)}\hat\Sigma_v^{-1})^{-1}\hat\eta_\ell'\sim\chi^2_p$,
where $\Pi_{11}^{(\ell)}$ selects the indices $\ell+d\,k$, $k=0,\dots,p-1$.
Also reported: $R^2_{g,\ell}=\lVert\hat V\hat\eta_\ell'\rVert^2/\lVert\tilde G_\ell\rVert^2$ and the
cross-sectional $R^2$ of $\bar r$ on $\hat\beta$. p-values are asymptotically normal.

---

## 7. Evaluation metrics and out-of-sample protocol

Given factors $F$, loadings $\beta$ and tangency weights $b=\hat\Sigma_F^{-1}\hat\mu_F$ estimated
on a training sample:

| Metric | Definition |
|---|---|
| Sharpe ratio | $\overline{x}/\operatorname{sd}(x)$ of the SDF portfolio $x_t=F_t'b$ (in sample: $\sqrt{\hat\mu_F'\hat\Sigma_F^{-1}\hat\mu_F}$) |
| pricing errors | $\alpha=\bar e$, $e_t=r_t-\beta F_t$ (in sample with OLS betas: the OLS intercepts) |
| RMS $\alpha$ | $\sqrt{\alpha'\alpha/N}$ |
| unexplained variation | $\frac1N\sum_i\operatorname{var}(e_i)/\operatorname{var}(r_i)$ |
| total $R^2$ | $1-\sum\lVert r_t-\beta F_t\rVert^2/\sum\lVert r_t\rVert^2$ |
| generalised correlation | canonical correlations between $\hat F$ and $F^{true}$ |

**Rolling protocol.** For each test date $t$, estimate on the preceding window
(rolling or expanding), compute $F_t=r_t'\hat\omega$, price with $\beta$ = the
window's OLS betas (`oos_loadings="ols"`, Bryzgalova et al.) or the projection
loadings $\hat\omega(\hat\omega'\hat\omega)^{-1}$ (`"projection"`, Lettau-Pelger),
form $x_t=F_t'b$, and pool $(e_t,x_t)$ over all test dates. IPCA reports pooled
out-of-sample total and predictive $R^2$.

---

## 8. Number of factors

| Estimator | Rule |
|---|---|
| Onatski (2010) | iterate $\delta=2|\hat b|$ from OLS of $\lambda_j,\dots,\lambda_{j+4}$ on $(j-1)^{2/3},\dots,(j+3)^{2/3}$; $\hat r=\max\{i\le r_{\max}:\lambda_i-\lambda_{i+1}\ge\delta\}$, $j=\hat r+1$, until convergence |
| Giglio-Xiu (2021, App. I.1) | $\hat p=\arg\min_{1\le j\le p_{\max}}[\lambda_j+j\,c(\log N+\log T)(N^{-1/2}+T^{-1/2})\operatorname{median}(\lambda_{1:p_{\max}})]-1$ |
| Bai-Ng (2002) | $\arg\min_k\log V(k)+k\,g(N,T)$, IC1-IC3 penalties |
| Ahn-Horenstein (2013) | $\arg\max_k\lambda_k/\lambda_{k+1}$ |

---

## 9. Complexity and numerical design

| Component | Algorithm | Cost |
|---|---|---|
| spectral operator | low-rank updates to $X=SRC$ | $O(TN(J+L))$ |
| leading eigenpairs | ARPACK Lanczos on $v\mapsto X'(Xv)$ / LAPACK `syevr` on the smaller Gram | $O(TN)$ per iteration / $O(TN\min(T,N))$ |
| loadings | least squares with intercept (QR/SVD) | $O(TNK)$ |
| IPCA managed portfolios | batched GEMM over $t$ | $O(\sum_tN_tL^2)$ |
| IPCA ALS iteration | batched $K\times K$ solves + one GEMM + Cholesky | $O(TL^2\tilde K^2+(L\tilde K)^3)$ |
| HAC / $\Pi$ matrices | lagged GEMMs on stacked moments | $O(Tq(dp+p+d)^2)$ |
| projectors | rank-revealing SVD bases, never explicit inverses | $O(TJ^2+NL^2)$ |

Parallel tasks (CV grid points × folds, rolling windows, bootstrap draws) are
independent and run through a backend-agnostic executor (serial, threads,
processes) with child seeds spawned from one `SeedSequence`, so results do not
depend on the backend or the number of workers.

---

## References

* Ahn, S. C. and Horenstein, A. R. (2013). Eigenvalue ratio test for the number of factors. *Econometrica* 81, 1203-1227.
* Bai, J. and Ng, S. (2002). Determining the number of factors in approximate factor models. *Econometrica* 70, 191-221.
* Bryzgalova, S., DeMiguel, V., Li, S. and Pelger, M. (2026). Asset-pricing factors with economic targets. Working paper.
* Giglio, S. and Xiu, D. (2021). Asset pricing with omitted factors. *Journal of Political Economy* 129, 1947-1990.
* Giglio, S., Xiu, D. and Zhang, D. Test assets and weak factors. *Journal of Finance*.
* Kelly, B., Pruitt, S. and Su, Y. (2019). Characteristics are covariances: A unified model of risk and return. *Journal of Financial Economics* 134, 501-524.
* Lettau, M. and Pelger, M. (2020). Estimating latent asset-pricing factors. *Journal of Econometrics* 218, 1-31.
* Lettau, M. and Pelger, M. (2020). Factors that fit the time series and cross-section of stock returns. *Review of Financial Studies* 33, 2274-2325.
* Newey, W. K. and West, K. D. (1994). Automatic lag selection in covariance matrix estimation. *Review of Economic Studies* 61, 631-653.
* Onatski, A. (2010). Determining the number of factors from empirical distribution of eigenvalues. *Review of Economics and Statistics* 92, 1004-1016.
