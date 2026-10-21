# Methodology

This document explains **how the package works as a process**: the steps from
raw data to results, which model to use for which question, how models are
tuned and evaluated, how results are validated, and the assumptions behind them.
The mathematics is in [TECHNICAL_DESCRIPTION.md](TECHNICAL_DESCRIPTION.md);
worked examples are the notebooks in [`examples/`](examples/).

---

## 1. The pipeline at a glance

```
 raw data ──► 1 prepare ──► 2 specify ──► 3 estimate ──► 4 evaluate ──► 5 select / tune
                                                              │                 │
                                                              ▼                 ▼
                                               6 infer (risk premia, tests)   7 out-of-sample study
```

| Step | What happens | Package objects |
|---|---|---|
| 1 Prepare | align dates, compute excess returns, lag characteristics, handle missing values | `FactorData`, `CharacteristicPanel`, `FactorData.dropna`, target constructors |
| 2 Specify | choose the model and its hyper-parameters | `PCA`, `RPPCA`, `XSTargetPCA`, `TSTargetPCA`, `XSTSTargetPCA`, `SPCA`, `IPCA`, or by name via `create_model` |
| 3 Estimate | fit on a training sample | `model.fit(data, **side_information)` |
| 4 Evaluate | asset-pricing metrics in and out of sample | `model.evaluate(R_test)`, `target_metrics()`, `IPCA.evaluate(panel)` |
| 5 Select | number of factors, γ, screening size | `selection.onatski / giglio_xiu / bai_ng`, `GridSearchCV`, `KFold`, `TimeSeriesSplit` |
| 6 Infer | risk premia of observable factors, alphas, specification tests | `ThreePassRiskPremia`, `SPCA.risk_premia`, `IPCA.test_*` |
| 7 Study | compare many models with rolling re-estimation, in parallel | `Engine`, `ModelSpec`, `Executor`, `lfm run config.json` |

---

## 2. Step 1 — data preparation

1. **Excess returns.** All estimators expect excess returns (returns minus the
   risk-free rate) in a $T\times N$ matrix with dates in rows. Pandas
   DataFrames are accepted and their labels are kept in the outputs.
2. **Balanced panels for return-based models.** PCA, RP-PCA, target PCA and
   SPCA require complete data. `FactorData.dropna(R, max_missing=...)` first
   drops assets with too many gaps, then incomplete dates. NaNs are never
   silently filled with zeros (a legacy behaviour that biased results).
3. **Unbalanced panels for IPCA.** `CharacteristicPanel` keeps missing
   observations as NaN; an observation is used only if its return and all of
   its instruments are observed. Characteristics must be **lagged** so that the
   instruments in row $t$ are known before the return in row $t$. Kelly-Pruitt-Su
   additionally rank-transform characteristics cross-sectionally to $[-0.5,0.5]$.
4. **Side information.** Pass observable factors (`factors=`), targets
   (`xs_target=`, `ts_target=`) or macro innovations (`macro=`) with the
   returns; the container validates their shapes against $T$ and $N$.
5. **Targets that depend on data** (alpha target, macro target) are not built
   by the user but **inside each estimation window** by the model. This is
   what prevents look-ahead bias in cross-validation and rolling studies.

---

## 3. Step 2 — choosing a model

| Question | Model | Why |
|---|---|---|
| What are the dominant sources of co-movement? | `PCA` | maximises explained variance |
| Which factors price the cross-section (high Sharpe ratio, low alphas)? | `RPPCA` | penalises pricing errors, finds weak but priced factors |
| Factors that explain what a benchmark model (e.g. FF5) misses? | `TSTargetPCA(ts_target="alpha")` | targets the reference model's alphas |
| Factors with interpretable loadings (characteristic patterns, macro exposures)? | `XSTargetPCA` | tilts loadings towards a target space |
| Both interpretability and pricing? | `XSTSTargetPCA` | combines the two targets |
| Risk premium of a given factor (macro, anomaly), robust to omitted factors? | `ThreePassRiskPremia` | Giglio-Xiu three-pass with valid standard errors |
| …when the factor is weak in the test assets? | `SPCA` | supervised screening recovers weak factors |
| Betas that depend on characteristics; large unbalanced panels of stocks? | `IPCA` | instrumented loadings, tests of characteristic alphas |

All spectral models are special cases of one estimator (TECHNICAL_DESCRIPTION
§2). Setting the target weights to zero recovers PCA, so the target weight
directly measures how much the economic information departs from pure
statistics.

---

## 4. Step 3 — estimation

Every model is fitted the same way, `model.fit(data, **side_information)`, and
exposes its results as attributes ending in `_` (see the README's *Outputs*
section). Estimation is deterministic: spectral models fix signs by positive
factor means, IPCA by its identification rules, and the Lanczos eigensolver
uses a fixed start vector. Repeated fits on the same data return identical
numbers.

---

## 5. Step 4 — evaluation

Asset-pricing quality is judged on three dimensions (Lettau-Pelger):

1. **Sharpe ratio** of the factors' tangency (SDF) portfolio — how much
   compensated risk the factors capture;
2. **RMS pricing error** $\sqrt{\alpha'\alpha/N}$ — how well the factors price
   the test assets;
3. **unexplained variation** — how much time-series variation is left.

Out of sample, everything that is estimated (weights, betas, tangency weights,
targets) comes from the training data only. For pricing new returns the package
offers both conventions used in the literature (OLS betas, or projection
loadings). Target models add the XS- and TS-metrics, which quantify how much of
the factors' structure is explained by the economic targets. IPCA is judged by
total and predictive $R^2$.

---

## 6. Step 5 — model selection and tuning

* **Number of factors.** Use several estimators on the spectrum (Onatski,
  Giglio-Xiu, Bai-Ng, Ahn-Horenstein) and look for agreement; the
  `variance_signal()` diagnostic shows which factors are above the noise level.
* **Target weights (γ).** Choose by cross-validation of the quantity of
  interest (Sharpe ratio by default; `neg_rms_alpha` if pricing errors matter
  more). The optimum is typically interior: too large γ over-weights noisy
  sample means. Contiguous folds (`KFold(5)`) follow Lettau-Pelger and
  Bryzgalova et al.; `TimeSeriesSplit(gap=…)` gives strictly forward-looking
  validation.
* **SPCA screening size.** Repeated random 3-fold CV of the hedging $R^2$
  (Giglio-Xiu-Zhang, Appendix E): `KFold(3, shuffle=True, n_repeats=…)`.
* Every (parameter, fold) pair is an independent task, so grids run in
  parallel; failed fits score `NaN` and are never selected.

---

## 7. Step 6 — inference

* **Risk premia of observable factors.** First check the weak-factor Wald test:
  only if the factor is spanned by the latent factors (rejection) is its premium
  identified. Then read $\hat\gamma_g$ and its Newey-West standard error. Use a
  zero-beta rate if the risk-free rate is not believed to price the assets.
* **Tradable factors.** `SPCA.risk_premia().extra["alpha"]` tests whether a
  traded factor earns a premium beyond its exposure to the latent factors.
* **IPCA specification tests.** `test_alpha` ($\Gamma_\alpha=0$: do
  characteristics proxy for mispricing?), `test_instrument` (does a
  characteristic matter for betas?), `test_observed_factors`. These are wild
  residual bootstraps; use ≥ 1000 draws for final results and `block_size>1`
  for serially dependent residuals.

---

## 8. Step 7 — out-of-sample studies

`Engine` runs the standard rolling protocol: for every test date each model is
re-estimated on the preceding window (rolling or expanding, optionally only
every k dates), and the pooled test-date results give out-of-sample Sharpe
ratios, pricing errors and unexplained variation (IPCA: $R^2$). The same study
can be described in a JSON/TOML file and run from the command line
(`lfm run config.json`), which makes it reproducible and easy to share.

**Parallelism.** `executor="process"` distributes models × windows over all
cores; `"thread"` suits large BLAS-bound problems. Because every task receives
its own random stream spawned from one seed, results are identical across
backends and worker counts. With many processes, limit BLAS threads
(`OMP_NUM_THREADS=1`) to avoid oversubscription.

---

## 9. Validation strategy

Correctness is established at four levels (details in
[docs/validation.md](docs/validation.md)):

1. **Against the authors' original code**: literal translations of Giglio-Xiu
   `PCAFM.m`, Giglio-Xiu-Zhang `SPCA_cv.m` and `rpPCA_cv.m`, and the original
   `ipca` package — identical results; the few places where the authors' code
   departs from their paper are documented and follow the paper.
2. **Against the papers' mathematics** where no code exists (target PCA):
   brute-force constructions of the operators, the equivalent $T\times T$
   formulation, optimality for the paper's objective.
3. **Statistical properties** by Monte Carlo: coverage of confidence
   intervals, size and power of bootstrap tests, recovery of the number of
   factors, detection of weak factors.
4. **Engineering**: 86 automated tests (a fast tier on every run, a slow tier
   that also executes all notebooks), type checking, linting, and CI on Python
   3.10-3.14 × Linux/Windows/macOS. Performance optimisations were accepted only
   after reproducing the previous outputs to machine precision.

---

## 10. Assumptions and limitations

* **Large-$N$, large-$T$ asymptotics.** Inference (three-pass, SPCA) relies on
  many test assets and a long sample; with small $N$ the latent factors are
  estimated with noise that the asymptotic standard errors ignore.
* **Stationarity** of the factor structure within each estimation window;
  use rolling windows if loadings change over time (or IPCA, where loadings
  move with characteristics).
* **Linear factor model.** All estimators are linear in returns; non-linear
  conditioning is only available through IPCA's instruments.
* **Balanced panels** for the return-based models (see Step 1).
* **Sample vs. population premia.** Any estimator recovers the premium
  *realised in the sample*; standard errors measure the distance to the
  population premium.
