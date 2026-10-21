
"""
Target PCA factor models — Bryzgalova et al. (2023).
  - PCA, XSTarget, TSTarget, XSTSTarget
  - compute_xs_target / compute_ts_target build the target matrices
  - All models: uncentered R (paper footnote 4), OLS loadings,
    spectral eigenvectors stored in results.eigenvectors (for centrality).
"""

import warnings
import numpy as np
import scipy.linalg
from dataclasses import dataclass
from typing import Tuple, Union, Optional, Dict, List
from statsmodels.tsa.vector_ar.var_model import VAR
import pandas as pd

def fit_factor_model_and_get_residuals(
    data,
    use_pca=True,
    n_components=None,
    var_lags=1,
    scale_before_pca=True,
    column_names=None,
):
    """Fit optional PCA then VAR(p) to `data`; return (residuals, fitted_var, pca)."""

    # ------------------------------------------------------------------
    # 1. Input normalisation
    # ------------------------------------------------------------------
    if isinstance(data, np.ndarray):
        if column_names is None:
            column_names = [f'Var_{i+1}' for i in range(data.shape[1])]
        data = pd.DataFrame(data, columns=column_names)
    else:
        data = data.copy()

    print("=" * 70)
    print("Factor Model Summary")
    print("=" * 70)
    print(f"Input shape        : {data.shape}  (observations x variables)")
    print(f"Pipeline           : {'PCA → VAR' if use_pca else 'VAR only'}")

    # ------------------------------------------------------------------
    # 2. Optional PCA step
    # ------------------------------------------------------------------
    pca = None
    var_input = data  # default: feed raw data into VAR

    if use_pca:
        
        from sklearn.decomposition import PCA as SklearnPCA
        from sklearn.preprocessing import StandardScaler
        
        if scale_before_pca:
            scaler      = StandardScaler()
            pca_input   = scaler.fit_transform(data)
            print(f"Pre-PCA scaling    : ✓ StandardScaler (zero mean, unit variance)")
        else:
            pca_input   = data.values
            print(f"Pre-PCA scaling    : ✗ None (raw covariance PCA)")
        
        pca = SklearnPCA(n_components=n_components)
        factors = pca.fit_transform(pca_input)

        # Build meaningful column names
        n_comp_actual = pca.n_components_
        factor_names = [f'PC_{i+1}' for i in range(n_comp_actual)]
        var_input = pd.DataFrame(factors, columns=factor_names, index=data.index)

        explained = pca.explained_variance_ratio_
        print(f"PCA components     : {n_comp_actual}")
        print(f"Variance explained : {explained.sum():.1%}  "
              f"(per component: {', '.join(f'{e:.1%}' for e in explained)})")
    else:
        print(f"Variables entering VAR: {data.shape[1]}")
        if data.shape[1] > 20:
            print(
                f"  ⚠  Warning: {data.shape[1]} variables → "
                f"{data.shape[1]**2} VAR coefficients. "
                "Consider use_pca=True to avoid overfitting."
            )

    # ------------------------------------------------------------------
    # 3. VAR(p) estimation
    # ------------------------------------------------------------------
    model = VAR(var_input)
    fitted_var = model.fit(maxlags=var_lags, ic=None)

    residuals = fitted_var.resid
    
    std_innov = (residuals - residuals.mean()) / residuals.std()

    print(f"VAR lag order      : {fitted_var.k_ar}")
    print(f"Observations in    : {len(var_input)}")
    print(f"Residuals out      : {len(residuals)}")
    print("\nResidual statistics:")
    print(residuals.describe())
    print("=" * 70)

    return residuals, std_innov, fitted_var, pca

# =============================================================================
# RESULTS STRUCTURE
# =============================================================================

@dataclass
class FactorModelResults:
    """Results from factor model estimation"""
    # Core estimates
    factors: np.ndarray                    # T x K factors
    loadings: np.ndarray                   # K x N loadings
    residuals: np.ndarray                  # T x N residuals

    # Eigenvalues & eigenvectors of the model's (N, N) operator M.
    # Each model has a DIFFERENT M:
    #   PCA:          M = Σ̂_R = (1/T) R_dm' R_dm
    #   XSTarget:     M = (1/NT) W_xs^{1/2} R' R W_xs^{1/2}      (raw, uncentered R)
    #   TSTarget:     M = (1/NT) R' W_ts R                    (raw R)
    #   XSTSTarget:   M_15 = (1/NT) C R' W_ts R C               (raw, uncentered R)
    # Consequently eigenvalues across models live on different scales.
    # For cross-model-comparable Absorption Ratios (Eq. 1), use
    # operator_trace as the denominator:
    #   AR^j = λ_j / operator_trace
    #   AR   = sum_j λ_j / operator_trace  (fraction of this operator's
    #                                       spectral energy captured by top-K)
    # This produces a scalar in [0, 1] comparable across all four models.
    eigenvalues: np.ndarray                # K eigenvalues of M  (top-K)
    eigenvectors: np.ndarray               # (N, K) eigenvectors of M
    operator_trace: float                  # trace(M) = sum of ALL N eigenvalues
                                           # of the model's own operator M

    # Performance metrics (Lettau & Pelger 2020)
    max_sharpe_ratio: float                # Maximum achievable Sharpe ratio
    max_sr_weights: np.ndarray             # K weights for max SR
    rms_alpha: float                       # Pricing errors (bps/month)
    sigma_e: float                         # Unexplained variance ratio
    alphas: np.ndarray                     # N pricing errors
    r_squared: float                       # Total variance explained

    # Comparable metrics (use these to compare models)
    explained_variance_per_factor: np.ndarray  # % variance per factor
    eigenvalues_normalized: np.ndarray         # Signal-to-noise ratios

    # Target metrics (equations 23 & 24)
    xs_metric: Optional[np.ndarray]        # XS-Metric per factor (K,), or None
    ts_metric: Optional[float]             # TS-Metric (scalar), or None

    # Internal
    unexplained_variation: float           # Average residual variance
    unexplained_variance_total: float      # Total unexplained variance


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _normalize_factor_signs(factors: np.ndarray) -> np.ndarray:
    """Make factors have positive mean"""
    result = factors.copy()
    for k in range(result.shape[1]):
        if np.mean(result[:, k]) < 0:
            result[:, k] *= -1
    return result


def _build_fmps(R_raw: np.ndarray, loadings: np.ndarray) -> np.ndarray:
    """FMP = R_raw Λ' (ΛΛ')^{-1}; (T, K) portfolios whose means equal factor risk premia."""
    gram     = loadings @ loadings.T                          # (K, K)
    gram_inv = np.linalg.solve(gram, np.eye(gram.shape[0]))  # (K, K)
    return R_raw @ loadings.T @ gram_inv                      # (T, K)


def _compute_rms_alpha(R_raw: np.ndarray, fmps: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """RMS pricing error from  R_i = α_i + β_i' FMP + ε_i.  Returns (rms_α bps/m, α, residuals)."""
    T, N = R_raw.shape
    X         = np.column_stack([np.ones(T), fmps])   # (T, 1+K)
    alphas    = np.zeros(N)
    residuals = np.zeros((T, N))

    for i in range(N):
        params          = np.linalg.lstsq(X, R_raw[:, i], rcond=None)[0]
        alphas[i]       = params[0]
        residuals[:, i] = R_raw[:, i] - X @ params

    rms_alpha = float(np.sqrt(np.mean(alphas ** 2)) * 10_000)   # bps / month
    return rms_alpha, alphas, residuals


def _compute_sigma_e(residuals: np.ndarray, returns: np.ndarray) -> float:
    """σ̄²_e = (1/N) Σ var(ε_i)/var(R_i).  Residuals are from FMP regression."""
    var_e = np.var(residuals, axis=0, ddof=1)   # (N,) residual variances
    var_R = np.var(returns,   axis=0, ddof=1)   # (N,) return  variances
    return float(np.mean(var_e / var_R))         # scalar, Eq. 21


def _compute_max_sharpe_ratio(fmps: np.ndarray) -> Tuple[float, np.ndarray]:
    """Max SR of FMP tangency portfolio:  SR = √(μ' Σ^{-1} μ).  Returns (sr, weights)."""
    K     = fmps.shape[1]
    mu    = fmps.mean(axis=0)                                           # (K,)
    Sig   = (np.cov(fmps.T) if K > 1
             else np.array([[np.var(fmps[:, 0], ddof=1)]]))            # (K, K)

    try:
        weights = np.linalg.solve(Sig, mu)          # Σ⁻¹ μ
        sr_sq   = float(mu @ weights)               # μ' Σ⁻¹ μ
        max_sr  = float(np.sqrt(max(sr_sq, 0.0)))   # guard numerical noise
    except np.linalg.LinAlgError:
        max_sr  = 0.0
        weights = np.zeros(K)

    return max_sr, weights


def _compute_explained_variance_per_factor(returns: np.ndarray, factors: np.ndarray,
                                           loadings: np.ndarray) -> np.ndarray:
    """Variance explained by each factor as % of total variance"""
    total_var = np.var(returns)
    K = factors.shape[1]
    explained = np.zeros(K)
    for k in range(K):
        fitted_k = np.outer(factors[:, k], loadings[k, :])
        explained[k] = 100 * np.var(fitted_k) / total_var
    return explained


def _compute_normalized_eigenvalues(returns: np.ndarray, factors: np.ndarray,
                                    loadings: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Normalized eigenvalues (signal-to-noise ratios).
    Comparable across all models.
    """
    T, N = returns.shape
    K = factors.shape[1]

    var_unexplained = 0.0
    for i in range(N):
        y = returns[:, i]
        valid = np.isfinite(y)
        if np.sum(valid) > K:
            params = np.linalg.lstsq(factors[valid], y[valid], rcond=None)[0]
            residuals = y[valid] - factors[valid] @ params
            var_unexplained += np.var(residuals, ddof=1)
        else:
            var_unexplained += np.var(y[valid], ddof=1)

    cov_factors = np.cov(factors.T) if K > 1 else np.array([[np.var(factors)]])
    M = loadings.T @ cov_factors @ loadings
    eigenvalues_raw = np.linalg.eigvalsh(M)
    eigenvalues_raw = np.sort(eigenvalues_raw)[::-1][:K]
    eigenvalues_normalized = eigenvalues_raw / var_unexplained

    return eigenvalues_normalized, var_unexplained


def _projection_matrix(G: np.ndarray) -> np.ndarray:
    """Orthogonal projector  P = G (G'G)^{-1} G'  onto col(G), lstsq-stable."""
    # G (T,G'G)^{-1} G' via lstsq: solve (G'G) X = G' for X = (G'G)^{-1} G'
    # then P = G @ X
    X = np.linalg.lstsq(G.T @ G, G.T, rcond=None)[0]   # (K, T) or (L, N)
    return G @ X


def _compute_xs_metric(factor_weights: np.ndarray, xs_target: np.ndarray) -> np.ndarray:
    """XS-Metric per factor: 100·||P_Λ ω̂_k|| / ||ω̂_k|| (paper Eq. after 6)."""
    P_xs = _projection_matrix(xs_target)          # (N, N)
    K = factor_weights.shape[1]
    xs_metrics = np.zeros(K)
    for k in range(K):
        omega_k    = factor_weights[:, k]          # (N,)
        norm_omega = np.linalg.norm(omega_k)
        if norm_omega < 1e-12:
            xs_metrics[k] = 0.0
        else:
            xs_metrics[k] = 100.0 * np.linalg.norm(P_xs @ omega_k) / norm_omega
    return xs_metrics


def _compute_ts_metric(factor_weights: np.ndarray, ts_target: np.ndarray,
                       returns: np.ndarray) -> float:
    """TS-Metric: 100·||P_G R P_ω̂||_F / ||P_G R||_F (paper Eq. 14)."""
    P_G     = _projection_matrix(ts_target)        # (T, T)

    # P_ω̂ = ω̂ (ω̂'ω̂)^{-1} ω̂'  — (N, N) symmetric idempotent
    om = factor_weights
    P_omega = om @ np.linalg.inv(om.T @ om) @ om.T   # (N, N)

    num = np.linalg.norm(P_G @ returns @ P_omega, 'fro')
    den = np.linalg.norm(P_G @ returns, 'fro')

    if den < 1e-12:
        return 0.0
    return float(100.0 * num / den)


# =============================================================================
# TARGET MATRIX CONSTRUCTORS  (Eq. 3 & 5 of the paper)
# =============================================================================

def compute_ts_target(returns: np.ndarray, ts_ref: np.ndarray, publication_lag: int = 0,
                      kind: str = "alpha") -> np.ndarray:
    """Build the time-series target G^{TS} from a reference series F^ref.

    kind="alpha": G^{TS} = (I_T − F̃ (F̃'F̃)⁻¹ F^ref') 1_T (Bryzgalova et al.,
    Eq. 10 in the version of July 8, 2026).  F̃ is the demeaned reference series and F^ref the raw one, so
    that R' G^{TS} / T = α, the intercepts of the time-series regressions of
    the assets on F^ref.  (Using F̃' in both places would give F̃'1 = 0 and
    hence G^{TS} = 1_T, i.e. RP-PCA without any reference information.)

    kind="comove": G^{TS} = F̃, so the target moments R'F̃/T are the assets'
    covariances with the reference series and P_G weights the part of
    returns that moves with it (general TS-target form, their Eq. 11).

    publication_lag > 0 uses the first T − lag rows only (the reference
    series is published with a delay): the last rows of G^{TS} are zero and
    R' G^{TS} / T equals the subsample moment.
    """
    if kind not in ("alpha", "comove"):
        raise ValueError(f"unknown ts target kind: {kind!r}")
    T = ts_ref.shape[0]
    lag = int(publication_lag or 0)
    Ts = T - lag
    F_ref = ts_ref[:Ts]                                       # (Ts, K_ref)

    # F̃ = F^ref − mean(F^ref)                                shape (Ts, K_ref)
    F_dm = F_ref - F_ref.mean(axis=0)
    if kind == "comove":
        G_ts = np.zeros((T, F_dm.shape[1]))
        G_ts[:Ts] = F_dm * (T / Ts)
        return G_ts
    FtF_inv = np.linalg.inv(F_dm.T @ F_dm + 1e-12 * np.eye(F_dm.shape[1]))

    # G = 1 − F̃ (F̃'F̃)⁻¹ F^ref' 1, rescaled so that R'G/T = α on the subsample
    ones = np.ones((Ts, 1))
    g = ones - F_dm @ FtF_inv @ (F_ref.T @ ones)              # (Ts, 1)
    G_ts = np.zeros((T, 1))
    G_ts[:Ts] = g * (T / Ts)
    return G_ts


def compute_xs_target(returns: np.ndarray, xs_ref: np.ndarray) -> np.ndarray:
    """Build Λ^{XS} = R' Y^{macro} (Y'Y)^{-1} (Eq. 6). (N, L) macro-mimicking loadings."""
    # ((Y^macro)ᵀ Y^macro)⁻¹  →  (L, L)
    GtG_inv = np.linalg.inv(xs_ref.T @ xs_ref)
    # Rᵀ Y^macro (Y^macroᵀ Y^macro)⁻¹  →  (N, L)
    return (returns.T @ xs_ref) @ GtG_inv


# =============================================================================
# STANDARD PCA (Section 2.1)
# =============================================================================

class PCA:
    """Standard PCA on (1/T) R'R, paper Section 2.1.  Stores β_PCA=√N·v̂ in eigenvectors."""

    def __init__(self, n_factors: int):
        self.n_factors = n_factors
        self.factors_  = None
        self.loadings_ = None
        self.mean_     = None

    def fit(self, returns: np.ndarray) -> FactorModelResults:
        """Fit standard PCA on demeaned R."""
        T, N = returns.shape
        K = self.n_factors

        # Demean (paper uses R - R̄)
        self.mean_ = returns.mean(axis=0)
        R = returns - self.mean_

        # SVD of demeaned returns
        U, S, Vt = np.linalg.svd(R, full_matrices=False)

        # ── Paper normalization: (1/N) β' β = I_K ────────────────────────────
        # β_PCA (N×K) = √N · Vt[:K,:].T
        # In K×N storage: loadings = √N · Vt[:K,:]
        self.loadings_ = np.sqrt(N) * Vt[:K, :]          # (K, N)

        # Normalize factor signs before computing F so sign convention is stable
        # F = (1/N) R β  =  R @ loadings.T / N
        self.factors_ = R @ self.loadings_.T / N          # (T, K)
        self.factors_ = _normalize_factor_signs(self.factors_)

        # Re-align loadings sign to match factors (flip columns where factor flipped)
        # After sign flip of factor k, the fitted value F[:,k] @ loadings[k,:] is
        # unchanged only if we also flip loadings[k,:].
        # Detect flips by comparing sign of original vs sign-normalized factor.
        F_unsigned = R @ (np.sqrt(N) * Vt[:K, :]).T / N
        for k in range(K):
            if np.sign(self.factors_[:, k].mean()) != np.sign(F_unsigned[:, k].mean()):
                self.loadings_[k, :] *= -1

        return self._compute_results(returns, R)

    def predict(self, returns: np.ndarray) -> FactorModelResults:
        """Project OOS returns onto fitted loadings."""
        R = returns - self.mean_
        N = self.loadings_.shape[1]                    # number of assets from training

        # (1/N) R β  with β = loadings.T   (using stored (1/N)β'β = I_K)
        factors = R @ self.loadings_.T / N             # (T_new, K)
        factors = _normalize_factor_signs(factors)

        # Loadings are fixed from training; recompute only for OOS metrics
        # Use the same paper formula: loadings unchanged (out-of-sample)
        loadings = self.loadings_.copy()

        return self._compute_results(returns, R, factors, loadings)

    def _compute_results(self, returns, R, factors=None, loadings=None):
        T, N = returns.shape
        K = self.n_factors

        if factors is None:
            factors = self.factors_
        if loadings is None:
            loadings = self.loadings_

        # ── Eigenvalues of Σ̂_R = (1/T) R'R  →  λ_k = S_k² / T ─────────────
        # Do NOT divide by N. These are the eigenvalues the paper refers to
        # when it says "K largest eigenvalues of Σ̂_R".
        _, S, Vt_full = np.linalg.svd(R, full_matrices=False)
        eigenvalues  = (S[:K]**2) / T                  # λ_k = S_k²/T  ✓
        eigenvectors = np.sqrt(N) * Vt_full[:K, :].T   # β_PCA  (N × K), paper convention

        # trace(Σ̂_R) = sum of ALL N eigenvalues of Σ̂_R = (1/T)||R||_F² = S²/T summed
        # This is the correct denominator for the Absorption Ratio (Eq. 1)
        # in PCA's own operator units.
        operator_trace = float(np.sum(S**2) / T)

        # Fitted values and residuals
        fitted          = factors @ loadings            # (T, N)
        residuals       = R - fitted
        unexplained_var = np.mean(residuals**2)

        # ── FMPs on raw returns — shared input for all three metrics ─────────
        fmps = _build_fmps(returns, loadings)

        rms_alpha, alphas, fmp_residuals = _compute_rms_alpha(returns, fmps)
        sigma_e          = _compute_sigma_e(fmp_residuals, returns)
        max_sr, max_sr_w = _compute_max_sharpe_ratio(fmps)
        r_squared        = 1.0 - unexplained_var / np.var(returns)

        explained_var_per_factor = _compute_explained_variance_per_factor(
            returns, factors, loadings)
        eigenvalues_normalized, var_unexplained = _compute_normalized_eigenvalues(
            returns, factors, loadings)

        return FactorModelResults(
            factors=factors, loadings=loadings, residuals=residuals,
            eigenvalues=eigenvalues, eigenvectors=eigenvectors,
            operator_trace=operator_trace,
            max_sharpe_ratio=max_sr, max_sr_weights=max_sr_w,
            rms_alpha=rms_alpha, sigma_e=sigma_e, alphas=alphas,
            r_squared=r_squared,
            explained_variance_per_factor=explained_var_per_factor,
            eigenvalues_normalized=eigenvalues_normalized,
            xs_metric=None,
            ts_metric=None,
            unexplained_variation=unexplained_var,
            unexplained_variance_total=var_unexplained
        )


# =============================================================================
# XS-TARGET PCA (Equation 5 / 7)
# =============================================================================

class XSTargetPCA:
    """XS-target PCA (Section 2.2).  M=(1/NT)W^{1/2}R'RW^{1/2}; F̂=unit-norm R·ψ; β̂=R'F̂; stores V_K."""

    def __init__(self, n_factors: int, gamma_xs: float = 1.0):
        self.n_factors       = n_factors
        self.gamma_xs        = gamma_xs
        self.factors_        = None
        self.loadings_       = None
        self.mean_           = None      # kept for interface compat with other classes
        self._psi            = None      # (N, K) W^{1/2} V_K — factor extractor
        self._V_K            = None      # (N, K) spectral eigenvectors of M_NN
        self._operator_trace = None      # scalar trace(M_NN)
        self._lambda_topk    = None      # top-K eigenvalues of M_NN — for predict()
        self._xs_target      = None      # stored for XS-Metric

    def fit(self, returns: np.ndarray, xs_target: np.ndarray) -> FactorModelResults:
        """Fit XS-target PCA on uncentered R with target Λ^{XS} (Prop 1)."""
        T, N = returns.shape
        K    = self.n_factors
        L    = xs_target.shape[1]

        # ── Paper convention (footnote 4): uncentered R ──────────────────────
        # We still record the sample mean for OOS centering interfaces.
        self.mean_ = returns.mean(axis=0)
        R = returns                                     # UNCENTERED

        self._xs_target = xs_target

        # ── Eq. 3: cross-sectional projection matrix ─────────────────────────
        P_Lambda = _projection_matrix(xs_target)        # (N, N)

        # ── W_xs = I_N + γ_XS (N/L) P_Λ  (from Eq. 5) ────────────────────────
        W_xs = np.eye(N) + self.gamma_xs * (N / L) * P_Lambda   # (N, N) SPD

        # ── Symmetric square root of W_xs (unique, reproducible) ─────────────
        # W_xs = Q Λ Q'  →  W_xs^{1/2} = Q Λ^{1/2} Q'  (symmetric, W^{1/2}·W^{1/2} = W).
        # The asymmetric form Q Λ^{1/2} satisfies W_sqrt · W_sqrt' = W but is
        # sign-sensitive across eigh runs — avoid it.
        eigvals_W, eigvecs_W = np.linalg.eigh(W_xs)
        sqrt_eigvals = np.sqrt(np.maximum(eigvals_W, 0))
        W_sqrt = (eigvecs_W * sqrt_eigvals) @ eigvecs_W.T      # (N, N) symmetric

        # ── Kernel-trick dual of Proposition 1 ───────────────────────────────
        # Paper's operator is the T×T matrix  M_TT = (1/NT) R W_xs R'.
        # With A = R W_xs^{1/2}:
        #     M_TT = A A' / NT
        #     M_NN = A' A / NT
        # share non-zero eigenvalues.  We eigendecompose the N×N form (cheaper
        # when N ≤ T) and lift the factors via the kernel-trick identity below.
        A    = R @ W_sqrt                               # (T, N)
        M_NN = A.T @ A / (N * T)                        # (N, N) symmetric PSD

        # trace(M_NN) = trace(M_TT) = sum of ALL eigenvalues — denominator for
        # the Absorption Ratio (Eq. 1) making AR comparable across models.
        operator_trace = float(np.trace(M_NN))

        # Eigendecomposition
        eigvals_all, eigvecs_all = np.linalg.eigh(M_NN)
        idx         = np.argsort(eigvals_all)[::-1]
        lam_all     = eigvals_all[idx]
        V_all       = eigvecs_all[:, idx]
        lam_topk    = lam_all[:K]                       # (K,)
        V_K         = V_all[:, :K]                      # (N, K)  unit-norm

        # Factor extractor  ψ = W^{1/2} V_K  (asset space, N×K)
        psi = W_sqrt @ V_K                              # (N, K)

        # ── Paper's factors: unit-norm T-space eigenvectors of M_TT ──────────
        # Kernel-trick lift:  F_TT[:, k] = (A V_K[:, k]) / ‖A V_K[:, k]‖.
        # Since A V_K = R W^{1/2} V_K = R ψ, this equals (R ψ) normalised.
        # The norm is √(NT · λ_k) because A V_K = A V and
        # ‖A V_K[:, k]‖² = V_K[:, k]' A' A V_K[:, k] = V_K[:, k]' (NT · M_NN) V_K[:, k]
        #                = NT · λ_k.
        F_unnorm  = R @ psi                              # (T, K)   = A V_K
        col_norms = np.linalg.norm(F_unnorm, axis=0, keepdims=True)
        col_norms = np.where(col_norms < 1e-12, 1.0, col_norms)
        F_paper   = F_unnorm / col_norms                 # (T, K)   unit-norm columns

        # Sign convention: positive mean, applied consistently
        F_paper = _normalize_factor_signs(F_paper)

        # ── Paper's loadings (from Proposition 1): β̂_XS = R' F̂_XS ───────────
        # Since F̂_XS has unit-norm columns, (F̂' F̂) = I_K and this equals OLS.
        beta_paper = R.T @ F_paper                       # (N, K)

        # Store in (K, N) convention
        self.factors_  = F_paper                         # (T, K)
        self.loadings_ = beta_paper.T                    # (K, N)

        # Internal state for centrality / predict()
        self._V_K            = V_K
        self._psi            = psi
        self._lambda_topk    = lam_topk
        self._operator_trace = operator_trace

        return self._compute_results(returns, R, lam_topk, V_K, operator_trace)

    def predict(self, returns: np.ndarray) -> FactorModelResults:
        """Project OOS (uncentered) returns onto training ψ; loadings fixed."""
        R = returns                                      # UNCENTERED

        F_unnorm = R @ self._psi
        col_norms = np.linalg.norm(F_unnorm, axis=0, keepdims=True)
        col_norms = np.where(col_norms < 1e-12, 1.0, col_norms)
        factors  = F_unnorm / col_norms
        factors  = _normalize_factor_signs(factors)

        loadings = self.loadings_                        # fixed from training

        # Eigenvalues not meaningful OOS (no training-subspace variance decomp)
        eigenvalues    = np.zeros(self.n_factors)
        eigenvectors   = self._V_K.copy()                # V_K for centrality
        operator_trace = self._operator_trace

        return self._compute_results(
            returns, R, eigenvalues, eigenvectors, operator_trace,
            factors=factors, loadings=loadings, is_oos=True,
        )

    def _compute_results(self, returns, R, eigenvalues, eigenvectors,
                         operator_trace,
                         factors=None, loadings=None, is_oos: bool = False):
        if factors is None:
            factors = self.factors_
        if loadings is None:
            loadings = self.loadings_

        fitted          = factors @ loadings             # (T, N)
        residuals       = R - fitted
        unexplained_var = np.mean(residuals**2)

        # FMPs on raw returns (shared input for Lettau-Pelger metrics)
        fmps = _build_fmps(returns, loadings)

        rms_alpha, alphas, fmp_residuals = _compute_rms_alpha(returns, fmps)
        sigma_e   = _compute_sigma_e(fmp_residuals, returns)
        max_sr, max_sr_weights = _compute_max_sharpe_ratio(fmps)
        r_squared = 1 - unexplained_var / np.var(returns)

        explained_var_per_factor = _compute_explained_variance_per_factor(
            returns, factors, loadings)
        eigenvalues_normalized, var_unexplained = _compute_normalized_eigenvalues(
            returns, factors, loadings)

        # XS-Metric (Eq. from paper Section 2.2): 100 · ||P_Λ ω̂|| / ||ω̂||.
        # The correct vector is the factor extractor ω̂ = ψ, not V_K.
        if self._xs_target is not None:
            factor_weights = self._psi if self._psi is not None else eigenvectors
            xs_metric = _compute_xs_metric(factor_weights, self._xs_target)
        else:
            xs_metric = None

        # Store TRUE spectral eigenvectors V_K (unit-norm columns) — the
        # correct object for the Kritzman et al. centrality formula Eq. 2.
        norms = np.linalg.norm(eigenvectors, axis=0, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        eigenvectors_norm = eigenvectors / norms

        return FactorModelResults(
            factors=factors, loadings=loadings, residuals=residuals,
            eigenvalues=eigenvalues, eigenvectors=eigenvectors_norm,
            operator_trace=operator_trace,
            max_sharpe_ratio=max_sr, max_sr_weights=max_sr_weights,
            rms_alpha=rms_alpha, sigma_e=sigma_e, alphas=alphas,
            r_squared=r_squared,
            explained_variance_per_factor=explained_var_per_factor,
            eigenvalues_normalized=eigenvalues_normalized,
            xs_metric=xs_metric,
            ts_metric=None,
            unexplained_variation=unexplained_var,
            unexplained_variance_total=var_unexplained,
        )


# =============================================================================
# TS-TARGET PCA (Equation 11)   *** FIXED ***
# =============================================================================

class TSTargetPCA:
    """TS-target PCA (Section 2.3).  M=(1/NT)R'W_ts R; F̂=R·ω̂; β̂=OLS; stores ω̂=eigvecs of M."""

    def __init__(self, n_factors: int, gamma_ts: float = 1.0):
        """
        Parameters
        ----------
        n_factors : int
        gamma_ts  : float  TS penalty weight (γ_TS in Eq. 7)
        """
        self.n_factors       = n_factors
        self.gamma_ts        = gamma_ts
        self.factors_        = None
        self.loadings_       = None
        self.mean_           = None    # kept for interface compat
        self._psi            = None    # (N, K) ω̂_TS — for predict() & centrality
        self._operator_trace = None    # scalar trace(M) — for comparable AR
        self._ts_target      = None    # stored for TS-Metric computation

    def fit(self, returns: np.ndarray, ts_target: np.ndarray) -> FactorModelResults:
        """Fit TS-target PCA on uncentered R with target G^{TS} (Prop 2)."""
        T, N = returns.shape
        K    = self.n_factors

        # ── Paper convention: uncentered R throughout ────────────────────────
        self.mean_      = returns.mean(axis=0)           # kept for OOS interface
        R               = returns                         # UNCENTERED
        self._ts_target = ts_target

        # ── Eq. 7 projection: P_G^{TS} = G(G'G)^{-1}G' ───────────────────────
        P_G = _projection_matrix(ts_target)              # (T, T)

        # ── Eq. 8: M = (1/NT) R' (I_T + γ_TS P_G) R ──────────────────────────
        W_ts = np.eye(T) + self.gamma_ts * P_G           # (T, T)
        M    = (R.T @ W_ts @ R) / (N * T)                # (N, N) symmetric PSD

        # trace(M) — denominator for comparable Absorption Ratio (Eq. 1)
        operator_trace = float(np.trace(M))
        self._operator_trace = operator_trace

        # ── Prop 2: ω̂_TS = top-K eigenvectors of M ──────────────────────────
        eigvals_all, eigvecs_all = np.linalg.eigh(M)
        idx          = np.argsort(eigvals_all)[::-1]
        eigenvalues  = eigvals_all[idx][:K]
        omega_ts     = eigvecs_all[:, idx][:, :K]        # (N, K)  unit-norm

        self._psi = omega_ts                              # ω̂_TS — used in predict()

        # ── Prop 2: F̂_TS = R · ω̂_TS  (UNCENTERED R, matching the paper) ────
        F_hat = R @ omega_ts                              # (T, K)
        F_hat = _normalize_factor_signs(F_hat)

        # ── Paper text (after Prop 2): β̂_TS = OLS(R on F̂_TS) ───────────────
        # "the loadings β̂_TS estimated by an OLS regression of R on F̂_TS"
        FtF_inv      = np.linalg.inv(F_hat.T @ F_hat)
        beta_hat     = FtF_inv @ F_hat.T @ R              # (K, N)

        self.factors_  = F_hat
        self.loadings_ = beta_hat

        return self._compute_results(returns, R, eigenvalues, omega_ts,
                                     operator_trace)

    def predict(self, returns: np.ndarray) -> FactorModelResults:
        """Project OOS (uncentered) returns onto training ω̂_TS; loadings fixed."""
        R = returns                                       # UNCENTERED

        factors = R @ self._psi                           # (T_oos, K)
        factors = _normalize_factor_signs(factors)

        loadings       = self.loadings_                   # fixed from training
        eigenvalues    = np.zeros(self.n_factors)
        eigenvectors   = self._psi.copy()                 # ω̂_TS for centrality
        operator_trace = self._operator_trace

        return self._compute_results(
            returns, R, eigenvalues, eigenvectors, operator_trace,
            factors=factors, loadings=loadings, is_oos=True,
        )

    def _compute_results(self, returns, R, eigenvalues, eigenvectors,
                         operator_trace,
                         factors=None, loadings=None, is_oos: bool = False):
        if factors is None:
            factors = self.factors_
        if loadings is None:
            loadings = self.loadings_

        fitted          = factors @ loadings              # (T, N)
        residuals       = R - fitted
        unexplained_var = np.mean(residuals**2)

        # FMPs (shared input for Lettau-Pelger metrics)
        fmps = _build_fmps(returns, loadings)

        rms_alpha, alphas, fmp_residuals = _compute_rms_alpha(returns, fmps)
        sigma_e   = _compute_sigma_e(fmp_residuals, returns)
        max_sr, max_sr_weights = _compute_max_sharpe_ratio(fmps)
        r_squared = 1 - unexplained_var / np.var(returns)

        explained_var_per_factor = _compute_explained_variance_per_factor(
            returns, factors, loadings)
        eigenvalues_normalized, var_unexplained = _compute_normalized_eigenvalues(
            returns, factors, loadings)

        # TS-Metric (paper Eq. 14):  100 · ||P_G R P_ω̂||_F / ||P_G R||_F
        # P_ω̂ is the N×N projection onto factor-portfolio weights ω̂.
        # In-sample only (OOS P_G is undefined because T changes).
        if self._ts_target is not None and not is_oos:
            ts_metric = _compute_ts_metric(
                factor_weights = self._psi,    # ω̂_TS, (N, K)
                ts_target      = self._ts_target,
                returns        = returns,
            )
        else:
            ts_metric = None

        return FactorModelResults(
            factors=factors, loadings=loadings, residuals=residuals,
            eigenvalues=eigenvalues, eigenvectors=eigenvectors,
            operator_trace=operator_trace,
            max_sharpe_ratio=max_sr, max_sr_weights=max_sr_weights,
            rms_alpha=rms_alpha, sigma_e=sigma_e, alphas=alphas,
            r_squared=r_squared,
            explained_variance_per_factor=explained_var_per_factor,
            eigenvalues_normalized=eigenvalues_normalized,
            xs_metric=None,
            ts_metric=ts_metric,
            unexplained_variation=unexplained_var,
            unexplained_variance_total=var_unexplained,
        )


# =============================================================================
# XS-TS-TARGET PCA (Equation 12)
# =============================================================================

class XSTSTargetPCA:
    """Combined XS-TS PCA (Section 2.4).  M=(1/NT)C R'W_ts R C; F̂=R·ω̂; stores Q_K."""

    def __init__(self, n_factors: int, gamma_xs: float = 1.0, gamma_ts: float = 1.0):
        self.n_factors       = n_factors
        self.gamma_xs        = gamma_xs
        self.gamma_ts        = gamma_ts
        self.factors_        = None
        self.loadings_       = None
        self.mean_           = None      # kept for interface compat
        self._omega          = None      # ω̂^{XS-TS} (N, K) — factor extractor
        self._Q_K            = None      # Q_K (N, K)       — spectral eigenvectors
        self._gamma_tilde    = None      # scalar γ̃_XS
        self._operator_trace = None      # scalar trace(M)
        self._xs_target      = None      # stored for XS-Metric
        self._ts_target      = None      # stored for TS-Metric

    def fit(self, returns: np.ndarray,
            xs_target: np.ndarray,
            ts_target: np.ndarray) -> FactorModelResults:
        """Fit combined XS-TS PCA on uncentered R (Prop 3, Eq. 11)."""
        T, N = returns.shape
        K    = self.n_factors
        L    = xs_target.shape[1]

        self._xs_target = xs_target
        self._ts_target = ts_target

        # ── Paper convention: uncentered R throughout ────────────────────────
        self.mean_ = returns.mean(axis=0)                # kept for OOS interface
        R          = returns                             # UNCENTERED

        # ── XS ingredients: γ̃_XS and C = I_N + γ̃_XS P_Λ^{XS}  (Prop 3) ─────
        P_Lambda    = _projection_matrix(xs_target)      # (N, N)
        gamma_tilde = np.sqrt(self.gamma_xs * N / L + 1) - 1
        C           = np.eye(N) + gamma_tilde * P_Lambda # (N, N) SPD
        self._gamma_tilde = gamma_tilde

        # ── TS ingredient: W_TS = I_T + γ_TS P_G^{TS} ────────────────────────
        P_G  = _projection_matrix(ts_target)             # (T, T)
        W_ts = np.eye(T) + self.gamma_ts * P_G           # (T, T)

        # ── Eq. 11: M = (1/NT) C R' W_TS R C   (symmetric N×N) ───────────────
        inner = (R.T @ W_ts @ R) / (N * T)               # (N, N)
        M     = C @ inner @ C                             # (N, N) symmetric PSD

        # Comparable Absorption Ratio: use trace(M) as denominator (Eq. 1).
        operator_trace = float(np.trace(M))
        self._operator_trace = operator_trace

        # ── Q_K: top-K eigenvectors of M (spectral) ──────────────────────────
        eigvals_all, eigvecs_all = np.linalg.eigh(M)
        idx          = np.argsort(eigvals_all)[::-1]
        eigenvalues  = eigvals_all[idx][:K]
        Q_K          = eigvecs_all[:, idx][:, :K]        # (N, K) unit-norm
        self._Q_K    = Q_K

        # ── Factor weights  ω̂^{XS-TS} = (C / (γ̃+1)) Q_K   (Eq. after 11) ────
        # Equivalent form (paper):
        #   ω̂ = ( 1/(γ̃+1) I_N + γ̃/(γ̃+1) P_Λ^{XS} ) Q_K
        # which equals C/(γ̃+1) · Q_K since C = I + γ̃ P_Λ.
        omega = (C / (gamma_tilde + 1.0)) @ Q_K          # (N, K)
        self._omega = omega

        # ── Factors (Prop 3): F̂_{XS-TS} = R · ω̂   (UNCENTERED R) ────────────
        F_hat = R @ omega                                 # (T, K)
        F_hat = _normalize_factor_signs(F_hat)

        # ── Loadings (Step 3, Eq. 12): OLS regression of R on F̂ ─────────────
        FtF_inv = np.linalg.inv(F_hat.T @ F_hat)
        beta    = FtF_inv @ F_hat.T @ R                   # (K, N)

        self.factors_  = F_hat
        self.loadings_ = beta

        return self._compute_results(returns, R, eigenvalues, Q_K,
                                     operator_trace)

    def predict(self, returns: np.ndarray) -> FactorModelResults:
        """Project OOS returns onto training ω̂^{XS-TS}; loadings fixed."""
        R = returns                                      # UNCENTERED

        factors = R @ self._omega                        # (T_oos, K)
        factors = _normalize_factor_signs(factors)

        loadings       = self.loadings_
        eigenvalues    = np.zeros(self.n_factors)
        eigenvectors   = self._Q_K.copy()                # Q_K for centrality
        operator_trace = self._operator_trace

        return self._compute_results(
            returns, R, eigenvalues, eigenvectors, operator_trace,
            factors=factors, loadings=loadings, is_oos=True,
        )

    def _compute_results(self, returns, R, eigenvalues, eigenvectors,
                         operator_trace,
                         factors=None, loadings=None, is_oos: bool = False):
        if factors is None:
            factors = self.factors_
        if loadings is None:
            loadings = self.loadings_

        fitted          = factors @ loadings             # (T, N)
        residuals       = R - fitted
        unexplained_var = np.mean(residuals**2)

        # FMPs (shared input for Lettau-Pelger metrics)
        fmps = _build_fmps(returns, loadings)

        rms_alpha, alphas, fmp_residuals = _compute_rms_alpha(returns, fmps)
        sigma_e   = _compute_sigma_e(fmp_residuals, returns)
        max_sr, max_sr_weights = _compute_max_sharpe_ratio(fmps)
        r_squared = 1 - unexplained_var / np.var(returns)

        explained_var_per_factor = _compute_explained_variance_per_factor(
            returns, factors, loadings)
        eigenvalues_normalized, var_unexplained = _compute_normalized_eigenvalues(
            returns, factors, loadings)

        # ── XS-Metric (paper, Section 2.2): 100 · ||P_Λ ω̂|| / ||ω̂|| ─────────
        # ω̂ is the factor-extraction weight, not Q_K.
        xs_metric = (
            _compute_xs_metric(self._omega, self._xs_target)
            if self._xs_target is not None else None
        )

        # ── TS-Metric (paper Eq. 14): 100 · ||P_G R P_ω̂||_F / ||P_G R||_F ──
        # P_ω̂ is the N×N projection onto ω̂. In-sample only.
        if self._ts_target is not None and not is_oos:
            ts_metric = _compute_ts_metric(
                factor_weights = self._omega,
                ts_target      = self._ts_target,
                returns        = returns,
            )
        else:
            ts_metric = None

        # Store TRUE spectral eigenvectors Q_K (unit-norm) for centrality
        norms = np.linalg.norm(eigenvectors, axis=0, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        eigenvectors_norm = eigenvectors / norms

        return FactorModelResults(
            factors=factors, loadings=loadings, residuals=residuals,
            eigenvalues=eigenvalues, eigenvectors=eigenvectors_norm,
            operator_trace=operator_trace,
            max_sharpe_ratio=max_sr, max_sr_weights=max_sr_weights,
            rms_alpha=rms_alpha, sigma_e=sigma_e, alphas=alphas,
            r_squared=r_squared,
            explained_variance_per_factor=explained_var_per_factor,
            eigenvalues_normalized=eigenvalues_normalized,
            xs_metric=xs_metric,
            ts_metric=ts_metric,
            unexplained_variation=unexplained_var,
            unexplained_variance_total=var_unexplained,
        )


# =============================================================================
# ONATSKI TEST
# =============================================================================

def onatski_test(returns: np.ndarray, max_factors: int = 20,
                 delta: float = 0.01) -> int:
    """Onatski (2010) ED test for K; returns estimated number of factors."""
    T, N = returns.shape
    R = returns - returns.mean(axis=0)

    _, S, _ = np.linalg.svd(R, full_matrices=False)
    eigenvalues = (S**2) / (N * T)

    K = max_factors
    for j in range(2, min(max_factors, len(eigenvalues)) + 1):
        lambda_j   = eigenvalues[j - 1]
        lambda_jp1 = eigenvalues[j]

        sum_diffs = sum(
            (eigenvalues[i - 1] - eigenvalues[i])**2
            for i in range(j + 1, min(j + 5, len(eigenvalues)))
        )
        delta_sq_hat = (1 / 4) * sum_diffs if sum_diffs > 0 else 0

        if (lambda_j - lambda_jp1) < delta * delta_sq_hat:
            K = j - 1
            break

    return K


# =============================================================================
# CROSS-VALIDATION EXTENSION   *** FIXED ***
# =============================================================================

@dataclass
class CVResult:
    """Aggregated cross-validation performance for one configuration"""
    gamma_xs: Optional[float]
    gamma_ts: Optional[float]
    n_factors: int
    mean_sigma_e: float
    std_sigma_e: float
    mean_max_sharpe: float
    mean_rms_alpha: float
    mean_r_squared: float


def time_series_cv_splits(
    n_time: int,
    n_splits: int = 5,
    min_train_frac: float = 0.6,
    test_frac: float = 0.1,
    purge_frac: float = 0.02
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Yield rolling (train, test) index pairs for time-series CV."""
    splits = []
    min_train  = int(n_time * min_train_frac)
    test_size  = max(12, int(n_time * test_frac))
    purge_size = max(1,  int(n_time * purge_frac))

    test_start_points = np.linspace(
        min_train + purge_size,
        n_time - test_size,
        n_splits,
        dtype=int
    )

    for test_start in test_start_points:
        train_idx = np.arange(0, test_start - purge_size)
        test_idx  = np.arange(test_start, min(test_start + test_size, n_time))

        if len(train_idx) < 20 or len(test_idx) < 6:
            continue

        splits.append((train_idx, test_idx))

    return splits


def cross_validate_target_model(
    returns: np.ndarray,
    xs_target: Optional[np.ndarray] = None,
    ts_target: Optional[np.ndarray] = None,
    model_class: type = XSTSTargetPCA,
    gamma_xs_candidates: Union[List[float], np.ndarray] = (0.0, 0.5, 1.0, 2.0, 5.0),
    gamma_ts_candidates: Union[List[float], np.ndarray] = (0.0, 0.5, 1.0, 2.0, 5.0),
    n_factors_candidates: Union[List[int], np.ndarray]  = (1, 2, 3, 4, 5, 6, 8),
    n_splits: int = 5,
    primary_metric: str = 'sigma_e',   # 'sigma_e' (↓), 'max_sharpe' (↑), 'rms_alpha' (↓)
    verbose: bool = False
) -> Dict[Tuple[Optional[float], Optional[float], int], CVResult]:
    """Grid-search over γ for a target-PCA class; returns a CVResult."""
    T, N = returns.shape

    gamma_xs_cand = np.asarray(gamma_xs_candidates)
    gamma_ts_cand = np.asarray(gamma_ts_candidates)
    K_cand        = np.asarray(n_factors_candidates, dtype=int)

    cv_splits = time_series_cv_splits(T, n_splits=n_splits)

    cv_results: Dict[Tuple[Optional[float], Optional[float], int], CVResult] = {}

    for g_xs in gamma_xs_cand:
        for g_ts in gamma_ts_cand:
            for K in K_cand:

                if verbose:
                    print(f"  CV → γ_xs={g_xs:4.2f} | γ_ts={g_ts:4.2f} | K={K:2d}   ", end="")

                sigma_e_list   = []
                max_sr_list    = []
                rms_alpha_list = []
                r2_list        = []

                for train_idx, test_idx in cv_splits:
                    ret_tr = returns[train_idx]
                    ret_te = returns[test_idx]

                    # ── FIX: xs_target is (N, L) — no time slicing needed ─────
                    xs_tr = xs_target                  # None or full (N, L)

                    # ts_target is (T, J) — slice to training window
                    ts_tr = None if ts_target is None else ts_target[train_idx]

                    # Instantiate and fit
                    if model_class is PCA:
                        model  = PCA(n_factors=K)
                        model.fit(ret_tr)
                    elif model_class is XSTargetPCA:
                        model  = XSTargetPCA(n_factors=K, gamma_xs=g_xs)
                        model.fit(ret_tr, xs_tr)
                    elif model_class is TSTargetPCA:
                        model  = TSTargetPCA(n_factors=K, gamma_ts=g_ts)
                        model.fit(ret_tr, ts_tr)
                    else:  # XSTSTargetPCA
                        model  = XSTSTargetPCA(n_factors=K, gamma_xs=g_xs, gamma_ts=g_ts)
                        model.fit(ret_tr, xs_tr, ts_tr)

                    # Evaluate out-of-sample
                    res_oos = model.predict(ret_te)

                    sigma_e_list.append(res_oos.sigma_e)
                    max_sr_list.append(res_oos.max_sharpe_ratio)
                    rms_alpha_list.append(res_oos.rms_alpha)
                    r2_list.append(res_oos.r_squared)

                mean_sigma = np.mean(sigma_e_list)
                mean_sr    = np.mean(max_sr_list)
                mean_rms   = np.mean(rms_alpha_list)
                mean_r2    = np.mean(r2_list)

                key = (
                    float(g_xs) if g_xs > 0 else None,
                    float(g_ts) if g_ts > 0 else None,
                    int(K)
                )

                cv_results[key] = CVResult(
                    gamma_xs=float(g_xs) if g_xs > 0 else None,
                    gamma_ts=float(g_ts) if g_ts > 0 else None,
                    n_factors=K,
                    mean_sigma_e=mean_sigma,
                    std_sigma_e=np.std(sigma_e_list),
                    mean_max_sharpe=mean_sr,
                    mean_rms_alpha=mean_rms,
                    mean_r_squared=mean_r2
                )

                if verbose:
                    print(f"   σ_e = {mean_sigma:.4f}   SR = {mean_sr:.3f}")

    return cv_results


def select_best_configuration(
    cv_results: Dict[Tuple[Optional[float], Optional[float], int], CVResult],
    primary: str = 'sigma_e',
    secondary: Optional[str] = 'max_sharpe'
) -> Tuple[Optional[float], Optional[float], int, CVResult]:
    """
    Select optimal (gamma_xs, gamma_ts, K) by primary and secondary metrics.

    primary / secondary choices: 'sigma_e' (↓), 'rms_alpha' (↓), 'max_sharpe' (↑)
    """
    best_key    = None
    best_score  = np.inf if primary in ('sigma_e', 'rms_alpha') else -np.inf
    best_result = None

    lower_is_better = primary in ('sigma_e', 'rms_alpha')

    for key, res in cv_results.items():
        val = getattr(res, f'mean_{primary}')

        is_better = (val < best_score) if lower_is_better else (val > best_score)

        if is_better or best_key is None:
            best_key    = key
            best_score  = val
            best_result = res
            continue

        # Secondary tie-breaker
        if abs(val - best_score) < 1e-6 and secondary is not None:
            sec_val  = getattr(res, f'mean_{secondary}')
            best_sec = getattr(best_result, f'mean_{secondary}')
            sec_lower = secondary in ('sigma_e', 'rms_alpha')
            if (sec_val < best_sec) if sec_lower else (sec_val > best_sec):
                best_key    = key
                best_score  = val
                best_result = res

    g_xs, g_ts, K = best_key
    return g_xs, g_ts, K, best_result
   
# =============================================================================
# EXAMPLE USAGE (commented out)
# =============================================================================
"""
if __name__ == "__main__":
    # returns    : np.ndarray (T, N)
    # xs_target  : np.ndarray (N, L)  or None
    # ts_target  : np.ndarray (T, J)  or None  — pass RAW (not pre-demeaned)

    cv_dict = cross_validate_target_model(
        returns=returns,
        xs_target=xs_target,
        ts_target=ts_target,
        model_class=XSTSTargetPCA,
        gamma_xs_candidates=np.linspace(0, 4, 9),
        gamma_ts_candidates=np.linspace(0, 3, 7),
        n_factors_candidates=range(1, 9),
        n_splits=5,
        primary_metric='sigma_e',
        verbose=True,
    )

    best_g_xs, best_g_ts, best_K, best_res = select_best_configuration(cv_dict)

    print("\\nOptimal configuration:")
    print(f"  γ_xs   = {best_g_xs}")
    print(f"  γ_ts   = {best_g_ts}")
    print(f"  K      = {best_K}")
    print(f"  σ_e    = {best_res.mean_sigma_e:.4f} ± {best_res.std_sigma_e:.4f}")
    print(f"  max SR = {best_res.mean_max_sharpe:.3f}")
    print(f"  RMS α  = {best_res.mean_rms_alpha:.2f} bps")
"""