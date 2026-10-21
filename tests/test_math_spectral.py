"""Independent verification of the spectral estimators against brute-force formulas.

Every test rebuilds the paper's operator naively (explicit T x T / N x N
matrices, matrix inverses, scipy.linalg.sqrtm) and compares it with the
efficient engine.
"""

import numpy as np
import pytest
import scipy.linalg as sla

import latent_factor_models as lfm
from latent_factor_models.models.targets import alpha_target


def subspace_equal(A, B, atol=1e-8):
    """True if col(A) == col(B) (via projectors)."""
    PA = A @ np.linalg.pinv(A)
    PB = B @ np.linalg.pinv(B)
    return np.allclose(PA, PB, atol=atol)


@pytest.fixture(scope="module")
def sim():
    return lfm.simulate_factor_returns(T=240, N=40, seed=11)


def naive_operator(R, G=None, gamma_ts=0.0, Lam=None, gamma_xs=0.0):
    T, N = R.shape
    W_ts = np.eye(T)
    if G is not None:
        W_ts = W_ts + gamma_ts * G @ np.linalg.inv(G.T @ G) @ G.T
    W_xs_sqrt = np.eye(N)
    if Lam is not None:
        L = Lam.shape[1]
        P = Lam @ np.linalg.inv(Lam.T @ Lam) @ Lam.T
        W_xs_sqrt = np.real(sla.sqrtm(np.eye(N) + gamma_xs * N / L * P))
    return W_xs_sqrt @ R.T @ W_ts @ R @ W_xs_sqrt / (N * T), W_xs_sqrt


def top_vecs(M, k):
    vals, vecs = np.linalg.eigh(M)
    order = np.argsort(vals)[::-1][:k]
    return vals[order], vecs[:, order]


def test_pca_equals_covariance_eigenvectors(sim):
    R = sim.returns
    m = lfm.PCA(3).fit(R)
    cov = np.cov(R, rowvar=False, ddof=0)
    vals, V = top_vecs(cov, 3)
    assert subspace_equal(m.weights_, V)
    np.testing.assert_allclose(m.eigenvalues_ * R.shape[1], vals, rtol=1e-10)


def test_uncentered_pca_is_second_moment(sim):
    R = sim.returns
    m = lfm.PCA(3, centered=False).fit(R)
    assert subspace_equal(m.weights_, top_vecs(R.T @ R, 3)[1])


@pytest.mark.parametrize("gamma", [-1.0, 0.0, 10.0, 50.0])
def test_rppca_operator(sim, gamma):
    R = sim.returns
    T, N = R.shape
    m = lfm.RPPCA(3, gamma=gamma).fit(R)
    rbar = R.mean(axis=0)
    M = R.T @ R / T + gamma * np.outer(rbar, rbar)  # Lettau-Pelger
    vals, V = top_vecs(M, 3)
    assert subspace_equal(m.weights_, V)
    np.testing.assert_allclose(m.eigenvalues_ * N, vals, rtol=1e-10)


def test_rppca_minus_one_is_pca(sim):
    a = lfm.RPPCA(3, gamma=-1).fit(sim.returns)
    b = lfm.PCA(3).fit(sim.returns)
    np.testing.assert_allclose(a.weights_, b.weights_, atol=1e-10)


def test_ts_target_mean_equals_rppca(sim):
    R = sim.returns
    a = lfm.TSTargetPCA(3, gamma_ts=7.0, ts_target="mean").fit(R)
    b = lfm.RPPCA(3, gamma=7.0).fit(R)
    np.testing.assert_allclose(a.weights_, b.weights_, atol=1e-10)


def test_ts_target_operator(sim):
    R = sim.returns
    G = alpha_target(sim.reference)
    m = lfm.TSTargetPCA(3, gamma_ts=15.0).fit(R, factors=sim.reference)
    M, _ = naive_operator(R, G=G, gamma_ts=15.0)
    vals, V = top_vecs(M, 3)
    np.testing.assert_allclose(m.operator_, M, atol=1e-10)
    assert subspace_equal(m.weights_, V)
    np.testing.assert_allclose(m.eigenvalues_, vals, rtol=1e-10)


def test_xs_target_matches_paper_both_forms(sim):
    R, Lam = sim.returns, sim.xs_target
    T, N = R.shape
    m = lfm.XSTargetPCA(3, gamma_xs=0.5).fit(R, xs_target=Lam)
    M, Wsqrt = naive_operator(R, Lam=Lam, gamma_xs=0.5)
    # N x N form (A3): omega = W^{1/2} Q_K up to scale
    vals, Q = top_vecs(M, 3)
    assert subspace_equal(m.weights_, Wsqrt @ Q)
    np.testing.assert_allclose(m.eigenvalues_, vals, rtol=1e-9)
    # T x T form (Prop. 1 / A4): factors = top eigenvectors of R W R'/(NT)
    P = Lam @ np.linalg.inv(Lam.T @ Lam) @ Lam.T
    W = np.eye(N) + 0.5 * N / Lam.shape[1] * P
    _, F_tt = top_vecs(R @ W @ R.T / (N * T), 3)
    assert subspace_equal(m.factors_, F_tt)
    # C^2 = W_XS closed form
    np.testing.assert_allclose(Wsqrt @ Wsqrt, W, atol=1e-9)
    gt = np.sqrt(0.5 * N / Lam.shape[1] + 1) - 1
    np.testing.assert_allclose(np.eye(N) + gt * P, Wsqrt, atol=1e-8)


def test_xsts_target_solves_weighted_low_rank_problem(sim):
    """Objective (11): min tr(W_TS E W_XS E'), E = R - F beta'.

    The optimum is F beta' = R C Q Q' C^{-1}.
    """
    R, Lam = sim.returns, sim.xs_target
    G = alpha_target(sim.reference)
    m = lfm.XSTSTargetPCA(2, gamma_xs=0.3, gamma_ts=5.0).fit(
        R, xs_target=Lam, factors=sim.reference
    )
    M, C = naive_operator(R, G=G, gamma_ts=5.0, Lam=Lam, gamma_xs=0.3)
    vals, Q = top_vecs(M, 2)
    assert subspace_equal(m.weights_, C @ Q)
    np.testing.assert_allclose(m.eigenvalues_, vals, rtol=1e-9)

    # the estimator attains the minimum of the weighted objective among rank-2 fits
    T, _N = R.shape
    Wts = np.eye(T) + 5.0 * G @ np.linalg.inv(G.T @ G) @ G.T
    Wxs = C @ C

    def objective(fit):
        E = R - fit
        return np.trace(Wts @ E @ Wxs @ E.T)

    best = objective(R @ C @ Q @ Q.T @ np.linalg.inv(C))
    rng = np.random.default_rng(0)
    for _ in range(20):
        Qr = np.linalg.qr(Q + 0.05 * rng.standard_normal(Q.shape))[0]
        assert objective(R @ C @ Qr @ Qr.T @ np.linalg.inv(C)) >= best - 1e-8


def test_alpha_target_reproduces_intercepts(sim):
    """R'G/T equals the time-series alphas of the assets on the reference factors (eq. 10)."""
    R, F = sim.returns, sim.reference
    T = R.shape[0]
    G = alpha_target(F)
    X = np.column_stack([np.ones(T), F])
    alphas = np.linalg.lstsq(X, R, rcond=None)[0][0]
    np.testing.assert_allclose(R.T @ G[:, 0] / T, alphas, atol=1e-10)


def test_projector_invariance_of_targets(sim):
    """Rotating/rescaling a target leaves the estimator unchanged."""
    R, Lam = sim.returns, sim.xs_target
    a = lfm.XSTargetPCA(3, 1.0).fit(R, xs_target=Lam)
    b = lfm.XSTargetPCA(3, 1.0).fit(R, xs_target=-3.7 * Lam)
    np.testing.assert_allclose(a.weights_, b.weights_, atol=1e-10)


def test_oos_factors_are_fixed_weights(sim):
    R = sim.returns
    m = lfm.TSTargetPCA(3).fit(R[:200], factors=sim.reference[:200])
    np.testing.assert_allclose(m.transform(R[200:]), R[200:] @ m.weights_)
    # single-period OOS factor is not renormalised (legacy bug)
    np.testing.assert_allclose(m.transform(R[200:201]), R[200:201] @ m.weights_)


def test_loadings_are_ols_with_intercept(sim):
    R = sim.returns
    m = lfm.RPPCA(3).fit(R)
    X = np.column_stack([np.ones(R.shape[0]), m.factors_])
    coef = np.linalg.lstsq(X, R, rcond=None)[0]
    np.testing.assert_allclose(m.intercepts_, coef[0], atol=1e-10)
    np.testing.assert_allclose(m.loadings_, coef[1:].T, atol=1e-10)


def test_factor_signs_positive_mean_and_consistent(sim):
    m = lfm.PCA(3).fit(sim.returns)
    assert np.all(m.factors_.mean(axis=0) >= 0)
    np.testing.assert_allclose(m.factors_, sim.returns @ m.weights_)


def test_standardized_rppca_matches_weighted_operator(sim):
    """Lettau-Pelger empirical version: PCA of Q X'(I + gamma/T 11')X Q, Q = diag(1/sigma)."""
    R = sim.returns
    T, _N = R.shape
    s = R.std(axis=0)
    Xs = R / s
    M = Xs.T @ (np.eye(T) + 10.0 / T * np.ones((T, T))) @ Xs / T
    _, Q = top_vecs(M, 3)
    m = lfm.RPPCA(3, gamma=10.0, standardize=True).fit(R)
    assert subspace_equal(m.weights_, Q / s[:, None])
    np.testing.assert_allclose(m.transform(R), R @ m.weights_)


def test_projection_loadings(sim):
    R = sim.returns
    m = lfm.RPPCA(3, gamma=10.0).fit(R)
    W = m.weights_
    fitted = m.transform(R) @ m.projection_loadings().T
    np.testing.assert_allclose(fitted, R @ W @ np.linalg.pinv(W), atol=1e-10)
    # for covariance PCA the OLS betas coincide with the projection loadings
    p = lfm.PCA(3).fit(R)
    np.testing.assert_allclose(p.loadings_, p.projection_loadings(), atol=1e-8)
    ev = m.evaluate(R[:50], loadings="projection")
    assert ev["rms_alpha"] >= 0


def test_generalized_correlation_and_variance_signal(sim):
    from latent_factor_models.evaluation import generalized_correlation

    rot = np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 0.0], [3.0, 0.0, -1.0]])
    gc = generalized_correlation(sim.factors @ rot, sim.factors)
    np.testing.assert_allclose(gc, 1.0, atol=1e-10)
    m = lfm.RPPCA(3, gamma=10.0).fit(sim.returns)
    vs = m.variance_signal()
    assert vs.shape == (3,) and np.all(np.diff(vs) <= 0) and vs[-1] > 0
