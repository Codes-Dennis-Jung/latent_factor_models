"""Verification of metrics, HAC, inference and factor-number estimators against naive formulas."""

import numpy as np
import pytest

import latent_factor_models as lfm
from latent_factor_models.evaluation import metrics
from latent_factor_models.inference.hac import long_run_covariance, newey_west_lag
from latent_factor_models.inference.three_pass import pi_matrices
from latent_factor_models.selection import bai_ng, eigenvalue_ratio, giglio_xiu, onatski


@pytest.fixture(scope="module")
def sim():
    return lfm.simulate_factor_returns(T=300, N=50, seed=4)


def test_max_sharpe_closed_form(sim):
    F = sim.factors
    mu, S = F.mean(0), np.cov(F, rowvar=False)
    assert np.isclose(metrics.max_sharpe_ratio(F), np.sqrt(mu @ np.linalg.solve(S, mu)))


def test_in_sample_alpha_equals_ols_intercepts(sim):
    R = sim.returns
    m = lfm.PCA(3).fit(R)
    e, alpha = metrics.pricing_errors(R, m.factors_, m.loadings_)
    np.testing.assert_allclose(alpha, m.intercepts_, atol=1e-10)
    rms = np.sqrt(np.mean(m.intercepts_**2))
    assert np.isclose(m.evaluate()["rms_alpha"], rms)
    sig = np.mean(e.var(0, ddof=1) / R.var(0, ddof=1))
    assert np.isclose(m.evaluate()["unexplained_variation"], sig)


def test_xs_ts_metric_naive(sim):
    R, Lam = sim.returns, sim.xs_target
    w = np.random.default_rng(1).standard_normal((R.shape[1], 2))
    P = Lam @ np.linalg.inv(Lam.T @ Lam) @ Lam.T
    np.testing.assert_allclose(
        metrics.xs_metric(w, Lam), 100 * np.linalg.norm(P @ w, axis=0) / np.linalg.norm(w, axis=0)
    )
    G = sim.reference
    PG = G @ np.linalg.inv(G.T @ G) @ G.T
    Pw = w @ np.linalg.inv(w.T @ w) @ w.T
    naive = 100 * np.linalg.norm(PG @ R @ Pw) / np.linalg.norm(PG @ R)
    assert np.isclose(metrics.ts_metric(R, w, G), naive)


def test_long_run_covariance_loop():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((120, 3))
    q = 4
    T = A.shape[0]
    S = sum(np.outer(A[t], A[t]) for t in range(T)) / T
    for s in range(1, q + 1):
        w = 1 - s / (q + 1)
        for t in range(s, T):
            S += w * (np.outer(A[t], A[t - s]) + np.outer(A[t - s], A[t])) / T
    np.testing.assert_allclose(long_run_covariance(A, q), S, atol=1e-12)


def test_pi_matrices_match_legacy_loop():
    """Vectorised Pi matrices equal the loop of the legacy code (column-major vec)."""
    rng = np.random.default_rng(2)
    T, p, d, q = 80, 2, 3, 3
    V, Z = rng.standard_normal((T, p)), rng.standard_normal((T, d))
    pis = pi_matrices(V, Z, q)
    vec = lambda m: m.flatten("F")  # noqa: E731
    P11 = np.zeros((d * p, d * p))
    P12 = np.zeros((d * p, p))
    P13 = np.zeros((d * p, d))
    for t in range(T):
        a = vec(np.outer(Z[t], V[t]))
        P11 += np.outer(a, a) / T
        P12 += np.outer(a, V[t]) / T
        P13 += np.outer(a, Z[t]) / T
        for s in range(1, min(t, q) + 1):
            w = (1 - s / (q + 1)) / T
            b = vec(np.outer(Z[t - s], V[t - s]))
            P11 += w * (np.outer(a, b) + np.outer(b, a))
            P12 += w * (np.outer(a, V[t - s]) + np.outer(b, V[t]))
            P13 += w * (np.outer(a, Z[t - s]) + np.outer(b, Z[t]))
    np.testing.assert_allclose(pis.pi11, P11, atol=1e-12)
    np.testing.assert_allclose(pis.pi12, P12, atol=1e-12)
    np.testing.assert_allclose(pis.pi13, P13, atol=1e-12)


def test_newey_west_lag_rules():
    assert newey_west_lag(100) == 4
    assert newey_west_lag(512, "stock-watson") == 6
    with pytest.raises(ValueError):
        newey_west_lag(1)


def test_three_pass_rotation_invariance(sim):
    """Gamma_g and its s.e. do not depend on the first-pass normalisation (PCA vs RP-PCA(-1))."""
    a = lfm.ThreePassRiskPremia(lfm.PCA(3)).fit(sim.returns, sim.observable)
    b = lfm.ThreePassRiskPremia(lfm.RPPCA(3, gamma=-1)).fit(sim.returns, sim.observable)
    np.testing.assert_allclose(a.gamma_g, b.gamma_g, rtol=1e-8)
    np.testing.assert_allclose(a.se, b.se, rtol=1e-8)


def test_three_pass_coverage_monte_carlo():
    """95% confidence intervals of gamma_g cover the true premium roughly 95% of the time."""
    hits = []
    for s in range(60):
        sim = lfm.simulate_factor_returns(T=400, N=60, seed=100 + s)
        r = lfm.ThreePassRiskPremia(lfm.PCA(3)).fit(sim.returns, sim.observable[:, :1])
        true = sim.risk_premia[0]  # proxy = factor 1 + noise
        hits.append(abs(r.gamma_g[0] - true) <= 1.96 * r.se[0])
    assert 0.85 <= np.mean(hits) <= 1.0


def _spiked_eigenvalues(k, rng, n=200, T=400):
    X = rng.standard_normal((T, n))
    F = rng.standard_normal((T, k)) * np.linspace(6, 3, k)
    X += F @ rng.standard_normal((k, n))
    Xc = X - X.mean(0)
    return np.linalg.svd(Xc, compute_uv=False) ** 2 / (n * T), X


@pytest.mark.parametrize("k", [1, 3, 5])
def test_number_of_factor_estimators(k):
    rng = np.random.default_rng(k)
    lam, X = _spiked_eigenvalues(k, rng)
    assert onatski(lam, r_max=10) == k
    assert eigenvalue_ratio(lam) == k
    assert bai_ng(X, k_max=10) == k
    assert giglio_xiu(lam, X.shape[1], X.shape[0]) == k
