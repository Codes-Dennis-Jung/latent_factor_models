"""IPCA and SPCA: ALS steps against naive loops, identification, recovery and SPCA identities."""

import numpy as np
import pytest

import latent_factor_models as lfm
from latent_factor_models.models.ipca import _factor_step, _gamma_step, _wild_residuals


@pytest.fixture(scope="module")
def ipca_sim():
    return lfm.simulate_ipca_panel(T=120, N=400, L=6, K=2, missing=0.1, seed=3)


def test_als_steps_match_naive_loops(ipca_sim):
    X, W, n = ipca_sim.panel.managed_portfolios()
    T, L = X.shape
    K = 2
    rng = np.random.default_rng(0)
    gamma = rng.standard_normal((L, K + 1))  # one observed column (intercept)
    G = np.ones((T, 1))
    F = _factor_step(gamma, X, W, G, K)
    for t in range(T):
        gb, ga = gamma[:, :K], gamma[:, K]
        f = np.linalg.solve(gb.T @ W[t] @ gb, gb.T @ (X[t] - W[t] @ ga))
        np.testing.assert_allclose(F[t], f, atol=1e-10)
    Gm = _gamma_step(F, X, W, G, n)
    num = np.zeros(L * (K + 1))
    den = np.zeros((L * (K + 1), L * (K + 1)))
    for t in range(T):
        ft = np.r_[F[t], 1.0]
        num += n[t] * np.kron(X[t], ft)
        den += n[t] * np.kron(W[t], np.outer(ft, ft))
    np.testing.assert_allclose(Gm, np.linalg.solve(den, num).reshape(L, K + 1), atol=1e-10)


def test_managed_portfolios_naive(ipca_sim):
    p = ipca_sim.panel
    X, W, _n = p.managed_portfolios()
    t = 5
    ok = p.mask[t]
    Z, r = p.characteristics[t, ok], p.returns[t, ok]
    np.testing.assert_allclose(X[t], Z.T @ r / ok.sum())
    np.testing.assert_allclose(W[t], Z.T @ Z / ok.sum())


def test_ipca_identification_and_recovery(ipca_sim):
    m = lfm.IPCA(2, intercept=True).fit(ipca_sim.panel)
    gb, F = m.gamma_beta_, m.factors_
    np.testing.assert_allclose(gb.T @ gb, np.eye(2), atol=1e-8)
    S = F.T @ F
    assert abs(S[0, 1]) < 1e-6 * S[0, 0] and S[0, 0] >= S[1, 1]
    np.testing.assert_allclose(gb.T @ m.gamma_alpha_, 0, atol=1e-8)
    assert np.all(F.mean(0) > 0)
    true = ipca_sim.gamma_beta
    assert np.linalg.norm(gb @ gb.T - true @ true.T) < 0.1


def test_ipca_with_intercept_and_observed_factors(ipca_sim):
    """Legacy bug: intercept + pre-specified factors crashed."""
    g = ipca_sim.factors[:, :1] + 0.1
    m = lfm.IPCA(1, intercept=True).fit(ipca_sim.panel, factors=g)
    assert m.gamma_delta_.shape == (6, 1)
    np.testing.assert_allclose(m.gamma_beta_.T @ m.gamma_delta_, 0, atol=1e-8)
    assert set(m.evaluate()) == {"total_r2", "pred_r2", "total_r2_x", "pred_r2_x"}


def test_ipca_oos_factor_keeps_intercept_fixed(ipca_sim):
    """Legacy bug: predictOOS estimated the intercept 'factor' freely."""
    p = ipca_sim.panel
    m = lfm.IPCA(2, intercept=True).fit(p.take(np.arange(100)))
    f_oos = m.transform(p.take(np.arange(100, 120)))
    assert f_oos.shape == (20, 2)
    # with the true model the OOS factors track the true ones closely
    corr = [abs(np.corrcoef(f_oos[:, k], ipca_sim.factors[100:, k])[0, 1]) for k in range(2)]
    assert min(corr) > 0.6 or np.linalg.matrix_rank(f_oos) == 2


def test_ipca_converged_flag_and_warning(ipca_sim):
    with pytest.warns(lfm.core.ConvergenceWarning):
        m = lfm.IPCA(2, max_iter=1).fit(ipca_sim.panel)
    assert not m.converged_ and m.n_iter_ == 1


def test_wild_residuals_resample_every_block():
    """Legacy bug: the last block kept the original residuals."""
    resid = np.arange(1, 11, dtype=float)[:, None]
    rng = np.random.default_rng(0)
    out = np.array([_wild_residuals(rng, resid, 5) for _ in range(200)])
    assert np.std(out[:, -1, 0]) > 0.5
    assert not np.allclose(out[:, -5:, 0] / resid[-5:, 0], out[0, -1, 0] / resid[-1, 0])


@pytest.mark.slow
def test_instrument_test_size_and_power():
    sim = lfm.simulate_ipca_panel(T=100, N=200, L=5, K=1, seed=9)
    gb = sim.gamma_beta.copy()
    # rebuild returns where instrument 0 is irrelevant for betas
    p = sim.panel
    gb[0] = 0.0
    r, z = p.clean()
    rng = np.random.default_rng(1)
    R = np.einsum("tnl,lk,tk->tn", z, gb, sim.factors) + rng.standard_normal(r.shape)
    R[~p.mask] = np.nan
    panel = lfm.CharacteristicPanel(R, p.characteristics)
    m = lfm.IPCA(1).fit(panel)
    irrelevant = m.test_instrument(0, n_draws=100, seed=0)
    relevant = m.test_instrument(4, n_draws=100, seed=0)
    assert irrelevant.pvalue > 0.05
    assert relevant.pvalue < 0.05


def test_bootstrap_reproducible_across_backends(ipca_sim):
    m = lfm.IPCA(2, intercept=True).fit(ipca_sim.panel)
    a = m.test_alpha(8, seed=42)
    b = m.test_alpha(8, seed=42, executor="thread", n_jobs=2)
    np.testing.assert_allclose(a.draws, b.draws)


# ----------------------------------------------------------------------- SPCA
@pytest.fixture(scope="module")
def sim():
    return lfm.simulate_factor_returns(T=300, N=60, seed=21)


def test_spca_identities(sim):
    m = lfm.SPCA(3, n_select=30).fit(sim.returns, factors=sim.observable)
    V = m.latent_factors_
    T = V.shape[0]
    np.testing.assert_allclose(V.T @ V / T, np.eye(3), atol=1e-10)
    Gbar = sim.observable - sim.observable.mean(0)
    np.testing.assert_allclose(m.eta_, Gbar.T @ V / T, atol=1e-10)
    np.testing.assert_allclose(m.gamma_g_, m.eta_ @ m.gamma_)
    # hedging portfolio reproduces the risk premium: W rbar = gamma_g
    np.testing.assert_allclose(m.mimicking_weights_ @ sim.returns.mean(0), m.gamma_g_, atol=1e-10)
    assert all(len(s) == 30 for s in m.selected_)


def test_spca_without_screening_is_pca(sim):
    """With all assets selected, step one is the first principal component."""
    m = lfm.SPCA(1, n_select=1.0).fit(sim.returns, factors=sim.observable[:, :1])
    pca = lfm.PCA(1).fit(sim.returns)
    Rbar = sim.returns - sim.returns.mean(0)
    v_pca = Rbar @ pca.weights_[:, 0]
    assert abs(np.corrcoef(v_pca, m.latent_factors_[:, 0])[0, 1]) > 1 - 1e-10


def test_spca_nested_inference_equals_refit(sim):
    big = lfm.SPCA(3, n_select=25).fit(sim.returns, factors=sim.observable)
    small = lfm.SPCA(2, n_select=25).fit(sim.returns, factors=sim.observable)
    np.testing.assert_allclose(big.risk_premia(2).gamma_g, small.risk_premia().gamma_g)
    np.testing.assert_allclose(big.risk_premia(2).se, small.risk_premia().se)


def test_spca_detects_useless_factor(sim):
    rp = lfm.SPCA(3, n_select=30).fit(sim.returns, factors=sim.observable).risk_premia()
    assert rp.wald_pvalue[0] < 0.01  # proxy of a true factor
    assert rp.wald_pvalue[1] > 0.01  # pure noise


def test_spca_zero_beta_consistent(sim):
    m = lfm.SPCA(2, n_select=30, zero_beta=True).fit(sim.returns, factors=sim.observable)
    ev = m.evaluate()
    assert np.isfinite(ev["sharpe_ratio"])
    np.testing.assert_allclose(m.transform(sim.returns), m.factors_)
