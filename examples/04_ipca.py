"""IPCA (Kelly, Pruitt and Su, 2019): characteristics-instrumented loadings.

Run: python examples/04_ipca.py
"""

import numpy as np

import latent_factor_models as lfm


def main() -> None:
    sim = lfm.simulate_ipca_panel(T=200, N=300, L=6, K=2, missing=0.1, irrelevant=1, seed=0)
    panel = sim.panel  # unbalanced (10% missing); instrument 0 is irrelevant for betas

    model = lfm.IPCA(n_factors=2).fit(panel)
    print(f"converged={model.converged_} after {model.n_iter_} iterations")
    print("fit:", {k: round(v, 4) for k, v in model.evaluate().items()})
    err = np.linalg.norm(
        model.gamma_beta_ @ model.gamma_beta_.T - sim.gamma_beta @ sim.gamma_beta.T
    )
    print(f"distance to the true Gamma_beta subspace: {err:.4f}")

    # Unrestricted model and bootstrap test of H0: Gamma_alpha = 0 (parallel threads)
    unrestricted = lfm.IPCA(2, intercept=True).fit(panel)
    print(unrestricted.test_alpha(n_draws=200, seed=1, executor="thread", n_jobs=4))

    # Instrument significance (whole row of Gamma_beta), restricted model
    for ell in range(panel.L):
        print(f"instrument {ell}:", model.test_instrument(ell, n_draws=100, seed=ell))

    # Out of sample: expanding window, refit every 20 periods after 120 periods
    print(
        lfm.Engine([lfm.IPCA(2)]).out_of_sample(panel, window=120, refit_every=20, expanding=True)
    )


if __name__ == "__main__":
    main()
