"""PCA: statistical factors from the covariance matrix of returns.

Run: python examples/01_pca.py
"""

import numpy as np

import latent_factor_models as lfm
from latent_factor_models.selection import covariance_eigenvalues, onatski


def main() -> None:
    sim = lfm.simulate_factor_returns(T=600, N=100, seed=0)
    R = sim.returns

    # How many factors? Onatski (2010) on the covariance spectrum.
    k = onatski(covariance_eigenvalues(R), r_max=10)
    print(f"Onatski estimate of the number of factors: {k}")

    model = lfm.PCA(n_factors=3).fit(R)
    print("absorption ratios:", np.round(model.absorption_ratio(), 3))
    print("in-sample metrics:", {k: round(v, 4) for k, v in model.evaluate().items()})

    # Out of sample: weights fixed on the first 400 months, applied to the rest
    oos = lfm.PCA(3).fit(R[:400]).evaluate(R[400:])
    print("out-of-sample metrics:", {k: round(v, 4) for k, v in oos.items()})

    # PCA struggles with the weak-but-priced third factor
    corr = np.corrcoef(model.factors_.T, sim.factors.T)[:3, 3:]
    print("|corr(estimated, true factors)|:\n", np.round(np.abs(corr), 2))


if __name__ == "__main__":
    main()
