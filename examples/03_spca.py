"""Supervised PCA (Giglio, Xiu and Zhang): risk premia of (weak) observable factors.

Run: python examples/03_spca.py
"""

import numpy as np

import latent_factor_models as lfm


def main() -> None:
    sim = lfm.simulate_factor_returns(T=600, N=100, seed=0)
    R, G = sim.returns, sim.observable  # G = [proxy of factor 1, useless factor]

    # Tune the screening size by repeated random 3-fold CV of the hedging R2
    cv = lfm.KFold(3, shuffle=True, n_repeats=5, seed=0)
    gs = lfm.GridSearchCV(
        lfm.SPCA(3), {"n_select": [10, 25, 50, 100]}, cv=cv, scoring="hedging_r2"
    ).fit(R, factors=G)
    model = gs.best_model_
    print("selected n_select:", gs.best_params_["n_select"])

    rp = model.risk_premia()
    print(rp.summary(["proxy", "useless"]))
    print(
        "true premium of factor 1:",
        round(float(sim.risk_premia[0]), 4),
        "| sample mean of proxy:",
        round(float(G[:, 0].mean()), 4),
    )
    print(
        "alpha of tradable proxy:",
        np.round(rp.extra["alpha"], 4),
        "s.e.",
        np.round(rp.extra["alpha_se"], 4),
    )
    print("in-sample hedging R2:", np.round(model.hedging_r2_, 3))

    # Giglio-Xiu three-pass estimator with a PCA first pass, for comparison
    tp = lfm.ThreePassRiskPremia(lfm.PCA(3)).fit(R, G)
    print("\nthree-pass (PCA):\n" + tp.summary(["proxy", "useless"]))


if __name__ == "__main__":
    main()
