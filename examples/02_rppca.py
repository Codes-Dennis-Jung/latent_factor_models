"""RP-PCA (Lettau and Pelger, 2020): PCA with a penalty on pricing errors.

Run: python examples/02_rppca.py
"""

import latent_factor_models as lfm


def main() -> None:
    sim = lfm.simulate_factor_returns(T=600, N=100, seed=0)
    R = sim.returns

    print(f"{'gamma':>6} {'SR':>8} {'RMS a':>8} {'sigma_e':>8}")
    for gamma in (-1, 0, 10, 20, 50):  # gamma = -1 is covariance PCA
        ev = lfm.RPPCA(3, gamma=gamma).fit(R).evaluate()
        print(
            f"{gamma:>6} {ev['sharpe_ratio']:8.3f} {ev['rms_alpha']:8.3f} "
            f"{ev['unexplained_variation']:8.3f}"
        )

    # Choose gamma by contiguous 5-fold cross-validation of the Sharpe ratio
    gs = lfm.GridSearchCV(
        lfm.RPPCA(3), {"gamma": [-1, 0, 5, 10, 20, 50]}, cv=5, scoring="sharpe"
    ).fit(R)
    print("cross-validated gamma:", gs.best_params_, f"(mean CV SR {gs.best_score_:.3f})")

    # Rolling out-of-sample comparison with PCA (240-period window, refit every 12)
    best = lfm.RPPCA(3, gamma=gs.best_params_["gamma"])
    print(lfm.Engine([lfm.PCA(3), best]).out_of_sample(R, window=240, refit_every=12))


if __name__ == "__main__":
    main()
