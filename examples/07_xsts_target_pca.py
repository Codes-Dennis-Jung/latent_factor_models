"""XS-TS-target PCA (Bryzgalova et al., 2026): both targets at once, tuned by CV.

Run: python examples/07_xsts_target_pca.py
"""

import numpy as np

import latent_factor_models as lfm


def main() -> None:
    sim = lfm.simulate_factor_returns(T=600, N=100, seed=0)
    data = sim.to_data("reference")
    L, N = sim.xs_target.shape[1], sim.returns.shape[1]

    grid = {"gamma_xs": [0.0, L / N, 4 * L / N], "gamma_ts": [0.0, 5.0, 15.0]}
    gs = lfm.GridSearchCV(
        lfm.XSTSTargetPCA(3), grid, cv=5, scoring="sharpe", executor="process", n_jobs=-1
    ).fit(data)
    print("best parameters:", gs.best_params_, f"CV SR={gs.best_score_:.3f}")
    for row in sorted(gs.result_.to_records(), key=lambda r: -r["mean_score"])[:5]:
        print(row)

    tm = gs.best_model_.target_metrics()
    print("target metrics:", {k: np.round(v, 1) for k, v in tm.items()})


if __name__ == "__main__":  # required for the process backend on Windows/macOS
    main()
