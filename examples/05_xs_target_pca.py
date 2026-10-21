"""XS-target PCA (Bryzgalova et al., 2026): loadings targeted to a cross-sectional space.

Run: python examples/05_xs_target_pca.py
"""

import latent_factor_models as lfm


def main() -> None:
    sim = lfm.simulate_factor_returns(T=600, N=100, seed=0)
    R = sim.returns
    N, L = R.shape[1], sim.xs_target.shape[1]

    print(f"{'gamma_xs':>9} {'XS-metric(SDF)':>15} {'SR':>7} {'RMS a':>7}")
    for gamma in (0.0, 0.25 * L / N, L / N, 4 * L / N):
        m = lfm.XSTargetPCA(3, gamma_xs=gamma).fit(R, xs_target=sim.xs_target)
        ev, tm = m.evaluate(), m.target_metrics()
        print(
            f"{gamma:9.4f} {tm['xs_metric_sdf']:15.2f} {ev['sharpe_ratio']:7.3f} "
            f"{ev['rms_alpha']:7.3f}"
        )

    # Macro target Lambda = R'Y(Y'Y)^-1, rebuilt inside each estimation window
    m = lfm.XSTargetPCA(3, gamma_xs=L / N, xs_target="macro").fit(R, macro=sim.macro)
    print("macro-target XS-metric per factor:", m.target_metrics()["xs_metric"].round(1))


if __name__ == "__main__":
    main()
