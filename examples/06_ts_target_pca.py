"""TS-target PCA (Bryzgalova et al., 2026): factors that explain a time-series target.

The alpha target uses reference factors (here a proxy of the market factor),
so the extracted factors focus on what the reference model leaves unpriced.

Run: python examples/06_ts_target_pca.py
"""

import latent_factor_models as lfm


def main() -> None:
    sim = lfm.simulate_factor_returns(T=600, N=100, seed=0)
    data = sim.to_data("reference")  # factors = reference model for the alpha target

    for gamma in (0.0, 5.0, 15.0, 50.0):
        m = lfm.TSTargetPCA(3, gamma_ts=gamma, ts_target="alpha").fit(data)
        ev = m.evaluate()
        print(
            f"gamma_ts={gamma:5.1f}  TS-metric={m.target_metrics()['ts_metric']:6.2f}  "
            f"SR={ev['sharpe_ratio']:.3f}  RMS a={ev['rms_alpha']:.3f}"
        )

    # The alpha target is rebuilt in every training window, so there is no look-ahead
    print(
        lfm.Engine([lfm.PCA(3), lfm.TSTargetPCA(3, gamma_ts=15)]).out_of_sample(
            data, window=240, refit_every=12
        )
    )


if __name__ == "__main__":
    main()
