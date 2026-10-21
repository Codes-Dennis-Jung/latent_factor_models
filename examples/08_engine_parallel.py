"""Compare all return-based models in and out of sample with the parallel engine.

Run: python examples/08_engine_parallel.py
"""

import time

import latent_factor_models as lfm


def main() -> None:
    sim = lfm.simulate_factor_returns(T=600, N=100, seed=0)
    data = sim.to_data("reference")
    L, N = sim.xs_target.shape[1], sim.returns.shape[1]
    models = [
        lfm.ModelSpec("pca", {"n_factors": 3}),
        lfm.ModelSpec("rppca", {"n_factors": 3, "gamma": 10}),
        lfm.ModelSpec("ts_target_pca", {"n_factors": 3, "gamma_ts": 15}),
        lfm.ModelSpec("xs_target_pca", {"n_factors": 3, "gamma_xs": L / N}),
        lfm.ModelSpec("xsts_target_pca", {"n_factors": 3, "gamma_xs": L / N, "gamma_ts": 15}),
    ]
    res = None
    for backend in ("serial", "process"):
        engine = lfm.Engine(models, executor=backend, n_jobs=-1)
        t0 = time.perf_counter()
        res = engine.run(data, window=240, refit_every=1)
        print(f"{backend:>8} backend: {time.perf_counter() - t0:.2f}s")
    print(res)


if __name__ == "__main__":  # required for the process backend on Windows/macOS
    main()
