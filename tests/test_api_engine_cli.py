"""Public API, data handling, parallel engine, cross-validation and command-line interface."""

import json

import numpy as np
import pytest

import latent_factor_models as lfm
from latent_factor_models.cli import main, read_matrix
from latent_factor_models.engine.parallel import Executor, resolve_n_jobs, spawn_seeds


@pytest.fixture(scope="module")
def sim():
    return lfm.simulate_factor_returns(T=260, N=30, seed=2)


ALL_RETURN_MODELS = [
    (lfm.PCA, {}),
    (lfm.RPPCA, {"gamma": 5}),
    (lfm.XSTargetPCA, {"gamma_xs": 0.5}),
    (lfm.TSTargetPCA, {}),
    (lfm.XSTSTargetPCA, {}),
    (lfm.SPCA, {"n_select": 15}),
]


@pytest.mark.parametrize(("cls", "params"), ALL_RETURN_MODELS)
def test_common_interface(sim, cls, params):
    m = cls(2, **params).fit(sim.to_data("reference"))
    assert m.factors_.shape == (260, 2)
    assert m.weights_.shape == m.loadings_.shape == (30, 2)
    np.testing.assert_allclose(m.residuals().mean(0), m.intercepts_, atol=1e-10)
    oos = m.evaluate(sim.returns[:50])
    assert set(oos) == {"sharpe_ratio", "rms_alpha", "unexplained_variation", "total_r2"}
    clone = m.clone()
    assert clone.get_params() == m.get_params() and not hasattr(clone, "is_fitted_")


def test_not_fitted_and_validation(sim):
    with pytest.raises(lfm.NotFittedError):
        lfm.PCA(2).transform(sim.returns)
    with pytest.raises(ValueError):
        lfm.PCA(0).fit(sim.returns)
    with pytest.raises(ValueError):
        lfm.RPPCA(1, gamma=-2).fit(sim.returns)
    with pytest.raises(lfm.DataError):
        lfm.XSTargetPCA(1).fit(sim.returns)  # target missing
    with pytest.raises(lfm.DataError):
        lfm.SPCA(1).fit(sim.returns)  # factors missing
    bad = sim.returns.copy()
    bad[0, 0] = np.nan
    with pytest.raises(lfm.DataError):
        lfm.PCA(1).fit(bad)
    with pytest.raises(lfm.DataError):
        lfm.PCA(1).fit(sim.returns, factors=np.ones((10, 1)))


def test_registry(sim):
    assert lfm.create_model("rp-pca", n_factors=2).name == "rppca"
    assert "xsts_target_pca" in lfm.available_models()
    with pytest.raises(KeyError):
        lfm.create_model("nope")


def test_pandas_roundtrip(sim):
    pd = pytest.importorskip("pandas")
    idx = pd.date_range("2000-01-31", periods=260, freq="ME")
    df = pd.DataFrame(sim.returns, index=idx, columns=[f"a{i}" for i in range(30)])
    m = lfm.PCA(2).fit(df)
    assert m.data_.index[0] == idx[0] and m.data_.assets[0] == "a0"
    np.testing.assert_allclose(m.transform(df.iloc[:5]), sim.returns[:5] @ m.weights_)
    long = pd.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "t": [1, 2, 1, 2],
            "r": [0.1, 0.2, 0.3, 0.4],
            "z1": [1.0, 2, 3, 4],
            "z2": [1.0, 1, 1, 1],
        }
    )
    p = lfm.CharacteristicPanel.from_long(long, "id", "t", "r", ["z1", "z2"])
    assert p.returns.shape == (2, 2) and p.L == 2


def test_target_windows_have_no_lookahead(sim):
    """The alpha target is rebuilt from the training window only."""
    data = sim.to_data("reference")
    m = lfm.TSTargetPCA(2).fit(data.take(np.arange(100)))
    from latent_factor_models.models.targets import alpha_target

    np.testing.assert_allclose(m.spec_.ts_target, alpha_target(sim.reference[:100]))


def test_executor_backends_agree():
    items = list(range(20))
    serial = Executor("serial").map(np.square, items)
    assert Executor("thread", 3).map(np.square, items) == serial
    assert resolve_n_jobs(-1) >= 1
    s1 = [np.random.default_rng(s).random() for s in spawn_seeds(1, 3)]
    s2 = [np.random.default_rng(s).random() for s in spawn_seeds(1, 3)]
    assert s1 == s2


@pytest.mark.slow
def test_process_backend_grid_search(sim):
    gs_serial = lfm.GridSearchCV(lfm.RPPCA(2), {"gamma": [-1, 10]}, cv=3).fit(sim.returns)
    gs_proc = lfm.GridSearchCV(
        lfm.RPPCA(2), {"gamma": [-1, 10]}, cv=3, executor="process", n_jobs=2
    ).fit(sim.returns)
    np.testing.assert_allclose(gs_serial.result_.scores, gs_proc.result_.scores)


def test_cv_splitters():
    folds = list(lfm.KFold(4).split(20))
    assert len(folds) == 4 and all(len(te) == 5 for _, te in folds)
    assert np.array_equal(folds[0][1], np.arange(5))
    rnd = list(lfm.KFold(3, shuffle=True, n_repeats=2, seed=0).split(30))
    assert len(rnd) == 6
    ts = list(lfm.TimeSeriesSplit(3, test_size=10, gap=2).split(100))
    for tr, te in ts:
        assert tr.max() + 2 < te.min()


def test_grid_search_and_spca_cv(sim):
    gs = lfm.GridSearchCV(
        lfm.SPCA(2),
        {"n_select": [5, 15, 30]},
        cv=lfm.KFold(3, shuffle=True, seed=0),
        scoring="hedging_r2",
    ).fit(sim.returns, factors=sim.observable)
    assert gs.best_params_["n_select"] in (5, 15, 30)
    assert gs.best_model_.is_fitted_
    scores = lfm.cross_validate(lfm.RPPCA(2), sim.returns, cv=4, scoring="neg_rms_alpha")
    assert scores.shape == (4,) and np.all(scores <= 0)


def test_engine_in_and_out_of_sample(sim):
    eng = lfm.Engine(
        [
            lfm.ModelSpec("pca", {"n_factors": 2}),
            {"name": "ts_target_pca", "params": {"n_factors": 2}},
            lfm.RPPCA(2),
        ]
    )
    res = eng.run(sim.to_data("reference"), window=200, refit_every=15)
    assert len(res.records) == 6
    oos = [r for r in res.records if r["sample"] == "out-of-sample"]
    assert all(r["n_windows"] == 4 for r in oos)
    assert "pca" in str(res)


def test_engine_matches_manual_rolling(sim):
    """One-step-ahead engine output equals a hand-written rolling loop."""
    R = sim.returns
    res = lfm.Engine([lfm.PCA(2)]).out_of_sample(R, window=240)
    resid, sdf = [], []
    for t in range(240, 260):
        m = lfm.PCA(2).fit(R[t - 240 : t])
        f = R[t] @ m.weights_
        resid.append(R[t] - m.loadings_ @ f)
        sdf.append(f @ lfm.evaluation.tangency_weights(m.factors_))
    rec = res.records[0]
    assert np.isclose(rec["rms_alpha"], np.sqrt(np.mean(np.mean(resid, 0) ** 2)))
    assert np.isclose(rec["sharpe_ratio"], np.mean(sdf) / np.std(sdf, ddof=1))


# ------------------------------------------------------------------------ CLI
def test_cli_models(capsys):
    assert main(["models", "-v"]) == 0
    assert "rppca" in capsys.readouterr().out


def test_cli_simulate_fit_run(tmp_path, capsys):
    d = tmp_path / "data"
    assert main(["simulate", "--T", "150", "--N", "20", "-o", str(d)]) == 0
    R, idx, cols = read_matrix(d / "returns.csv")
    assert R.shape == (150, 20) and cols[0] == "a1" and idx is None
    out = tmp_path / "fit"
    assert (
        main(
            [
                "fit",
                "rppca",
                "--returns",
                str(d / "returns.csv"),
                "-p",
                "n_factors=2",
                "-p",
                "gamma=10",
                "-o",
                str(out),
            ]
        )
        == 0
    )
    summary = json.loads((out / "summary.json").read_text())
    assert summary["N"] == 20 and "sharpe_ratio" in summary["metrics"]
    assert (
        main(
            [
                "fit",
                "spca",
                "--returns",
                str(d / "returns.csv"),
                "--factors",
                str(d / "observable.csv"),
                "-p",
                "n_factors=2",
            ]
        )
        == 0
    )
    cfg = {
        "data": {"returns": "data/returns.csv", "factors": "data/reference.csv"},
        "models": [
            {"name": "pca", "params": {"n_factors": 2}},
            {"name": "ts_target_pca", "params": {"n_factors": 2}},
        ],
        "evaluation": {"window": 120, "refit_every": 10},
        "output": "results.csv",
    }
    (tmp_path / "cfg.json").write_text(json.dumps(cfg))
    assert main(["run", str(tmp_path / "cfg.json")]) == 0
    assert (tmp_path / "results.csv").read_text().count("\n") == 5


def test_cli_errors(tmp_path, capsys):
    assert main(["fit", "pca", "--returns", str(tmp_path / "missing.csv")]) == 1
    assert "error" in capsys.readouterr().err
    with pytest.raises(SystemExit) as exc:
        main(["fit"])
    assert exc.value.code == 2


def test_cli_ipca_simulation(tmp_path):
    assert main(["simulate", "--kind", "ipca", "--T", "30", "--N", "40", "-o", str(tmp_path)]) == 0
    stacked = np.loadtxt(tmp_path / "panel.csv", delimiter=",", skiprows=1)
    m = lfm.IPCA(1).fit(stacked)
    assert m.factors_.shape == (30, 1)
