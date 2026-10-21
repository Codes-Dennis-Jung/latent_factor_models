# Command-line interface `lfm`

Installing the package provides the `lfm` command (also available as
`python -m latent_factor_models`). It follows standard CLI conventions:

* `--help` on every command and `--version`;
* results go to stdout, diagnostics and errors to stderr;
* exit status `0` on success, `1` on data/runtime errors (with an
  `lfm: error: ...` message), `2` on usage errors.

```text
lfm models [-v]                         list estimators (with parameters and aliases)
lfm simulate [--kind factor|ipca] [--T 600] [--N 100] [--seed 0] [-o DIR]
lfm fit MODEL --returns R.csv [--factors G.csv] [--xs-target L.csv]
              [--ts-target G.csv] [--macro Y.csv] [-p key=value ...] [-o DIR]
lfm run CONFIG.json|CONFIG.toml [--backend serial|thread|process] [--n-jobs N] [-o OUT.csv]
```

## CSV conventions

Files are comma separated. A header row is optional (detected automatically).
A non-numeric first column is treated as the index (e.g. dates). Empty, `NA`
or `nan` cells are missing values. Return-based models need complete data;
use `FactorData.dropna` in Python to clean panels.

## Examples

```bash
# synthetic data with known ground truth
lfm simulate --T 600 --N 100 -o synthetic_data

# one model, estimates written to fit_output/{factors,weights,loadings}.csv + summary.json
lfm fit rppca --returns synthetic_data/returns.csv -p n_factors=3 -p gamma=10 -o fit_output

# TS-target PCA with the alpha target built from reference factors
lfm fit ts_target_pca --returns synthetic_data/returns.csv \
    --factors synthetic_data/reference.csv -p n_factors=3 -p gamma_ts=15

# SPCA risk premia (JSON includes gamma_g, se, t, p and Wald p-values)
lfm fit spca --returns synthetic_data/returns.csv --factors synthetic_data/observable.csv \
    -p n_factors=3 -p n_select=50

# a full in-sample and rolling out-of-sample comparison, in parallel
lfm run examples/config_example.json --backend process --n-jobs -1
```

## Configuration file

```json
{
  "data": {"returns": "returns.csv", "factors": "reference.csv", "xs_target": "xs.csv"},
  "models": [
    {"name": "pca", "params": {"n_factors": 3}},
    {"name": "rppca", "params": {"n_factors": 3, "gamma": 10}, "label": "RP-PCA"}
  ],
  "evaluation": {"window": 240, "refit_every": 12, "expanding": false,
                 "oos_loadings": "ols"},
  "execution": {"backend": "process", "n_jobs": -1},
  "output": "results.csv"
}
```

Paths are relative to the configuration file. For IPCA, replace `data` with
`{"panel": "panel.csv", "entity": "id", "time": "date", "target": "ret",
"characteristics": ["z1", "z2"]}` (requires pandas). TOML works the same way
on Python ≥ 3.11.
