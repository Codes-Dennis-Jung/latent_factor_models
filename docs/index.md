# latent-factor-models documentation

| Topic | Document |
|---|---|
| Design, layers, class hierarchy, extension guide | [architecture.md](architecture.md) |
| PCA | [models/pca.md](models/pca.md) |
| RP-PCA (Lettau-Pelger) | [models/rppca.md](models/rppca.md) |
| SPCA (Giglio-Xiu-Zhang) | [models/spca.md](models/spca.md) |
| IPCA (Kelly-Pruitt-Su) | [models/ipca.md](models/ipca.md) |
| XS-target PCA | [models/xs_target_pca.md](models/xs_target_pca.md) |
| TS-target PCA | [models/ts_target_pca.md](models/ts_target_pca.md) |
| XS-TS-target PCA | [models/xsts_target_pca.md](models/xsts_target_pca.md) |
| Risk-premium inference (three-pass, Newey-West, Wald) | [inference.md](inference.md) |
| Metrics, number of factors, cross-validation, engine | [evaluation.md](evaluation.md) |
| Command line | [cli.md](cli.md) |
| Legacy code, validation and errors fixed | [../archive/README.md](../archive/README.md) |

Every model document has the same structure: **technical description**
(what the model is for), **methodology** (the estimator as in the paper),
**implementation notes** (how the package computes it, how it was validated,
what was fixed relative to the legacy code), an **example** and the
**parameters**. Runnable versions of all examples are in `examples/`.
