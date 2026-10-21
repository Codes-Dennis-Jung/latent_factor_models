# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Dennis Jung
"""Command-line interface ``lfm``.

Sub-commands
------------
``lfm models``
    List the registered estimators and their parameters.
``lfm simulate``
    Write a synthetic data set (CSV files) with known ground truth.
``lfm fit MODEL``
    Fit one model to CSV data and print/save metrics and estimates.
``lfm run CONFIG``
    Run the experiment engine from a JSON (or TOML) configuration.

Exit status: ``0`` success, ``1`` data or runtime error, ``2`` usage error.

CSV conventions: comma separated, optional header row, optional first
column of non-numeric labels (dates) which is used as index.
"""

from __future__ import annotations

import argparse
import ast
import csv
import inspect
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .core.base import PortfolioFactorModel
from .core.data import CharacteristicPanel, FactorData
from .core.exceptions import LatentFactorError
from .core.registry import _REGISTRY, available_models, create_model, get_model_class
from .engine.runner import Engine, ModelSpec
from .simulation import simulate_factor_returns, simulate_ipca_panel


# ------------------------------------------------------------------- CSV I/O
def _is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return text.strip().lower() in ("", "nan", "na")
    return True


def read_matrix(path: str | Path) -> tuple[np.ndarray, list[str] | None, list[str] | None]:
    """Read a numeric CSV matrix with optional header row and label column.

    Returns
    -------
    values, index labels (or None), column labels (or None)
    """
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = [r for r in csv.reader(fh) if r]
    if not rows:
        raise LatentFactorError(f"{path}: empty file")
    header = None
    if not all(_is_number(c) for c in rows[0]):
        header, rows = rows[0], rows[1:]
    has_index = any(not _is_number(r[0]) for r in rows)
    index = [r[0] for r in rows] if has_index else None
    body = [r[1:] if has_index else r for r in rows]
    try:
        values = np.array(
            [[float(c) if c.strip() not in ("", "NA", "na") else np.nan for c in r] for r in body],
            dtype=float,
        )
    except ValueError as exc:
        raise LatentFactorError(f"{path}: non-numeric value ({exc})") from None
    if header is not None and has_index:
        header = header[1:]
    return values, index, header


def write_matrix(
    path: Path,
    values: np.ndarray,
    columns: Sequence[str] | None = None,
    index: Sequence[Any] | None = None,
) -> None:
    """Write a 2-D array as CSV with optional header and index column."""
    values = np.atleast_2d(np.asarray(values))
    if values.shape[0] == 1 and index is not None and len(index) != 1:
        values = values.T
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if columns is not None:
            w.writerow((["index"] if index is not None else []) + list(columns))
        for i, row in enumerate(values):
            lead = [index[i]] if index is not None else []
            w.writerow(lead + [repr(float(x)) for x in row])


def _parse_value(text: str) -> Any:
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def _parse_params(items: Sequence[str] | None) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"--param expects key=value, got {item!r}")
        key, value = item.split("=", 1)
        params[key.strip()] = _parse_value(value.strip())
    return params


def _load_factor_data(
    returns: str,
    factors: str | None = None,
    xs_target: str | None = None,
    ts_target: str | None = None,
    macro: str | None = None,
) -> FactorData:
    R, index, assets = read_matrix(returns)
    extra: dict[str, Any] = {
        name: read_matrix(path)[0]
        for name, path in (
            ("factors", factors),
            ("xs_target", xs_target),
            ("ts_target", ts_target),
            ("macro", macro),
        )
        if path
    }
    return FactorData(R, index=index, assets=assets, **extra)


def _load_panel(spec: dict[str, Any], base: Path) -> CharacteristicPanel:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise LatentFactorError("characteristic panels from CSV require pandas") from exc
    frame = pd.read_csv(base / spec["panel"])
    return CharacteristicPanel.from_long(
        frame, spec["entity"], spec["time"], spec["target"], spec["characteristics"]
    )


# ---------------------------------------------------------------- commands
def _cmd_models(args: argparse.Namespace) -> int:
    for name in available_models():
        cls = get_model_class(name)
        aliases = sorted(k for k, v in _REGISTRY.items() if v is cls and k != name)
        doc = (inspect.getdoc(cls) or "").splitlines()[0]
        print(f"{name:<18} {cls.__name__:<14} {doc}")
        if args.verbose:
            sig = inspect.signature(cls.__init__)
            params = ", ".join(str(p) for p in list(sig.parameters.values())[1:])
            print(f"{'':<18} params: {params}")
            if aliases:
                print(f"{'':<18} aliases: {', '.join(aliases)}")
    return 0


def _cmd_simulate(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.kind == "factor":
        sim = simulate_factor_returns(T=args.T, N=args.N, seed=args.seed)
        assets = [f"a{i + 1}" for i in range(args.N)]
        write_matrix(out / "returns.csv", sim.returns, assets)
        write_matrix(out / "observable.csv", sim.observable, ["proxy", "useless"])
        write_matrix(out / "reference.csv", sim.reference, ["reference"])
        write_matrix(out / "macro.csv", sim.macro, ["macro1", "macro2"])
        write_matrix(out / "xs_target.csv", sim.xs_target, ["target"])
        write_matrix(
            out / "true_factors.csv",
            sim.factors,
            [f"f{k + 1}" for k in range(sim.factors.shape[1])],
        )
    else:
        ip = simulate_ipca_panel(T=args.T, N=args.N, seed=args.seed)
        p = ip.panel
        t_idx, i_idx = np.nonzero(p.mask)
        with open(out / "panel.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["entity", "time", "ret"] + [f"z{j + 1}" for j in range(p.L)])
            for t, i in zip(t_idx, i_idx, strict=True):
                chars = (repr(float(z)) for z in p.characteristics[t, i])
                w.writerow([i, t, repr(float(p.returns[t, i])), *chars])
    print(f"wrote synthetic {args.kind} data to {out.resolve()}")
    return 0


def _cmd_fit(args: argparse.Namespace) -> int:
    params = _parse_params(args.param)
    model = create_model(args.model, **params)
    if not isinstance(model, PortfolioFactorModel):
        raise LatentFactorError("`lfm fit` supports return-based models; use `lfm run` for IPCA")
    data = _load_factor_data(args.returns, args.factors, args.xs_target, args.ts_target, args.macro)
    model.fit(data)
    result: dict[str, Any] = {
        "model": repr(model),
        "T": data.T,
        "N": data.N,
        "metrics": model.evaluate(),
    }
    if hasattr(model, "target_metrics"):
        result["target_metrics"] = {
            k: np.asarray(v).tolist() for k, v in model.target_metrics().items()
        }
    if hasattr(model, "risk_premia"):
        rp = model.risk_premia()
        result["risk_premia"] = {
            k: np.asarray(getattr(rp, k)).tolist()
            for k in ("gamma_g", "se", "tstat", "pvalue", "wald_pvalue")
        }
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(text, encoding="utf-8")
        K = model.n_factors
        cols = [f"F{k + 1}" for k in range(K)]
        write_matrix(out / "factors.csv", model.factors_, cols, data.index)
        write_matrix(out / "weights.csv", model.weights_, cols, data.assets)
        write_matrix(out / "loadings.csv", model.loadings_, cols, data.assets)
        print(f"saved estimates to {out.resolve()}", file=sys.stderr)
    return 0


def _read_config(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".toml":
        try:
            import tomllib
        except ModuleNotFoundError as exc:  # pragma: no cover - Python 3.10
            raise LatentFactorError("TOML configs need Python >= 3.11; use JSON") from exc
        return dict(tomllib.loads(path.read_text(encoding="utf-8")))
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _cmd_run(args: argparse.Namespace) -> int:
    path = Path(args.config)
    cfg = _read_config(path)
    base = path.parent
    data_cfg = cfg.get("data", {})
    if "panel" in data_cfg:
        data: Any = _load_panel(data_cfg, base)
    else:

        def rel(key: str) -> str | None:
            return str(base / data_cfg[key]) if key in data_cfg else None

        returns = rel("returns")
        if returns is None:
            raise LatentFactorError("config needs data.returns or data.panel")
        data = _load_factor_data(
            returns, rel("factors"), rel("xs_target"), rel("ts_target"), rel("macro")
        )
    models = [
        ModelSpec(m["name"], dict(m.get("params", {})), m.get("label"))
        for m in cfg.get("models", [])
    ]
    if not models:
        raise LatentFactorError("config lists no models")
    ex = cfg.get("execution", {})
    engine = Engine(
        models,
        executor=args.backend or ex.get("backend", "serial"),
        n_jobs=args.n_jobs if args.n_jobs is not None else ex.get("n_jobs"),
    )
    ev = cfg.get("evaluation", {})
    result = engine.run(
        data,
        window=ev.get("window"),
        refit_every=ev.get("refit_every", 1),
        expanding=ev.get("expanding", False),
        oos_loadings=ev.get("oos_loadings", "ols"),
    )
    print(result)
    output = args.out or cfg.get("output")
    if output:
        out = base / output if args.out is None else Path(output)
        keys = list(dict.fromkeys(k for r in result.records for k in r))
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(result.records)
        print(f"saved results to {out.resolve()}", file=sys.stderr)
    return 0


# ------------------------------------------------------------------ parser
def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser (exposed for documentation and tests)."""
    parser = argparse.ArgumentParser(
        prog="lfm",
        description="Latent factor models for asset pricing.",
        epilog="Run `lfm COMMAND --help` for command-specific options.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    p = sub.add_parser("models", help="list available models")
    p.add_argument("-v", "--verbose", action="store_true", help="show parameters and aliases")
    p.set_defaults(func=_cmd_models)

    p = sub.add_parser("simulate", help="write synthetic data (CSV)")
    p.add_argument("--kind", choices=("factor", "ipca"), default="factor")
    p.add_argument("--T", type=int, default=600, help="number of periods")
    p.add_argument("--N", type=int, default=100, help="number of assets")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("-o", "--out", default="synthetic_data", help="output directory")
    p.set_defaults(func=_cmd_simulate)

    p = sub.add_parser("fit", help="fit one model to CSV data")
    p.add_argument("model", help="model name, see `lfm models`")
    p.add_argument("--returns", required=True, help="T x N returns CSV")
    p.add_argument("--factors", help="T x d observable/reference factors CSV")
    p.add_argument("--xs-target", dest="xs_target", help="N x L cross-sectional target CSV")
    p.add_argument("--ts-target", dest="ts_target", help="T x J time-series target CSV")
    p.add_argument("--macro", help="T x L macro innovations CSV")
    p.add_argument(
        "-p",
        "--param",
        action="append",
        metavar="KEY=VALUE",
        help="model parameter, repeatable (e.g. -p n_factors=3 -p gamma=10)",
    )
    p.add_argument("-o", "--out", help="directory for estimates (CSV) and summary.json")
    p.set_defaults(func=_cmd_fit)

    p = sub.add_parser("run", help="run the experiment engine from a JSON/TOML config")
    p.add_argument("config", help="configuration file")
    p.add_argument("--backend", choices=("serial", "thread", "process"), help="override backend")
    p.add_argument("--n-jobs", dest="n_jobs", type=int, help="override number of workers")
    p.add_argument("-o", "--out", help="results CSV (overrides config `output`)")
    p.set_defaults(func=_cmd_run)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of the ``lfm`` command."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (LatentFactorError, ValueError, KeyError, FileNotFoundError) as exc:
        print(f"lfm: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
