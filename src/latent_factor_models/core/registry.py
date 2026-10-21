# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dennis Jung
"""Name-based model registry used by the engine and the command line."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from .base import LatentFactorModel

_REGISTRY: dict[str, type[LatentFactorModel]] = {}

C = TypeVar("C", bound=type[LatentFactorModel])


def register(*aliases: str) -> Callable[[C], C]:
    """Class decorator registering a model under its ``name`` and optional aliases."""

    def decorator(cls: C) -> C:
        for key in (cls.name, *aliases):
            key = key.lower()
            if key in _REGISTRY and _REGISTRY[key] is not cls:
                raise ValueError(f"model name {key!r} already registered")
            _REGISTRY[key] = cls
        return cls

    return decorator


def get_model_class(name: str) -> type[LatentFactorModel]:
    """Look up a registered model class by (case-insensitive) name or alias."""
    try:
        return _REGISTRY[name.lower()]
    except KeyError:
        raise KeyError(f"unknown model {name!r}; available: {available_models()}") from None


def create_model(name: str, **params: Any) -> LatentFactorModel:
    """Instantiate a registered model, e.g. ``create_model("rppca", n_factors=3, gamma=10)``."""
    return get_model_class(name)(**params)


def available_models() -> list[str]:
    """Canonical names of all registered models."""
    return sorted({cls.name for cls in _REGISTRY.values()})
