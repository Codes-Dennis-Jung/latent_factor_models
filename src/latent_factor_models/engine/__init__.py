# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dennis Jung
"""Execution engine: parallel backends and the experiment runner."""

from .parallel import Executor, spawn_seeds
from .runner import Engine, EngineResult, ModelSpec

__all__ = ["Engine", "EngineResult", "Executor", "ModelSpec", "spawn_seeds"]
