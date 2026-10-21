# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Dennis Jung
"""Synthetic data generators."""

from .synthetic import (
    SimulatedFactorData,
    SimulatedIPCAData,
    simulate_factor_returns,
    simulate_ipca_panel,
)

__all__ = [
    "SimulatedFactorData",
    "SimulatedIPCAData",
    "simulate_factor_returns",
    "simulate_ipca_panel",
]
