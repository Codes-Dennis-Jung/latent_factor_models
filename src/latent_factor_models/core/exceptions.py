# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Dennis Jung
"""Exception hierarchy of the package."""

from __future__ import annotations


class LatentFactorError(Exception):
    """Base class for all package-specific errors."""


class NotFittedError(LatentFactorError, AttributeError):
    """Raised when a model is used before :meth:`fit` was called."""


class DataError(LatentFactorError, ValueError):
    """Raised for malformed, inconsistent or missing input data."""


class ConvergenceWarning(UserWarning):
    """Issued when an iterative estimator stops before reaching its tolerance."""
