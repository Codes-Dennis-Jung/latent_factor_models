# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dennis Jung
"""Allow ``python -m latent_factor_models``."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
