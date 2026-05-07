from __future__ import annotations

import numpy as np

from fleet_replacement.config import BETProductivityConfig


def bet_productivity_logistic(
    year: int,
    config: BETProductivityConfig,
) -> float:
    """Compute BET productivity from a logistic growth curve at a given year."""
    base = 1.0 / (1.0 + np.exp(-config.k * (year - config.t0)))
    return float(config.start + (config.max - config.start) * base)
