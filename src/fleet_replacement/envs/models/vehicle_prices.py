from __future__ import annotations

import numpy as np

from fleet_replacement.config import BETPriceConfig


def DT_price_step(current_price: float) -> float:
    """Return DT purchase price for a process with zero drift and volatility."""
    return float(current_price)


def BET_price_mean_reversion_step(
    current_price: float,
    config: BETPriceConfig,
    rng: np.random.Generator,
) -> float:
    """Advance BET purchase price by one mean-reverting step with Gaussian noise."""
    reversion = config.mean_reversion_strength * (config.long_term_mean - current_price)
    shock = rng.normal(0.0, config.purchase_price_volatility)
    return float(current_price + reversion + shock)
