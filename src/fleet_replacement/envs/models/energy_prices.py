from __future__ import annotations

import numpy as np

from fleet_replacement.config import StochasticPriceConfig


def _geometric_brownian_step(
    current_price: float,
    growth_rate: float,
    volatility: float,
    rng: np.random.Generator,
) -> float:
    """Advance a positive price process by one geometric Brownian motion step."""
    noise = rng.normal(0.0, 1.0)
    drift = growth_rate - 0.5 * volatility**2
    diffusion = volatility * noise
    return float(current_price * np.exp(drift + diffusion))


def diesel_price_step(
    current_price: float,
    config: StochasticPriceConfig,
    rng: np.random.Generator,
) -> float:
    """Update diesel price using the configured geometric Brownian process."""
    return _geometric_brownian_step(
        current_price=current_price,
        growth_rate=config.growth_rate,
        volatility=config.volatility,
        rng=rng,
    )


def electricity_price_step(
    current_price: float,
    config: StochasticPriceConfig,
    rng: np.random.Generator,
) -> float:
    """Update electricity price using the configured geometric Brownian process."""
    return _geometric_brownian_step(
        current_price=current_price,
        growth_rate=config.growth_rate,
        volatility=config.volatility,
        rng=rng,
    )
