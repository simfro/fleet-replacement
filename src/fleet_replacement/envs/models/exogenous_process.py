from __future__ import annotations

import numpy as np

from fleet_replacement.config import (
    BETProductivityConfig,
    BETPriceConfig,
    StochasticPriceConfig,
)


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


def BET_productivity_logistic(
    year: int,
    config: BETProductivityConfig,
) -> float:
    """Compute BET productivity from a logistic growth curve at a given year."""
    base = 1.0 / (1.0 + np.exp(-config.k * (year - config.t0)))
    return float(config.start + (config.max - config.start) * base)
