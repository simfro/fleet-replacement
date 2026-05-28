from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SimulationPeriodConfig:
    base_year: int
    final_year: int


@dataclass(frozen=True)
class StochasticPriceConfig:
    initial_price: float
    growth_rate: float
    volatility: float


@dataclass(frozen=True)
class DTPriceConfig:
    initial_price: float


@dataclass(frozen=True)
class BETPriceConfig:
    initial_price: float
    long_term_mean: float
    mean_reversion_strength: float
    purchase_price_volatility: float


@dataclass(frozen=True)
class BETProductivityConfig:
    start: float
    max: float
    k_min: float
    k_max: float
    t0_min: int
    t0_max: int


@dataclass(frozen=True)
class OperationalConfig:
    fuel_consumption_l_per_km: float
    electricity_consumption_kwh_per_km: float
    driver_salary_annual: float
    income_per_km: float
    annual_mileage_km: float


@dataclass(frozen=True)
class EconomicConfig:
    interest_rate: float
    loan_fraction: float
    economic_lifetime: int
    discount_factor: float


@dataclass(frozen=True)
class VehicleManagementConfig:
    fleet_size: int
    max_age: int
    max_replacements: int


@dataclass(frozen=True)
class ResidualValueConfig:
    initial_depreciation: float
    annual_depreciation_rate: float
    market_elasticity: float
    floor_fraction: float


@dataclass(frozen=True)
class ForecastRatesConfig:
    """Growth rates and convergence beliefs used by LookaheadAgent.

    All values are fractional (e.g. ``0.02`` = 2 % per year).  Defaults to
    0.0, meaning the agent assumes current observed values remain constant.

    ``BET_price_gap_closure_rate`` is the fraction of the BET–DT purchase-price
    gap the agent believes will close each year.  E.g. ``0.05`` implies parity
    after 20 years.  Set to ``0.0`` to keep the gap constant.
    """

    diesel_price_growth: float = 0.0
    electricity_price_growth: float = 0.0
    purchase_price_growth_DT: float = 0.0
    BET_price_gap_closure_rate: float = 0.0
    BET_productivity_growth: float = 0.0


@dataclass(frozen=True)
class LookaheadConfig:
    """Configuration for the deterministic lookahead planning model."""

    horizon: int
    forecast_rates: ForecastRatesConfig = field(default_factory=ForecastRatesConfig)


@dataclass(frozen=True)
class EnvConfig:
    simulation_period: SimulationPeriodConfig
    diesel_price: StochasticPriceConfig
    electricity_price: StochasticPriceConfig
    DT_price: DTPriceConfig
    BET_price: BETPriceConfig
    BET_productivity: BETProductivityConfig
    operational: OperationalConfig
    economic: EconomicConfig
    vehicle_management: VehicleManagementConfig
    residual_value: ResidualValueConfig
    lookahead: LookaheadConfig = field(
        default_factory=lambda: LookaheadConfig(horizon=10)
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EnvConfig:
        simulation_period = _require_section(data, "simulation_period")
        diesel_price = _require_section(data, "diesel_price")
        electricity_price = _require_section(data, "electricity_price")
        dt_price = _require_section(data, "DT_price")
        bet_price = _require_section(data, "BET_price")
        bet_productivity = _require_section(data, "BET_productivity")
        operational = _require_section(data, "operational")
        economic = _require_section(data, "economic")
        vehicle_management = _require_section(data, "vehicle_management")
        residual_value = _require_section(data, "residual_value")

        return cls(
            simulation_period=SimulationPeriodConfig(
                base_year=int(simulation_period["base_year"]),
                final_year=int(simulation_period["final_year"]),
            ),
            diesel_price=StochasticPriceConfig(
                initial_price=float(diesel_price["initial_price"]),
                growth_rate=float(diesel_price["growth_rate"]),
                volatility=float(diesel_price["volatility"]),
            ),
            electricity_price=StochasticPriceConfig(
                initial_price=float(electricity_price["initial_price"]),
                growth_rate=float(electricity_price["growth_rate"]),
                volatility=float(electricity_price["volatility"]),
            ),
            DT_price=DTPriceConfig(
                initial_price=float(dt_price["initial_price"]),
            ),
            BET_price=BETPriceConfig(
                initial_price=float(bet_price["initial_price"]),
                long_term_mean=float(bet_price["long_term_mean"]),
                mean_reversion_strength=float(bet_price["mean_reversion_strength"]),
                purchase_price_volatility=float(bet_price["purchase_price_volatility"]),
            ),
            BET_productivity=BETProductivityConfig(
                start=float(bet_productivity["start"]),
                max=float(bet_productivity["max"]),
                k_min=float(bet_productivity["k_min"]),
                k_max=float(bet_productivity["k_max"]),
                t0_min=int(bet_productivity["t0_min"]),
                t0_max=int(bet_productivity["t0_max"]),
            ),
            operational=OperationalConfig(
                fuel_consumption_l_per_km=float(
                    operational["fuel_consumption_l_per_km"]
                ),
                electricity_consumption_kwh_per_km=float(
                    operational["electricity_consumption_kwh_per_km"]
                ),
                driver_salary_annual=float(operational["driver_salary_annual"]),
                income_per_km=float(operational["income_per_km"]),
                annual_mileage_km=float(operational["annual_mileage_km"]),
            ),
            economic=EconomicConfig(
                interest_rate=float(economic["interest_rate"]),
                loan_fraction=float(economic["loan_fraction"]),
                economic_lifetime=int(economic["economic_lifetime"]),
                discount_factor=float(economic["discount_factor"]),
            ),
            vehicle_management=VehicleManagementConfig(
                fleet_size=int(vehicle_management["fleet_size"]),
                max_age=int(vehicle_management["max_age"]),
                max_replacements=int(vehicle_management["max_replacements"]),
            ),
            residual_value=ResidualValueConfig(
                initial_depreciation=float(residual_value["initial_depreciation"]),
                annual_depreciation_rate=float(
                    residual_value["annual_depreciation_rate"]
                ),
                market_elasticity=float(residual_value["market_elasticity"]),
                floor_fraction=float(residual_value["floor_fraction"]),
            ),
            lookahead=_parse_lookahead(data.get("lookahead")),
        )


def _parse_lookahead(section: dict[str, Any] | None) -> LookaheadConfig:
    """Parse the optional ``lookahead`` section; returns defaults when absent."""
    if not isinstance(section, dict):
        return LookaheadConfig(horizon=10)
    rates_raw = section.get("forecast_rates", {})
    rates = ForecastRatesConfig(
        diesel_price_growth=float(rates_raw.get("diesel_price_growth", 0.0)),
        electricity_price_growth=float(rates_raw.get("electricity_price_growth", 0.0)),
        purchase_price_growth_DT=float(rates_raw.get("purchase_price_growth_DT", 0.0)),
        BET_price_gap_closure_rate=float(
            rates_raw.get("BET_price_gap_closure_rate", 0.0)
        ),
        BET_productivity_growth=float(rates_raw.get("BET_productivity_growth", 0.0)),
    )
    return LookaheadConfig(
        horizon=int(section["horizon"]),
        forecast_rates=rates,
    )


def load_env_config(config_path: str | Path) -> EnvConfig:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping at top-level: {path}")
    config = EnvConfig.from_dict(data)
    _validate_config(config)
    return config


def _require_section(data: dict[str, Any], section_name: str) -> dict[str, Any]:
    section = data.get(section_name)
    if not isinstance(section, dict):
        raise ValueError(f"Missing or invalid section '{section_name}'")
    return section


def _validate_config(config: EnvConfig) -> None:
    if config.simulation_period.base_year >= config.simulation_period.final_year:
        raise ValueError("simulation_period.base_year must be less than final_year")
    if config.vehicle_management.fleet_size <= 0:
        raise ValueError("vehicle_management.fleet_size must be > 0")
    if config.vehicle_management.max_age <= 0:
        raise ValueError("vehicle_management.max_age must be > 0")
    if config.vehicle_management.max_replacements < 0:
        raise ValueError("vehicle_management.max_replacements must be >= 0")
    if (
        config.vehicle_management.max_replacements
        > config.vehicle_management.fleet_size
    ):
        raise ValueError(
            "vehicle_management.max_replacements must be <= vehicle_management.fleet_size"
        )
    if not 0.0 <= config.economic.loan_fraction <= 1.0:
        raise ValueError("economic.loan_fraction must be in [0, 1]")
    if config.economic.economic_lifetime <= 0:
        raise ValueError("economic.economic_lifetime must be > 0")
    if not 0.0 < config.economic.discount_factor <= 1.0:
        raise ValueError("economic.discount_factor must be in (0, 1]")
