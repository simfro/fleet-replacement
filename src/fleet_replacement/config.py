from __future__ import annotations

from dataclasses import dataclass
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
class DieselVehicleConfig:
    initial_price: float


@dataclass(frozen=True)
class ElectricVehicleConfig:
    initial_price: float
    long_term_mean: float
    mean_reversion_strength: float
    purchase_price_volatility: float


@dataclass(frozen=True)
class BetProductivityConfig:
    start: float
    max: float
    k: float
    t0: int


@dataclass(frozen=True)
class OperationalConfig:
    fuel_consumption_l_per_km: float
    electricity_consumption_kwh_per_km: float
    driver_salary_annual: float
    income_per_km: float
    annual_mileage_km: float


@dataclass(frozen=True)
class LoanConfig:
    interest_rate: float
    fraction: float
    economic_lifetime: int


@dataclass(frozen=True)
class VehicleManagementConfig:
    max_age: int
    max_replacements: int


@dataclass(frozen=True)
class ResidualValueConfig:
    initial_depreciation: float
    annual_depreciation_rate: float
    market_elasticity: float
    floor_fraction: float


@dataclass(frozen=True)
class DiscountConfig:
    gamma: float


@dataclass(frozen=True)
class FleetInitializationConfig:
    initial_diesel_fraction: float
    initial_age_min: int
    initial_age_max: int


@dataclass(frozen=True)
class EnvConfig:
    simulation_period: SimulationPeriodConfig
    fleet_size: int
    diesel_price: StochasticPriceConfig
    electricity_price: StochasticPriceConfig
    diesel_vehicle: DieselVehicleConfig
    electric_vehicle: ElectricVehicleConfig
    bet_productivity: BetProductivityConfig
    operational: OperationalConfig
    loan: LoanConfig
    vehicle_management: VehicleManagementConfig
    residual_value: ResidualValueConfig
    discount: DiscountConfig
    fleet_initialization: FleetInitializationConfig
    bet_variant_multipliers: tuple[float, float, float]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EnvConfig:
        simulation_period = _require_section(data, "simulation_period")
        diesel_price = _require_section(data, "diesel_price")
        electricity_price = _require_section(data, "electricity_price")
        diesel_vehicle = _require_section(data, "diesel_vehicle")
        electric_vehicle = _require_section(data, "electric_vehicle")
        bet_productivity = _require_section(data, "bet_productivity")
        operational = _require_section(data, "operational")
        loan = _require_section(data, "loan")
        vehicle_management = _require_section(data, "vehicle_management")
        residual_value = _require_section(data, "residual_value")
        discount = _require_section(data, "discount")
        fleet_initialization = _require_section(data, "fleet_initialization")

        multipliers = data.get("bet_variant_multipliers", [1.0, 1.15, 1.3])
        if not isinstance(multipliers, list) or len(multipliers) != 3:
            raise ValueError(
                "'bet_variant_multipliers' must be a list of exactly 3 values"
            )

        return cls(
            simulation_period=SimulationPeriodConfig(
                base_year=int(simulation_period["base_year"]),
                final_year=int(simulation_period["final_year"]),
            ),
            fleet_size=int(data.get("fleet_size", 10)),
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
            diesel_vehicle=DieselVehicleConfig(
                initial_price=float(diesel_vehicle["initial_price"]),
            ),
            electric_vehicle=ElectricVehicleConfig(
                initial_price=float(electric_vehicle["initial_price"]),
                long_term_mean=float(electric_vehicle["long_term_mean"]),
                mean_reversion_strength=float(
                    electric_vehicle["mean_reversion_strength"]
                ),
                purchase_price_volatility=float(
                    electric_vehicle["purchase_price_volatility"]
                ),
            ),
            bet_productivity=BetProductivityConfig(
                start=float(bet_productivity["start"]),
                max=float(bet_productivity["max"]),
                k=float(bet_productivity["k"]),
                t0=int(bet_productivity["t0"]),
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
            loan=LoanConfig(
                interest_rate=float(loan["interest_rate"]),
                fraction=float(loan["fraction"]),
                economic_lifetime=int(loan["economic_lifetime"]),
            ),
            vehicle_management=VehicleManagementConfig(
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
            discount=DiscountConfig(
                gamma=float(discount["gamma"]),
            ),
            fleet_initialization=FleetInitializationConfig(
                initial_diesel_fraction=float(
                    fleet_initialization["initial_diesel_fraction"]
                ),
                initial_age_min=int(fleet_initialization["initial_age_min"]),
                initial_age_max=int(fleet_initialization["initial_age_max"]),
            ),
            bet_variant_multipliers=(
                float(multipliers[0]),
                float(multipliers[1]),
                float(multipliers[2]),
            ),
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
    if config.fleet_size <= 0:
        raise ValueError("fleet_size must be > 0")
    if not 0.0 <= config.fleet_initialization.initial_diesel_fraction <= 1.0:
        raise ValueError(
            "fleet_initialization.initial_diesel_fraction must be in [0, 1]"
        )
    if config.fleet_initialization.initial_age_min < 0:
        raise ValueError("fleet_initialization.initial_age_min must be >= 0")
    if config.fleet_initialization.initial_age_max > config.vehicle_management.max_age:
        raise ValueError(
            "fleet_initialization.initial_age_max must be <= vehicle_management.max_age"
        )
    if (
        config.fleet_initialization.initial_age_min
        > config.fleet_initialization.initial_age_max
    ):
        raise ValueError(
            "fleet_initialization.initial_age_min must be <= initial_age_max"
        )
    if config.vehicle_management.max_age <= 0:
        raise ValueError("vehicle_management.max_age must be > 0")
    if not 0.0 < config.discount.gamma <= 1.0:
        raise ValueError("discount.gamma must be in (0, 1]")
