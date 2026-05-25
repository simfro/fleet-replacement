"""
Lookahead fleet replacement planning model implemented with linopy.

The model is solved iteratively inside a stochastic simulation:
at each step the caller constructs a FleetReplacementData from the
current environment observation, calls solve(), and extracts the
first-period decisions via best_immediate_actions().

Index conventions
-----------------
time         : 0 .. horizon-1       (planning periods)
slot         : 0 .. fleet_size-1    (fleet slots)
vehicle_type : "DT" | "BET"
vehicle_age  : 0 .. max_age
decision     : "Keep" | "Replace_DT" | "Replace_BET"

Variables
---------
R[time, slot, vehicle_type, vehicle_age]
    Binary. 1 iff the vehicle of (type, age) occupies fleet slot `slot` at time t.

x[time, slot, vehicle_type, vehicle_age, decision]
    Binary. 1 iff decision `d` is taken for the vehicle of (type, age)
    in slot `slot` at time t.
    Defined only for t in 0 .. horizon-2  (no decision at terminal period).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import linopy
import numpy as np
import xarray as xr

from fleet_replacement.config import EnvConfig

# ---------------------------------------------------------------------------
# Domain enumerations
# ---------------------------------------------------------------------------


class VehicleType(StrEnum):
    DT = "DT"
    BET = "BET"


class Decision(StrEnum):
    KEEP = "Keep"
    REPLACE_DT = "Replace_DT"
    REPLACE_BET = "Replace_BET"


# Ordered lists used as xarray coordinate values
VEHICLE_TYPES: list[VehicleType] = list(VehicleType)
DECISIONS: list[Decision] = list(Decision)


@dataclass(frozen=True)
class InitialVehicle:
    """Vehicle occupying a fleet slot at the start of the planning horizon."""

    vehicle_type: VehicleType
    age: int
    purchase_price: float


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InfoState:
    """
    Current information state forwarded from the simulation environment.

    All prices are in the same currency unit used by the environment.
    """

    year: int
    energy_price_diesel: float
    energy_price_electricity: float
    purchase_price_DT: float
    purchase_price_BET: float
    productivity_BET: float


@dataclass(frozen=True)
class ModelParams:
    """
    Deterministic lookahead parameters.

    Decoupled from ``EnvConfig`` so that the planning model can use different
    (e.g. mean-path) assumptions than the stochastic environment process.
    """

    horizon: int
    max_age: int
    # Operational
    income_per_km: float
    annual_mileage_km: float
    fuel_consumption_l_per_km: float
    el_consumption_kWh_per_km: float
    driver_salary_annual: float
    # Productivity cap (domain constraint applied during forecast generation)
    BET_productivity_max: float
    # Financial
    interest_rate: float
    loan_fraction: float
    economic_lifetime: int
    # Discount factor
    gamma: float

    @classmethod
    def from_env_config(cls, config: EnvConfig, horizon: int) -> ModelParams:
        """Construct ModelParams from an EnvConfig."""
        return cls(
            horizon=horizon,
            max_age=config.vehicle_management.max_age,
            income_per_km=config.operational.income_per_km,
            annual_mileage_km=config.operational.annual_mileage_km,
            fuel_consumption_l_per_km=config.operational.fuel_consumption_l_per_km,
            el_consumption_kWh_per_km=config.operational.electricity_consumption_kwh_per_km,
            driver_salary_annual=config.operational.driver_salary_annual,
            BET_productivity_max=config.BET_productivity.max,
            interest_rate=config.economic.interest_rate,
            loan_fraction=config.economic.loan_fraction,
            economic_lifetime=config.economic.economic_lifetime,
            gamma=config.economic.discount_factor,
        )


@dataclass(frozen=True)
class ForecastParams:
    """
    Linear growth rates for constructing price and productivity forecasts.

    All rates are fractional annual growth (e.g. ``0.02`` = 2 % per year).
    Defaults to zero growth (flat forecast from current observed values).
    """

    diesel_price_growth: float = 0.0
    electricity_price_growth: float = 0.0
    purchase_price_growth_DT: float = 0.0
    purchase_price_growth_BET: float = 0.0
    BET_productivity_growth: float = 0.0


@dataclass
class Forecast:
    """
    Pre-computed linear price and productivity forecasts over a planning horizon.

    All arrays have shape ``(horizon,)`` with index ``t=0`` representing the
    current period.  Build with :func:`make_forecast`.
    """

    purchase_price_DT: np.ndarray
    purchase_price_BET: np.ndarray
    energy_price_diesel: np.ndarray
    energy_price_electricity: np.ndarray
    productivity_BET: np.ndarray


@dataclass
class FleetReplacementData:
    """
    Complete data package passed to ``build_model``.

    ``initial_fleet`` is a list of ``(vehicle_type, age)`` tuples, one per slot,
    ordered by slot index (0-based).  This matches the format of the environment
    observation arrays ``is_electric`` and ``age``.

    Derived index arrays are built automatically on construction.
    """

    params: ModelParams
    info_state: InfoState
    # One entry per fleet slot, ordered by slot index (0-based)
    initial_fleet: list[InitialVehicle]

    # Derived (populated in __post_init__)
    time_span: np.ndarray = field(init=False)
    decision_time_span: np.ndarray = field(init=False)
    fleet_slots: np.ndarray = field(init=False)
    vehicle_types: list[VehicleType] = field(init=False)
    vehicle_ages: np.ndarray = field(init=False)
    decisions: list[Decision] = field(init=False)

    def __post_init__(self) -> None:
        self.time_span = np.arange(self.params.horizon)
        self.decision_time_span = self.time_span[:-1]  # no decision at terminal step
        self.fleet_slots = np.arange(len(self.initial_fleet))
        self.vehicle_types = VEHICLE_TYPES
        self.vehicle_ages = np.arange(self.params.max_age + 1)
        self.decisions = DECISIONS

    @property
    def fleet_size(self) -> int:
        return len(self.initial_fleet)

    @classmethod
    def from_env_observation(
        cls,
        obs: dict,
        params: ModelParams,
    ) -> FleetReplacementData:
        """
        Build FleetReplacementData from a Gymnasium environment observation dict.

        Expects the observation format produced by ``FleetReplacementEnv``:
          obs["fleet"]["is_electric"]   : array of 0/1, shape (fleet_size,)
          obs["fleet"]["age"]           : array of int,  shape (fleet_size,)
          obs["fleet"]["purchase_price"]: array of float, shape (fleet_size,)
          obs["information_state"]["current_year"]        : shape (1,)
          obs["information_state"]["energy_price_diesel"] : shape (1,)
          obs["information_state"]["energy_price_electricity"]: shape (1,)
          obs["information_state"]["purchase_price_DT"]   : shape (1,)
          obs["information_state"]["purchase_price_BET"]  : shape (1,)
          obs["information_state"]["productivity_BET"]    : shape (1,)
        """
        fleet_obs = obs["fleet"]
        info_obs = obs["information_state"]

        is_electric = fleet_obs["is_electric"]
        ages = fleet_obs["age"]
        purchase_prices = fleet_obs["purchase_price"]

        initial_fleet = [
            InitialVehicle(
                vehicle_type=(
                    VehicleType.BET if int(is_electric[i]) == 1 else VehicleType.DT
                ),
                age=int(ages[i]),
                purchase_price=float(purchase_prices[i]),
            )
            for i in range(len(ages))
        ]

        info_state = InfoState(
            year=int(info_obs["current_year"][0]),
            energy_price_diesel=float(info_obs["energy_price_diesel"][0]),
            energy_price_electricity=float(info_obs["energy_price_electricity"][0]),
            purchase_price_DT=float(info_obs["purchase_price_DT"][0]),
            purchase_price_BET=float(info_obs["purchase_price_BET"][0]),
            productivity_BET=float(info_obs["productivity_BET"][0]),
        )

        return cls(
            params=params,
            info_state=info_state,
            initial_fleet=initial_fleet,
        )


# ---------------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------------


def forecast_purchase_price_DT(
    info_state: InfoState,
    horizon: int,
    forecast_params: ForecastParams,
) -> np.ndarray:
    """Linear forecast for DT purchase price."""
    ts = np.arange(horizon, dtype=float)
    return info_state.purchase_price_DT * (
        1 + forecast_params.purchase_price_growth_DT * ts
    )


def forecast_purchase_price_BET(
    info_state: InfoState,
    horizon: int,
    forecast_params: ForecastParams,
) -> np.ndarray:
    """Linear forecast for BET purchase price."""
    ts = np.arange(horizon, dtype=float)
    return info_state.purchase_price_BET * (
        1 + forecast_params.purchase_price_growth_BET * ts
    )


def forecast_energy_price_diesel(
    info_state: InfoState,
    horizon: int,
    forecast_params: ForecastParams,
) -> np.ndarray:
    """Linear forecast for diesel energy price."""
    ts = np.arange(horizon, dtype=float)
    return info_state.energy_price_diesel * (
        1 + forecast_params.diesel_price_growth * ts
    )


def forecast_energy_price_electricity(
    info_state: InfoState,
    horizon: int,
    forecast_params: ForecastParams,
) -> np.ndarray:
    """Linear forecast for electricity energy price."""
    ts = np.arange(horizon, dtype=float)
    return info_state.energy_price_electricity * (
        1 + forecast_params.electricity_price_growth * ts
    )


def forecast_productivity_BET(
    info_state: InfoState,
    horizon: int,
    BET_productivity_max: float,
    forecast_params: ForecastParams,
) -> np.ndarray:
    """Linear forecast for BET productivity, capped at ``BET_productivity_max``."""
    ts = np.arange(horizon, dtype=float)
    return np.minimum(
        info_state.productivity_BET
        * (1 + forecast_params.BET_productivity_growth * ts),
        BET_productivity_max,
    )


def make_forecast(
    info_state: InfoState,
    horizon: int,
    BET_productivity_max: float,
    forecast_params: ForecastParams | None = None,
) -> Forecast:
    """
    Assemble a :class:`Forecast` by calling the individual forecast functions.

    Each series defaults to a linear projection from the current ``InfoState``
    using the growth rates in ``forecast_params``.  Override any of the
    ``forecast_*`` functions to customise a specific series.

    Parameters
    ----------
    info_state:
        Current observed prices and BET productivity.
    horizon:
        Number of planning periods to forecast.
    BET_productivity_max:
        Hard upper cap on BET productivity (relative to DT = 1).
    forecast_params:
        Growth rates for each series.  Defaults to :class:`ForecastParams`
        (all zeros, i.e. flat forecasts).
    """
    if forecast_params is None:
        forecast_params = ForecastParams()
    return Forecast(
        purchase_price_DT=forecast_purchase_price_DT(
            info_state, horizon, forecast_params
        ),
        purchase_price_BET=forecast_purchase_price_BET(
            info_state, horizon, forecast_params
        ),
        energy_price_diesel=forecast_energy_price_diesel(
            info_state, horizon, forecast_params
        ),
        energy_price_electricity=forecast_energy_price_electricity(
            info_state, horizon, forecast_params
        ),
        productivity_BET=forecast_productivity_BET(
            info_state, horizon, BET_productivity_max, forecast_params
        ),
    )


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------


def build_model(data: FleetReplacementData, forecast: Forecast) -> linopy.Model:
    """
    Construct and return an unsolved linopy.Model for fleet replacement.

    Variables, constraints, and objective are added by sub-functions.
    Call ``solve(model)`` to find the optimal solution.
    """
    model = linopy.Model()

    # R[time, slot, vehicle_type, vehicle_age]
    # Binary fleet-state variable.
    model.add_variables(
        coords={
            "time": data.time_span,
            "slot": data.fleet_slots,
            "vehicle_type": data.vehicle_types,
            "vehicle_age": data.vehicle_ages,
        },
        binary=True,
        name="R",
    )

    # x[time, slot, vehicle_type, vehicle_age, decision]
    # Binary decision variable for all non-terminal periods.
    model.add_variables(
        coords={
            "time": data.decision_time_span,
            "slot": data.fleet_slots,
            "vehicle_type": data.vehicle_types,
            "vehicle_age": data.vehicle_ages,
            "decision": data.decisions,
        },
        binary=True,
        name="x",
    )

    add_constraints(model, data)
    add_objective(model, data, forecast)

    return model


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------


def add_constraints(model: linopy.Model, data: FleetReplacementData) -> None:
    """Add all fleet replacement constraints to the model."""
    p = data.params
    R = model.variables["R"]
    x = model.variables["x"]
    max_age = p.max_age

    # Initial fleet: fix vehicle state at t=0 for each slot
    for slot_idx, vehicle in enumerate(data.initial_fleet):
        model.add_constraints(
            R.sel(
                time=0,
                slot=slot_idx,
                vehicle_type=vehicle.vehicle_type,
                vehicle_age=vehicle.age,
            )
            == 1,
            name=f"initial_fleet_slot_{slot_idx}",
        )

    # Exactly one vehicle occupies each fleet slot at every time period
    model.add_constraints(
        R.sum(dims=["vehicle_type", "vehicle_age"]) == 1,
        name="single_vehicle_per_slot_and_time",
    )

    # Exactly one decision is made for each fleet slot at every decision period
    model.add_constraints(
        x.sum(dims=["vehicle_type", "vehicle_age", "decision"]) == 1,
        name="single_decision_per_slot_and_time",
    )

    # A decision can only be made for a vehicle present in the slot
    model.add_constraints(
        x <= R.isel(time=slice(0, len(data.decision_time_span))),
        name="no_decision_for_nonexistent_vehicle",
    )

    # A vehicle at maximum age must be replaced (cannot keep)
    model.add_constraints(
        x.sel(vehicle_age=max_age, decision=Decision.KEEP) == 0,
        name="replace_at_max_age",
    )

    # Keep transition: a kept vehicle ages by one period
    model.add_constraints(
        R.isel(time=slice(1, None), vehicle_age=slice(1, max_age + 1)).assign_coords(
            time=data.decision_time_span,
            vehicle_age=data.vehicle_ages[:-1],
        )
        >= x.sel(decision=Decision.KEEP).isel(vehicle_age=slice(0, max_age)),
        name="keep_transitions",
    )

    # Replace_DT transition: new DT at age 0 must exist the period after replacement
    model.add_constraints(
        R.sel(vehicle_type=VehicleType.DT, vehicle_age=0)
        .isel(time=slice(1, None))
        .assign_coords(time=data.decision_time_span)
        >= x.sel(decision=Decision.REPLACE_DT),
        name="replace_DT_transitions",
    )

    # Replace_BET transition: new BET at age 0 must exist the period after replacement
    model.add_constraints(
        R.sel(vehicle_type=VehicleType.BET, vehicle_age=0)
        .isel(time=slice(1, None))
        .assign_coords(time=data.decision_time_span)
        >= x.sel(decision=Decision.REPLACE_BET),
        name="replace_BET_transitions",
    )


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------


def _build_purchase_price(
    data: FleetReplacementData, forecast: Forecast
) -> xr.DataArray:
    """
    Build a DataArray of purchase prices indexed by (time, slot, vehicle_type, vehicle_age).

    Used as a static coefficient for the annual interest cost term in the objective.

    The purchase price of the vehicle present at (t, slot, type, age) is:
    - Original vehicle in slot s: age == initial_fleet[s].age + t  → historical purchase price
    - Replacement vehicle: bought at decision time tau = t - age - 1  → forecast price at tau

    The two tracks are mutually exclusive: the original vehicle's tau would be
    -(initial_age + 1) < 0, so the replacement loop never touches those entries.
    """
    h = data.params.horizon
    max_age = data.params.max_age
    dt_idx = data.vehicle_types.index(VehicleType.DT)
    bet_idx = data.vehicle_types.index(VehicleType.BET)

    pp_vals = np.zeros((h, data.fleet_size, len(data.vehicle_types), max_age + 1))

    # Replacement vehicle prices: vehicle at age a at time t was bought at tau = t - a - 1
    for t in range(h):
        for a in range(max_age + 1):
            tau = t - a - 1
            if tau >= 0:
                pp_vals[t, :, dt_idx, a] = forecast.purchase_price_DT[tau]
                pp_vals[t, :, bet_idx, a] = forecast.purchase_price_BET[tau]

    # Original vehicle prices (complement — their tau is always negative)
    for s, vehicle in enumerate(data.initial_fleet):
        v_idx = data.vehicle_types.index(vehicle.vehicle_type)
        for t in range(h):
            a = vehicle.age + t
            if a <= max_age:
                pp_vals[t, s, v_idx, a] = vehicle.purchase_price

    return xr.DataArray(
        pp_vals,
        dims=["time", "slot", "vehicle_type", "vehicle_age"],
        coords={
            "time": data.time_span,
            "slot": data.fleet_slots,
            "vehicle_type": data.vehicle_types,
            "vehicle_age": data.vehicle_ages,
        },
    )


def add_objective(
    model: linopy.Model, data: FleetReplacementData, forecast: Forecast
) -> None:
    """Add the revenue-minus-costs maximisation objective to the model."""
    p = data.params
    R = model.variables["R"]
    h = p.horizon

    # Discount factors
    discount = xr.DataArray(
        [p.gamma**t for t in range(h)],
        dims=["time"],
        coords={"time": data.time_span},
    )

    # BET productivity trajectory from forecast (DT productivity is always 1)
    productivity = xr.DataArray(
        np.stack([np.ones(h), forecast.productivity_BET], axis=1),
        dims=["time", "vehicle_type"],
        coords={"time": data.time_span, "vehicle_type": data.vehicle_types},
    )

    # Energy consumption (per km) by vehicle type: DT=diesel, BET=electricity
    energy_consumption = xr.DataArray(
        [p.fuel_consumption_l_per_km, p.el_consumption_kWh_per_km],
        dims=["vehicle_type"],
        coords={"vehicle_type": data.vehicle_types},
    )

    # Energy price trajectories from forecast
    energy_price = xr.DataArray(
        np.stack(
            [forecast.energy_price_diesel, forecast.energy_price_electricity], axis=1
        ),
        dims=["time", "vehicle_type"],
        coords={"time": data.time_span, "vehicle_type": data.vehicle_types},
    )

    purchase_price = _build_purchase_price(data, forecast)

    obj_revenue = (
        p.income_per_km * p.annual_mileage_km * (discount * productivity * R).sum()
    )
    obj_energy_cost = (
        p.annual_mileage_km
        * (discount * productivity * energy_consumption * energy_price * R).sum()
    )
    obj_interest_cost = (
        p.interest_rate * p.loan_fraction * (discount * purchase_price * R).sum()
    )

    model.add_objective(obj_revenue - obj_energy_cost - obj_interest_cost, sense="max")


# ---------------------------------------------------------------------------
# Solve and result extraction
# ---------------------------------------------------------------------------


def solve(model: linopy.Model) -> linopy.Model:
    """
    Solve a pre-built fleet replacement model.

    Returns the solved model. Raises ``RuntimeError`` if no optimal solution
    is found.
    """
    status, condition = model.solve(solver_name="highs")
    if condition != "optimal":
        raise RuntimeError(
            f"Model did not solve to optimality. "
            f"Status: {status!r}, condition: {condition!r}"
        )
    return model


def best_immediate_actions(
    model: linopy.Model, data: FleetReplacementData
) -> list[Decision]:
    """
    Extract the optimal first-period decision for each fleet slot.

    Returns a list of decision strings (e.g. "Keep", "Replace_DT", "Replace_BET"),
    one per slot, ordered by slot index.

    This is the value fed back into the stochastic simulation at each step.
    """
    x = model.solution["x"]  # xarray.DataArray of solved values
    t0 = int(data.decision_time_span[0])
    x_t0 = x.sel(time=t0)  # shape: (slot, vehicle_type, vehicle_age, decision)

    actions: list[Decision] = []
    for slot in data.fleet_slots:
        x_slot = x_t0.sel(slot=slot)
        # Find the (vehicle_type, vehicle_age, decision) combination with value > 0.5
        idx = x_slot.values.argmax()
        flat_idx = np.unravel_index(idx, x_slot.shape)
        decision_idx = flat_idx[-1]  # last coordinate is 'decision'
        actions.append(data.decisions[decision_idx])

    return actions
