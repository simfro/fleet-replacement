from enum import Enum
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from fleet_replacement.config import EnvConfig, load_env_config
from fleet_replacement.envs.models import (
    BET_price_mean_reversion_step,
    BET_productivity_logistic,
    diesel_price_step,
    DT_price_step,
    electricity_price_step,
)


class Actions(Enum):
    KEEP = 0
    REPLACE_DT = 1
    REPLACE_BET = 2


class FleetReplacementEnv(gym.Env):
    def __init__(
        self,
        config: EnvConfig | None = None,
        config_path: str | Path | None = None,
        render_mode=None,
    ):
        if config is None:
            default_config_path = (
                Path(__file__).resolve().parents[3] / "configs" / "env.yaml"
            )
            config = load_env_config(config_path or default_config_path)

        self.config = config
        self.fleet_size = self.config.vehicle_management.fleet_size
        self.max_vehicle_age = self.config.vehicle_management.max_age
        self.start_year = self.config.simulation_period.base_year
        self.final_year = self.config.simulation_period.final_year
        self._current_year = self.start_year
        self.render_mode = render_mode

        # is_electric: 0 = Diesel Truck (DT), 1 = Battery Electric Truck (BET)
        self.observation_space = spaces.Dict(
            {
                "fleet": spaces.Dict(
                    {
                        "is_electric": spaces.MultiBinary(self.fleet_size),
                        "age": spaces.Box(
                            low=0,
                            high=self.max_vehicle_age,
                            shape=(self.fleet_size,),
                            dtype=np.int32,
                        ),
                        "purchase_price": spaces.Box(
                            low=0.0,
                            high=np.finfo(np.float32).max,
                            shape=(self.fleet_size,),
                            dtype=np.float32,
                        ),
                    }
                ),
                "information_state": spaces.Dict(
                    {
                        "current_year": spaces.Box(
                            low=self.start_year,
                            high=self.final_year,
                            shape=(1,),
                            dtype=np.int32,
                        ),
                        "energy_price_diesel": spaces.Box(
                            low=0.0,
                            high=np.finfo(np.float32).max,
                            shape=(1,),
                            dtype=np.float32,
                        ),
                        "energy_price_electricity": spaces.Box(
                            low=0.0,
                            high=np.finfo(np.float32).max,
                            shape=(1,),
                            dtype=np.float32,
                        ),
                        "purchase_price_DT": spaces.Box(
                            low=0.0,
                            high=np.finfo(np.float32).max,
                            shape=(1,),
                            dtype=np.float32,
                        ),
                        "purchase_price_BET": spaces.Box(
                            low=0.0,
                            high=np.finfo(np.float32).max,
                            shape=(1,),
                            dtype=np.float32,
                        ),
                        "productivity_BET": spaces.Box(
                            low=0.0,
                            high=1.0,
                            shape=(1,),
                            dtype=np.float32,
                        ),
                    }
                ),
            }
        )

        self._fleet = self._initial_fleet(self.fleet_size)
        self._info_state = self._initial_info_state()

        # One action per vehicle:
        # Actions.KEEP = keep, Actions.REPLACE_DT = replace with DT,
        # Actions.REPLACE_BET = replace with BET (medium battery).
        self.action_space = spaces.MultiDiscrete(
            np.full(self.fleet_size, len(Actions), dtype=np.int64)
        )

    def _get_obs(self):
        """Return the current observation dictionary for the agent.

        The observation contains the fleet state and the current market
        information state, matching the declared observation space.
        """
        information_state_obs = {
            "current_year": np.array(
                [self._info_state["current_year"]], dtype=np.int32
            ),
            "energy_price_diesel": np.array(
                [self._info_state["energy_price_diesel"]], dtype=np.float32
            ),
            "energy_price_electricity": np.array(
                [self._info_state["energy_price_electricity"]], dtype=np.float32
            ),
            "purchase_price_DT": np.array(
                [self._info_state["purchase_price_DT"]], dtype=np.float32
            ),
            "purchase_price_BET": np.array(
                [self._info_state["purchase_price_BET"]], dtype=np.float32
            ),
            "productivity_BET": np.array(
                [self._info_state["productivity_BET"]], dtype=np.float32
            ),
        }
        return {"fleet": self._fleet, "information_state": information_state_obs}

    def _get_info(self):
        """Return auxiliary metadata for debugging and analysis."""

        return {
            "total_reward": "---",
            "revenue": "---",
            "cost": "---",
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self._current_year = self.start_year
        self._fleet = self._initial_fleet(self.fleet_size)
        self._info_state = self._initial_info_state()

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def _initial_fleet(self, fleet_size: int):
        """Build the deterministic starting fleet state for an episode.

        All vehicles start as diesel trucks with age 0 and purchase prices set to
        the configured initial DT purchase price.
        """
        return {
            "is_electric": np.zeros(fleet_size, dtype=np.int8),
            "age": np.zeros(fleet_size, dtype=np.int32),
            "purchase_price": np.full(
                fleet_size,
                self.config.DT_price.initial_price,
                dtype=np.float32,
            ),
        }

    def _initial_info_state(self):
        """Create the deterministic initial market information state.

        The state starts at the simulation base year and uses configured
        initial prices for diesel, electricity, DT purchase, and BET purchase.
        """
        return {
            "current_year": self.start_year,
            "energy_price_diesel": self.config.diesel_price.initial_price,
            "energy_price_electricity": self.config.electricity_price.initial_price,
            "purchase_price_DT": self.config.DT_price.initial_price,
            "purchase_price_BET": self.config.BET_price.initial_price,
            "productivity_BET": self.config.BET_productivity.start,
        }

    def step(self, action):
        self._update_fleet(action)
        self._update_info_state()

        observation = self._get_obs()
        info = self._get_info()
        terminated = self._current_year >= self.final_year

        reward = self._get_reward(action)

        return (
            observation,
            reward,
            terminated,
            False,
            info,
        )

    def _get_reward(self, action):
        """Calculate the reward for the current step based on fleet and info state."""

        revenue = 0.0
        diesel_cost = 0.0
        electricity_cost = 0.0
        interest_cost = 0.0
        depericiation_cost = 0.0
        for i in range(self.fleet_size):
            if self._fleet["is_electric"][i]:
                revenue += self._revenue_BET()
                electricity_cost += self._electricity_cost()
            else:
                revenue += self._revenue_DT()
                diesel_cost += self._diesel_cost()
            interest_cost += self._interest_cost(self._fleet["purchase_price"][i])
            depericiation_cost += self._annual_depreciation(
                age=self._fleet["age"][i],
                purchase_price=self._fleet["purchase_price"][i],
            )

        salary_cost = self.config.operational.driver_salary_annual * self.fleet_size

        return revenue - (
            diesel_cost
            + electricity_cost
            + interest_cost
            + depericiation_cost
            + salary_cost
        )

    def _sale_result(self, purchase_price, age, is_electric):
        return self.residual_value(purchase_price, age, is_electric) - self.book_value(
            purchase_price, age
        )

    def residual_value(self, purchase_price, age, is_electric):
        P0 = purchase_price + 0.01  # Avoid zero division
        P_new = self._info_state[
            "purchase_price_BET" if is_electric else "purchase_price_DT"
        ]

        d0 = self.config.residual_value.initial_depreciation
        r = self.config.residual_value.annual_depreciation_rate

        # Core linear piece after immediate drop
        frac = max(0.0, (1 - d0) - r * age)

        # Market adjustment elasticity (0=none, 1=proportional)
        k = self.config.residual_value.market_elasticity
        market_adj = (P_new / P0) ** k

        value = P0 * frac * market_adj

        # Floor at a fraction of purchase price to avoid zero or negative residual values for old vehicles
        floor_val = self.config.residual_value.floor_fraction * P0

        return max(value, floor_val)

    def book_value(self, purchase_price, age):
        return max(
            0.0, purchase_price * (1 - age / self.config.economic.economic_lifetime)
        )

    def _annual_depreciation(self, age, purchase_price):
        if age > self.config.economic.economic_lifetime:
            return 0.0
        else:
            return purchase_price / self.config.economic.economic_lifetime

    def _revenue_BET(self):
        return (
            self._info_state["productivity_BET"]
            * self.config.operational.income_per_km
            * self.config.operational.annual_mileage_km
        )

    def _revenue_DT(self):
        return (
            self.config.operational.income_per_km
            * self.config.operational.annual_mileage_km
        )

    def _interest_cost(self, purchase_price):
        return (
            purchase_price
            * self.config.economic.loan_fraction
            * self.config.economic.interest_rate
        )

    def _electricity_cost(self):
        milage = (
            self.config.operational.annual_mileage_km
            * self._info_state["productivity_BET"]
        )
        electricity_consumption = (
            milage * self.config.operational.electricity_consumption_kwh_per_km
        )
        electricity_cost = (
            electricity_consumption * self._info_state["energy_price_electricity"]
        )
        return electricity_cost

    def _diesel_cost(self):
        milage = self.config.operational.annual_mileage_km
        fuel_consumption = milage * self.config.operational.fuel_consumption_l_per_km
        diesel_cost = fuel_consumption * self._info_state["energy_price_diesel"]
        return diesel_cost

    def _update_fleet(self, action):
        """Apply one replacement/keep action per vehicle and mutate fleet state.

        For each vehicle index, the action either increments age (keep) or
        replaces the vehicle with a DT/BET at age 0 using the current market
        purchase price from the information state.
        """
        for idx, a in enumerate(action):
            if a == Actions.KEEP.value:
                self._fleet["age"][idx] += 1
            elif a == Actions.REPLACE_DT.value:
                self._fleet["is_electric"][idx] = 0
                self._fleet["age"][idx] = 0
                self._fleet["purchase_price"][idx] = self._info_state[
                    "purchase_price_DT"
                ]
            elif a == Actions.REPLACE_BET.value:
                self._fleet["is_electric"][idx] = 1
                self._fleet["age"][idx] = 0
                self._fleet["purchase_price"][idx] = self._info_state[
                    "purchase_price_BET"
                ]
            else:
                raise ValueError(f"Invalid action {a} for vehicle {idx}")

    def _update_info_state(self):
        """Advance simulation year and update all market information variables.

        The update follows the transition model: geometric Brownian motion for
        energy prices, constant DT purchase price, mean-reverting BET purchase
        price, and logistic BET productivity growth.
        """
        self._current_year += 1
        self._info_state = {
            "current_year": self._current_year,
            "energy_price_diesel": diesel_price_step(
                current_price=float(self._info_state["energy_price_diesel"]),
                config=self.config.diesel_price,
                rng=self.np_random,
            ),
            "energy_price_electricity": electricity_price_step(
                current_price=float(self._info_state["energy_price_electricity"]),
                config=self.config.electricity_price,
                rng=self.np_random,
            ),
            "purchase_price_DT": DT_price_step(
                current_price=float(self._info_state["purchase_price_DT"])
            ),
            "purchase_price_BET": BET_price_mean_reversion_step(
                current_price=float(self._info_state["purchase_price_BET"]),
                config=self.config.BET_price,
                rng=self.np_random,
            ),
            "productivity_BET": BET_productivity_logistic(
                year=self._current_year,
                config=self.config.BET_productivity,
            ),
        }

    def render(self):
        """Print the current environment state to the terminal."""
        if self.render_mode not in (None, "human"):
            raise ValueError(
                f"Unsupported render_mode '{self.render_mode}'. Use None or 'human'."
            )

        fleet = self._fleet
        info_state = self._info_state
        info = self._get_info()

        n_electric = int(np.sum(fleet["is_electric"]))
        n_diesel = int(self.fleet_size - n_electric)
        avg_age = float(np.mean(fleet["age"]))

        print("\n=== FleetReplacementEnv ===")
        print(f"Year: {self._current_year}")
        print("Fleet")
        print(f"  Total vehicles: {self.fleet_size}")
        print(f"  Diesel: {n_diesel} | Electric: {n_electric}")
        print(f"  Average age: {avg_age:.2f}")
        print("Information state")
        print(f"  Diesel energy price: {float(info_state['energy_price_diesel']):.3f}")
        print(
            f"  Electricity energy price: {float(info_state['energy_price_electricity']):.3f}"
        )
        print(f"  DT purchase price: {float(info_state['purchase_price_DT']):,.0f}")
        print(f"  BET purchase price: {float(info_state['purchase_price_BET']):,.0f}")
        print(f"  BET productivity: {float(info_state['productivity_BET']):.3f}")
