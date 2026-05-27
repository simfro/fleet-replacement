from enum import Enum
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from fleet_replacement.config import EnvConfig, load_env_config
from fleet_replacement.envs.models import (
    BET_price_step,
    BET_productivity_logistic,
    compute_reward,
    diesel_price_step,
    DT_price_step,
    electricity_price_step,
    REWARD_COMPONENT_KEYS,
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
        self._reward: dict[str, float]

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

        return {"reward": self._reward}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self._current_year = self.start_year
        self._fleet = self._initial_fleet(self.fleet_size)
        self._info_state = self._initial_info_state()
        self._reward = dict.fromkeys(REWARD_COMPONENT_KEYS, 0.0)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def _initial_fleet(self, fleet_size: int):
        """Build the deterministic starting fleet state for an episode.

        Vehicles are initialised with staggered ages so the fleet is not
        artificially synchronised: slot ``i`` starts at age
        ``i % (max_age + 1)``.  All vehicles start as diesel trucks with
        purchase prices set to the configured initial DT purchase price.
        """
        ages = np.array(
            [i % (self.max_vehicle_age + 1) for i in range(fleet_size)],
            dtype=np.int32,
        )
        return {
            "is_electric": np.zeros(fleet_size, dtype=np.int8),
            "age": ages,
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

        self._reward = compute_reward(
            self._fleet, self._info_state, action, self.config
        )
        reward = self._reward["total_reward"]

        observation = self._get_obs()
        info = self._get_info()

        terminated = self._current_year >= self.final_year

        return (
            observation,
            reward,
            terminated,
            False,
            info,
        )

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
            "purchase_price_BET": BET_price_step(
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

        if info["reward_components"]:
            rc = info["reward_components"]
            print("Reward components")
            print(f"  Revenue: {float(rc['revenue']):,.2f}")
            print(f"  Total cost: {float(rc['total_cost']):,.2f}")
            print(f"  Sale result: {float(rc['sale_result']):,.2f}")
            print(f"  Total reward: {float(rc['total_reward']):,.2f}")
