from enum import Enum
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from fleet_replacement.config import EnvConfig, load_env_config


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
        # TODO: Save important internal information that is not part of the observation but may be useful for analysis.
        return {
            "current_year": self._current_year,
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
        reward = 1  # self._calculate_reward()

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
        self._current_year += 1
        self._info_state = {
            "current_year": self._current_year,
            "energy_price_diesel": self._update_energy_price_diesel(),
            "energy_price_electricity": self._update_energy_price_electricity(),
            "purchase_price_DT": self._update_purchase_price_DT(),
            "purchase_price_BET": self._update_purchase_price_BET(),
            "productivity_BET": self._update_productivity_BET(),
        }

    def _update_energy_price_diesel(self) -> float:
        diesel_price = float(self._info_state["energy_price_diesel"])
        growth = self.config.diesel_price.growth_rate
        volatility = self.config.diesel_price.volatility
        noise = self.np_random.normal(0.0, 1.0)
        drift = growth - 0.5 * volatility**2
        diffusion = volatility * noise
        return float(diesel_price * np.exp(drift + diffusion))

    def _update_energy_price_electricity(self) -> float:
        electricity_price = float(self._info_state["energy_price_electricity"])
        growth = self.config.electricity_price.growth_rate
        volatility = self.config.electricity_price.volatility
        noise = self.np_random.normal(0.0, 1.0)
        drift = growth - 0.5 * volatility**2
        diffusion = volatility * noise
        return float(electricity_price * np.exp(drift + diffusion))

    def _update_purchase_price_BET(self) -> float:
        purchase_price = float(self._info_state["purchase_price_BET"])
        bet_cfg = self.config.BET_price
        reversion = bet_cfg.mean_reversion_strength * (
            bet_cfg.long_term_mean - purchase_price
        )
        shock = self.np_random.normal(0.0, bet_cfg.purchase_price_volatility)
        return float(purchase_price + reversion + shock)

    def _update_purchase_price_DT(self) -> float:
        return float(self._info_state["purchase_price_DT"])

    def _update_productivity_BET(self) -> float:
        bet_prod_cfg = self.config.BET_productivity
        base = 1.0 / (
            1.0 + np.exp(-bet_prod_cfg.k * (self._current_year - bet_prod_cfg.t0))
        )
        return float(
            bet_prod_cfg.start + (bet_prod_cfg.max - bet_prod_cfg.start) * base
        )
